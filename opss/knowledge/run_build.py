#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

from __future__ import annotations

"""CLI entry point for building the PiscesLx knowledge store.

Builds a FAISS-indexed mmap-backed knowledge store from raw text
corpora using the 0.5B POPSS knowledge encoder, then writes it to
``memory_store_path`` so :class:`YvMemoryRouter` can load it at
inference / training time.

Usage
-----
    # Basic: read text from a file, write store to the default path
    python -m opss.knowledge.run_build --corpus my_data.txt

    # Specify store path and slot count explicitly
    python -m opss.knowledge.run_build --corpus books/ \\
        --store-path ./knowledge_store/7B/ \\
        --slots 100_000_000 --chunk-size 256

    # Streaming mode (no corpus file — feed stdin line by line)
    cat big_corpus.txt | python -m opss.knowledge.run_build --stream --slots 1_000_000

    # Build from a YvConfig JSON (reuses ``memory_store_path`` and
    # the ``knowledge_store_*`` fields)
    python -m opss.knowledge.run_build --config model_config.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from opss.knowledge import POPSSKnowledgeBuilder, POPSSKnowledgeBuilderConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PiscesLx knowledge store from text corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input sources (mutually exclusive groups)
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--corpus", "-c", type=str, default=None,
        help="Path to a text file or directory of text files",
    )
    src.add_argument(
        "--stream", "-s", action="store_true",
        help="Read text from stdin line by line",
    )

    # Store configuration
    parser.add_argument(
        "--store-path", "-o", type=str, default="./knowledge_store/7B/",
        help="Output directory for the FAISS knowledge store (default: ./knowledge_store/7B/)",
    )
    parser.add_argument(
        "--slots", type=int, default=0,
        help="Number of knowledge slots (0 = auto from available text)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=256,
        help="Token chunk size (default: 256)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=32,
        help="Token overlap between chunks (default: 32)",
    )
    parser.add_argument(
        "--contrastive-epochs", type=int, default=3,
        help="NT-Xent contrastive refinement epochs (default: 3, 0 = skip)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size (default: 64)",
    )

    # Encoder configuration
    parser.add_argument(
        "--encoder-hidden", type=int, default=640,
        help="Encoder hidden dimension (default: 640)",
    )
    parser.add_argument(
        "--encoder-layers", type=int, default=16,
        help="Encoder transformer layers (default: 16)",
    )
    parser.add_argument(
        "--encoder-experts", type=int, default=4,
        help="Encoder MoE experts (default: 4)",
    )
    parser.add_argument(
        "--encoder-heads", type=int, default=10,
        help="Encoder attention heads (default: 10)",
    )

    # FAISS index configuration
    parser.add_argument(
        "--index-type", type=str, default="ivfpq",
        choices=["ivfpq", "ivfflat"],
        help="FAISS index type (default: ivfpq)",
    )
    parser.add_argument(
        "--index-nlist", type=int, default=4096,
        help="IVF cluster count (default: 4096)",
    )
    parser.add_argument(
        "--index-m", type=int, default=16,
        help="PQ sub-quantizers (default: 16)",
    )
    parser.add_argument(
        "--index-nbits", type=int, default=8,
        help="Bits per PQ sub-quantizer (default: 8)",
    )

    # Runtime
    parser.add_argument(
        "--device", type=str, default=None,
        help='Computation device (default: auto — cuda if available else cpu)',
    )
    parser.add_argument(
        "--use-fp8", action="store_true",
        help="Use FP8 for the encoder forward pass",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YvConfig JSON file; overrides all other CLI flags",
    )

    return parser.parse_args(argv)


def _load_corpus(path: str) -> list[str]:
    """Load text(s) from a file or directory."""
    p = Path(path)
    if p.is_file():
        return [p.read_text(encoding="utf-8")]
    if p.is_dir():
        texts: list[str] = []
        for f in sorted(p.rglob("*.txt")):
            texts.append(f.read_text(encoding="utf-8"))
        if not texts:
            print(f"Warning: no .txt files found in {path}", file=sys.stderr)
        return texts
    print(f"Error: {path} is not a file or directory", file=sys.stderr)
    sys.exit(1)


def _stream_corpus(total_slots: int) -> list[str]:
    """Read lines from stdin up to ``total_slots`` chunks."""
    texts: list[str] = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            texts.append(line)
            if total_slots > 0 and len(texts) >= total_slots:
                break
    return texts


def build_from_config(config_path: str) -> None:
    """Load a YvConfig JSON and invoke the builder with its settings."""
    try:
        from model.config import YvConfig
    except ImportError:
        print(
            "Cannot import YvConfig; make sure the project root is on sys.path",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = YvConfig.from_json(config_path)
    store_path = cfg.memory_store_path or "./knowledge_store/7B/"

    builder_cfg = POPSSKnowledgeBuilderConfig(
        encoder_hidden=cfg.knowledge_encoder_hidden,
        encoder_layers=cfg.knowledge_encoder_layers,
        encoder_experts=cfg.knowledge_encoder_experts,
        knowledge_slots=cfg.knowledge_store_slots,
        chunk_size=cfg.knowledge_store_chunk_size,
        chunk_overlap=cfg.knowledge_store_chunk_overlap,
        contrastive_epochs=cfg.knowledge_store_contrastive_epochs,
        batch_size=cfg.knowledge_store_batch_size,
        index_type=cfg.knowledge_store_index_type,
        index_nlist=cfg.knowledge_store_index_nlist,
        index_m=cfg.knowledge_store_index_m,
        index_nbits=cfg.knowledge_store_index_nbits,
        store_path=store_path,
        device=cfg.device if hasattr(cfg, "device") else None,
        use_fp8=cfg.knowledge_store_use_fp8,
        knowledge_dim=cfg.memory_knowledge_dim,
    )
    builder = POPSSKnowledgeBuilder(builder_cfg)

    print(f"Building knowledge store from config: {config_path}")
    print(f"  store_path  = {store_path}")
    print(f"  slots       = {builder_cfg.knowledge_slots:,}")
    print(f"  chunk_size  = {builder_cfg.chunk_size}")
    print(f"  device      = {builder_cfg.device}")
    print()

    texts = []
    if hasattr(cfg, "_corpus_path") and cfg._corpus_path:
        texts = _load_corpus(cfg._corpus_path)

    if texts:
        builder.build_from_texts(texts)
    else:
        print("No corpus found. Use --corpus or --stream to provide text.")
        print("Run with --help for usage.")
        sys.exit(0)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Config mode ────────────────────────────────────────────────
    if args.config is not None:
        build_from_config(args.config)
        return

    # ── Direct CLI mode ────────────────────────────────────────────
    store_path = args.store_path
    os.makedirs(store_path, exist_ok=True)

    builder_cfg = POPSSKnowledgeBuilderConfig(
        encoder_hidden=args.encoder_hidden,
        encoder_layers=args.encoder_layers,
        encoder_experts=args.encoder_experts,
        encoder_heads=args.encoder_heads,
        knowledge_slots=args.slots,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        contrastive_epochs=args.contrastive_epochs,
        batch_size=args.batch_size,
        index_type=args.index_type,
        index_nlist=args.index_nlist,
        index_m=args.index_m,
        index_nbits=args.index_nbits,
        store_path=store_path,
        device=args.device,
        use_fp8=args.use_fp8,
    )
    builder = POPSSKnowledgeBuilder(builder_cfg)

    # ── Load corpus ────────────────────────────────────────────────
    if args.stream:
        print(f"Reading text from stdin (target: {args.slots:,} slots)...")
        texts = _stream_corpus(args.slots)
    elif args.corpus:
        print(f"Loading corpus from {args.corpus}...")
        texts = _load_corpus(args.corpus)
    else:
        print("No input source specified. Use --corpus, --stream, or --config.")
        print("Run with --help for usage.")
        sys.exit(1)

    print(f"Loaded {len(texts)} text(s), building store...")

    if args.stream:
        import itertools
        result = builder.build_from_stream(iter(texts), total_slots=args.slots)
    else:
        result = builder.build_from_texts(texts)

    print(f"\nDone! {result['num_slots']:,} slots written to {result['store_path']}")


if __name__ == "__main__":
    main()
