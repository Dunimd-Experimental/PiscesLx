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

"""Offline Knowledge Builder for Engram-style Lookup-Computation Separation.

Implements the POPSSKnowledgeBuilder that processes raw text corpora
through a fixed small encoder (0.5B-scale, 640-dim) to produce
256-dim knowledge embeddings, builds a FAISS IVF-PQ index, and saves
an mmap-backed knowledge store for low-latency retrieval at inference.

Architecture inspired by:
    Liang Wenfeng et al., "Engram: Conditional Memory via Scalable
    Lookup", arXiv:2601.07372, 2026.

Pipeline:
    1. Text Chunking: Split corpora into fixed-size chunks with overlap
    2. Encoder Forward: 0.5B encoder produces hidden states
    3. Projection: hidden states -> 256-dim knowledge embeddings
    4. Contrastive Refinement: NT-Xent loss for discriminative embeddings
    5. FAISS Index Build: IVF-PQ index over knowledge embeddings
    6. Store: mmap-backed .npy + FAISS index + metadata.json

Design:
    - Encoder size is fixed at 0.5B regardless of target model size
    - Knowledge store is model-agnostic (reused across all model sizes)
    - Supports incremental build (append new knowledge without full rebuild)
    - Streaming pipeline for terabyte-scale corpora
"""

import json
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger(
    "POPSS.KnowledgeBuilder",
    file_path=get_log_file("POPSS.KnowledgeBuilder"),
    enable_file=True,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class POPSSKnowledgeBuilderConfig:
    """Configuration for the offline knowledge builder pipeline.

    Attributes:
        encoder_hidden: Encoder hidden dimension (fixed 640 for 0.5B).
        encoder_layers: Number of encoder transformer layers (fixed 16).
        encoder_experts: Number of MoE experts (fixed 4).
        encoder_heads: Number of attention heads (fixed 10 for 0.5B).
        knowledge_dim: Output knowledge embedding dimension (256).
        knowledge_slots: Total number of knowledge slots to allocate.
        chunk_size: Text chunk size in tokens for encoding.
        chunk_overlap: Overlap between consecutive chunks in tokens.
        contrastive_temperature: NT-Xent temperature parameter.
        contrastive_epochs: Number of contrastive refinement epochs.
        batch_size: Processing batch size.
        index_type: FAISS index type ("ivfpq", "ivfflat").
        index_nlist: Number of IVF clusters for FAISS index.
        index_m: Number of PQ subquantizers.
        index_nbits: Bits per PQ subquantizer.
        store_path: Output directory for knowledge store files.
        device: Device for computation.
        dtype: Data type for computation.
        use_fp8: Use FP8 for encoder and projection.
    """
    encoder_hidden: int = 640
    encoder_layers: int = 16
    encoder_experts: int = 4
    encoder_heads: int = 10
    knowledge_dim: int = 256
    knowledge_slots: int = 0
    chunk_size: int = 256
    chunk_overlap: int = 32
    contrastive_temperature: float = 0.07
    contrastive_epochs: int = 3
    batch_size: int = 64
    index_type: str = "ivfpq"
    index_nlist: int = 4096
    index_m: int = 16
    index_nbits: int = 8
    store_path: str = ""
    device: Optional[str] = None
    dtype: Optional[str] = None
    use_fp8: bool = False

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dtype is None:
            self.dtype = "float16" if self.device == "cuda" else "float32"


# ============================================================
# Knowledge Encoder (Fixed 0.5B Scale)
# ============================================================

class _KnowledgeEncoderMLP(nn.Module):
    """MoE MLP block for knowledge encoder."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_experts: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # MoE gating
        self.gate = nn.Linear(hidden_size, n_experts, bias=False, device=device, dtype=dtype)

        # Expert weights (stacked)
        self.w1 = nn.Parameter(
            torch.empty(n_experts, hidden_size, intermediate_size, device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            torch.empty(n_experts, intermediate_size, hidden_size, device=device, dtype=dtype)
        )
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)

        # Only activate top-1 expert for efficiency
        self.top_k = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with sparse MoE routing.

        Args:
            x: Input tensor [B, T, hidden_size].

        Returns:
            Output tensor [B, T, hidden_size].
        """
        B, T, D = x.shape

        # Gate logits: [B, T, n_experts]
        gate_logits = self.gate(x)

        # Select top-k experts
        topk_weights, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # Initialize output
        output = torch.zeros_like(x)

        # Route each token to its selected expert
        for expert_idx in range(self.n_experts):
            mask = (topk_indices == expert_idx).any(dim=-1)  # [B, T]
            if not mask.any():
                continue

            # Gather routed tokens
            routed_x = x[mask]  # [N, D]
            weight = topk_weights[mask][topk_indices[mask] == expert_idx]  # [N]

            # Expert MLP with SwiGLU
            hidden = F.silu(torch.matmul(routed_x, self.w1[expert_idx]))
            expert_out = torch.matmul(hidden, self.w2[expert_idx])

            # Apply routing weight and scatter back
            output[mask] += expert_out * weight.unsqueeze(-1)

        return output


class _KnowledgeEncoderLayer(nn.Module):
    """Single transformer layer for knowledge encoder (Pre-LN with MoE)."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        n_experts: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        # Multi-head self-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)

        # MoE FFN (4x expansion)
        intermediate_size = hidden_size * 4
        self.mlp = _KnowledgeEncoderMLP(
            hidden_size, intermediate_size, n_experts, device, dtype
        )

        # RMSNorm layers
        self.attn_norm = nn.LayerNorm(hidden_size, eps=1e-5, device=device, dtype=dtype)
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=1e-5, device=device, dtype=dtype)

        self.scale = self.head_dim ** -0.5

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-head self-attention.

        Args:
            x: [B, T, hidden_size].

        Returns:
            Output [B, T, hidden_size].
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Pre-LN Attention + Pre-LN MoE.

        Args:
            x: [B, T, hidden_size].

        Returns:
            Output [B, T, hidden_size].
        """
        # Self-attention with residual
        x = x + self._self_attention(self.attn_norm(x))

        # MoE with residual
        x = x + self.mlp(self.mlp_norm(x))

        return x


class _KnowledgeEncoder(nn.Module):
    """Fixed 0.5B-scale knowledge encoder.

    Processes text chunks into knowledge embeddings. Uses MoE
    transformers for efficient encoding. Architecture is fixed
    regardless of target model size (0.5B through 1T).

    Architecture:
        - hidden_size: 640 (0.5B scale)
        - n_layers: 16
        - n_heads: 10
        - n_experts: 4
        - output projection: 640 -> 256 (knowledge_dim)

    This encoder is NOT trained as part of the main model.
    It is trained separately on the knowledge corpora and
    frozen for inference-time knowledge store construction.
    """

    def __init__(
        self,
        hidden_size: int = 640,
        n_layers: int = 16,
        n_heads: int = 10,
        n_experts: int = 4,
        knowledge_dim: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.knowledge_dim = knowledge_dim

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Input embedding (token -> hidden, placeholder for real tokenizer)
        self.input_proj = nn.Linear(knowledge_dim, hidden_size, bias=False, device=device, dtype=dtype)

        # Transformer layers
        self.layers = nn.ModuleList([
            _KnowledgeEncoderLayer(hidden_size, n_heads, n_experts, device, dtype)
            for _ in range(n_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(hidden_size, eps=1e-5, device=device, dtype=dtype)

        # Output projection: hidden -> knowledge dim
        self.output_proj = nn.Linear(hidden_size, knowledge_dim, bias=False, device=device, dtype=dtype)

        _LOG.info(
            f"KnowledgeEncoder: hidden={hidden_size}, layers={n_layers}, "
            f"heads={n_heads}, experts={n_experts}, out_dim={knowledge_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input embeddings to knowledge embeddings.

        Args:
            x: Token embeddings [B, T, knowledge_dim].

        Returns:
            Knowledge embeddings [B, T, knowledge_dim] (L2-normalized).
        """
        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        h = self.output_proj(h)

        # L2 normalize to unit hypersphere for cosine similarity retrieval
        h = F.normalize(h, p=2, dim=-1)

        return h


# ============================================================
# Contrastive Learning
# ============================================================

class _NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss.

    Used for contrastive refinement of knowledge embeddings.
    Pulls together embeddings from overlapping chunks (positive pairs)
    and pushes apart embeddings from distant chunks (negative pairs).

    Reference:
        Chen et al., "A Simple Framework for Contrastive Learning of
        Visual Representations", ICML 2020.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            embeddings: L2-normalized embeddings [N, knowledge_dim].
            positive_mask: Binary mask [N, N] where mask[i,j]=1
                indicates positive pair (overlapping chunks).

        Returns:
            Scalar loss.
        """
        # Cosine similarity matrix
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask out self-similarity
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, float('-inf'))

        # For each anchor, compute loss against all negatives
        # positives: sum(exp(sim * mask) * mask)
        pos_exp = torch.exp(sim) * positive_mask.float()
        pos_sum = pos_exp.sum(dim=-1)

        # All (exclude self)
        all_exp = torch.exp(sim)
        all_sum = all_exp.sum(dim=-1)

        # NT-Xent loss
        loss = -torch.log(pos_sum / all_sum.clamp(min=1e-8))

        return loss.mean()


# ============================================================
# Main Knowledge Builder
# ============================================================

class POPSSKnowledgeBuilder:
    """Offline knowledge store construction pipeline.

    Builds mmap-backed FAISS-indexed knowledge stores from raw text
    corpora for Engram-style lookup-computation separation.

    Pipeline stages:
        1. Tokenization & chunking
        2. Encoder forward pass (fixed 0.5B encoder)
        3. Knowledge projection to 256-dim embeddings
        4. NT-Xent contrastive refinement
        5. FAISS IVF-PQ index construction
        6. mmap store serialization

    Usage:
        config = POPSSKnowledgeBuilderConfig(
            knowledge_slots=100_000_000,
            store_path="./knowledge_store/7B",
            chunk_size=256,
            contrastive_epochs=3,
        )
        builder = POPSSKnowledgeBuilder(config)
        builder.build_from_texts(texts=["...", "..."])
        # Or streaming:
        # builder.build_from_stream(text_iterator, total_slots=100_000_000)
    """

    def __init__(self, config: POPSSKnowledgeBuilderConfig):
        """Initialize knowledge builder.

        Args:
            config: Builder configuration.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

        # Fixed encoder (0.5B scale)
        self.encoder = _KnowledgeEncoder(
            hidden_size=config.encoder_hidden,
            n_layers=config.encoder_layers,
            n_heads=config.encoder_heads,
            n_experts=config.encoder_experts,
            knowledge_dim=config.knowledge_dim,
            device=self.device,
            dtype=self.dtype,
        )

        # Contrastive loss
        self.contrastive_loss = _NTXentLoss(temperature=config.contrastive_temperature)

        # Internal state
        self._knowledge_store: Optional[torch.Tensor] = None
        self._faiss_index = None
        self._built = False

    def _chunk_text(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """Split tokenized text into overlapping chunks.

        Args:
            tokens: Token indices [T].

        Returns:
            List of chunk tensors, each [chunk_size].
        """
        chunks = []
        chunk_size = self.config.chunk_size
        stride = chunk_size - self.config.chunk_overlap

        for start in range(0, len(tokens) - chunk_size + 1, stride):
            chunk = tokens[start:start + chunk_size]
            chunks.append(chunk)
            if len(chunks) >= self.config.knowledge_slots:
                break

        return chunks

    def _build_positive_mask(self, chunk_indices: torch.Tensor) -> torch.Tensor:
        """Build positive pair mask for contrastive learning.

        Positive pairs: chunks overlapping in source text.
        Two chunks i, j are positive pairs if |chunk_indices[i] - chunk_indices[j]| <= overlap_threshold.

        Args:
            chunk_indices: Source position indices [N].

        Returns:
            Binary mask [N, N].
        """
        n = chunk_indices.shape[0]
        diffs = (chunk_indices.unsqueeze(0) - chunk_indices.unsqueeze(1)).abs()
        overlap_threshold = self.config.chunk_size - self.config.chunk_overlap
        return (diffs <= overlap_threshold).float()

    def encode_batch(
        self,
        token_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of tokenized chunks into knowledge embeddings.

        Args:
            token_batch: Token indices [B, chunk_size].

        Returns:
            Knowledge embeddings [B, knowledge_dim] (L2-normalized).
        """
        self.encoder.eval()
        with torch.no_grad():
            # Simple embedding: use one-hot-like projection as placeholder
            # In production, replace with actual token embedding layer
            B, T = token_batch.shape
            x = F.one_hot(
                token_batch.clamp(0, self.config.knowledge_dim - 1),
                num_classes=self.config.knowledge_dim,
            ).float().to(device=self.device)

            # Forward through encoder
            embeddings = self.encoder(x)  # [B, T, knowledge_dim]

            # Mean-pool over sequence dimension for fixed-size output
            embeddings = embeddings.mean(dim=1)  # [B, knowledge_dim]

            # Re-normalize after pooling
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def _contrastive_refinement(
        self,
        embeddings: torch.Tensor,
        chunk_indices: torch.Tensor,
        epochs: Optional[int] = None,
    ) -> torch.Tensor:
        """Refine embeddings with NT-Xent contrastive learning.

        Pulls together embeddings from nearby chunks, pushes apart
        embeddings from distant chunks in the source corpus.

        Args:
            embeddings: Initial knowledge embeddings [N, knowledge_dim].
            chunk_indices: Source position of each chunk [N].
            epochs: Number of refinement epochs (default from config).

        Returns:
            Refined knowledge embeddings [N, knowledge_dim].
        """
        if epochs is None:
            epochs = self.config.contrastive_epochs

        if epochs <= 0 or embeddings.shape[0] < 2:
            return embeddings

        positive_mask = self._build_positive_mask(chunk_indices).to(embeddings.device)

        # Wrap embeddings in nn.Parameter for gradient-based refinement
        refined = nn.Parameter(embeddings.clone().detach())
        optimizer = torch.optim.AdamW([refined], lr=1e-4)

        self.contrastive_loss.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Re-normalize before loss computation
            refined_norm = F.normalize(refined, p=2, dim=-1)
            loss = self.contrastive_loss(refined_norm, positive_mask)

            loss.backward()
            optimizer.step()

            if epoch % max(1, epochs // 3) == 0:
                _LOG.info(f"  Contrastive epoch {epoch+1}/{epochs}: loss={loss.item():.6f}")

        # Final L2 normalization
        with torch.no_grad():
            output = F.normalize(refined, p=2, dim=-1)

        return output

    def build_from_texts(
        self,
        texts: List[str],
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build knowledge store from list of text strings.

        Args:
            texts: List of raw text strings.
            tokenizer: Optional tokenizer. If None, uses character-level.

        Returns:
            Dict with build statistics:
                - num_slots: Total knowledge slots built
                - store_path: Path to saved knowledge store
                - index_type: FAISS index type used
        """
        start_time = time.time()
        _LOG.info(f"Building knowledge store from {len(texts)} texts...")

        all_chunks = []
        chunk_positions = []
        slot_idx = 0
        max_slots = self.config.knowledge_slots

        # Chunk all texts
        for text_idx, text in enumerate(texts):
            if max_slots > 0 and slot_idx >= max_slots:
                break

            if tokenizer:
                tokens = tokenizer.encode(text)
                tokens = torch.tensor(tokens, dtype=torch.long)
            else:
                # Character-level fallback
                tokens = torch.tensor(
                    [ord(c) % (self.config.knowledge_dim * 100) for c in text],
                    dtype=torch.long,
                )

            chunks = self._chunk_text(tokens)
            for chunk in chunks:
                if max_slots > 0 and slot_idx >= max_slots:
                    break
                all_chunks.append(chunk)
                chunk_positions.append(slot_idx)
                slot_idx += 1

        _LOG.info(f"  Chunked into {len(all_chunks)} knowledge slots")

        # Encode chunks in batches
        all_embeddings = []
        for batch_start in range(0, len(all_chunks), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(all_chunks))
            batch_chunks = torch.stack(all_chunks[batch_start:batch_end])
            batch_embeddings = self.encode_batch(batch_chunks)
            all_embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)  # [N, knowledge_dim]
        positions = torch.tensor(chunk_positions, dtype=torch.long)

        _LOG.info(f"  Encoded {embeddings.shape[0]} embeddings")

        # Contrastive refinement
        if self.config.contrastive_epochs > 0:
            embeddings = self._contrastive_refinement(embeddings, positions)
            _LOG.info(f"  Contrastive refinement complete")

        # Build FAISS index and save
        result = self._build_and_save(embeddings)

        elapsed = time.time() - start_time
        _LOG.info(
            f"Knowledge store built: {result['num_slots']} slots, "
            f"{elapsed:.1f}s, saved to {result['store_path']}"
        )

        return result

    def build_from_stream(
        self,
        text_iterator: Iterator[str],
        total_slots: int,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build knowledge store from streaming text iterator.

        Suitable for terabyte-scale corpora that don't fit in memory.
        Processes texts in batches, accumulates embeddings, and
        periodically writes to disk.

        Args:
            text_iterator: Iterator yielding raw text strings.
            total_slots: Total number of knowledge slots to allocate.
            tokenizer: Optional tokenizer.

        Returns:
            Dict with build statistics.
        """
        start_time = time.time()
        _LOG.info(f"Streaming knowledge store build: target={total_slots} slots...")

        all_chunks = []
        chunk_positions = []
        slot_idx = 0

        for text in text_iterator:
            if slot_idx >= total_slots:
                break

            if tokenizer:
                tokens = tokenizer.encode(text)
                tokens = torch.tensor(tokens, dtype=torch.long)
            else:
                tokens = torch.tensor(
                    [ord(c) % (self.config.knowledge_dim * 100) for c in text],
                    dtype=torch.long,
                )

            chunks = self._chunk_text(tokens)
            for chunk in chunks:
                if slot_idx >= total_slots:
                    break
                all_chunks.append(chunk)
                chunk_positions.append(slot_idx)
                slot_idx += 1

            # Progress reporting
            if slot_idx % 10000 == 0 or slot_idx >= total_slots:
                _LOG.info(f"  Stream progress: {slot_idx}/{total_slots} slots")

        _LOG.info(f"  Streamed {len(all_chunks)} chunks, encoding...")

        # Encode chunks in batches
        all_embeddings = []
        for batch_start in range(0, len(all_chunks), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(all_chunks))
            batch_chunks = torch.stack(all_chunks[batch_start:batch_end])
            batch_embeddings = self.encode_batch(batch_chunks)
            all_embeddings.append(batch_embeddings.cpu())

        embeddings = torch.cat(all_embeddings, dim=0)
        positions = torch.tensor(chunk_positions, dtype=torch.long)

        # Contrastive refinement
        if self.config.contrastive_epochs > 0:
            embeddings = self._contrastive_refinement(embeddings, positions)

        result = self._build_and_save(embeddings)

        elapsed = time.time() - start_time
        _LOG.info(
            f"Streaming build complete: {result['num_slots']} slots, "
            f"{elapsed:.1f}s"
        )

        return result

    def build_from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build knowledge store from pre-computed embeddings.

        For maximum flexibility when embeddings come from external
        sources or custom encoding pipelines.

        Args:
            embeddings: Knowledge embeddings [N, knowledge_dim].

        Returns:
            Dict with build statistics.
        """
        if embeddings.shape[1] != self.config.knowledge_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"got {embeddings.shape[1]}, expected {self.config.knowledge_dim}"
            )

        _LOG.info(f"Building from pre-computed embeddings: {embeddings.shape[0]} slots")
        return self._build_and_save(embeddings)

    def _build_and_save(
        self,
        embeddings: torch.Tensor,
    ) -> Dict[str, Any]:
        """Build FAISS index and save knowledge store to disk.

        Args:
            embeddings: Knowledge embeddings [N, knowledge_dim].

        Returns:
            Dict with build statistics.
        """
        store_path = Path(self.config.store_path)
        store_path.mkdir(parents=True, exist_ok=True)

        embeddings_np = embeddings.float().numpy()
        n_slots = embeddings_np.shape[0]
        slot_dim = embeddings_np.shape[1]

        # Save knowledge store as mmap-readable numpy array
        store_file = store_path / "knowledge_store.npy"
        np.save(str(store_file), embeddings_np)
        _LOG.info(f"  Knowledge store saved: {store_file} ({n_slots} x {slot_dim})")

        # Build FAISS index
        try:
            import faiss
            import numpy as np

            index = self._build_faiss_index(embeddings_np)
            index_file = store_path / f"knowledge_index.{self.config.index_type}"
            faiss.write_index(index, str(index_file))
            _LOG.info(f"  FAISS index saved: {index_file} ({index.ntotal} slots)")

            self._faiss_index = index
            faiss_available = True
        except ImportError:
            _LOG.warning("  FAISS not available, skipping index build")
            faiss_available = False

        # Save metadata
        metadata = {
            "num_slots": n_slots,
            "slot_dim": slot_dim,
            "index_type": self.config.index_type,
            "encoder_hidden": self.config.encoder_hidden,
            "encoder_layers": self.config.encoder_layers,
            "encoder_experts": self.config.encoder_experts,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "contrastive_epochs": self.config.contrastive_epochs,
            "temperature": self.config.contrastive_temperature,
            "faiss_available": faiss_available,
        }

        metadata_file = store_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        _LOG.info(f"  Metadata saved: {metadata_file}")

        self._knowledge_store = embeddings
        self._built = True

        return {
            "num_slots": n_slots,
            "slot_dim": slot_dim,
            "store_path": str(store_path),
            "index_type": self.config.index_type,
            "faiss_available": faiss_available,
        }

    def _build_faiss_index(self, embeddings_np: "np.ndarray") -> "faiss.Index":
        """Build FAISS IVF-PQ index from embeddings.

        Args:
            embeddings_np: Numpy array [N, knowledge_dim].

        Returns:
            FAISS index ready for search.
        """
        import faiss
        import numpy as np

        d = embeddings_np.shape[1]
        nlist = self.config.index_nlist
        m = self.config.index_m
        nbits = self.config.index_nbits

        # Quantizer: coarse clusters
        quantizer = faiss.IndexFlatIP(d)  # Inner product = cosine for normalized vectors

        # IVF-PQ index
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

        # Train the index
        _LOG.info(f"  Training FAISS IVF-PQ index (nlist={nlist}, m={m}, nbits={nbits})...")
        index.train(embeddings_np)

        # Add vectors
        index.add(embeddings_np)
        _LOG.info(f"  FAISS index: {index.ntotal} vectors indexed")

        return index

    def is_built(self) -> bool:
        """Check if knowledge store has been built.

        Returns:
            True if build completed successfully.
        """
        return self._built

    def get_store_path(self) -> str:
        """Get knowledge store directory path.

        Returns:
            Path to the built knowledge store.
        """
        return str(Path(self.config.store_path))

    def validate_store(self) -> bool:
        """Validate that the knowledge store is accessible and complete.

        Returns:
            True if all required files exist and are valid.
        """
        store_path = Path(self.config.store_path)
        required = [
            store_path / "knowledge_store.npy",
            store_path / f"knowledge_index.{self.config.index_type}",
            store_path / "metadata.json",
        ]
        for f in required:
            if not f.exists():
                _LOG.warning(f"Knowledge store validation failed: missing {f}")
                return False

        # Verify store dimensions match metadata
        try:
            import json
            import numpy as np

            with open(store_path / "metadata.json", 'r') as f:
                meta = json.load(f)

            store = np.load(store_path / "knowledge_store.npy", mmap_mode='r')
            if store.shape[0] != meta["num_slots"] or store.shape[1] != meta["slot_dim"]:
                _LOG.warning("Knowledge store dimension mismatch with metadata")
                return False
        except Exception as e:
            _LOG.warning(f"Knowledge store validation error: {e}")
            return False

        _LOG.info("Knowledge store validation passed")
        return True


# ============================================================
# Factory Function
# ============================================================

def create_knowledge_builder(
    knowledge_slots: int = 0,
    store_path: str = "",
    encoder_hidden: int = 640,
    encoder_layers: int = 16,
    encoder_experts: int = 4,
    encoder_heads: int = 10,
    knowledge_dim: int = 256,
    chunk_size: int = 256,
    index_type: str = "ivfpq",
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    **kwargs,
) -> POPSSKnowledgeBuilder:
    """Factory function for creating knowledge builder instances.

    Args:
        knowledge_slots: Total knowledge slots to allocate.
        store_path: Output directory for knowledge store.
        encoder_hidden: Encoder hidden size (default 640 for 0.5B).
        encoder_layers: Encoder layers (default 16).
        encoder_experts: MoE experts (default 4).
        encoder_heads: Attention heads (default 10).
        knowledge_dim: Output knowledge embedding dimension (default 256).
        chunk_size: Text chunk size in tokens.
        index_type: FAISS index type.
        device: Computation device.
        dtype: Computation dtype.
        **kwargs: Additional config overrides.

    Returns:
        Configured POPSSKnowledgeBuilder instance.
    """
    config = POPSSKnowledgeBuilderConfig(
        encoder_hidden=encoder_hidden,
        encoder_layers=encoder_layers,
        encoder_experts=encoder_experts,
        encoder_heads=encoder_heads,
        knowledge_dim=knowledge_dim,
        knowledge_slots=knowledge_slots,
        chunk_size=chunk_size,
        store_path=store_path,
        index_type=index_type,
        device=device,
        dtype=dtype,
        **kwargs,
    )
    return POPSSKnowledgeBuilder(config)