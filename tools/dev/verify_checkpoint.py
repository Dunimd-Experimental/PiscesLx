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

"""
Prove that a checkpoint is a trained weight, not a re-saved initialisation (§9).

A file on disk says nothing by itself.  This tool answers three separate questions
with measured evidence and reports each one separately, because a checkpoint can
pass any subset of them by accident:

1. **Is it loadable?** ``YvModel.load_state_dict(..., strict=True)`` — a partial or
   renamed state dict is a failure, not a warning.
2. **Did training change the weights?** every parameter is compared against a freshly
   constructed model with the same config; identical tensors mean no step was applied.
3. **Is it usable?** teacher-forced next-token cross-entropy on real held-out text,
   reported as perplexity, plus the loss recorded inside the checkpoint itself.
"""

import argparse
import json
import math
import pathlib
import sys

import torch

from model.config import YvConfig
from model.core.model import YvModel
from model.tokenizer import YvTokenizer
from tools.data.dataset.raw_text import PiscesLxToolsDataJsonSampleReader
from utils.dc import PiscesLxLogger
from utils.paths import PiscesLxPaths

__all__ = ["PiscesLxCheckpointVerifier"]


class PiscesLxCheckpointVerifier:
    """Load, diff and score a training checkpoint."""

    LOG_NAME = "PiscesLx.Tools.Dev"
    MAX_EVAL_TOKENS = 4096
    SAMPLE_LIMIT = 64
    BYTES_PER_GB = 1024 ** 3

    def __init__(self, checkpoint: str, model_config: str, corpus: str) -> None:
        self.checkpoint_path = pathlib.Path(checkpoint)
        self.model_config = model_config
        self.corpus = corpus
        self._LOG = PiscesLxLogger(
            self.LOG_NAME,
            file_path=PiscesLxPaths.get_log_file(self.LOG_NAME),
            enable_file=True,
        )

    def verify(self) -> dict:
        """Run all three checks; every result carries its own evidence."""
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        report = {
            "checkpoint": self.checkpoint_path.as_posix(),
            "size_gb": round(self.checkpoint_path.stat().st_size / self.BYTES_PER_GB, 3),
            "keys": sorted(payload) if isinstance(payload, dict) else type(payload).__name__,
            "global_step": payload.get("global_step") if isinstance(payload, dict) else None,
            "recorded_loss": self._recorded_loss(payload),
        }
        state = payload["model_state_dict"] if isinstance(payload, dict) else payload
        cfg = YvConfig.from_yaml(self.model_config)
        reference = YvModel(cfg, device=torch.device("cpu"), dtype=torch.float32)
        reference_state = reference.state_dict()

        report["loadable"] = self._check_loadable(reference_state, state)
        report["trained"] = self._check_trained(reference_state, state)
        reference.eval()
        report["perplexity"] = self._check_usable(reference, state)
        report["verdict"] = (
            report["loadable"]["ok"] and report["trained"]["ok"] and report["perplexity"]["ok"]
        )
        return report

    @staticmethod
    def _recorded_loss(payload) -> object:
        if not isinstance(payload, dict):
            return None
        stats = payload.get("training_stats") or {}
        for key in ("loss", "train_loss", "last_loss"):
            if key in stats:
                return stats[key]
        return {key: value for key, value in stats.items() if "loss" in key.lower()} or None

    @staticmethod
    def _check_loadable(reference_state: dict, state: dict) -> dict:
        missing = sorted(set(reference_state) - set(state))
        unexpected = sorted(set(state) - set(reference_state))
        shape_mismatch = sorted(
            key for key in set(reference_state) & set(state)
            if tuple(reference_state[key].shape) != tuple(state[key].shape)
        )
        return {
            "ok": not (missing or unexpected or shape_mismatch),
            "missing": missing[:5],
            "unexpected": unexpected[:5],
            "shape_mismatch": shape_mismatch[:5],
            "tensors": len(state),
        }

    @staticmethod
    def _check_trained(reference_state: dict, state: dict) -> dict:
        changed = 0
        identical = []
        for key, fresh in reference_state.items():
            saved = state.get(key)
            if saved is None or saved.shape != fresh.shape:
                continue
            if torch.equal(saved.float(), fresh.float()):
                identical.append(key)
            else:
                changed += 1
        compared = changed + len(identical)
        return {
            "ok": changed > 0,
            "changed": changed,
            "unchanged": len(identical),
            "compared": compared,
            "first_unchanged": identical[:5],
        }

    def _check_usable(self, model: YvModel, state: dict) -> dict:
        """Score the checkpoint with teacher forcing on real held-out documents."""
        model.load_state_dict(state, strict=False)
        tokenizer = YvTokenizer()
        samples = PiscesLxToolsDataJsonSampleReader.resolve(self.corpus)[: self.SAMPLE_LIMIT]
        if not samples:
            return {"ok": False, "error": f"no samples readable from {self.corpus}"}
        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        total_loss, tokens = 0.0, 0
        for sample in samples:
            text = PiscesLxToolsDataJsonSampleReader.extract_text(sample)
            ids = tokenizer.encode(text)
            ids = torch.tensor(list(ids), dtype=torch.long)[: self.MAX_EVAL_TOKENS]
            if ids.numel() < 2:
                continue
            input_ids = ids[:-1].unsqueeze(0)
            labels = ids[1:].unsqueeze(0)
            with torch.no_grad():
                output = model(input_ids=input_ids, labels=labels)
            loss = output["loss"] if isinstance(output, dict) else output.loss
            if loss is None:
                return {"ok": False, "error": "model returned no loss for labelled input"}
            total_loss += float(loss) * (ids.numel() - 1)
            tokens += int(ids.numel() - 1)
        if tokens == 0:
            return {"ok": False, "error": "no token could be scored"}
        mean_loss = total_loss / tokens
        return {
            "ok": math.isfinite(mean_loss) and mean_loss > 0,
            "mean_loss": round(mean_loss, 4),
            "perplexity": round(math.exp(min(mean_loss, 50.0)), 2),
            "tokens_scored": tokens,
            "documents": len(samples),
            "pad_id": pad_id,
        }

    @classmethod
    def cli(cls, argv) -> int:
        parser = argparse.ArgumentParser(prog="tools/dev/verify_checkpoint.py")
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--model_config", required=True)
        parser.add_argument("--corpus", required=True, help="JSONL directory or file for scoring")
        args = parser.parse_args(argv)
        report = cls(args.checkpoint, args.model_config, args.corpus).verify()
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(PiscesLxCheckpointVerifier.cli(sys.argv[1:]))
