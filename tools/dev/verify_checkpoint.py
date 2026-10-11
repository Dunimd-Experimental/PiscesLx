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
import torch.nn.functional as F

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
    SIZE_DIGITS = 3
    LOSS_DIGITS = 4
    SCORE_DIGITS = 2
    REPORT_LIST_ITEMS = 5
    # exp() overflows float near 709; the cap exists only to keep the report numeric,
    # and it must not clamp a score that is legitimately above the uniform baseline.
    PERPLEXITY_EXPONENT_CAP = 700.0

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
            "size_gb": round(self.checkpoint_path.stat().st_size / self.BYTES_PER_GB, self.SIZE_DIGITS),
            "keys": sorted(payload) if isinstance(payload, dict) else type(payload).__name__,
            "global_step": payload.get("global_step") if isinstance(payload, dict) else None,
            "recorded_loss": self._recorded_loss(payload),
        }
        state = payload["model_state_dict"] if isinstance(payload, dict) else payload
        cfg = YvConfig.from_yaml(self.model_config)
        reference = YvModel(cfg, device=torch.device("cpu"), dtype=torch.float32)
        # Optional components are built on first use, so the checkpoint records which ones
        # it was trained with; the reference must declare exactly that set before its keys
        # can be compared strictly.
        reference.materialize_components(
            payload.get("materialized_components") if isinstance(payload, dict) else [])
        reference_state = reference.state_dict()

        report["loadable"] = self._check_loadable(reference_state, state)
        report["trained"] = self._check_trained(reference_state, state)
        reference.eval()
        # The scoring stage runs the full model, so a defect inside it is itself the
        # finding; report it as a failed stage instead of losing the two answers above.
        try:
            report["perplexity"] = self._check_usable(reference, state)
        except Exception as exc:
            report["perplexity"] = {"ok": False,
                                    "error": f"{type(exc).__name__}: {exc}"}
        # A weight whose next-token prediction is worse than a uniform distribution is not
        # usable, however many tensors the optimiser moved: 90 steps of this run drove the
        # summed objective from 120 to 39.7 while next-token CE stayed at 81.3 nats
        # (uniform = 11.9), because that decline was auxiliary terms. The verdict therefore
        # requires the LM signal itself, not just loadability and changed tensors.
        report["verdict"] = (
            report["loadable"]["ok"]
            and report["trained"]["ok"]
            and report["perplexity"]["ok"]
            and bool(report["perplexity"].get("beats_uniform", False))
        )
        return report

    def _round(self, value: float) -> float:
        return round(value, self.SCORE_DIGITS)

    @staticmethod
    def _recorded_loss(payload) -> object:
        if not isinstance(payload, dict):
            return None
        stats = payload.get("training_stats") or {}
        for key in ("loss", "train_loss", "last_loss"):
            if key in stats:
                return stats[key]
        return {key: value for key, value in stats.items() if "loss" in key.lower()} or None

    def _check_loadable(self, reference_state: dict, state: dict) -> dict:
        missing = sorted(set(reference_state) - set(state))
        unexpected = sorted(set(state) - set(reference_state))
        shape_mismatch = sorted(
            key for key in set(reference_state) & set(state)
            if tuple(reference_state[key].shape) != tuple(state[key].shape)
        )
        return {
            "ok": not (missing or unexpected or shape_mismatch),
            "missing": missing[: self.REPORT_LIST_ITEMS],
            "unexpected": unexpected[: self.REPORT_LIST_ITEMS],
            "shape_mismatch": shape_mismatch[: self.REPORT_LIST_ITEMS],
            "tensors": len(state),
        }

    def _check_trained(self, reference_state: dict, state: dict) -> dict:
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
            "first_unchanged": identical[: self.REPORT_LIST_ITEMS],
        }

    def _check_usable(self, model: YvModel, state: dict) -> dict:
        """Score the checkpoint with teacher forcing on real held-out documents."""
        model.load_state_dict(state, strict=True)
        tokenizer = YvTokenizer()
        samples = PiscesLxToolsDataJsonSampleReader.resolve(self.corpus)[: self.SAMPLE_LIMIT]
        if not samples:
            return {"ok": False, "error": f"no samples readable from {self.corpus}"}
        pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        total_loss, copy_loss, tokens = 0.0, 0.0, 0
        for sample in samples:
            text = PiscesLxToolsDataJsonSampleReader.extract_text(sample)
            ids = tokenizer.encode(text)
            ids = torch.tensor(list(ids), dtype=torch.long)[: self.MAX_EVAL_TOKENS]
            if ids.numel() < 2:
                continue
            input_ids = ids[:-1].unsqueeze(0)
            with torch.no_grad():
                output = model(input_ids=input_ids)
            # Score from the returned logits, never from the model's `loss`: that field is
            # the summed objective (routing + MTP + reasoning auxiliary terms included), so
            # exponentiating it produced "perplexity 5.2e21", and its decline can while away
            # while next-token prediction does not improve at all.
            logits = output["logits"] if isinstance(output, dict) else output.logits
            if logits is None:
                return {"ok": False, "error": "model returned no logits for labelled input"}
            next_token = F.cross_entropy(logits[0].float(), ids[1:], reduction="none")
            copy_target = F.cross_entropy(logits[0].float(), ids[:-1], reduction="none")
            vocab_width = logits.size(-1)
            total_loss += float(next_token.sum())
            copy_loss += float(copy_target.sum())
            tokens += int(next_token.numel())
        if tokens == 0:
            return {"ok": False, "error": "no token could be scored"}
        mean_loss = total_loss / tokens
        # A uniform model over the vocabulary scores ln(V) nats. Reporting the number
        # without it lets "perplexity 5e21" look like a metric bug instead of what it is:
        # a model predicting the next token worse than uniform, because its logits still
        # peak on the input token (see copy_target_loss).
        uniform_loss = math.log(vocab_width)
        return {
            "ok": math.isfinite(mean_loss) and mean_loss > 0,
            "mean_loss": round(mean_loss, self.LOSS_DIGITS),
            "perplexity": self._round(math.exp(min(mean_loss, self.PERPLEXITY_EXPONENT_CAP))),
            "uniform_baseline_loss": round(uniform_loss, self.LOSS_DIGITS),
            "copy_target_loss": round(copy_loss / tokens, self.LOSS_DIGITS),
            "beats_uniform": mean_loss < uniform_loss,
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
