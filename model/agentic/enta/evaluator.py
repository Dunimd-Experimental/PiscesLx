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

"""EntaEvaluator — evaluates a model checkpoint without loading the full model.

The evaluation is lightweight: it inspects checkpoint metadata (loss curves,
step count, config snapshot) and optionally runs a small probe set, but
never loads the full student model into memory.
"""

import json
import os
from typing import Any, Dict


class EntaEvaluator:
    """Lightweight evaluator for model checkpoints.

    Does NOT load the full model.  Reads metadata from the checkpoint
    directory to build a capability profile dict.
    """

    def __init__(self) -> None:
        pass

    def evaluate(self, checkpoint_path: str, cfg: dict | None = None) -> Dict[str, Any]:
        """Build a capability profile from checkpoint metadata.

        The returned dict contains:
        - ``training_steps`` — how many steps the model was trained for
        - ``avg_loss`` — final average loss (from training log, if available)
        - ``perplexity`` — derived from avg_loss
        - ``capability_score`` — a synthetic [0, 1] score based on the above

        Args:
            checkpoint_path: Path to the checkpoint directory.
            cfg: Optional config dict (unused in base evaluator, but
                passed through for extensibility).

        Returns:
            A dict with capability profile keys.
        """
        profile: Dict[str, Any] = {
            "training_steps": 0,
            "avg_loss": float("inf"),
            "perplexity": float("inf"),
            "capability_score": 0.0,
        }

        if not os.path.isdir(checkpoint_path):
            return profile

        # Try to read training metadata from a JSON log in the checkpoint dir.
        meta_path = os.path.join(checkpoint_path, "training_meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                profile["training_steps"] = int(meta.get("step", 0))
                loss_val = meta.get("loss", None)
                if loss_val is not None and isinstance(loss_val, (int, float)):
                    profile["avg_loss"] = float(loss_val)
                    profile["perplexity"] = float(
                        min(1e6, 2.71828 ** float(loss_val))
                    )
            except (json.JSONDecodeError, OSError):
                pass

        # Try the simpler training.log fallback.
        if profile["avg_loss"] == float("inf"):
            log_path = os.path.join(checkpoint_path, "training.log")
            if os.path.isfile(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        last_loss = None
                        for line in f:
                            if "loss" in line.lower():
                                parts = line.split()
                                for i, p in enumerate(parts):
                                    if p == "loss" and i + 1 < len(parts):
                                        try:
                                            last_loss = float(parts[i + 1].rstrip(","))
                                        except ValueError:
                                            pass
                        if last_loss is not None:
                            profile["avg_loss"] = last_loss
                            profile["perplexity"] = float(
                                min(1e6, 2.71828 ** last_loss)
                            )
                except OSError:
                    pass

        # Compute a synthetic capability score.
        if profile["avg_loss"] < float("inf") and profile["avg_loss"] > 0:
            # Lower loss → higher score, bounded in [0, 1].
            score = 1.0 / (1.0 + profile["avg_loss"])
            profile["capability_score"] = round(float(score), 4)

        return profile
