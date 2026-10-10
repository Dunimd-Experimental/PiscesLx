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

"""``train_status`` tool -- read the live status of a training run.

Combines two signals into one call:

* the tail of the run log (from ``train_launch``'s ``log_path``, or resolved
  from a ``run_id``),
* the most recent structured metrics snapshot
  (``training_metrics_*.json`` written by ``tools/train/monitoring.py``),

and returns the latest ``loss`` / ``learning_rate`` / ``grad_norm`` / ``step``
plus the raw log tail so the trainer can self-assess convergence.
"""

from __future__ import annotations

import glob
import json
import pathlib
from typing import Any

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

logger = PiscesLxLogger(
    "EnTA.Tools.TrainStatus",
    file_path=get_log_file("EnTA.Tools.TrainStatus"),
    enable_file=True,
)

from enta.tools.base import build_tool

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[5]
RUNS_ROOT = PROJECT_ROOT / "logs" / "train_runs"


def _resolve_log_path(log_path: str | None, run_id: str | None) -> str | None:
    if log_path:
        return log_path
    if run_id:
        candidate = RUNS_ROOT / run_id / "train.log"
        if candidate.exists():
            return str(candidate)
    return None


def _read_tail(log_path: str, tail_lines: int) -> str:
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return f"(cannot read log: {exc})"
    if not lines:
        return "(empty log)"
    return "".join(lines[-tail_lines:])


def _latest_metrics() -> dict[str, Any] | None:
    files = sorted(glob.glob(str(PROJECT_ROOT / "logs" / "training_metrics_*.json")))
    if not files:
        return None
    try:
        with open(files[-1], "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed reading %s: %s", files[-1], exc)
        return None
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return None


async def _train_status_execute(**kwargs: Any) -> dict[str, Any]:
    log_path = _resolve_log_path(kwargs.get("log_path"), kwargs.get("run_id"))
    tail_lines = int(kwargs.get("tail_lines", 60))

    if not log_path:
        return {
            "success": False,
            "error": "Provide 'log_path' (from train_launch) or 'run_id'.",
        }

    raw_tail = _read_tail(log_path, tail_lines)
    metrics = _latest_metrics()

    result: dict[str, Any] = {
        "success": True,
        "log_path": log_path,
        "latest_loss": None,
        "lr": None,
        "grad_norm": None,
        "step": None,
        "raw_tail": raw_tail,
    }
    if metrics:
        result["latest_loss"] = metrics.get("loss")
        result["lr"] = metrics.get("learning_rate")
        result["grad_norm"] = metrics.get("grad_norm")
        result["step"] = metrics.get("step", metrics.get("global_step"))
    return result


EncreTrainStatusTool = build_tool(
    name="train_status",
    description= (
        "Read the live status of a training run. Returns the latest structured "
        "metrics (loss, learning_rate, grad_norm, step) parsed from "
        "training_metrics_*.json plus the tail of the run log. Pass the "
        "log_path returned by train_launch, or a run_id to resolve it."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "log_path": {"type": "string", "description": "Path to the run log (from train_launch)."},
            "run_id": {"type": "string", "description": "Run id returned by train_launch (alternative to log_path)."},
            "tail_lines": {"type": "integer", "description": "Number of log lines to return (default 60)."},
        },
        "required": [],
    },
    execute=_train_status_execute,
    intents=["general", "training"],
    category="training",
    is_concurrency_safe=lambda _: True,
)
