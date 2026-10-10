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

"""``train_eval`` tool -- evaluate a training checkpoint via ``YvEntaTrainer``.

Wraps the model-side ``YvEntaTrainer.from_yvconfig(...).evaluate(checkpoint_path)``
call.  ``evaluate`` returns a metadata-level capability profile *without* loading
the full model, so this tool is cheap enough to call repeatedly during the eval
phase of the training loop.  Project root and the ``enta/backend`` package root
are added to ``sys.path`` on demand so the model-side module is importable from
the agent tool.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

logger = PiscesLxLogger(
    "EnTA.Tools.TrainEval",
    file_path=get_log_file("EnTA.Tools.TrainEval"),
    enable_file=True,
)

from enta.tools.base import build_tool

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[5]
ENTA_BACKEND = PROJECT_ROOT / "enta" / "backend"


def _ensure_imports() -> tuple:
    """Make ``model.*`` and ``enta`` importable, then return the needed symbols."""
    for p in (str(PROJECT_ROOT), str(ENTA_BACKEND)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from model.agentic.enta import YvEntaTrainer  # noqa: F401  (heavy import)
    from model.config import YvConfig

    return YvEntaTrainer, YvConfig


async def _train_eval_execute(**kwargs: Any) -> dict[str, Any]:
    checkpoint_path = kwargs.get("checkpoint_path", "")
    enta_config = kwargs.get("enta_config") or str(
        PROJECT_ROOT / "configs" / "teachers.yaml"
    )

    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return {
            "success": False,
            "error": f"checkpoint_path '{checkpoint_path}' does not exist.",
        }

    try:
        YvEntaTrainer, YvConfig = _ensure_imports()
        cfg = YvConfig.load_encre_yaml()
        trainer = YvEntaTrainer.from_yvconfig(cfg)
        profile = trainer.evaluate(checkpoint_path)
    except Exception as exc:
        logger.exception("train_eval failed")
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    if not isinstance(profile, dict):
        profile = {"profile": profile}
    return {"success": True, "checkpoint_path": checkpoint_path, **profile}


EncreTrainEvalTool = build_tool(
    name="train_eval",
    description=(
        "Evaluate a training checkpoint using YvEntaTrainer.evaluate (a "
        "metadata-level capability profile; the full model is not loaded, so "
        "this is cheap and repeatable). Pass the checkpoint path returned by "
        "train_checkpoints and optionally an EnTA config path."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "checkpoint_path": {"type": "string", "description": "Path to a checkpoint_step_*.pt file (from train_checkpoints)."},
            "enta_config": {"type": "string", "description": "EnTA config yaml (default configs/teachers.yaml)."},
        },
        "required": ["checkpoint_path"],
    },
    execute=_train_eval_execute,
    intents=["general", "training"],
    category="training",
)
