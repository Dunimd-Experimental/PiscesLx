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

"""``train_checkpoints`` tool -- list and resolve training checkpoints.

Checkpoints written by the pipeline follow the ``checkpoint_step_<step>.pt``
naming convention (see ``tools/train/pipeline.py`` and ``run_reporter.py``).
This tool lists them under a given ``output_dir`` and returns the latest one,
which is the natural input for ``train_eval`` or for ``train_launch`` with
``resume_ckpt``.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

logger = PiscesLxLogger(
    "EnTA.Tools.TrainCheckpoints",
    file_path=get_log_file("EnTA.Tools.TrainCheckpoints"),
    enable_file=True,
)

from enta.tools.base import build_tool


async def _train_checkpoints_execute(**kwargs: Any) -> dict[str, Any]:
    output_dir = kwargs.get("output_dir", "")
    action = kwargs.get("action", "list")

    if not output_dir or not os.path.isdir(output_dir):
        return {
            "success": False,
            "error": f"output_dir '{output_dir}' does not exist or was not provided.",
        }

    pattern = os.path.join(output_dir, "checkpoint_step_*.pt")
    checkpoints = sorted(glob.glob(pattern))

    if action == "latest":
        latest = checkpoints[-1] if checkpoints else None
        return {
            "success": True,
            "action": "latest",
            "latest": latest,
            "count": len(checkpoints),
        }

    return {
        "success": True,
        "action": "list",
        "checkpoints": checkpoints,
        "latest": checkpoints[-1] if checkpoints else None,
        "count": len(checkpoints),
    }


EncreTrainCheckpointsTool = build_tool(
    name="train_checkpoints",
    description=(
        "List or resolve training checkpoints under an output directory. "
        "Checkpoints follow 'checkpoint_step_<step>.pt'. Use action='latest' to "
        "get the most recent checkpoint path (for train_eval or resuming a run)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "output_dir": {"type": "string", "description": "Directory containing checkpoint_step_*.pt files."},
            "action": {
                "type": "string",
                "enum": ["list", "latest"],
                "description": "list = all checkpoints; latest = newest checkpoint path only.",
            },
        },
        "required": ["output_dir"],
    },
    execute=_train_checkpoints_execute,
    intents=["general", "training"],
    category="training",
    is_concurrency_safe=lambda _: True,
)
