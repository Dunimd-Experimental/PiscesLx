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

"""``train_orchestrate`` tool -- drive the end-to-end distill+RL training loop.

This is the trainer-side orchestration surface.  It does **not** re-implement
the training loop (that lives in ``tools/train/orchestrator.py:_run_enta_loop``);
instead it:

1. registers the four canonical phases (data_generation -> distill -> rl -> eval)
   as a dependency-ordered task graph in the in-process ledger, and
2. launches a single ``manage.py train --enta`` run (with the requested phase
   subset and iteration cap) as a detached background process via the shared
   ``train_launch`` helpers, returning a run handle for monitoring.

The heavy loop runs out-of-band; the agent keeps control and can poll progress
with ``train_status`` / ``train_checkpoints`` / ``train_eval``.
"""

from __future__ import annotations

import uuid
from typing import Any

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

logger = PiscesLxLogger(
    "EnTA.Tools.TrainOrchestrate",
    file_path=get_log_file("EnTA.Tools.TrainOrchestrate"),
    enable_file=True,
)

from enta.tools.base import build_tool
from enta.tools.builtin.train_launch import build_train_argv, launch_train
from enta.utils.task_store import get_store
from enta.training import TRAINING_PHASES, TRAINING_PHASE_GLOSS


async def _train_orchestrate_execute(**kwargs: Any) -> dict[str, Any]:
    phases = kwargs.get("phases") or TRAINING_PHASES
    max_iterations = kwargs.get("max_iterations")
    enta_config = kwargs.get("enta_config")
    model_size = kwargs.get("model_size")
    dataset = kwargs.get("dataset")
    resume_ckpt = kwargs.get("resume_ckpt")

    # 1) Register the phase roadmap in the ledger.
    store = get_store()
    tasks: list[dict[str, Any]] = []
    previous_id: str | None = None
    for phase in phases:
        if phase not in TRAINING_PHASES:
            return {
                "success": False,
                "error": f"Unknown phase '{phase}'. Expected one of {TRAINING_PHASES}.",
            }
        task_record = store.create(
            name=f"train:{phase}",
            description=TRAINING_PHASE_GLOSS[phase],
            task_type="train_orchestrate",
            prompt=f"{phase}: {TRAINING_PHASE_GLOSS[phase]}",
        )
        store.update(task_record.id, status="pending")
        if previous_id:
            store.update(task_record.id, dependencies=[previous_id])
        tasks.append(
            {
                "id": task_record.id,
                "name": f"train:{phase}",
                "status": "pending",
                "dependencies": [previous_id] if previous_id else [],
            }
        )
        previous_id = task_record.id

    # 2) Launch the EnTA loop as a detached background process.
    try:
        argv = build_train_argv(
            mode="enta",
            model_size=model_size,
            dataset=dataset,
            resume_ckpt=resume_ckpt,
            enta_config=enta_config,
            phases=phases,
            max_iterations=max_iterations,
        )
        info = launch_train(argv)
    except Exception as exc:
        logger.exception("train_orchestrate launch failed")
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "success": True,
        "run_id": info["run_id"],
        "pid": info["pid"],
        "log_path": info["log_path"],
        "phases": phases,
        "tasks": tasks,
        "note": (
            "End-to-end EnTA loop launched in background. Monitor with "
            "train_status, list checkpoints with train_checkpoints, and "
            "evaluate with train_eval once a checkpoint is produced."
        ),
    }


EncreTrainOrchestrateTool = build_tool(
    name="train_orchestrate",
    description=(
        "Drive the end-to-end distill+RL hybrid training loop. Registers the "
        "four canonical phases (data_generation -> distill -> rl -> eval) as a "
        "task roadmap, then launches 'manage.py train --enta' as a detached "
        "background run and returns a run handle. Poll progress with "
        "train_status / train_checkpoints / train_eval."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "phases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered phases to run (default all four).",
            },
            "max_iterations": {"type": "integer", "description": "Max outer-loop iterations for the EnTA run."},
            "model_size": {"type": "string"},
            "dataset": {"type": "string"},
            "resume_ckpt": {"type": "string"},
            "enta_config": {"type": "string", "description": "EnTA config yaml (default configs/teachers.yaml)."},
        },
        "required": [],
    },
    execute=_train_orchestrate_execute,
    intents=["general", "training"],
    category="training",
)
