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

"""``train_plan`` tool -- inject the EnTA trainer persona and record a
training goal as a dependency-ordered task graph in the in-process ledger.

This is the trainer-side planning surface: unlike ``workflow`` (which merely
records a generic goal), ``train_plan`` first surfaces the trainer persona and
then maps the goal onto the four canonical training phases defined in
``enta.training``.  Each phase is registered as a pending task so the model can
execute them with the dedicated ``train_*`` tools and mark progress later.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

logger = PiscesLxLogger(
    "EnTA.Tools.TrainPlan",
    file_path=get_log_file("EnTA.Tools.TrainPlan"),
    enable_file=True,
)

from enta.tools.base import build_tool
from enta.utils.task_store import get_store
from enta.training import (
    TRAINER_SYSTEM_PROMPT,
    TRAINING_PHASES,
    TRAINING_PHASE_GLOSS,
)


def _parse_inline_steps(goal: str) -> list[str]:
    """Extract a numbered / bulleted / paragraph list embedded in ``goal``."""
    text = (goal or "").strip()
    if not text:
        return []

    numbered: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(?:\d+[\.\)、]\s+|[-*]\s+)(.+)$", line)
        if m:
            numbered.append(m.group(1).strip())
    if numbered:
        return numbered

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    return [text]


def _phase_tasks(goal: str) -> list[tuple[str, str]]:
    """Return (phase, description) pairs, either from an explicit phase list
    embedded in the goal or from the standard four-phase ordering."""
    steps = _parse_inline_steps(goal)
    if steps and any(s.strip().lower() in TRAINING_PHASES for s in steps):
        pairs: list[tuple[str, str]] = []
        for s in steps:
            key = s.strip().lower()
            if key in TRAINING_PHASES:
                pairs.append((key, TRAINING_PHASE_GLOSS[key]))
        return pairs
    return [(p, TRAINING_PHASE_GLOSS[p]) for p in TRAINING_PHASES]


async def _train_plan_execute(**kwargs: Any) -> dict[str, Any]:
    goal = kwargs.get("goal", "")
    include_persona = bool(kwargs.get("persona", True))

    if not goal:
        return {
            "success": False,
            "error": "'goal' is required.",
        }

    store = get_store()
    phase_pairs = _phase_tasks(goal)

    tasks: list[dict[str, Any]] = []
    previous_id: str | None = None
    for phase, description in phase_pairs:
        task_record = store.create(
            name=f"train:{phase}",
            description=description,
            task_type="train_plan",
            prompt=f"{phase}: {description}",
        )
        store.update(task_record.id, status="pending")
        if previous_id:
            store.update(task_record.id, dependencies=[previous_id])
        tasks.append(
            {
                "id": task_record.id,
                "name": f"train:{phase}",
                "description": description,
                "status": "pending",
                "dependencies": [previous_id] if previous_id else [],
            }
        )
        previous_id = task_record.id

    return {
        "success": True,
        "persona": TRAINER_SYSTEM_PROMPT if include_persona else None,
        "phases": [p for p, _ in phase_pairs],
        "tasks": tasks,
    }


EncreTrainPlanTool = build_tool(
    name="train_plan",
    description=(
        "Inject the EnTA trainer persona and decompose a training goal into "
        "the four canonical, dependency-ordered phases (data_generation -> "
        "distill -> rl -> eval), registering each as a pending task in the "
        "in-process ledger. The persona string is returned so the model can "
        "adopt the trainer identity; the phases are then driven by the other "
        "train_* tools. Returns a runnable training roadmap."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "The training objective. May be a free-form goal (mapped "
                    "onto the four default phases) or a numbered list that "
                    "names a subset of phases (data_generation / distill / rl "
                    "/ eval) to run."
                ),
            },
            "persona": {
                "type": "boolean",
                "description": "Whether to return the trainer persona in the result (default true).",
            },
        },
        "required": ["goal"],
    },
    execute=_train_plan_execute,
    intents=["general", "training"],
    category="training",
)
