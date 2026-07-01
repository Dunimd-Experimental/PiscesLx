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

from typing import Any

from enta.tools.base import build_tool
from enta.utils.task_store import get_store


async def _task_create_execute(**kwargs: Any) -> str:
    name = kwargs.get("name", "")
    description = kwargs.get("description", "")
    task_type = kwargs.get("task_type", "bash")
    prompt = kwargs.get("prompt", "")
    parent_id = kwargs.get("parent_id")

    record = get_store().create(
        name=name,
        description=description,
        task_type=task_type,
        prompt=prompt,
        parent_id=parent_id,
    )
    return f"Task created: {record.id}"


EncreTaskCreateTool = build_tool(
    name="task_create",
    description="Create a new sub-task tracked in the process task ledger.",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Task name"},
            "description": {"type": "string", "description": "Task description"},
            "task_type": {
                "type": "string",
                "enum": ["bash", "agent", "workflow"],
                "description": "Type of task",
            },
            "prompt": {"type": "string", "description": "Task prompt/instructions"},
            "parent_id": {"type": "string", "description": "Parent task ID"},
        },
        "required": ["name", "task_type", "prompt"],
    },
    execute=_task_create_execute,
    intents=["general", "coding", "data", "research"],
)
