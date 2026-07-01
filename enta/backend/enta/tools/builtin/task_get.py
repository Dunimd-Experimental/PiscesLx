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


async def _task_get_execute(**kwargs: Any) -> str:
    task_id = kwargs.get("task_id", "")
    record = get_store().get(task_id)
    if record is None:
        return f"Error: Task not found: {task_id}"

    lines = [
        f"ID: {record.id}",
        f"Name: {record.name}",
        f"Type: {record.task_type}",
        f"Status: {record.status}",
        f"Description: {record.description}",
    ]
    if record.result:
        lines.append(f"Result: {record.result[:500]}")
    if record.error:
        lines.append(f"Error: {record.error}")
    if record.parent_id:
        lines.append(f"Parent: {record.parent_id}")
    return "\n".join(lines)


EncreTaskGetTool = build_tool(
    name="task_get",
    description="Get details of a specific task by ID",
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to retrieve",
            },
        },
        "required": ["task_id"],
    },
    execute=_task_get_execute,
    intents=["general", "coding", "data", "research"],
    is_concurrency_safe=lambda _: True,
)
