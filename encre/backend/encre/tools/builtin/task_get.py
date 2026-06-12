#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

from encre.task.manager import EncreTaskManager
from encre.tools.base import build_tool


async def _task_get_execute(**kwargs: Any) -> str:
    task_id = kwargs.get("task_id", "")
    task = EncreTaskManager.get_task(task_id)
    if task is None:
        return f"Error: Task not found: {task_id}"

    lines = [
        f"ID: {task.id}",
        f"Name: {task.name}",
        f"Type: {task.task_type}",
        f"Status: {task.status}",
        f"Description: {task.description}",
    ]
    if task.result:
        lines.append(f"Result: {task.result[:500]}")
    if task.error:
        lines.append(f"Error: {task.error}")
    if task.parent_id:
        lines.append(f"Parent: {task.parent_id}")
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
