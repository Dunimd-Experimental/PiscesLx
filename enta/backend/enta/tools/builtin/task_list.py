#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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



from typing import Any

from enta.tools.base import build_tool
from enta.utils.task_store import get_store


async def _task_list_execute(**kwargs: Any) -> str:
    status = kwargs.get("status")
    records = get_store().list(status=status)

    if not records:
        return "No tasks found."

    lines: list[str] = []
    status_icon = {
        "pending": "○",
        "running": "●",
        "completed": "✓",
        "failed": "✗",
        "killed": "⊘",
        "cancelled": "⊘",
    }
    for r in records:
        icon = status_icon.get(r.status, "○")
        lines.append(f"{icon} {r.id}: {r.name} ({r.status})")

    return "\n".join(lines)


EncreTaskListTool = build_tool(
    name="task_list",
    description="List all tasks with optional status filter",
    input_schema={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["pending", "running", "completed", "failed", "killed", "cancelled"],
                "description": "Filter by status",
            },
        },
    },
    execute=_task_list_execute,
    intents=["general", "coding", "data", "research"],
    is_concurrency_safe=lambda _: True,
)
