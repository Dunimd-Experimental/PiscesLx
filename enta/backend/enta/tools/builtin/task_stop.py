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


async def _task_stop_execute(**kwargs: Any) -> str:
    task_id = kwargs.get("task_id", "")
    if not task_id:
        return "Error: task_id is required."

    record = get_store().get(task_id)
    if record is None:
        return f"Error: task '{task_id}' not found."

    if record.status == "completed":
        return f"Task '{task_id}' already completed."
    if record.status in {"cancelled", "killed"}:
        return f"Task '{task_id}' already {record.status}."

    get_store().update(
        task_id,
        status="cancelled",
        error="Stopped by user request",
    )
    return f"Task '{task_id}' stopped."


EncreTaskStopTool = build_tool(
    name="task_stop",
    description="Stop a running background task by its ID",
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the background task to stop",
            },
        },
        "required": ["task_id"],
    },
    execute=_task_stop_execute,
    intents=["general", "coding", "data"],
)
