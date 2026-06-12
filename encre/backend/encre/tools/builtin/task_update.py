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

from typing import Any

from encre.task.manager import EncreTaskManager
from encre.tools.base import build_tool


async def _task_update_execute(**kwargs: Any) -> str:
    task_id = kwargs.get("task_id", "")
    status = kwargs.get("status")
    result = kwargs.get("result")
    error = kwargs.get("error")

    success = EncreTaskManager.update_task(
        task_id=task_id,
        status=status,
        result=result,
        error=error,
    )
    if success:
        return f"Task {task_id} updated successfully."
    return f"Error: Task not found: {task_id}"


EncreTaskUpdateTool = build_tool(
    name="task_update",
    description="Update the status or result of a task",
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {"type": "string", "description": "The task ID to update"},
            "status": {
                "type": "string",
                "enum": ["pending", "running", "completed", "failed", "killed"],
                "description": "New status",
            },
            "result": {"type": "string", "description": "Task result content"},
            "error": {"type": "string", "description": "Error message if failed"},
        },
        "required": ["task_id"],
    },
    execute=_task_update_execute,
    intents=["general", "coding", "data", "research"],
)
