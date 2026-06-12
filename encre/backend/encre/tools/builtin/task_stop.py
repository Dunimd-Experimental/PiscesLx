#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
# ...
# Licensed under the Apache License, Version 2.0.

from typing import Any

from encre.tools.base import build_tool
from encre.utils.types import TaskStatus


async def _task_stop_execute(**kwargs: Any) -> str:
    from encre.task.manager import EncreTaskManager

    task_id = kwargs.get("task_id", "")
    if not task_id:
        return "Error: task_id is required."

    task = EncreTaskManager.get_task(task_id)
    if task is None:
        return f"Error: task '{task_id}' not found."

    if task.status == TaskStatus.COMPLETED:
        return f"Task '{task_id}' already completed."
    if task.status == TaskStatus.CANCELLED:
        return f"Task '{task_id}' already cancelled."

    EncreTaskManager.update_task(task_id, status=TaskStatus.CANCELLED, error="Stopped by user request")
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
