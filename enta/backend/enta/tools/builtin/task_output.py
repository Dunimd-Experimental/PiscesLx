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



import asyncio
import json
from typing import Any

from enta.tools.base import build_tool
from enta.utils.task_store import get_store


async def _task_output_execute(**kwargs: Any) -> str:
    task_id = kwargs.get("task_id", "")
    block = kwargs.get("block", True)
    timeout_ms = kwargs.get("timeout", 30000)

    if not task_id:
        return "Error: task_id is required."

    record = get_store().get(task_id)
    if record is None:
        return f"Error: task '{task_id}' not found."

    if block and record.status in {"pending", "running"}:
        deadline = asyncio.get_event_loop().time() + (timeout_ms / 1000.0)
        while record.status in {"pending", "running"}:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.2, remaining))
            record = get_store().get(task_id)
            if record is None:
                return f"Error: task '{task_id}' disappeared."

    status_icon = {
        "pending": "⏳",
        "running": "\U0001f504",
        "completed": "✅",
        "failed": "❌",
        "cancelled": "\U0001f6ab",
        "killed": "\U0001f6ab",
    }.get(record.status, "❓")

    payload = {
        "id": record.id,
        "name": record.name,
        "status": record.status,
        "icon": status_icon,
        "result": record.result or "",
        "error": record.error or "",
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }
    if record.result and len(record.result) > 5000:
        payload["result"] = record.result[:5000] + "\n... (truncated)"
    if record.error and len(record.error) > 2000:
        payload["error"] = record.error[:2000] + "\n... (truncated)"
    return json.dumps(payload, ensure_ascii=False, indent=2)


EncreTaskOutputTool = build_tool(
    name="task_output",
    description=(
        "Retrieve output from a running or completed background task. "
        "Can block waiting for completion, or return current status."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the task to get output from",
            },
            "block": {
                "type": "boolean",
                "description": "Wait for task completion (default: true)",
                "default": True,
            },
            "timeout": {
                "type": "integer",
                "description": "Max wait time in milliseconds (default: 30000)",
                "default": 30000,
            },
        },
        "required": ["task_id"],
    },
    execute=_task_output_execute,
    intents=["general", "coding", "data"],
    is_concurrency_safe=lambda _: True,
)
