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


async def _todo_execute(**kwargs: Any) -> str:
    todos = kwargs.get("todos", [])
    summary = kwargs.get("summary", "")
    reset = kwargs.get("reset", False)

    lines: list[str] = []
    if summary:
        lines.append(f"Summary: {summary}")
        lines.append("")

    if reset:
        lines.append("Todo list has been reset.")

    for item in todos:
        status_icon = {"pending": "○", "in_progress": "●", "completed": "✓"}
        icon = status_icon.get(item.get("status", "pending"), "○")
        priority_flag = {"high": "\U0001f534", "medium": "\U0001f7e1", "low": "\U0001f7e2"}
        flag = priority_flag.get(item.get("priority", "medium"), "\U0001f7e1")
        lines.append(f"{icon} {flag} {item.get('content', '')}")

    if not lines:
        return "No todo items."

    return "\n".join(lines)


EncreTodoTool = build_tool(
    name="todo",
    description="Create or update a structured todo list for task tracking",
    input_schema={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},  # noqa: E501
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["content", "status", "priority"],
                },
                "description": "List of todo items",
            },
            "summary": {
                "type": "string",
                "description": "Summary of work accomplished",
            },
            "reset": {
                "type": "boolean",
                "description": "Whether to reset the todo list",
            },
        },
        "required": ["todos"],
    },
    execute=_todo_execute,
    intents=["general", "coding", "data", "research"],
)
