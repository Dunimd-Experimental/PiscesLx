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
from enta.tools.builtin._shell_manager import BackgroundShellManager


async def _bash_output_execute(**kwargs: Any) -> str:
    shell_id = str(kwargs.get("id", "")).strip()
    if not shell_id:
        return "Error: id is required"

    mgr = BackgroundShellManager.instance()

    wait = bool(kwargs.get("wait", False))
    if wait:
        timeout = max(0.0, min(60.0, float(kwargs.get("wait_seconds", 5.0))))
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            snap = mgr.read_new_output(shell_id)
            if "error" in snap:
                return f"Error: {snap['error']}"
            if snap["stdout"] or snap["stderr"] or not snap["running"]:
                return json.dumps(snap, ensure_ascii=False)
            if asyncio.get_event_loop().time() >= deadline:
                return json.dumps(snap, ensure_ascii=False)
            await asyncio.sleep(0.15)

    snap = mgr.read_new_output(shell_id)
    if "error" in snap:
        return f"Error: {snap['error']}"
    return json.dumps(snap, ensure_ascii=False)


async def _bash_kill_execute(**kwargs: Any) -> str:
    shell_id = str(kwargs.get("id", "")).strip()
    if not shell_id:
        return "Error: id is required"
    force = bool(kwargs.get("force", False))
    result = await BackgroundShellManager.instance().kill(shell_id, force=force)
    if "error" in result:
        return f"Error: {result['error']}"
    return json.dumps(result, ensure_ascii=False)


async def _bash_list_execute(**kwargs: Any) -> str:
    shells = BackgroundShellManager.instance().list_shells()
    return json.dumps(shells, ensure_ascii=False)


EncreBashOutputTool = build_tool(
    name="bash_output",
    description=(
        "Read new output from a backgrounded shell (started via bash with "
        "run_in_background=true). Returns only bytes accumulated since the "
        "last read for that shell. With wait=true, blocks up to wait_seconds "
        "for new output or completion."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The shell id returned from bash(run_in_background=true).",
            },
            "wait": {
                "type": "boolean",
                "description": "If true, poll until new bytes arrive or the shell exits.",
            },
            "wait_seconds": {
                "type": "number",
                "description": "Max seconds to wait when wait=true (default 5, capped at 60).",
            },
        },
        "required": ["id"],
    },
    execute=_bash_output_execute,
    intents=["general", "coding"],
    is_concurrency_safe=lambda _: True,
)

EncreBashKillTool = build_tool(
    name="bash_kill",
    description=(
        "Stop a backgrounded shell. By default sends SIGTERM (or terminate "
        "on Windows). Pass force=true to escalate to SIGKILL after a short "
        "grace period."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The shell id to kill.",
            },
            "force": {
                "type": "boolean",
                "description": "Use SIGKILL / hard-terminate.",
            },
        },
        "required": ["id"],
    },
    execute=_bash_kill_execute,
    intents=["general", "coding"],
)

EncreBashListTool = build_tool(
    name="bash_list",
    description=(
        "List all backgrounded shells (running and exited) tracked in this "
        "session. Returns ids, commands, running flags, and exit codes."
    ),
    input_schema={
        "type": "object",
        "properties": {},
    },
    execute=_bash_list_execute,
    intents=["general", "coding"],
    is_concurrency_safe=lambda _: True,
)
