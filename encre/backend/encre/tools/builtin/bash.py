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

import asyncio
import json
import os
from typing import Any

from encre.tools.base import build_tool
from encre.tools.builtin._shell_manager import BackgroundShellManager


async def _bash_execute(**kwargs: Any) -> str:
    command = kwargs.get("command", "")
    if not command:
        return "Error: command is required"

    if bool(kwargs.get("run_in_background", False)):
        cwd = kwargs.get("cwd") or None
        mgr = BackgroundShellManager.instance()
        try:
            rec = await mgr.spawn(command, cwd=cwd)
        except Exception as exc:
            return f"Error spawning background shell: {exc}"
        return json.dumps({
            "id": rec.id,
            "running": True,
            "command": rec.command,
            "cwd": rec.cwd,
            "started_at": rec.started_at,
            "hint": "Use bash_output with this id to read output, bash_kill to stop.",
        }, ensure_ascii=False)

    timeout = int(kwargs.get("timeout", 120))
    cwd = kwargs.get("cwd") or None
    process = None
    try:
        # Use asyncio subprocess instead of the native sandbox so that
        # task.cancel() / CancelledError can actually kill the running
        # process when the user pauses -- run_in_executor would not.
        process = await asyncio.create_subprocess_exec(
            "bash" if os.name != "nt" else "cmd.exe",
            "-c" if os.name != "nt" else "/c",
            command,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception:
                pass
            return f"Error: Command timed out after {timeout}s"
        except asyncio.CancelledError:
            # User paused -- kill the process immediately
            try:
                process.kill()
            except Exception:
                pass
            raise

        output = (stdout or b"").decode("utf-8", errors="replace")
        stderr_text = (stderr or b"").decode("utf-8", errors="replace")
        if stderr_text:
            if output:
                output += "\n"
            output += stderr_text
        exit_code = await process.wait()
        if exit_code != 0:
            output += f"\nCommand exited with code {exit_code}"
        return output
    except asyncio.CancelledError:
        # Ensure process is killed when the task is cancelled
        if process and process.returncode is None:
            try:
                process.kill()
            except Exception:
                pass
        raise
    except Exception as e:
        if process and process.returncode is None:
            try:
                process.kill()
            except Exception:
                pass
        return f"Error executing command: {e}"


EncreBashTool = build_tool(
    name="bash",
    description=(
        "Execute a shell command. By default runs synchronously and returns "
        "the full stdout/stderr. Set run_in_background=true to spawn the "
        "command, return a shell id immediately, and stream output later via "
        "bash_output / bash_kill. Use background mode for dev servers, "
        "watchers, or any long-running command."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds for foreground execution (default: 120). Ignored in background mode.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command",
            },
            "run_in_background": {
                "type": "boolean",
                "description": (
                    "If true, spawn the command as a backgrounded shell and "
                    "return a shell id. Use bash_output to read its output "
                    "and bash_kill to stop it."
                ),
            },
            "dangerous": {
                "type": "boolean",
                "description": "Explicitly mark as dangerous to bypass safety checks",
            },
        },
        "required": ["command"],
    },
    execute=_bash_execute,
    intents=["general", "coding", "data"],
)
