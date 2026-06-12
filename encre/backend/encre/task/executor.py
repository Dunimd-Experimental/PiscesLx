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
from encre.task.types import EncreTask


class EncreTaskExecutor:
    def __init__(self) -> None:
        self._manager = EncreTaskManager()

    async def execute_task(self, task_id: str) -> str:
        task = self._manager.get_task(task_id)
        if task is None:
            return f"Error: Task not found: {task_id}"

        self._manager.update_task(task_id, status="running")

        try:
            if task.task_type == "bash":
                result = await self._execute_bash(task)
            elif task.task_type == "agent":
                result = await self._execute_agent(task)
            elif task.task_type == "workflow":
                result = await self._execute_workflow(task)
            else:
                result = f"Error: Unknown task type: {task.task_type}"

            self._manager.update_task(task_id, status="completed", result=result)
            return result
        except Exception as e:
            error_msg = str(e)
            self._manager.update_task(task_id, status="failed", error=error_msg)
            return f"Error: {error_msg}"

    async def _execute_bash(self, task: EncreTask) -> str:
        import asyncio
        import subprocess
        import sys

        kwargs = dict(
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        proc = await asyncio.create_subprocess_shell(
            task.prompt,
            **kwargs,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode("utf-8", errors="replace")
        if stderr:
            output += "\n" + stderr.decode("utf-8", errors="replace")
        return output

    async def _execute_agent(self, task: EncreTask) -> str:
        from encre.loop import EncreLoop
        from encre.session import EncreSession
        from encre.config import EncreConfig

        config = EncreConfig()
        if task.metadata:
            for key in ("model", "api_key", "base_url", "max_tokens"):
                if key in task.metadata:
                    setattr(config, key, task.metadata[key])

        session = EncreSession(config)
        session.add_message(
            "system",
            f"You are executing a subtask: {task.description or task.name}",
        )
        loop = EncreLoop(config, session)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task.prompt},
        ]
        return await loop._run_sub_agent(task.prompt, [])

    async def _execute_workflow(self, task: EncreTask) -> str:
        steps = task.prompt.split("\n")
        results: list[str] = []
        for step in steps:
            step = step.strip()
            if not step:
                continue
            result = await self._execute_bash(
                EncreTask(
                    id="",
                    name="step",
                    description="",
                    task_type="bash",
                    prompt=step,
                )
            )
            results.append(f"$ {step}\n{result}")
        return "\n---\n".join(results)