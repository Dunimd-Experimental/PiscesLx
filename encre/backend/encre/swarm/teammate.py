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
import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from encre.swarm.mailbox import EncreMailbox

if TYPE_CHECKING:
    from encre.tools.base import EncreTool
    from encre.config import EncreConfig


@dataclass
class TeammateHandle:
    teammate_id: str
    name: str
    status: str = "pending"
    result: str = ""
    error: str = ""
    mailbox: EncreMailbox | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)


class EncreTeammate:
    def __init__(
        self,
        name: str,
        task: str,
        tools: "list[EncreTool] | None" = None,
        config: "EncreConfig | None" = None,
        allowed_tools: "list[str] | None" = None,
    ):
        self.teammate_id = str(uuid.uuid4())
        self.name = name
        self.task = task
        self.tools = tools or []
        self.config = config
        self.allowed_tools = allowed_tools
        self.mailbox = EncreMailbox(owner_id=f"{name}:{self.teammate_id[:8]}")
        self._run_task: asyncio.Task | None = None
        self._run_handle: TeammateHandle | None = None

    async def run(self) -> TeammateHandle:
        handle = TeammateHandle(
            teammate_id=self.teammate_id,
            name=self.name,
            status="running",
            mailbox=self.mailbox,
        )
        self._run_task = asyncio.create_task(self._run(handle))
        handle._task = self._run_task
        self._run_handle = handle
        return handle

    async def _run(self, handle: TeammateHandle) -> None:
        try:
            from encre.agent import EncreAgent
            from encre.config import EncreConfig
            from encre.tools.registry import ToolRegistry

            config = self.config or EncreConfig(max_turns=15)
            tool_registry = ToolRegistry()
            for tool in self.tools:
                tool_registry.register(tool)
            agent = EncreAgent(config=config, tool_registry=tool_registry)

            parts: list[str] = []
            from encre.utils.types import TextDelta, ToolResult
            async for event in agent.run(self.task):
                if isinstance(event, TextDelta) and event.text:
                    parts.append(event.text)
                elif isinstance(event, ToolResult):
                    content = event.content if event.content else str(event.is_error)
                    await self.mailbox.send(self.mailbox, content)
            handle.result = "".join(parts)
            handle.status = "completed"
        except asyncio.CancelledError:
            handle.status = "cancelled"
            handle.error = "Cancelled by user"
            # Re-raise so the caller (e.g. SwarmManager or asyncio.gather)
            # knows this task was cancelled rather than completed.
            raise
        except Exception as e:
            handle.error = str(e)
            handle.status = "failed"

    async def cancel(self) -> None:
        """Cancel the teammate's running agent task.

        Sends a cancellation request to the underlying asyncio Task and
        updates the handle status to 'cancelled'.  Safe to call even if
        the teammate has not been started or has already finished.
        """
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
        if self._run_handle is not None:
            self._run_handle.status = "cancelled"
