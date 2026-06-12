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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from encre.swarm.teammate import EncreTeammate, TeammateHandle


@dataclass
class SwarmProgress:
    total: int = 0
    completed: int = 0
    failed: int = 0
    running: int = 0
    pending: int = 0
    killed: int = 0
    details: dict[str, str] = field(default_factory=dict)


class EncreSwarmManager:
    def __init__(self, max_concurrent: int = 30):
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._teammates: "dict[str, EncreTeammate]" = {}
        self._handles: "dict[str, TeammateHandle]" = {}

    async def spawn(self, teammate: "EncreTeammate") -> "TeammateHandle":
        self._teammates[teammate.teammate_id] = teammate
        async with self._semaphore:
            handle = await teammate.run()
            self._handles[teammate.teammate_id] = handle
            return handle

    async def spawn_many(self, teammates: "list[EncreTeammate]") -> "list[TeammateHandle]":
        coros = [self.spawn(t) for t in teammates]
        results = await asyncio.gather(*coros, return_exceptions=True)
        handles: "list[TeammateHandle]" = []
        for r in results:
            if isinstance(r, Exception):
                from encre.swarm.teammate import TeammateHandle
                handles.append(
                    TeammateHandle(
                        teammate_id="error",
                        name="spawn_failed",
                        status="failed",
                        error=str(r),
                    )
                )
            else:
                handles.append(r)
        return handles

    async def await_all(self, timeout: float | None = None) -> "list[TeammateHandle]":
        tasks = []
        for h in self._handles.values():
            if h._task is not None:
                tasks.append(h._task)
        if timeout is not None:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        else:
            await asyncio.gather(*tasks, return_exceptions=True)
        return list(self._handles.values())

    async def await_any(self) -> "TeammateHandle | None":
        tasks = [h._task for h in self._handles.values() if h._task is not None]
        if not tasks:
            return None
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            for h in self._handles.values():
                if h._task is task:
                    return h
        return None

    async def cancel_all(self) -> None:
        for t in self._teammates.values():
            await t.cancel()
        self._teammates.clear()
        self._handles.clear()

    def get_progress(self) -> SwarmProgress:
        progress = SwarmProgress()
        for h in self._handles.values():
            progress.total += 1
            if h.status == "completed":
                progress.completed += 1
            elif h.status == "failed":
                progress.failed += 1
            elif h.status == "running":
                progress.running += 1
            elif h.status == "pending":
                progress.pending += 1
            elif h.status == "killed":
                progress.killed += 1
            progress.details[h.teammate_id] = h.status
        return progress

    def get_handle(self, teammate_id: str) -> "TeammateHandle | None":
        return self._handles.get(teammate_id)
