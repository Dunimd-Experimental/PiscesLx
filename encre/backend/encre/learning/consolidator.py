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
import logging

from encre.agent import EncreAgent

logger = logging.getLogger("encre.learning.consolidator")


class MemoryConsolidator:
    def __init__(self, agent: EncreAgent, interval: int = 3600) -> None:
        self._agent = agent
        self._interval = interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Memory consolidator started (interval=%ds)", self._interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Memory consolidator stopped")

    async def _loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._interval)
            if not self._running:
                break
            await self._consolidate()

    async def consolidate_now(self) -> None:
        await self._consolidate()

    async def _consolidate(self) -> None:
        memory_system = getattr(self._agent, "memory_system", None)
        if memory_system is None:
            return

        try:
            if hasattr(memory_system, "consolidate") and callable(memory_system.consolidate):
                result = await memory_system.consolidate()
                logger.info("Memory consolidation completed: %s", result)
        except Exception as e:
            logger.warning("Memory consolidation failed: %s", e)