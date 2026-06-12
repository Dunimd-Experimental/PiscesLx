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
from typing import Any

from encre.agent import EncreAgent

logger = logging.getLogger("encre.learning")


class LearningEngine:
    TOOL_CALL_THRESHOLD = 5

    def __init__(self, agent: EncreAgent) -> None:
        self._agent = agent
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._running = True
        logger.info("Learning engine started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Learning engine stopped")

    async def analyze_run(self, tool_names: list[str], prompt: str) -> None:
        if not self._running:
            return
        if len(tool_names) < self.TOOL_CALL_THRESHOLD:
            return
        task = asyncio.create_task(self._crystallize(tool_names, prompt))
        self._tasks.append(task)
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)

    async def _crystallize(self, tool_names: list[str], prompt: str) -> None:
        from encre.learning.skill_generator import SkillGenerator
        generator = SkillGenerator(self._agent)
        skill_def = generator.generate(tool_names, prompt)
        if skill_def is None:
            return
        await generator.register(skill_def)