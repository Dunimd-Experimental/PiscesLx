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
import random
import time
from dataclasses import dataclass


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after: float = 0.0
    remaining: int = 0


class EncreRateLimiter:
    def __init__(
        self,
        per_minute: int = 60,
        per_hour: int = 500,
        per_day: int = 5000,
        max_concurrent: int = 10,
    ) -> None:
        self.per_minute = per_minute
        self.per_hour = per_hour
        self.per_day = per_day
        self.max_concurrent = max_concurrent
        self._windows: dict[str, list[float]] = {}
        self._concurrent_count: int = 0

    def check(self, tool_name: str) -> RateLimitResult:
        now = time.time()
        if tool_name not in self._windows:
            self._windows[tool_name] = []
        window = self._windows[tool_name]
        window[:] = [ts for ts in window if now - ts < 86400]
        minute_count = sum(1 for ts in window if now - ts < 60)
        if minute_count >= self.per_minute:
            oldest = min(ts for ts in window if now - ts < 60)
            return RateLimitResult(False, 60.0 - (now - oldest) + 0.1, 0)
        hour_count = sum(1 for ts in window if now - ts < 3600)
        if hour_count >= self.per_hour:
            return RateLimitResult(False, 3600.0 - (now - min(window)) + 0.1, 0)
        day_count = len(window)
        if day_count >= self.per_day:
            return RateLimitResult(False, 86400.0 - (now - window[0]) + 0.1, 0)
        window.append(now)
        return RateLimitResult(True, 0.0, max(0, self.per_day - day_count - 1))

    async def acquire_slot(self) -> None:
        while self._concurrent_count >= self.max_concurrent:
            await asyncio.sleep(0.05)
        self._concurrent_count += 1

    def release_slot(self) -> None:
        self._concurrent_count = max(0, self._concurrent_count - 1)

    def backoff(self, attempts: int = 0) -> float:
        base = min(2 ** attempts, 60)
        return base + random.random() * 0.5

    def reset(self) -> None:
        self._windows.clear()
        self._concurrent_count = 0

    @property
    def active_tools(self) -> list[str]:
        now = time.time()
        active: list[str] = []
        for name, window in self._windows.items():
            if any(now - ts < 86400 for ts in window):
                active.append(name)
        return active
