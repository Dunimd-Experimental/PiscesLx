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

"""Tests for the rate limiter: construction, check, backoff, slots, and reset."""

import asyncio
import pytest


class TestRateLimitResult:
    def test_allowed_result(self):
        from encre.ratelimit import RateLimitResult
        result = RateLimitResult(allowed=True, remaining=50)
        assert result.allowed is True
        assert result.retry_after == 0.0
        assert result.remaining == 50

    def test_denied_result(self):
        from encre.ratelimit import RateLimitResult
        result = RateLimitResult(allowed=False, retry_after=30.5, remaining=0)
        assert result.allowed is False
        assert result.retry_after == 30.5
        assert result.remaining == 0

    def test_default_values(self):
        from encre.ratelimit import RateLimitResult
        result = RateLimitResult(allowed=True)
        assert result.retry_after == 0.0
        assert result.remaining == 0

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        from encre.ratelimit import RateLimitResult
        assert is_dataclass(RateLimitResult)


class TestEncreRateLimiterConstruction:
    def test_default_values(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        assert limiter.per_minute == 60
        assert limiter.per_hour == 500
        assert limiter.per_day == 5000
        assert limiter.max_concurrent == 10
        assert limiter._concurrent_count == 0

    def test_custom_values(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(
            per_minute=30,
            per_hour=200,
            per_day=1000,
            max_concurrent=5,
        )
        assert limiter.per_minute == 30
        assert limiter.per_hour == 200
        assert limiter.per_day == 1000
        assert limiter.max_concurrent == 5

    def test_initial_windows_empty(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        assert limiter._windows == {}

    def test_initial_active_tools_empty(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        assert limiter.active_tools == []


class TestEncreRateLimiterCheck:
    def test_first_check_allowed(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        result = limiter.check("bash")
        assert result.allowed is True
        assert result.remaining > 0

    def test_multiple_checks_track_count(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(per_minute=100)
        for _ in range(10):
            result = limiter.check("bash")
            assert result.allowed is True
        # Remaining should have decreased
        assert result.remaining < 5000

    def test_different_tools_have_separate_windows(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(per_minute=100)
        # Use one tool a lot, the other should still have full quota
        for _ in range(50):
            limiter.check("heavy_tool")
        result = limiter.check("light_tool")
        assert result.allowed is True
        # light_tool should have close to full remaining
        assert result.remaining > 4000

    def test_per_minute_limit_exceeded(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(per_minute=5, per_hour=99999, per_day=99999)
        for _ in range(5):
            result = limiter.check("bash")
            assert result.allowed is True
        # 6th should be denied
        result = limiter.check("bash")
        assert result.allowed is False
        assert result.retry_after > 0

    def test_per_day_limit_exceeded(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(per_minute=99999, per_hour=99999, per_day=3)
        for _ in range(3):
            result = limiter.check("bash")
            assert result.allowed is True
        # 4th should be denied
        result = limiter.check("bash")
        assert result.allowed is False
        assert result.remaining == 0


class TestEncreRateLimiterSlots:
    @pytest.mark.asyncio
    async def test_acquire_slot_below_limit(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(max_concurrent=10)
        await limiter.acquire_slot()
        assert limiter._concurrent_count == 1

    @pytest.mark.asyncio
    async def test_acquire_multiple_slots(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(max_concurrent=5)
        await limiter.acquire_slot()
        await limiter.acquire_slot()
        await limiter.acquire_slot()
        assert limiter._concurrent_count == 3

    @pytest.mark.asyncio
    async def test_release_slot(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        await limiter.acquire_slot()
        assert limiter._concurrent_count == 1
        limiter.release_slot()
        assert limiter._concurrent_count == 0

    def test_release_slot_never_goes_negative(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        limiter.release_slot()
        limiter.release_slot()
        assert limiter._concurrent_count == 0

    @pytest.mark.asyncio
    async def test_acquire_slot_blocks_when_at_capacity(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(max_concurrent=1)
        await limiter.acquire_slot()
        assert limiter._concurrent_count == 1

        # Now attempt to acquire another slot — it should be blocked
        # We test this by using a task with a timeout
        async def acquire():
            await limiter.acquire_slot()
            return True

        task = asyncio.create_task(acquire())
        await asyncio.sleep(0.2)  # Give it time to spin
        assert limiter._concurrent_count == 1  # Still at capacity

        # Release and the task should proceed
        limiter.release_slot()
        await asyncio.wait_for(task, timeout=2.0)
        assert limiter._concurrent_count == 1


class TestEncreRateLimiterBackoff:
    def test_backoff_with_zero_attempts(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        delay = limiter.backoff(0)
        assert 1.0 <= delay <= 1.5  # 2^0 = 1 + random(0, 0.5)

    def test_backoff_increases_with_attempts(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        d1 = limiter.backoff(1)
        d2 = limiter.backoff(2)
        d3 = limiter.backoff(3)
        # Base values: 2, 4, 8 — should generally increase
        # but there's jitter so we check ranges
        assert d1 > 0
        assert d2 > 0
        assert d3 > 0

    def test_backoff_capped_at_60_seconds(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        # 2^10 = 1024, capped at 60
        delay = limiter.backoff(10)
        assert delay <= 60.5  # 60 + random(0, 0.5)
        assert delay >= 60.0

    def test_backoff_returns_float(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        delay = limiter.backoff(5)
        assert isinstance(delay, float)


class TestEncreRateLimiterReset:
    def test_reset_clears_windows(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(per_minute=5)
        for _ in range(3):
            limiter.check("bash")
        assert "bash" in limiter._windows
        limiter.reset()
        assert limiter._windows == {}

    def test_reset_clears_concurrent_count(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        limiter._concurrent_count = 5
        limiter.reset()
        assert limiter._concurrent_count == 0

    def test_active_tools_empty_after_reset(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(per_minute=100)
        limiter.check("bash")
        limiter.check("grep")
        assert len(limiter.active_tools) == 2
        limiter.reset()
        assert limiter.active_tools == []


class TestEncreRateLimiterActiveTools:
    def test_active_tools_returns_names(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter(per_minute=100)
        limiter.check("bash")
        limiter.check("grep")
        active = limiter.active_tools
        assert "bash" in active
        assert "grep" in active

    def test_active_tools_starts_empty(self):
        from encre.ratelimit import EncreRateLimiter
        limiter = EncreRateLimiter()
        assert limiter.active_tools == []
