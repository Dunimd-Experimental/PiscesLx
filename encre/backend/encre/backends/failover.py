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
import contextlib
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from encre.backends.base import BaseBackend
from encre.utils.types import BackendError, BackendEvent, BackendFinish


@dataclass
class BackendHealth:
    name: str
    healthy: bool = True
    consecutive_failures: int = 0
    last_checked: float = 0.0
    last_error: str = ""
    total_requests: int = 0
    total_failures: int = 0

    def record_success(self) -> None:
        self.healthy = True
        self.consecutive_failures = 0
        self.total_requests += 1

    def record_failure(self, error: str) -> None:
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_requests += 1
        self.last_error = error
        if self.consecutive_failures >= 3:
            self.healthy = False


class FailoverBackend(BaseBackend):
    """Backend that chains multiple backends with automatic failover.

    When the primary backend fails (timeout, rate limit, API error), the next
    backend in the chain is tried.  Health status is tracked and unhealthy
    backends are skipped.

    Events are buffered from each backend attempt so that partial output
    from a failed backend is never yielded to the caller.  Only a clean,
    complete stream from a successful backend reaches the consumer.

    Usage:
        failover = FailoverBackend([
            ("primary", OpenAIBackend(model="gpt-5", api_key="...")),
            ("fallback", AnthropicBackend(model="claude-sonnet-4-6", api_key="...")),
            ("last_resort", DeepSeekBackend(model="deepseek-v4-pro", api_key="...")),
        ])
    """

    MAX_CONSECUTIVE_FAILURES = 3
    RECOVERY_GRACE_PERIOD = 300.0

    def __init__(self, backends: list[tuple[str, BaseBackend]]) -> None:
        if not backends:
            raise ValueError("At least one backend is required")
        self._backends: list[tuple[str, BaseBackend]] = backends
        self._health: dict[str, BackendHealth] = {
            name: BackendHealth(name=name) for name, _ in backends
        }
        self._primary = backends[0][1]
        self._active_name: str = backends[0][0]
        self._lock = asyncio.Lock()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
        enable_caching: bool = False,
    ) -> AsyncGenerator[BackendEvent, None]:
        """Send a chat completion request with automatic failover.

        Events from each backend attempt are buffered.  If the current
        backend fails (exception or BackendError), the buffer is discarded
        and the next healthy backend is tried.  Only a complete, successful
        stream reaches the caller.
        """
        errors: list[str] = []

        for name, backend in self._backends:
            health = self._health[name]
            if not health.healthy:
                if time.time() - health.last_checked < self.RECOVERY_GRACE_PERIOD:
                    continue
                health.healthy = True
                health.consecutive_failures = 0

            health.last_checked = time.time()

            buffer: list[BackendEvent] = []
            failed = False
            error_msg = ""

            try:
                async with self._lock:
                    self._active_name = name

                async for event in backend.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    enable_caching=enable_caching,
                ):
                    if isinstance(event, BackendError):
                        error_msg = event.error
                        failed = True
                        break
                    buffer.append(event)
                    if isinstance(event, BackendFinish):
                        break

            except Exception as e:
                error_msg = str(e)
                failed = True

            if failed:
                health.record_failure(error_msg)
                errors.append(f"[{name}] {error_msg}")
                continue

            health.record_success()
            for event in buffer:
                yield event
            return

        yield BackendError(
            error=f"All backends failed: {'; '.join(errors)}"
        )

    def supports_tool_calling(self) -> bool:
        return self._primary.supports_tool_calling()

    def context_window_size(self) -> int:
        return self._primary.context_window_size()

    def supports_thinking(self) -> bool:
        return self._primary.supports_thinking()

    def supports_prompt_caching(self) -> bool:
        return self._primary.supports_prompt_caching()

    def count_tokens(self, text: str) -> int:
        return self._primary.count_tokens(text)

    def get_health(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "healthy": h.healthy,
                "consecutive_failures": h.consecutive_failures,
                "total_requests": h.total_requests,
                "total_failures": h.total_failures,
                "last_error": h.last_error,
            }
            for name, h in self._health.items()
        }

    @property
    def active_backend_name(self) -> str:
        return self._active_name

    async def aclose(self) -> None:
        for _, backend in self._backends:
            with contextlib.suppress(Exception):
                await backend.aclose()
