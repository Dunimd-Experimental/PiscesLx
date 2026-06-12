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
"""
Exponential backoff retry mechanism for LLM API calls.

Provides a configurable retry decorator/context that handles transient HTTP
errors (429 rate limits, 502/503/504 server errors), connection timeouts, and
other network-level failures.  The retry logic uses full jitter for the sleep
interval to avoid thundering-herd problems when multiple clients hit a rate
limit simultaneously.

Design decisions
----------------
- **Exponential backoff with full jitter**: sleep = random(0, base * 2^attempt)
  This avoids synchronised retries across multiple instances.
- **Separate retry budgets for different error classes**: 429 (rate limit) gets
  more retries than 5xx (server errors) because rate limits are typically
  self-correcting within seconds.
- **Async-first**: designed for use with ``httpx.AsyncClient`` and ``asyncio``.
- **Non-intrusive**: the decorator preserves the original function's signature
  and type annotations via ``functools.wraps``.

Typical usage::

    from encre.backends.retry import retry_with_backoff, RetryConfig

    config = RetryConfig(max_retries=5, base_delay=2.0)

    @retry_with_backoff(config=config)
    async def call_llm_api(client, payload):
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
"""

import asyncio
import inspect
import random
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class RetryConfig:
    """Configuration for the exponential backoff retry behaviour.

    Attributes:
        max_retries: Maximum number of retry attempts before giving up.
            Default 5 (covers most transient failure scenarios).
        base_delay: Base delay in seconds for the exponential backoff
            calculation.  Actual delay = random(0, base_delay * 2^attempt).
            Default 2.0 — slightly higher than 1.0 to better handle API
            cold-start latency (common with DeepSeek, etc.).
        max_delay: Maximum delay in seconds between retries.  Prevents the
            backoff from growing unbounded.  Default 120.0.
        retryable_status_codes: HTTP status codes that trigger a retry.
            Default includes 429 (rate limit), 502 (bad gateway), 503 (service
            unavailable), and 504 (gateway timeout).
        retryable_exceptions: Exception types that trigger a retry.
            Default includes httpx.TimeoutException, httpx.ConnectError,
            httpx.RemoteProtocolError, and httpx.TransportError.
        rate_limit_retries: Separate (higher) retry budget for 429 responses.
            Default 8, since rate limits are usually short-lived.
    """

    max_retries: int = 8
    base_delay: float = 2.0
    max_delay: float = 120.0
    retryable_status_codes: set[int] = field(
        default_factory=lambda: {429, 502, 503, 504}
    )
    retryable_exceptions: set[type[Exception]] = field(
        default_factory=lambda: {
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.TransportError,
        }
    )
    rate_limit_retries: int = 8


# Default configuration suitable for most LLM API providers.
DEFAULT_RETRY_CONFIG = RetryConfig()


def retry_with_backoff(
    config: RetryConfig = DEFAULT_RETRY_CONFIG,
) -> Callable[[F], F]:
    """Decorator that adds exponential backoff retry logic to an async function.

    The decorator intercepts :class:`httpx.HTTPStatusError` and
    :class:`httpx.RequestError` exceptions raised by the wrapped function.
    When the status code or exception type matches the retryable set, it waits
    for ``random(0, config.base_delay * 2^attempt)`` seconds before retrying.

    429 (rate limit) responses receive a separate, higher retry budget
    (``config.rate_limit_retries``) because rate limits are typically
    self-correcting within a few seconds.

    Args:
        config: A :class:`RetryConfig` instance controlling retry behaviour.
            Uses :data:`DEFAULT_RETRY_CONFIG` if not provided.

    Returns:
        A decorator that wraps the target async function with retry logic.

    Raises:
        httpx.HTTPStatusError: If the response status code is not retryable
            (e.g. 400, 401, 403, 404) or if all retry attempts are exhausted.
        The last exception: If all retry attempts are exhausted for a
            non-HTTP-status error (e.g. connection timeout).

    Example::

        @retry_with_backoff(RetryConfig(max_retries=3, base_delay=0.5))
        async def fetch_data(client, params):
            resp = await client.get("https://api.example.com/data", params=params)
            resp.raise_for_status()
            return resp.json()
    """

    def decorator(func: F) -> F:
        if inspect.isasyncgenfunction(func):
            return _wrap_async_gen(func, config)  # type: ignore[return-value]

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as exc:
                    last_exception = exc
                    status_code = exc.response.status_code

                    # Non-retryable status codes — fail immediately.
                    if status_code not in config.retryable_status_codes:
                        raise

                    # Use a separate (higher) retry budget for 429 rate limits.
                    if status_code == 429:
                        if attempt >= config.rate_limit_retries:
                            raise
                    elif attempt >= config.max_retries:
                        raise

                    delay = _compute_delay(attempt, config)
                    await asyncio.sleep(delay)

                except tuple(config.retryable_exceptions) as exc:
                    last_exception = exc
                    if attempt >= config.max_retries:
                        raise

                    delay = _compute_delay(attempt, config)
                    await asyncio.sleep(delay)

            # All retries exhausted — re-raise the last exception.
            if last_exception is not None:
                raise last_exception

        return wrapper  # type: ignore[return-value]

    return decorator


def _wrap_async_gen(func, config):
    """Wrap an async generator function with retry logic.

    On transient failures the generator is re-invoked from scratch; events from
    a failed attempt are discarded so the caller only sees a clean stream.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        last_exception: Exception | None = None

        for attempt in range(config.max_retries + 1):
            try:
                async for item in func(*args, **kwargs):
                    yield item
                return  # stream completed successfully
            except httpx.HTTPStatusError as exc:
                last_exception = exc
                status_code = exc.response.status_code

                if status_code not in config.retryable_status_codes:
                    raise

                if status_code == 429:
                    if attempt >= config.rate_limit_retries:
                        raise
                elif attempt >= config.max_retries:
                    raise

                delay = _compute_delay(attempt, config)
                await asyncio.sleep(delay)

            except tuple(config.retryable_exceptions) as exc:
                last_exception = exc
                if attempt >= config.max_retries:
                    raise

                delay = _compute_delay(attempt, config)
                await asyncio.sleep(delay)

        if last_exception is not None:
            raise last_exception

    return wrapper


def _compute_delay(attempt: int, config: RetryConfig) -> float:
    """Compute the sleep delay for a given retry attempt using full jitter.

    The formula is::

        delay = random(0, min(config.base_delay * 2^attempt, config.max_delay))

    Full jitter (random between 0 and the exponential cap) is preferred over
    equal jitter (random between cap/2 and cap) because it produces a smoother
    distribution of retry times across multiple clients, reducing the
    thundering-herd effect.

    Args:
        attempt: The current retry attempt number (0-indexed).
        config: The :class:`RetryConfig` instance.

    Returns:
        A float delay in seconds between 0 and ``config.max_delay``.
    """
    cap = min(config.base_delay * (2**attempt), config.max_delay)
    return random.uniform(0, cap)