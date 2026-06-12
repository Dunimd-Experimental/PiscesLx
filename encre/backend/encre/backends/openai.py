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
OpenAI backend — GPT-4.1 family, GPT-5.x, o3, o4-mini (2026 lineup).

As of May 2026, OpenAI's model lineup has evolved significantly:

- **GPT-4.1 family** (Nano / Mini / full): The mid-range workhorse replacing
  GPT-4o (deprecated Jan 2026). All variants support 1,048,576 (1M) token
  context windows and prompt caching at 75% off cached input tokens.
  Pricing: $2/$8 per 1M tokens (full), $0.40/$1.60 (Mini), $0.10/$0.40 (Nano).

- **GPT-5.x series** (5.2 / 5.4 / 5.5): Tiered flagship models with varying
  context windows (128K-1M) and capability levels. GPT-5.5 is the most capable
  with 1M context at $5/$30 per 1M tokens.

- **o3 / o4-mini**: Reasoning-optimised models with 200K context windows and
  extended output limits (100K tokens). Designed for complex multi-step
  reasoning, code generation, and analysis tasks.

- **GPT-4o**: Fully deprecated as of January 2026. Kept in the registry for
  backward compatibility only; all new sessions should use GPT-4.1.

This backend extends :class:`OpenAISSEBackend` and inherits all SSE streaming,
tool call buffering, and non-stream fallback logic.  The only customisation
is the default API endpoint (``https://api.openai.com/v1``) and the default
model (``gpt-4.1``).
"""

from typing import Any

import httpx

from encre.backends.openai_sse import OpenAISSEBackend
from encre.backends.retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    retry_with_backoff,
)
from encre.logging_config import get_logger
from encre.utils.types import BackendEvent, create_backend_thinking

logger = get_logger("encre.backends.openai")


class OpenAIBackend(OpenAISSEBackend):
    """OpenAI backend for the 2026 model lineup.

    Supports GPT-4.1 (Nano/Mini/full), GPT-5.x (5.2/5.4/5.5), o3, and
    o4-mini.  The default model is ``gpt-4.1``, which replaces the deprecated
    ``gpt-4o``.

    All models support tool calling, image inputs, and prompt caching.
    The GPT-4.1 family offers 1M token context windows at competitive prices.
    o3 and o4-mini add extended reasoning capabilities with 200K context and
    100K output token limits.

    2026 pricing summary:
        - GPT-4.1: $2.00/$8.00 per 1M tokens (input/output)
        - GPT-4.1 Mini: $0.40/$1.60 per 1M tokens
        - GPT-4.1 Nano: $0.10/$0.40 per 1M tokens
        - GPT-5.2: $1.75/$14.00 per 1M tokens (128K context)
        - GPT-5.4: $2.50/$15.00 per 1M tokens (400K context)
        - GPT-5.5: $5.00/$30.00 per 1M tokens (1M context)
        - o3: $2.00/$8.00 per 1M tokens (200K context)
        - o4-mini: $1.10/$4.40 per 1M tokens (200K context)
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gpt-4.1",
        **kwargs: Any,
    ) -> None:
        """Initialise the OpenAI backend.

        Args:
            api_key: OpenAI API key.  If empty, falls back to the
                ``OPENAI_API_KEY`` environment variable (handled by the caller).
            base_url: Custom API base URL.  Defaults to
                ``https://api.openai.com/v1``.  Can be changed to use Azure
                OpenAI or other OpenAI-compatible endpoints.
            model: Model name.  Defaults to ``gpt-4.1`` (the 2026 replacement
                for GPT-4o).  Other valid values: ``gpt-4.1-mini``,
                ``gpt-4.1-nano``, ``gpt-5.2``, ``gpt-5.4``, ``gpt-5.5``,
                ``o3``, ``o4-mini``.
            **kwargs: Additional arguments passed to :class:`OpenAISSEBackend`.
        """
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.retry_config: RetryConfig = kwargs.pop(
            "retry_config", DEFAULT_RETRY_CONFIG
        )
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    # ── Overrides ─────────────────────────────────────────────────────

    async def _send_request(self, body: dict[str, Any]) -> httpx.Response:
        """Send the request with exponential-backoff retry.

        Wraps the HTTP POST to OpenAI's chat completions endpoint with
        :func:`retry_with_backoff` to handle transient errors (429 rate limits,
        502/503/504 server errors, connection timeouts).

        Args:
            body: The JSON request body containing model, messages, tools, etc.

        Returns:
            An :class:`httpx.Response` with a streaming body.

        Raises:
            httpx.HTTPStatusError: If the response status code is not retryable
                or if all retry attempts are exhausted.
        """
        logger.debug(
            f"Sending request to OpenAI: model={self.model}",
            extra={"model": self.model},
        )
        try:
            _retry_decorator = retry_with_backoff(config=self.retry_config)
            _retried_request = _retry_decorator(lambda: self._client.send(
                self._client.build_request(
                    "POST", self._get_endpoint(), json=body
                ),
                stream=True,
            ))
            return await _retried_request()
        except Exception:
            logger.error(
                f"OpenAI backend request failed",
                extra={"model": self.model},
                exc_info=True,
            )
            raise

    # ── Reasoning content extraction ────────────────────────────────
    # OpenAI o3/o4 models emit reasoning_content in the response delta.
    # Extract and emit as BackendThinking events so the UI can display them.

    def _extract_extra_stream_events(
        self, delta: dict[str, Any]
    ) -> list[BackendEvent]:
        reasoning = delta.get("reasoning_content")
        if reasoning:
            return [create_backend_thinking(reasoning)]
        return []

    def _extract_extra_non_stream_events(
        self, message: dict[str, Any]
    ) -> list[BackendEvent]:
        reasoning = message.get("reasoning_content")
        if reasoning:
            return [create_backend_thinking(reasoning)]
        return []

    def context_window_size(self) -> int:
        """Return the context window size for the current model.

        Returns 1,048,576 (1M) for GPT-4.1 family models, 200,000 for o3/o4-mini,
        and 128,000 as the default fallback.  The actual value depends on the
        model selected at initialisation.

        2026 reference:
            - GPT-4.1 / Mini / Nano: 1,048,576 tokens
            - GPT-5.2: 128,000 tokens
            - GPT-5.4: 400,000 tokens
            - GPT-5.5: 1,048,576 tokens
            - o3 / o4-mini: 200,000 tokens
        """
        model_lower = self.model.lower()
        if "nano" in model_lower or "mini" in model_lower or "4.1" in model_lower:
            return 1048576
        if model_lower.startswith("o3") or model_lower.startswith("o4"):
            return 200000
        if "5.5" in model_lower:
            return 1048576
        if "5.4" in model_lower:
            return 400000
        return 128000

    def supports_thinking(self) -> bool:
        """Return True for reasoning models (o3, o4-mini).

        These models emit extended thinking tokens as part of their response,
        which are surfaced as :class:`BackendThinking` events in the agent loop.
        """
        model_lower = self.model.lower()
        return model_lower.startswith("o3") or model_lower.startswith("o4")