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

"""
DeepSeek backend — V4-Flash, V4-Pro (2026 lineup).

As of May 2026, DeepSeek's model lineup has been updated:

- **DeepSeek V4-Flash**: The default chat model with 1M context window,
  384K output tokens, and aggressive cache discounts (92% off cache hits).
  Pricing: $0.14/$0.28 per 1M tokens (input/output).  Replaces the
  deprecated ``deepseek-chat`` model.

- **DeepSeek V4-Pro**: The enhanced reasoning model with 1M context window,
  384K output tokens, and 80% cache hit discount.  Pricing: $1.74/$3.48 per
  1M tokens.  Replaces the deprecated ``deepseek-reasoner`` model.

- **Legacy models**: ``deepseek-chat`` and ``deepseek-reasoner`` are
  deprecated as of July 2026.  They are kept in the registry for backward
  compatibility but map to their V4 equivalents.

Both V4 models support:
- Tool/function calling (OpenAI-compatible format)
- Thinking/reasoning tokens (emitted as ``reasoning_content`` in the API)
- Prompt caching (80-92% discount on cache hits)
- 1M token context windows
- 384K token output limits

This backend extends :class:`OpenAISSEBackend` because DeepSeek uses an
OpenAI-compatible API.  The only customisation is the extraction of
``reasoning_content`` from the response delta, which is emitted as
:class:`BackendThinking` events.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend
from encre.utils.types import BackendEvent, create_backend_thinking


class DeepSeekBackend(OpenAISSEBackend):
    """DeepSeek backend for the 2026 V4 model lineup.

    Supports DeepSeek V4-Flash (default) and V4-Pro via the OpenAI-compatible
    API at ``https://api.deepseek.com/v1``.  The legacy ``deepseek-chat`` and
    ``deepseek-reasoner`` model names are also accepted and map to their V4
    equivalents in the registry.

    The key difference from the base :class:`OpenAISSEBackend` is the
    extraction of ``reasoning_content`` from the response delta, which
    contains the model's chain-of-thought reasoning.  This is emitted as
    :class:`BackendThinking` events so the agent loop can surface them
    appropriately.

    2026 pricing summary:
        - DeepSeek V4-Flash: $0.14/$0.28 per 1M tokens (92% cache discount)
        - DeepSeek V4-Pro: $1.74/$3.48 per 1M tokens (80% cache discount)
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "deepseek-v4-flash",
        **kwargs: Any,
    ) -> None:
        """Initialise the DeepSeek backend.

        Args:
            api_key: DeepSeek API key.
            base_url: Custom API base URL.  Defaults to
                ``https://api.deepseek.com/v1``.
            model: Model name.  Defaults to ``deepseek-chat`` (maps to
                V4-Flash in the registry).  Other valid values:
                ``deepseek-v4-flash``, ``deepseek-v4-pro``,
                ``deepseek-reasoner``.
            **kwargs: Additional arguments passed to :class:`OpenAISSEBackend`.
        """
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    # ── Overrides ─────────────────────────────────────────────────────

    def _build_request_data(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> dict[str, Any]:
        data = super()._build_request_data(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if self.supports_thinking():
            data["thinking"] = {"type": "enabled"}
        return data

    def _extract_extra_stream_events(
        self, delta: dict[str, Any]
    ) -> list[BackendEvent]:
        """Emit BackendThinking for reasoning_content deltas.

        DeepSeek V4 models emit ``reasoning_content`` in the response delta
        alongside the regular ``content`` field.  This method extracts those
        reasoning tokens and converts them to :class:`BackendThinking` events.

        Args:
            delta: The ``delta`` object from a streaming SSE chunk.

        Returns:
            A list of :class:`BackendThinking` events, or an empty list if
            no reasoning content is present.
        """
        reasoning = delta.get("reasoning_content")
        if reasoning:
            return [create_backend_thinking(reasoning)]
        return []

    def _extract_extra_non_stream_events(
        self, message: dict[str, Any]
    ) -> list[BackendEvent]:
        """Emit BackendThinking for reasoning_content in non-stream responses.

        Args:
            message: The ``message`` object from a non-streaming response.

        Returns:
            A list of :class:`BackendThinking` events, or an empty list if
            no reasoning content is present.
        """
        reasoning = message.get("reasoning_content")
        if reasoning:
            return [create_backend_thinking(reasoning)]
        return []

    def context_window_size(self) -> int:
        """Return the context window size for DeepSeek V4 models.

        Both V4-Flash and V4-Pro support 1,048,576 (1M) token context windows.
        Legacy models (deepseek-chat/reasoner) had 64K context.
        """
        return 1048576

    def supports_thinking(self) -> bool:
        """DeepSeek V4 models support reasoning/thinking tokens."""
        return True

    def supports_prompt_caching(self) -> bool:
        """DeepSeek V4 models support prompt caching (80-92% discount)."""
        return True
