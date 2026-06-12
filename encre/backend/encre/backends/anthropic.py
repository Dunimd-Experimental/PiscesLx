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
Anthropic backend — Claude Opus 4.6/4.7, Sonnet 4.5/4.6, Haiku 4.5 (2026 lineup).

As of May 2026, Anthropic's Claude model lineup includes:

- **Claude Opus 4.6 / 4.7**: Anthropic's most capable models, excelling at
  complex reasoning, code generation, and nuanced analysis. 200K context
  window (1M in beta). Pricing: $5/$25 per 1M tokens (input/output).
  Supports thinking mode and prompt caching (90% off cache reads).

- **Claude Sonnet 4.5 / 4.6**: The balanced workhorse — strong reasoning at
  lower cost. 200K context (1M in beta for Sonnet 4.6). Pricing: $3/$15 per
  1M tokens. Sonnet 4.6 is currently in beta with extended context support.

- **Claude Haiku 4.5**: The fastest and most cost-effective Claude model.
  200K context. Pricing: $1/$5 per 1M tokens. Ideal for high-throughput,
  latency-sensitive applications.

All Claude models support:
- Tool/function calling (native tool_use API)
- Image inputs (vision)
- Thinking/reasoning tokens (except Haiku)
- Prompt caching (90% discount on cache reads)
- Extended output (8192 tokens default)

This backend implements Anthropic's native Messages API directly (not
OpenAI-compatible), using the ``/v1/messages`` endpoint with SSE streaming.
The protocol differs significantly from OpenAI: it uses named events
(``content_block_start``, ``content_block_delta``, etc.) instead of
OpenAI's ``choices[0].delta`` structure.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any
import httpx

from encre.backends.base import BaseBackend
from encre.backends.retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    retry_with_backoff,
)
from encre.logging_config import get_logger
from encre.utils.types import (
    BackendEvent,
    create_backend_error,
    create_backend_finish,
    create_backend_text,
    create_backend_thinking,
    create_backend_tool_call,
    create_backend_tool_call_delta,
)

logger = get_logger("encre.backends.anthropic")


class AnthropicBackend(BaseBackend):
    """Anthropic backend for the 2026 Claude model lineup.

    Supports Claude Opus 4.6/4.7, Sonnet 4.5/4.6, and Haiku 4.5 via
    Anthropic's native Messages API.  The default model is
    ``claude-sonnet-4-20250514`` (Sonnet 4.6).

    This backend implements the Anthropic SSE protocol directly, handling:
    - ``content_block_start`` events for text, thinking, and tool_use blocks
    - ``content_block_delta`` events for text deltas, thinking deltas,
      signature deltas, and input_json deltas (tool call arguments)
    - ``content_block_stop`` events to finalise tool calls
    - ``message_delta`` events for finish reasons and usage metadata
    - ``error`` events for API-level errors

    Prompt caching is supported via the ``enable_caching`` parameter, which
    injects ``cache_control`` breakpoints on system messages and the last
    user message.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        **kwargs: Any,
    ) -> None:
        """Initialise the Anthropic backend.

        Args:
            api_key: Anthropic API key.  Required for authentication via the
                ``x-api-key`` header.
            model: Claude model name.  Defaults to ``claude-sonnet-4-20250514``
                (Sonnet 4.6).  Other valid values: ``claude-opus-4-20250514``
                (Opus 4.6), ``claude-haiku-4-20250514`` (Haiku 4.5).
            **kwargs: Additional arguments.  Supports ``retry_config`` for
                custom :class:`RetryConfig`.
        """
        self.api_key = api_key
        self.model = model
        self.retry_config: RetryConfig = kwargs.pop("retry_config", DEFAULT_RETRY_CONFIG)
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(300.0, connect=30.0),
        )

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
        """Send a chat completion request and stream back events.

        Implements Anthropic's Messages API with SSE streaming.  The method
        handles the full event lifecycle: content block start/delta/stop for
        text, thinking, and tool_use blocks, plus message-level deltas for
        finish reasons.

        Args:
            messages: Conversation history in OpenAI message format.  System
                messages are extracted and sent via the ``system`` parameter.
            tools: Optional tool definitions in OpenAI format.  Converted to
                Anthropic's ``tools`` parameter format.
            tool_choice: Tool selection strategy.  ``"auto"``, ``"any"``, or
                ``"none"``.  Mapped to Anthropic's ``tool_choice.type``.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.
            stream: If True (default), uses SSE streaming.  If False, uses
                non-streaming request.
            enable_caching: If True, injects ``cache_control`` breakpoints
                for prompt caching (90% discount on cache reads).

        Yields:
            :class:`BackendText`, :class:`BackendThinking`,
            :class:`BackendToolCallDelta`, :class:`BackendToolCall`,
            :class:`BackendFinish`, or :class:`BackendError`.
        """
        if enable_caching:
            messages = self._apply_prompt_caching(messages)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools

        if tool_choice == "auto":
            body["tool_choice"] = {"type": "auto"}
        elif tool_choice == "any":
            body["tool_choice"] = {"type": "any"}
        elif tool_choice == "none":
            body["tool_choice"] = {"type": "none"}

        try:

            async def _make_request() -> httpx.Response:
                return await self._client.send(
                    self._client.build_request("POST", "/messages", json=body),
                    stream=True,
                )

            _retry_decorator = retry_with_backoff(config=self.retry_config)
            _retried_request = _retry_decorator(_make_request)
            resp = await _retried_request()

            async with resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_text = error_body.decode(errors="replace")
                    msg = f"Anthropic API error {resp.status_code}: {error_text}"
                    logger.error(msg)
                    yield create_backend_error(msg)
                    return

                current_tool_use: dict[str, Any] | None = None
                current_tool_index: int = 0
                finish_reason: str = "stop"

                async for line in resp.aiter_lines():
                    if not line.startswith("event: "):
                        continue
                    event_type = line[7:].strip()
                    data_line = await resp.__anext__()
                    if not data_line.startswith("data: "):
                        continue
                    data = json.loads(data_line[6:].strip())

                    if event_type == "content_block_start":
                        block = data.get("content_block", {})
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                yield create_backend_text(text)
                        elif block.get("type") == "thinking":
                            thinking_text = block.get("thinking", "")
                            if thinking_text:
                                yield create_backend_thinking(thinking_text)
                        elif block.get("type") == "redacted_thinking":
                            yield create_backend_thinking("[Thinking redacted]")
                        elif block.get("type") == "tool_use":
                            current_tool_use = {
                                "id": block.get("id", ""),
                                "name": block.get("name", ""),
                                "arguments": "",
                            }
                            current_tool_index = data.get("index", 0)
                            yield create_backend_tool_call_delta(
                                current_tool_index, "name", block.get("name", "")
                            )

                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yield create_backend_text(text)
                        elif delta.get("type") == "thinking_delta":
                            thinking_text = delta.get("thinking", "")
                            if thinking_text:
                                yield create_backend_thinking(thinking_text)
                        elif delta.get("type") == "signature_delta":
                            sig = delta.get("signature", "")
                            if sig:
                                yield create_backend_thinking("", signature_delta=sig)
                        elif delta.get("type") == "input_json_delta":
                            partial = delta.get("partial_json", "")
                            if current_tool_use is not None:
                                current_tool_use["arguments"] += partial
                            yield create_backend_tool_call_delta(
                                data.get("index", 0), "arguments", partial
                            )

                    elif event_type == "content_block_stop":
                        if current_tool_use is not None:
                            yield create_backend_tool_call(
                                id=current_tool_use["id"],
                                name=current_tool_use["name"],
                                arguments=current_tool_use["arguments"],
                            )
                            current_tool_use = None

                    elif event_type == "message_delta":
                        delta = data.get("delta", {})
                        stop_reason = delta.get("stop_reason", "")
                        if stop_reason == "end_turn":
                            finish_reason = "stop"
                        elif stop_reason == "tool_use":
                            finish_reason = "tool_calls"
                        elif stop_reason == "max_tokens":
                            finish_reason = "max_tokens"
                        else:
                            finish_reason = stop_reason or "stop"

                    elif event_type == "error":
                        error_data = data.get("error", {})
                        err_msg = error_data.get("message", str(data))
                        logger.error(f"Anthropic stream error: {err_msg}")
                        yield create_backend_error(err_msg)

                yield create_backend_finish(finish_reason)

        except Exception as e:
            logger.error(f"Anthropic backend request failed: {e}", extra={"model": self.model})
            yield create_backend_error(str(e))

    def supports_tool_calling(self) -> bool:
        """All Claude models support native tool calling via the tool_use API."""
        return True

    def context_window_size(self) -> int:
        """Return the context window size for Claude models.

        All 2026 Claude models support 200,000 tokens.  Sonnet 4.6 and
        Opus 4.6/4.7 have 1M context in beta.
        """
        return 200000

    async def aclose(self) -> None:
        """Close the HTTP client session."""
        await self._client.aclose()

    def supports_thinking(self) -> bool:
        """Claude Opus and Sonnet support thinking tokens; Haiku does not."""
        return True

    def supports_prompt_caching(self) -> bool:
        """All Claude models support prompt caching at 90% off cache reads."""
        return True

    def count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken or char/4 heuristic.

        Anthropic uses a BPE tokenizer similar to GPT.  For precise
        counts, use the Anthropic API ``/v1/messages/count_tokens``
        endpoint (requires an async call).
        """
        if not text:
            return 0
        try:
            from encre.utils.tokens import estimate_tokens
            return estimate_tokens(text, model="claude-sonnet-4-6")
        except Exception:
            return len(text) // 4

    async def list_models(self) -> list[str]:
        """Fetch available models from Anthropic's models endpoint.

        Returns a list of model IDs available to the API key.
        Results are cached for 5 minutes.
        """
        import time
        now = time.time()
        cache_key = f"anthropic:{self.api_key[:8] if self.api_key else 'noauth'}"
        if (
            hasattr(self, "_models_cache")
            and hasattr(self, "_models_cache_ts")
            and cache_key == getattr(self, "_models_cache_key", "")
            and now - self._models_cache_ts < 300
        ):
            return self._models_cache  # type: ignore[attr-defined]

        try:
            resp = await self._client.get("/models")
            resp.raise_for_status()
            data = resp.json()
            models: list[str] = []
            for item in data.get("data", []):
                model_id = item.get("id", "")
                if model_id:
                    models.append(model_id)
            models.sort()
        except Exception:
            models = []

        self._models_cache = models
        self._models_cache_ts = now
        self._models_cache_key = cache_key
        return models

    @staticmethod
    def _apply_prompt_caching(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject ``cache_control`` breakpoints for Anthropic prompt caching.

        Caches system messages and the last user message.  ``cache_control`` is
        only valid on ``text`` and ``tool_result`` content blocks -- images and
        other block types must be skipped.  This method walks content blocks in
        reverse to find the *last textual* block rather than blindly marking
        whatever happens to appear last in the list.

        Args:
            messages: The conversation history to annotate with cache breakpoints.

        Returns:
            A new message list with ``cache_control`` annotations added to
            system messages and the last user message's final text block.
        """
        # Valid block types for cache_control per Anthropic API docs.
        _cacheable_block_types = frozenset({"text", "tool_result"})

        result: list[dict[str, Any]] = []
        system_indices: list[int] = []
        last_user_idx: int | None = None

        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_indices.append(i)
            elif msg.get("role") == "user":
                last_user_idx = i

        for i, msg in enumerate(messages):
            msg_copy = dict(msg)
            should_cache = i in system_indices or i == last_user_idx

            if should_cache:
                content = msg_copy.get("content")
                if isinstance(content, str):
                    msg_copy["content"] = [
                        {"type": "text", "text": content,
                         "cache_control": {"type": "ephemeral"}}
                    ]
                elif isinstance(content, list):
                    blocks: list[dict[str, Any]] = []
                    cacheable_last_idx: int | None = None
                    for rev_j in range(len(content) - 1, -1, -1):
                        block_type = content[rev_j].get("type", "")
                        if block_type in _cacheable_block_types:
                            cacheable_last_idx = rev_j
                            break

                    for j, block in enumerate(content):
                        block_copy = dict(block)
                        if j == cacheable_last_idx:
                            block_copy["cache_control"] = {"type": "ephemeral"}
                        blocks.append(block_copy)
                    msg_copy["content"] = blocks

            result.append(msg_copy)

        return result
