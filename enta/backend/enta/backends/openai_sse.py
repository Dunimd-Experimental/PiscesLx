#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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
Shared SSE (Server-Sent Events) streaming backend for OpenAI-compatible APIs.

This module provides :class:`OpenAISSEBackend`, a reusable base class that
implements the common SSE streaming protocol shared by OpenAI, DeepSeek, Groq,
and any OpenAI-compatible provider.  Subclasses only need to configure the
API endpoint, authentication, and model name -- the SSE parsing, tool call
buffering, and non-stream fallback are all handled here.

Protocol details
----------------
The OpenAI chat completions SSE protocol emits ``data:`` lines with these
event types:

- ``data: {"choices":[{"delta":{"content":"Hello"}}]}`` -- text delta
- ``data: {"choices":[{"delta":{"tool_calls":[{"function":{"name":"search","arguments":""}}]}}]}`` -- tool call start  # noqa: E501
- ``data: {"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"q\":"}}]}}]}`` -- tool call argument delta  # noqa: E501
- ``data: {"choices":[{"delta":{}}]}`` -- empty delta (signals end of a choice)
- ``data: [DONE]`` -- stream complete
- ``data: {"error":{...}}`` -- error event

Tool call buffering
-------------------
Tool call arguments arrive as a sequence of string deltas that must be
concatenated and parsed as JSON.  The backend buffers these deltas in
:attr:`_tool_call_buffers`, keyed by tool call index, and emits a complete
:class:`BackendToolCall` only when the stream finishes or the choice ends.

Non-stream fallback
-------------------
When ``stream=False``, the backend sends a non-streaming request and parses
the single JSON response body into the same :class:`BackendEvent` types,
ensuring the agent loop always receives a uniform event stream regardless
of the streaming setting.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from enta.backends.base import BaseBackend
from enta.backends.retry import DEFAULT_RETRY_CONFIG, RetryConfig, retry_with_backoff
from enta.utils.types import (
    BackendError,
    BackendEvent,
    BackendFinish,
    BackendText,
    BackendToolCall,
    BackendToolCallDelta,
)


def _normalize_usage(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize provider-specific usage fields to standard keys.

    OpenAI-compatible APIs return ``prompt_tokens`` / ``completion_tokens``,
    while Anthropic uses ``input_tokens`` / ``output_tokens``.  This function
    maps both conventions to the standard ``input_tokens`` / ``output_tokens`` /
    ``total_tokens`` triplet.
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for src, dst in [
        ("prompt_tokens", "input_tokens"),
        ("completion_tokens", "output_tokens"),
        ("total_tokens", "total_tokens"),
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
    ]:
        if src in raw:
            out[dst] = raw[src]
    return out


class OpenAISSEBackend(BaseBackend):
    """Base backend for OpenAI-protocol SSE streaming.

    Handles the common SSE parsing, tool call buffering, and non-stream
    fallback logic shared by OpenAI, DeepSeek, Groq, and OpenAI-compatible
    providers.  Subclasses set :attr:`api_base_url`, :attr:`api_key`, and
    :attr:`model` in their ``__init__``, then call :meth:`chat` to stream
    responses.

    The backend uses :class:`httpx.AsyncClient` for HTTP communication and
    :func:`retry_with_backoff` for automatic retry of transient errors.

    Subclass responsibilities:
        - Set ``self.api_base_url``, ``self.api_key``, ``self.model``.
        - Override :meth:`context_window_size` if the model has a non-default
          context window.
        - Optionally override :meth:`_prepare_request_kwargs` to add
          provider-specific request parameters.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        retry_config: RetryConfig = DEFAULT_RETRY_CONFIG,
        http_timeout: float = 120.0,
        **kwargs: Any,
    ) -> None:
        """Initialise the SSE backend.

        Args:
            api_key: API key for authentication.  May be empty for local
                providers (Ollama, Local).
            base_url: Base URL of the API endpoint.  If empty, the subclass
                should set ``self.api_base_url`` in its own ``__init__``.
            model: Model name to use for completions.
            retry_config: :class:`RetryConfig` for transient error retries.
            http_timeout: HTTP request timeout in seconds.  Default 120s
                (covers long-thinking models like o3 and Claude Opus).
            **kwargs: Additional provider-specific parameters.
        """
        self.api_key = api_key
        self.api_base_url = base_url.rstrip("/").removesuffix("/chat/completions")
        self.model = model
        self.retry_config = retry_config
        self.http_timeout = http_timeout
        self._client: httpx.AsyncClient | None = None
        self._tool_call_buffers: dict[int, dict[str, Any]] = {}

    def _get_client(self) -> httpx.AsyncClient:
        """Return (or create) the shared HTTP client.

        Uses a lazy initialisation pattern so that the client is only created
        when the first request is made.  The client is configured with:
        - A default timeout of ``self.http_timeout`` seconds.
        - No environment proxy override (``trust_env=False``) for security.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=30.0,   # 30s for cold-start APIs (DeepSeek etc.)
                    read=self.http_timeout,
                    write=30.0,
                    pool=30.0,
                ),
                trust_env=False,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the HTTP client and release all connections."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Model listing ──────────────────────────────────────────────────

    async def list_models(self) -> list[str]:
        """Fetch available models from the provider's GET /models endpoint.

        Results are cached with a 5-minute TTL to avoid excessive API calls.
        """
        import time
        now = time.time()
        cache_key = f"{self.api_base_url}:{self.api_key[:8] if self.api_key else 'noauth'}"
        if (
            hasattr(self, "_models_cache")
            and hasattr(self, "_models_cache_ts")
            and cache_key == getattr(self, "_models_cache_key", "")
            and now - self._models_cache_ts < 300
        ):
            return self._models_cache  # type: ignore[attr-defined]

        client = self._get_client()
        headers = self._build_headers()
        try:
            response = await client.get(
                f"{self.api_base_url}/models",
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            models: list[str] = []
            for item in data.get("data", []):
                mid = item.get("id", "")
                if mid:
                    models.append(mid)
            # Sort user-facing models first, then others
            models.sort()
        except Exception:
            models = []

        self._models_cache = models
        self._models_cache_ts = now
        self._models_cache_key = cache_key
        return models

    def supports_tool_calling(self) -> bool:
        """All OpenAI-protocol backends support tool calling by default."""
        return True

    def context_window_size(self) -> int:
        """Return the default context window size (128K tokens).

        Subclasses should override this to return the actual model's context
        window (e.g. 1,048,576 for GPT-4.1 or DeepSeek V4).
        """
        return 128000

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers for the API request.

        Includes the ``Authorization`` header when ``self.api_key`` is set,
        and always includes ``Content-Type: application/json``.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_request_data(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Build the JSON request body for the chat completions endpoint.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Optional tool definitions.
            tool_choice: Tool selection strategy.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: Whether to request a streaming response.

        Returns:
            A dictionary ready to be serialised as JSON for the API request body.
        """
        data: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max(max_tokens, 1),
            "stream": stream,
        }
        if tools:
            data["tools"] = tools
            data["tool_choice"] = tool_choice
        # Always request reasoning/thinking tokens.  Models that don't
        # support it will ignore this field -- it's harmless.
        data["thinking"] = {"type": "enabled"}
        return data

    def _prepare_request_kwargs(self) -> dict[str, Any]:
        """Return additional keyword arguments for the HTTP POST request.

        Subclasses can override this to add provider-specific parameters
        (e.g. ``extra_body`` for DeepSeek's ``thinking`` parameter, or
        custom headers for Anthropic's ``anthropic-version`` header).

        Returns:
            An empty dict by default.  Subclasses should return a dict of
            kwargs to merge into the ``httpx.AsyncClient.post()`` call.
        """
        return {}

    def _extract_extra_stream_events(
        self, delta: dict[str, Any]
    ) -> list[BackendEvent]:
        """Extract ``reasoning_content`` from SSE deltas.

        All OpenAI-protocol backends check for reasoning tokens.  Models
        that don't emit them simply return empty deltas -- harmless.
        """
        from enta.utils.types import create_backend_thinking
        reasoning = delta.get("reasoning_content")
        if reasoning:
            return [create_backend_thinking(reasoning)]
        return []

    def _extract_extra_non_stream_events(
        self, message: dict[str, Any]
    ) -> list[BackendEvent]:
        """Extract ``reasoning_content`` from non-stream responses."""
        from enta.utils.types import create_backend_thinking
        reasoning = message.get("reasoning_content")
        if reasoning:
            return [create_backend_thinking(reasoning)]
        return []

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

        When ``stream=True``, the response is consumed as an SSE stream and
        events are yielded incrementally.  When ``stream=False``, the entire
        response is fetched first, then parsed into the same event types.

        The method is decorated with :func:`retry_with_backoff` for automatic
        retry of transient HTTP errors.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Optional tool definitions.
            tool_choice: Tool selection strategy.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: Whether to use SSE streaming.
            enable_caching: Ignored by the base class; subclasses (e.g.
                Anthropic) may use this to enable prompt caching headers.

        Yields:
            :class:`BackendText`, :class:`BackendToolCallDelta`,
            :class:`BackendToolCall`, :class:`BackendFinish`.
        """
        client = self._get_client()
        headers = self._build_headers()
        data = self._build_request_data(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        extra_kwargs = self._prepare_request_kwargs()

        # DEBUG: log request body to find 400 cause
        if stream:
            async for event in self._stream_response(client, headers, data, extra_kwargs):
                yield event
        else:
            async for event in self._non_stream_response(client, headers, data, extra_kwargs):
                yield event

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        data: dict[str, Any],
        extra_kwargs: dict[str, Any],
    ) -> AsyncGenerator[BackendEvent, None]:
        """Consume an SSE stream and yield parsed events.

        Uses ``httpx.AsyncClient`` with ``auth=Bearer`` and streams the
        response line by line.  Each ``data:`` line is parsed as JSON and
        converted to the appropriate :class:`BackendEvent`.

        The method handles:
        - Text deltas (``delta.content``)
        - Tool call deltas (``delta.tool_calls``) with argument buffering
        - Empty deltas (signals end of a choice)
        - ``[DONE]`` sentinel (signals end of stream)
        - Error events (``data: {"error": ...}``)
        """
        @retry_with_backoff(config=self.retry_config)
        async def _do_stream() -> AsyncGenerator[BackendEvent, None]:
            _t_req = time.time()
            _log = logging.getLogger("enta.backend")
            _msg_count = len(data.get("messages", []))
            _sys_len = len(str([m for m in data.get("messages", []) if m.get("role") == "system"]))
            _log.info("[http] POST %s/chat/completions model=%s msgs=%d sys_chars=%d thinking=%s timeout=%.0fs",  # noqa: E501
                       self.api_base_url, data.get("model", "?"), _msg_count, _sys_len,
                       data.get("thinking", {}).get("type", "disabled"), self.http_timeout)
            _log.info("[http] calling client.stream()...")
            try:
                response = await client.send(
                    client.build_request("POST", f"{self.api_base_url}/chat/completions",
                                         headers=headers, json=data),
                    stream=True,
                )
            except Exception as exc:
                _log.error(f"[http] client.send() raised {type(exc).__name__}: {exc}")
                raise
            _log.info("[http] client.send() done status=%d elapsed=%.2fs",
                       response.status_code, time.time() - _t_req)
            try:
                if response.status_code >= 400:
                    body = await response.aread()
                    _log.error(
                        "HTTP %s response body: %s",
                        response.status_code,
                        body.decode("utf-8", errors="replace")[:2000],
                    )
                response.raise_for_status()
                self._tool_call_buffers.clear()

                _stream_usage: dict[str, Any] | None = None
                _t_first = time.time()
                _first_event = True

                async for line in response.aiter_lines():
                    if _first_event:
                        _log.info("[http] first SSE line after %.2fs", time.time() - _t_first)
                        _first_event = False
                    if not line.startswith("data: "):
                        continue

                    payload = line[6:].strip()

                    # End-of-stream sentinel.
                    if payload == "[DONE]":
                        # Flush any buffered tool calls.
                        for tc in self._tool_call_buffers.values():
                            yield BackendToolCall(
                                id=tc["id"],
                                name=tc["name"],
                                arguments=tc["arguments"],
                            )
                        self._tool_call_buffers.clear()
                        yield BackendFinish(reason="stop", usage=_stream_usage)
                        return

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    # Error event.
                    if "error" in chunk:
                        error_msg = chunk["error"].get("message", str(chunk["error"]))
                        yield BackendError(error=error_msg)
                        return

                    choices = chunk.get("choices", [])
                    if not choices:
                        # Usage may appear in a choices-less final chunk
                        if "usage" in chunk:
                            _stream_usage = _normalize_usage(chunk["usage"])
                        continue

                    delta = choices[0].get("delta", {})
                    finish_reason = choices[0].get("finish_reason")

                    # Capture usage from final chunk
                    if "usage" in chunk:
                        _stream_usage = _normalize_usage(chunk["usage"])

                    # Extra events (reasoning_content, etc.)
                    for extra in self._extract_extra_stream_events(delta):
                        yield extra

                    # Text delta.
                    content = delta.get("content")
                    if content:
                        yield BackendText(text=content)

                    # Tool call deltas.
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        for tc in tool_calls:
                            idx = tc.get("index", 0)
                            if idx not in self._tool_call_buffers:
                                self._tool_call_buffers[idx] = {
                                    "id": tc.get("id") or f"call_{idx}",
                                    "name": "",
                                    "arguments": "",
                                }
                            buf = self._tool_call_buffers[idx]

                            func = tc.get("function", {})
                            func_name = func.get("name", "")
                            if func_name:
                                buf["name"] += func_name

                            func_args = func.get("arguments", "")
                            if func_args:
                                buf["arguments"] += func_args

                            if func_name:
                                yield BackendToolCallDelta(
                                    index=idx, key="name", value=func_name
                                )
                            if func_args:
                                yield BackendToolCallDelta(
                                    index=idx, key="arguments", value=func_args
                                )

                    # Finish reason signals the end of this choice.
                    if finish_reason:
                        for tc in self._tool_call_buffers.values():
                            yield BackendToolCall(
                                id=tc["id"],
                                name=tc["name"],
                                arguments=tc["arguments"],
                            )
                        self._tool_call_buffers.clear()
                        yield BackendFinish(reason=finish_reason, usage=_stream_usage)
            finally:
                await response.aclose()

        async for event in _do_stream():
            yield event

    async def _non_stream_response(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        data: dict[str, Any],
        extra_kwargs: dict[str, Any],
    ) -> AsyncGenerator[BackendEvent, None]:
        """Send a non-streaming request and parse the single JSON response.

        The response is parsed into the same :class:`BackendEvent` types as
        the streaming path, ensuring the agent loop receives a uniform event
        stream regardless of the streaming setting.

        This method is decorated with :func:`retry_with_backoff` for automatic
        retry of transient HTTP errors.
        """
        @retry_with_backoff(config=self.retry_config)
        async def _do_non_stream() -> AsyncGenerator[BackendEvent, None]:
            response = await client.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                **extra_kwargs,
            )
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                error_msg = result["error"].get("message", str(result["error"]))
                yield BackendError(error=error_msg)
                return

            choices = result.get("choices", [])
            _non_stream_usage = _normalize_usage(result["usage"]) if result.get("usage") else None
            if not choices:
                yield BackendFinish(reason="stop", usage=_non_stream_usage)
                return

            choice = choices[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            # Extra events (reasoning_content, etc.)
            for extra in self._extract_extra_non_stream_events(message):
                yield extra

            # Text content.
            content = message.get("content", "")
            if content:
                yield BackendText(text=content)

            # Tool calls.
            tool_calls = message.get("tool_calls", [])
            for tc in tool_calls:
                func = tc.get("function", {})
                yield BackendToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=func.get("arguments", "{}"),
                )

            yield BackendFinish(reason=finish_reason, usage=_non_stream_usage)

        async for event in _do_non_stream():
            yield event
