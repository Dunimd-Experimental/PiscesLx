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
Google backend — Gemini 2.5 Pro, Gemini 2.5 Flash (2026 lineup).

As of May 2026, Google's Gemini model lineup includes:

- **Gemini 2.5 Pro**: Google's most capable model with 1M token context
  window, multimodal support (text, image, video, audio), and thinking mode.
  Pricing: $1.25/$10 per 1M tokens (short context <=200K), $2.50/$15 per 1M
  tokens (long context >200K).

- **Gemini 2.5 Flash**: The fast, cost-effective variant with 1M context and
  similar multimodal capabilities.  Pricing: $0.15/$0.60 per 1M tokens.

Both models support:
- Tool/function calling (via Google's functionCall/functionResponse protocol)
- Thinking/reasoning tokens
- Multimodal inputs (text, images, video, audio)
- Google Search grounding (optional, via ``enable_grounding``)
- Streaming and non-streaming responses

This backend implements Google's Generative Language API directly (not
OpenAI-compatible), using the ``streamGenerateContent`` and
``generateContent`` endpoints.  The protocol uses a different message format
than OpenAI: roles are ``user``/``model``/``function``, and tool calls use
``functionCall``/``functionResponse`` blocks instead of OpenAI's
``tool_calls`` array.

Retry logic via :func:`retry_with_backoff` handles transient HTTP errors
(429, 502, 503, 504) and network timeouts for both streaming and non-streaming
requests.
"""

import asyncio
import json
import random
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from encre.backends.base import BaseBackend
from encre.utils.types import (
    BackendEvent,
    create_backend_error,
    create_backend_finish,
    create_backend_text,
    create_backend_tool_call,
    create_backend_tool_call_delta,
)


class GoogleBackend(BaseBackend):
    """Google backend for the 2026 Gemini model lineup.

    Supports Gemini 2.5 Pro (default) and Gemini 2.5 Flash via Google's
    Generative Language API at ``https://generativelanguage.googleapis.com/v1beta``.

    This backend handles the full protocol conversion between OpenAI's message
    format and Google's format, including:
    - Role mapping: ``user`` -> ``user``, ``assistant`` -> ``model``,
      ``tool`` -> ``function``, ``system`` -> ``systemInstruction``
    - Tool conversion: OpenAI ``function`` tools -> Google ``functionDeclaration``
    - Content block conversion: text, image_url, and image_data blocks
    - Tool call buffering for streaming responses
    - Finish reason mapping (STOP, MAX_TOKENS, SAFETY, etc.)

    Optional Google Search grounding can be enabled via the ``enable_grounding``
    parameter, which adds a ``googleSearch`` tool to the request.
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gemini-2.5-pro",
        enable_grounding: bool = False,
        **_kwargs: Any,
    ) -> None:
        """Initialise the Google backend.

        Args:
            api_key: Google AI Studio API key.  Required for authentication
                via the ``key`` query parameter.
            base_url: Custom API base URL.  Defaults to
                ``https://generativelanguage.googleapis.com/v1beta``.
            model: Model name.  Defaults to ``gemini-2.5-pro``.  Other valid
                values: ``gemini-2.5-flash``.
            enable_grounding: If True, enables Google Search grounding for
                real-time information retrieval.
            **_kwargs: Additional arguments (currently unused).
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") or self.DEFAULT_BASE_URL
        self.model = model
        self.enable_grounding = enable_grounding
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=30.0),
        )

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Convert OpenAI-format messages to Google Generative Language format.

        Handles role mapping, content block conversion, and system instruction
        extraction.  Tool call messages from the assistant are converted to
        ``functionCall`` blocks, and tool response messages are converted to
        ``functionResponse`` blocks.

        Args:
            messages: OpenAI-format message list.

        Returns:
            A tuple of (contents, system_instruction) where ``contents`` is the
            converted message list and ``system_instruction`` is the extracted
            system prompt (or None).
        """
        contents: list[dict[str, Any]] = []
        system_instruction: dict[str, Any] | None = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = {
                    "parts": [{"text": content}]
                }
                continue

            parts: list[dict[str, Any]] = []

            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url:
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_url,
                                },
                            })
                    elif item.get("type") == "image" or item.get("type") == "image_data":
                        parts.append({
                            "inline_data": {
                                "mime_type": item.get("mime_type", "image/jpeg"),
                                "data": item.get("data", ""),
                            }
                        })
            elif isinstance(content, dict):
                parts.append({"text": json.dumps(content)})

            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        func_args = func.get("arguments", "")
                        parts.append({
                            "functionCall": {
                                "name": func.get("name", ""),
                                "args": (
                                    json.loads(func_args)
                                    if func_args else {}
                                ),
                            },
                        })
                else:
                    mapped_role = "model"
                    contents.append({"role": mapped_role, "parts": parts})
                    continue

            elif role == "tool":
                tool_name = msg.get("name", "")
                resp_content = (
                    content if isinstance(content, str)
                    else json.dumps(content)
                )
                mapped_content: list[dict[str, Any]] = [{
                    "functionResponse": {
                        "name": tool_name,
                        "response": {"content": resp_content},
                    }
                }]
                mapped_role = "function"
                contents.append({"role": mapped_role, "parts": mapped_content})
                continue

            if role == "user":
                mapped_role = "user"
            elif role == "assistant":
                mapped_role = "model"
            else:
                mapped_role = "user"

            contents.append({"role": mapped_role, "parts": parts})

        return contents, system_instruction

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tools to Google function declarations.

        Args:
            tools: OpenAI-format tool list (``[{"type": "function", "function": {...}}]``).

        Returns:
            A list of Google-format tool declarations with ``functionDeclarations``.
        """
        function_declarations: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                declaration: dict[str, Any] = {
                    "name": func.get("name", ""),
                }
                if func.get("description"):
                    declaration["description"] = func["description"]
                if func.get("parameters"):
                    declaration["parameters"] = func["parameters"]
                function_declarations.append(declaration)
        return [{"functionDeclarations": function_declarations}]

    def _build_body(
        self,
        contents: list[dict[str, Any]],
        system_instruction: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Build the request body for Gemini API calls."""
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }

        if tool_choice == "any":
            generation_config["toolConfig"] = {
                "functionCallingConfig": {"mode": "ANY"}
            }
        elif tool_choice == "none":
            generation_config["toolConfig"] = {
                "functionCallingConfig": {"mode": "NONE"}
            }
        elif tool_choice == "auto":
            generation_config["toolConfig"] = {
                "functionCallingConfig": {"mode": "AUTO"}
            }

        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        if system_instruction:
            body["systemInstruction"] = system_instruction

        if tools:
            body["tools"] = self._convert_tools(tools)

        if self.enable_grounding:
            body["tools"] = [*body.get("tools", []), {"googleSearch": {}}]

        return body

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
        _enable_caching: bool = False,
    ) -> AsyncGenerator[BackendEvent, None]:
        """Send a chat completion request and stream back events.

        Implements Google's Generative Language API with SSE streaming for
        ``streamGenerateContent`` and non-streaming for ``generateContent``.
        Both paths use exponential backoff retry for transient errors.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Optional tool definitions in OpenAI format.
            tool_choice: Tool selection strategy (``"auto"``, ``"any"``, ``"none"``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: If True (default), uses SSE streaming.
            enable_caching: Not yet supported by Google's API (ignored).

        Yields:
            :class:`BackendText`, :class:`BackendToolCallDelta`,
            :class:`BackendToolCall`, :class:`BackendFinish`, or
            :class:`BackendError`.
        """
        contents, system_instruction = self._convert_messages(messages)
        body = self._build_body(
            contents, system_instruction,
            tools, tool_choice, temperature, max_tokens,
        )

        endpoint = (
            f"/models/{self.model}:streamGenerateContent"
            if stream
            else f"/models/{self.model}:generateContent"
        )
        url = (
            f"{self.base_url}{endpoint}?key={self.api_key}&alt=sse"
            if stream
            else f"{self.base_url}{endpoint}?key={self.api_key}"
        )

        try:
            if stream:
                async for event in self._stream_with_retry(url, body):
                    yield event
            else:
                async for event in self._non_stream_with_retry(url, body):
                    yield event
        except Exception as e:
            yield create_backend_error(str(e))

    async def _stream_with_retry(
        self, url: str, body: dict[str, Any]
    ) -> AsyncGenerator[BackendEvent, None]:
        """Stream response with exponential backoff retry.

        Retries on 429/502/503/504 status codes, timeouts, and connection
        errors.  On retry the entire stream is re-requested from scratch.
        """
        max_retries = 5
        rate_limit_retries = 8
        base_delay = 1.0
        max_delay = 60.0

        for attempt in range(max(rate_limit_retries, max_retries) + 1):
            try:
                async for event in self._do_stream(url, body):
                    yield event
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 502, 503, 504}:
                    yield create_backend_error(
                        f"Gemini API error {exc.response.status_code}"
                    )
                    return
                if exc.response.status_code == 429 and attempt >= rate_limit_retries:
                    yield create_backend_error("Gemini rate limit exhausted")
                    return
                if exc.response.status_code != 429 and attempt >= max_retries:
                    yield create_backend_error("Gemini server error retries exhausted")
                    return
            except (httpx.TimeoutException, httpx.ConnectError,
                    httpx.RemoteProtocolError, httpx.TransportError):
                if attempt >= max_retries:
                    yield create_backend_error("Gemini network error retries exhausted")
                    return

            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(random.uniform(0, delay))

    async def _non_stream_with_retry(
        self, url: str, body: dict[str, Any]
    ) -> AsyncGenerator[BackendEvent, None]:
        """Non-streaming response with exponential backoff retry."""
        max_retries = 5
        rate_limit_retries = 8
        base_delay = 1.0
        max_delay = 60.0

        for attempt in range(max(rate_limit_retries, max_retries) + 1):
            try:
                async for event in self._do_non_stream(url, body):
                    yield event
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {429, 502, 503, 504}:
                    yield create_backend_error(
                        f"Gemini API error {exc.response.status_code}"
                    )
                    return
                if exc.response.status_code == 429 and attempt >= rate_limit_retries:
                    yield create_backend_error("Gemini rate limit exhausted")
                    return
                if exc.response.status_code != 429 and attempt >= max_retries:
                    yield create_backend_error("Gemini server error retries exhausted")
                    return
            except (httpx.TimeoutException, httpx.ConnectError,
                    httpx.RemoteProtocolError, httpx.TransportError):
                if attempt >= max_retries:
                    yield create_backend_error("Gemini network error retries exhausted")
                    return

            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(random.uniform(0, delay))

    async def _do_stream(
        self, url: str, body: dict[str, Any]
    ) -> AsyncGenerator[BackendEvent, None]:
        """Execute a single streaming request to Gemini API."""
        async with self._client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                resp.raise_for_status()
                yield create_backend_error(
                    f"Gemini API error {resp.status_code}: {error_body.decode()}"
                )
                return

            tool_call_buffers: dict[int, dict[str, Any]] = {}
            current_idx = 0
            finish_reason: str = "stop"
            accumulated_text: dict[int, str] = {}

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload:
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                candidates = chunk.get("candidates", [])
                if not candidates:
                    continue

                candidate = candidates[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                for part_idx, part in enumerate(parts):
                    if "text" in part:
                        text = part.get("text", "")
                        if text:
                            if part_idx not in accumulated_text:
                                accumulated_text[part_idx] = ""
                            prev = accumulated_text[part_idx]
                            new_part = (
                                text[len(prev):] if text.startswith(prev)
                                else text
                            )
                            accumulated_text[part_idx] = text
                            if new_part:
                                yield create_backend_text(new_part)
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        buf_idx = current_idx
                        current_idx += 1
                        name = fc.get("name", "")
                        args = fc.get("args", {})
                        args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                        tool_call_buffers[buf_idx] = {
                            "id": f"call_{buf_idx}",
                            "name": name,
                            "arguments": args_str,
                        }
                        yield create_backend_tool_call_delta(buf_idx, "name", name)
                        yield create_backend_tool_call_delta(buf_idx, "arguments", args_str)

                finish = candidate.get("finishReason")
                if finish:
                    finish_reason = self._map_finish_reason(finish)

            for idx in sorted(tool_call_buffers.keys()):
                buf = tool_call_buffers[idx]
                yield create_backend_tool_call(
                    id=buf["id"],
                    name=buf["name"],
                    arguments=buf["arguments"],
                )

            yield create_backend_finish(finish_reason)

    async def _do_non_stream(
        self, url: str, body: dict[str, Any]
    ) -> AsyncGenerator[BackendEvent, None]:
        """Execute a single non-streaming request to Gemini API."""
        resp = await self._client.post(url, json=body)
        resp.raise_for_status()

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "unknown")
            yield create_backend_error(f"Gemini blocked: {block_reason}")
            return

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "STOP")
        mapped_reason = self._map_finish_reason(finish_reason)

        content = candidate.get("content", {})
        parts = content.get("parts", [])

        tool_call_buffers: dict[int, dict[str, Any]] = {}
        current_idx = 0

        for part in parts:
            if "text" in part:
                yield create_backend_text(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                buf_idx = current_idx
                current_idx += 1
                name = fc.get("name", "")
                args = fc.get("args", {})
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                tool_call_buffers[buf_idx] = {
                    "id": f"call_{buf_idx}",
                    "name": name,
                    "arguments": args_str,
                }
                yield create_backend_tool_call_delta(buf_idx, "name", name)
                yield create_backend_tool_call_delta(buf_idx, "arguments", args_str)

        for idx in sorted(tool_call_buffers.keys()):
            buf = tool_call_buffers[idx]
            yield create_backend_tool_call(
                id=buf["id"],
                name=buf["name"],
                arguments=buf["arguments"],
            )

        yield create_backend_finish(mapped_reason)

    def _map_finish_reason(self, reason: str) -> str:
        """Map Google finish reasons to unified finish reasons.

        Google uses uppercase finish reasons (STOP, MAX_TOKENS, SAFETY, etc.)
        which are mapped to the unified format used by the agent loop.

        Args:
            reason: Google's finish reason string.

        Returns:
            A unified finish reason string (``"stop"``, ``"max_tokens"``, ``"error"``).
        """
        mapping = {
            "STOP": "stop",
            "MAX_TOKENS": "max_tokens",
            "SAFETY": "error",
            "RECITATION": "error",
            "MALFORMED_FUNCTION_CALL": "error",
            "OTHER": "stop",
        }
        return mapping.get(reason, "stop")

    def supports_tool_calling(self) -> bool:
        """Gemini models support function calling via functionDeclarations."""
        return True

    def context_window_size(self) -> int:
        """Return the context window size for Gemini models.

        Both Gemini 2.5 Pro and 2.5 Flash support 1,048,576 (1M) token
        context windows.
        """
        return 1048576

    def count_tokens(self, text: str) -> int:
        """Estimate token count using tiktoken or char/4 heuristic.

        Uses Gemini-compatible token estimation.  For precise counts,
        use the Cloud AI API ``countTokens`` endpoint.
        """
        if not text:
            return 0
        try:
            from encre.utils.tokens import estimate_tokens
            return estimate_tokens(text, model="gemini-pro")
        except Exception:
            return len(text) // 4

    async def list_models(self) -> list[str]:
        """Fetch available models from Google's models endpoint.

        Returns a list of model names available to the API key.
        Results are cached for 5 minutes.
        """
        import time
        now = time.time()
        cache_key = f"google:{self.api_key[:8] if self.api_key else 'noauth'}"
        if (
            hasattr(self, "_models_cache")
            and hasattr(self, "_models_cache_ts")
            and cache_key == getattr(self, "_models_cache_key", "")
            and now - self._models_cache_ts < 300
        ):
            return self._models_cache  # type: ignore[attr-defined]

        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
            models: list[str] = []
            for item in data.get("models", []):
                name = item.get("name", "")
                if name:
                    name = name.replace("models/", "")
                    models.append(name)
            models.sort()
        except Exception:
            models = []

        self._models_cache = models
        self._models_cache_ts = now
        self._models_cache_key = cache_key
        return models

    async def aclose(self) -> None:
        """Close the HTTP client session."""
        await self._client.aclose()

    def supports_thinking(self) -> bool:
        """Gemini 2.5 models support thinking/reasoning tokens."""
        return True

    def supports_grounding(self) -> bool:
        """Gemini models support Google Search grounding."""
        return True
