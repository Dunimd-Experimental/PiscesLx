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
AWS Bedrock backend — Converse API (Claude, Llama, Mistral, etc.).

AWS Bedrock is a managed service that provides access to foundation models
from Anthropic, Meta, Mistral, AI21, Cohere, and Amazon via a single API.
This backend uses the Bedrock Converse API, which provides a unified
interface for all supported models.

Supported model families (2026):
- **Anthropic Claude**: Opus 4.6/4.7, Sonnet 4.5/4.6, Haiku 4.5
- **Meta Llama**: Llama 3.3 70B, Llama 4 Scout
- **Mistral**: Mistral Large 2, Mistral Small
- **Amazon**: Nova Pro, Nova Lite
- **AI21**: Jamba 1.5
- **Cohere**: Command R+

Key characteristics:
- Requires AWS credentials (access key + secret key + region)
- Uses the ``aws-sdk`` (``boto3``) for authentication and API calls
- Supports streaming via the Converse Stream API
- Tool calling is supported for Claude and Llama models
- Context window varies by model (4K-200K)
- Pricing varies by model and region

This backend implements the Bedrock Converse API directly using ``boto3``,
not the OpenAI-compatible endpoint.  The Converse API provides a unified
interface that normalises model responses across providers.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

from encre.backends.base import BaseBackend
from encre.utils.types import (
    BackendEvent,
    create_backend_error,
    create_backend_finish,
    create_backend_text,
    create_backend_tool_call,
    create_backend_tool_call_delta,
)


class BedrockBackend(BaseBackend):
    """AWS Bedrock backend using the Converse API.

    Provides access to foundation models from Anthropic, Meta, Mistral,
    Amazon, AI21, and Cohere via AWS Bedrock's unified Converse API.

    Authentication is handled via AWS credentials (``aws_access_key_id``,
    ``aws_secret_access_key``, ``region_name``).  If not provided, the
    default AWS credential chain is used (environment variables, ~/.aws/credentials).

    Args:
        model: Bedrock model ID (e.g., ``"anthropic.claude-sonnet-4-20250514"``).
            Defaults to ``"anthropic.claude-sonnet-4-20250514"``.
        aws_access_key_id: Optional AWS access key ID.
        aws_secret_access_key: Optional AWS secret access key.
        region_name: AWS region name (e.g., ``"us-west-2"``).  Defaults to
            ``"us-east-1"``.
        **kwargs: Additional arguments (currently unused).
    """

    def __init__(
        self,
        model: str = "anthropic.claude-sonnet-4-20250514",
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
        region_name: str = "us-east-1",
        **_kwargs: Any,
    ) -> None:
        """Initialise the Bedrock backend.

        Args:
            model: Bedrock model ID.  Defaults to
                ``anthropic.claude-sonnet-4-20250514``.
            aws_access_key_id: Optional AWS access key ID.
            aws_secret_access_key: Optional AWS secret access key.
            region_name: AWS region.  Defaults to ``us-east-1``.
            **_kwargs: Additional arguments (currently unused).
        """
        self.model = model
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self._client = None

    async def _ensure_client(self) -> None:
        """Lazy-initialise the Bedrock runtime client.

        Creates a ``boto3`` Bedrock Runtime client with the configured
        credentials.  If no credentials are provided, the default AWS
        credential chain is used.

        The synchronous boto3 session creation is run in a thread pool
        executor to avoid blocking the async event loop.

        Raises:
            ImportError: If ``boto3`` is not installed.
        """
        if self._client is not None:
            return
        import asyncio

        def _create_client():
            try:
                import boto3
                session_kwargs: dict[str, Any] = {"region_name": self.region_name}
                if self.aws_access_key_id and self.aws_secret_access_key:
                    session_kwargs["aws_access_key_id"] = self.aws_access_key_id
                    session_kwargs["aws_secret_access_key"] = self.aws_secret_access_key
                session = boto3.Session(**session_kwargs)
                return session.client("bedrock-runtime")
            except ImportError as e:
                raise ImportError(
                    "boto3 not installed. Install with: pip install boto3"
                ) from e

        loop = asyncio.get_running_loop()
        self._client = await loop.run_in_executor(None, _create_client)

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Convert OpenAI-format messages to Bedrock Converse API format.

        Handles role mapping (``system`` → ``system``, ``assistant`` →
        ``assistant``, ``user`` → ``user``, ``tool`` → ``user`` with
        ``toolResult`` content blocks).

        Args:
            messages: OpenAI-format message list.

        Returns:
            A tuple of (converted_messages, system_message) where
            ``system_message`` is the extracted system prompt (or None).
        """
        converted: list[dict[str, Any]] = []
        system_message: dict[str, Any] | None = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_message = {"content": content}
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    content_blocks: list[dict[str, Any]] = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        func_args = func.get("arguments", "{}")
                        content_blocks.append({
                            "toolUse": {
                                "toolUseId": tc.get(
                                    "id", f"tooluse_{hash(str(tc))}"
                                ),
                                "name": func.get("name", ""),
                                "input": (
                                    json.loads(func_args) if func_args else {}
                                ),
                            }
                        })
                    converted.append({"role": "assistant", "content": content_blocks})
                else:
                    converted.append({"role": "assistant", "content": [{"text": content}]})
                continue

            if role == "tool":
                tool_use_id = msg.get("tool_call_id", f"tooluse_{hash(str(msg))}")
                converted.append({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{
                                "text": (
                                    content if isinstance(content, str)
                                    else json.dumps(content)
                                ),
                            }],
                        }
                    }],
                })
                continue

            if isinstance(content, str):
                converted.append({"role": "user", "content": [{"text": content}]})
            elif isinstance(content, list):
                blocks = []
                for item in content:
                    if item.get("type") == "text":
                        blocks.append({"text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        blocks.append(self._convert_image_block(item))
                converted.append({"role": "user", "content": blocks})

        return converted, system_message

    @staticmethod
    def _convert_image_block(item: dict[str, Any]) -> dict[str, Any]:
        """Convert an OpenAI-format image block to Bedrock image format.

        Handles both base64 data URLs (``data:image/jpeg;base64,...``)
        and raw URL references.  For data URLs, the base64 payload is
        decoded into raw bytes.  For external URLs, the raw bytes are
        fetched or passed through.

        Args:
            item: An OpenAI content block with ``type: "image_url"``.

        Returns:
            A Bedrock-format image block with ``image.format`` and
            ``image.source.bytes``.
        """
        import base64
        image_url = item.get("image_url", {})
        url = image_url.get("url", "")
        mime_type = "image/jpeg"

        if url.startswith("data:"):
            header, b64_data = url.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")
            image_bytes = base64.b64decode(b64_data)
        else:
            image_bytes = url.encode("utf-8")

        fmt = mime_type.split("/")[-1] if "/" in mime_type else "jpeg"
        return {
            "image": {
                "format": fmt,
                "source": {"bytes": image_bytes},
            }
        }

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-format tools to Bedrock Converse API tool format.

        Args:
            tools: OpenAI-format tool list.

        Returns:
            A list of Bedrock-format tool specifications.
        """
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                tool_spec: dict[str, Any] = {
                    "toolSpec": {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                    }
                }
                if func.get("parameters"):
                    tool_spec["toolSpec"]["inputSchema"] = {
                        "json": func["parameters"]
                    }
                converted.append(tool_spec)
        return converted

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

        Uses the Bedrock Converse API (streaming or non-streaming) to
        generate responses from the configured model.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Optional tool definitions in OpenAI format.
            tool_choice: Tool selection strategy (``"auto"``, ``"any"``, ``"none"``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: If True (default), uses the Converse Stream API.
            enable_caching: Not supported by Bedrock Converse API (ignored).

        Yields:
            :class:`BackendText`, :class:`BackendToolCallDelta`,
            :class:`BackendToolCall`, :class:`BackendFinish`, or
            :class:`BackendError`.
        """
        try:
            await self._ensure_client()
        except ImportError as e:
            yield create_backend_error(str(e))
            return

        import asyncio

        converted_messages, system_message = self._convert_messages(messages)

        inference_config: dict[str, Any] = {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }

        params: dict[str, Any] = {
            "modelId": self.model,
            "messages": converted_messages,
            "inferenceConfig": inference_config,
        }

        if system_message:
            params["system"] = [{"text": system_message["content"]}]

        if tools:
            params["toolConfig"] = {
                "tools": self._convert_tools(tools),
            }
            if tool_choice == "any":
                params["toolConfig"]["toolChoice"] = {"any": {}}
            elif tool_choice == "none":
                params["toolConfig"]["toolChoice"] = {"none": {}}

        try:
            loop = asyncio.get_running_loop()

            if stream:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.converse_stream(**params),
                )

                stream = response.get("stream", [])
                finish_reason: str = "stop"
                tool_use_buffer: dict[str, Any] | None = None
                tool_index: int = 0

                for event in stream:
                    if "contentBlockStart" in event:
                        start = event["contentBlockStart"]
                        tool_use = start.get("toolUse", {})
                        if tool_use:
                            tool_use_buffer = {
                                "toolUseId": tool_use.get("toolUseId", ""),
                                "name": tool_use.get("name", ""),
                                "input": "",
                            }
                            yield create_backend_tool_call_delta(
                                tool_index, "name", tool_use.get("name", "")
                            )

                    elif "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]
                        text_delta = delta.get("delta", {}).get("text", "")
                        if text_delta:
                            yield create_backend_text(text_delta)
                        tool_input_delta = (
                            delta.get("delta", {})
                            .get("toolInput", {})
                            .get("input", "")
                        )
                        if tool_input_delta and tool_use_buffer is not None:
                            if isinstance(tool_input_delta, str):
                                tool_use_buffer["input"] += tool_input_delta
                            td = tool_input_delta
                            arg_val = (
                                td if isinstance(td, str)
                                else json.dumps(td)
                            )
                            yield create_backend_tool_call_delta(
                                tool_index, "arguments", arg_val,
                            )

                    elif "contentBlockStop" in event:
                        if tool_use_buffer is not None:
                            buf_input = tool_use_buffer["input"]
                            arg_str = (
                                buf_input if isinstance(buf_input, str)
                                else json.dumps(buf_input)
                            )
                            yield create_backend_tool_call(
                                id=tool_use_buffer["toolUseId"],
                                name=tool_use_buffer["name"],
                                arguments=arg_str,
                            )
                            tool_use_buffer = None
                            tool_index += 1

                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason", "")
                        if stop_reason == "end_turn":
                            finish_reason = "stop"
                        elif stop_reason == "tool_use":
                            finish_reason = "tool_calls"
                        elif stop_reason == "max_tokens":
                            finish_reason = "max_tokens"
                        else:
                            finish_reason = stop_reason or "stop"

                    elif "internalServerException" in event:
                        yield create_backend_error("Bedrock internal server error")
                        return

                yield create_backend_finish(finish_reason)

            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.converse(**params),
                )

                output = response.get("output", {})
                message = output.get("message", {})
                content = message.get("content", [])

                for block in content:
                    if "text" in block:
                        yield create_backend_text(block["text"])
                    elif "toolUse" in block:
                        tool_use = block["toolUse"]
                        yield create_backend_tool_call(
                            id=tool_use.get("toolUseId", ""),
                            name=tool_use.get("name", ""),
                            arguments=json.dumps(tool_use.get("input", {})),
                        )

                stop_reason = response.get("stopReason", "end_turn")
                if stop_reason == "end_turn":
                    yield create_backend_finish("stop")
                elif stop_reason == "tool_use":
                    yield create_backend_finish("tool_calls")
                else:
                    yield create_backend_finish(stop_reason or "stop")

        except Exception as e:
            yield create_backend_error(str(e))

    def supports_tool_calling(self) -> bool:
        """Bedrock Converse API supports tool calling for Claude and Llama models."""
        return True

    def context_window_size(self) -> int:
        """Return a conservative context window estimate for Bedrock models.

        Context window varies by model (Claude: 200K, Llama: 131K, etc.).
        Returns 200000 as a safe upper bound for Claude models.
        """
        return 200000

    async def aclose(self) -> None:
        """Close the Bedrock client if initialised."""
        if self._client is not None:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._client.close)
            except Exception:
                pass
        self._client = None

    def supports_thinking(self) -> bool:
        """Bedrock supports thinking for Claude Opus and Sonnet models."""
        return True
