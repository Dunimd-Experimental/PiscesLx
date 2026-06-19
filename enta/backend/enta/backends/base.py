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
Abstract base class for all LLM backends.

Defines the :class:`BaseBackend` interface that every provider-specific backend
must implement. The core contract is the :meth:`chat` method, which accepts a
conversation history (OpenAI-format message list) and optional tool definitions,
then yields a stream of :class:`BackendEvent` items.

Lifecycle
---------
1. Instantiate the backend with provider-specific credentials and model name.
2. Call ``chat()`` in an ``async for`` loop to consume the event stream.
3. Call ``aclose()`` when done to release HTTP clients and GPU memory.

BackendEvent types emitted by chat()
-------------------------------------
- :class:`BackendText` -- a text delta (streaming chunk).
- :class:`BackendThinking` -- reasoning/thinking tokens (Anthropic, DeepSeek, Gemini).
- :class:`BackendToolCallDelta` -- partial tool call name or arguments.
- :class:`BackendToolCall` -- a complete tool call ready for execution.
- :class:`BackendFinish` -- signals the end of the response with a finish reason.
- :class:`BackendError` -- a non-recoverable error that terminated the stream.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from enta.utils.types import BackendEvent


class BaseBackend(ABC):
    """Abstract base class for LLM provider backends.

    Every backend in the ``enta.backends`` package extends this class and
    implements the abstract methods below.  The class also provides default
    implementations for optional capabilities (thinking, prompt caching, token
    counting) that subclasses may override when the provider supports them.

    Provider backends and their 2026 model support:

    +-----------------------+-----------------------------------------------+
    | Backend               | 2026 models                                   |
    +-----------------------+-----------------------------------------------+
    | OpenAIBackend         | GPT-4.1, GPT-4.1 Mini/Nano, GPT-5.x, o3,     |
    |                       | o4-mini (GPT-4o deprecated)                   |
    | AnthropicBackend      | Claude Opus 4.6/4.7, Sonnet 4.5/4.6,         |
    |                       | Haiku 4.5                                      |
    | GoogleBackend         | Gemini 2.5 Pro, Gemini 2.5 Flash              |
    | DeepSeekBackend       | DeepSeek V4-Flash, V4-Pro                     |
    |                       | (deepseek-chat/reasoner deprecated Jul 2026)  |
    | GroqBackend           | Llama 3.3 70B, Llama 4 Scout, GPT-OSS 120B   |
    | OllamaBackend         | Any model served by a local Ollama instance    |
    | LocalBackend          | Any Hugging Face transformers model            |
    | BedrockBackend        | Claude, Llama, Mistral via AWS Bedrock         |
    | OpenAICompatibleBackend| vLLM, SGLang, LiteLLM, llama.cpp, etc.       |
    +-----------------------+-----------------------------------------------+
    """

    @abstractmethod
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

        This is the central method of every backend.  It accepts an OpenAI-format
        message list (``[{"role": "user", "content": "..."}, ...]``) and yields
        :class:`BackendEvent` items as the response is produced.

        Args:
            messages: Conversation history in OpenAI message format. Each message
                has ``role`` (``"system"``, ``"user"``, ``"assistant"``, ``"tool"``)
                and ``content`` (string or list of content blocks).
            tools: Optional list of tool definitions in OpenAI function-calling
                format.  When provided, the model may request tool invocations.
            tool_choice: Controls tool selection behaviour.
                ``"auto"`` -- model decides; ``"any"`` -- must use a tool;
                ``"none"`` -- no tool usage; or a specific ``{"type": "function", "function": {"name": "..."}}``.  # noqa: E501
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum number of tokens to generate in the response.
            stream: If True (default), yields text/tool deltas as they arrive.
                If False, yields the complete response as a single burst.
            enable_caching: If True, enables prompt caching optimisations
                (Anthropic, OpenAI, DeepSeek V4 support this).

        Yields:
            BackendEvent items: :class:`BackendText`, :class:`BackendThinking`,
            :class:`BackendToolCallDelta`, :class:`BackendToolCall`,
            :class:`BackendFinish`, or :class:`BackendError`.
        """
        ...

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Return True if the backend/model supports function/tool calling.

        Backends that return False will have tool definitions stripped before
        the request is sent to the provider.
        """
        ...

    @abstractmethod
    def context_window_size(self) -> int:
        """Return the maximum context window size in tokens.

        This value is used by the agent loop to decide when context compaction
        is needed.  The returned value should reflect the model's actual limit,
        not a provider default.

        2026 reference values:
        - GPT-4.1 family: 1,048,576 (1M)
        - GPT-5.x: 128,000-400,000 (varies by variant)
        - Claude Opus/Sonnet 4.6: 200,000 (1M in beta)
        - Gemini 2.5 Pro: 1,048,576 (1M)
        - DeepSeek V4: 1,048,576 (1M)
        - Groq models: 131,072
        - Ollama: varies by model (default 8,192-131,072)
        """
        ...

    def supports_thinking(self) -> bool:
        """Return True if the backend can extract reasoning/thinking tokens.

        All 2026 backends support extracting ``reasoning_content`` from
        response deltas.  Whether the model actually emits thinking tokens
        is the model's decision -- the backend simply passes them through
        when present.
        """
        return True

    def supports_prompt_caching(self) -> bool:
        """Return True if the backend can request prompt caching.

        Most 2026 providers support some form of prompt caching.  The
        backend may inject cache-control headers or prefixes, but the
        provider decides whether to honor them.
        """
        return True

    def count_tokens(self, text: str) -> int:
        """Estimate the token count for a given text string.

        Returns -1 when the backend cannot provide an accurate count (the
        default).  Subclasses that have access to a tokenizer should override
        this to return a precise count.
        """
        return -1

    async def list_models(self) -> list[str]:
        """Return the list of available model IDs from this provider.

        Default implementation returns an empty list. Subclasses that support
        OpenAI-compatible APIs override this to call ``GET /models``.
        """
        return []

    async def aclose(self) -> None:
        """Release any resources held by this backend.

        This includes closing HTTP client sessions (httpx.AsyncClient),
        shutting down thread pools (LocalBackend), and releasing GPU memory.
        Called by the agent loop when the backend is no longer needed.
        """
        pass
