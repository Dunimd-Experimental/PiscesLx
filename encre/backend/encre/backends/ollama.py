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
Ollama backend — locally-hosted models via the Ollama API.

Ollama is a local model runner that supports hundreds of open-source models
including Llama 3.x, Mistral, Qwen 2.5, DeepSeek, Gemma 2, Phi-4, and many
more.  Models run locally on CPU or GPU, with no data leaving the machine.

Key characteristics:
- No API key required (runs on localhost by default)
- OpenAI-compatible API at ``http://localhost:11434/v1``
- Supports tool/function calling (model-dependent)
- Context window varies by model (typically 4K-128K), dynamically queried
- No built-in prompt caching or thinking support
- Free and fully offline

This backend extends :class:`OpenAISSEBackend` because Ollama provides an
OpenAI-compatible API endpoint.  The default base URL is
``http://localhost:11434/v1``.

Context window detection
------------------------
On the first ``chat()`` call, this backend queries Ollama's native
``/api/show`` endpoint to retrieve the model's actual ``context_length``
(``num_ctx`` parameter).  The retrieved value is cached and returned by
``context_window_size()``.  If the query fails, the conservative default
of 8192 is used.
"""

from collections.abc import AsyncGenerator
from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend
from encre.utils.types import BackendEvent


class OllamaBackend(OpenAISSEBackend):
    """Ollama backend for locally-hosted open-source models.

    Connects to a local Ollama instance at ``http://localhost:11434/v1``
    (configurable via ``base_url``).  Supports any model available in the
    local Ollama library.

    This backend inherits all SSE streaming, tool calling, and error handling
    from :class:`OpenAISSEBackend`, with the addition of dynamic context
    window detection via Ollama's native API.

    Note:
        Tool calling support depends on the specific model being used.
        Some models (e.g., Llama 3.1+ and Qwen 2.5) support native tool
        calling, while others may not.
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "llama3.2",
        **kwargs: Any,
    ) -> None:
        """Initialise the Ollama backend.

        Args:
            api_key: Not required for local Ollama (defaults to empty string).
            base_url: Custom API base URL.  Defaults to
                ``http://localhost:11434/v1``.
            model: Model name.  Defaults to ``llama3.2``.  Must be a model
                that has been pulled into the local Ollama instance.
            **kwargs: Additional arguments passed to :class:`OpenAISSEBackend`.
        """
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)
        self._context_window_cache: int | None = None

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
        """Send a chat completion request, fetching model info first if needed.

        On the first call, queries Ollama's ``/api/show`` endpoint to
        retrieve the model's actual ``context_length`` for accurate
        context window reporting.
        """
        if self._context_window_cache is None:
            await self._fetch_model_context_window()
        async for event in super().chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            enable_caching=enable_caching,
        ):
            yield event

    def context_window_size(self) -> int:
        """Return the context window size for the loaded model.

        On first access, returns the conservative default of 8192.  After
        the first ``chat()`` call, returns the actual ``context_length``
        retrieved from Ollama's ``/api/show`` endpoint.
        """
        return self._context_window_cache or 8192

    async def _fetch_model_context_window(self) -> None:
        """Query Ollama's /api/show for the model's actual context length.

        Ollama stores the effective context length as ``num_ctx`` in the
        model info.  This method queries the native Ollama API (not the
        OpenAI-compatible endpoint) and caches the result.
        """
        try:
            client = self._get_client()
            base = self.api_base_url.rstrip("/").removesuffix("/v1")
            resp = await client.post(
                f"{base}/api/show",
                json={"name": self.model},
            )
            if resp.status_code == 200:
                data = resp.json()
                ctx = (
                    data.get("model_info", {}).get("context_length", 0)
                    or data.get("model_info", {}).get(
                        f"{self.model}.context_length", 0
                    )
                )
                if ctx and ctx > 0:
                    self._context_window_cache = ctx
                    return
        except Exception:
            pass
        self._context_window_cache = 8192

    async def list_models(self) -> list[str]:
        """Fetch available models from the local Ollama instance.

        Uses Ollama's native ``/api/tags`` endpoint since the OpenAI
        compatible ``/v1/models`` may not include all locally pulled models.
        """
        import time
        now = time.time()
        cache_key = f"ollama:{self.api_base_url}"
        if (
            hasattr(self, "_models_cache")
            and hasattr(self, "_models_cache_ts")
            and cache_key == getattr(self, "_models_cache_key", "")
            and now - self._models_cache_ts < 300
        ):
            return self._models_cache  # type: ignore[attr-defined]

        try:
            client = self._get_client()
            base = self.api_base_url.rstrip("/").removesuffix("/v1")
            resp = await client.get(f"{base}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                models.sort()
            else:
                models = await super().list_models()
        except Exception:
            models = await super().list_models()

        self._models_cache = models
        self._models_cache_ts = now
        self._models_cache_key = cache_key
        return models
