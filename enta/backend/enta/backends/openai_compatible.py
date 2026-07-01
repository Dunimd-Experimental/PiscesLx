#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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
OpenAI-compatible backend -- generic adapter for any OpenAI-compatible API.

Many LLM providers offer APIs that are compatible with OpenAI's chat
completion format, including:

- **vLLM**: Self-hosted inference server
- **Together AI**: Cloud API for open-source models
- **Fireworks AI**: Fast inference platform
- **Perplexity**: Search-augmented LLM API
- **OpenRouter**: Unified API for multiple providers
- **Any local/self-hosted server** using the OpenAI protocol

This backend extends :class:`OpenAISSEBackend` and supports:

- SSE streaming with text, tool calls, and reasoning/thinking tokens
- Provider-specific reasoning content (``reasoning_content`` delta field)
- Context window sizing from the model config (defaults to 1M for large models)
- Automatic URL normalisation (appends ``/v1`` if missing)
"""

import re
from typing import Any

from enta.backends.openai_sse import OpenAISSEBackend

# Known 1M-context model families (prefix or regex patterns) for auto-detection
_LARGE_CONTEXT_PATTERNS: list[str] = [
    "gpt-4.1", "gpt-5.5", "gpt-5.6", "gpt-5.4", "gpt-6",
    "deepseek-v4", "deepseek-chat", "deepseek-reasoner",
    "claude-opus-4-7", "claude-sonnet-4-6", "claude-4",
    "gemini-2.5", "gemini-2.0", "gemini-3",
    "qwen3", "qwen-max", "qwen-plus", "qwen-long",
    "llama-4", "llama-3.3",
    "mixtral", "mistral-large", "mistral-medium",
    "yi-large", "yi-medium",
    "hunyuan", "ernie-4",
    "doubao", "glm-5",
    "kimi-k2", "minimax-m2",
]

_SMALL_CONTEXT_PATTERNS: list[str] = [
    # Legacy models that are NOT 1M
    "gpt-4o-2024", "gpt-4-1106", "gpt-4-0125",
    "gpt-3.5", "text-embedding",
    "deepseek-v1", "deepseek-v2",
    "glm-4", "glm-3",
]


class OpenAICompatibleBackend(OpenAISSEBackend):
    """Generic backend for any OpenAI-compatible API endpoint.

    Works with any provider serving an OpenAI-compatible chat completions API.
    Supports reasoning/thinking tokens (``reasoning_content``) from providers
    like OpenRouter, Together AI, Fireworks, vLLM, and others.

    Configure with the provider's base URL and desired model name.

    Examples:
        - vLLM: ``base_url="http://localhost:8000/v1"``
        - Together AI: ``base_url="https://api.together.xyz/v1"``
        - Fireworks AI: ``base_url="https://api.fireworks.ai/inference/v1"``
        - Perplexity: ``base_url="https://api.perplexity.ai"``
        - OpenRouter: ``base_url="https://openrouter.ai/api/v1"``

    Args:
        api_key: API key for the provider.
        base_url: Base URL of the OpenAI-compatible API endpoint.
        model: Model name to use.
        context_window: Optional explicit context window size override.
        **kwargs: Additional arguments passed to :class:`OpenAISSEBackend`.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gpt-4.1-mini",
        context_window: int = 0,
        **kwargs: Any,
    ) -> None:
        # Normalise the base URL: many providers use /v1, but users may omit it.
        base_url = (base_url or "").rstrip("/")
        if base_url and not base_url.endswith("/v1"):
            # Check if it already has a versioned path like /api/v1 or /inference/v1
            if not re.search(r"/v\d+$", base_url):
                base_url = base_url + "/v1"
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)
        self._context_window = context_window

    # ── Context window ──────────────────────────────────────────────────

    def context_window_size(self) -> int:
        """Return the context window size for this model.

        Explicitly configured value takes priority.  Otherwise, heuristics
        based on known model families are used.  Defaults to 128 000 tokens
        as a safe fallback for unknown models.
        """
        if self._context_window > 0:
            return self._context_window
        return _detect_context_window(self.model)


def _detect_context_window(model: str) -> int:
    """Guess the context window size from the model name.

    For known large-context model families (GPT-4.1, DeepSeek V4, Claude 4,
    etc.) returns 1 048 576 (1M).  For unreckognised models returns 256 000
    as a reasonable modern default since most 2026 models have >= 200K.
    """
    model_lower = model.lower()
    for pattern in _SMALL_CONTEXT_PATTERNS:
        if pattern in model_lower:
            return 128_000
    for pattern in _LARGE_CONTEXT_PATTERNS:
        if pattern in model_lower:
            return 1_048_576
    return 256_000
