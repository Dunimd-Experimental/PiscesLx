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
Groq backend — Llama 3.3 70B, Llama 4 Scout, GPT-OSS 120B (2026 lineup).

As of May 2026, Groq's inference platform offers ultra-low-latency access to:

- **Llama 3.3 70B (by Meta)**: A 70B-parameter model optimised for
  instruction-following and general chat.  131K context window.
  Pricing: $0.59/$0.79 per 1M tokens (input/output).

- **Llama 4 Scout (by Meta)**: Meta's latest 2026 model with improved
  reasoning and multilingual capabilities.  131K context window.
  Pricing: $0.29/$0.39 per 1M tokens.

- **GPT-OSS 120B (by Groq)**: Groq's own open-source 120B model, designed
  for high-throughput production workloads.  131K context window.
  Pricing: $0.49/$0.69 per 1M tokens.

All Groq models support:
- Tool/function calling (OpenAI-compatible format)
- 131K token context windows
- Ultra-low-latency inference via Groq's LPU hardware
- OpenAI-compatible API (can use OpenAISSEBackend directly)

This backend extends :class:`OpenAISSEBackend` because Groq uses an
OpenAI-compatible API.  The default base URL is
``https://api.groq.com/openai/v1``.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class GroqBackend(OpenAISSEBackend):
    """Groq backend for ultra-low-latency inference.

    Supports Llama 3.3 70B, Llama 4 Scout, and GPT-OSS 120B via Groq's
    OpenAI-compatible API at ``https://api.groq.com/openai/v1``.

    This backend inherits all SSE streaming, tool calling, and error handling
    from :class:`OpenAISSEBackend` without modification, as Groq's API is
    fully OpenAI-compatible.

    2026 pricing summary:
        - Llama 3.3 70B: $0.59/$0.79 per 1M tokens
        - Llama 4 Scout: $0.29/$0.39 per 1M tokens
        - GPT-OSS 120B: $0.49/$0.69 per 1M tokens
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "llama-3.3-70b-versatile",
        **kwargs: Any,
    ) -> None:
        """Initialise the Groq backend.

        Args:
            api_key: Groq API key.
            base_url: Custom API base URL.  Defaults to
                ``https://api.groq.com/openai/v1``.
            model: Model name.  Defaults to ``llama-3.3-70b-versatile``.
                Other valid values: ``llama-4-scout``, ``gpt-oss-120b``.
            **kwargs: Additional arguments passed to :class:`OpenAISSEBackend`.
        """
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def context_window_size(self) -> int:
        """Return the context window size for Groq models.

        All Groq-hosted models currently support 131,072 (128K) token
        context windows.
        """
        return 131072