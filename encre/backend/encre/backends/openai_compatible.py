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
OpenAI-compatible backend — generic adapter for any OpenAI-compatible API.

Many LLM providers offer APIs that are compatible with OpenAI's chat
completion format, including:

- **vLLM**: Self-hosted inference server
- **Together AI**: Cloud API for open-source models
- **Fireworks AI**: Fast inference platform
- **Perplexity**: Search-augmented LLM API
- **OpenRouter**: Unified API for multiple providers
- **Any local/self-hosted server** using the OpenAI protocol

This backend extends :class:`OpenAISSEBackend` and allows users to specify
any base URL and model name, making it a universal adapter for the growing
ecosystem of OpenAI-compatible providers.

Key characteristics:
- Fully configurable base URL and model name
- Inherits all SSE streaming, tool calling, and error handling
- No provider-specific customisations needed
- Ideal for self-hosted or custom API endpoints
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class OpenAICompatibleBackend(OpenAISSEBackend):
    """Generic backend for any OpenAI-compatible API endpoint.

    This is a universal adapter that works with any provider serving an
    OpenAI-compatible chat completions API.  Configure it with the
    provider's base URL and desired model name.

    Examples:
        - vLLM: ``base_url="http://localhost:8000/v1"``
        - Together AI: ``base_url="https://api.together.xyz/v1"``
        - Fireworks AI: ``base_url="https://api.fireworks.ai/inference/v1"``
        - Perplexity: ``base_url="https://api.perplexity.ai"``
        - OpenRouter: ``base_url="https://openrouter.ai/api/v1"``

    Args:
        api_key: API key for the provider.
        base_url: Base URL of the OpenAI-compatible API endpoint.
            This is **required** and must be provided.
        model: Model name to use.  Defaults to ``"gpt-4.1-mini"``.
        **kwargs: Additional arguments passed to :class:`OpenAISSEBackend`.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gpt-4.1-mini",
        **kwargs: Any,
    ) -> None:
        """Initialise the OpenAI-compatible backend.

        Args:
            api_key: API key for the provider.
            base_url: Base URL of the OpenAI-compatible API endpoint.
                Must be provided (no default).
            model: Model name.  Defaults to ``gpt-4.1-mini``.
            **kwargs: Additional arguments passed to :class:`OpenAISSEBackend`.
        """
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def context_window_size(self) -> int:
        """Return a conservative context window estimate.

        Context window varies widely across providers and models.  Returns
        128000 as a reasonable default for most modern models.
        """
        return 128000