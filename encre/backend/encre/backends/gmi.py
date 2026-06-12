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
GMI Cloud backend — GPU cloud infrastructure with multi-model API.

GMI Cloud provides GPU cloud infrastructure with access to top AI models
including Claude, GPT, DeepSeek, Gemini, Kimi, Qwen, and more through an
OpenAI-compatible API.

Base URL: https://api.gmi-serving.com/v1
Authentication: GMI_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend
from encre.utils.types import BackendEvent, create_backend_thinking


class GMIBackend(OpenAISSEBackend):
    """GMI Cloud backend for GPU-accelerated multi-model access.

    Provides access to various models via GMI Cloud's OpenAI-compatible API.
    Supports thinking/reasoning tokens when available.
    """

    DEFAULT_BASE_URL = "https://api.gmi-serving.com/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "deepseek-ai/DeepSeek-V3.2",
        **kwargs: Any,
    ) -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def _extract_extra_stream_events(
        self, delta: dict[str, Any]
    ) -> list[BackendEvent]:
        reasoning = delta.get("reasoning_content")
        if reasoning:
            return [create_backend_thinking(reasoning)]
        return []

    def _extract_extra_non_stream_events(
        self, message: dict[str, Any]
    ) -> list[BackendEvent]:
        reasoning = message.get("reasoning_content")
        if reasoning:
            return [create_backend_thinking(reasoning)]
        return []

    def context_window_size(self) -> int:
        return 163000

    def supports_thinking(self) -> bool:
        return True