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
Kimi / Moonshot backend — Kimi-K2, Kimi-K2.5, Kimi-K2.6 (2026 lineup).

Moonshot AI's Kimi series models offer strong long-context reasoning and
coding capabilities.  The API is OpenAI-compatible.

Models:
- Kimi K2: Base reasoning model
- Kimi K2.5: Enhanced version with tool calling
- Kimi K2.6: Latest with extended capabilities

Two endpoints:
- Global: https://api.moonshot.cn/v1  (KIMI_API_KEY)
- China:  https://api.moonshot.cn/v1  (KIMI_CN_API_KEY, alias kimi-cn)

Base URL: https://api.moonshot.cn/v1
Authentication: KIMI_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend
from encre.utils.types import BackendEvent, create_backend_thinking


class KimiBackend(OpenAISSEBackend):
    """Kimi (Moonshot) backend for the Kimi K2 model series.

    Supports Kimi K2, K2.5, and K2.6 via Moonshot AI's OpenAI-compatible API.
    """

    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "kimi-k2.6",
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
        return 262144

    def supports_thinking(self) -> bool:
        return True