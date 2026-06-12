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
MiniMax backend — MiniMax-M2.1, M2.5, M2.7 (2026 lineup).

MiniMax offers powerful Chinese and English LLMs with both OpenAI-compatible
and Anthropic-compatible endpoints.  This backend uses the OpenAI-compatible
endpoint.

Models:
- MiniMax-M2.1: Balanced general-purpose model
- MiniMax-M2.5: Enhanced reasoning with tool calling
- MiniMax-M2.7: Latest flagship model

Two endpoints:
- Global: https://api.minimax.chat/v1  (MINIMAX_API_KEY)
- China:  https://api.minimax.chat/v1  (MINIMAX_CN_API_KEY, alias minimax-cn)

Base URL: https://api.minimax.chat/v1
Authentication: MINIMAX_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class MiniMaxBackend(OpenAISSEBackend):
    """MiniMax backend for the MiniMax M2 model series.

    Supports MiniMax-M2.1, M2.5, and M2.7 via MiniMax's OpenAI-compatible API.
    """

    DEFAULT_BASE_URL = "https://api.minimax.chat/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "MiniMax-M2.7",
        **kwargs: Any,
    ) -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def context_window_size(self) -> int:
        return 196000

    def supports_thinking(self) -> bool:
        return True