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
GLM (Zhipu AI) backend — GLM-4.5, GLM-4.6, GLM-4.7 (2026 lineup).

Zhipu AI's GLM series models are among the leading Chinese LLMs, offering
strong reasoning, coding, and multilingual capabilities.  The API is
OpenAI-compatible and supports thinking/reasoning tokens.

Models:
- GLM-4.5: Balanced general-purpose model
- GLM-4.6: Enhanced reasoning with tool calling
- GLM-4.7: Latest flagship with extended context

Base URL: https://open.bigmodel.cn/api/paas/v4
Authentication: GLM_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend
from encre.utils.types import BackendEvent, create_backend_thinking


class GLMBackend(OpenAISSEBackend):
    """GLM (Zhipu AI) backend for the GLM-4.x model series.

    Supports GLM-4.5, GLM-4.6, and GLM-4.7 via Zhipu AI's OpenAI-compatible
    API.  Thinking/reasoning tokens are extracted from ``reasoning_content``.
    """

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "glm-4.7",
        **kwargs: Any,
    ) -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def _build_request_data(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
    ) -> dict[str, Any]:
        data = super()._build_request_data(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        return data

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
        return 131072

    def supports_thinking(self) -> bool:
        return True