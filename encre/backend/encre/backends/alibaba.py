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
Alibaba DashScope backend — Qwen series models (2026 lineup).

Alibaba Cloud's DashScope (Model Studio) provides access to the Qwen series
models including Qwen-Max, Qwen-Plus, Qwen-Flash, and QwQ (reasoning models).
The API is OpenAI-compatible.

Models:
- Qwen-Max series: Flagship models (qwen-max, qwen3-max)
- Qwen-Plus series: Balanced performance (qwen-plus, qwen3-plus)
- Qwen-Flash series: Fast and economical (qwen-flash, qwen3-flash)
- QwQ series: Reasoning-focused (qwq-plus)
- Qwen-Coder: Coding-optimized

Two service types:
- Standard: provider="alibaba", env=DASHSCOPE_API_KEY
- Coding Plan: provider="alibaba-coding-plan", env=DASHSCOPE_API_KEY

Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
Authentication: DASHSCOPE_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend
from encre.utils.types import BackendEvent, create_backend_thinking


class AlibabaBackend(OpenAISSEBackend):
    """Alibaba DashScope backend for the Qwen model series.

    Supports Qwen-Max, Qwen-Plus, Qwen-Flash, QwQ, and Qwen-Coder models
    via Alibaba Cloud's OpenAI-compatible API.
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "qwen-plus",
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
        if self.model and self.model.lower().startswith("qwq"):
            data["enable_thinking"] = True
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