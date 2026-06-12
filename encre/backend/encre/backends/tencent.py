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
Tencent TokenHub backend — Hunyuan and third-party model access.

Tencent Cloud's TokenHub platform provides access to Hunyuan models and
third-party models (DeepSeek, GLM, Kimi, MiniMax, etc.) through an
OpenAI-compatible API.

Models:
- Hunyuan Hy3 (preview): Latest flagship
- Hunyuan TurboS / Turbo: Previous generation
- Third-party: DeepSeek, GLM, Kimi, MiniMax, etc.

Base URL: https://tokenhub.tencentmaas.com/v1
Authentication: TOKENHUB_API_KEY environment variable or explicit api_key.

Aliases: tencent, tokenhub, tencentmaas
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend
from encre.utils.types import BackendEvent, create_backend_thinking


class TencentBackend(OpenAISSEBackend):
    """Tencent TokenHub backend for Hunyuan and third-party models.

    Provides access to Tencent Cloud's LLM platform via an OpenAI-compatible
    API.  Both Hunyuan native models and third-party models are supported.
    """

    DEFAULT_BASE_URL = "https://tokenhub.tencentmaas.com/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "hy3-preview",
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
        return 131072

    def supports_thinking(self) -> bool:
        return True