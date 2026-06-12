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
Xiaomi MiMo backend — MiMo-V2-Flash, MiMo-V2.5, MiMo-V2.5-Pro (2026 lineup).

Xiaomi's MiMo series models offer open-source, high-performance LLMs with
blazing-fast inference speeds and strong coding capabilities.  The API is
OpenAI-compatible and supports reasoning tokens.

Models:
- MiMo-V2-Flash: 309B MoE, 150 tok/s, SWE-Bench 73.4%
- MiMo-V2-omni: Multimodal variant
- MiMo-V2-Pro: Enhanced reasoning
- MiMo-V2.5: Improved general performance
- MiMo-V2.5-Pro: Premium reasoning

Base URL: https://api.xiaomimimo.com/v1
           https://platform.xiaomimimo.com/v1 (alias)
Authentication: XIAOMI_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend
from encre.utils.types import BackendEvent, create_backend_thinking


class XiaomiBackend(OpenAISSEBackend):
    """Xiaomi MiMo backend for the MiMo model series.

    Supports MiMo-V2-Flash, MiMo-V2.5, and MiMo-V2.5-Pro via Xiaomi's
    OpenAI-compatible API.  Reasoning tokens are extracted from
    ``reasoning_content``.
    """

    DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "mimo-v2.5-pro",
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