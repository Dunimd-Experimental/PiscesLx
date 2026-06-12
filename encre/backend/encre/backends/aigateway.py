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
AI Gateway backend — generic API gateway with OpenAI-compatible endpoints.

AI Gateway provides a unified API gateway for accessing various LLM providers
through a single endpoint.  This backend is compatible with any gateway that
exposes an OpenAI-compatible chat completions API.

Base URL: Configurable via AI_GATEWAY_BASE_URL or explicit base_url.
Authentication: AI_GATEWAY_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class AIGatewayBackend(OpenAISSEBackend):
    """AI Gateway backend for generic gateway access.

    Works with any AI API gateway that exposes an OpenAI-compatible
    chat completions endpoint.
    """

    DEFAULT_BASE_URL = ""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gpt-4.1-mini",
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def context_window_size(self) -> int:
        return 128000