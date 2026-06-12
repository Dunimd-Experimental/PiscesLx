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
Arcee AI backend — domain-adapted LLM API.

Arcee AI provides domain-adapted LLMs with OpenAI-compatible API access,
specializing in enterprise-grade models for specific verticals.

Base URL: https://api.arcee.ai/v2
Authentication: ARCEEAI_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class ArceeBackend(OpenAISSEBackend):
    """Arcee AI backend for domain-adapted models."""

    DEFAULT_BASE_URL = "https://api.arcee.ai/v2"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "arcee-v2",
        **kwargs: Any,
    ) -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def context_window_size(self) -> int:
        return 128000