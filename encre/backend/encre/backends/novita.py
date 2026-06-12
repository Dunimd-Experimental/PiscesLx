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
NovitaAI backend — 200+ models, Model API, Agent Sandbox, GPU Cloud.

NovitaAI provides access to 200+ open-source and proprietary models through
an OpenAI-compatible API.  It also offers Agent Sandbox and GPU Cloud services.

Base URL: https://api.novita.ai/v3/openai
Authentication: NOVITA_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class NovitaBackend(OpenAISSEBackend):
    """NovitaAI backend for 200+ model access.

    Provides access to open-source and proprietary models through NovitaAI's
    OpenAI-compatible API gateway.
    """

    DEFAULT_BASE_URL = "https://api.novita.ai/v3/openai"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "mistralai/mixtral-8x22b-instruct",
        **kwargs: Any,
    ) -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def context_window_size(self) -> int:
        return 128000