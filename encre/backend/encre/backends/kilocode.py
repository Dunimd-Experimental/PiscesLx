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
Kilo Code backend — unified API gateway for multi-model access.

Kilo Code's AI Gateway provides a unified API endpoint that routes requests
to many models (Anthropic, OpenAI, Google, etc.) through a single API key
and endpoint.

Base URL: https://api.kilo.ai/api/gateway
Authentication: KILOCODE_API_KEY environment variable or explicit api_key.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class KiloCodeBackend(OpenAISSEBackend):
    """Kilo Code Gateway backend for unified multi-model access.

    Routes requests through Kilo Code's AI Gateway, which supports various
    models from multiple providers through a single endpoint.
    """

    DEFAULT_BASE_URL = "https://api.kilo.ai/api/gateway"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "kilocode/kilo/auto",
        **kwargs: Any,
    ) -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def context_window_size(self) -> int:
        return 1000000