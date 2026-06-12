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
LM Studio backend — local model server with OpenAI-compatible API.

LM Studio runs local LLMs through an OpenAI-compatible HTTP server.
No API key is typically required for local usage, but one can be set
if the server is configured with authentication.

Base URL: http://localhost:1234/v1 (default LM Studio port)
Authentication: Optional, LM_API_KEY environment variable.
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class LMStudioBackend(OpenAISSEBackend):
    """LM Studio backend for locally-hosted models.

    Connects to LM Studio's local HTTP server which serves models
    through an OpenAI-compatible API at ``http://localhost:1234/v1``.
    """

    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "local-model",
        **kwargs: Any,
    ) -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def context_window_size(self) -> int:
        return 8192