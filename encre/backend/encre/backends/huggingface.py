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
Hugging Face backend — Inference API for open-source models.

Hugging Face provides access to thousands of open-source models through
its Inference API, which now supports an OpenAI-compatible chat completion
endpoint.

Base URL: https://api-inference.huggingface.co/v1/
Authentication: HF_TOKEN environment variable or explicit api_key.

Aliases: huggingface, hf
"""

from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class HuggingFaceBackend(OpenAISSEBackend):
    """Hugging Face Inference API backend.

    Provides access to thousands of open-source models via Hugging Face's
    OpenAI-compatible API endpoint.
    """

    DEFAULT_BASE_URL = "https://api-inference.huggingface.co/v1/"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "meta-llama/Llama-3.3-70B-Instruct",
        **kwargs: Any,
    ) -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def context_window_size(self) -> int:
        return 128000