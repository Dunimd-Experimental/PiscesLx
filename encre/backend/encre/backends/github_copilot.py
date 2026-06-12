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
GitHub Copilot backend — uses GitHub Copilot subscription via OpenAI-compatible API.

GitHub Copilot provides an OpenAI-compatible chat completion API for
subscribers.  Authentication is handled via a GitHub token (OAuth device
code flow) or a COPILOT_GITHUB_TOKEN environment variable.

Note: This backend expects a valid GitHub token.  The OAuth device code
flow must be completed externally before using this backend.

Authentication (priority order):
1. Explicit ``api_key`` parameter
2. ``COPILOT_GITHUB_TOKEN`` environment variable
3. ``GH_TOKEN`` environment variable
4. Output of ``gh auth token`` (CLI)

Base URL: https://api.githubcopilot.com (determined dynamically)
"""

import os
import subprocess
from typing import Any

from encre.backends.openai_sse import OpenAISSEBackend


class GitHubCopilotBackend(OpenAISSEBackend):
    """GitHub Copilot backend for Copilot subscribers.

    Uses GitHub Copilot's OpenAI-compatible chat API.  Authentication
    requires a GitHub token with Copilot access.
    """

    DEFAULT_BASE_URL = "https://api.githubcopilot.com"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gpt-4o-copilot",
        **kwargs: Any,
    ) -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        resolved_key = api_key or self._resolve_github_token()
        super().__init__(api_key=resolved_key, base_url=base_url, model=model, **kwargs)

    @staticmethod
    def _resolve_github_token() -> str:
        token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
        if not token:
            try:
                result = subprocess.run(
                    ["gh", "auth", "token"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    token = result.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
        return token

    def context_window_size(self) -> int:
        return 128000