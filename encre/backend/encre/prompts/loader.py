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
Prompt file loader.

Reads plaintext ``.prompt`` files from ``blocks/``, ``skills/``, etc. at
runtime and caches them for the lifetime of the process.

Usage::

    loader = PromptLoader()
    content = loader.load("identity")
    content = loader.load_with_context("permission", mode="bypass", guidance="...")
"""

from __future__ import annotations

import os
from typing import Any

_PROMPTS_ROOT: str = os.path.abspath(os.path.dirname(__file__))
_CACHE: dict[str, str] = {}


class PromptLoader:
    """Loads and caches prompt files.

    Prompt files live under ``<prompts_root>/<category>/<name>.prompt``.
    """

    def __init__(self, root: str | None = None) -> None:
        self._root = root or _PROMPTS_ROOT

    def load(self, name: str, category: str = "blocks") -> str:
        """Load a prompt file, cache, return content.

        Parameters
        ----------
        name:
            Prompt file name (without ``.prompt`` suffix), e.g. ``"identity"``.
        category:
            Sub-directory under the prompts root, e.g. ``"blocks"``, ``"skills"``.
        """
        cache_key = f"{category}/{name}"
        if cache_key in _CACHE:
            return _CACHE[cache_key]

        path = os.path.join(self._root, category, f"{name}.prompt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompt file not found: {path}. "
                f"Ensure the file exists under prompts/{category}/."
            )

        _CACHE[cache_key] = content
        return content

    def load_with_context(self, name: str, category: str = "blocks", **ctx: Any) -> str:
        """Load a prompt and substitute ``{{key}}`` placeholders with *ctx* values."""
        content = self.load(name, category=category)
        for key, val in ctx.items():
            content = content.replace(f"{{{{{key}}}}}", str(val))
        return content

    def get_block_path(self, name: str, category: str = "blocks") -> str:
        """Return the absolute path to a prompt file (for diagnostics)."""
        return os.path.join(self._root, category, f"{name}.prompt")

    def clear_cache(self) -> None:
        _CACHE.clear()

    @property
    def root(self) -> str:
        return self._root
