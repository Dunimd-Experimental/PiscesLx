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

"""Shared token estimation using native Rust tokenizer, tiktoken, or fallback."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Native Rust tokenizer (fastest path)
# ---------------------------------------------------------------------------
_HAS_NATIVE: bool = False
try:
    from encre._native import count_tokens as _native_count_tokens
    _HAS_NATIVE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# tiktoken availability (checked once)
# ---------------------------------------------------------------------------
_TIKTOKEN_AVAILABLE: bool = False
_ENCODING_CACHE: dict[str, Any] = {}

try:
    import tiktoken  # noqa: F401
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Model -> tiktoken encoding name mapping
# ---------------------------------------------------------------------------
_MODEL_ENCODING_MAP: dict[str, str] = {
    # GPT-4 family
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-4.1-mini": "o200k_base",
    "gpt-4.1-nano": "o200k_base",
    "gpt-4o-2024-08-06": "o200k_base",
    "gpt-4o-2024-05-13": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-1106-preview": "cl100k_base",
    "gpt-4-0125-preview": "cl100k_base",
    # GPT-3.5 family
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-1106": "cl100k_base",
    "gpt-3.5-turbo-0125": "cl100k_base",
    "gpt-3.5-turbo-instruct": "cl100k_base",
    # o-series
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o4-mini": "o200k_base",
    # Text embedding
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}

# Default encoding for unknown models
_DEFAULT_ENCODING = "cl100k_base"


def _get_encoding(model: str) -> Any:
    """Return a tiktoken Encoding for *model*, cached per encoding name."""
    if not _TIKTOKEN_AVAILABLE:
        return None

    encoding_name = _MODEL_ENCODING_MAP.get(model, _DEFAULT_ENCODING)

    if encoding_name not in _ENCODING_CACHE:
        try:
            import tiktoken
            _ENCODING_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
        except Exception:
            return None

    return _ENCODING_CACHE.get(encoding_name)


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate the number of tokens in *text* for the given *model*.

    Priority order: native Rust tokenizer → tiktoken → char/4 heuristic.
    """
    if not text:
        return 0

    if _HAS_NATIVE:
        try:
            return _native_count_tokens(text)
        except Exception:
            pass

    if _TIKTOKEN_AVAILABLE:
        enc = _get_encoding(model)
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:
                pass

    return len(text) // 4


def count_message_tokens(
    messages: list[dict[str, Any]],
    model: str = "gpt-4o",
) -> int:
    """Estimate the total token count for a list of chat messages.

    Walks through content strings, list-based content blocks, and tool calls,
    summing token estimates for each piece of text.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content, model)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    total += estimate_tokens(text, model)
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                total += estimate_tokens(tc.get("name", ""), model)
                total += estimate_tokens(tc.get("arguments", ""), model)
        # Per-message overhead (role, formatting tokens)
        total += 8
    return total


def estimate_tokens_simple(text: str) -> int:
    """Estimate tokens with the naive char/4 heuristic (same as old behavior).

    Kept for compatibility with callers that don't care about model-specific
    tokenization but want a single consistent import point.
    """
    return estimate_tokens(text, model="gpt-4o")


def is_tiktoken_available() -> bool:
    """Return True if tiktoken is installed and usable."""
    return _TIKTOKEN_AVAILABLE
