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

"""Tests for GroqBackend — construction, capabilities, context window, tokens."""

import asyncio

import pytest

from encre.backends.groq import GroqBackend


# ===========================================================================
# Construction
# ===========================================================================

class TestGroqBackendConstruction:
    """Test GroqBackend instantiation with various parameters."""

    def test_create_default(self):
        """Default model is llama-3.3-70b-versatile, base URL is api.groq.com."""
        be = GroqBackend(api_key="gsk-test")
        assert be.model == "llama-3.3-70b-versatile"
        assert be.api_key == "gsk-test"
        assert be.api_base_url == "https://api.groq.com/openai/v1"

    def test_create_with_custom_model(self):
        """Explicit model is stored correctly."""
        be = GroqBackend(api_key="gsk-test", model="llama-4-scout")
        assert be.model == "llama-4-scout"

    def test_create_with_gpt_oss_model(self):
        """GPT-OSS 120B model."""
        be = GroqBackend(api_key="gsk-test", model="gpt-oss-120b")
        assert be.model == "gpt-oss-120b"

    def test_create_with_custom_base_url(self):
        """Custom base_url overrides the default."""
        be = GroqBackend(
            api_key="gsk-test",
            base_url="https://custom-groq.example.com/openai/v1",
        )
        assert be.api_base_url == "https://custom-groq.example.com/openai/v1"

    def test_create_with_empty_api_key(self):
        """Empty API key is allowed."""
        be = GroqBackend()
        assert be.api_key == ""
        assert be.model == "llama-3.3-70b-versatile"

    def test_create_with_http_timeout(self):
        """http_timeout kwarg is forwarded."""
        be = GroqBackend(api_key="gsk-test", http_timeout=30.0)
        assert be.http_timeout == 30.0


# ===========================================================================
# Capability checks
# ===========================================================================

class TestGroqBackendCapabilities:
    """Test supports_tool_calling, supports_thinking, supports_prompt_caching."""

    def test_supports_tool_calling(self):
        """Groq models support OpenAI-compatible tool calling."""
        be = GroqBackend(api_key="gsk-test")
        assert be.supports_tool_calling() is True

    def test_supports_tool_calling_different_models(self):
        """Tool calling is True for all Groq-hosted models."""
        models = ["llama-3.3-70b-versatile", "llama-4-scout", "gpt-oss-120b"]
        for m in models:
            be = GroqBackend(api_key="gsk-test", model=m)
            assert be.supports_tool_calling() is True, f"model={m}"

    def test_supports_thinking_returns_bool(self):
        """Groq does not override supports_thinking; returns inherited bool."""
        be = GroqBackend(api_key="gsk-test")
        result = be.supports_thinking()
        assert isinstance(result, bool)

    def test_supports_prompt_caching_returns_bool(self):
        """Prompt caching flag is a boolean (inherited default)."""
        be = GroqBackend(api_key="gsk-test")
        result = be.supports_prompt_caching()
        assert isinstance(result, bool)


# ===========================================================================
# Context window size
# ===========================================================================

class TestGroqBackendContextWindow:
    """Test context_window_size() for Groq models."""

    def test_context_window_size_default(self):
        """All Groq models: 131,072 tokens (128K)."""
        be = GroqBackend(api_key="gsk-test")
        assert be.context_window_size() == 131072

    def test_context_window_size_scout(self):
        """Llama 4 Scout: 131,072 tokens."""
        be = GroqBackend(api_key="gsk-test", model="llama-4-scout")
        assert be.context_window_size() == 131072

    def test_context_window_size_gpt_oss(self):
        """GPT-OSS 120B: 131,072 tokens."""
        be = GroqBackend(api_key="gsk-test", model="gpt-oss-120b")
        assert be.context_window_size() == 131072

    def test_context_window_positive(self):
        """Context window is always positive."""
        be = GroqBackend(api_key="gsk-test")
        assert be.context_window_size() > 0
        assert isinstance(be.context_window_size(), int)


# ===========================================================================
# Token counting and model attribute
# ===========================================================================

class TestGroqBackendTokens:
    """Test count_tokens() and model attribute."""

    def test_count_tokens_returns_int(self):
        """count_tokens returns an integer."""
        be = GroqBackend(api_key="gsk-test")
        result = be.count_tokens("hello world")
        assert isinstance(result, int)

    def test_count_tokens_empty_string(self):
        """Empty string should not crash."""
        be = GroqBackend(api_key="gsk-test")
        result = be.count_tokens("")
        assert isinstance(result, int)

    def test_count_tokens_long_text(self):
        """Long text should not crash."""
        be = GroqBackend(api_key="gsk-test")
        result = be.count_tokens("Groq ultra-low-latency inference. " * 200)
        assert isinstance(result, int)

    def test_model_attribute(self):
        """model attribute matches constructor argument."""
        be = GroqBackend(api_key="gsk-test", model="llama-4-scout")
        assert be.model == "llama-4-scout"
        assert isinstance(be.model, str)


# ===========================================================================
# Request data building
# ===========================================================================

class TestGroqBackendRequestBuilding:
    """Test _build_request_data inherited from OpenAISSEBackend."""

    def test_build_request_includes_max_tokens(self):
        """Request body includes max_tokens."""
        be = GroqBackend(api_key="gsk-test")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
        )
        assert data["max_tokens"] == 1024

    def test_build_request_includes_model(self):
        """Request body includes model name."""
        be = GroqBackend(api_key="gsk-test", model="llama-4-scout")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert data["model"] == "llama-4-scout"

    def test_build_request_stream_flag(self):
        """Streaming is enabled/disabled via the flag."""
        be = GroqBackend(api_key="gsk-test")
        data_stream = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
        assert data_stream["stream"] is True

        data_nostream = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )
        assert data_nostream["stream"] is False


# ===========================================================================
# Lifecycle
# ===========================================================================

class TestGroqBackendLifecycle:
    """Test resource cleanup."""

    def test_aclose_does_not_raise(self):
        """aclose() should work without a prior request (lazy client)."""
        be = GroqBackend(api_key="gsk-test")
        asyncio.run(be.aclose())

    def test_aclose_idempotent(self):
        """aclose() called twice should not raise."""

        async def _double():
            be = GroqBackend(api_key="gsk-test")
            await be.aclose()
            await be.aclose()

        asyncio.run(_double())
