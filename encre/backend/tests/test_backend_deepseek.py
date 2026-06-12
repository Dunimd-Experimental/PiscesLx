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

"""Tests for DeepSeekBackend — construction, capabilities, context window, tokens."""

import asyncio

import pytest

from encre.backends.deepseek import DeepSeekBackend


# ===========================================================================
# Construction
# ===========================================================================

class TestDeepSeekBackendConstruction:
    """Test DeepSeekBackend instantiation with various parameters."""

    def test_create_default(self):
        """Default model is deepseek-chat, base URL is api.deepseek.com/v1."""
        be = DeepSeekBackend(api_key="sk-test")
        assert be.model == "deepseek-chat"
        assert be.api_key == "sk-test"
        assert be.api_base_url == "https://api.deepseek.com/v1"

    def test_create_with_custom_model(self):
        """Explicit model is stored correctly."""
        be = DeepSeekBackend(api_key="sk-test", model="deepseek-v4-flash")
        assert be.model == "deepseek-v4-flash"

    def test_create_with_v4_pro_model(self):
        """DeepSeek V4-Pro model."""
        be = DeepSeekBackend(api_key="sk-test", model="deepseek-v4-pro")
        assert be.model == "deepseek-v4-pro"

    def test_create_with_reasoner_model(self):
        """Legacy deepseek-reasoner model."""
        be = DeepSeekBackend(api_key="sk-test", model="deepseek-reasoner")
        assert be.model == "deepseek-reasoner"

    def test_create_with_custom_base_url(self):
        """Custom base_url overrides the default."""
        be = DeepSeekBackend(
            api_key="sk-test",
            base_url="https://custom.deepseek.example.com/v1",
        )
        assert be.api_base_url == "https://custom.deepseek.example.com/v1"

    def test_create_with_empty_api_key(self):
        """Empty API key is allowed."""
        be = DeepSeekBackend()
        assert be.api_key == ""
        assert be.model == "deepseek-chat"

    def test_create_with_http_timeout(self):
        """http_timeout is forwarded to OpenAISSEBackend."""
        be = DeepSeekBackend(api_key="sk-test", http_timeout=90.0)
        assert be.http_timeout == 90.0


# ===========================================================================
# Capability checks
# ===========================================================================

class TestDeepSeekBackendCapabilities:
    """Test supports_tool_calling, supports_thinking, supports_prompt_caching."""

    def test_supports_tool_calling(self):
        """DeepSeek V4 models support tool calling."""
        be = DeepSeekBackend(api_key="sk-test")
        assert be.supports_tool_calling() is True

    def test_supports_tool_calling_different_models(self):
        """Tool calling is True for all DeepSeek V4 models."""
        models = ["deepseek-chat", "deepseek-v4-flash", "deepseek-v4-pro"]
        for m in models:
            be = DeepSeekBackend(api_key="sk-test", model=m)
            assert be.supports_tool_calling() is True, f"model={m}"

    def test_supports_thinking(self):
        """DeepSeek V4 models support reasoning/thinking tokens."""
        be = DeepSeekBackend(api_key="sk-test")
        assert be.supports_thinking() is True

    def test_supports_thinking_different_models(self):
        """Thinking is supported by all V4 models."""
        models = ["deepseek-chat", "deepseek-v4-flash", "deepseek-v4-pro"]
        for m in models:
            be = DeepSeekBackend(api_key="sk-test", model=m)
            assert be.supports_thinking() is True, f"model={m}"

    def test_supports_prompt_caching(self):
        """DeepSeek V4 supports prompt caching (80-92% discount)."""
        be = DeepSeekBackend(api_key="sk-test")
        assert be.supports_prompt_caching() is True


# ===========================================================================
# Context window size
# ===========================================================================

class TestDeepSeekBackendContextWindow:
    """Test context_window_size() for DeepSeek models."""

    def test_context_window_size_default(self):
        """All DeepSeek V4 models: 1,048,576 tokens (1M)."""
        be = DeepSeekBackend(api_key="sk-test")
        assert be.context_window_size() == 1048576

    def test_context_window_size_v4_flash(self):
        """V4-Flash: 1M tokens."""
        be = DeepSeekBackend(api_key="sk-test", model="deepseek-v4-flash")
        assert be.context_window_size() == 1048576

    def test_context_window_size_v4_pro(self):
        """V4-Pro: 1M tokens."""
        be = DeepSeekBackend(api_key="sk-test", model="deepseek-v4-pro")
        assert be.context_window_size() == 1048576

    def test_context_window_size_chat(self):
        """Legacy deepseek-chat: 1M tokens (maps to V4)."""
        be = DeepSeekBackend(api_key="sk-test", model="deepseek-chat")
        assert be.context_window_size() == 1048576

    def test_context_window_positive(self):
        """Context window is always positive."""
        be = DeepSeekBackend(api_key="sk-test")
        assert be.context_window_size() > 0
        assert isinstance(be.context_window_size(), int)


# ===========================================================================
# Token counting and model attribute
# ===========================================================================

class TestDeepSeekBackendTokens:
    """Test count_tokens() and model attribute."""

    def test_count_tokens_returns_int(self):
        """count_tokens() returns an integer."""
        be = DeepSeekBackend(api_key="sk-test")
        result = be.count_tokens("hello world")
        assert isinstance(result, int)

    def test_count_tokens_empty_string(self):
        """Empty string should not crash."""
        be = DeepSeekBackend(api_key="sk-test")
        result = be.count_tokens("")
        assert isinstance(result, int)

    def test_count_tokens_long_text(self):
        """Long text should not crash."""
        be = DeepSeekBackend(api_key="sk-test")
        result = be.count_tokens("Test " * 500)
        assert isinstance(result, int)

    def test_model_attribute(self):
        """model attribute matches constructor argument."""
        be = DeepSeekBackend(api_key="sk-test", model="deepseek-v4-pro")
        assert be.model == "deepseek-v4-pro"
        assert isinstance(be.model, str)


# ===========================================================================
# Request data building
# ===========================================================================

class TestDeepSeekBackendRequestBuilding:
    """Test _build_request_data inherited from OpenAISSEBackend."""

    def test_build_request_includes_max_tokens(self):
        """Request body includes max_tokens parameter."""
        be = DeepSeekBackend(api_key="sk-test")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=512,
        )
        assert data["max_tokens"] == 512

    def test_build_request_includes_model(self):
        """Request body includes the model name."""
        be = DeepSeekBackend(api_key="sk-test", model="deepseek-v4-flash")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert data["model"] == "deepseek-v4-flash"

    def test_build_request_includes_temperature(self):
        """Temperature is included in request data."""
        be = DeepSeekBackend(api_key="sk-test")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7,
        )
        assert data["temperature"] == 0.7


# ===========================================================================
# Lifecycle
# ===========================================================================

class TestDeepSeekBackendLifecycle:
    """Test resource cleanup."""

    def test_aclose_does_not_raise(self):
        """aclose() should work without a prior request (lazy client)."""
        be = DeepSeekBackend(api_key="sk-test")
        asyncio.run(be.aclose())

    def test_aclose_idempotent(self):
        """aclose() called twice should not raise."""

        async def _double():
            be = DeepSeekBackend(api_key="sk-test")
            await be.aclose()
            await be.aclose()

        asyncio.run(_double())
