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

"""Tests for OpenAIBackend — construction, capabilities, context window, tokens."""

import asyncio

import pytest

from encre.backends.openai import OpenAIBackend


# ===========================================================================
# Construction
# ===========================================================================

class TestOpenAIBackendConstruction:
    """Test OpenAIBackend instantiation with various parameter combinations."""

    def test_create_default(self):
        """Default model is gpt-4.1, default base_url is api.openai.com/v1."""
        be = OpenAIBackend(api_key="sk-test")
        assert be.model == "gpt-4.1"
        assert be.api_key == "sk-test"
        assert be.api_base_url == "https://api.openai.com/v1"

    def test_create_with_custom_model(self):
        """Explicit model name is stored correctly."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-4.1-mini")
        assert be.model == "gpt-4.1-mini"

    def test_create_with_base_url(self):
        """Custom base_url overrides the default OpenAI endpoint."""
        be = OpenAIBackend(
            api_key="sk-test",
            base_url="https://custom.openai.example.com/v1",
        )
        assert be.api_base_url == "https://custom.openai.example.com/v1"

    def test_create_with_nano_model(self):
        """GPT-4.1 Nano variant."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-4.1-nano")
        assert be.model == "gpt-4.1-nano"

    def test_create_with_o3_model(self):
        """o3 reasoning model."""
        be = OpenAIBackend(api_key="sk-test", model="o3")
        assert be.model == "o3"

    def test_create_with_gpt5_5_model(self):
        """GPT-5.5 top-tier model."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-5.5")
        assert be.model == "gpt-5.5"

    def test_create_with_empty_api_key(self):
        """Empty API key is allowed (caller may use env var)."""
        be = OpenAIBackend()
        assert be.api_key == ""
        assert be.model == "gpt-4.1"

    def test_create_passes_http_timeout(self):
        """http_timeout kwarg is forwarded to the parent SSE backend."""
        be = OpenAIBackend(api_key="sk-test", http_timeout=60.0)
        assert be.http_timeout == 60.0


# ===========================================================================
# Capability checks
# ===========================================================================

class TestOpenAIBackendCapabilities:
    """Test supports_tool_calling, supports_thinking, supports_prompt_caching."""

    def test_supports_tool_calling(self):
        """All OpenAI models support tool calling."""
        be = OpenAIBackend(api_key="sk-test")
        assert be.supports_tool_calling() is True

    def test_supports_tool_calling_different_models(self):
        """Tool calling is True regardless of model choice."""
        models = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5.5", "o3"]
        for m in models:
            be = OpenAIBackend(api_key="sk-test", model=m)
            assert be.supports_tool_calling() is True, f"model={m}"

    def test_supports_thinking_gpt4_1(self):
        """GPT-4.1 does NOT emit thinking tokens."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-4.1")
        assert be.supports_thinking() is False

    def test_supports_thinking_gpt4_1_mini(self):
        """GPT-4.1 Mini does NOT emit thinking tokens."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-4.1-mini")
        assert be.supports_thinking() is False

    def test_supports_thinking_gpt5(self):
        """GPT-5.x models do NOT emit thinking tokens."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-5.2")
        assert be.supports_thinking() is False

    def test_supports_thinking_o3(self):
        """o3 IS a reasoning model — emits thinking tokens."""
        be = OpenAIBackend(api_key="sk-test", model="o3")
        assert be.supports_thinking() is True

    def test_supports_thinking_o4_mini(self):
        """o4-mini IS a reasoning model — emits thinking tokens."""
        be = OpenAIBackend(api_key="sk-test", model="o4-mini")
        assert be.supports_thinking() is True

    def test_supports_prompt_caching_returns_bool(self):
        """Prompt caching flag is a boolean."""
        be = OpenAIBackend(api_key="sk-test")
        result = be.supports_prompt_caching()
        assert isinstance(result, bool)


# ===========================================================================
# Context window size
# ===========================================================================

class TestOpenAIBackendContextWindow:
    """Test context_window_size() for every model variant."""

    def test_context_gpt4_1(self):
        """GPT-4.1: 1,048,576 tokens."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-4.1")
        assert be.context_window_size() == 1048576

    def test_context_gpt4_1_mini(self):
        """GPT-4.1 Mini: 1,048,576 tokens."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-4.1-mini")
        assert be.context_window_size() == 1048576

    def test_context_gpt4_1_nano(self):
        """GPT-4.1 Nano: 1,048,576 tokens."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-4.1-nano")
        assert be.context_window_size() == 1048576

    def test_context_o3(self):
        """o3: 200,000 tokens."""
        be = OpenAIBackend(api_key="sk-test", model="o3")
        assert be.context_window_size() == 200000

    def test_context_o4_mini(self):
        """o4-mini: 1,048,576 tokens ('mini' substring matches first)."""
        be = OpenAIBackend(api_key="sk-test", model="o4-mini")
        assert be.context_window_size() == 1048576

    def test_context_gpt5_2(self):
        """GPT-5.2: 128,000 tokens (default fallback)."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-5.2")
        assert be.context_window_size() == 128000

    def test_context_gpt5_4(self):
        """GPT-5.4: 400,000 tokens."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-5.4")
        assert be.context_window_size() == 400000

    def test_context_gpt5_5(self):
        """GPT-5.5: 1,048,576 tokens."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-5.5")
        assert be.context_window_size() == 1048576

    def test_context_always_positive(self):
        """Context window is always a positive integer."""
        models = ["gpt-4.1", "gpt-4.1-mini", "gpt-5.2", "o3", "o4-mini"]
        for m in models:
            be = OpenAIBackend(api_key="sk-test", model=m)
            assert be.context_window_size() > 0, f"model={m}"
            assert isinstance(be.context_window_size(), int), f"model={m}"


# ===========================================================================
# Token counting and model attribute
# ===========================================================================

class TestOpenAIBackendTokens:
    """Test count_tokens and model attribute access."""

    def test_count_tokens_returns_int(self):
        """count_tokens() returns an integer (may be -1 without tiktoken)."""
        be = OpenAIBackend(api_key="sk-test")
        result = be.count_tokens("hello world")
        assert isinstance(result, int)

    def test_count_tokens_empty_string(self):
        """Empty string should not crash token counting."""
        be = OpenAIBackend(api_key="sk-test")
        result = be.count_tokens("")
        assert isinstance(result, int)

    def test_count_tokens_long_text(self):
        """Long text should not crash token counting."""
        be = OpenAIBackend(api_key="sk-test")
        result = be.count_tokens("The quick brown fox jumps over the lazy dog. " * 100)
        assert isinstance(result, int)

    def test_model_attribute_access(self):
        """model attribute reflects the constructor argument."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-4.1-mini")
        assert be.model == "gpt-4.1-mini"
        assert isinstance(be.model, str)

    def test_model_default(self):
        """Default model is gpt-4.1."""
        be = OpenAIBackend(api_key="sk-test")
        assert be.model == "gpt-4.1"


# ===========================================================================
# Request data / token parameter construction
# ===========================================================================

class TestOpenAIBackendRequestBuilding:
    """Test _build_request_data and token parameter handling."""

    def test_build_request_includes_max_tokens(self):
        """_build_request_data propagates max_tokens to the request body."""
        be = OpenAIBackend(api_key="sk-test")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=2048,
        )
        assert "max_tokens" in data
        assert data["max_tokens"] == 2048

    def test_build_request_default_max_tokens(self):
        """Default max_tokens is 4096."""
        be = OpenAIBackend(api_key="sk-test")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert data["max_tokens"] == 4096

    def test_build_request_includes_model(self):
        """Request body includes the model name."""
        be = OpenAIBackend(api_key="sk-test", model="gpt-4.1-mini")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert data["model"] == "gpt-4.1-mini"

    def test_build_request_includes_messages(self):
        """Request body includes conversation messages."""
        be = OpenAIBackend(api_key="sk-test")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        data = be._build_request_data(messages=messages)
        assert data["messages"] == messages

    def test_build_request_stream_default(self):
        """Streaming is enabled by default."""
        be = OpenAIBackend(api_key="sk-test")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert data["stream"] is True

    def test_build_request_non_stream(self):
        """Non-streaming mode can be requested."""
        be = OpenAIBackend(api_key="sk-test")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )
        assert data["stream"] is False

    def test_build_request_with_tools(self):
        """Tool definitions are included when provided."""
        be = OpenAIBackend(api_key="sk-test")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        data = be._build_request_data(
            messages=[{"role": "user", "content": "search for cats"}],
            tools=tools,
        )
        assert "tools" in data
        assert data["tools"] == tools
        assert "tool_choice" in data

    def test_build_request_without_tools(self):
        """Tools and tool_choice are omitted when no tools provided."""
        be = OpenAIBackend(api_key="sk-test")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert "tools" not in data
        assert "tool_choice" not in data


# ===========================================================================
# Lifecycle
# ===========================================================================

class TestOpenAIBackendLifecycle:
    """Test resource cleanup and lifecycle."""

    def test_aclose_does_not_raise(self):
        """aclose() should work even without a prior request (lazy client)."""
        be = OpenAIBackend(api_key="sk-test")
        # Should not raise — _client may be None (lazy init).
        asyncio.run(be.aclose())

    def test_aclose_idempotent(self):
        """Calling aclose() twice should not raise."""
        be = OpenAIBackend(api_key="sk-test")

        async def _double_close():
            await be.aclose()
            await be.aclose()

        asyncio.run(_double_close())
