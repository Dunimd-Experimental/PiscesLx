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

"""Tests for AnthropicBackend — construction, capabilities, context window,
thinking config, prompt caching, and token counting."""

import asyncio

import pytest

from encre.backends.anthropic import AnthropicBackend


# ===========================================================================
# Construction
# ===========================================================================

class TestAnthropicBackendConstruction:
    """Test AnthropicBackend instantiation with various parameters."""

    def test_create_default(self):
        """Default model is claude-sonnet-4-20250514."""
        be = AnthropicBackend(api_key="sk-ant-test")
        assert be.model == "claude-sonnet-4-20250514"
        assert be.api_key == "sk-ant-test"

    def test_create_with_custom_model(self):
        """Explicit model name is stored."""
        be = AnthropicBackend(
            api_key="sk-ant-test",
            model="claude-opus-4-20250514",
        )
        assert be.model == "claude-opus-4-20250514"

    def test_create_with_haiku_model(self):
        """Haiku 4.5 model."""
        be = AnthropicBackend(
            api_key="sk-ant-test",
            model="claude-haiku-4-20250514",
        )
        assert be.model == "claude-haiku-4-20250514"

    def test_create_with_empty_api_key(self):
        """Empty API key is allowed for construction."""
        be = AnthropicBackend()
        assert be.api_key == ""
        assert be.model == "claude-sonnet-4-20250514"

    def test_create_initializes_http_client(self):
        """AnthropicBackend creates its own httpx.AsyncClient."""
        be = AnthropicBackend(api_key="sk-ant-test")
        assert be._client is not None
        assert hasattr(be._client, "base_url")


# ===========================================================================
# Capability checks
# ===========================================================================

class TestAnthropicBackendCapabilities:
    """Test supports_tool_calling, supports_thinking, supports_prompt_caching."""

    def test_supports_tool_calling(self):
        """All Claude models support native tool_use."""
        be = AnthropicBackend(api_key="sk-ant-test")
        assert be.supports_tool_calling() is True

    def test_supports_tool_calling_different_models(self):
        """Tool calling is True for all Claude models."""
        models = [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250514",
        ]
        for m in models:
            be = AnthropicBackend(api_key="sk-ant-test", model=m)
            assert be.supports_tool_calling() is True, f"model={m}"

    def test_supports_thinking_opus(self):
        """Claude Opus supports thinking."""
        be = AnthropicBackend(api_key="sk-ant-test", model="claude-opus-4-20250514")
        assert be.supports_thinking() is True

    def test_supports_thinking_sonnet(self):
        """Claude Sonnet supports thinking."""
        be = AnthropicBackend(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        assert be.supports_thinking() is True

    def test_supports_thinking_haiku(self):
        """Claude Haiku also returns True for supports_thinking (all Claude do)."""
        be = AnthropicBackend(api_key="sk-ant-test", model="claude-haiku-4-20250514")
        assert be.supports_thinking() is True

    def test_supports_prompt_caching(self):
        """All Claude models support prompt caching at 90% discount."""
        be = AnthropicBackend(api_key="sk-ant-test")
        assert be.supports_prompt_caching() is True

    def test_supports_prompt_caching_different_models(self):
        """Prompt caching is True for all Claude models."""
        models = [
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-haiku-4-20250514",
        ]
        for m in models:
            be = AnthropicBackend(api_key="sk-ant-test", model=m)
            assert be.supports_prompt_caching() is True, f"model={m}"


# ===========================================================================
# Context window size
# ===========================================================================

class TestAnthropicBackendContextWindow:
    """Test context_window_size() for Claude models."""

    def test_context_window_size_opus(self):
        """Claude Opus: 200,000 tokens."""
        be = AnthropicBackend(api_key="sk-ant-test", model="claude-opus-4-20250514")
        assert be.context_window_size() == 200000

    def test_context_window_size_sonnet(self):
        """Claude Sonnet: 200,000 tokens."""
        be = AnthropicBackend(api_key="sk-ant-test", model="claude-sonnet-4-20250514")
        assert be.context_window_size() == 200000

    def test_context_window_size_haiku(self):
        """Claude Haiku: 200,000 tokens."""
        be = AnthropicBackend(api_key="sk-ant-test", model="claude-haiku-4-20250514")
        assert be.context_window_size() == 200000

    def test_context_window_positive(self):
        """Context window is always positive."""
        be = AnthropicBackend(api_key="sk-ant-test")
        assert be.context_window_size() > 0
        assert isinstance(be.context_window_size(), int)


# ===========================================================================
# Token counting
# ===========================================================================

class TestAnthropicBackendTokens:
    """Test count_tokens() for Anthropic backend."""

    def test_count_tokens_returns_int(self):
        """count_tokens returns an integer."""
        be = AnthropicBackend(api_key="sk-ant-test")
        result = be.count_tokens("hello world")
        assert isinstance(result, int)

    def test_count_tokens_empty_string(self):
        """Empty string should not crash."""
        be = AnthropicBackend(api_key="sk-ant-test")
        result = be.count_tokens("")
        assert isinstance(result, int)

    def test_count_tokens_long_text(self):
        """Long text should not crash."""
        be = AnthropicBackend(api_key="sk-ant-test")
        result = be.count_tokens("Testing token counting. " * 200)
        assert isinstance(result, int)


# ===========================================================================
# Prompt caching — _apply_prompt_caching static method
# ===========================================================================

class TestAnthropicBackendPromptCaching:
    """Test the _apply_prompt_caching static method for cache_control injection."""

    def test_caches_system_message(self):
        """System messages receive cache_control breakpoints."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = AnthropicBackend._apply_prompt_caching(messages)
        assert len(result) == 2
        # System message (index 0) should have cache_control.
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert "cache_control" in sys_content[-1]

    def test_caches_last_user_message(self):
        """The last user message receives a cache_control breakpoint."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Second question"},
        ]
        result = AnthropicBackend._apply_prompt_caching(messages)
        assert len(result) == 2
        # First user message should NOT be cached (not last).
        content0 = result[0]["content"]
        if isinstance(content0, list):
            has_cache_0 = any("cache_control" in str(b) for b in content0)
            assert not has_cache_0
        # Last user message SHOULD be cached.
        content1 = result[1]["content"]
        assert isinstance(content1, list)
        assert "cache_control" in content1[-1]

    def test_caches_system_and_last_user(self):
        """Both system messages and the last user message are cached."""
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        result = AnthropicBackend._apply_prompt_caching(messages)
        # System (idx 0): cached.
        sys_content = result[0]["content"]
        assert isinstance(sys_content, list)
        assert "cache_control" in sys_content[-1]
        # Assistant (idx 2): NOT cached.
        asst_content = result[2]["content"]
        if isinstance(asst_content, list):
            has_cache = any("cache_control" in str(b) for b in asst_content)
            assert not has_cache
        elif isinstance(asst_content, str):
            pass  # string content won't have cache_control
        # Last user (idx 3): cached.
        last_user_content = result[3]["content"]
        assert isinstance(last_user_content, list)
        assert "cache_control" in last_user_content[-1]

    def test_handles_content_list_with_images(self):
        """Cache control skips non-cacheable blocks (images) — caches last text block."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": "abc123", "media_type": "image/png"}},
                    {"type": "text", "text": "Describe this image."},
                ],
            },
        ]
        result = AnthropicBackend._apply_prompt_caching(messages)
        blocks = result[0]["content"]
        # The image block (index 0) should NOT have cache_control.
        assert "cache_control" not in blocks[0]
        # The text block (index 1, last cacheable) SHOULD have cache_control.
        assert "cache_control" in blocks[1]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    def test_does_not_cache_assistant_messages(self):
        """Assistant messages are not annotated with cache_control."""
        messages = [
            {"role": "assistant", "content": "I am Claude."},
        ]
        result = AnthropicBackend._apply_prompt_caching(messages)
        # Assistant message content stays as a string (not wrapped in list).
        assert isinstance(result[0]["content"], str)

    def test_no_messages_returns_empty(self):
        """Empty message list returns empty list."""
        result = AnthropicBackend._apply_prompt_caching([])
        assert result == []


# ===========================================================================
# Chat method signature / parameter inspection
# ===========================================================================

class TestAnthropicBackendChatSignature:
    """Test that the chat() method accepts expected parameters."""

    def test_chat_is_async_generator(self):
        """Chat method should be an async generator function."""
        import inspect
        assert inspect.iscoroutinefunction(AnthropicBackend.chat) or \
            inspect.isasyncgenfunction(AnthropicBackend.chat)

    def test_chat_accepts_enable_caching(self):
        """Chat method signature includes enable_caching parameter."""
        import inspect
        sig = inspect.signature(AnthropicBackend.chat)
        params = sig.parameters
        assert "enable_caching" in params

    def test_chat_accepts_max_tokens(self):
        """Chat method signature includes max_tokens parameter."""
        import inspect
        sig = inspect.signature(AnthropicBackend.chat)
        params = sig.parameters
        assert "max_tokens" in params
        assert params["max_tokens"].default == 4096

    def test_chat_accepts_tools(self):
        """Chat method accepts optional tool definitions."""
        import inspect
        sig = inspect.signature(AnthropicBackend.chat)
        assert "tools" in sig.parameters


# ===========================================================================
# Lifecycle
# ===========================================================================

class TestAnthropicBackendLifecycle:
    """Test resource cleanup."""

    def test_aclose_closes_client(self):
        """aclose() closes the httpx client."""

        async def _close():
            be = AnthropicBackend(api_key="sk-ant-test")
            assert be._client is not None
            await be.aclose()
            # After close, the client should be closed.
            assert be._client.is_closed

        asyncio.run(_close())

    def test_aclose_silent_on_error(self):
        """aclose() does not raise on repeated calls."""

        async def _double_close():
            be = AnthropicBackend(api_key="sk-ant-test")
            await be.aclose()
            await be.aclose()  # Idempotent.

        asyncio.run(_double_close())
