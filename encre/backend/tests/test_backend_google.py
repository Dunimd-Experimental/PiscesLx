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

"""Tests for GoogleBackend — construction, capabilities, context window,
grounding, message conversion, and token counting."""

import asyncio

import pytest

from encre.backends.google import GoogleBackend


# ===========================================================================
# Construction
# ===========================================================================

class TestGoogleBackendConstruction:
    """Test GoogleBackend instantiation with various parameters."""

    def test_create_default(self):
        """Default model is gemini-2.5-pro, default base URL is Google AI Studio."""
        be = GoogleBackend(api_key="fake-key")
        assert be.model == "gemini-2.5-pro"
        assert be.api_key == "fake-key"
        assert "generativelanguage.googleapis.com" in be.base_url

    def test_create_with_flash_model(self):
        """Gemini 2.5 Flash model."""
        be = GoogleBackend(api_key="fake-key", model="gemini-2.5-flash")
        assert be.model == "gemini-2.5-flash"

    def test_create_with_custom_base_url(self):
        """Custom base_url overrides the default Google endpoint."""
        be = GoogleBackend(
            api_key="fake-key",
            base_url="https://custom-google.example.com/v1beta",
        )
        assert be.base_url == "https://custom-google.example.com/v1beta"

    def test_create_with_grounding_enabled(self):
        """Google Search grounding can be enabled at construction."""
        be = GoogleBackend(api_key="fake-key", enable_grounding=True)
        assert be.enable_grounding is True

    def test_create_grounding_disabled_by_default(self):
        """Grounding is disabled by default."""
        be = GoogleBackend(api_key="fake-key")
        assert be.enable_grounding is False

    def test_create_with_empty_api_key(self):
        """Empty API key is allowed."""
        be = GoogleBackend()
        assert be.api_key == ""
        assert be.model == "gemini-2.5-pro"

    def test_create_initializes_http_client(self):
        """GoogleBackend creates its own httpx.AsyncClient."""
        be = GoogleBackend(api_key="fake-key")
        assert be._client is not None


# ===========================================================================
# Capability checks
# ===========================================================================

class TestGoogleBackendCapabilities:
    """Test supports_tool_calling, supports_thinking, supports_grounding."""

    def test_supports_tool_calling(self):
        """Gemini models support function calling."""
        be = GoogleBackend(api_key="fake-key")
        assert be.supports_tool_calling() is True

    def test_supports_tool_calling_different_models(self):
        """Tool calling is True for all Gemini 2.5 models."""
        models = ["gemini-2.5-pro", "gemini-2.5-flash"]
        for m in models:
            be = GoogleBackend(api_key="fake-key", model=m)
            assert be.supports_tool_calling() is True, f"model={m}"

    def test_supports_thinking(self):
        """Gemini 2.5 models support thinking/reasoning."""
        be = GoogleBackend(api_key="fake-key")
        assert be.supports_thinking() is True

    def test_supports_grounding(self):
        """Gemini models support Google Search grounding."""
        be = GoogleBackend(api_key="fake-key")
        assert be.supports_grounding() is True

    def test_supports_prompt_caching_returns_bool(self):
        """Prompt caching flag is a boolean (inherited default)."""
        be = GoogleBackend(api_key="fake-key")
        result = be.supports_prompt_caching()
        assert isinstance(result, bool)


# ===========================================================================
# Context window size
# ===========================================================================

class TestGoogleBackendContextWindow:
    """Test context_window_size() for Gemini models."""

    def test_context_window_size_default(self):
        """Gemini 2.5 Pro: 1,048,576 tokens (1M)."""
        be = GoogleBackend(api_key="fake-key")
        assert be.context_window_size() == 1048576

    def test_context_window_size_flash(self):
        """Gemini 2.5 Flash: 1M tokens."""
        be = GoogleBackend(api_key="fake-key", model="gemini-2.5-flash")
        assert be.context_window_size() == 1048576

    def test_context_window_positive(self):
        """Context window is always positive."""
        be = GoogleBackend(api_key="fake-key")
        assert be.context_window_size() > 0
        assert isinstance(be.context_window_size(), int)


# ===========================================================================
# Token counting and model attribute
# ===========================================================================

class TestGoogleBackendTokens:
    """Test count_tokens() and model attribute."""

    def test_count_tokens_returns_int(self):
        """count_tokens returns an integer."""
        be = GoogleBackend(api_key="fake-key")
        result = be.count_tokens("hello world")
        assert isinstance(result, int)

    def test_count_tokens_empty_string(self):
        """Empty string should not crash."""
        be = GoogleBackend(api_key="fake-key")
        result = be.count_tokens("")
        assert isinstance(result, int)

    def test_count_tokens_long_text(self):
        """Long text should not crash."""
        be = GoogleBackend(api_key="fake-key")
        result = be.count_tokens("Google Gemini token counting. " * 300)
        assert isinstance(result, int)

    def test_model_attribute(self):
        """model attribute matches constructor argument."""
        be = GoogleBackend(api_key="fake-key", model="gemini-2.5-flash")
        assert be.model == "gemini-2.5-flash"
        assert isinstance(be.model, str)


# ===========================================================================
# Message conversion (internal protocol mapping)
# ===========================================================================

class TestGoogleBackendMessageConversion:
    """Test _convert_messages and _convert_tools for OpenAI-to-Google format."""

    def test_convert_simple_user_message(self):
        """Simple user message is converted to Google format."""
        be = GoogleBackend(api_key="fake-key")
        messages = [{"role": "user", "content": "Hello, Gemini!"}]
        contents, system_instruction = be._convert_messages(messages)
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert len(contents[0]["parts"]) == 1
        assert contents[0]["parts"][0]["text"] == "Hello, Gemini!"

    def test_convert_system_message_to_instruction(self):
        """System messages are extracted into systemInstruction, not contents."""
        be = GoogleBackend(api_key="fake-key")
        messages = [
            {"role": "system", "content": "You are a helpful bot."},
            {"role": "user", "content": "Help me."},
        ]
        contents, system_instruction = be._convert_messages(messages)
        assert len(contents) == 1  # Only the user message
        assert system_instruction is not None
        assert system_instruction["parts"][0]["text"] == "You are a helpful bot."

    def test_convert_assistant_message(self):
        """Assistant role is mapped to 'model' role."""
        be = GoogleBackend(api_key="fake-key")
        messages = [{"role": "assistant", "content": "I can help with that."}]
        contents, _ = be._convert_messages(messages)
        assert len(contents) == 1
        assert contents[0]["role"] == "model"

    def test_convert_empty_messages(self):
        """Empty message list produces empty contents."""
        be = GoogleBackend(api_key="fake-key")
        contents, system_instruction = be._convert_messages([])
        assert contents == []
        assert system_instruction is None

    def test_convert_tools_openai_to_google_format(self):
        """OpenAI tool definitions are converted to Google functionDeclarations."""
        be = GoogleBackend(api_key="fake-key")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
        result = be._convert_tools(tools)
        assert len(result) == 1
        assert "functionDeclarations" in result[0]
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert decls[0]["description"] == "Get current weather"

    def test_convert_tools_skips_non_function_types(self):
        """Non-function tool types are skipped during conversion."""
        be = GoogleBackend(api_key="fake-key")
        tools = [
            {"type": "code_interpreter"},
            {"type": "function", "function": {"name": "calc"}},
        ]
        result = be._convert_tools(tools)
        assert len(result) > 0
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1  # Only the function tool
        assert decls[0]["name"] == "calc"

    def test_convert_tools_empty_list(self):
        """Empty tool list produces empty declaration list."""
        be = GoogleBackend(api_key="fake-key")
        result = be._convert_tools([])
        assert len(result) == 1
        assert result[0]["functionDeclarations"] == []

    def test_convert_tools_without_description(self):
        """Tool without description is still converted."""
        be = GoogleBackend(api_key="fake-key")
        tools = [
            {
                "type": "function",
                "function": {"name": "simple_tool"},
            }
        ]
        result = be._convert_tools(tools)
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "simple_tool"
        assert "description" not in decls[0]


# ===========================================================================
# Finish reason mapping
# ===========================================================================

class TestGoogleBackendFinishReason:
    """Test _map_finish_reason for Google-to-unified mapping."""

    def test_map_stop(self):
        be = GoogleBackend(api_key="fake-key")
        assert be._map_finish_reason("STOP") == "stop"

    def test_map_max_tokens(self):
        be = GoogleBackend(api_key="fake-key")
        assert be._map_finish_reason("MAX_TOKENS") == "max_tokens"

    def test_map_safety(self):
        be = GoogleBackend(api_key="fake-key")
        assert be._map_finish_reason("SAFETY") == "error"

    def test_map_recitation(self):
        be = GoogleBackend(api_key="fake-key")
        assert be._map_finish_reason("RECITATION") == "error"

    def test_map_unknown_fallback(self):
        be = GoogleBackend(api_key="fake-key")
        assert be._map_finish_reason("UNKNOWN_REASON") == "stop"


# ===========================================================================
# Lifecycle
# ===========================================================================

class TestGoogleBackendLifecycle:
    """Test resource cleanup."""

    def test_aclose_closes_client(self):
        """aclose() closes the httpx client."""

        async def _close():
            be = GoogleBackend(api_key="fake-key")
            assert be._client is not None
            await be.aclose()
            assert be._client.is_closed

        asyncio.run(_close())

    def test_aclose_idempotent(self):
        """aclose() called twice should not raise."""

        async def _double_close():
            be = GoogleBackend(api_key="fake-key")
            await be.aclose()
            await be.aclose()

        asyncio.run(_double_close())
