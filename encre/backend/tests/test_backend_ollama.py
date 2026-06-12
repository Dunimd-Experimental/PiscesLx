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

"""Tests for OllamaBackend — construction, capabilities, context window, tokens."""

import asyncio

import pytest

from encre.backends.ollama import OllamaBackend


# ===========================================================================
# Construction
# ===========================================================================

class TestOllamaBackendConstruction:
    """Test OllamaBackend instantiation with various parameters."""

    def test_create_default(self):
        """Default model is llama3.2, base URL is localhost:11434/v1."""
        be = OllamaBackend()
        assert be.model == "llama3.2"
        assert be.api_base_url == "http://localhost:11434/v1"
        assert be.api_key == ""

    def test_create_with_base_url(self):
        """Custom base_url for a remote Ollama instance."""
        be = OllamaBackend(base_url="http://192.168.1.100:11434/v1")
        assert be.api_base_url == "http://192.168.1.100:11434/v1"

    def test_create_with_model(self):
        """Custom model name (must exist in the local Ollama library)."""
        be = OllamaBackend(model="llama3.1")
        assert be.model == "llama3.1"

    def test_create_with_qwen_model(self):
        """Qwen 2.5 model name."""
        be = OllamaBackend(model="qwen2.5")
        assert be.model == "qwen2.5"

    def test_create_with_mistral_model(self):
        """Mistral model name."""
        be = OllamaBackend(model="mistral")
        assert be.model == "mistral"

    def test_create_no_api_key_required(self):
        """Ollama runs locally — no API key needed."""
        be = OllamaBackend()
        assert be.api_key == ""

    def test_create_with_http_timeout(self):
        """http_timeout kwarg is forwarded."""
        be = OllamaBackend(http_timeout=60.0)
        assert be.http_timeout == 60.0

    def test_create_full_custom(self):
        """Full custom configuration: remote host, specific model."""
        be = OllamaBackend(
            base_url="http://gpu-server:11434/v1",
            model="deepseek-r1:8b",
            http_timeout=300.0,
        )
        assert be.api_base_url == "http://gpu-server:11434/v1"
        assert be.model == "deepseek-r1:8b"
        assert be.http_timeout == 300.0


# ===========================================================================
# Capability checks
# ===========================================================================

class TestOllamaBackendCapabilities:
    """Test supports_tool_calling, supports_thinking, supports_prompt_caching."""

    def test_supports_tool_calling_returns_bool(self):
        """Tool calling depends on the model — returns a boolean."""
        be = OllamaBackend()
        result = be.supports_tool_calling()
        assert isinstance(result, bool)

    def test_supports_tool_calling_default_true(self):
        """Ollama inherits True from OpenAISSEBackend (model-dependent in practice)."""
        be = OllamaBackend()
        assert be.supports_tool_calling() is True

    def test_supports_thinking_returns_bool(self):
        """Thinking support flag is a boolean."""
        be = OllamaBackend()
        result = be.supports_thinking()
        assert isinstance(result, bool)

    def test_supports_prompt_caching_returns_bool(self):
        """Prompt caching is not supported locally — returns bool."""
        be = OllamaBackend()
        result = be.supports_prompt_caching()
        assert isinstance(result, bool)


# ===========================================================================
# Context window size
# ===========================================================================

class TestOllamaBackendContextWindow:
    """Test context_window_size() for Ollama models."""

    def test_context_window_size_default(self):
        """Returns conservative 8192 as safe default for local models."""
        be = OllamaBackend()
        assert be.context_window_size() == 8192

    def test_context_window_size_different_models(self):
        """Context window is 8192 regardless of model (conservative default)."""
        models = ["llama3.2", "llama3.1", "qwen2.5", "mistral", "deepseek-r1:8b"]
        for m in models:
            be = OllamaBackend(model=m)
            assert be.context_window_size() == 8192, f"model={m}"

    def test_context_window_positive(self):
        """Context window is always positive."""
        be = OllamaBackend()
        assert be.context_window_size() > 0
        assert isinstance(be.context_window_size(), int)


# ===========================================================================
# Token counting and model attribute
# ===========================================================================

class TestOllamaBackendTokens:
    """Test count_tokens() and model attribute."""

    def test_count_tokens_returns_int(self):
        """count_tokens returns an integer."""
        be = OllamaBackend()
        result = be.count_tokens("hello world")
        assert isinstance(result, int)

    def test_count_tokens_empty_string(self):
        """Empty string should not crash."""
        be = OllamaBackend()
        result = be.count_tokens("")
        assert isinstance(result, int)

    def test_count_tokens_long_text(self):
        """Long text should not crash."""
        be = OllamaBackend()
        result = be.count_tokens("Ollama local model token counting. " * 200)
        assert isinstance(result, int)

    def test_model_attribute(self):
        """model attribute matches constructor argument."""
        be = OllamaBackend(model="mistral")
        assert be.model == "mistral"
        assert isinstance(be.model, str)

    def test_model_default(self):
        """Default model is llama3.2."""
        be = OllamaBackend()
        assert be.model == "llama3.2"


# ===========================================================================
# Request data building
# ===========================================================================

class TestOllamaBackendRequestBuilding:
    """Test _build_request_data inherited from OpenAISSEBackend."""

    def test_build_request_includes_max_tokens(self):
        """Request body includes max_tokens."""
        be = OllamaBackend()
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=512,
        )
        assert data["max_tokens"] == 512

    def test_build_request_includes_model(self):
        """Request body includes the model name."""
        be = OllamaBackend(model="qwen2.5")
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert data["model"] == "qwen2.5"

    def test_build_request_with_tools(self):
        """Tool definitions are included when provided."""
        be = OllamaBackend()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        data = be._build_request_data(
            messages=[{"role": "user", "content": "read the file"}],
            tools=tools,
        )
        assert "tools" in data
        assert data["tools"] == tools
        assert "tool_choice" in data

    def test_build_request_without_tools(self):
        """tools and tool_choice omitted when no tools provided."""
        be = OllamaBackend()
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert "tools" not in data
        assert "tool_choice" not in data

    def test_build_request_default_stream(self):
        """Streaming is enabled by default."""
        be = OllamaBackend()
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert data["stream"] is True

    def test_build_request_default_temperature(self):
        """Default temperature is 0.0."""
        be = OllamaBackend()
        data = be._build_request_data(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert data["temperature"] == 0.0


# ===========================================================================
# Lifecycle
# ===========================================================================

class TestOllamaBackendLifecycle:
    """Test resource cleanup."""

    def test_aclose_does_not_raise(self):
        """aclose() should work without a prior request (lazy client)."""
        be = OllamaBackend()
        asyncio.run(be.aclose())

    def test_aclose_idempotent(self):
        """aclose() called twice should not raise."""

        async def _double():
            be = OllamaBackend()
            await be.aclose()
            await be.aclose()

        asyncio.run(_double())
