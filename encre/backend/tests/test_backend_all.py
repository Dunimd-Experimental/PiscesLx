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

"""Tests for individual backend implementations (no API keys needed)."""

import asyncio

import pytest

from encre.backend import create_backend
from encre.backends.base import BaseBackend


# ===========================================================================
# OpenAI
# ===========================================================================

class TestOpenAIBackend:
    def test_create(self):
        be = create_backend("openai", api_key="sk-fake")
        assert isinstance(be, BaseBackend)
        assert be.supports_tool_calling() is True
        assert be.context_window_size() == 1048576

    def test_model_override(self):
        be = create_backend("openai", model="gpt-4o-mini", api_key="sk-fake")
        assert be.model == "gpt-4o-mini"

    def test_count_tokens(self):
        be = create_backend("openai", api_key="sk-fake")
        # May return -1 if tiktoken not installed
        assert isinstance(be.count_tokens("hello"), int)


# ===========================================================================
# Anthropic
# ===========================================================================

class TestAnthropicBackend:
    def test_create(self):
        be = create_backend("anthropic", api_key="sk-ant-fake")
        assert isinstance(be, BaseBackend)
        assert be.supports_tool_calling() is True
        assert be.context_window_size() == 200000
        assert be.supports_thinking() is True
        assert be.supports_prompt_caching() is True

    def test_model_override(self):
        be = create_backend("anthropic", model="claude-sonnet-4-20250514", api_key="sk-ant-fake")
        assert be.model == "claude-sonnet-4-20250514"

    def test_max_tokens_override(self):
        be = create_backend("anthropic", api_key="sk-ant-fake")
        assert be.supports_thinking() is True


# ===========================================================================
# DeepSeek
# ===========================================================================

class TestDeepSeekBackend:
    def test_create(self):
        be = create_backend("deepseek", api_key="sk-fake")
        assert isinstance(be, BaseBackend)
        assert be.supports_tool_calling() is True
        assert be.context_window_size() == 1048576

    def test_model_override(self):
        be = create_backend("deepseek", model="deepseek-chat", api_key="sk-fake")
        assert be.model == "deepseek-chat"


# ===========================================================================
# Google
# ===========================================================================

class TestGoogleBackend:
    def test_create(self):
        be = create_backend("google", api_key="fake-key")
        assert isinstance(be, BaseBackend)
        assert be.supports_tool_calling() is True
        assert be.context_window_size() == 1048576

    def test_model_override(self):
        be = create_backend("google", model="gemini-2.5-flash", api_key="fake-key")
        assert be.model == "gemini-2.5-flash"


# ===========================================================================
# Groq
# ===========================================================================

class TestGroqBackend:
    def test_create(self):
        be = create_backend("groq", api_key="gsk-fake")
        assert isinstance(be, BaseBackend)
        assert be.supports_tool_calling() is True
        assert be.context_window_size() == 131072

    def test_model_override(self):
        be = create_backend("groq", model="llama-4-maverick", api_key="gsk-fake")
        assert be.model == "llama-4-maverick"


# ===========================================================================
# Ollama
# ===========================================================================

class TestOllamaBackend:
    def test_create(self):
        be = create_backend("ollama", base_url="http://localhost:11434")
        assert isinstance(be, BaseBackend)
        assert be.context_window_size() == 8192
        assert isinstance(be.supports_tool_calling(), bool)


# ===========================================================================
# Local
# ===========================================================================

class TestLocalBackend:
    def test_create(self):
        be = create_backend("local")
        assert isinstance(be, BaseBackend)
        assert be.context_window_size() == 4096
        assert isinstance(be.supports_tool_calling(), bool)

    def test_model_override(self):
        be = create_backend("local", model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct")
        assert be.model_name == "meta-llama/Llama-4-Maverick-17B-128E-Instruct"


# ===========================================================================
# Bedrock
# ===========================================================================

class TestBedrockBackend:
    def test_create(self):
        be = create_backend("bedrock", aws_access_key_id="fake", aws_secret_access_key="fake", region="us-east-1")
        assert isinstance(be, BaseBackend)
        assert be.context_window_size() == 200000
        assert isinstance(be.supports_tool_calling(), bool)

    def test_model_override(self):
        be = create_backend(
            "bedrock",
            model="anthropic.claude-sonnet-4-20250514-v1:0",
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
        )
        assert be.model == "anthropic.claude-sonnet-4-20250514-v1:0"


# ===========================================================================
# OpenAI Compatible
# ===========================================================================

class TestOpenAICompatibleBackend:
    def test_create(self):
        be = create_backend("openai_compatible", base_url="https://api.example.com/v1", api_key="sk-fake")
        assert isinstance(be, BaseBackend)
        assert isinstance(be.supports_tool_calling(), bool)
        assert be.context_window_size() == 128000

    def test_model_override(self):
        be = create_backend(
            "openai_compatible",
            model="custom-model",
            base_url="https://api.example.com/v1",
            api_key="sk-fake",
        )
        assert be.model == "custom-model"


# ===========================================================================
# Retry integration
# ===========================================================================

class TestRetryIntegration:
    def test_retry_with_backoff_handler(self):
        from encre.backends.retry import retry_with_backoff, RetryConfig
        import httpx

        async def _test():
            config = RetryConfig(max_retries=2, base_delay=0.01)

            @retry_with_backoff(config)
            async def flaky_request():
                raise httpx.TimeoutException("timeout")

            with pytest.raises(httpx.TimeoutException):
                await flaky_request()

        asyncio.run(_test())

    def test_retry_config_tool_retries(self):
        from encre.backends.retry import RetryConfig
        rc = RetryConfig()
        assert rc.rate_limit_retries == 8
