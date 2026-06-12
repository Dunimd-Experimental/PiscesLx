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

"""Tests for backends: model registry, backend factory, BaseBackend ABC,
and retry configuration.
"""

import asyncio

import pytest

from encre.backends.registry import BackendRegistry, ModelInfo, REGISTRY, resolve_model_info
from encre.backends.base import BaseBackend
from encre.backends.retry import RetryConfig, DEFAULT_RETRY_CONFIG
from encre.backend import create_backend


# ===========================================================================
# ModelInfo dataclass
# ===========================================================================

class TestModelInfo:
    def test_create_default(self):
        mi = ModelInfo(name="test-model", provider="openai")
        assert mi.name == "test-model"
        assert mi.provider == "openai"
        assert mi.context_window == 128000
        assert mi.max_output_tokens == 8192
        assert mi.supports_tools is True
        assert mi.supports_streaming is True

    def test_create_with_aliases(self):
        mi = ModelInfo(
            name="gpt-4o",
            provider="openai",
            aliases=["gpt4o", "4o"],
        )
        assert mi.aliases == ["gpt4o", "4o"]

    def test_equality(self):
        a = ModelInfo(name="m1", provider="openai")
        b = ModelInfo(name="m1", provider="openai")
        assert a == b

    def test_different_names_not_equal(self):
        a = ModelInfo(name="m1", provider="openai")
        b = ModelInfo(name="m2", provider="openai")
        assert a != b


# ===========================================================================
# BackendRegistry: register / unregister / resolve
# ===========================================================================

class TestBackendRegistryRegistration:
    def test_register_and_resolve_exact(self):
        registry = BackendRegistry()
        mi = ModelInfo(name="my-model", provider="anthropic", context_window=300000)
        registry.register(mi)
        resolved = registry.resolve("my-model")
        assert resolved is not None
        assert resolved.name == "my-model"
        assert resolved.context_window == 300000

    def test_register_with_aliases_resolve_alias(self):
        registry = BackendRegistry()
        mi = ModelInfo(name="my-model-2", provider="openai", aliases=["mm2", "alias2"])
        registry.register(mi)
        r1 = registry.resolve("mm2")
        assert r1 is not None
        assert r1.name == "my-model-2"
        r2 = registry.resolve("alias2")
        assert r2 is not None
        assert r2.name == "my-model-2"

    def test_unregister_removes_entry(self):
        registry = BackendRegistry()
        mi = ModelInfo(name="temp-model", provider="openai", aliases=["tm"])
        registry.register(mi)
        assert registry.resolve("temp-model") is not None
        registry.unregister("temp-model")
        assert registry.resolve("temp-model") is None

    def test_resolve_nonexistent(self):
        registry = BackendRegistry()
        assert registry.resolve("nonexistent-model-xyz") is None

    def test_register_overwrite(self):
        registry = BackendRegistry()
        mi1 = ModelInfo(name="overwrite-test", provider="openai", context_window=1000)
        mi2 = ModelInfo(name="overwrite-test", provider="openai", context_window=2000)
        registry.register(mi1)
        registry.register(mi2)
        r = registry.resolve("overwrite-test")
        assert r.context_window == 2000

    def test_list_models(self):
        registry = BackendRegistry()
        mi = ModelInfo(name="list-all-test", provider="groq")
        registry.register(mi)
        all_models = registry.list_models()
        assert any(m.name == "list-all-test" for m in all_models)


# ===========================================================================
# Global REGISTRY and resolve_model_info
# ===========================================================================

class TestResolveModelInfo:
    def test_resolve_known_model(self):
        info = resolve_model_info("gpt-4.1")
        assert info.name == "gpt-4.1"
        assert info.provider == "openai"
        assert info.context_window == 1048576

    def test_resolve_registered_custom_model(self):
        mi = ModelInfo(name="known-model", provider="anthropic", context_window=999000)
        REGISTRY.register(mi)
        info = resolve_model_info("known-model")
        assert info.context_window == 999000
        REGISTRY.unregister("known-model")

    def test_resolve_unregistered_openai(self):
        info = resolve_model_info("gpt-5-imaginary")
        assert info.provider == "openai"
        assert info.context_window == 1048576
        assert info.supports_tools is True

    def test_resolve_unregistered_anthropic(self):
        info = resolve_model_info("claude-opus-5-imaginary")
        assert info.provider == "anthropic"
        assert info.context_window == 200000
        assert info.supports_thinking is True
        assert info.supports_prompt_caching is True

    def test_resolve_unregistered_google(self):
        info = resolve_model_info("gemini-3-imaginary")
        assert info.provider == "google"
        assert info.context_window == 1048576

    def test_resolve_unregistered_deepseek(self):
        info = resolve_model_info("deepseek-v4-imaginary")
        assert info.provider == "deepseek"
        assert info.context_window == 1048576

    def test_resolve_unregistered_groq(self):
        info = resolve_model_info("groq-model-imaginary", provider="groq")
        assert info.provider == "groq"
        assert info.context_window == 131072

    def test_resolve_unregistered_ollama(self):
        info = resolve_model_info("some-ollama-model", provider="ollama")
        assert info.provider == "ollama"
        assert info.context_window == 8192

    def test_resolve_unregistered_local(self):
        info = resolve_model_info("my-local-model", provider="local")
        assert info.provider == "local"
        assert info.context_window == 4096
        assert info.max_output_tokens == 2048

    def test_resolve_honors_explicit_provider(self):
        info = resolve_model_info("some-unknown-model", provider="bedrock")
        assert info.provider == "bedrock"
        assert info.context_window == 200000

    def test_global_registry_has_known_models(self):
        info = REGISTRY.resolve("gpt-4.1")
        assert info is not None
        info2 = REGISTRY.resolve("claude-sonnet-4.6")
        assert info2 is not None


# ===========================================================================
# BaseBackend ABC compliance
# ===========================================================================

class TestBaseBackendABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BaseBackend()

    def test_concrete_subclasses_instantiate(self):
        from encre.backends.local import LocalBackend
        be = LocalBackend()
        assert isinstance(be, BaseBackend)

    def test_supports_tool_calling_is_abstract(self):
        assert "supports_tool_calling" in BaseBackend.__abstractmethods__

    def test_chat_is_abstract(self):
        assert "chat" in BaseBackend.__abstractmethods__

    def test_context_window_size_is_abstract(self):
        assert "context_window_size" in BaseBackend.__abstractmethods__

    def test_default_supports_thinking(self):
        from encre.backends.local import LocalBackend
        be = LocalBackend()
        assert hasattr(be, "supports_thinking")
        assert isinstance(be.supports_thinking(), bool)

    def test_default_supports_prompt_caching(self):
        from encre.backends.local import LocalBackend
        be = LocalBackend()
        assert hasattr(be, "supports_prompt_caching")
        assert isinstance(be.supports_prompt_caching(), bool)

    def test_default_count_tokens(self):
        from encre.backends.local import LocalBackend
        be = LocalBackend()
        assert be.count_tokens("hello") == -1

    def test_aclose_noop(self):
        from encre.backends.local import LocalBackend
        be = LocalBackend()
        asyncio.run(be.aclose())


# ===========================================================================
# Backend factory: create_backend()
# ===========================================================================

class TestCreateBackend:
    def test_create_openai(self):
        be = create_backend("openai")
        assert isinstance(be, BaseBackend)

    def test_create_anthropic(self):
        be = create_backend("anthropic")
        assert isinstance(be, BaseBackend)

    def test_create_ollama(self):
        be = create_backend("ollama")
        assert isinstance(be, BaseBackend)

    def test_create_deepseek(self):
        be = create_backend("deepseek")
        assert isinstance(be, BaseBackend)

    def test_create_google(self):
        be = create_backend("google")
        assert isinstance(be, BaseBackend)

    def test_create_groq(self):
        be = create_backend("groq")
        assert isinstance(be, BaseBackend)

    def test_create_local(self):
        be = create_backend("local")
        assert isinstance(be, BaseBackend)

    def test_create_bedrock(self):
        be = create_backend("bedrock")
        assert isinstance(be, BaseBackend)

    def test_create_openai_compatible(self):
        be = create_backend("openai_compatible", base_url="https://api.example.com/v1")
        assert isinstance(be, BaseBackend)

    def test_create_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_backend("nonexistent_backend")

    def test_create_backend_passes_kwargs(self):
        be = create_backend("openai", model="gpt-4o-mini", api_key="sk-test")
        assert be.model == "gpt-4o-mini"


# ===========================================================================
# RetryConfig
# ===========================================================================

class TestRetryConfig:
    def test_default_config(self):
        rc = RetryConfig()
        assert rc.max_retries == 5
        assert rc.base_delay == 1.0
        assert rc.max_delay == 60.0
        assert 429 in rc.retryable_status_codes
        assert 502 in rc.retryable_status_codes
        assert 503 in rc.retryable_status_codes
        assert 504 in rc.retryable_status_codes

    def test_default_retry_config_is_retry_config(self):
        assert isinstance(DEFAULT_RETRY_CONFIG, RetryConfig)

    def test_custom_config(self):
        rc = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            retryable_status_codes={429, 500},
        )
        assert rc.max_retries == 5
        assert rc.base_delay == 2.0
        assert rc.max_delay == 120.0
        assert rc.retryable_status_codes == {429, 500}

    def test_zero_retries_disables_retry(self):
        rc = RetryConfig(max_retries=0)
        assert rc.max_retries == 0

    def test_retry_on_exceptions_default(self):
        import httpx
        rc = RetryConfig()
        assert httpx.TimeoutException in rc.retryable_exceptions
        assert httpx.ConnectError in rc.retryable_exceptions


# ===========================================================================
# Backend-specific capability checks
# ===========================================================================

class TestBackendCapabilities:
    def test_openai_capabilities(self):
        be = create_backend("openai", api_key="sk-fake")
        assert be.supports_tool_calling() is True
        assert be.context_window_size() > 0
        assert isinstance(be.supports_thinking(), bool)

    def test_anthropic_capabilities(self):
        be = create_backend("anthropic", api_key="sk-ant-fake")
        assert be.supports_tool_calling() is True
        assert be.context_window_size() > 0
        assert be.supports_thinking() is True
        assert be.supports_prompt_caching() is True

    def test_deepseek_capabilities(self):
        be = create_backend("deepseek", api_key="sk-fake")
        assert be.supports_tool_calling() is True
        assert be.context_window_size() > 0

    def test_local_capabilities(self):
        be = create_backend("local")
        assert be.context_window_size() > 0
        assert isinstance(be.supports_tool_calling(), bool)

    def test_ollama_capabilities(self):
        be = create_backend("ollama")
        assert be.context_window_size() > 0
        assert isinstance(be.supports_tool_calling(), bool)


# ===========================================================================
# Config integration with backends
# ===========================================================================

class TestConfigBackendIntegration:
    def test_config_server_backend_type_default(self):
        from encre.config import EncreConfig
        cfg = EncreConfig()
        assert cfg.backend_type == "openai"
        be = create_backend(cfg.backend_type, api_key="sk-fake")
        assert isinstance(be, BaseBackend)

    def test_config_with_kwargs(self):
        from encre.config import EncreConfig
        cfg = EncreConfig(
            backend_type="anthropic",
            backend_kwargs={"max_tokens": 32768},
        )
        assert cfg.backend_type == "anthropic"
        assert cfg.backend_kwargs["max_tokens"] == 32768
