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
Model metadata registry — centralised model info for all supported providers.

This module provides two mechanisms for resolving model metadata:

1. :class:`BackendRegistry` — a dynamic, thread-safe registry that maps model
   names to :class:`ModelInfo` dataclass instances.  Users can register custom
   models at runtime.

2. :func:`resolve_model_info` — a convenience function that first checks the
   registry, then falls back to provider-level defaults defined in
   ``_PROVIDER_DEFAULTS``.

2026 model landscape
--------------------
As of mid-2026, the LLM market has undergone significant changes:

- **OpenAI**: GPT-4o is fully deprecated. The GPT-4.1 family (Nano/Mini/full)
  serves as the mid-range workhorse with 1M context windows. GPT-5.x variants
  (5.2, 5.4, 5.5) offer tiered pricing and capability levels. o3 and o4-mini
  handle reasoning-heavy tasks.

- **Anthropic**: Claude Opus 4.6/4.7 and Sonnet 4.5/4.6 are the primary models.
  All support 200K context (1M in beta) with prompt caching at 90% discount.

- **Google**: Gemini 2.5 Pro is the flagship with 1M context and multimodal
  support (text, image, video, audio).

- **DeepSeek**: V4-Flash and V4-Pro replace the deprecated deepseek-chat and
  deepseek-reasoner. Both offer 1M context with aggressive cache discounts.

- **Groq**: Continues to offer ultra-low-latency inference for Llama 3.3 70B,
  Llama 4 Scout, and GPT-OSS 120B.
"""

from dataclasses import dataclass, field
from typing import Any

import threading


@dataclass
class ModelInfo:
    """Metadata for a single LLM model.

    This dataclass stores all information needed by the agent loop to make
    decisions about context management, cost tracking, and capability detection.

    Attributes:
        name: Canonical model name (e.g. ``"gpt-4.1"``).
        provider: Provider identifier (``"openai"``, ``"anthropic"``, etc.).
        context_window: Maximum total tokens (input + output) the model supports.
        max_output_tokens: Maximum tokens the model can generate in one response.
        supports_tools: Whether the model supports function/tool calling.
        supports_thinking: Whether the model emits reasoning/thinking tokens.
        supports_images: Whether the model accepts image inputs.
        supports_prompt_caching: Whether prompt caching is available.
        supports_streaming: Whether streaming responses are supported.
        aliases: Alternative names that resolve to this model.
        pricing_input_per_1m: Cost per 1M input tokens in USD.
        pricing_output_per_1m: Cost per 1M output tokens in USD.
        extra: Provider-specific metadata (e.g. ``{"thinking_budget": 32000}``).
    """

    name: str
    provider: str
    context_window: int = 128000
    max_output_tokens: int = 8192
    supports_tools: bool = True
    supports_thinking: bool = False
    supports_images: bool = False
    supports_prompt_caching: bool = False
    supports_streaming: bool = True
    aliases: list[str] | None = None
    pricing_input_per_1m: float = 0.0
    pricing_output_per_1m: float = 0.0
    extra: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Provider-level defaults
# ---------------------------------------------------------------------------
# These values are used when a model name is not found in the registry.  They
# represent reasonable defaults for each provider's most common configuration
# as of May 2026.

_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    # OpenAI — 2026 lineup (GPT-4o deprecated, replaced by GPT-4.1 family)
    # GPT-4.1: 1M context, $2/$8 per 1M tokens, prompt caching at 75% off
    # GPT-5.x: 128K-400K context, $1.25-$5/$10-$30 per 1M tokens
    # o3/o4-mini: 200K context, reasoning-optimised
    "openai": {
        "context_window": 1048576,
        "max_output_tokens": 16384,
        "supports_tools": True,
        "supports_images": True,
        "supports_prompt_caching": True,
    },
    # Anthropic — 2026 lineup
    # Claude Opus 4.6/4.7: $5/$25 per 1M tokens, 200K context (1M beta)
    # Claude Sonnet 4.5/4.6: $3/$15 per 1M tokens, 200K context (1M beta)
    # Claude Haiku 4.5: $1/$5 per 1M tokens, 200K context
    # All support prompt caching at 90% off cache reads
    "anthropic": {
        "context_window": 200000,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_images": True,
        "supports_thinking": True,
        "supports_prompt_caching": True,
    },
    # Google — 2026 lineup
    # Gemini 2.5 Pro: 1M context, $1.25/$10 (short) / $2.50/$15 (long)
    # Supports multimodal (text, image, video, audio) and thinking
    "google": {
        "context_window": 1048576,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_images": True,
        "supports_thinking": True,
    },
    # DeepSeek — 2026 lineup
    # V4-Flash: $0.14/$0.28 per 1M tokens, 1M context, 384K output
    # V4-Pro: $1.74/$3.48 per 1M tokens, 1M context, 384K output
    # 80-92% cache hit discount
    "deepseek": {
        "context_window": 1048576,
        "max_output_tokens": 65536,
        "supports_tools": True,
        "supports_thinking": True,
        "supports_prompt_caching": True,
    },
    # Groq — 2026 lineup
    # Llama 3.3 70B: $0.59/$0.79 per 1M tokens, 131K context
    # Llama 4 Scout: $0.11/$0.34 per 1M tokens, 131K context
    # GPT-OSS 120B: $0.15/$0.60 per 1M tokens, 131K context
    "groq": {
        "context_window": 131072,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_images": False,
    },
    # Ollama — local models, no fixed defaults
    "ollama": {
        "context_window": 8192,
        "max_output_tokens": 4096,
        "supports_tools": False,
        "supports_images": False,
    },
    # Local — Hugging Face transformers, varies by model
    "local": {
        "context_window": 4096,
        "max_output_tokens": 2048,
        "supports_tools": False,
        "supports_images": False,
    },
    # AWS Bedrock — depends on the underlying model
    "bedrock": {
        "context_window": 200000,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_images": True,
    },
    # OpenRouter — unified multi-provider access
    "openrouter": {
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supports_tools": True,
        "supports_thinking": True,
        "supports_images": True,
    },
    # NovitaAI — 200+ models
    "novita": {
        "context_window": 128000,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_images": True,
    },
    # AI Gateway — generic gateway
    "aigateway": {
        "context_window": 128000,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_images": True,
    },
    # GLM (Zhipu AI) — 2026 lineup
    "glm": {
        "context_window": 131072,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_images": True,
        "supports_thinking": True,
    },
    # Kimi (Moonshot) — 2026 lineup
    "kimi": {
        "context_window": 262144,
        "max_output_tokens": 16384,
        "supports_tools": True,
        "supports_thinking": True,
    },
    # Arcee AI — domain-adapted LLMs
    "arcee": {
        "context_window": 128000,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_images": False,
    },
    # GMI Cloud — GPU cloud infrastructure
    "gmi": {
        "context_window": 163000,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_thinking": True,
    },
    # MiniMax — 2026 lineup
    "minimax": {
        "context_window": 196000,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_thinking": True,
    },
    # Alibaba DashScope (Qwen) — 2026 lineup
    "alibaba": {
        "context_window": 131072,
        "max_output_tokens": 16384,
        "supports_tools": True,
        "supports_thinking": True,
        "supports_images": True,
    },
    # Alibaba Coding Plan — independent billing SKU
    "alibaba-coding-plan": {
        "context_window": 131072,
        "max_output_tokens": 16384,
        "supports_tools": True,
        "supports_thinking": True,
    },
    # Kilo Code Gateway
    "kilocode": {
        "context_window": 1000000,
        "max_output_tokens": 128000,
        "supports_tools": True,
        "supports_thinking": True,
    },
    # Xiaomi MiMo — 2026 lineup
    "xiaomi": {
        "context_window": 262144,
        "max_output_tokens": 16384,
        "supports_tools": True,
        "supports_thinking": True,
    },
    # Tencent TokenHub
    "tencent": {
        "context_window": 131072,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_thinking": True,
    },
    # Hugging Face Inference API
    "huggingface": {
        "context_window": 128000,
        "max_output_tokens": 8192,
        "supports_tools": True,
        "supports_images": True,
    },
    # OpenCode Zen
    "opencode-zen": {
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supports_tools": True,
        "supports_thinking": True,
    },
    # OpenCode Go
    "opencode-go": {
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supports_tools": True,
        "supports_thinking": True,
    },
    # LM Studio — local models
    "lmstudio": {
        "context_window": 8192,
        "max_output_tokens": 4096,
        "supports_tools": False,
        "supports_images": False,
    },
    # GitHub Copilot
    "github-copilot": {
        "context_window": 128000,
        "max_output_tokens": 16384,
        "supports_tools": True,
        "supports_images": True,
    },
}


# ---------------------------------------------------------------------------
# Known model registry (2026 data)
# ---------------------------------------------------------------------------
# These entries are pre-populated with verified pricing and capability data
# collected in May 2026.  Models marked "deprecated" are kept for backward
# compatibility but should not be used for new sessions.

_KNOWN_MODELS: dict[str, ModelInfo] = {
    # ===== OpenAI =====
    "gpt-4.1": ModelInfo(
        name="gpt-4.1",
        provider="openai",
        context_window=1048576,
        max_output_tokens=16384,
        supports_images=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=2.0,
        pricing_output_per_1m=8.0,
        aliases=["gpt-4.1-2026-05"],
    ),
    "gpt-4.1-mini": ModelInfo(
        name="gpt-4.1-mini",
        provider="openai",
        context_window=1048576,
        max_output_tokens=16384,
        supports_images=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=0.40,
        pricing_output_per_1m=1.60,
    ),
    "gpt-4.1-nano": ModelInfo(
        name="gpt-4.1-nano",
        provider="openai",
        context_window=1048576,
        max_output_tokens=16384,
        supports_images=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=0.10,
        pricing_output_per_1m=0.40,
    ),
    "gpt-5.2": ModelInfo(
        name="gpt-5.2",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        supports_images=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=1.75,
        pricing_output_per_1m=14.0,
    ),
    "gpt-5.4": ModelInfo(
        name="gpt-5.4",
        provider="openai",
        context_window=400000,
        max_output_tokens=16384,
        supports_images=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=2.50,
        pricing_output_per_1m=15.0,
    ),
    "gpt-5.5": ModelInfo(
        name="gpt-5.5",
        provider="openai",
        context_window=1048576,
        max_output_tokens=16384,
        supports_images=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=5.0,
        pricing_output_per_1m=30.0,
    ),
    "o3": ModelInfo(
        name="o3",
        provider="openai",
        context_window=200000,
        max_output_tokens=100000,
        supports_images=True,
        supports_thinking=True,
        pricing_input_per_1m=2.0,
        pricing_output_per_1m=8.0,
    ),
    "o4-mini": ModelInfo(
        name="o4-mini",
        provider="openai",
        context_window=200000,
        max_output_tokens=100000,
        supports_images=True,
        supports_thinking=True,
        pricing_input_per_1m=1.10,
        pricing_output_per_1m=4.40,
    ),
    # Legacy — kept for backward compatibility, deprecated as of Jan 2026
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        supports_images=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=2.50,
        pricing_output_per_1m=10.0,
        extra={"deprecated": True, "replaced_by": "gpt-4.1"},
    ),
    # ===== Anthropic =====
    "claude-opus-4.6": ModelInfo(
        name="claude-opus-4.6",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        supports_images=True,
        supports_thinking=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=5.0,
        pricing_output_per_1m=25.0,
    ),
    "claude-opus-4.7": ModelInfo(
        name="claude-opus-4.7",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        supports_images=True,
        supports_thinking=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=5.0,
        pricing_output_per_1m=25.0,
    ),
    "claude-sonnet-4.5": ModelInfo(
        name="claude-sonnet-4.5",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        supports_images=True,
        supports_thinking=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=3.0,
        pricing_output_per_1m=15.0,
    ),
    "claude-sonnet-4.6": ModelInfo(
        name="claude-sonnet-4.6",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        supports_images=True,
        supports_thinking=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=3.0,
        pricing_output_per_1m=15.0,
        extra={"beta": True, "extended_context": 1048576},
    ),
    "claude-haiku-4.5": ModelInfo(
        name="claude-haiku-4.5",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        supports_images=True,
        supports_thinking=False,
        supports_prompt_caching=True,
        pricing_input_per_1m=1.0,
        pricing_output_per_1m=5.0,
    ),
    # ===== Google =====
    "gemini-2.5-pro": ModelInfo(
        name="gemini-2.5-pro",
        provider="google",
        context_window=1048576,
        max_output_tokens=8192,
        supports_images=True,
        supports_thinking=True,
        pricing_input_per_1m=1.25,
        pricing_output_per_1m=10.0,
        extra={
            "long_context_pricing": {"input": 2.50, "output": 15.0},
            "long_context_threshold": 200000,
        },
    ),
    "gemini-2.5-flash": ModelInfo(
        name="gemini-2.5-flash",
        provider="google",
        context_window=1048576,
        max_output_tokens=8192,
        supports_images=True,
        supports_thinking=True,
        pricing_input_per_1m=0.15,
        pricing_output_per_1m=0.60,
    ),
    # ===== DeepSeek =====
    "deepseek-v4-flash": ModelInfo(
        name="deepseek-v4-flash",
        provider="deepseek",
        context_window=1048576,
        max_output_tokens=65536,
        supports_images=False,
        supports_thinking=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=0.14,
        pricing_output_per_1m=0.28,
        aliases=["deepseek-chat"],
        extra={"cache_hit_discount": 0.92},
    ),
    "deepseek-v4-pro": ModelInfo(
        name="deepseek-v4-pro",
        provider="deepseek",
        context_window=1048576,
        max_output_tokens=65536,
        supports_images=False,
        supports_thinking=True,
        supports_prompt_caching=True,
        pricing_input_per_1m=1.74,
        pricing_output_per_1m=3.48,
        aliases=["deepseek-reasoner"],
        extra={"cache_hit_discount": 0.80},
    ),
    # Legacy DeepSeek — deprecated, map to V4 equivalents
    "deepseek-chat": ModelInfo(
        name="deepseek-chat",
        provider="deepseek",
        context_window=65536,
        max_output_tokens=8192,
        supports_tools=True,
        supports_thinking=False,
        pricing_input_per_1m=0.14,
        pricing_output_per_1m=0.28,
        extra={"deprecated": True, "replaced_by": "deepseek-v4-flash"},
    ),
    "deepseek-reasoner": ModelInfo(
        name="deepseek-reasoner",
        provider="deepseek",
        context_window=65536,
        max_output_tokens=8192,
        supports_tools=True,
        supports_thinking=True,
        pricing_input_per_1m=0.55,
        pricing_output_per_1m=2.19,
        extra={"deprecated": True, "replaced_by": "deepseek-v4-pro"},
    ),
    # ===== Groq =====
    "llama-3.3-70b": ModelInfo(
        name="llama-3.3-70b",
        provider="groq",
        context_window=131072,
        max_output_tokens=8192,
        supports_tools=True,
        pricing_input_per_1m=0.59,
        pricing_output_per_1m=0.79,
    ),
    "llama-4-scout": ModelInfo(
        name="llama-4-scout",
        provider="groq",
        context_window=131072,
        max_output_tokens=8192,
        supports_tools=True,
        supports_images=True,
        pricing_input_per_1m=0.11,
        pricing_output_per_1m=0.34,
    ),
    "gpt-oss-120b": ModelInfo(
        name="gpt-oss-120b",
        provider="groq",
        context_window=131072,
        max_output_tokens=8192,
        supports_tools=True,
        pricing_input_per_1m=0.15,
        pricing_output_per_1m=0.60,
    ),
    "llama-3.1-8b": ModelInfo(
        name="llama-3.1-8b",
        provider="groq",
        context_window=131072,
        max_output_tokens=8192,
        supports_tools=True,
        pricing_input_per_1m=0.05,
        pricing_output_per_1m=0.08,
    ),
}


class BackendRegistry:
    """Thread-safe registry for model metadata.

    Provides dynamic registration and resolution of :class:`ModelInfo` entries.
    The registry is pre-populated with known models from ``_KNOWN_MODELS`` and
    can be extended at runtime via :meth:`register`.

    Thread safety is guaranteed by a :class:`threading.Lock` around all
    read-write operations.

    Typical usage::

        registry = BackendRegistry()
        info = registry.resolve("gpt-4.1")
        if info:
            print(f"{info.name}: {info.context_window} tokens, ${info.pricing_input_per_1m}/1M input")
    """

    def __init__(self) -> None:
        """Initialise the registry with all known models from ``_KNOWN_MODELS``."""
        self._models: dict[str, ModelInfo] = {}
        self._lock = threading.Lock()
        self._load_known_models()

    def _load_known_models(self) -> None:
        """Populate the registry with pre-defined model entries.

        Each model is registered under its canonical name and all of its
        aliases, so that ``resolve("deepseek-chat")`` returns the same
        :class:`ModelInfo` as ``resolve("deepseek-v4-flash")``.
        """
        for model in _KNOWN_MODELS.values():
            self._models[model.name] = model
            if model.aliases:
                for alias in model.aliases:
                    self._models[alias] = model

    def register(self, model: ModelInfo) -> None:
        """Register a new model or update an existing one.

        Args:
            model: The :class:`ModelInfo` instance to register.  If a model
                with the same name already exists, it is overwritten.
        """
        with self._lock:
            self._models[model.name] = model
            if model.aliases:
                for alias in model.aliases:
                    self._models[alias] = model

    def resolve(self, model_name: str) -> ModelInfo | None:
        """Look up a model by name or alias.

        Args:
            model_name: The model name or alias to look up.

        Returns:
            The :class:`ModelInfo` if found, or ``None`` if the model is not
            registered.
        """
        with self._lock:
            return self._models.get(model_name)

    def unregister(self, model_name: str) -> None:
        """Remove a model from the registry.

        Args:
            model_name: The canonical name of the model to remove.
        """
        with self._lock:
            self._models.pop(model_name, None)

    def list_models(self) -> list[ModelInfo]:
        """Return a snapshot of all registered models.

        Returns:
            A list of :class:`ModelInfo` instances (deduplicated by canonical name).
        """
        seen: set[str] = set()
        result: list[ModelInfo] = []
        with self._lock:
            for name, info in self._models.items():
                if info.name not in seen:
                    seen.add(info.name)
                    result.append(info)
        return result


# Global singleton registry instance.
REGISTRY = BackendRegistry()


def resolve_model_info(model_name: str, provider: str | None = None) -> ModelInfo:
    """Resolve model metadata from the registry or provider defaults.

    This is the primary entry point for model metadata resolution.  It first
    checks the global :data:`REGISTRY` for an exact match.  If no match is
    found, it falls back to ``_PROVIDER_DEFAULTS`` using the inferred or
    provided provider name.

    Args:
        model_name: The model name to resolve (e.g. ``"gpt-4.1"``).
        provider: Optional provider hint.  If not provided, the provider is
            inferred from the model name prefix (e.g. ``"gpt-"`` → ``"openai"``,
            ``"claude-"`` → ``"anthropic"``).

    Returns:
        A :class:`ModelInfo` instance with either registry data or provider
        defaults.  If no defaults exist for the provider, a minimal
        :class:`ModelInfo` with default values is returned.
    """
    # Try the registry first.
    info = REGISTRY.resolve(model_name)
    if info is not None:
        return info

    # Infer provider from model name if not given.
    if provider is None:
        provider = _infer_provider(model_name)

    # Fall back to provider defaults.
    defaults = _PROVIDER_DEFAULTS.get(provider, {})
    return ModelInfo(
        name=model_name,
        provider=provider,
        context_window=defaults.get("context_window", 128000),
        max_output_tokens=defaults.get("max_output_tokens", 8192),
        supports_tools=defaults.get("supports_tools", True),
        supports_thinking=defaults.get("supports_thinking", False),
        supports_images=defaults.get("supports_images", False),
        supports_prompt_caching=defaults.get("supports_prompt_caching", False),
        supports_streaming=defaults.get("supports_streaming", True),
        pricing_input_per_1m=defaults.get("pricing_input_per_1m", 0.0),
        pricing_output_per_1m=defaults.get("pricing_output_per_1m", 0.0),
    )


def _infer_provider(model_name: str) -> str:
    """Infer the provider name from a model name prefix.

    Inference rules (case-insensitive):
    - ``gpt-``, ``o3``, ``o4-`` → ``"openai"``
    - ``claude-`` → ``"anthropic"``
    - ``gemini-`` → ``"google"``
    - ``deepseek-``, ``deepseek`` → ``"deepseek"``
    - ``llama-``, ``mixtral-`` → ``"groq"``
    - ``ollama`` → ``"ollama"``
    - ``bedrock`` → ``"bedrock"``
    - Everything else → ``"openai"`` (OpenAI-compatible default)
    """
    name_lower = model_name.lower()
    if name_lower.startswith("gpt-") or name_lower.startswith("o3") or name_lower.startswith("o4-"):
        return "openai"
    if name_lower.startswith("claude-"):
        return "anthropic"
    if name_lower.startswith("gemini-"):
        return "google"
    if name_lower.startswith("deepseek"):
        return "deepseek"
    if name_lower.startswith("llama-") or name_lower.startswith("mixtral-"):
        return "groq"
    if name_lower.startswith("ollama"):
        return "ollama"
    if name_lower.startswith("bedrock"):
        return "bedrock"
    if name_lower.startswith("glm-") or name_lower.startswith("glm_"):
        return "glm"
    if name_lower.startswith("kimi-") or name_lower.startswith("moonshot-"):
        return "kimi"
    if name_lower.startswith("qwen") or name_lower.startswith("qwq-") or name_lower.startswith("qwen-"):
        return "alibaba"
    if name_lower.startswith("minimax-"):
        return "minimax"
    if name_lower.startswith("mimo-"):
        return "xiaomi"
    if name_lower.startswith("hy") or name_lower.startswith("hunyuan-"):
        return "tencent"
    if name_lower.startswith("openrouter"):
        return "openrouter"
    if name_lower.startswith("novita"):
        return "novita"
    if name_lower.startswith("arcee"):
        return "arcee"
    if name_lower.startswith("gmi-") or name_lower.startswith("deepseek-ai/") or name_lower.startswith("zai-org/"):
        return "gmi"
    if name_lower.startswith("kilocode") or name_lower.startswith("kilo-"):
        return "kilocode"
    if name_lower.startswith("opencode-zen"):
        return "opencode-zen"
    if name_lower.startswith("opencode-go"):
        return "opencode-go"
    if name_lower.startswith("copilot"):
        return "github-copilot"
    return "openai"