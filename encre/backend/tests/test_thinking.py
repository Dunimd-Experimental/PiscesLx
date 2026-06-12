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

"""Tests for resolve_thinking_config and get_thinking_budget_tokens from encre.thinking."""

import inspect
import pytest

from encre.thinking.config import resolve_thinking_config, get_thinking_budget_tokens
from encre.utils.types import (
    AdaptiveThinking,
    EnabledThinking,
    DisabledThinking,
)


class TestResolveThinkingConfig:
    """Verify resolve_thinking_config returns correct ThinkingConfig based on inputs."""

    def test_returns_config_as_is_when_provided(self):
        """If an explicit config is passed, it should be returned unchanged."""
        explicit = EnabledThinking(enabled=True, budget_tokens=5000)
        result = resolve_thinking_config(explicit, "any-model")
        assert result is explicit
        assert result.enabled is True
        assert result.budget_tokens == 5000

    def test_none_config_with_claude_model_adaptive(self):
        """None config + Claude-like model name -> AdaptiveThinking."""
        result = resolve_thinking_config(None, "claude-sonnet-4-20250514")
        assert isinstance(result, AdaptiveThinking)
        assert result.enabled is True
        assert result.min_tokens == 1024
        assert result.max_tokens == 8192

    def test_none_config_with_sonnet_model_adaptive(self):
        """None config + sonnet in name -> AdaptiveThinking (checks for 'sonnet')."""
        result = resolve_thinking_config(None, "sonnet-20240229")
        assert isinstance(result, AdaptiveThinking)
        assert result.enabled is True

    def test_none_config_with_opus_model_adaptive(self):
        """None config + Claude model -> AdaptiveThinking (checks for 'claude')."""
        result = resolve_thinking_config(None, "claude-opus-4-20250515")
        assert isinstance(result, AdaptiveThinking)
        assert result.enabled is True

    def test_none_config_with_non_claude_model_disabled(self):
        """None config + non-Claude model -> DisabledThinking."""
        result = resolve_thinking_config(None, "gpt-4o")
        assert isinstance(result, DisabledThinking)
        assert result.enabled is False

    def test_none_config_with_deepseek_model_disabled(self):
        result = resolve_thinking_config(None, "deepseek-v3")
        assert isinstance(result, DisabledThinking)
        assert result.enabled is False

    def test_none_config_with_gemini_model_disabled(self):
        result = resolve_thinking_config(None, "gemini-2.5-pro")
        assert isinstance(result, DisabledThinking)
        assert result.enabled is False

    def test_disabled_config_passed_through(self):
        """Explicit DisabledThinking is returned as-is."""
        disabled = DisabledThinking()
        result = resolve_thinking_config(disabled, "claude-sonnet-4-20250514")
        assert result is disabled
        assert result.enabled is False

    def test_adaptive_config_passed_through(self):
        """Explicit AdaptiveThinking is returned as-is."""
        adaptive = AdaptiveThinking(enabled=True, budget_ratio=0.75)
        result = resolve_thinking_config(adaptive, "gpt-4o")
        assert result is adaptive
        assert result.budget_ratio == 0.75


class TestGetThinkingBudget:
    """Verify get_thinking_budget_tokens returns correct token budgets."""

    def test_adaptive_returns_max_tokens(self):
        """AdaptiveThinking budget = max_tokens."""
        config = AdaptiveThinking(enabled=True, max_tokens=8192)
        assert get_thinking_budget_tokens(config) == 8192

    def test_enabled_returns_budget_tokens(self):
        """EnabledThinking budget = budget_tokens."""
        config = EnabledThinking(budget_tokens=16000)
        assert get_thinking_budget_tokens(config) == 16000

    def test_disabled_returns_zero(self):
        """DisabledThinking always returns 0."""
        config = DisabledThinking()
        assert get_thinking_budget_tokens(config) == 0

    def test_adaptive_custom_max_tokens(self):
        config = AdaptiveThinking(enabled=True, max_tokens=16000)
        assert get_thinking_budget_tokens(config) == 16000

    def test_enabled_custom_budget_tokens(self):
        config = EnabledThinking(budget_tokens=0)
        assert get_thinking_budget_tokens(config) == 0


class TestResolveThinkingConfigSignature:
    """Verify the function signature matches expectations."""

    def test_accepts_config_and_model(self):
        sig = inspect.signature(resolve_thinking_config)
        params = list(sig.parameters.keys())
        assert "config" in params
        assert "model" in params

    def test_two_parameters(self):
        sig = inspect.signature(resolve_thinking_config)
        assert len(sig.parameters) == 2
