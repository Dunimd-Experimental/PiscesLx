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

"""Tests for failover and router backends."""

import asyncio

import pytest

from encre.backends.failover import FailoverBackend, BackendHealth
from encre.backends.router import RouterBackend, CostTracker, Route, TaskCategory
from encre.backend import create_backend
from encre.backends.base import BaseBackend


# ===========================================================================
# BackendHealth
# ===========================================================================

class TestBackendHealth:
    def test_initial_healthy(self):
        bh = BackendHealth(name="openai")
        assert bh.healthy is True
        assert bh.consecutive_failures == 0
        assert bh.total_failures == 0
        assert bh.total_requests == 0

    def test_record_failure(self):
        bh = BackendHealth(name="openai")
        bh.record_failure("timeout")
        assert bh.consecutive_failures == 1
        assert bh.total_failures == 1
        assert bh.total_requests == 1
        assert bh.last_error == "timeout"

    def test_record_success_resets_consecutive(self):
        bh = BackendHealth(name="openai")
        bh.record_failure("err1")
        bh.record_failure("err2")
        bh.record_success()
        assert bh.consecutive_failures == 0
        assert bh.total_requests == 3
        assert bh.healthy is True

    def test_consecutive_failures_threshold(self):
        bh = BackendHealth(name="openai")
        for i in range(3):
            bh.record_failure(f"error {i}")
        assert bh.healthy is False

    def test_manual_recovery(self):
        import time
        bh = BackendHealth(name="openai")
        for _ in range(3):
            bh.record_failure("timeout")
        assert bh.healthy is False
        # Simulate grace period passing and probe
        bh.last_checked = time.time() - 400
        bh.healthy = True
        bh.consecutive_failures = 0
        assert bh.healthy is True


# ===========================================================================
# FailoverBackend
# ===========================================================================

class TestFailoverBackend:
    def test_create(self):
        be1 = create_backend("openai", api_key="sk-fake")
        be2 = create_backend("anthropic", api_key="sk-ant-fake")
        fb = FailoverBackend(backends=[("primary", be1), ("fallback", be2)])
        assert fb is not None
        assert isinstance(fb, BaseBackend)

    def test_active_name_starts_first(self):
        be1 = create_backend("openai", api_key="sk-fake")
        be2 = create_backend("anthropic", api_key="sk-ant-fake")
        fb = FailoverBackend(backends=[("openai", be1), ("anthropic", be2)])
        assert fb.active_backend_name == "openai"

    def test_get_health(self):
        be1 = create_backend("openai", api_key="sk-fake")
        be2 = create_backend("anthropic", api_key="sk-ant-fake")
        fb = FailoverBackend(backends=[("openai", be1), ("anthropic", be2)])
        health = fb.get_health()
        assert "openai" in health
        assert "anthropic" in health
        assert health["openai"]["healthy"] is True

    def test_context_window_size(self):
        be1 = create_backend("openai", api_key="sk-fake")
        be2 = create_backend("anthropic", api_key="sk-ant-fake")
        fb = FailoverBackend(backends=[("p", be1), ("f", be2)])
        assert fb.context_window_size() > 0

    def test_supports_tool_calling(self):
        be1 = create_backend("openai", api_key="sk-fake")
        be2 = create_backend("anthropic", api_key="sk-ant-fake")
        fb = FailoverBackend(backends=[("p", be1), ("f", be2)])
        assert isinstance(fb.supports_tool_calling(), bool)

    def test_supports_thinking(self):
        be1 = create_backend("openai", api_key="sk-fake")
        be2 = create_backend("anthropic", api_key="sk-ant-fake")
        fb = FailoverBackend(backends=[("p", be1), ("f", be2)])
        assert isinstance(fb.supports_thinking(), bool)

    def test_supports_prompt_caching(self):
        be1 = create_backend("openai", api_key="sk-fake")
        be2 = create_backend("anthropic", api_key="sk-ant-fake")
        fb = FailoverBackend(backends=[("p", be1), ("f", be2)])
        assert isinstance(fb.supports_prompt_caching(), bool)

    def test_three_backend_chain(self):
        be1 = create_backend("openai", api_key="sk-fake")
        be2 = create_backend("anthropic", api_key="sk-ant-fake")
        be3 = create_backend("deepseek", api_key="sk-fake")
        fb = FailoverBackend(backends=[("a", be1), ("b", be2), ("c", be3)])
        assert fb.active_backend_name == "a"
        assert len(fb.get_health()) == 3

    def test_empty_backends_raises(self):
        with pytest.raises(ValueError, match="At least one backend"):
            FailoverBackend(backends=[])

    def test_count_tokens(self):
        be1 = create_backend("openai", api_key="sk-fake")
        fb = FailoverBackend(backends=[("p", be1)])
        assert isinstance(fb.count_tokens("hello"), int)

    def test_aclose(self):
        be1 = create_backend("openai", api_key="sk-fake")
        fb = FailoverBackend(backends=[("p", be1)])
        asyncio.run(fb.aclose())


# ===========================================================================
# Route
# ===========================================================================

class TestRoute:
    def test_create(self):
        backend = create_backend("openai", api_key="sk-fake")
        route = Route(category=TaskCategory.CODING, backend=backend)
        assert route.category == TaskCategory.CODING
        assert route.priority == 0

    def test_matches_coding_prompt(self):
        backend = create_backend("openai", api_key="sk-fake")
        route = Route(category=TaskCategory.CODING, backend=backend)
        confidence = route.matches("write a python function to sort a list")
        assert confidence > 0.0

    def test_matches_research_prompt(self):
        backend = create_backend("openai", api_key="sk-fake")
        route = Route(category=TaskCategory.RESEARCH, backend=backend)
        confidence = route.matches("research the best database for microservices")
        assert confidence > 0.0

    def test_matches_no_match(self):
        backend = create_backend("openai", api_key="sk-fake")
        route = Route(category=TaskCategory.CODING, backend=backend)
        confidence = route.matches("hello how are you")
        assert confidence == 0.0


# ===========================================================================
# RouterBackend
# ===========================================================================

class TestRouterBackend:
    def test_create(self):
        routes = {TaskCategory.CODING: create_backend("openai", api_key="sk-fake")}
        default = create_backend("openai", api_key="sk-fake")
        rb = RouterBackend(routes=routes, default=default)
        assert isinstance(rb, BaseBackend)

    def test_last_route_default(self):
        routes = {TaskCategory.CODING: create_backend("openai", api_key="sk-fake")}
        default = create_backend("openai", api_key="sk-fake")
        rb = RouterBackend(routes=routes, default=default)
        assert rb.last_route == "default"

    def test_context_window_size(self):
        default = create_backend("openai", api_key="sk-fake")
        rb = RouterBackend(routes={}, default=default)
        assert rb.context_window_size() > 0

    def test_supports_tool_calling(self):
        default = create_backend("openai", api_key="sk-fake")
        rb = RouterBackend(routes={}, default=default)
        assert isinstance(rb.supports_tool_calling(), bool)

    def test_supports_thinking(self):
        default = create_backend("openai", api_key="sk-fake")
        rb = RouterBackend(routes={}, default=default)
        assert isinstance(rb.supports_thinking(), bool)

    def test_cost_tracker_enabled(self):
        default = create_backend("openai", api_key="sk-fake")
        rb = RouterBackend(routes={}, default=default, track_costs=True)
        assert rb.cost_tracker is not None

    def test_cost_tracker_disabled(self):
        default = create_backend("openai", api_key="sk-fake")
        rb = RouterBackend(routes={}, default=default, track_costs=False)
        assert rb.cost_tracker is None

    def test_route_stats_defaults(self):
        routes = {TaskCategory.CODING: create_backend("openai", api_key="sk-fake")}
        default = create_backend("openai", api_key="sk-fake")
        rb = RouterBackend(routes=routes, default=default)
        stats = rb.route_stats
        assert isinstance(stats, dict)
        assert "default" in stats
        assert TaskCategory.CODING in stats


# ===========================================================================
# CostTracker
# ===========================================================================

class TestCostTracker:
    def test_create(self):
        ct = CostTracker()
        assert ct.total_cost_usd == 0.0
        assert ct.total_input_tokens == 0
        assert ct.total_output_tokens == 0
        assert ct.cache_hit_tokens == 0

    def test_record_usage(self):
        ct = CostTracker()
        ct.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.0005)
        assert ct.total_input_tokens == 100
        assert ct.total_output_tokens == 50
        assert ct.total_cost_usd == 0.0005
        assert ct.requests_by_model["gpt-4o"] == 1

    def test_multiple_models(self):
        ct = CostTracker()
        ct.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.0005)
        ct.record(model="claude-sonnet-4-20250514", input_tokens=200, output_tokens=100, cost_usd=0.003)
        assert ct.total_input_tokens == 300
        assert ct.total_output_tokens == 150
        assert ct.total_cost_usd == 0.0035
        assert len(ct.cost_by_model) == 2
        assert ct.requests_by_model["gpt-4o"] == 1
        assert ct.requests_by_model["claude-sonnet-4-20250514"] == 1

    def test_with_cache(self):
        ct = CostTracker()
        ct.record(model="claude-sonnet-4-6", input_tokens=1000, output_tokens=50,
                  cost_usd=0.01, cache_hit=500, cache_savings=0.005)
        assert ct.cache_hit_tokens == 500
        assert ct.cache_savings_usd == 0.005

    def test_to_dict(self):
        ct = CostTracker()
        ct.record(model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.0005)
        d = ct.to_dict()
        assert d["total_input_tokens"] == 100
        assert d["total_output_tokens"] == 50
        assert "cost_by_model" in d
        assert "requests_by_model" in d


# ===========================================================================
# TaskCategory
# ===========================================================================

class TestTaskCategory:
    def test_all_categories_exist(self):
        assert TaskCategory.CLASSIFICATION == "classification"
        assert TaskCategory.REASONING == "reasoning"
        assert TaskCategory.CODING == "coding"
        assert TaskCategory.RESEARCH == "research"
        assert TaskCategory.WRITING == "writing"
        assert TaskCategory.PLANNING == "planning"
        assert TaskCategory.EXECUTION == "execution"
        assert TaskCategory.SUMMARIZATION == "summarization"
