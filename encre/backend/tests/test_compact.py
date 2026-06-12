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

"""Tests for compaction subsystem: strategies, semantic compactor, context partitioner."""

import asyncio

import pytest

from encre.compact.engine import EncreCompactEngine
from encre.compact.strategies import (
    EncreAlwaysCompactStrategy,
    EncreAutoCompactStrategy,
    EncreTokenBudgetStrategy,
    EncreBudgetReductionStrategy,
    EncreSemanticCompactStrategy,
    EncreSnipStrategy,
    EncreMicroCompactStrategy,
    EncreContextCollapseStrategy,
    EncreMultiStagePipeline,
)
from encre.compact.semantic import (
    SemanticToolOutputCompactor,
    ContextPartitioner,
    ContextPartition,
    ContextTier,
)


def _m(role, content, name=None):
    msg = {"role": role, "content": content}
    if name:
        msg["name"] = name
    return msg


def _make_messages(turns):
    msgs = [_m("system", "You are an assistant.")]
    for i in range(turns):
        msgs.append(_m("user", f"Question {i}"))
        msgs.append(_m("assistant", f"Answer {i}"))
    return msgs


# ===========================================================================
# ContextTier / ContextPartition
# ===========================================================================

class TestContextTier:
    def test_create(self):
        ct = ContextTier(name="test", messages=[_m("user", "hello")])
        assert ct.name == "test"
        assert len(ct.messages) == 1

    def test_token_count(self):
        ct = ContextTier(name="test", messages=[_m("user", "hello world")])
        assert ct.token_count() > 0


class TestContextPartition:
    def test_defaults(self):
        cp = ContextPartition()
        assert cp.system == []
        assert cp.hot == []
        assert cp.warm == []
        assert cp.cold == []
        assert cp.reference == []

    def test_with_messages(self):
        cp = ContextPartition(
            system=[_m("system", "You are helpful.")],
            hot=[_m("user", "latest question")],
        )
        msgs = cp.to_messages()
        assert len(msgs) == 2

    def test_total_tokens(self):
        cp = ContextPartition(hot=[_m("user", "hello world")])
        assert cp.total_tokens() > 0


class TestContextPartitioner:
    def test_partition(self):
        partitioner = ContextPartitioner()
        messages = [
            _m("system", "You are an assistant."),
            _m("user", "Hello"),
            _m("assistant", "Hi there"),
            _m("user", "Can you help me?"),
            _m("assistant", "Sure, what do you need?"),
        ]
        result = partitioner.partition(messages)
        assert isinstance(result, ContextPartition)
        assert len(result.hot) > 0
        assert len(result.system) == 1


# ===========================================================================
# SemanticToolOutputCompactor
# ===========================================================================

class TestSemanticToolOutputCompactor:
    def setup_method(self):
        self.compactor = SemanticToolOutputCompactor()

    def test_compact_grep(self):
        big = "file.py:1:line1\nfile.py:2:line2\n" * 600
        result = self.compactor.compact_tool_output("grep", big)
        assert len(result) < len(big)

    def test_compact_glob(self):
        big = "\n".join(f"/path/to/file{i}.py" for i in range(800))
        result = self.compactor.compact_tool_output("glob", big)
        assert "files" in result.lower() or "glob" in result.lower()

    def test_compact_bash(self):
        big = "error line 1\n" * 700
        result = self.compactor.compact_tool_output("bash", big)
        assert len(result) < len(big)

    def test_compact_file_read(self):
        big = "def foo():\n    pass\n" * 500
        result = self.compactor.compact_tool_output("file_read", big)
        assert len(result) < len(big)

    def test_compact_web_fetch(self):
        html = "<html><head><title>Test</title></head><body>" + "<p>content</p>" * 600 + "</body></html>"
        result = self.compactor.compact_tool_output("web_fetch", html)
        assert len(result) < len(html)

    def test_compact_task_list(self):
        big = '{"id": "1", "subject": "test"}\n' * 20
        result = self.compactor.compact_tool_output("task_list", big)
        assert len(result) < 700

    def test_compact_unknown_truncates(self):
        big = "x" * 10000
        result = self.compactor.compact_tool_output("unknown_tool", big)
        assert len(result) <= 10000

    def test_short_output_passthrough(self):
        short = "short output"
        result = self.compactor.compact_tool_output("grep", short)
        assert result == short


# ===========================================================================
# Compaction Strategies
# ===========================================================================

class TestCompactionStrategies:
    def test_always_compact_should(self):
        s = EncreAlwaysCompactStrategy()
        assert asyncio.run(s.should_compact(_make_messages(8), 128000)) is True

    def test_always_compact_few(self):
        s = EncreAlwaysCompactStrategy()
        assert asyncio.run(s.should_compact(_make_messages(2), 128000)) is False

    def test_always_compact_execute(self):
        s = EncreAlwaysCompactStrategy()
        msgs = _make_messages(8)
        result = asyncio.run(s.compact(msgs, 128000))
        assert len(result) <= len(msgs)

    def test_token_budget_should(self):
        s = EncreTokenBudgetStrategy(budget_ratio=0.5)
        msgs = [_m("user", "x" * 10000)]
        assert isinstance(asyncio.run(s.should_compact(msgs, 1000)), bool)

    def test_token_budget_execute(self):
        s = EncreTokenBudgetStrategy(budget_ratio=0.5)
        msgs = _make_messages(20)
        result = asyncio.run(s.compact(msgs, 128000))
        assert len(result) <= len(msgs)

    def test_budget_reduction_execute(self):
        s = EncreBudgetReductionStrategy(max_chars_per_message=100)
        msgs = [_m("user", "x" * 5000)]
        result = asyncio.run(s.compact(msgs, 128000))
        assert len(result[0]["content"]) < 5000  # was truncated

    def test_budget_reduction_should(self):
        s = EncreBudgetReductionStrategy(max_chars_per_message=100)
        msgs = [_m("user", "x" * 5000)]
        assert asyncio.run(s.should_compact(msgs, 128000)) is True

    def test_snip_execute(self):
        s = EncreSnipStrategy(keep_recent_turns=3)
        msgs = _make_messages(20)
        result = asyncio.run(s.compact(msgs, 128000))
        assert len(result) <= len(msgs)

    def test_micro_compact_execute(self):
        s = EncreMicroCompactStrategy()
        msgs = _make_messages(10)
        result = asyncio.run(s.compact(msgs, 128000))
        assert result is not None

    def test_micro_compact_large_content(self):
        s = EncreMicroCompactStrategy()
        msgs = [_m("user", "x" * 5000)]
        assert asyncio.run(s.should_compact(msgs, 128000)) is True

    def test_context_collapse_execute(self):
        s = EncreContextCollapseStrategy()
        msgs = _make_messages(20)
        result = asyncio.run(s.compact(msgs, 128000))
        assert result is not None

    def test_semantic_should(self):
        s = EncreSemanticCompactStrategy()
        msgs = [_m("tool", "x" * 10000)]
        assert isinstance(asyncio.run(s.should_compact(msgs, 128000)), bool)

    def test_semantic_execute(self):
        s = EncreSemanticCompactStrategy()
        msgs = [_m("user", "test")]
        result = asyncio.run(s.compact(msgs, 128000))
        assert result is not None

    def test_multi_stage_has_six_stages(self):
        pipeline = EncreMultiStagePipeline()
        assert len(pipeline._stages) >= 6

    def test_multi_stage_execute(self):
        pipeline = EncreMultiStagePipeline()
        msgs = _make_messages(5)
        result = asyncio.run(pipeline.compact(msgs, 128000))
        assert result is not None

    def test_multi_stage_should(self):
        pipeline = EncreMultiStagePipeline()
        assert isinstance(asyncio.run(pipeline.should_compact(_make_messages(2), 128000)), bool)

    def test_auto_compact_strategy(self):
        s = EncreAutoCompactStrategy(threshold_ratio=0.5)
        msgs = [_m("user", "x" * 50000)]
        assert isinstance(asyncio.run(s.should_compact(msgs, 1000)), bool)

    def test_auto_compact_execute(self):
        s = EncreAutoCompactStrategy(threshold_ratio=0.5)
        msgs = _make_messages(2)
        result = asyncio.run(s.compact(msgs, 128000))
        assert result is not None


# ===========================================================================
# EncreCompactEngine
# ===========================================================================

class TestCompactEngine:
    def test_create(self):
        engine = EncreCompactEngine()
        assert engine is not None

    def test_with_strategy(self):
        s = EncreAlwaysCompactStrategy()
        engine = EncreCompactEngine(strategy=s)
        assert engine is not None

    def test_should_compact(self):
        engine = EncreCompactEngine()
        msgs = _make_messages(2)
        assert isinstance(asyncio.run(engine.should_compact(msgs, 128000)), bool)

    def test_compact(self):
        engine = EncreCompactEngine()
        msgs = _make_messages(30)
        result = asyncio.run(engine.compact(msgs, 128000))
        assert result is not None

    def test_set_strategy(self):
        engine = EncreCompactEngine()
        s = EncreAlwaysCompactStrategy()
        engine.set_strategy(s)
        assert engine._strategy is s
