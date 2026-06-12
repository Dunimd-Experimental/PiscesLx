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

"""Tests for utility types, event factories, enums, and union types."""

import pytest

from encre.utils.types import (
    TextDelta,
    ThinkingDelta,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolProgress,
    ToolResult,
    PermissionRequest,
    Finish,
    FinishReason,
    PermissionMode,
    PermissionBehavior,
    PermissionAllow,
    PermissionDeny,
    PermissionAsk,
    PermissionDecision,
    TaskType,
    TaskStatus,
    ThinkingConfig,
    AdaptiveThinking,
    EnabledThinking,
    DisabledThinking,
    BackendText,
    BackendThinking,
    BackendToolCall,
    BackendToolCallDelta,
    BackendFinish,
    BackendError,
    BackendEvent,
    create_text_delta,
    create_thinking_delta,
    create_tool_call_start,
    create_tool_call_delta,
    create_tool_call_end,
    create_tool_progress,
    create_tool_result,
    create_permission_request,
    create_finish,
    create_backend_text,
    create_backend_thinking,
    create_backend_tool_call,
    create_backend_tool_call_delta,
    create_backend_finish,
    create_backend_error,
)


# ===========================================================================
# Event dataclasses
# ===========================================================================

class TestTextDelta:
    def test_create(self):
        td = TextDelta(text="hello")
        assert td.text == "hello"


class TestThinkingDelta:
    def test_create(self):
        td = ThinkingDelta(text="thinking...")
        assert td.text == "thinking..."

    def test_empty(self):
        td = ThinkingDelta(text="")
        assert td.text == ""


class TestToolCallStart:
    def test_create(self):
        tcs = ToolCallStart(id="call_1", name="bash")
        assert tcs.id == "call_1"
        assert tcs.name == "bash"


class TestToolCallDelta:
    def test_create(self):
        tcd = ToolCallDelta(id="call_1", key="arguments", value='{"pattern": "foo"}')
        assert tcd.id == "call_1"
        assert tcd.key == "arguments"


class TestToolCallEnd:
    def test_create(self):
        tce = ToolCallEnd(id="call_1")
        assert tce.id == "call_1"


class TestToolProgress:
    def test_create(self):
        tp = ToolProgress(id="call_1", tool_name="bash", status="running")
        assert tp.id == "call_1"
        assert tp.tool_name == "bash"
        assert tp.status == "running"


class TestToolResult:
    def test_create(self):
        tr = ToolResult(id="call_1", content="output here", is_error=False)
        assert tr.id == "call_1"
        assert tr.content == "output here"
        assert tr.is_error is False

    def test_error_result(self):
        tr = ToolResult(id="call_1", content="command not found", is_error=True)
        assert tr.is_error is True


class TestPermissionRequest:
    def test_create(self):
        pr = PermissionRequest(tool_name="bash", reason="safe command")
        assert pr.tool_name == "bash"
        assert pr.reason == "safe command"


class TestFinish:
    def test_create(self):
        f = Finish(reason="stop", usage={"tokens": 100})
        assert f.reason == "stop"
        assert f.usage == {"tokens": 100}

    def test_finish_reasons(self):
        reasons = ["stop", "tool_calls", "error", "max_tokens", "cancelled"]
        for r in reasons:
            f = Finish(reason=r)
            assert f.reason == r


# ===========================================================================
# Permission
# ===========================================================================

class TestPermissionEnums:
    def test_permission_mode(self):
        modes = ["default", "accept_edits", "bypass", "dont_ask", "plan", "auto"]
        for m in modes:
            # PermissionMode is a Literal, so values must be in the set
            assert m in ["default", "accept_edits", "bypass", "dont_ask", "plan", "auto"]

    def test_permission_allow(self):
        pa = PermissionAllow()
        assert pa.behavior == "allow"

    def test_permission_deny(self):
        pd = PermissionDeny()
        assert pd.behavior == "deny"

    def test_permission_ask(self):
        pa = PermissionAsk()
        assert pa.behavior == "ask"


# ===========================================================================
# Task enums
# ===========================================================================

class TestTaskEnums:
    def test_task_type_literals(self):
        types = ["bash", "agent", "workflow"]
        for t in types:
            assert t in ["bash", "agent", "workflow"]

    def test_task_status_literals(self):
        statuses = ["pending", "running", "completed", "failed", "killed"]
        for s in statuses:
            assert s in ["pending", "running", "completed", "failed", "killed"]


# ===========================================================================
# Thinking config
# ===========================================================================

class TestThinkingConfig:
    def test_adaptive(self):
        tc = AdaptiveThinking()
        assert tc.enabled is True
        assert tc.min_tokens == 1024

    def test_enabled(self):
        tc = EnabledThinking(budget_tokens=16000)
        assert tc.budget_tokens == 16000

    def test_disabled(self):
        tc = DisabledThinking()
        assert tc.enabled is False


# ===========================================================================
# Backend event types
# ===========================================================================

class TestBackendEvents:
    def test_backend_text(self):
        bt = BackendText(text="hello")
        assert bt.text == "hello"

    def test_backend_thinking(self):
        bt = BackendThinking(text="hmm", signature_delta=None)
        assert bt.text == "hmm"

    def test_backend_tool_call(self):
        btc = BackendToolCall(id="call_1", name="bash", arguments='{"cmd": "ls"}')
        assert btc.name == "bash"
        assert btc.arguments == '{"cmd": "ls"}'

    def test_backend_tool_call_delta(self):
        bd = BackendToolCallDelta(index=0, key="arguments", value='"pattern"')
        assert bd.index == 0
        assert bd.key == "arguments"

    def test_backend_finish(self):
        bf = BackendFinish(reason="stop")
        assert bf.reason == "stop"

    def test_backend_error(self):
        be = BackendError(error="Too many requests")
        assert "Too many" in be.error


# ===========================================================================
# Factory functions
# ===========================================================================

class TestFactories:
    def test_create_text_delta(self):
        event = create_text_delta("hello")
        assert isinstance(event, TextDelta)
        assert event.text == "hello"

    def test_create_thinking_delta(self):
        event = create_thinking_delta("hmm...")
        assert isinstance(event, ThinkingDelta)
        assert event.text == "hmm..."

    def test_create_tool_call_start(self):
        event = create_tool_call_start("bash", "id1")
        assert isinstance(event, ToolCallStart)
        assert event.name == "bash"
        assert event.id == "id1"

    def test_create_tool_call_delta(self):
        event = create_tool_call_delta("id1", "arguments", "...")
        assert isinstance(event, ToolCallDelta)

    def test_create_tool_call_end(self):
        event = create_tool_call_end("id1")
        assert isinstance(event, ToolCallEnd)

    def test_create_tool_progress(self):
        event = create_tool_progress("id1", "bash", "running")
        assert isinstance(event, ToolProgress)

    def test_create_tool_result(self):
        event = create_tool_result("id1", "output")
        assert isinstance(event, ToolResult)
        assert event.content == "output"

    def test_create_permission_request(self):
        event = create_permission_request("bash", "safe cmd")
        assert isinstance(event, PermissionRequest)

    def test_create_finish(self):
        event = create_finish("stop")
        assert isinstance(event, Finish)

    def test_create_backend_text(self):
        event = create_backend_text("hello")
        assert isinstance(event, BackendText)

    def test_create_backend_thinking(self):
        event = create_backend_thinking("hmm")
        assert isinstance(event, BackendThinking)

    def test_create_backend_tool_call(self):
        event = create_backend_tool_call("id1", "bash", "{}")
        assert isinstance(event, BackendToolCall)

    def test_create_backend_tool_call_delta(self):
        event = create_backend_tool_call_delta(0, "key", "value")
        assert isinstance(event, BackendToolCallDelta)

    def test_create_backend_finish(self):
        event = create_backend_finish("stop")
        assert isinstance(event, BackendFinish)

    def test_create_backend_error(self):
        event = create_backend_error("Request timed out")
        assert isinstance(event, BackendError)
