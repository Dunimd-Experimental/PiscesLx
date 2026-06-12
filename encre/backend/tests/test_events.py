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

"""Tests for AgentEvent union, BackendEvent union, and factory function variants."""

import pytest

from encre.utils.types import (
    AgentEvent,
    BackendEvent,
    BackendText,
    BackendThinking,
    BackendToolCall,
    BackendToolCallDelta,
    BackendFinish,
    BackendError,
    TextDelta,
    ThinkingDelta,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolProgress,
    ToolResult,
    PermissionRequest,
    Finish,
)


class TestAgentEventUnion:
    """Verify every member type passes isinstance check against AgentEvent."""

    def test_text_delta_in_union(self):
        e = TextDelta(text="hello")
        assert isinstance(e, TextDelta)

    def test_thinking_delta_in_union(self):
        e = ThinkingDelta(text="thinking...")
        assert isinstance(e, ThinkingDelta)

    def test_tool_call_start_in_union(self):
        e = ToolCallStart(name="bash", id="call_1")
        assert isinstance(e, ToolCallStart)

    def test_tool_call_delta_in_union(self):
        e = ToolCallDelta(id="call_1", key="args", value="{}")
        assert isinstance(e, ToolCallDelta)

    def test_tool_call_end_in_union(self):
        e = ToolCallEnd(id="call_1")
        assert isinstance(e, ToolCallEnd)

    def test_tool_progress_in_union(self):
        e = ToolProgress(id="call_1", tool_name="bash", status="running")
        assert isinstance(e, ToolProgress)

    def test_tool_result_in_union(self):
        e = ToolResult(id="call_1", content="output", is_error=False)
        assert isinstance(e, ToolResult)

    def test_permission_request_in_union(self):
        e = PermissionRequest(tool_name="bash", reason="safe")
        assert isinstance(e, PermissionRequest)

    def test_finish_in_union(self):
        e = Finish(reason="stop")
        assert isinstance(e, Finish)


class TestBackendEventUnion:
    """Verify every backend event type passes isinstance check."""

    def test_backend_text_in_union(self):
        e = BackendText(text="hello")
        assert isinstance(e, BackendText)

    def test_backend_thinking_in_union(self):
        e = BackendThinking(text="hmm...", signature_delta=None)
        assert isinstance(e, BackendThinking)

    def test_backend_tool_call_in_union(self):
        e = BackendToolCall(id="c1", name="bash", arguments="{}")
        assert isinstance(e, BackendToolCall)

    def test_backend_tool_call_delta_in_union(self):
        e = BackendToolCallDelta(index=0, key="k", value="v")
        assert isinstance(e, BackendToolCallDelta)

    def test_backend_finish_in_union(self):
        e = BackendFinish(reason="stop")
        assert isinstance(e, BackendFinish)

    def test_backend_error_in_union(self):
        e = BackendError(error="timeout")
        assert isinstance(e, BackendError)

    def test_backend_thinking_with_signature(self):
        e = BackendThinking(text="deep thought", signature_delta="sig123")
        assert e.signature_delta == "sig123"


class TestFactoryFunctionEdgeCases:
    """Test factory functions for edge cases and full kwarg coverage."""

    def test_create_finish_with_usage(self):
        from encre.utils.types import create_finish
        f = create_finish("stop", usage={"prompt_tokens": 10, "completion_tokens": 20})
        assert f.usage == {"prompt_tokens": 10, "completion_tokens": 20}

    def test_create_finish_without_usage(self):
        from encre.utils.types import create_finish
        f = create_finish("error")
        assert f.usage is None

    def test_create_finish_all_reasons(self):
        from encre.utils.types import create_finish
        for reason in ["stop", "tool_calls", "error", "max_tokens", "cancelled"]:
            f = create_finish(reason)
            assert f.reason == reason

    def test_create_text_delta_empty(self):
        from encre.utils.types import create_text_delta
        e = create_text_delta("")
        assert e.text == ""

    def test_create_text_delta_multiline(self):
        from encre.utils.types import create_text_delta
        e = create_text_delta("line1\nline2\nline3")
        assert "line2" in e.text

    def test_create_tool_result_with_error(self):
        from encre.utils.types import create_tool_result
        e = create_tool_result("call_err", "command failed", is_error=True)
        assert e.is_error is True

    def test_create_permission_request(self):
        from encre.utils.types import create_permission_request
        e = create_permission_request("bash", "Running potentially dangerous command")
        assert e.tool_name == "bash"

    def test_create_backend_thinking_with_signature(self):
        from encre.utils.types import create_backend_thinking
        e = create_backend_thinking("deep thoughts", signature_delta="sig_abc")
        assert e.signature_delta == "sig_abc"

    def test_create_backend_thinking_without_signature(self):
        from encre.utils.types import create_backend_thinking
        e = create_backend_thinking("just thinking")
        assert e.signature_delta is None

    def test_create_backend_error_long_message(self):
        from encre.utils.types import create_backend_error
        e = create_backend_error("A" * 1000)
        assert len(e.error) == 1000


class TestFinishReasonVariants:
    """Verify every FinishReason literal works."""

    def test_finish_stop(self):
        f = Finish(reason="stop")
        assert f.reason == "stop"

    def test_finish_tool_calls(self):
        f = Finish(reason="tool_calls")
        assert f.reason == "tool_calls"

    def test_finish_error(self):
        f = Finish(reason="error")
        assert f.reason == "error"

    def test_finish_max_tokens(self):
        f = Finish(reason="max_tokens")
        assert f.reason == "max_tokens"

    def test_finish_cancelled(self):
        f = Finish(reason="cancelled")
        assert f.reason == "cancelled"


class TestToolResultPatterns:
    """Test ToolResult success and error patterns."""

    def test_tool_result_success(self):
        tr = ToolResult(id="t1", content="file contents here", is_error=False)
        assert tr.is_error is False
        assert len(tr.content) > 0

    def test_tool_result_error(self):
        tr = ToolResult(id="t2", content="Permission denied", is_error=True)
        assert tr.is_error is True

    def test_tool_result_empty_content(self):
        tr = ToolResult(id="t3", content="", is_error=False)
        assert tr.content == ""
        assert tr.is_error is False

    def test_tool_result_large_content(self):
        big = "x" * 5000
        tr = ToolResult(id="t4", content=big, is_error=False)
        assert len(tr.content) == 5000
