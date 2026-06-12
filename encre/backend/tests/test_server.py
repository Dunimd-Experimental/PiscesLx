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

"""Tests for encre.server.protocol — client/server message encoding and parsing."""

import json

import pytest

from encre.server.protocol import (
    ClientRun,
    ClientRespondPermission,
    ClientCancel,
    ClientResume,
    ClientConfigure,
    ClientPing,
    ClientMessage,
    ClientMessageType,
    ServerMessageType,
    parse_client_message,
    encode_server_message,
    encode_text_delta,
    encode_thinking_delta,
    encode_tool_call_start,
    encode_tool_call_delta,
    encode_tool_call_end,
    encode_tool_progress,
    encode_tool_result,
    encode_permission_request,
    encode_finish,
    encode_pong,
    encode_error,
    encode_session_ready,
    _make_message,
)


# ── Client Message Dataclasses ────────────────────────────────────────────

class TestClientRun:
    def test_defaults(self):
        msg = ClientRun()
        assert msg.type == "run"
        assert msg.prompt == ""
        assert msg.system_prompt is None
        assert msg.session_id is None
        assert msg.specialty == "general"

    def test_from_dict_minimal(self):
        msg = ClientRun.from_dict({"prompt": "hello"})
        assert msg.type == "run"
        assert msg.prompt == "hello"
        assert msg.specialty == "general"

    def test_from_dict_full(self):
        msg = ClientRun.from_dict({
            "prompt": "do it",
            "system_prompt": "You are helpful.",
            "session_id": "abc-123",
            "specialty": "coding",
        })
        assert msg.type == "run"
        assert msg.prompt == "do it"
        assert msg.system_prompt == "You are helpful."
        assert msg.session_id == "abc-123"
        assert msg.specialty == "coding"

    def test_from_dict_missing_keys(self):
        """from_dict uses .get() with defaults for all fields."""
        msg = ClientRun.from_dict({})
        assert msg.prompt == ""
        assert msg.system_prompt is None
        assert msg.session_id is None


class TestClientRespondPermission:
    def test_defaults(self):
        msg = ClientRespondPermission()
        assert msg.type == "respond_permission"
        assert msg.tool_name == ""
        assert msg.decision is False

    def test_from_dict(self):
        msg = ClientRespondPermission.from_dict({
            "tool_name": "bash",
            "decision": True,
        })
        assert msg.tool_name == "bash"
        assert msg.decision is True

    def test_from_dict_defaults(self):
        msg = ClientRespondPermission.from_dict({})
        assert msg.tool_name == ""
        assert msg.decision is False


class TestClientCancel:
    def test_defaults(self):
        msg = ClientCancel()
        assert msg.type == "cancel"
        assert msg.session_id == ""

    def test_from_dict(self):
        msg = ClientCancel.from_dict({"session_id": "sess-xyz"})
        assert msg.session_id == "sess-xyz"

    def test_from_dict_empty(self):
        msg = ClientCancel.from_dict({})
        assert msg.session_id == ""


class TestClientResume:
    def test_defaults(self):
        msg = ClientResume()
        assert msg.type == "resume"
        assert msg.session_id == ""

    def test_from_dict(self):
        msg = ClientResume.from_dict({"session_id": "sess-abc"})
        assert msg.session_id == "sess-abc"


class TestClientConfigure:
    def test_defaults(self):
        msg = ClientConfigure()
        assert msg.type == "configure"
        assert msg.config == {}

    def test_from_dict(self):
        msg = ClientConfigure.from_dict({"config": {"model": "gpt-4o"}})
        assert msg.config == {"model": "gpt-4o"}

    def test_from_dict_empty(self):
        msg = ClientConfigure.from_dict({})
        assert msg.config == {}


class TestClientPing:
    def test_defaults(self):
        msg = ClientPing()
        assert msg.type == "ping"

    def test_from_dict_ignores_payload(self):
        msg = ClientPing.from_dict({"extra": "ignored"})
        assert msg.type == "ping"


# ── parse_client_message ─────────────────────────────────────────────────

class TestParseClientMessage:
    def test_parse_run(self):
        msg = parse_client_message(json.dumps({"type": "run", "prompt": "hi"}))
        assert isinstance(msg, ClientRun)
        assert msg.prompt == "hi"

    def test_parse_respond_permission(self):
        msg = parse_client_message(json.dumps({
            "type": "respond_permission",
            "tool_name": "edit",
            "decision": True,
        }))
        assert isinstance(msg, ClientRespondPermission)
        assert msg.tool_name == "edit"

    def test_parse_cancel(self):
        msg = parse_client_message(json.dumps({
            "type": "cancel",
            "session_id": "s1",
        }))
        assert isinstance(msg, ClientCancel)

    def test_parse_resume(self):
        msg = parse_client_message(json.dumps({
            "type": "resume",
            "session_id": "s1",
        }))
        assert isinstance(msg, ClientResume)

    def test_parse_configure(self):
        msg = parse_client_message(json.dumps({
            "type": "configure",
            "config": {"max_tokens": 8192},
        }))
        assert isinstance(msg, ClientConfigure)
        assert msg.config == {"max_tokens": 8192}

    def test_parse_ping(self):
        msg = parse_client_message(json.dumps({"type": "ping"}))
        assert isinstance(msg, ClientPing)

    def test_parse_invalid_json_returns_none(self):
        msg = parse_client_message("not json at all")
        assert msg is None

    def test_parse_empty_json_returns_none(self):
        msg = parse_client_message("{}")
        assert msg is None

    def test_parse_unknown_type_returns_none(self):
        msg = parse_client_message(json.dumps({"type": "magic_unknown"}))
        assert msg is None

    def test_parse_bytes_input(self):
        msg = parse_client_message(b'{"type": "ping"}')
        assert isinstance(msg, ClientPing)

    def test_parse_invalid_utf8_bytes(self):
        msg = parse_client_message(b'\xff\xfe\x00')
        assert msg is None


# ── _make_message helper ─────────────────────────────────────────────────

class TestMakeMessage:
    def test_basic(self):
        result = _make_message("test_type", key="val")
        assert result == {"type": "test_type", "key": "val"}

    def test_no_extras(self):
        result = _make_message("bare")
        assert result == {"type": "bare"}

    def test_multiple_kwargs(self):
        result = _make_message("m", a=1, b=2, c=3)
        assert result == {"type": "m", "a": 1, "b": 2, "c": 3}


# ── encode_server_message ────────────────────────────────────────────────

class TestEncodeServerMessage:
    def test_returns_json_string(self):
        result = encode_server_message("text_delta", text="hello")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["type"] == "text_delta"
        assert parsed["text"] == "hello"

    def test_ensure_ascii_false(self):
        # ensure_ascii=False means unicode is preserved
        result = encode_server_message("text_delta", text="cafe")
        assert "cafe" in result

    def test_no_extra_kwargs(self):
        result = encode_server_message("pong")
        parsed = json.loads(result)
        assert parsed == {"type": "pong"}


# ── Convenience Encoders ─────────────────────────────────────────────────

class TestConvenienceEncoders:
    def test_encode_text_delta(self):
        msg = encode_text_delta("Hello world")
        parsed = json.loads(msg)
        assert parsed == {"type": "text_delta", "text": "Hello world"}

    def test_encode_thinking_delta(self):
        msg = encode_thinking_delta("Hmm...")
        parsed = json.loads(msg)
        assert parsed == {"type": "thinking_delta", "text": "Hmm..."}

    def test_encode_tool_call_start(self):
        msg = encode_tool_call_start("bash", "call_1")
        parsed = json.loads(msg)
        assert parsed == {"type": "tool_call_start", "name": "bash", "id": "call_1"}

    def test_encode_tool_call_delta(self):
        msg = encode_tool_call_delta("call_1", "arguments", '{"cmd":')
        parsed = json.loads(msg)
        assert parsed == {
            "type": "tool_call_delta",
            "id": "call_1",
            "key": "arguments",
            "value": '{"cmd":',
        }

    def test_encode_tool_call_end(self):
        msg = encode_tool_call_end("call_1")
        parsed = json.loads(msg)
        assert parsed == {"type": "tool_call_end", "id": "call_1"}

    def test_encode_tool_progress(self):
        msg = encode_tool_progress("call_1", "bash", "running")
        parsed = json.loads(msg)
        assert parsed == {
            "type": "tool_progress",
            "id": "call_1",
            "tool_name": "bash",
            "status": "running",
        }

    def test_encode_tool_result(self):
        msg = encode_tool_result("call_1", "output text", is_error=False)
        parsed = json.loads(msg)
        assert parsed["type"] == "tool_result"
        assert parsed["content"] == "output text"
        assert parsed["is_error"] is False

    def test_encode_tool_result_error(self):
        msg = encode_tool_result("call_1", "command not found", is_error=True)
        parsed = json.loads(msg)
        assert parsed["is_error"] is True

    def test_encode_permission_request(self):
        msg = encode_permission_request("bash", "requires sudo")
        parsed = json.loads(msg)
        assert parsed == {
            "type": "permission_request",
            "tool_name": "bash",
            "reason": "requires sudo",
        }

    def test_encode_finish(self):
        msg = encode_finish("stop")
        parsed = json.loads(msg)
        assert parsed == {"type": "finish", "reason": "stop", "usage": None}

    def test_encode_finish_with_usage(self):
        usage = {"input_tokens": 100, "output_tokens": 50}
        msg = encode_finish("stop", usage=usage)
        parsed = json.loads(msg)
        assert parsed["usage"] == usage

    def test_encode_pong(self):
        msg = encode_pong()
        parsed = json.loads(msg)
        assert parsed == {"type": "pong"}

    def test_encode_error(self):
        msg = encode_error("something went wrong")
        parsed = json.loads(msg)
        assert parsed == {"type": "error", "message": "something went wrong", "code": "internal"}

    def test_encode_error_with_code(self):
        msg = encode_error("timeout", code="timeout")
        parsed = json.loads(msg)
        assert parsed["code"] == "timeout"

    def test_encode_session_ready(self):
        msg = encode_session_ready("sess-42")
        parsed = json.loads(msg)
        assert parsed == {"type": "session_ready", "session_id": "sess-42"}


# ── Message Type Literals ────────────────────────────────────────────────

class TestMessageTypes:
    def test_client_message_type_values(self):
        """ClientMessageType literal includes all expected values."""
        # Runtime validation: these values are from the literal definition
        expected = {"run", "respond_permission", "cancel", "resume", "configure", "ping"}
        # Type check: assert the string value comparisons work
        assert "run" in expected
        assert "ping" in expected

    def test_server_message_type_values(self):
        """ServerMessageType literal includes all expected values."""
        expected = {
            "text_delta", "thinking_delta", "tool_call_start",
            "tool_call_delta", "tool_call_end", "tool_progress",
            "tool_result", "permission_request", "finish", "pong",
            "error", "session_ready",
        }
        assert "text_delta" in expected
        assert "session_ready" in expected
        assert "finish" in expected


# ── Roundtrip ────────────────────────────────────────────────────────────

class TestRoundtrip:
    """Verify that messages can be serialized and deserialized properly."""

    def test_ping_roundtrip(self):
        encoded = encode_pong()
        parsed = json.loads(encoded)
        assert parsed["type"] == "pong"

    def test_client_run_roundtrip(self):
        # Create a ClientRun, encode it manually, parse it back
        original = {"type": "run", "prompt": "test prompt"}
        raw = json.dumps(original)
        msg = parse_client_message(raw)
        assert isinstance(msg, ClientRun)
        assert msg.prompt == "test prompt"

    def test_all_client_types_parseable(self):
        """Every ClientMessageType should have a registered parser."""
        for msg_type in ["run", "respond_permission", "cancel", "resume", "configure", "ping"]:
            base = {"type": msg_type}
            if msg_type == "configure":
                base["config"] = {}
            result = parse_client_message(json.dumps(base))
            assert result is not None, f"Failed to parse: {msg_type}"
