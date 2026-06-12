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

"""Tests for EncreSession and SessionCheckpoint."""

import time
import pytest

from encre.config import EncreConfig
from encre.session import EncreSession, SessionCheckpoint


class TestSessionCheckpoint:
    """Verify SessionCheckpoint dataclass."""

    def test_checkpoint_creation(self):
        cp = SessionCheckpoint(
            checkpoint_id="cp_001",
            label="test checkpoint",
            messages=[],
            tool_call_count=0,
            turn_count=0,
        )
        assert cp.checkpoint_id == "cp_001"
        assert cp.label == "test checkpoint"
        assert cp.messages == []
        assert cp.tool_call_count == 0
        assert cp.turn_count == 0

    def test_checkpoint_default_values(self):
        cp = SessionCheckpoint(checkpoint_id="cp_default")
        assert cp.label == ""
        assert cp.messages == []
        assert cp.tool_call_count == 0
        assert cp.turn_count == 0
        assert cp.metadata == {}
        assert cp.created_at == 0.0

    def test_checkpoint_with_data(self):
        msgs = [{"role": "user", "content": "hello"}]
        cp = SessionCheckpoint(
            checkpoint_id="cp_002",
            label="snapshot",
            messages=msgs,
            tool_call_count=3,
            turn_count=2,
            metadata={"key": "value"},
            created_at=1234567890.0,
        )
        assert cp.messages == msgs
        assert cp.tool_call_count == 3
        assert cp.turn_count == 2
        assert cp.metadata == {"key": "value"}
        assert cp.created_at == 1234567890.0


class TestEncreSessionConstruction:
    """Verify EncreSession can be constructed."""

    def test_session_creation(self):
        config = EncreConfig()
        session = EncreSession(config)
        assert session is not None

    def test_session_id_is_string(self):
        config = EncreConfig()
        session = EncreSession(config)
        assert isinstance(session.id, str)
        assert len(session.id) > 0

    def test_session_messages_empty_initially(self):
        config = EncreConfig()
        session = EncreSession(config)
        assert session.messages == []

    def test_session_created_at_is_recent(self):
        config = EncreConfig()
        before = time.time()
        session = EncreSession(config)
        after = time.time()
        assert before - 1 <= session.created_at <= after + 1

    def test_session_updated_at_equals_created_at_initially(self):
        config = EncreConfig()
        session = EncreSession(config)
        # timestamps come from two separate time.time() calls, so allow small delta
        assert abs(session.updated_at - session.created_at) < 0.01

    def test_session_tool_call_count_zero_initially(self):
        config = EncreConfig()
        session = EncreSession(config)
        assert session.tool_call_count == 0

    def test_session_turn_count_zero_initially(self):
        config = EncreConfig()
        session = EncreSession(config)
        assert session.turn_count == 0

    def test_session_config_stored(self):
        config = EncreConfig(max_tokens=9999)
        session = EncreSession(config)
        assert session.config is config
        assert session.config.max_tokens == 9999


class TestEncreSessionMessages:
    """Verify add_message and related methods."""

    def test_add_message(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.add_message("user", "hello world")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "hello world"

    def test_add_multiple_messages(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.add_message("system", "You are helpful.")
        session.add_message("user", "Question")
        session.add_message("assistant", "Answer")
        assert len(session.messages) == 3

    def test_add_message_updates_updated_at(self):
        config = EncreConfig()
        session = EncreSession(config)
        original = session.updated_at
        time.sleep(0.01)
        session.add_message("user", "hi")
        assert session.updated_at > original

    def test_add_message_with_extra_kwargs(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.add_message("assistant", "hello", tool_calls=[{"id": "t1", "name": "bash"}])
        assert "tool_calls" in session.messages[0]

    def test_add_tool_result(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.add_tool_result("call_abc", "ls output", is_error=False)
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "tool"
        assert session.messages[0]["tool_call_id"] == "call_abc"
        assert session.tool_call_count == 1

    def test_add_tool_result_increments_count(self):
        config = EncreConfig()
        session = EncreSession(config)
        assert session.tool_call_count == 0
        session.add_tool_result("t1", "out1")
        session.add_tool_result("t2", "out2")
        assert session.tool_call_count == 2

    def test_add_message_content(self):
        config = EncreConfig()
        session = EncreSession(config)
        blocks = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
        session.add_message_content("user", blocks)
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert isinstance(session.messages[0]["content"], list)


class TestEncreSessionCheckpoints:
    """Verify checkpoint/rollback functionality."""

    def test_checkpoint_creates_id(self):
        config = EncreConfig()
        session = EncreSession(config)
        cid = session.checkpoint("my label")
        assert isinstance(cid, str)
        assert len(cid) > 0

    def test_checkpoint_list_non_empty(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.add_message("user", "important")
        session.checkpoint("snapshot")
        checkpoints = session.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0]["label"] == "snapshot"

    def test_rollback_restores_messages(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.add_message("user", "before checkpoint")
        cid = session.checkpoint("backup")
        session.add_message("assistant", "after checkpoint")
        assert len(session.messages) == 2
        success = session.rollback(cid)
        assert success is True
        assert len(session.messages) == 1
        assert session.messages[0]["content"] == "before checkpoint"

    def test_rollback_nonexistent_returns_false(self):
        config = EncreConfig()
        session = EncreSession(config)
        assert session.rollback("nonexistent") is False

    def test_clear_checkpoints(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.checkpoint("cp1")
        session.checkpoint("cp2")
        assert len(session.list_checkpoints()) == 2
        session.clear_checkpoints()
        assert len(session.list_checkpoints()) == 0


class TestEncreSessionUtility:
    """Verify utility methods: clear, expiry, serialization."""

    def test_clear_history(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.add_message("user", "hello")
        assert len(session.messages) == 1
        session.clear_history()
        assert len(session.messages) == 0
        assert session.tool_call_count == 0
        assert session.turn_count == 0

    def test_is_expired_false_for_new_session(self):
        config = EncreConfig(session_max_age_hours=24.0)
        session = EncreSession(config)
        assert session.is_expired() is False

    def test_is_max_turns_reached(self):
        config = EncreConfig(max_turns=10)
        session = EncreSession(config)
        session.turn_count = 10
        assert session.is_max_turns_reached() is True
        session.turn_count = 5
        assert session.is_max_turns_reached() is False

    def test_to_dict(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.add_message("user", "hello")
        d = session.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == session.id
        assert "messages" in d

    def test_estimate_tokens_static(self):
        count = EncreSession.estimate_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_estimate_tokens_empty(self):
        count = EncreSession.estimate_tokens("")
        assert count == 0

    def test_count_messages_tokens_static(self):
        msgs = [{"role": "user", "content": "hello world"}]
        count = EncreSession.count_messages_tokens(msgs)
        assert isinstance(count, int)
        assert count > 0

    def test_get_context_messages_returns_copy(self):
        config = EncreConfig()
        session = EncreSession(config)
        session.add_message("user", "hi")
        ctx = session.get_context_messages()
        assert len(ctx) == 1
        ctx.append({"role": "assistant", "content": "extra"})
        assert len(session.messages) == 1  # original unchanged

    def test_from_dict_roundtrip(self):
        config = EncreConfig()
        session1 = EncreSession(config)
        session1.add_message("user", "hello")
        data = session1.to_dict()
        session2 = EncreSession.from_dict(data, config)
        assert session2.id == session1.id
        assert len(session2.messages) == 1
        assert session2.messages[0]["content"] == "hello"
