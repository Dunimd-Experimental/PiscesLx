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

"""Tests for encre.telemetry — agent event recording and session summaries.

Note: test_security_config.py already covers basic ToolCallRecord, TurnRecord,
RetryRecord, record_tool_call, record_turn, record_retry, get_summary, flush,
reset, and disabled telemetry. This file adds edge case and comprehensive tests.
"""

import time

import pytest

from encre.telemetry import EncreTelemetry, ToolCallRecord, TurnRecord, RetryRecord


# ── Edge Cases: Empty Telemetry ──────────────────────────────────────────

class TestEmptyTelemetry:
    def test_get_summary_with_no_data(self):
        tel = EncreTelemetry()
        summary = tel.get_summary()
        assert summary["total_tool_calls"] == 0
        assert summary["total_turns"] == 0
        assert summary["successful_tool_calls"] == 0
        assert summary["failed_tool_calls"] == 0
        assert summary["avg_tool_latency_ms"] == 0.0
        assert summary["avg_turn_latency_ms"] == 0.0
        assert summary["total_events"] == 0
        assert summary["compactions"] == 0
        assert summary["tool_usage"] == {}
        assert summary["total_retries"] == 0
        assert summary["retry_by_error"] == {}

    def test_flush_with_no_data(self):
        tel = EncreTelemetry()
        result = tel.flush()
        assert isinstance(result, dict)
        assert result["total_tool_calls"] == 0


# ── Tool Call Records ────────────────────────────────────────────────────

class TestToolCallRecordFields:
    def test_default_tokens_used(self):
        rec = ToolCallRecord(tool_name="test", latency_ms=100.0, success=True)
        assert rec.tokens_used == 0

    def test_default_error_message(self):
        rec = ToolCallRecord(tool_name="test", latency_ms=100.0, success=True)
        assert rec.error_message == ""

    def test_timestamp_auto_generated(self):
        before = time.time()
        rec = ToolCallRecord(tool_name="test", latency_ms=100.0, success=True)
        after = time.time()
        assert before <= rec.timestamp <= after

    def test_with_error_message(self):
        rec = ToolCallRecord(
            tool_name="bash",
            latency_ms=0.0,
            success=False,
            error_message="command not found",
        )
        assert rec.error_message == "command not found"
        assert rec.success is False


# ── Turn Records ─────────────────────────────────────────────────────────

class TestTurnRecordFields:
    def test_default_compact_triggered(self):
        rec = TurnRecord(turn_number=1, event_count=5, latency_ms=3000.0)
        assert rec.compact_triggered is False

    def test_default_token_usage(self):
        rec = TurnRecord(turn_number=1, event_count=5, latency_ms=3000.0)
        assert rec.token_usage == {}

    def test_with_token_usage(self):
        rec = TurnRecord(
            turn_number=2,
            event_count=10,
            latency_ms=5000.0,
            token_usage={"prompt": 1000, "completion": 200},
        )
        assert rec.token_usage["prompt"] == 1000
        assert rec.token_usage["completion"] == 200

    def test_with_compact_triggered(self):
        rec = TurnRecord(
            turn_number=3,
            event_count=8,
            latency_ms=4000.0,
            compact_triggered=True,
        )
        assert rec.compact_triggered is True


# ── Retry Records ────────────────────────────────────────────────────────

class TestRetryRecordFields:
    def test_timestamp_auto_generated(self):
        before = time.time()
        rec = RetryRecord(attempt=1, error_type="exception", error_detail="timeout", delay_s=2.0)
        after = time.time()
        assert before <= rec.timestamp <= after

    def test_full_record(self):
        rec = RetryRecord(
            attempt=3,
            error_type="http_status",
            error_detail="503 Service Unavailable",
            delay_s=5.0,
        )
        assert rec.attempt == 3
        assert rec.error_type == "http_status"
        assert rec.error_detail == "503 Service Unavailable"
        assert rec.delay_s == 5.0


# ── Comprehensive Summary Tests ──────────────────────────────────────────

class TestSummaryComprehensive:
    def setup_method(self):
        self.tel = EncreTelemetry()

    def test_tool_usage_counts_unique(self):
        self.tel.record_tool_call("bash", 100.0, True)
        self.tel.record_tool_call("bash", 200.0, True)
        self.tel.record_tool_call("edit", 150.0, True)
        self.tel.record_tool_call("grep", 50.0, True)
        self.tel.record_tool_call("bash", 300.0, True)
        summary = self.tel.get_summary()
        assert summary["tool_usage"]["bash"] == 3
        assert summary["tool_usage"]["edit"] == 1
        assert summary["tool_usage"]["grep"] == 1

    def test_successful_vs_failed_counts(self):
        self.tel.record_tool_call("bash", 100.0, True)
        self.tel.record_tool_call("bash", 100.0, False, error_message="fail")
        self.tel.record_tool_call("edit", 100.0, True)
        summary = self.tel.get_summary()
        assert summary["total_tool_calls"] == 3
        assert summary["successful_tool_calls"] == 2
        assert summary["failed_tool_calls"] == 1

    def test_avg_latencies(self):
        self.tel.record_tool_call("a", 100.0, True)
        self.tel.record_tool_call("b", 200.0, True)
        self.tel.record_tool_call("c", 300.0, True)
        summary = self.tel.get_summary()
        assert summary["avg_tool_latency_ms"] == 200.0

    def test_avg_turn_latencies(self):
        self.tel.record_turn(1, 5, 1000.0)
        self.tel.record_turn(2, 3, 3000.0)
        summary = self.tel.get_summary()
        assert summary["avg_turn_latency_ms"] == 2000.0

    def test_total_events(self):
        self.tel.record_turn(1, 5, 1000.0)
        self.tel.record_turn(2, 3, 2000.0)
        self.tel.record_turn(3, 7, 1500.0)
        summary = self.tel.get_summary()
        assert summary["total_events"] == 15

    def test_compactions_count(self):
        self.tel.record_turn(1, 5, 1000.0, compact_triggered=False)
        self.tel.record_turn(2, 3, 2000.0, compact_triggered=True)
        self.tel.record_turn(3, 7, 1500.0, compact_triggered=True)
        summary = self.tel.get_summary()
        assert summary["compactions"] == 2

    def test_session_duration(self):
        tel = EncreTelemetry()
        # Session started at _session_started_at
        summary = tel.get_summary()
        assert summary["session_duration_s"] >= 0.0

    def test_retry_summary(self):
        self.tel.record_retry(1, "http_status", "429", 1.0)
        self.tel.record_retry(2, "http_status", "503", 2.0)
        self.tel.record_retry(1, "exception", "timeout", 3.0)
        summary = self.tel.get_summary()
        assert summary["total_retries"] == 3


# ── Reset Behavior ───────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_all_lists(self):
        tel = EncreTelemetry()
        tel.record_tool_call("bash", 100.0, True)
        tel.record_turn(1, 2, 1000.0)
        tel.record_retry(1, "e", "d", 1.0)
        assert len(tel.tool_calls) == 1
        assert len(tel.turns) == 1
        assert len(tel.retries) == 1

        tel.reset()
        assert len(tel.tool_calls) == 0
        assert len(tel.turns) == 0
        assert len(tel.retries) == 0

    def test_reset_resets_session_start(self):
        tel = EncreTelemetry()
        old_start = tel._session_started_at
        tel.reset()
        assert tel._session_started_at >= old_start

    def test_summary_after_reset_is_empty(self):
        tel = EncreTelemetry()
        tel.record_tool_call("bash", 100.0, True)
        tel.reset()
        summary = tel.get_summary()
        assert summary["total_tool_calls"] == 0
        assert summary["total_turns"] == 0
        assert summary["total_retries"] == 0


# ── Disabled Telemetry ───────────────────────────────────────────────────

class TestDisabledTelemetry:
    def test_record_tool_call_noop(self):
        tel = EncreTelemetry(enabled=False)
        tel.record_tool_call("bash", 100.0, True)
        assert len(tel.tool_calls) == 0

    def test_record_turn_noop(self):
        tel = EncreTelemetry(enabled=False)
        tel.record_turn(1, 2, 1000.0)
        assert len(tel.turns) == 0

    def test_record_retry_noop(self):
        tel = EncreTelemetry(enabled=False)
        tel.record_retry(1, "e", "d", 1.0)
        assert len(tel.retries) == 0

    def test_constructor_default_enabled(self):
        tel = EncreTelemetry()
        assert tel.enabled is True

    def test_constructor_explicitly_disabled(self):
        tel = EncreTelemetry(enabled=False)
        assert tel.enabled is False

    def test_flush_works_when_disabled(self):
        tel = EncreTelemetry(enabled=False)
        result = tel.flush()
        assert isinstance(result, dict)
        assert result["total_tool_calls"] == 0


# ── Timestamp Consistency ────────────────────────────────────────────────

class TestTimestampConsistency:
    def test_records_are_ordered_by_time(self):
        tel = EncreTelemetry()
        tel.record_tool_call("first", 100.0, True)
        tel.record_tool_call("second", 200.0, True)
        tel.record_tool_call("third", 300.0, True)
        timestamps = [t.timestamp for t in tel.tool_calls]
        assert timestamps == sorted(timestamps)

    def test_turn_records_are_ordered(self):
        tel = EncreTelemetry()
        for i in range(5):
            tel.record_turn(i + 1, 2, 1000.0)
        assert len(tel.turns) == 5
        timestamps = [t.timestamp for t in tel.turns]
        assert timestamps == sorted(timestamps)
