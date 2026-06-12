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

"""Tests for the recovery system: ErrorRecoveryEngine, RetryableExecutor,
RecoveryState, and correction inference."""

import asyncio

import pytest

from encre.recovery import (
    ErrorRecoveryEngine,
    RetryableExecutor,
    RecoveryState,
    RecoveryDecision,
    RecoveryAction,
    ErrorCategory,
    classify_error,
    compute_backoff,
    RecoveryConfig,
)


# ===========================================================================
# ErrorCategory
# ===========================================================================

class TestErrorCategory:
    def test_all_categories_exist(self):
        assert hasattr(ErrorCategory, "TRANSIENT")
        assert hasattr(ErrorCategory, "INPUT_MALFORMED")
        assert hasattr(ErrorCategory, "TOOL_NOT_FOUND")
        assert hasattr(ErrorCategory, "PERMISSION_DENIED")
        assert hasattr(ErrorCategory, "TOOL_EXECUTION")
        assert hasattr(ErrorCategory, "SANDBOX_ERROR")
        assert hasattr(ErrorCategory, "MODEL_ERROR")
        assert hasattr(ErrorCategory, "UNKNOWN")


class TestClassifyError:
    def test_transient_timeout(self):
        assert classify_error("ConnectionError: timeout") == ErrorCategory.TRANSIENT

    def test_transient_rate_limit(self):
        assert classify_error("Error: rate limit exceeded") == ErrorCategory.TRANSIENT

    def test_permission_denied(self):
        assert classify_error("PermissionError: permission denied") == ErrorCategory.PERMISSION_DENIED

    def test_permission_blocked_by_hook(self):
        assert classify_error("blocked by hook") == ErrorCategory.PERMISSION_DENIED

    def test_model_error(self):
        assert classify_error("APIError: api error in backend") == ErrorCategory.MODEL_ERROR

    def test_input_malformed(self):
        assert classify_error("Error: invalid json in payload") == ErrorCategory.INPUT_MALFORMED

    def test_tool_not_found(self):
        assert classify_error("Error: unknown tool 'fake_tool'") == ErrorCategory.TOOL_NOT_FOUND

    def test_sandbox_error(self):
        assert classify_error("sandbox failure") == ErrorCategory.SANDBOX_ERROR

    def test_unknown(self):
        assert classify_error("SomeWeirdError: unknown failure") == ErrorCategory.UNKNOWN


# ===========================================================================
# RecoveryAction
# ===========================================================================

class TestRecoveryAction:
    def test_all_actions_exist(self):
        assert hasattr(RecoveryAction, "RETRY")
        assert hasattr(RecoveryAction, "RETRY_WITH_BACKOFF")
        assert hasattr(RecoveryAction, "RETRY_WITH_FIXED_ARGS")
        assert hasattr(RecoveryAction, "FALLBACK_TOOL")
        assert hasattr(RecoveryAction, "DEGRADE")
        assert hasattr(RecoveryAction, "SKIP")
        assert hasattr(RecoveryAction, "ABORT_TURN")
        assert hasattr(RecoveryAction, "ABORT_SESSION")


# ===========================================================================
# RecoveryState
# ===========================================================================

class TestRecoveryState:
    def test_initial_state(self):
        state = RecoveryState(tool_name="test_tool", tool_args={"path": "/tmp/test"})
        assert state.tool_name == "test_tool"
        assert state.tool_args == {"path": "/tmp/test"}
        assert state.attempts == 0
        assert state.recovery_history == []
        assert state.succeeded is False
        assert state.last_error == ""
        assert state.last_category == ErrorCategory.UNKNOWN

    def test_record_decision(self):
        state = RecoveryState(tool_name="test_tool", tool_args={})
        decision = RecoveryDecision(action=RecoveryAction.RETRY, reason="try again")
        state.recovery_history.append(decision)
        assert len(state.recovery_history) == 1
        assert state.recovery_history[0].action == RecoveryAction.RETRY

    def test_with_error_info(self):
        state = RecoveryState(
            tool_name="test_tool",
            tool_args={"x": 1},
            last_error="something broke",
            last_category=ErrorCategory.TRANSIENT,
        )
        assert state.last_error == "something broke"
        assert state.last_category == ErrorCategory.TRANSIENT


# ===========================================================================
# RecoveryDecision
# ===========================================================================

class TestRecoveryDecision:
    def test_retry_decision(self):
        d = RecoveryDecision(action=RecoveryAction.RETRY, reason="timeout", retry_delay_seconds=1.0)
        assert d.action == RecoveryAction.RETRY
        assert d.reason == "timeout"
        assert d.retry_delay_seconds == 1.0

    def test_fallback_decision(self):
        d = RecoveryDecision(
            action=RecoveryAction.FALLBACK_TOOL,
            reason="tool unavailable",
            fallback_tool="alternative_tool",
        )
        assert d.fallback_tool == "alternative_tool"

    def test_fixed_args_decision(self):
        d = RecoveryDecision(
            action=RecoveryAction.RETRY_WITH_FIXED_ARGS,
            reason="bad arg",
            modified_args={"arg1": "fixed"},
        )
        assert d.modified_args == {"arg1": "fixed"}

    def test_abort_decision(self):
        d = RecoveryDecision(action=RecoveryAction.ABORT_TURN, reason="unrecoverable")
        assert d.action == RecoveryAction.ABORT_TURN


# ===========================================================================
# ErrorRecoveryEngine
# ===========================================================================

class TestErrorRecoveryEngine:
    def test_decide_transient_first_attempt(self):
        engine = ErrorRecoveryEngine()
        decision = engine.decide(
            tool_name="file_read",
            tool_args={"path": "/tmp/test"},
            error_message="ConnectionError: timeout",
            attempt=0,
        )
        assert decision.action == RecoveryAction.RETRY_WITH_BACKOFF
        assert decision.retry_delay_seconds > 0

    def test_decide_permission_skip(self):
        engine = ErrorRecoveryEngine()
        decision = engine.decide(
            tool_name="file_read",
            tool_args={"path": "/tmp/test"},
            error_message="PermissionError: permission denied",
            attempt=0,
        )
        assert decision.action == RecoveryAction.SKIP

    def test_decide_max_retries_with_fallback(self):
        engine = ErrorRecoveryEngine()
        decision = engine.decide(
            tool_name="file_read",
            tool_args={"path": "/tmp/test"},
            error_message="RuntimeError: something broke",
            attempt=3,
            fallback_depth=0,
        )
        # file_read has "bash" as fallback at depth 0
        assert decision.action in (RecoveryAction.FALLBACK_TOOL, RecoveryAction.SKIP)

    def test_decide_max_retries_no_fallback_chain(self):
        engine = ErrorRecoveryEngine()
        decision = engine.decide(
            tool_name="nonexistent_tool",
            tool_args={},
            error_message="RuntimeError: persistent failure",
            attempt=5,
        )
        assert decision.action in (RecoveryAction.SKIP, RecoveryAction.ABORT_TURN)

    def test_decide_tool_not_found(self):
        engine = ErrorRecoveryEngine()
        decision = engine.decide(
            tool_name="nonexistent",
            tool_args={},
            error_message="Error: unknown tool 'nonexistent'",
            attempt=0,
        )
        assert decision.action == RecoveryAction.ABORT_TURN

    def test_decide_model_error(self):
        engine = ErrorRecoveryEngine()
        decision = engine.decide(
            tool_name="file_read",
            tool_args={"path": "/tmp/test"},
            error_message="APIError: internal server error",
            attempt=0,
        )
        assert decision.action == RecoveryAction.RETRY_WITH_BACKOFF

    def test_decide_input_malformed_first_attempt(self):
        engine = ErrorRecoveryEngine()
        decision = engine.decide(
            tool_name="file_write",
            tool_args={"path": "/tmp/test"},
            error_message="Error: invalid json in payload",
            attempt=0,
        )
        assert decision.action == RecoveryAction.RETRY_WITH_FIXED_ARGS

    def test_decide_input_malformed_second_attempt(self):
        engine = ErrorRecoveryEngine()
        decision = engine.decide(
            tool_name="file_write",
            tool_args={"path": "/tmp/test"},
            error_message="Error: invalid json in payload",
            attempt=1,
        )
        assert decision.action == RecoveryAction.ABORT_TURN

    def test_session_error_limit(self):
        engine = ErrorRecoveryEngine()
        engine._session_errors = 10  # reaches _max_session_errors
        decision = engine.decide(
            tool_name="test",
            tool_args={},
            error_message="any error",
            attempt=0,
        )
        assert decision.action == RecoveryAction.ABORT_SESSION

    def test_reset_session(self):
        engine = ErrorRecoveryEngine()
        engine._session_errors = 5
        engine._last_errors = ["e1", "e2", "e3"]
        engine.reset_session()
        assert engine._session_errors == 0
        assert engine._last_errors == []

    def test_infer_correction_fallback(self):
        state = RecoveryState(tool_name="bad_tool", tool_args={})
        state.recovery_history = [
            RecoveryDecision(action=RecoveryAction.FALLBACK_TOOL, reason="unavailable", fallback_tool="good_tool"),
        ]
        correction = ErrorRecoveryEngine.infer_correction(state)
        assert "good_tool" in correction

    def test_infer_correction_fixed_args(self):
        state = RecoveryState(tool_name="test", tool_args={})
        state.recovery_history = [
            RecoveryDecision(
                action=RecoveryAction.RETRY_WITH_FIXED_ARGS,
                reason="bad arg",
                modified_args={"x": 1},
            ),
        ]
        correction = ErrorRecoveryEngine.infer_correction(state)
        assert "modified args" in correction.lower()

    def test_infer_correction_empty_history(self):
        state = RecoveryState(tool_name="test", tool_args={})
        assert ErrorRecoveryEngine.infer_correction(state) == ""

    def test_infer_correction_from_history(self):
        history = [
            RecoveryDecision(action=RecoveryAction.RETRY_WITH_BACKOFF, reason="timeout", retry_delay_seconds=5.0),
        ]
        correction = ErrorRecoveryEngine.infer_correction_from_history(history, "test_tool")
        assert "backoff" in correction.lower()

    def test_infer_correction_from_history_empty(self):
        assert ErrorRecoveryEngine.infer_correction_from_history([], "test") == ""

    def test_infer_correction_from_history_last_wins(self):
        history = [
            RecoveryDecision(action=RecoveryAction.RETRY, reason="first"),
            RecoveryDecision(action=RecoveryAction.FALLBACK_TOOL, reason="last", fallback_tool="better_tool"),
        ]
        correction = ErrorRecoveryEngine.infer_correction_from_history(history, "old_tool")
        assert "better_tool" in correction


# ===========================================================================
# RetryableExecutor
# ===========================================================================

class TestRetryableExecutor:
    def test_execute_success(self):
        async def _test():
            executor = RetryableExecutor()
            call_count = 0

            async def succeed(args):
                nonlocal call_count
                call_count += 1
                return "success"

            result = await executor.execute(
                tool_name="test",
                tool_args={},
                execute_fn=succeed,
            )
            assert result.succeeded is True
            assert result.final_result == "success"
            assert call_count == 1

        asyncio.run(_test())

    def test_execute_retry_then_success(self):
        async def _test():
            executor = RetryableExecutor()
            call_count = 0

            async def flaky(args):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("timeout transient failure")
                return "eventual success"

            result = await executor.execute(
                tool_name="test",
                tool_args={},
                execute_fn=flaky,
            )
            assert result.succeeded is True
            assert result.final_result == "eventual success"
            assert call_count == 3

        asyncio.run(_test())

    def test_execute_skips_on_permission_error(self):
        async def _test():
            executor = RetryableExecutor()
            call_count = 0

            async def permission_denied(args):
                nonlocal call_count
                call_count += 1
                raise PermissionError("permission denied")

            result = await executor.execute(
                tool_name="test",
                tool_args={},
                execute_fn=permission_denied,
            )
            # Permission denied → SKIP, which marks state as succeeded
            assert result.succeeded is True
            assert "Skipped" in result.final_result
            assert call_count == 1

        asyncio.run(_test())

    def test_execute_collects_recovery_history(self):
        async def _test():
            executor = RetryableExecutor()

            async def always_fails(args):
                raise ConnectionError("timeout")

            result = await executor.execute(
                tool_name="unknown_tool",
                tool_args={},
                execute_fn=always_fails,
            )
            assert len(result.recovery_history) > 0

        asyncio.run(_test())

    def test_execute_on_retry_callback(self):
        async def _test():
            executor = RetryableExecutor()
            callbacks = []

            def on_retry(decision, attempt):
                callbacks.append((decision.action, attempt))

            call_count = 0

            async def flaky(args):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ConnectionError("timeout")
                return "ok"

            result = await executor.execute(
                tool_name="test",
                tool_args={},
                execute_fn=flaky,
                on_retry=on_retry,
            )
            assert result.succeeded is True
            assert len(callbacks) >= 1

        asyncio.run(_test())

    def test_custom_recovery_engine(self):
        async def _test():
            engine = ErrorRecoveryEngine()
            executor = RetryableExecutor(recovery=engine)

            async def works(args):
                return "done"

            result = await executor.execute("t", {}, works)
            assert result.succeeded is True
            assert result.final_result == "done"

        asyncio.run(_test())


# ===========================================================================
# RecoveryConfig
# ===========================================================================

class TestRecoveryConfig:
    def test_constants(self):
        assert RecoveryConfig.MAX_RETRIES == 3
        assert RecoveryConfig.MAX_FALLBACK_DEPTH == 2
        assert RecoveryConfig.BASE_BACKOFF_SECONDS == 1.0
        assert RecoveryConfig.MAX_BACKOFF_SECONDS == 60.0
        assert RecoveryConfig.BACKOFF_MULTIPLIER == 2.0

    def test_fallback_chains(self):
        assert "file_read" in RecoveryConfig.FALLBACK_CHAINS
        assert "grep" in RecoveryConfig.FALLBACK_CHAINS
        assert "glob" in RecoveryConfig.FALLBACK_CHAINS
        assert "web_fetch" in RecoveryConfig.FALLBACK_CHAINS
        assert "file_edit" in RecoveryConfig.FALLBACK_CHAINS

    def test_error_patterns(self):
        patterns = RecoveryConfig.ERROR_PATTERNS
        assert patterns["timed out"] == ErrorCategory.TRANSIENT
        assert patterns["rate limit"] == ErrorCategory.TRANSIENT
        assert patterns["permission denied"] == ErrorCategory.PERMISSION_DENIED
        assert patterns["unknown tool"] == ErrorCategory.TOOL_NOT_FOUND
        assert patterns["api error"] == ErrorCategory.MODEL_ERROR
        assert patterns["sandbox"] == ErrorCategory.SANDBOX_ERROR


# ===========================================================================
# compute_backoff
# ===========================================================================

class TestComputeBackoff:
    def test_first_attempt(self):
        delay = compute_backoff(0, base=1.0, max_wait=60.0)
        assert 0.75 <= delay <= 1.25  # base * 2^0 = 1.0, +/- 25% jitter

    def test_third_attempt(self):
        delay = compute_backoff(3, base=1.0, max_wait=60.0)
        assert 6.0 <= delay <= 10.0  # base * 2^3 = 8.0, +/- 25%

    def test_respects_max_wait(self):
        delay = compute_backoff(10, base=1.0, max_wait=30.0)
        assert delay <= 37.5  # max_wait + 25% jitter cap
