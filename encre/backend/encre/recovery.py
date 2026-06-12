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

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable


class ErrorCategory(Enum):
    """Classification of execution errors for recovery routing."""
    TRANSIENT = auto()        # Rate limit, timeout, network — retry with backoff
    INPUT_MALFORMED = auto()  # Bad JSON, missing required param — don't retry blindly
    TOOL_NOT_FOUND = auto()   # Unknown tool — can't recover
    PERMISSION_DENIED = auto()  # Blocked by safety/hook/user — don't retry
    TOOL_EXECUTION = auto()   # Tool ran but returned error — may retry or fallback
    SANDBOX_ERROR = auto()    # Sandbox failure — may retry
    MODEL_ERROR = auto()      # Backend/model error — retry with backoff
    UNKNOWN = auto()          # Unexpected — retry once


class RecoveryAction(Enum):
    RETRY = auto()            # Retry the same tool with same args
    RETRY_WITH_BACKOFF = auto()  # Retry with exponential backoff
    RETRY_WITH_FIXED_ARGS = auto()  # Retry with modified args (corrected)
    FALLBACK_TOOL = auto()    # Try a different tool for the same intent
    DEGRADE = auto()          # Accept partial result and continue
    SKIP = auto()             # Skip this tool entirely
    ABORT_TURN = auto()       # Abort this turn, feed error to model
    ABORT_SESSION = auto()    # Unrecoverable, end the session


@dataclass
class RecoveryDecision:
    action: RecoveryAction
    reason: str = ""
    retry_delay_seconds: float = 0.0
    fallback_tool: str = ""
    fallback_args: dict[str, Any] | None = None
    modified_args: dict[str, Any] | None = None


@dataclass
class RecoveryState:
    """Tracks recovery state for a single tool execution within a turn."""
    tool_name: str
    tool_args: dict[str, Any]
    attempts: int = 0
    max_retries: int = 3
    last_error: str = ""
    last_category: ErrorCategory = ErrorCategory.UNKNOWN
    recovery_history: list[RecoveryDecision] = field(default_factory=list)
    succeeded: bool = False
    final_result: str = ""


class RecoveryConfig:
    """Global recovery behavior configuration."""

    MAX_RETRIES = 3
    MAX_FALLBACK_DEPTH = 2
    BASE_BACKOFF_SECONDS = 1.0
    MAX_BACKOFF_SECONDS = 60.0
    BACKOFF_MULTIPLIER = 2.0

    # Fallback chains: if tool X fails, try tool Y
    FALLBACK_CHAINS: dict[str, list[str]] = {
        "file_read": ["bash"],   # Use cat/head if read fails
        "grep": ["bash"],        # Use grep command directly
        "glob": ["bash"],        # Use ls/find
        "web_fetch": ["bash"],   # Use curl
        "web_search": ["web_fetch"],  # Try direct fetch
        "file_edit": ["bash"],   # Use sed
        "file_write": ["bash"],  # Use cat/echo
    }

    # Error classification patterns
    ERROR_PATTERNS: dict[str, ErrorCategory] = {
        "timed out": ErrorCategory.TRANSIENT,
        "timeout": ErrorCategory.TRANSIENT,
        "connection refused": ErrorCategory.TRANSIENT,
        "connection error": ErrorCategory.TRANSIENT,
        "rate limit": ErrorCategory.TRANSIENT,
        "too many requests": ErrorCategory.TRANSIENT,
        "service unavailable": ErrorCategory.TRANSIENT,
        "internal server error": ErrorCategory.TRANSIENT,
        "bad gateway": ErrorCategory.TRANSIENT,
        "gateway timeout": ErrorCategory.TRANSIENT,
        "invalid json": ErrorCategory.INPUT_MALFORMED,
        "json decode error": ErrorCategory.INPUT_MALFORMED,
        "unknown tool": ErrorCategory.TOOL_NOT_FOUND,
        "permission denied": ErrorCategory.PERMISSION_DENIED,
        "blocked by hook": ErrorCategory.PERMISSION_DENIED,
        "denied by user": ErrorCategory.PERMISSION_DENIED,
        "sandbox": ErrorCategory.SANDBOX_ERROR,
        "api error": ErrorCategory.MODEL_ERROR,
    }


def classify_error(error_message: str) -> ErrorCategory:
    """Classify an error message into a recovery category."""
    lower = error_message.lower()
    for pattern, category in RecoveryConfig.ERROR_PATTERNS.items():
        if pattern in lower:
            return category
    return ErrorCategory.UNKNOWN


def compute_backoff(attempt: int, base: float = 1.0, max_wait: float = 60.0) -> float:
    """Exponential backoff with jitter."""
    wait = min(base * (2 ** attempt), max_wait)
    # Add jitter: +/- 25%
    import random
    jitter = wait * 0.25 * (random.random() * 2 - 1)
    return max(0.0, wait + jitter)


class ErrorRecoveryEngine:
    """Multi-path error recovery engine for tool execution failures.

    Recovery paths (7 total):
      1. RETRY_SAME  — transient error → immediate retry
      2. RETRY_BACKOFF — rate limit/timeout → exponential backoff retry
      3. FIX_ARGS    — malformed input → try to infer correct args and retry
      4. FALLBACK    — tool unavailable → try fallback chain
      5. DEGRADE     — partial success acceptable → continue with partial result
      6. SKIP        — non-critical tool → skip and continue
      7. ABORT       — unrecoverable → feed error to model for replanning

    Termination conditions (10 total):
      1. Max retries exceeded (per-tool)
      2. Max fallback depth exceeded
      3. Max session errors exceeded
      4. Permission denied (no retry)
      5. Tool not found (no retry)
      6. Critical system error (abort)
      7. Same error repeats 3+ times (stuck loop)
      8. Backoff total time exceeds turn timeout
      9. All fallback tools exhausted
      10. Manual abort requested
    """

    def __init__(self) -> None:
        self._session_errors: int = 0
        self._max_session_errors: int = 10
        self._last_errors: list[str] = []  # sliding window for stuck detection

    def decide(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        error_message: str,
        attempt: int,
        fallback_depth: int = 0,
    ) -> RecoveryDecision:
        """Determine the recovery action for a tool execution failure."""

        category = classify_error(error_message)
        # Only count non-transient errors — transient errors (timeouts, rate limits)
        # are self-resolving and should not silently kill the session.
        if category not in (ErrorCategory.TRANSIENT, ErrorCategory.MODEL_ERROR):
            self._session_errors += 1
        self._last_errors.append(error_message[:200])
        if len(self._last_errors) > 10:
            self._last_errors.pop(0)

        # Termination condition: session-level max errors
        if self._session_errors >= self._max_session_errors:
            return RecoveryDecision(
                action=RecoveryAction.ABORT_SESSION,
                reason=f"Max session errors ({self._max_session_errors}) reached",
            )

        # Termination condition: stuck loop (same error 3+ times)
        if self._is_stuck(error_message):
            return RecoveryDecision(
                action=RecoveryAction.ABORT_TURN,
                reason="Same error repeated — likely stuck in a loop",
            )

        # Termination condition: max retries per tool
        if attempt >= RecoveryConfig.MAX_RETRIES:
            if fallback_depth < RecoveryConfig.MAX_FALLBACK_DEPTH:
                fallback = self._get_fallback(tool_name, fallback_depth)
                if fallback:
                    return RecoveryDecision(
                        action=RecoveryAction.FALLBACK_TOOL,
                        reason=f"Max retries ({attempt}) — falling back to {fallback}",
                        fallback_tool=fallback,
                    )
            return RecoveryDecision(
                action=RecoveryAction.SKIP,
                reason=f"Max retries ({attempt}) and no fallback available",
            )

        # Route by category
        if category == ErrorCategory.TRANSIENT:
            delay = compute_backoff(attempt, RecoveryConfig.BASE_BACKOFF_SECONDS)
            return RecoveryDecision(
                action=RecoveryAction.RETRY_WITH_BACKOFF,
                reason=f"Transient error — retry {attempt + 1}/{RecoveryConfig.MAX_RETRIES}",
                retry_delay_seconds=delay,
            )

        if category == ErrorCategory.MODEL_ERROR:
            delay = compute_backoff(attempt, 2.0, 120.0)
            return RecoveryDecision(
                action=RecoveryAction.RETRY_WITH_BACKOFF,
                reason=f"Model/API error — retry with longer backoff",
                retry_delay_seconds=delay,
            )

        if category == ErrorCategory.INPUT_MALFORMED:
            if attempt == 0:
                return RecoveryDecision(
                    action=RecoveryAction.RETRY_WITH_FIXED_ARGS,
                    reason="Malformed input — attempting arg correction",
                )
            return RecoveryDecision(
                action=RecoveryAction.ABORT_TURN,
                reason="Cannot fix malformed input after retry",
            )

        if category == ErrorCategory.PERMISSION_DENIED:
            return RecoveryDecision(
                action=RecoveryAction.SKIP,
                reason="Permission denied — skipping tool",
            )

        if category == ErrorCategory.TOOL_NOT_FOUND:
            return RecoveryDecision(
                action=RecoveryAction.ABORT_TURN,
                reason="Tool not found — asking model to replan",
            )

        if category == ErrorCategory.TOOL_EXECUTION:
            if fallback_depth < RecoveryConfig.MAX_FALLBACK_DEPTH:
                fallback = self._get_fallback(tool_name, fallback_depth)
                if fallback:
                    return RecoveryDecision(
                        action=RecoveryAction.FALLBACK_TOOL,
                        reason=f"Tool execution failed — falling back to {fallback}",
                        fallback_tool=fallback,
                    )
            return RecoveryDecision(
                action=RecoveryAction.RETRY,
                reason=f"Tool execution failed — retry {attempt + 1}",
            )

        # UNKNOWN
        if attempt == 0:
            delay = compute_backoff(0, 1.0)
            return RecoveryDecision(
                action=RecoveryAction.RETRY_WITH_BACKOFF,
                reason="Unknown error — retry once with backoff",
                retry_delay_seconds=delay,
            )
        return RecoveryDecision(
            action=RecoveryAction.ABORT_TURN,
            reason="Unknown error persists — asking model to replan",
        )

    def reset_session(self) -> None:
        self._session_errors = 0
        self._last_errors.clear()

    def _is_stuck(self, error: str) -> bool:
        if len(self._last_errors) < 3:
            return False
        # Check if last 3 errors are essentially the same
        recent = self._last_errors[-3:]
        normalized = [e[:100].lower().strip() for e in recent]
        return len(set(normalized)) == 1

    @staticmethod
    def infer_correction(state: RecoveryState) -> str:
        """Derive a human-readable correction from recovery history."""
        if not state.recovery_history:
            return ""
        return ErrorRecoveryEngine._infer_from_decisions(state.recovery_history, state.tool_name)

    @staticmethod
    def infer_correction_from_history(recovery_history: list[RecoveryDecision], tool_name: str = "") -> str:
        """Derive a human-readable correction from a list of recovery decisions."""
        if not recovery_history:
            return ""
        return ErrorRecoveryEngine._infer_from_decisions(recovery_history, tool_name)

    @staticmethod
    def _infer_from_decisions(decisions: list[RecoveryDecision], tool_name: str) -> str:
        for decision in reversed(decisions):
            if decision.action == RecoveryAction.FALLBACK_TOOL:
                return f"use fallback tool '{decision.fallback_tool}' instead of '{tool_name}'"
            if decision.action == RecoveryAction.RETRY_WITH_FIXED_ARGS:
                return f"retry with modified args: {decision.modified_args}"
            if decision.action == RecoveryAction.RETRY_WITH_BACKOFF:
                return f"retry with backoff ({decision.retry_delay_seconds:.0f}s delay)"
        last = decisions[-1]
        return f"{last.action.name}: {last.reason}" if last.reason else last.action.name

    @staticmethod
    def _get_fallback(tool_name: str, depth: int) -> str | None:
        chain = RecoveryConfig.FALLBACK_CHAINS.get(tool_name, [])
        if depth < len(chain):
            return chain[depth]
        return None


class RetryableExecutor:
    """Wraps a tool execution function with automatic retry and recovery.

    Usage:
        executor = RetryableExecutor(recovery_engine)
        result = await executor.execute(
            tool_name="file_read",
            tool_args={"path": "/tmp/data.json"},
            execute_fn=lambda args: tool.execute(**args),
        )
    """

    def __init__(self, recovery: ErrorRecoveryEngine | None = None) -> None:
        self.recovery = recovery or ErrorRecoveryEngine()

    async def execute(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        execute_fn: Callable[[dict[str, Any]], Any],
        on_retry: Callable[[RecoveryDecision, int], None] | None = None,
    ) -> RecoveryState:
        """Execute a tool with automatic retry/recovery."""
        state = RecoveryState(tool_name=tool_name, tool_args=dict(tool_args))
        fallback_depth = 0
        current_args = dict(tool_args)
        current_tool = tool_name

        while state.attempts < RecoveryConfig.MAX_RETRIES + 2:  # +2 for fallback attempts
            try:
                result = await execute_fn(current_args)
                if isinstance(result, str) and result.startswith("Error"):
                    raise RuntimeError(result)
                state.succeeded = True
                state.final_result = str(result) if result else ""
                return state

            except Exception as e:
                error_msg = str(e)
                state.last_error = error_msg
                state.last_category = classify_error(error_msg)

                decision = self.recovery.decide(
                    current_tool, current_args, error_msg,
                    state.attempts, fallback_depth,
                )
                state.recovery_history.append(decision)

                if on_retry:
                    on_retry(decision, state.attempts)

                if decision.action == RecoveryAction.RETRY:
                    state.attempts += 1
                    continue

                elif decision.action == RecoveryAction.RETRY_WITH_BACKOFF:
                    await asyncio.sleep(decision.retry_delay_seconds)
                    state.attempts += 1
                    continue

                elif decision.action == RecoveryAction.RETRY_WITH_FIXED_ARGS:
                    if decision.modified_args:
                        current_args = decision.modified_args
                    state.attempts += 1
                    continue

                elif decision.action == RecoveryAction.FALLBACK_TOOL:
                    current_tool = decision.fallback_tool
                    current_args = decision.fallback_args or current_args
                    fallback_depth += 1
                    state.attempts += 1
                    continue

                elif decision.action == RecoveryAction.DEGRADE:
                    state.succeeded = True
                    state.final_result = f"[Degraded] {error_msg}"
                    return state

                elif decision.action == RecoveryAction.SKIP:
                    state.succeeded = True
                    state.final_result = f"[Skipped] {error_msg}"
                    return state

                elif decision.action in (RecoveryAction.ABORT_TURN, RecoveryAction.ABORT_SESSION):
                    state.succeeded = False
                    state.final_result = f"[Aborted] {error_msg}"
                    return state

        state.succeeded = False
        state.final_result = f"[Exhausted] Max retries exceeded: {state.last_error}"
        return state
