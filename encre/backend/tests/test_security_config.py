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

"""Tests for SSRF guard, rate limiter, config, telemetry, sandbox types."""

import pytest

from encre.ssrf import EncreSSRFGuard
from encre.ratelimit import EncreRateLimiter, RateLimitResult
from encre.config import EncreConfig
from encre.telemetry import EncreTelemetry, ToolCallRecord, TurnRecord, RetryRecord
from encre.logging_config import setup_logging, get_logger


class TestSSRFGuard:
    def setup_method(self):
        self.guard = EncreSSRFGuard()

    def test_validate_url_allows_public(self):
        result = self.guard.validate_url("https://example.com/resource")
        assert result is True

    def test_validate_url_blocks_private_ip(self):
        result = self.guard.validate_url("http://127.0.0.1/admin")
        assert result is False

    def test_validate_url_blocks_metadata(self):
        result = self.guard.validate_url("http://169.254.169.254/metadata")
        assert result is False

    def test_validate_url_rejects_non_http(self):
        assert self.guard.validate_url("ftp://example.com/file") is False

    def test_validate_url_rejects_invalid(self):
        assert self.guard.validate_url("not-a-url") is False

    def test_is_blocked_hostname_private(self):
        assert self.guard.is_blocked_hostname("127.0.0.1") is True
        assert self.guard.is_blocked_hostname("10.0.0.1") is True
        assert self.guard.is_blocked_hostname("192.168.1.1") is True

    def test_is_blocked_hostname_public(self):
        assert self.guard.is_blocked_hostname("8.8.8.8") is False

    def test_extract_safe_hostname(self):
        hostname = self.guard.extract_safe_hostname("https://example.com/path")
        assert hostname == "example.com"

    def test_extract_safe_hostname_blocked(self):
        hostname = self.guard.extract_safe_hostname("http://127.0.0.1/admin")
        assert hostname is None

    def test_clear_dns_cache(self):
        self.guard.clear_dns_cache()
        assert len(self.guard._dns_cache) == 0


class TestRateLimiter:
    def test_create(self):
        rl = EncreRateLimiter(per_minute=60)
        assert rl.per_minute == 60

    def test_defaults(self):
        rl = EncreRateLimiter()
        assert rl.per_minute == 60
        assert rl.per_hour == 500
        assert rl.max_concurrent == 10

    def test_rate_limit_result(self):
        rr = RateLimitResult(allowed=True, remaining=5)
        assert rr.allowed is True
        assert rr.remaining == 5

    def test_rate_limit_result_denied(self):
        rr = RateLimitResult(allowed=False, remaining=0, retry_after=10.0)
        assert rr.allowed is False
        assert rr.retry_after == 10.0

    def test_first_request_allowed(self):
        rl = EncreRateLimiter(per_minute=999)
        result = rl.check("tool_a")
        assert result.allowed is True

    def test_different_keys_independent(self):
        rl = EncreRateLimiter(per_minute=999)
        rl.check("tool_a")
        result = rl.check("tool_b")
        assert result.allowed is True


class TestConfig:
    def test_defaults(self):
        cfg = EncreConfig()
        assert cfg.max_turns > 0
        assert cfg.max_tokens > 0
        assert cfg.model != ""

    def test_custom_config(self):
        cfg = EncreConfig(
            model="claude-sonnet-4-20250514",
            backend_type="anthropic",
            max_turns=25,
            max_tokens=32768,
        )
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.backend_type == "anthropic"
        assert cfg.max_turns == 25

    def test_backend_kwargs(self):
        cfg = EncreConfig(backend_kwargs={"temperature": 0.7})
        assert cfg.backend_kwargs["temperature"] == 0.7

    def test_permission_mode_default(self):
        cfg = EncreConfig()
        assert cfg.permission_mode == "default"

    def test_sandbox_enabled_default(self):
        cfg = EncreConfig()
        assert cfg.sandbox_enabled is True

    def test_tool_result_max_chars(self):
        cfg = EncreConfig(tool_result_max_chars=50000)
        assert cfg.tool_result_max_chars == 50000


class TestTelemetry:
    def setup_method(self):
        self.tel = EncreTelemetry()

    def test_tool_call_record(self):
        tcr = ToolCallRecord(
            tool_name="bash", latency_ms=1500.0, success=True, tokens_used=100
        )
        assert tcr.tool_name == "bash"
        assert tcr.latency_ms == 1500.0
        assert tcr.success is True
        assert tcr.tokens_used == 100

    def test_turn_record(self):
        tr = TurnRecord(turn_number=1, event_count=2, latency_ms=3000.0)
        assert tr.turn_number == 1
        assert tr.event_count == 2
        assert tr.latency_ms == 3000.0

    def test_retry_record(self):
        rr = RetryRecord(
            attempt=2, error_type="http_status", error_detail="429", delay_s=1.0
        )
        assert rr.attempt == 2
        assert rr.error_type == "http_status"
        assert rr.error_detail == "429"
        assert rr.delay_s == 1.0

    def test_record_tool_call(self):
        self.tel.record_tool_call("bash", 2000.0, True, 100)
        assert len(self.tel.tool_calls) == 1

    def test_record_turn(self):
        self.tel.record_turn(1, 2, 3000.0)
        assert len(self.tel.turns) == 1

    def test_record_retry(self):
        self.tel.record_retry(1, "http_status", "429", 1.0)
        assert len(self.tel.retries) == 1

    def test_get_summary(self):
        self.tel.record_tool_call("bash", 1000.0, True, 100)
        summary = self.tel.get_summary()
        assert isinstance(summary, dict)
        assert summary["total_tool_calls"] == 1

    def test_flush(self):
        self.tel.record_tool_call("bash", 1000.0, True, 100)
        result = self.tel.flush()
        assert isinstance(result, dict)
        assert result["total_tool_calls"] == 1

    def test_reset(self):
        self.tel.record_tool_call("bash", 1000.0, True, 100)
        self.tel.reset()
        assert len(self.tel.tool_calls) == 0

    def test_disabled_telemetry(self):
        tel = EncreTelemetry(enabled=False)
        tel.record_tool_call("bash", 1000.0, True, 100)
        assert len(tel.tool_calls) == 0


class TestLoggingConfig:
    def test_setup_logging(self):
        setup_logging(level="WARNING")
        # setup_logging returns None (configures global state)

    def test_get_logger(self):
        logger = get_logger("test.module")
        assert logger is not None


class TestSandboxTypes:
    def test_sandbox_config(self):
        from encre.sandbox.types import SandboxConfig

        cfg = SandboxConfig(image="ubuntu:22.04", timeout=30)
        assert cfg.image == "ubuntu:22.04"
        assert cfg.timeout == 30

    def test_sandbox_config_defaults(self):
        from encre.sandbox.types import SandboxConfig

        cfg = SandboxConfig()
        assert cfg.image == "python:3.11-slim"
        assert cfg.network == "none"
        assert cfg.memory_limit == "512m"

    def test_sandbox_result(self):
        from encre.sandbox.types import SandboxResult

        sr = SandboxResult(stdout="success", stderr="", exit_code=0, duration_ms=1200.0)
        assert sr.exit_code == 0
        assert sr.stdout == "success"

    def test_sandbox_result_error(self):
        from encre.sandbox.types import SandboxResult

        sr = SandboxResult(
            stdout="", stderr="command not found", exit_code=1, timed_out=True
        )
        assert sr.exit_code == 1
        assert sr.timed_out is True
