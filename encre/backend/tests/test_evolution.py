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

"""Tests for evolution subsystem: learner, optimizer, reflex, meta-cognition."""

import pytest

from encre.evolution.learner import EncreEvolutionLearner, SuccessRecord, ErrorRecord
from encre.evolution.optimizer import EncreStrategyOptimizer, ToolStrategy
from encre.evolution.reflex import EncreReflexLoop, ReflexResult
from encre.evolution.meta import EncreMetaCognition, CapabilityProfile
from encre.evolution.config import EvolutionConfig


class TestEvolutionConfig:
    def test_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.learner_enabled is True
        assert cfg.optimizer_enabled is True
        assert cfg.reflex_enabled is True
        assert cfg.meta_enabled is True

    def test_custom(self):
        cfg = EvolutionConfig(learner_enabled=False, optimizer_enabled=False)
        assert cfg.learner_enabled is False
        assert cfg.optimizer_enabled is False

    def test_create_default(self):
        cfg = EvolutionConfig.create_default()
        assert cfg.learner is not None
        assert cfg.optimizer is not None
        assert cfg.reflex is not None
        assert cfg.meta is not None

    def test_create_disabled(self):
        cfg = EvolutionConfig.create_disabled()
        assert cfg.learner_enabled is False
        assert cfg.meta_enabled is False


class TestRecords:
    def test_success_record(self):
        sr = SuccessRecord(tool_name="bash", intent_signature="run tests", param_pattern='{"cmd": "pytest"}', outcome="passed")
        assert sr.tool_name == "bash"
        assert sr.intent_signature == "run tests"
        assert sr.reuse_count == 0

    def test_error_record(self):
        er = ErrorRecord(tool_name="grep", error_type="no_match", error_context="no matches found", correction="use broader pattern")
        assert er.tool_name == "grep"
        assert er.error_type == "no_match"
        assert er.correction == "use broader pattern"
        assert er.resolved is False

    def test_record_serialization(self):
        sr = SuccessRecord(tool_name="bash", intent_signature="run", param_pattern="{}", outcome="ok")
        d = sr.to_dict()
        assert d["tool_name"] == "bash"
        sr2 = SuccessRecord.from_dict(d)
        assert sr2.tool_name == sr.tool_name


class TestEvolutionLearner:
    def setup_method(self):
        self.learner = EncreEvolutionLearner()

    def test_record_error(self):
        self.learner.record_error("bash", "timeout", "command timed out", "retry with backoff")
        assert len(self.learner._errors) == 1
        assert self.learner._errors[0].tool_name == "bash"

    def test_record_success(self):
        self.learner.record_success("bash", "run tests", {"cmd": "pytest"}, "passed")
        assert len(self.learner._successes) == 1
        assert self.learner._successes[0].tool_name == "bash"

    def test_record_correction_matches_open_error(self):
        self.learner.record_error("bash", "timeout", "command timed out when running git status", "")
        self.learner.record_correction("bash", "command timed out when running", "use timeout flag")
        assert self.learner._errors[0].resolved is True

    def test_record_correction_no_match(self):
        self.learner.record_error("bash", "timeout", "command timed out running git", "")
        self.learner.record_correction("bash", "completely different error about permissions", "fix perms")
        assert self.learner._errors[0].resolved is False

    def test_similar_error_reuse(self):
        self.learner.record_error("bash", "timeout", "command timed out", "")
        self.learner.record_error("bash", "timeout", "command timed out running git", "use timeout")
        # Second call should reuse the first record (similar context)
        assert len(self.learner._errors) == 1
        assert self.learner._errors[0].trigger_count == 1

    def test_mark_error_resolved(self):
        self.learner.record_error("bash", "timeout", "command timed out", "")
        self.learner.mark_error_resolved("bash", "command timed out")
        assert self.learner._errors[0].resolved is True

    def test_get_guidance(self):
        self.learner.record_error("bash", "timeout", "command timed out", "use timeout flag")
        self.learner.record_success("bash", "run tests", {"cmd": "pytest"}, "tests passed")
        guidance = self.learner.get_guidance("bash", "run tests with timeout")
        assert isinstance(guidance, str)

    def test_get_guidance_unknown_tool(self):
        guidance = self.learner.get_guidance("unknown_tool", "some context")
        assert guidance == ""

    def test_get_tool_best_params(self):
        self.learner.record_success("bash", "run tests", {"cmd": "pytest -v"}, "passed")
        params = self.learner.get_tool_best_params("bash", "run tests")
        assert params is not None
        assert params["cmd"] == "pytest -v"

    def test_get_tool_best_params_none(self):
        params = self.learner.get_tool_best_params("nonexistent", "test")
        assert params is None

    def test_get_statistics(self):
        self.learner.record_error("bash", "timeout", "err1", "fix1")
        self.learner.record_success("bash", "run", {}, "ok")
        stats = self.learner.get_statistics()
        assert stats["total_errors"] >= 1
        assert stats["total_successes"] >= 1

    def test_save_load(self):
        import tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "learner.json")
        self.learner._storage_path = path
        self.learner.record_error("bash", "timeout", "err", "fix")
        self.learner.save()
        learner2 = EncreEvolutionLearner(storage_path=path)
        assert learner2.load() is True
        assert len(learner2._errors) == 1
        import shutil
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)

    def test_load_nonexistent(self):
        learner = EncreEvolutionLearner(storage_path="/nonexistent/path/file.json")
        assert learner.load() is False

    def test_reset(self):
        self.learner.record_error("bash", "timeout", "err", "fix")
        self.learner.record_success("bash", "run", {}, "ok")
        self.learner.reset()
        assert len(self.learner._errors) == 0
        assert len(self.learner._successes) == 0


class TestStrategyOptimizer:
    def setup_method(self):
        self.optimizer = EncreStrategyOptimizer()

    def test_record_outcome_success(self):
        self.optimizer.record_outcome("bash", {"cmd": "ls -la"}, success=True, latency_ms=100)
        stats = self.optimizer.get_statistics()
        assert "bash" in stats
        assert stats["bash"]["total_samples"] == 1

    def test_record_outcome_failure(self):
        self.optimizer.record_outcome("bash", {"cmd": "rm -rf /"}, success=False)
        stats = self.optimizer.get_statistics()
        assert stats["bash"]["total_samples"] == 1

    def test_suggest_strategy(self):
        for _ in range(5):
            self.optimizer.record_outcome("bash", {"cmd": "ls -la"}, success=True)
        suggestion = self.optimizer.suggest_strategy("bash", "list files")
        assert suggestion is not None
        assert "_strategy_hint" in suggestion

    def test_suggest_strategy_insufficient_samples(self):
        self.optimizer.record_outcome("bash", {"cmd": "ls"}, success=True)
        # Only 1 sample, below MIN_SAMPLES_FOR_RECOMMENDATION
        suggestion = self.optimizer.suggest_strategy("bash", "list")
        assert suggestion is None

    def test_suggest_strategy_unknown_tool(self):
        assert self.optimizer.suggest_strategy("nonexistent", "test") is None

    def test_get_fallback(self):
        for _ in range(5):
            self.optimizer.record_outcome("bash", {"cmd": "ls -la"}, success=True)
        for _ in range(5):
            self.optimizer.record_outcome("bash", {"cmd": "pwd"}, success=True)
        fallback = self.optimizer.get_fallback("bash", {"cmd": "ls -la"})
        assert fallback is not None
        assert "_fallback_hint" in fallback

    def test_get_statistics(self):
        self.optimizer.record_outcome("bash", {"cmd": "ls"}, success=True)
        self.optimizer.record_outcome("grep", {"pattern": "foo"}, success=False)
        stats = self.optimizer.get_statistics()
        assert "bash" in stats
        assert "grep" in stats

    def test_reset(self):
        self.optimizer.record_outcome("bash", {"cmd": "ls"}, success=True)
        self.optimizer.reset()
        assert self.optimizer.suggest_strategy("bash", "test") is None


class TestReflexLoop:
    def setup_method(self):
        self.reflex = EncreReflexLoop(enabled=True)

    def test_reflect_empty_tools(self):
        result = self.reflex.reflect(turn_number=1, tool_results=[], turn_latency_ms=100)
        assert isinstance(result, ReflexResult)
        assert result.turn_number == 1
        assert result.score < 1.0
        assert len(result.issues) > 0

    def test_reflect_all_success(self):
        result = self.reflex.reflect(turn_number=2, tool_results=[
            {"tool_name": "file_read", "is_error": False},
            {"tool_name": "grep", "is_error": False},
        ], turn_latency_ms=2000)
        assert result.score > 0.5
        assert result.should_retry is False

    def test_reflect_all_errors(self):
        result = self.reflex.reflect(turn_number=3, tool_results=[
            {"tool_name": "bash", "is_error": True},
            {"tool_name": "bash", "is_error": True},
        ], turn_latency_ms=500)
        assert result.score < 0.5
        # error_rate = 2/2 = 1.0, should_retry = error_rate > 0.5 and total > 1
        assert result.should_retry is True

    def test_consecutive_failures_detected(self):
        for i in range(4):
            self.reflex.reflect(turn_number=i, tool_results=[
                {"tool_name": "bash", "is_error": True},
            ], turn_latency_ms=100)
        result = self.reflex.reflect(turn_number=5, tool_results=[
            {"tool_name": "bash", "is_error": True},
        ], turn_latency_ms=100)
        assert any("consecutive" in issue.lower() for issue in result.issues)

    def test_duplicate_calls_detected(self):
        result = self.reflex.reflect(turn_number=1, tool_results=[
            {"tool_name": "grep", "is_error": False},
            {"tool_name": "grep", "is_error": False},
            {"tool_name": "grep", "is_error": False},
            {"tool_name": "grep", "is_error": False},
        ], turn_latency_ms=100)
        assert any("repeated" in issue.lower() for issue in result.issues)

    def test_slow_turn_detected(self):
        result = self.reflex.reflect(turn_number=1, tool_results=[
            {"tool_name": "bash", "is_error": False},
        ], turn_latency_ms=120000)
        assert any("slow" in issue.lower() for issue in result.issues)

    def test_get_improvement_context(self):
        self.reflex.reflect(turn_number=1, tool_results=[
            {"tool_name": "bash", "is_error": True},
        ], turn_latency_ms=100)
        ctx = self.reflex.get_improvement_context()
        assert isinstance(ctx, str)

    def test_get_improvement_context_empty(self):
        reflex = EncreReflexLoop(enabled=True)
        assert reflex.get_improvement_context() == ""

    def test_get_trend_stable(self):
        for i in range(5):
            self.reflex.reflect(turn_number=i, tool_results=[
                {"tool_name": "bash", "is_error": False},
            ], turn_latency_ms=100)
        assert self.reflex.get_trend() == "stable"

    def test_get_average_score(self):
        self.reflex.reflect(turn_number=1, tool_results=[
            {"tool_name": "bash", "is_error": False},
        ], turn_latency_ms=100)
        avg = self.reflex.get_average_score()
        assert 0.0 <= avg <= 1.0

    def test_reset(self):
        self.reflex.reflect(turn_number=1, tool_results=[
            {"tool_name": "bash", "is_error": False},
        ], turn_latency_ms=100)
        self.reflex.reset()
        assert self.reflex.get_average_score() == 1.0
        assert self.reflex.get_improvement_context() == ""

    def test_disabled_reflex(self):
        reflex = EncreReflexLoop(enabled=False)
        result = reflex.reflect(turn_number=1, tool_results=[], turn_latency_ms=100)
        assert result.score == 1.0
        assert result.issues == []


class TestMetaCognition:
    def setup_method(self):
        self.meta = EncreMetaCognition()

    def test_capability_profile_default(self):
        profile = CapabilityProfile(domain="python")
        assert profile.domain == "python"
        assert profile.score == 0.5
        assert profile.confidence == 0.0
        assert profile.sample_count == 0

    def test_capability_profile_update(self):
        profile = CapabilityProfile(domain="python")
        profile.update(success=True, difficulty=0.5)
        assert profile.sample_count == 1
        assert profile.score > 0.5

    def test_assess_turn(self):
        self.meta.assess_turn("write a python function to read a file", [
            {"tool_name": "file_read", "is_error": False},
        ])
        profile = self.meta.get_profile("file_operations")
        assert isinstance(profile, dict)
        assert profile["score"] > 0.5

    def test_get_profile_unknown(self):
        result = self.meta.get_profile("unknown_domain")
        assert isinstance(result, dict)
        assert result["confidence"] == 0.0

    def test_get_all_profiles(self):
        self.meta.assess_turn("run tests with pytest", [
            {"tool_name": "bash", "is_error": False},
        ])
        all_profiles = self.meta.get_profile()
        assert isinstance(all_profiles, dict)

    def test_get_weakness_report(self):
        # Create low score with high confidence
        for _ in range(25):
            self.meta.assess_turn("use bash to run a broken command", [
                {"tool_name": "bash", "is_error": True},
            ])
        report = self.meta.get_weakness_report()
        assert isinstance(report, list)

    def test_should_delegate(self):
        should, reason = self.meta.should_delegate("design a system architecture")
        # No confidence yet, should not delegate
        assert should is False

    def test_get_self_awareness_context(self):
        ctx = self.meta.get_self_awareness_context()
        assert isinstance(ctx, str)

    def test_record_delegation(self):
        self.meta.record_delegation("complex task", "sub_agent", True)
        # Should not crash

    def test_reset(self):
        self.meta.assess_turn("run tests", [{"tool_name": "bash", "is_error": False}])
        self.meta.reset()
        assert self.meta.get_profile() == {}
