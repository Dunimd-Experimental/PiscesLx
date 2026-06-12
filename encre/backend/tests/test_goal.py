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

"""Tests for the goal system: definitions, results, statuses, events, and runners."""

import tempfile

import pytest


class TestGoalStatus:
    def test_all_status_values(self):
        from encre.goal import GoalStatus
        assert GoalStatus.PENDING is not None
        assert GoalStatus.IN_PROGRESS is not None
        assert GoalStatus.SUCCESS is not None
        assert GoalStatus.FAILED is not None
        assert GoalStatus.TIMEOUT is not None
        assert GoalStatus.MAX_ATTEMPTS is not None

    def test_status_is_enum(self):
        from encre.goal import GoalStatus
        from enum import Enum
        assert issubclass(GoalStatus, Enum)

    def test_status_string_conversion(self):
        from encre.goal import GoalStatus
        assert str(GoalStatus.PENDING) == "GoalStatus.PENDING"
        status = GoalStatus.SUCCESS
        assert status.name == "SUCCESS"


class TestGoalDefinition:
    def test_minimal_construction(self):
        from encre.goal import GoalDefinition
        goal = GoalDefinition(
            description="Test goal",
            success_criteria="Tests pass",
        )
        assert goal.description == "Test goal"
        assert goal.success_criteria == "Tests pass"

    def test_default_values(self):
        from encre.goal import GoalDefinition
        goal = GoalDefinition(
            description="Default test",
            success_criteria="All good",
        )
        assert goal.max_attempts == 20
        assert goal.timeout_seconds == 3600
        assert goal.evaluator_model == ""
        assert goal.evaluator_provider == ""

    def test_full_construction(self):
        from encre.goal import GoalDefinition
        goal = GoalDefinition(
            description="Complex task",
            success_criteria="Zero errors, all features work",
            max_attempts=10,
            timeout_seconds=1800,
            evaluator_model="gpt-4o-mini",
            evaluator_provider="openai",
        )
        assert goal.description == "Complex task"
        assert goal.success_criteria == "Zero errors, all features work"
        assert goal.max_attempts == 10
        assert goal.timeout_seconds == 1800
        assert goal.evaluator_model == "gpt-4o-mini"
        assert goal.evaluator_provider == "openai"

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        from encre.goal import GoalDefinition
        assert is_dataclass(GoalDefinition)

    def test_non_default_timeout(self):
        from encre.goal import GoalDefinition
        goal = GoalDefinition(
            description="Quick task",
            success_criteria="Done",
            timeout_seconds=300,
        )
        assert goal.timeout_seconds == 300

    def test_non_default_max_attempts(self):
        from encre.goal import GoalDefinition
        goal = GoalDefinition(
            description="Many attempts",
            success_criteria="Eventually",
            max_attempts=100,
        )
        assert goal.max_attempts == 100


class TestGoalResult:
    def test_default_construction(self):
        from encre.goal import GoalResult, GoalStatus
        result = GoalResult(status=GoalStatus.PENDING)
        assert result.status == GoalStatus.PENDING
        assert result.summary == ""
        assert result.attempts == 0
        assert result.elapsed_seconds == 0.0
        assert result.final_output == ""
        assert result.milestones == []

    def test_full_construction(self):
        from encre.goal import GoalResult, GoalStatus
        result = GoalResult(
            status=GoalStatus.SUCCESS,
            summary="All tests passed",
            attempts=3,
            elapsed_seconds=45.2,
            final_output="Task completed successfully",
            milestones=["Step 1 done", "Step 2 done", "All done"],
        )
        assert result.status == GoalStatus.SUCCESS
        assert result.summary == "All tests passed"
        assert result.attempts == 3
        assert result.elapsed_seconds == 45.2
        assert result.final_output == "Task completed successfully"
        assert len(result.milestones) == 3
        assert "Step 1 done" in result.milestones

    def test_failed_status(self):
        from encre.goal import GoalResult, GoalStatus
        result = GoalResult(
            status=GoalStatus.FAILED,
            summary="Could not complete the task",
            attempts=10,
            elapsed_seconds=600.0,
        )
        assert result.status == GoalStatus.FAILED
        assert result.attempts == 10

    def test_timeout_status(self):
        from encre.goal import GoalResult, GoalStatus
        result = GoalResult(
            status=GoalStatus.TIMEOUT,
            summary="Timed out",
            elapsed_seconds=3601.0,
        )
        assert result.status == GoalStatus.TIMEOUT

    def test_max_attempts_status(self):
        from encre.goal import GoalResult, GoalStatus
        result = GoalResult(
            status=GoalStatus.MAX_ATTEMPTS,
            summary="Reached max attempts",
            attempts=20,
        )
        assert result.status == GoalStatus.MAX_ATTEMPTS

    def test_milestones_are_mutable_list(self):
        from encre.goal import GoalResult, GoalStatus
        result = GoalResult(status=GoalStatus.IN_PROGRESS, milestones=["started"])
        result.milestones.append("middle")
        result.milestones.append("almost done")
        assert len(result.milestones) == 3

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        from encre.goal import GoalResult
        assert is_dataclass(GoalResult)


class TestGoalEvent:
    def test_construction(self):
        from encre.goal import GoalEvent, GoalStatus
        event = GoalEvent(
            status=GoalStatus.IN_PROGRESS,
            attempt=1,
            message="Working on it",
        )
        assert event.status == GoalStatus.IN_PROGRESS
        assert event.attempt == 1
        assert event.message == "Working on it"

    def test_default_values(self):
        from encre.goal import GoalEvent, GoalStatus
        event = GoalEvent(status=GoalStatus.PENDING)
        assert event.attempt == 0
        assert event.message == ""

    def test_success_event(self):
        from encre.goal import GoalEvent, GoalStatus
        event = GoalEvent(
            status=GoalStatus.SUCCESS,
            attempt=5,
            message="Goal achieved",
        )
        assert event.status == GoalStatus.SUCCESS
        assert event.attempt == 5

    def test_failed_event(self):
        from encre.goal import GoalEvent, GoalStatus
        event = GoalEvent(
            status=GoalStatus.FAILED,
            attempt=20,
            message="All attempts exhausted",
        )
        assert event.status == GoalStatus.FAILED

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        from encre.goal import GoalEvent
        assert is_dataclass(GoalEvent)


class TestEncreGoalRunnerConstruction:
    def test_construction_with_config(self):
        from encre.goal import EncreGoalRunner
        from encre.config import EncreConfig
        config = EncreConfig(model="gpt-4o", backend_type="openai")
        runner = EncreGoalRunner(config=config)
        assert runner is not None
        assert runner.config is config
        assert runner.tool_registry is not None
        assert runner.hook_system is not None
        assert runner.safety is not None
        assert runner.telemetry is not None

    def test_construction_with_all_params(self):
        from encre.goal import EncreGoalRunner
        from encre.config import EncreConfig
        from encre.tools.registry import ToolRegistry
        from encre.hooks.system import EncreHookSystem
        from encre.safety import EncreSafetyEngine

        config = EncreConfig(model="claude-sonnet-4-20250514", backend_type="anthropic")
        tools = ToolRegistry()
        hooks = EncreHookSystem()
        safety = EncreSafetyEngine(config)

        runner = EncreGoalRunner(
            config=config,
            tool_registry=tools,
            hook_system=hooks,
            safety=safety,
        )
        assert runner.tool_registry is tools
        assert runner.hook_system is hooks
        assert runner.safety is safety

    def test_evulator_system_prompt_is_string(self):
        from encre.goal import EncreGoalRunner
        from encre.config import EncreConfig
        config = EncreConfig(model="gpt-4o", backend_type="openai")
        runner = EncreGoalRunner(config=config)
        assert isinstance(runner.EVALUATOR_SYSTEM_PROMPT, str)
        assert "goal completion evaluator" in runner.EVALUATOR_SYSTEM_PROMPT.lower()

    def test_build_goal_prompt(self):
        from encre.goal import EncreGoalRunner, GoalDefinition
        from encre.config import EncreConfig
        config = EncreConfig(model="gpt-4o", backend_type="openai")
        runner = EncreGoalRunner(config=config)
        goal = GoalDefinition(
            description="Implement login",
            success_criteria="Login endpoint works with JWT",
        )
        prompt = runner._build_goal_prompt(goal)
        assert "GOAL: Implement login" in prompt
        assert "SUCCESS CRITERIA: Login endpoint works with JWT" in prompt
        assert "autonomously" in prompt.lower()


class TestEncreGoalLoopConstruction:
    def test_construction_basic(self):
        from encre.goal import EncreGoalLoop
        from encre.config import EncreConfig

        config = EncreConfig(model="gpt-4o", backend_type="openai")

        # Manually create a minimal mock for EncreAgent
        import tempfile
        class MockAgent:
            def __init__(self):
                self.config = config
                from encre.tools.registry import ToolRegistry
                from encre.hooks.system import EncreHookSystem
                from encre.safety import EncreSafetyEngine
                from encre.memdir.system import EncreMemorySystem
                from encre.skills.registry import EncreSkillRegistry
                from encre.telemetry import EncreTelemetry
                self.tool_registry = ToolRegistry()
                self.hook_system = EncreHookSystem()
                self.safety = EncreSafetyEngine(config)
                self.memory_system = EncreMemorySystem(auto_memory_path=tempfile.mkdtemp())
                self.skill_registry = EncreSkillRegistry()
                self.telemetry = EncreTelemetry(enabled=False)

        agent = MockAgent()
        loop = EncreGoalLoop(
            agent=agent,
            description="Test description",
            success_criteria="Test criteria",
        )
        assert loop is not None
        assert loop._description == "Test description"
        assert loop._success_criteria == "Test criteria"
        assert loop._max_attempts == 20
        assert loop._timeout_seconds == 3600
        assert loop.runner is not None

    def test_construction_with_custom_params(self):
        from encre.goal import EncreGoalLoop
        from encre.config import EncreConfig

        config = EncreConfig(model="gpt-4o", backend_type="openai")

        class MockAgent:
            def __init__(self):
                self.config = config
                from encre.tools.registry import ToolRegistry
                from encre.hooks.system import EncreHookSystem
                from encre.safety import EncreSafetyEngine
                from encre.memdir.system import EncreMemorySystem
                from encre.skills.registry import EncreSkillRegistry
                from encre.telemetry import EncreTelemetry
                self.tool_registry = ToolRegistry()
                self.hook_system = EncreHookSystem()
                self.safety = EncreSafetyEngine(config)
                self.memory_system = EncreMemorySystem(auto_memory_path=tempfile.mkdtemp())
                self.skill_registry = EncreSkillRegistry()
                self.telemetry = EncreTelemetry(enabled=False)

        agent = MockAgent()
        loop = EncreGoalLoop(
            agent=agent,
            description="Custom desc",
            success_criteria="Custom criteria",
            max_attempts=5,
            timeout_seconds=600,
        )
        assert loop._max_attempts == 5
        assert loop._timeout_seconds == 600

    def test_runner_uses_agent_properties(self):
        from encre.goal import EncreGoalLoop
        from encre.config import EncreConfig

        config = EncreConfig(model="gpt-4o", backend_type="openai")

        class MockAgent:
            def __init__(self):
                self.config = config
                from encre.tools.registry import ToolRegistry
                from encre.hooks.system import EncreHookSystem
                from encre.safety import EncreSafetyEngine
                from encre.memdir.system import EncreMemorySystem
                from encre.skills.registry import EncreSkillRegistry
                from encre.telemetry import EncreTelemetry
                self.tool_registry = ToolRegistry()
                self.hook_system = EncreHookSystem()
                self.safety = EncreSafetyEngine(config)
                self.memory_system = EncreMemorySystem(auto_memory_path=tempfile.mkdtemp())
                self.skill_registry = EncreSkillRegistry()
                self.telemetry = EncreTelemetry(enabled=False)

        agent = MockAgent()
        loop = EncreGoalLoop(agent=agent)
        # The runner should reference the agent's subsystems
        assert loop.runner.config is agent.config
        assert loop.runner.tool_registry is agent.tool_registry
        assert loop.runner.hook_system is agent.hook_system

    @pytest.mark.asyncio
    async def test_execute_raises_without_description(self):
        from encre.goal import EncreGoalLoop
        from encre.config import EncreConfig
        import pytest

        config = EncreConfig(model="gpt-4o", backend_type="openai")

        class MockAgent:
            def __init__(self):
                self.config = config
                from encre.tools.registry import ToolRegistry
                from encre.hooks.system import EncreHookSystem
                from encre.safety import EncreSafetyEngine
                from encre.memdir.system import EncreMemorySystem
                from encre.skills.registry import EncreSkillRegistry
                from encre.telemetry import EncreTelemetry
                self.tool_registry = ToolRegistry()
                self.hook_system = EncreHookSystem()
                self.safety = EncreSafetyEngine(config)
                self.memory_system = EncreMemorySystem(auto_memory_path=tempfile.mkdtemp())
                self.skill_registry = EncreSkillRegistry()
                self.telemetry = EncreTelemetry(enabled=False)

        agent = MockAgent()
        loop = EncreGoalLoop(agent=agent)

        with pytest.raises(ValueError, match="description"):
            await loop.execute()  # No description provided here or at construction
