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

"""Tests for EncreAgent: construction, properties, run, lifecycle."""

import inspect
import pytest

from encre.agent import EncreAgent
from encre.config import EncreConfig
from encre.utils.types import AgentEvent


class TestEncreAgentConstruction:
    """Verify EncreAgent can be constructed with various configurations."""

    def test_creation_with_no_args(self):
        agent = EncreAgent()
        assert agent is not None
        assert isinstance(agent.config, EncreConfig)

    def test_creation_with_explicit_config(self):
        config = EncreConfig(model="gpt-4o-mini", max_tokens=1000)
        agent = EncreAgent(config=config)
        assert agent.config is config
        assert agent.config.model == "gpt-4o-mini"

    def test_creation_with_config_defaults(self):
        agent = EncreAgent()
        assert agent.config.model == "gpt-4o"
        assert agent.config.backend_type == "openai"


class TestEncreAgentProperties:
    """Verify EncreAgent exposes expected attributes after construction."""

    def test_has_config(self):
        agent = EncreAgent()
        assert hasattr(agent, "config")
        assert isinstance(agent.config, EncreConfig)

    def test_has_tool_registry(self):
        agent = EncreAgent()
        assert hasattr(agent, "tool_registry")

    def test_has_hook_system(self):
        agent = EncreAgent()
        assert hasattr(agent, "hook_system")

    def test_has_safety(self):
        agent = EncreAgent()
        assert hasattr(agent, "safety")

    def test_has_memory_system(self):
        agent = EncreAgent()
        assert hasattr(agent, "memory_system")

    def test_has_skill_registry(self):
        agent = EncreAgent()
        assert hasattr(agent, "skill_registry")
        assert agent.skill_registry is not None

    def test_has_session(self):
        agent = EncreAgent()
        assert hasattr(agent, "session")

    def test_has_telemetry(self):
        agent = EncreAgent()
        assert hasattr(agent, "telemetry")

    def test_has_evolution(self):
        agent = EncreAgent()
        assert hasattr(agent, "evolution")

    def test_has_recovery(self):
        agent = EncreAgent()
        assert hasattr(agent, "recovery")

    def test_has_loop(self):
        agent = EncreAgent()
        assert hasattr(agent, "loop")


class TestEncreAgentRun:
    """Verify run() signature returns an AsyncGenerator."""

    def test_run_returns_async_generator(self):
        agent = EncreAgent()
        # run() is an async generator function
        assert inspect.isasyncgenfunction(agent.run)

    def test_run_signature(self):
        agent = EncreAgent()
        sig = inspect.signature(agent.run)
        params = list(sig.parameters.keys())
        assert "prompt" in params
        assert "system_prompt" in params

    def test_run_with_tools_returns_async_generator(self):
        agent = EncreAgent()
        assert inspect.isasyncgenfunction(agent.run_with_tools)

    def test_run_return_type_is_async_generator(self):
        import typing
        agent = EncreAgent()
        hints = typing.get_type_hints(agent.run)
        assert "return" in hints


class TestEncreAgentLifecycle:
    """Verify EncreAgent lifecycle methods exist."""

    def test_reset_exists(self):
        agent = EncreAgent()
        assert callable(agent.reset)

    def test_aclose_exists(self):
        agent = EncreAgent()
        assert callable(agent.aclose)

    def test_add_message_exists(self):
        agent = EncreAgent()
        assert callable(agent.add_message)

    def test_add_message_adds_to_session(self):
        agent = EncreAgent()
        assert len(agent.session.messages) == 0
        agent.add_message("user", "hello")
        assert len(agent.session.messages) == 1
        assert agent.session.messages[0]["role"] == "user"
        assert agent.session.messages[0]["content"] == "hello"

    def test_respond_permission_exists(self):
        agent = EncreAgent()
        assert callable(agent.respond_permission)

    def test_activate_skill_exists(self):
        agent = EncreAgent()
        assert callable(agent.activate_skill)


class TestEncreAgentGoalAndSwarm:
    """Verify goal() and swarm() factory methods exist."""

    def test_goal_returns_goal_loop(self):
        agent = EncreAgent()
        loop = agent.goal(
            description="Test goal",
            success_criteria="Tests pass",
            max_attempts=3,
        )
        assert loop is not None
        assert hasattr(loop, "execute")

    def test_swarm_returns_swarm_session(self):
        agent = EncreAgent()
        session = agent.swarm(
            goal="Build a TODO app",
            max_concurrent=2,
        )
        assert session is not None
        assert hasattr(session, "execute")

    def test_set_scheduler_exists(self):
        agent = EncreAgent()
        assert callable(agent.set_scheduler)
