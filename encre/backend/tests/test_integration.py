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

"""Integration tests: cross-subsystem wiring, agent composition, safety -> tool flow."""

import asyncio

import pytest

from encre.config import EncreConfig
from encre.agent import EncreAgent
from encre.tools.registry import ToolRegistry
from encre.tools.builtin import EncreBashTool, EncreFileReadTool, EncreFileWriteTool
from encre.safety import EncreSafetyEngine, DangerLevel, analyze_bash_command


# ===========================================================================
# SafetyEngine + Tool integration
# ===========================================================================


class TestSafetyToolIntegration:
    def setup_method(self):
        self.config = EncreConfig(permission_mode="default")
        self.safety = EncreSafetyEngine(config=self.config)

    def test_analyze_safe_bash(self):
        result = self.safety.analyze_bash("ls -la")
        assert result is not None
        assert result.danger_level == DangerLevel.SAFE

    def test_analyze_dangerous_rm(self):
        result = self.safety.analyze_bash("rm -rf /")
        assert result is not None
        assert result.danger_level in (DangerLevel.HIGH, DangerLevel.CRITICAL)

    def test_analyze_pipe(self):
        result = self.safety.analyze_bash("cat /etc/passwd | grep root")
        assert result is not None
        assert result.contains_pipe is True

    def test_permission_check_allow(self):
        async def _test():
            decision = await self.safety.check_tool_permission(
                "file_read", {"path": "test.py"}
            )
            assert decision is not None

        asyncio.run(_test())

    def test_permission_mode_bypass(self):
        async def _test():
            safety = EncreSafetyEngine(
                config=EncreConfig(permission_mode="bypass")
            )
            decision = await safety.check_tool_permission(
                "bash", {"cmd": "rm -rf /"}
            )
            assert decision is not None
            assert decision.behavior == "allow"

        asyncio.run(_test())

    def test_validate_tool_output_truncates(self):
        result = self.safety.validate_tool_output("bash", "some output")
        assert isinstance(result, str)
        assert "some output" in result

    def test_validate_tool_output_truncates_long(self):
        long_output = "x" * 200000
        result = self.safety.validate_tool_output("bash", long_output)
        assert len(result) <= self.config.tool_result_max_chars + 50

    def test_danger_level_enum(self):
        assert DangerLevel.SAFE is not None
        assert DangerLevel.LOW is not None
        assert DangerLevel.MEDIUM is not None
        assert DangerLevel.HIGH is not None
        assert DangerLevel.CRITICAL is not None

    def test_analyze_bash_command_function(self):
        result = analyze_bash_command("echo hello")
        assert result is not None
        assert result.danger_level == DangerLevel.SAFE

    def test_is_bash_safe(self):
        is_safe, reason = self.safety.is_bash_safe("echo hello")
        assert is_safe is True
        assert reason == ""

    def test_is_bash_safe_dangerous(self):
        is_safe, reason = self.safety.is_bash_safe("rm -rf /")
        assert is_safe is False
        assert len(reason) > 0


# ===========================================================================
# Agent creation and composition
# ===========================================================================


class TestAgentComposition:
    def _make_config(self):
        return EncreConfig(
            model="gpt-4o",
            backend_type="local",
            permission_mode="bypass",
            max_turns=1,
            max_tokens=1024,
        )

    def test_minimal_agent(self):
        agent = EncreAgent(config=self._make_config())
        assert agent is not None
        assert agent.config is not None

    def test_agent_reset(self):
        agent = EncreAgent(config=self._make_config())
        agent.reset()
        assert agent is not None

    def test_agent_has_tool_registry(self):
        agent = EncreAgent(config=self._make_config())
        assert agent.tool_registry is not None

    def test_agent_has_memory_system(self):
        # memory_system is None by default (not auto-created)
        # but can be explicitly provided with required auto_memory_path
        import tempfile
        from encre.memdir.system import EncreMemorySystem
        mem = EncreMemorySystem(auto_memory_path=tempfile.mkdtemp())
        agent = EncreAgent(config=self._make_config(), memory_system=mem)
        assert agent.memory_system is not None

    def test_agent_has_safety_engine(self):
        agent = EncreAgent(config=self._make_config())
        assert agent.safety is not None

    def test_agent_has_hook_system(self):
        agent = EncreAgent(config=self._make_config())
        assert agent.hook_system is not None

    def test_agent_has_skill_registry(self):
        agent = EncreAgent(config=self._make_config())
        assert agent.skill_registry is not None

    def test_agent_has_plugin_registry(self):
        agent = EncreAgent(config=self._make_config())
        assert agent.plugin_registry is not None

    def test_agent_has_evolution(self):
        agent = EncreAgent(config=self._make_config())
        assert agent.evolution is not None

    def test_agent_has_telemetry(self):
        agent = EncreAgent(config=self._make_config())
        assert agent.telemetry is not None

    def test_agent_has_recovery(self):
        agent = EncreAgent(config=self._make_config())
        assert agent.recovery is not None

    def test_agent_swarm_method(self):
        agent = EncreAgent(config=self._make_config())
        session = agent.swarm(goal="Test", max_concurrent=2)
        assert session is not None

    def test_agent_load_plugins(self):
        agent = EncreAgent(config=self._make_config())
        count = agent.load_plugins()
        assert isinstance(count, int)


# ===========================================================================
# ToolRegistry + Tool integration
# ===========================================================================


class TestToolRegistryIntegration:
    def setup_method(self):
        self.registry = ToolRegistry()

    def test_register_and_get(self):
        tool = EncreFileReadTool()
        self.registry.register(tool)
        retrieved = self.registry.get("file_read")
        assert retrieved is not None

    def test_register_many(self):
        tools = [EncreFileReadTool(), EncreFileWriteTool(), EncreBashTool()]
        self.registry.register_many(tools)
        assert self.registry.get("file_read") is not None
        assert self.registry.get("file_write") is not None
        assert self.registry.get("bash") is not None

    def test_get_nonexistent(self):
        assert self.registry.get("nonexistent_tool") is None

    def test_remove_tool(self):
        tool = EncreFileReadTool()
        self.registry.register(tool)
        del self.registry._tools["file_read"]
        assert self.registry.get("file_read") is None

    def test_all_tools(self):
        self.registry.register(EncreFileReadTool())
        tools = self.registry.all()
        names = [t.name for t in tools]
        assert "file_read" in names

    def test_get_openai_tools(self):
        self.registry.register(EncreFileReadTool())
        self.registry.register(EncreBashTool())
        tools_json = self.registry.get_openai_tools()
        assert len(tools_json) >= 2

    def test_get_anthropic_tools(self):
        self.registry.register(EncreFileReadTool())
        tools_json = self.registry.get_anthropic_tools()
        assert len(tools_json) >= 1

    def test_clear_tools(self):
        self.registry.register(EncreFileReadTool())
        self.registry._tools.clear()
        assert self.registry.all() == []
