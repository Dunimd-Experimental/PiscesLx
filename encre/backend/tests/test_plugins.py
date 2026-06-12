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

"""Tests for plugin system: manifest, plugin protocol, and registry."""

import pytest

from encre.plugins.types import EncrePlugin, PluginManifest, PluginSource
from encre.plugins.registry import PluginRegistry


class TestPluginManifest:
    def test_create_minimal(self):
        m = PluginManifest(name="test-plugin", version="0.1.0")
        assert m.name == "test-plugin"
        assert m.version == "0.1.0"
        assert m.source == PluginSource.INSTALLED

    def test_create_full(self):
        m = PluginManifest(
            name="full-plugin",
            version="1.0.0",
            description="A full plugin",
            author="Test Author",
            license="Apache-2.0",
            homepage="https://example.com",
            source=PluginSource.PROJECT,
            dependencies=["dep1", "dep2"],
            min_yim_version="0.2.0",
            tags=["database", "tools"],
            provides_tools=["db_query"],
            provides_skills=["db_skill"],
            provides_hooks=["on_session_start"],
            provides_backends=["postgres"],
        )
        assert m.name == "full-plugin"
        assert m.author == "Test Author"
        assert m.license == "Apache-2.0"
        assert m.source == PluginSource.PROJECT
        assert "dep1" in m.dependencies
        assert "database" in m.tags
        assert "db_query" in m.provides_tools

    def test_plugin_source_enum(self):
        assert PluginSource.BUNDLED.value == "bundled"
        assert PluginSource.INSTALLED.value == "installed"
        assert PluginSource.PROJECT.value == "project"
        assert PluginSource.USER.value == "user"

    def test_to_dict(self):
        m = PluginManifest(name="dict-test", version="0.1.0")
        d = m.to_dict()
        assert d["name"] == "dict-test"
        assert d["version"] == "0.1.0"
        assert d["source"] == "installed"


class TestEncrePlugin:
    def test_plugin_with_tools(self):
        class MyPlugin(EncrePlugin):
            manifest = PluginManifest(name="my-plugin", version="1.0.0")

            def get_tools(self):
                return ["fake_tool"]

        plugin = MyPlugin()
        assert plugin.manifest.name == "my-plugin"
        assert plugin.get_tools() == ["fake_tool"]

    def test_plugin_with_hooks(self):
        class HookPlugin(EncrePlugin):
            manifest = PluginManifest(name="hook-plugin", version="1.0.0")

            def get_hooks(self):
                return [("on_session_start", lambda: None)]

        plugin = HookPlugin()
        hooks = plugin.get_hooks()
        assert len(hooks) == 1
        assert hooks[0][0] == "on_session_start"

    def test_plugin_default_returns_empty(self):
        class EmptyPlugin(EncrePlugin):
            manifest = PluginManifest(name="empty", version="0.1.0")

        plugin = EmptyPlugin()
        assert plugin.get_tools() == []
        assert plugin.get_skills() == []
        assert plugin.get_hooks() == []
        assert plugin.get_backends() == {}


class TestPluginRegistry:
    def test_empty_registry(self):
        registry = PluginRegistry()
        assert registry.count == 0
        assert registry.active_count == 0

    def test_register_plugin(self):
        class TestPlugin(EncrePlugin):
            manifest = PluginManifest(name="reg-test", version="1.0.0")

        registry = PluginRegistry()
        registry.register(TestPlugin())
        assert registry.count == 1

    def test_register_duplicate_name(self):
        class DupPlugin(EncrePlugin):
            manifest = PluginManifest(name="dup", version="1.0.0")

        registry = PluginRegistry()
        registry.register(DupPlugin())
        registry.register(DupPlugin())
        assert registry.count == 1

    def test_activate_deactivate(self):
        class ActPlugin(EncrePlugin):
            manifest = PluginManifest(name="act-test", version="1.0.0")

        registry = PluginRegistry()
        plugin = ActPlugin()
        registry.register(plugin)
        assert registry.activate("act-test") is True
        assert registry.active_count == 1
        assert registry.deactivate("act-test") is True
        assert registry.active_count == 0

    def test_unregister(self):
        class UnregPlugin(EncrePlugin):
            manifest = PluginManifest(name="unreg-test", version="1.0.0")

        registry = PluginRegistry()
        registry.register(UnregPlugin())
        assert registry.unregister("unreg-test") is True
        assert registry.count == 0

    def test_get(self):
        class GetPlugin(EncrePlugin):
            manifest = PluginManifest(name="get-test", version="1.0.0")

        registry = PluginRegistry()
        plugin = GetPlugin()
        registry.register(plugin)
        assert registry.get("get-test") is not None
        assert registry.get_manifest("get-test") is not None

    def test_list_all(self):
        class ListPlugin(EncrePlugin):
            manifest = PluginManifest(name="list-test", version="1.0.0")

        registry = PluginRegistry()
        registry.register(ListPlugin())
        manifests = registry.list_all()
        assert len(manifests) == 1
        assert manifests[0].name == "list-test"

    def test_get_all_tools(self):
        class ToolPlugin(EncrePlugin):
            manifest = PluginManifest(name="tool-plugin", version="1.0.0")

            def get_tools(self):
                return ["tool1", "tool2"]

        registry = PluginRegistry()
        plugin = ToolPlugin()
        registry.register(plugin)
        registry.activate("tool-plugin")
        tools = registry.get_all_tools()
        assert isinstance(tools, list)

    def test_get_all_skills(self):
        class SkillPlugin(EncrePlugin):
            manifest = PluginManifest(name="skill-plugin", version="1.0.0")

            def get_skills(self):
                return ["skill1"]

        registry = PluginRegistry()
        plugin = SkillPlugin()
        registry.register(plugin)
        registry.activate("skill-plugin")
        skills = registry.get_all_skills()
        assert isinstance(skills, list)

    def test_get_all_hooks(self):
        class HookPlugin(EncrePlugin):
            manifest = PluginManifest(name="hook-plugin2", version="1.0.0")

            def get_hooks(self):
                return [("pre_tool_exec", lambda: None)]

        registry = PluginRegistry()
        plugin = HookPlugin()
        registry.register(plugin)
        registry.activate("hook-plugin2")
        hooks = registry.get_all_hooks()
        assert isinstance(hooks, dict)

    def test_get_all_backends(self):
        class BackendPlugin(EncrePlugin):
            manifest = PluginManifest(name="be-plugin", version="1.0.0")

            def get_backends(self):
                return {"custom": "FakeBackend"}

        registry = PluginRegistry()
        plugin = BackendPlugin()
        registry.register(plugin)
        registry.activate("be-plugin")
        backends = registry.get_all_backends()
        assert isinstance(backends, dict)
        assert "custom" in backends

    def test_reset(self):
        class ResetPlugin(EncrePlugin):
            manifest = PluginManifest(name="reset-test", version="1.0.0")

        registry = PluginRegistry()
        registry.register(ResetPlugin())
        registry.activate("reset-test")
        registry.reset()
        assert registry.count == 0
        assert registry.active_count == 0
