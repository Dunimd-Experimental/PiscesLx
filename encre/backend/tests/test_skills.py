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

"""Tests for the skills registry, bundled skill definitions, and skill lookup."""

import pytest


class TestBundledSkillDefinition:
    def test_minimal_construction(self):
        from encre.skills.types import BundledSkillDefinition

        async def _prompt_fn(args, ctx):
            return "test prompt"

        skill = BundledSkillDefinition(
            name="test_skill",
            description="A test skill",
            get_prompt_for_command=_prompt_fn,
        )
        assert skill.name == "test_skill"
        assert skill.description == "A test skill"
        assert skill.get_prompt_for_command is _prompt_fn
        assert skill.aliases == []

    def test_all_fields_populated(self):
        from encre.skills.types import BundledSkillDefinition, SkillContext, SkillSource

        async def _prompt_fn(args, ctx):
            return "custom prompt"

        skill = BundledSkillDefinition(
            name="full_skill",
            description="Fully populated skill",
            get_prompt_for_command=_prompt_fn,
            aliases=["fs", "full"],
            when_to_use=".py .rs",
            argument_hint="[target: file to process]",
            allowed_tools=["bash", "grep"],
            model="gpt-4o",
            disable_model_invocation=False,
            user_invocable=True,
            context=SkillContext.INLINE,
            source=SkillSource.BUNDLED,
            file_path="/path/to/skill.md",
        )
        assert skill.name == "full_skill"
        assert skill.aliases == ["fs", "full"]
        assert skill.when_to_use == ".py .rs"
        assert skill.argument_hint == "[target: file to process]"
        assert skill.allowed_tools == ["bash", "grep"]
        assert skill.model == "gpt-4o"
        assert skill.disable_model_invocation is False
        assert skill.user_invocable is True
        assert skill.context == SkillContext.INLINE
        assert skill.source == SkillSource.BUNDLED
        assert skill.file_path == "/path/to/skill.md"

    def test_default_values(self):
        from encre.skills.types import BundledSkillDefinition, SkillContext, SkillSource

        async def _prompt_fn(args, ctx):
            return "default test"

        skill = BundledSkillDefinition(
            name="defaults",
            description="Testing defaults",
            get_prompt_for_command=_prompt_fn,
        )
        assert skill.aliases == []
        assert skill.when_to_use == ""
        assert skill.argument_hint == ""
        assert skill.allowed_tools is None
        assert skill.model is None
        assert skill.disable_model_invocation is False
        assert skill.user_invocable is True
        assert skill.context == SkillContext.INLINE
        assert skill.source == SkillSource.BUNDLED
        assert skill.file_path == ""

    @pytest.mark.asyncio
    async def test_get_prompt_for_command_with_args(self):
        from encre.skills.types import BundledSkillDefinition

        async def _prompt_fn(args, ctx):
            return f"debug {args or 'nothing'}"

        skill = BundledSkillDefinition(
            name="echo",
            description="Echo skill",
            get_prompt_for_command=_prompt_fn,
        )
        result = await skill.get_prompt_for_command("file.py", {})
        assert result == "debug file.py"

    @pytest.mark.asyncio
    async def test_get_prompt_for_command_with_context(self):
        from encre.skills.types import BundledSkillDefinition

        async def _prompt_fn(args, ctx):
            return f"mode={ctx.get('mode', 'default')}"

        skill = BundledSkillDefinition(
            name="ctx_skill",
            description="Context skill",
            get_prompt_for_command=_prompt_fn,
        )
        result = await skill.get_prompt_for_command(None, {"mode": "verbose"})
        assert result == "mode=verbose"


class TestEncreSkillRegistry:
    def test_create_registry(self):
        from encre.skills.registry import EncreSkillRegistry
        registry = EncreSkillRegistry()
        assert registry is not None

    def test_register_and_lookup_by_name(self):
        from encre.skills.registry import EncreSkillRegistry
        from encre.skills.types import BundledSkillDefinition

        async def _prompt_fn(args, ctx):
            return "hello"

        registry = EncreSkillRegistry()
        skill = BundledSkillDefinition(
            name="greet",
            description="Greeting skill",
            get_prompt_for_command=_prompt_fn,
        )
        registry.register(skill)
        found = registry.lookup("greet")
        assert found is not None
        assert found.name == "greet"
        assert found.description == "Greeting skill"

    def test_lookup_nonexistent_returns_none(self):
        from encre.skills.registry import EncreSkillRegistry
        registry = EncreSkillRegistry()
        assert registry.lookup("nonexistent") is None

    def test_lookup_by_alias(self):
        from encre.skills.registry import EncreSkillRegistry
        from encre.skills.types import BundledSkillDefinition

        async def _prompt_fn(args, ctx):
            return "alias test"

        registry = EncreSkillRegistry()
        skill = BundledSkillDefinition(
            name="original",
            description="Original skill",
            get_prompt_for_command=_prompt_fn,
            aliases=["orig", "og"],
        )
        registry.register(skill)

        found = registry.lookup("orig")
        assert found is not None
        assert found.name == "original"

        found2 = registry.lookup("og")
        assert found2 is not None
        assert found2.name == "original"

    def test_register_multiple_skills(self):
        from encre.skills.registry import EncreSkillRegistry
        from encre.skills.types import BundledSkillDefinition

        async def _prompt_fn(args, ctx):
            return "multi"

        registry = EncreSkillRegistry()
        skill_a = BundledSkillDefinition(
            name="alpha", description="Alpha", get_prompt_for_command=_prompt_fn
        )
        skill_b = BundledSkillDefinition(
            name="beta", description="Beta", get_prompt_for_command=_prompt_fn
        )
        registry.register(skill_a)
        registry.register(skill_b)
        assert registry.lookup("alpha") is not None
        assert registry.lookup("beta") is not None

    def test_list_all_returns_registered_skills(self):
        from encre.skills.registry import EncreSkillRegistry
        from encre.skills.types import BundledSkillDefinition

        async def _prompt_fn(args, ctx):
            return "list"

        registry = EncreSkillRegistry()
        skill = BundledSkillDefinition(
            name="listable", description="Listable", get_prompt_for_command=_prompt_fn
        )
        registry.register(skill)

        all_skills = registry.list_all()
        assert len(all_skills) >= 1
        names = [s.name for s in all_skills]
        assert "listable" in names

    def test_register_with_same_source_priority_overwrites(self):
        from encre.skills.registry import EncreSkillRegistry
        from encre.skills.types import BundledSkillDefinition, SkillSource

        async def _prompt_fn(args, ctx):
            return "first"

        async def _prompt_fn2(args, ctx):
            return "second"

        registry = EncreSkillRegistry()
        skill1 = BundledSkillDefinition(
            name="same", description="First", get_prompt_for_command=_prompt_fn,
            source=SkillSource.BUNDLED
        )
        skill2 = BundledSkillDefinition(
            name="same", description="Second", get_prompt_for_command=_prompt_fn2,
            source=SkillSource.BUNDLED
        )
        registry.register(skill1)
        registry.register(skill2)
        # With same priority (BUNDLED=3), the second should NOT overwrite
        # because new_priority >= old_priority returns early
        found = registry.lookup("same")
        assert found is not None
        assert found.description == "First"

    def test_higher_priority_overwrites_lower(self):
        from encre.skills.registry import EncreSkillRegistry
        from encre.skills.types import BundledSkillDefinition, SkillSource

        async def _prompt_fn(args, ctx):
            return "managed"

        async def _prompt_fn2(args, ctx):
            return "bundled"

        registry = EncreSkillRegistry()
        skill_bundled = BundledSkillDefinition(
            name="override_test", description="Bundled version",
            get_prompt_for_command=_prompt_fn2, source=SkillSource.BUNDLED
        )
        skill_managed = BundledSkillDefinition(
            name="override_test", description="Managed version",
            get_prompt_for_command=_prompt_fn, source=SkillSource.MANAGED
        )
        registry.register(skill_bundled)
        registry.register(skill_managed)
        # MANAGED (0) has higher priority than BUNDLED (3), should overwrite
        found = registry.lookup("override_test")
        assert found.description == "Managed version"

    @pytest.mark.asyncio
    async def test_activate_returns_prompt(self):
        from encre.skills.registry import EncreSkillRegistry
        from encre.skills.types import BundledSkillDefinition

        async def _prompt_fn(args, ctx):
            return "activated prompt content"

        registry = EncreSkillRegistry()
        skill = BundledSkillDefinition(
            name="activable", description="Activatable",
            get_prompt_for_command=_prompt_fn
        )
        registry.register(skill)
        result = await registry.activate("activable")
        assert result == "activated prompt content"

    @pytest.mark.asyncio
    async def test_activate_nonexistent_returns_error(self):
        from encre.skills.registry import EncreSkillRegistry
        registry = EncreSkillRegistry()
        result = await registry.activate("ghost")
        assert "not found" in result


class TestCreateBundledSkills:
    def test_create_bundled_skills_populates_registry(self):
        from encre.skills.bundled import create_bundled_skills
        from encre.skills.registry import EncreSkillRegistry
        registry = EncreSkillRegistry()
        create_bundled_skills(registry)

        # All 5 bundled skills should be registered
        debug = registry.lookup("debug")
        assert debug is not None
        assert debug.name == "debug"
        assert "debug" in debug.description.lower() or "Debug" in debug.description

        loop = registry.lookup("loop")
        assert loop is not None
        assert loop.name == "loop"

        batch = registry.lookup("batch")
        assert batch is not None
        assert batch.name == "batch"

        verify = registry.lookup("verify")
        assert verify is not None
        assert verify.name == "verify"

        stuck = registry.lookup("stuck")
        assert stuck is not None
        assert stuck.name == "stuck"

    def test_bundled_skill_lookup_by_alias(self):
        from encre.skills.bundled import create_bundled_skills
        from encre.skills.registry import EncreSkillRegistry
        registry = EncreSkillRegistry()
        create_bundled_skills(registry)

        # debug aliases: dbg, diag, troubleshoot
        found = registry.lookup("dbg")
        assert found is not None
        assert found.name == "debug"

        # loop aliases: repeat, schedule, watch
        found = registry.lookup("schedule")
        assert found is not None
        assert found.name == "loop"

        # batch aliases: parallel, multi-agent, farm, orchestrate
        found = registry.lookup("parallel")
        assert found is not None
        assert found.name == "batch"

    def test_list_all_after_create_bundled_skills(self):
        from encre.skills.bundled import create_bundled_skills
        from encre.skills.registry import EncreSkillRegistry
        registry = EncreSkillRegistry()
        create_bundled_skills(registry)

        all_skills = registry.list_all()
        skill_names = {s.name for s in all_skills}
        assert skill_names >= {"debug", "loop", "batch", "verify", "stuck"}

    @pytest.mark.asyncio
    async def test_bundled_skill_activation(self):
        from encre.skills.bundled import create_bundled_skills
        from encre.skills.registry import EncreSkillRegistry
        registry = EncreSkillRegistry()
        create_bundled_skills(registry)

        result = await registry.activate("debug")
        assert result is not None
        assert len(result) > 0

    def test_bundled_skill_sources(self):
        from encre.skills.bundled import create_bundled_skills
        from encre.skills.registry import EncreSkillRegistry
        from encre.skills.types import SkillSource
        registry = EncreSkillRegistry()
        create_bundled_skills(registry)

        for skill in registry.list_all():
            assert skill.source == SkillSource.BUNDLED
