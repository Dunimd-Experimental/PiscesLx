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

"""Tests for the prompt system: base classes, builder, templates, and specializations."""

import pytest
from encre.utils.types import PermissionMode


class TestEncreBasePrompt:
    def test_base_prompt_is_abstract(self):
        from encre.prompts.base import EncreBasePrompt
        with pytest.raises(TypeError):
            EncreBasePrompt()  # Cannot instantiate ABC

    def test_base_prompt_has_abstract_methods(self):
        from encre.prompts.base import EncreBasePrompt
        assert hasattr(EncreBasePrompt, "build_system_prompt")
        assert hasattr(EncreBasePrompt, "build_tool_instructions")


class TestEncrePromptTemplate:
    def test_construction_defaults(self):
        from encre.prompts.base import EncrePromptTemplate
        tmpl = EncrePromptTemplate()
        assert tmpl is not None
        assert tmpl._specialty == "general"
        assert tmpl._builder is not None

    def test_construction_with_specialty(self):
        from encre.prompts.base import EncrePromptTemplate
        tmpl = EncrePromptTemplate(specialty="coding")
        assert tmpl._specialty == "coding"

    def test_construction_with_custom_builder(self):
        from encre.prompts.base import EncrePromptTemplate
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        tmpl = EncrePromptTemplate(builder=builder, specialty="research")
        assert tmpl._builder is builder
        assert tmpl._specialty == "research"

    def test_builder_property(self):
        from encre.prompts.base import EncrePromptTemplate
        tmpl = EncrePromptTemplate(specialty="data")
        assert tmpl.builder is tmpl._builder

    def test_build_system_prompt_returns_string(self):
        from encre.prompts.base import EncrePromptTemplate
        tmpl = EncrePromptTemplate(specialty="general")
        result = tmpl.build_system_prompt(mode="default")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_system_prompt_with_tools(self):
        from encre.prompts.base import EncrePromptTemplate
        tmpl = EncrePromptTemplate()
        tools = [
            {"function": {"name": "bash", "description": "Execute shell commands"}},
            {"function": {"name": "read", "description": "Read files"}},
        ]
        result = tmpl.build_system_prompt(mode="default", tools=tools)
        assert "bash" in result
        assert "read" in result

    def test_build_system_prompt_with_custom_instructions(self):
        from encre.prompts.base import EncrePromptTemplate
        tmpl = EncrePromptTemplate()
        result = tmpl.build_system_prompt(
            mode="default",
            custom_instructions="Always use Python 3.12 syntax.",
        )
        assert "Python 3.12" in result

    def test_build_system_prompt_reflects_specialty(self):
        from encre.prompts.base import EncrePromptTemplate
        coding_tmpl = EncrePromptTemplate(specialty="coding")
        research_tmpl = EncrePromptTemplate(specialty="research")

        coding_result = coding_tmpl.build_system_prompt(mode="default")
        research_result = research_tmpl.build_system_prompt(mode="default")

        # Different specialties produce different prompts
        assert coding_result != research_result
        assert "Software Engineering" in coding_result
        assert "Research" in research_result

    def test_build_system_prompt_reflects_permission_mode(self):
        from encre.prompts.base import EncrePromptTemplate
        tmpl = EncrePromptTemplate()

        bypass_result = tmpl.build_system_prompt(mode="bypass")
        plan_result = tmpl.build_system_prompt(mode="plan")

        assert "bypass" in bypass_result.lower()
        assert "plan" in plan_result.lower()

    def test_build_tool_instructions_empty_list(self):
        from encre.prompts.base import EncrePromptTemplate
        tmpl = EncrePromptTemplate()
        result = tmpl.build_tool_instructions([])
        assert "do not have access" in result.lower()

    def test_build_tool_instructions_with_names(self):
        from encre.prompts.base import EncrePromptTemplate
        tmpl = EncrePromptTemplate()
        result = tmpl.build_tool_instructions(["bash", "grep", "glob"])
        assert "bash" in result
        assert "grep" in result
        assert "glob" in result
        assert "Use them as needed" in result


class TestPromptBlock:
    def test_prompt_block_construction(self):
        from encre.prompts.system import PromptBlock
        block = PromptBlock(priority=10, name="test_block", content="Test content")
        assert block.priority == 10
        assert block.name == "test_block"
        assert block.content == "Test content"

    def test_prompt_block_with_context(self):
        from encre.prompts.system import PromptBlock
        block = PromptBlock(
            priority=50,
            name="templated",
            content="Hello {{username}}, welcome to {{project}}.",
        )
        ctx = {"username": "Alice", "project": "Encre"}
        filled = block.with_context(ctx)
        assert "Hello Alice" in filled.content
        assert "welcome to Encre" in filled.content
        assert filled.name == "templated"
        assert filled.priority == 50


class TestEncrePromptBuilder:
    def test_builder_construction(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        assert builder is not None
        assert builder._blocks == {}

    def test_add_block(self):
        from encre.prompts.system import EncrePromptBuilder, PromptBlock
        builder = EncrePromptBuilder()
        block = PromptBlock(priority=100, name="extra", content="Extra instructions")
        builder.add_block(block)
        assert "extra" in builder._blocks
        assert builder._blocks["extra"].content == "Extra instructions"

    def test_remove_block(self):
        from encre.prompts.system import EncrePromptBuilder, PromptBlock
        builder = EncrePromptBuilder()
        block = PromptBlock(priority=100, name="temporary", content="Temp")
        builder.add_block(block)
        assert "temporary" in builder._blocks
        builder.remove_block("temporary")
        assert "temporary" not in builder._blocks

    def test_remove_nonexistent_block_does_not_raise(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        builder.remove_block("nonexistent")  # Should not raise

    def test_add_custom_instructions(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        builder.add_custom_instructions("Focus on testing.")
        assert "custom" in builder._blocks
        assert "Focus on testing" in builder._blocks["custom"].content
        assert builder._blocks["custom"].priority == 200

    def test_build_default(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        result = builder.build()
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain default blocks
        assert "identity" in result.lower() or "helpful" in result.lower()

    def test_build_coding_specialty(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        result = builder.build(specialty="coding")
        assert "Software Engineering" in result

    def test_build_research_specialty(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        result = builder.build(specialty="research")
        assert "Research" in result

    def test_build_data_specialty(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        result = builder.build(specialty="data")
        assert "Data Analysis" in result

    def test_build_unknown_specialty_falls_back_to_general(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        result = builder.build(specialty="unknown_specialty")
        assert "General Assistant" in result

    def test_build_with_permission_mode(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()

        bypass = builder.build(mode="bypass")
        default = builder.build(mode="default")

        assert "full autonomy" in bypass.lower()
        assert "Ask for permission" in default

    def test_build_with_tools(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        tools = [{"function": {"name": "test_tool", "description": "A test tool"}}]
        result = builder.build(tools=tools)
        assert "test_tool" in result
        assert "A test tool" in result

    def test_build_with_custom_instructions(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        result = builder.build(custom_instructions="ALWAYS validate input first.")
        assert "ALWAYS validate input first" in result

    def test_build_with_context(self):
        from encre.prompts.system import EncrePromptBuilder
        builder = EncrePromptBuilder()
        result = builder.build_with_context(
            ctx={"username": "TestUser"},
            specialty="general",
        )
        # The identity block doesn't have {{username}} but the method
        # should still work without errors
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_with_context_variable_substitution(self):
        from encre.prompts.system import EncrePromptBuilder, PromptBlock
        builder = EncrePromptBuilder()
        builder.add_block(PromptBlock(
            priority=200,
            name="context_block",
            content="User {{user}} using version {{version}}",
        ))
        result = builder.build_with_context(
            ctx={"user": "Alice", "version": "1.0.0"},
            specialty="general",
        )
        assert "User Alice" in result
        assert "version 1.0.0" in result

    def test_custom_block_can_override_default(self):
        from encre.prompts.system import EncrePromptBuilder, PromptBlock
        builder = EncrePromptBuilder()
        # Override the identity block
        builder.add_block(PromptBlock(
            priority=0,
            name="identity",
            content="You are a friendly assistant.",
        ))
        result = builder.build()
        assert "friendly assistant" in result


class TestSpecializationPrompts:
    def test_coding_prompt_specialty(self):
        from encre.prompts.coding import EncreCodingPrompt
        cp = EncreCodingPrompt()
        assert cp._specialty == "coding"
        result = cp.build_system_prompt(mode="default")
        assert "Software Engineering" in result

    def test_general_prompt_specialty(self):
        from encre.prompts.general import EncreGeneralPrompt
        gp = EncreGeneralPrompt()
        assert gp._specialty == "general"
        result = gp.build_system_prompt(mode="default")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_research_prompt_specialty(self):
        from encre.prompts.research import EncreResearchPrompt
        rp = EncreResearchPrompt()
        assert rp._specialty == "research"
        result = rp.build_system_prompt(mode="default")
        assert "Research" in result

    def test_data_prompt_specialty(self):
        from encre.prompts.data import EncreDataPrompt
        dp = EncreDataPrompt()
        assert dp._specialty == "data"
        result = dp.build_system_prompt(mode="default")
        assert "Data Analysis" in result

    def test_specialization_build_with_tools_and_custom_instructions(self):
        from encre.prompts.coding import EncreCodingPrompt
        cp = EncreCodingPrompt()
        tools = [{"function": {"name": "bash", "description": "Run bash commands"}}]
        result = cp.build_system_prompt(
            mode="default",
            tools=tools,
            custom_instructions="Always write docstrings.",
        )
        assert "bash" in result
        assert "Always write docstrings" in result
        assert "Software Engineering" in result
