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

from encre.skills.types import BundledSkillDefinition, SkillContext, SkillSource


def create_bundled_skills(registry):
    from encre.skills.bundled.batch import _batch_prompt
    from encre.skills.bundled.code_review import _code_review_prompt
    from encre.skills.bundled.data_viz import _data_viz_prompt
    from encre.skills.bundled.debug import _debug_prompt
    from encre.skills.bundled.gen_test import _gen_test_prompt
    from encre.skills.bundled.loop import _loop_prompt
    from encre.skills.bundled.refactor import _refactor_prompt
    from encre.skills.bundled.stuck import _stuck_prompt
    from encre.skills.bundled.verify import _verify_prompt
    from encre.skills.bundled.web_research import _web_research_prompt
    from encre.skills.bundled.write_docs import _write_docs_prompt

    debug_skill = BundledSkillDefinition(
        name="debug",
        description=(
            "Systematic debugging workflow: gather logs, analyze root cause,"
            " isolate, fix, and verify errors"
        ),
        get_prompt_for_command=_debug_prompt,
        aliases=["dbg", "diag", "troubleshoot"],
        when_to_use=".log .txt .err .out .traceback",
        argument_hint="[target: file, module, or component to debug]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    loop_skill = BundledSkillDefinition(
        name="loop",
        description="Execute a command repeatedly on a schedule using [interval] <prompt> syntax",
        get_prompt_for_command=_loop_prompt,
        aliases=["repeat", "schedule", "watch"],
        when_to_use="",
        argument_hint="[seconds] <task description>",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    batch_skill = BundledSkillDefinition(
        name="batch",
        description=(
            "3-phase batch execution: research/plan, spawn parallel agents,"
            " track and synthesize results"
        ),
        get_prompt_for_command=_batch_prompt,
        aliases=["parallel", "multi-agent", "farm", "orchestrate"],
        when_to_use="",
        argument_hint="[high-level task description for batch processing]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.FORK,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    verify_skill = BundledSkillDefinition(
        name="verify",
        description=(
            "Code verification pipeline: static analysis, type checking,"
            " linting, test execution, build check, smoke test"
        ),
        get_prompt_for_command=_verify_prompt,
        aliases=["check", "validate", "test", "qa"],
        when_to_use=".py .rs .js .ts .go .java .cpp .c .h",
        argument_hint="[files or directories to verify, or 'all' for entire project]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    stuck_skill = BundledSkillDefinition(
        name="stuck",
        description=(
            "Self-diagnosis for stuck/looping agents: detect patterns,"
            " identify root cause, and apply recovery strategies"
        ),
        get_prompt_for_command=_stuck_prompt,
        aliases=["unstuck", "recover", "diagnose-loop", "self-fix"],
        when_to_use="",
        argument_hint="[description of what the agent is stuck on]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    web_research_skill = BundledSkillDefinition(
        name="web_research",
        description=(
            "Professional web research: multi-query discovery,"
            " cross-validation, source analysis, and structured synthesis"
        ),
        get_prompt_for_command=_web_research_prompt,
        aliases=["research", "search", "investigate"],
        when_to_use="",
        argument_hint="[topic or question to research]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    code_review_skill = BundledSkillDefinition(
        name="code_review",
        description=(
            "Expert code audit: correctness, security, performance,"
            " maintainability, and codebase fit analysis"
        ),
        get_prompt_for_command=_code_review_prompt,
        aliases=["review", "audit", "inspect"],
        when_to_use=".py .rs .js .ts .go .java .cpp .c .h",
        argument_hint="[files, modules, or pull request to review]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    refactor_skill = BundledSkillDefinition(
        name="refactor",
        description=(
            "Behavior-preserving code transformation: extract, rename,"
            " decouple, and restructure with zero regression"
        ),
        get_prompt_for_command=_refactor_prompt,
        aliases=["restructure", "cleanup", "improve"],
        when_to_use=".py .rs .js .ts .go .java",
        argument_hint="[files, modules, or components to refactor]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    write_docs_skill = BundledSkillDefinition(
        name="write_docs",
        description=(
            "Technical documentation writer: API reference, README, ADR,"
            " changelog, and tutorials with quality rigor"
        ),
        get_prompt_for_command=_write_docs_prompt,
        aliases=["document", "docs", "doc"],
        when_to_use="",
        argument_hint="[code, API, or project to document]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    data_viz_skill = BundledSkillDefinition(
        name="data_viz",
        description=(
            "Data analysis and visualization: descriptive stats, exploratory"
            " analysis, chart selection, and rigorous communication"
        ),
        get_prompt_for_command=_data_viz_prompt,
        aliases=["viz", "chart", "plot", "analytics"],
        when_to_use=".csv .json .xlsx .data",
        argument_hint="[data file or description of data to analyze]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    gen_test_skill = BundledSkillDefinition(
        name="gen_test",
        description=(
            "Test generation: unit tests, edge cases, error paths, and"
            " integration tests following project conventions"
        ),
        get_prompt_for_command=_gen_test_prompt,
        aliases=["test", "unittest", "spec"],
        when_to_use=".py .rs .js .ts .go .java",
        argument_hint="[files or modules to generate tests for]",
        disable_model_invocation=False,
        user_invocable=True,
        context=SkillContext.INLINE,
        source=SkillSource.BUNDLED,
        hidden=True,
    )

    registry.register(debug_skill)
    registry.register(loop_skill)
    registry.register(batch_skill)
    registry.register(verify_skill)
    registry.register(stuck_skill)
    registry.register(web_research_skill)
    registry.register(code_review_skill)
    registry.register(refactor_skill)
    registry.register(write_docs_skill)
    registry.register(data_viz_skill)
    registry.register(gen_test_skill)
