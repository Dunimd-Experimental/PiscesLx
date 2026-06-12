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

from __future__ import annotations
from encre.config import SubAgentConfig
from encre.prompts.loader import PromptLoader

_loader = PromptLoader()


def get_builtin_sub_agents() -> list[SubAgentConfig]:
    return [
        # ── General mode ──────────────────────────────────────
        SubAgentConfig(
            name="coder",
            description="Focused implementation agent for writing code",
            system_prompt=_loader.load("coder", category="skills"),
            hidden=True,
        ),
        SubAgentConfig(
            name="researcher",
            description="Thorough research agent for gathering and analyzing information",
            system_prompt=_loader.load("researcher", category="skills"),
            hidden=True,
        ),
        SubAgentConfig(
            name="critic",
            description="Quality auditor for reviewing code and other work",
            system_prompt=_loader.load("critic", category="skills"),
            hidden=True,
        ),
        # ── Workspace mode ────────────────────────────────────
        SubAgentConfig(
            name="architect",
            description="System designer for planning architecture before implementation",
            system_prompt=_loader.load("architect", category="skills"),
            hidden=True,
        ),
        SubAgentConfig(
            name="planner",
            description="Task breakdown specialist for decomposing goals into actionable steps",
            system_prompt=_loader.load("planner", category="skills"),
            hidden=True,
        ),
        # ── Plan/Spec mode ────────────────────────────────────
        SubAgentConfig(
            name="spec-writer",
            description="Requirements and specification specialist for writing precise specs",
            system_prompt=_loader.load("spec_writer", category="skills"),
            hidden=True,
        ),
        # ── Auto agent (lobster) mode ─────────────────────────
        SubAgentConfig(
            name="automation",
            description="Autonomous workflow orchestrator for monitoring, execution, and scheduling",
            system_prompt=_loader.load("automation", category="skills"),
            hidden=True,
        ),
        SubAgentConfig(
            name="monitor",
            description="Event and state watcher for detecting changes and triggers",
            system_prompt=_loader.load("monitor", category="skills"),
            hidden=True,
        ),
        SubAgentConfig(
            name="executor",
            description="Focused task completion agent for executing well-defined tasks",
            system_prompt=_loader.load("executor", category="skills"),
            hidden=True,
        ),
        SubAgentConfig(
            name="scheduler",
            description="Task and time orchestrator for managing schedules and retries",
            system_prompt=_loader.load("scheduler", category="skills"),
            hidden=True,
        ),
    ]
