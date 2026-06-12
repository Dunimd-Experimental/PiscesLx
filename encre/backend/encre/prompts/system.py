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

from dataclasses import dataclass, field
from typing import Any

from encre.prompts.loader import PromptLoader
from encre.utils.types import PermissionMode

_loader = PromptLoader()

# ── Block definitions ──────────────────────────────────────────────


@dataclass
class PromptBlock:
    priority: int
    name: str
    content: str
    condition: list[str] | None = None  # intents that trigger this block; None = always

    def with_context(self, ctx: dict[str, str]) -> PromptBlock:
        content = self.content
        for key, val in ctx.items():
            content = content.replace(f"{{{{{key}}}}}", val)
        return PromptBlock(
            priority=self.priority, name=self.name, content=content,
            condition=self.condition,
        )


# ── Core presets ────────────────────────────────────────────────────


def _identity_block() -> PromptBlock:
    return PromptBlock(priority=0, name="identity", condition=None, content=_loader.load("identity"))


def _tool_usage_block(tools: list[dict[str, Any]] | None = None) -> PromptBlock:
    return PromptBlock(
        priority=10, name="tool_usage",
        condition=None,
        content=_loader.load("tool_usage"),
    )


def _permission_block(mode: PermissionMode) -> PromptBlock:
    if mode == "bypass":
        guidance = "You have full autonomy to execute any tool without asking for permission. Use this responsibly."
    elif mode == "dont_ask":
        guidance = "Execute tasks directly without asking for confirmation. Only pause if an operation appears destructive and irreversible."
    elif mode == "accept_edits":
        guidance = "You may read, write, and edit files freely. Shell commands and web requests may require confirmation."
    elif mode == "plan":
        guidance = "First create a clear plan. Present it to the user for approval before executing any changes."
    elif mode == "spec":
        guidance = "First produce a complete specification. Present it to the user for review and approval before writing any code or making changes."
    elif mode == "auto":
        guidance = "Most operations are auto-approved. Dangerous operations (rm -rf, chmod 777, etc.) require confirmation."
    else:
        guidance = "Ask for permission before executing tools that modify files, run shell commands, or access the network."

    return PromptBlock(
        priority=20, name="permission", condition=None,
        content=_loader.load_with_context("permission", mode=mode, guidance=guidance),
    )


def _language_block(lang_pref: str, app_lang: str) -> PromptBlock | None:
    resolved = lang_pref if lang_pref != "auto" else app_lang
    if resolved == "zh":
        instruction = "IMPORTANT: You must always respond in Chinese (中文) throughout the entire conversation. Even if the user writes in another language, you must reply in Chinese. Do not switch to other languages under any circumstances."
    elif resolved == "en":
        instruction = "IMPORTANT: You must always respond in English throughout the entire conversation. Even if the user writes in another language, you must reply in English. Do not switch to other languages under any circumstances."
    else:
        return None
    return PromptBlock(priority=25, name="language", condition=None, content=instruction)


def _output_format_block() -> PromptBlock:
    return PromptBlock(priority=30, name="output_format", condition=["general", "coding", "data"], content=_loader.load("output_format"))


def _safety_block() -> PromptBlock:
    return PromptBlock(priority=5, name="safety", condition=None, content=_loader.load("safety"))


def _task_management_block() -> PromptBlock:
    return PromptBlock(priority=15, name="task_management", condition=["coding", "data"], content=_loader.load("task_management"))


def _specialty_coding_block() -> PromptBlock:
    return PromptBlock(priority=100, name="specialty", condition=["coding"], content=_loader.load("specialty_coding"))


def _specialty_research_block() -> PromptBlock:
    return PromptBlock(priority=100, name="specialty", condition=["research"], content=_loader.load("specialty_research"))


def _specialty_data_block() -> PromptBlock:
    return PromptBlock(priority=100, name="specialty", condition=["data"], content=_loader.load("specialty_data"))


def _specialty_general_block() -> PromptBlock:
    return PromptBlock(priority=100, name="specialty", condition=None, content=_loader.load("specialty_general"))


def _iwork_block(workspace_root: str, workspace_name: str, project_summary: str = "") -> PromptBlock:
    ctx = dict(workspace_name=workspace_name, workspace_root=workspace_root)
    content = _loader.load_with_context("workspace_mode", **ctx)
    if project_summary:
        snapshot = f"\n\n### Project Snapshot\n{project_summary}"
        content = content.replace("{{project_snapshot}}", snapshot)
    else:
        content = content.replace("{{project_snapshot}}", "")
    return PromptBlock(priority=2, name="iwork", condition=None, content=content)


def _plan_mode_block() -> PromptBlock:
    content = _loader.load("planner", category="skills")
    content = content.replace(
        "## Planner Sub-Agent — Task Breakdown Specialist\n\nYou are a planning sub-agent. Your job is to break down a goal into concrete, actionable tasks.",
        "## Planning Mode\n\nYou are in **planning mode**. Present a clear plan to the user for approval before executing any changes. Break the goal into concrete, actionable tasks.",
    )
    return PromptBlock(priority=50, name="plan_mode", condition=None, content=content)


def _spec_mode_block() -> PromptBlock:
    content = _loader.load("spec_writer", category="skills")
    content = content.replace(
        "## Spec Writer Sub-Agent — Requirements & Specification Specialist\n\nYou are a specification sub-agent. Your job is to transform ambiguous requirements into precise, complete, and actionable specifications.",
        "## Specification Mode\n\nYou are in **specification mode**. Produce a complete specification and present it to the user for review before writing any code. Transform the requirements into a precise, complete, and actionable specification.",
    )
    return PromptBlock(priority=50, name="spec_mode", condition=None, content=content)


def _normal_mode_block() -> PromptBlock:
    return PromptBlock(priority=2, name="mode", condition=None, content=_loader.load("general_mode"))


def _environment_block() -> PromptBlock:
    import platform as _platform
    import sys as _sys

    os_name = _platform.system() or _sys.platform
    if os_name == "Windows":
        details = f"Windows {_platform.version()} ({_platform.machine()})"
        shell_hint = "Use cmd, PowerShell, or bash commands. File paths use backslashes (\\)."
    elif os_name == "Darwin":
        details = f"macOS {_platform.mac_ver()[0]} ({_platform.machine()})"
        shell_hint = "Use bash or zsh. File paths use forward slashes (/)."
    elif os_name == "Linux":
        details = f"Linux ({_platform.machine()})"
        shell_hint = "Use bash. File paths use forward slashes (/)."
    else:
        details = os_name
        shell_hint = ""

    content = (
        f"You are running on: **{os_name}** — {details}\n"
        f"{shell_hint}"
    ).strip()
    return PromptBlock(priority=8, name="environment", condition=None, content=content)


def _current_datetime_block() -> PromptBlock:
    """Inject current date and time so the model has temporal awareness."""
    from datetime import datetime as _dt
    now = _dt.now()
    content = (
        f"## Current Date & Time\n"
        f"Today is: **{now.strftime('%A, %B %d, %Y')}**\n"
        f"Current time: **{now.strftime('%H:%M:%S')}**\n"
        f"Current year: **{now.year}**\n"
        f"\n"
        f"Use this temporal context when the task involves time-sensitive information, "
        f"news, events, scheduling, or any scenario where recency matters."
    ).strip()
    return PromptBlock(priority=9, name="current_datetime", condition=None, content=content)


# ── Builder ─────────────────────────────────────────────────────────


class EncrePromptBuilder:
    """Layered system prompt builder with priority-based block assembly."""

    def __init__(self) -> None:
        self._blocks: dict[str, PromptBlock] = {}

    def add_block(self, block: PromptBlock) -> None:
        self._blocks[block.name] = block

    def remove_block(self, name: str) -> None:
        self._blocks.pop(name, None)

    def add_custom_instructions(self, text: str) -> None:
        self.add_block(PromptBlock(priority=200, name="custom", content=text))

    def build(
        self,
        mode: PermissionMode = "default",
        tools: list[dict[str, Any]] | None = None,
        specialty: str = "general",
        custom_instructions: str = "",
        intents: list[str] | None = None,
        workspace_root: str = "",
        workspace_name: str = "",
        project_summary: str = "",
        language_preference: str = "auto",
        app_language: str = "zh",
    ) -> str:
        intents = intents or ["general"]

        # Collect blocks
        blocks: dict[str, PromptBlock] = dict(self._blocks)

        # Mode header — iWork takes priority over normal
        if workspace_root:
            mode_block = _iwork_block(workspace_root, workspace_name or workspace_root, project_summary)
        else:
            mode_block = _normal_mode_block()
        blocks[mode_block.name] = mode_block

        # Always-add core blocks (if not overridden)
        defaults = [
            _identity_block(),
            _safety_block(),
            _current_datetime_block(),
            _environment_block(),
            _tool_usage_block(tools),
            _task_management_block(),
            _permission_block(mode),
            _language_block(language_preference, app_language),
            _output_format_block(),
        ]
        for block in defaults:
            if block.name not in blocks:
                blocks[block.name] = block

        # Plan/Spec mode blocks — inject detailed instructions when in that mode
        if mode == "plan" and "plan_mode" not in blocks:
            blocks["plan_mode"] = _plan_mode_block()
        elif mode == "spec" and "spec_mode" not in blocks:
            blocks["spec_mode"] = _spec_mode_block()

        # Specialty block (if not overridden)
        if "specialty" not in blocks:
            specialty_map: dict[str, PromptBlock] = {}
            if "coding" in intents:
                specialty_map["coding"] = _specialty_coding_block()
            if "research" in intents:
                specialty_map["research"] = _specialty_research_block()
            if "data" in intents:
                specialty_map["data"] = _specialty_data_block()
            # specific specialty takes priority, fall back to general
            if specialty in specialty_map:
                blocks["specialty"] = specialty_map[specialty]
            elif specialty_map:
                blocks["specialty"] = next(iter(specialty_map.values()))
            else:
                blocks["specialty"] = _specialty_general_block()

        # Custom instructions
        if custom_instructions:
            blocks["custom"] = PromptBlock(
                priority=200, name="custom", condition=None, content=custom_instructions,
            )

        # Filter by condition, then sort by priority, then assemble
        filtered: list[PromptBlock] = []
        for block in blocks.values():
            if block.condition is None:
                filtered.append(block)
            elif any(i in block.condition for i in intents):
                filtered.append(block)

        sorted_blocks = sorted(filtered, key=lambda b: b.priority)
        parts: list[str] = []
        for block in sorted_blocks:
            content = block.content.strip()
            if content:
                parts.append(content)

        return "\n\n".join(parts)

    def build_with_context(
        self,
        ctx: dict[str, str],
        mode: PermissionMode = "default",
        tools: list[dict[str, Any]] | None = None,
        specialty: str = "general",
    ) -> str:
        """Build with template variable substitution ({{key}} replaced by ctx values)."""
        prompt = self.build(mode=mode, tools=tools, specialty=specialty)
        for key, val in ctx.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", val)
        return prompt
