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
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Callable

from encre.agent import EncreAgent

logger = logging.getLogger("encre.learning.skill_generator")


class SkillGenerator:
    def __init__(
        self,
        agent: EncreAgent,
        *,
        enrich_fn: Callable[[list[str], str], dict[str, Any]] | None = None,
    ) -> None:
        self._agent = agent
        self._enrich_fn = enrich_fn

    def generate(self, tool_names: list[str], prompt: str) -> dict[str, Any] | None:
        tool_counts: dict[str, int] = {}
        for name in tool_names:
            tool_counts[name] = tool_counts.get(name, 0) + 1

        top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:5]
        if not top_tools:
            return None

        name_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
        skill_name = f"auto-{top_tools[0][0]}-{name_hash}"

        enrichment: dict[str, Any] = {}
        if self._enrich_fn:
            try:
                enrichment = self._enrich_fn(tool_names, prompt)
            except Exception:
                pass

        description = enrichment.get("description") or self._build_description(prompt, top_tools)
        when_to_use = enrichment.get("when_to_use") or self._build_when_to_use(top_tools)
        steps = enrichment.get("steps") or self._build_steps(top_tools)
        generated_reason = enrichment.get("reason", "Repeated tool-use pattern detected")

        body_parts = [
            f"# {skill_name}",
            "",
            description,
            "",
            "## When to Use",
            "",
            when_to_use or "Tasks matching the detected tool pattern",
            "",
            "## Tools",
            "",
        ]
        for tool_name, count in top_tools:
            body_parts.append(f"- `{tool_name}` (used {count} times)")
        body_parts.extend(["", "## Steps", ""])
        for step in steps:
            body_parts.append(f"- {step}")

        body = "\n".join(body_parts)

        return {
            "name": skill_name,
            "description": description,
            "aliases": [skill_name],
            "source": "auto_generated",
            "when_to_use": when_to_use,
            "body": body,
            "generated_at": time.time(),
            "tool_names": tool_names,
            "reason": generated_reason,
        }

    def _build_description(self, prompt: str, top_tools: list[tuple[str, int]]) -> str:
        tools_str = ", ".join(f"{name}" for name, _ in top_tools[:3])
        prompt_preview = prompt[:80].replace("\n", " ")
        return f"Automatically generated skill for task involving {tools_str}. Prompt: {prompt_preview}"

    def _build_when_to_use(self, top_tools: list[tuple[str, int]]) -> str:
        patterns = []
        for name, _ in top_tools[:3]:
            if name in ("bash", "run_command", "execute"):
                patterns.append("*.py, *.sh, *.toml, *.json")
            elif name in ("file_read", "read_file", "grep", "glob"):
                patterns.append("source files, logs, configuration")
            elif name in ("web_search", "web_fetch", "web"):
                patterns.append("web research, documentation lookup")
            else:
                patterns.append(f"tasks requiring `{name}`")
        return ", ".join(patterns)

    def _build_steps(self, top_tools: list[tuple[str, int]]) -> list[str]:
        steps: list[str] = []
        for name, count in top_tools:
            if name in ("bash", "run_command", "execute"):
                steps.append(f"Execute command using `{name}` ({count} times)")
            elif name in ("file_read", "read_file"):
                steps.append(f"Read file contents using `{name}` ({count} times)")
            elif name in ("grep",):
                steps.append(f"Search codebase using `{name}` ({count} times)")
            elif name in ("glob",):
                steps.append(f"Find files using `{name}` ({count} times)")
            elif name in ("web_search", "web_fetch"):
                steps.append(f"Research using `{name}` ({count} times)")
            else:
                steps.append(f"Execute `{name}` ({count} times)")
        return steps

    async def register(self, skill_def: dict[str, Any]) -> None:
        skill_registry = getattr(self._agent, "skill_registry", None)
        if skill_registry is None:
            return

        from encre.skills.types import BundledSkillDefinition
        bundled = BundledSkillDefinition(
            name=skill_def["name"],
            description=skill_def["description"],
            source="auto_generated",
            prompt=skill_def["body"],
        )
        skill_registry._bundled[skill_def["name"]] = bundled

        skills_dir = self._get_auto_skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skills_dir / f"{skill_def['name']}.md"
        skill_content = self._format_skill_md(skill_def)
        skill_path.write_text(skill_content, encoding="utf-8")
        logger.info(
            "Generated skill: %s (%d tool calls, reason: %s)",
            skill_def["name"],
            len(skill_def["tool_names"]),
            skill_def.get("reason", "pattern detected"),
        )

    def _get_auto_skills_dir(self) -> Path:
        from encre.config import get_data_dir
        return get_data_dir() / "skills" / "auto_generated"

    def _format_skill_md(self, skill_def: dict[str, Any]) -> str:
        lines = ["---"]
        lines.append(f"name: {skill_def['name']}")
        lines.append(f"description: {skill_def['description']}")
        lines.append(f"source: {skill_def['source']}")
        if skill_def.get("aliases"):
            lines.append(f"aliases: {', '.join(skill_def['aliases'])}")
        if skill_def.get("when_to_use"):
            lines.append(f"when_to_use: {skill_def['when_to_use']}")
        lines.append("---")
        lines.append("")
        lines.append(skill_def["body"])
        return "\n".join(lines)