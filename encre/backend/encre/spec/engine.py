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

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from encre.prompts.loader import PromptLoader

_loader = PromptLoader()
_DEFAULT_SPEC_PROMPT = _loader.load("spec_generator", category="spec")


class SpecStatus(str, Enum):
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class SpecSection:
    title: str
    content: str


@dataclass
class SpecDocument:
    title: str
    sections: list[SpecSection] = field(default_factory=list)
    status: SpecStatus = SpecStatus.DRAFT
    feedback: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sections": [
                {"title": s.title, "content": s.content}
                for s in self.sections
            ],
            "status": self.status.value,
            "feedback": self.feedback,
            "raw_text": self.raw_text,
            "metadata": self.metadata,
        }

    def to_plan_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for section in self.sections:
            items.append({
                "id": f"spec-{section.title.lower().replace(' ', '-')}",
                "text": section.title,
                "status": "done",
            })
        if self.status == SpecStatus.APPROVED:
            items.append({
                "id": "spec-approved",
                "text": "Spec approved — ready for implementation",
                "status": "done",
            })
        return items

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            lines.append(section.content)
            lines.append("")
        if self.feedback:
            lines.append("## Feedback")
            lines.append("")
            lines.append(self.feedback)
        lines.append(f"\n**Status**: {self.status.value}")
        return "\n".join(lines)


class EncreSpecEngine:
    """Specification engine for generating, reviewing, and managing specs.

    Flow: generate draft -> review -> approve/reject -> execute.
    """

    def __init__(self, spec_prompt: str | None = None) -> None:
        self._spec_prompt = spec_prompt or _DEFAULT_SPEC_PROMPT
        self._current_spec: SpecDocument | None = None

    @property
    def current_spec(self) -> SpecDocument | None:
        return self._current_spec

    def build_spec_prompt(self, context: str) -> str:
        """Build the LLM prompt for generating a specification."""
        return f"{self._spec_prompt}\n\n## Context\n\n{context}\n\nGenerate the specification now."

    def parse_spec(self, title: str, llm_output: str) -> SpecDocument:
        """Parse LLM output into a structured SpecDocument."""
        sections: list[SpecSection] = []
        current_title = "Overview"
        current_lines: list[str] = []

        for line in llm_output.split("\n"):
            if line.startswith("## ") and not line.startswith("### "):
                if current_lines:
                    sections.append(SpecSection(
                        title=current_title,
                        content="\n".join(current_lines).strip(),
                    ))
                current_title = line.strip("## #").strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append(SpecSection(
                title=current_title,
                content="\n".join(current_lines).strip(),
            ))

        doc = SpecDocument(
            title=title,
            sections=sections,
            status=SpecStatus.DRAFT,
            raw_text=llm_output,
        )
        self._current_spec = doc
        return doc

    def approve(self) -> SpecDocument | None:
        """Mark the current spec as approved."""
        if self._current_spec is None:
            return None
        self._current_spec.status = SpecStatus.APPROVED
        return self._current_spec

    def reject(self, feedback: str) -> SpecDocument | None:
        """Reject the current spec with feedback."""
        if self._current_spec is None:
            return None
        self._current_spec.status = SpecStatus.REJECTED
        self._current_spec.feedback = feedback
        return self._current_spec

    def reset(self) -> None:
        self._current_spec = None
