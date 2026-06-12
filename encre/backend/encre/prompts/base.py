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

from abc import ABC, abstractmethod
from typing import Any

from encre.prompts.system import EncrePromptBuilder
from encre.utils.types import PermissionMode


class EncreBasePrompt(ABC):
    """Abstract base for prompt templates."""

    @abstractmethod
    def build_system_prompt(
        self,
        mode: PermissionMode,
        tools: list[dict[str, Any]] | None = None,
        custom_instructions: str = "",
    ) -> str:
        ...

    @abstractmethod
    def build_tool_instructions(self, tool_names: list[str]) -> str:
        ...


class EncrePromptTemplate(EncreBasePrompt):
    """General-purpose prompt template using the layered builder."""

    def __init__(self, builder: EncrePromptBuilder | None = None, specialty: str = "general") -> None:
        self._builder = builder or EncrePromptBuilder()
        self._specialty = specialty

    def build_system_prompt(
        self,
        mode: PermissionMode,
        tools: list[dict[str, Any]] | None = None,
        custom_instructions: str = "",
        intents: list[str] | None = None,
        workspace_root: str = "",
        workspace_name: str = "",
        project_summary: str = "",
        language_preference: str = "auto",
        app_language: str = "zh",
    ) -> str:
        return self._builder.build(
            mode=mode,
            tools=tools,
            specialty=self._specialty,
            custom_instructions=custom_instructions,
            intents=intents,
            workspace_root=workspace_root,
            workspace_name=workspace_name,
            project_summary=project_summary,
            language_preference=language_preference,
            app_language=app_language,
        )

    def build_tool_instructions(self, tool_names: list[str]) -> str:
        if not tool_names:
            return "You do not have access to any tools."
        tools_list = "\n".join(f"- {name}" for name in tool_names)
        return f"You have access to the following tools:\n{tools_list}\n\nUse them as needed to accomplish the task."

    @property
    def builder(self) -> EncrePromptBuilder:
        return self._builder
