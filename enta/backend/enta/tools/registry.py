#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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



from typing import Any

from enta.tools.base import EncreTool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, EncreTool] = {}

    def register(self, tool: EncreTool) -> None:
        self._tools[tool.name] = tool

    def register_many(self, tools: list[EncreTool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> EncreTool | None:
        return self._tools.get(name)

    def list_tools(self) -> dict[str, EncreTool]:
        return dict(self._tools)

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_format() for tool in self._tools.values()]

    def get_openai_tools_for_intents(self, intents: list[str]) -> list[dict[str, Any]]:
        """Return tools whose intents intersect with the given intent list."""
        result = []
        for tool in self._tools.values():
            tool_intents = getattr(tool, 'intents', ['general'])
            if any(i in intents for i in tool_intents):
                result.append(tool.to_openai_format())
        return result

    def get_anthropic_tools(self) -> list[dict[str, Any]]:
        return [tool.to_anthropic_format() for tool in self._tools.values()]

    def all(self) -> list[EncreTool]:
        return list(self._tools.values())
