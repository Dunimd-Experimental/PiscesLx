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
from typing import Any

from encre.prompts.base import EncrePromptTemplate
from encre.prompts.system import EncrePromptBuilder
from encre.utils.types import PermissionMode


class EncreCodingPrompt(EncrePromptTemplate):
    def __init__(self, builder: EncrePromptBuilder | None = None, specialty: str = "coding") -> None:
        super().__init__(builder=builder, specialty=specialty)

    def build_system_prompt(
        self,
        mode: PermissionMode = "default",
        tools: list[dict[str, Any]] | None = None,
        custom_instructions: str = "",
    ) -> str:
        return self._builder.build(
            mode=mode,
            tools=tools,
            specialty="coding",
            custom_instructions=custom_instructions,
        )
