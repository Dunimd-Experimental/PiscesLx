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
from typing import Any, Callable, Awaitable
from enum import Enum


class SkillContext(str, Enum):
    INLINE = "inline"
    FORK = "fork"


class SkillSource(str, Enum):
    MANAGED = "managed"
    USER = "user"
    PROJECT = "project"
    BUNDLED = "bundled"


_PRIORITY_ORDER: dict[SkillSource, int] = {
    SkillSource.MANAGED: 0,
    SkillSource.USER: 1,
    SkillSource.PROJECT: 2,
    SkillSource.BUNDLED: 3,
}


@dataclass
class BundledSkillDefinition:
    name: str
    description: str
    get_prompt_for_command: Callable[[str | None, dict[str, Any]], Awaitable[str]]
    aliases: list[str] = field(default_factory=list)
    when_to_use: str = ""
    argument_hint: str = ""
    allowed_tools: list[str] | None = None
    model: str | None = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    context: SkillContext = SkillContext.INLINE
    source: SkillSource = SkillSource.BUNDLED
    file_path: str = ""
    body: str = ""
    hidden: bool = False
    # Agent Skills standard fields
    license: str = ""
    compatibility: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
