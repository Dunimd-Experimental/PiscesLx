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

import re
from typing import Any

from encre.prompts.loader import PromptLoader
from encre.skills.types import BundledSkillDefinition, SkillContext, SkillSource

_loader = PromptLoader()
_LOOP_PATTERN = re.compile(r"^\s*\[(\d+(?:\.\d+)?)\]\s*(.+)", re.DOTALL)


async def _loop_prompt(args: str | None, ctx: dict[str, Any]) -> str:
    if args is None:
        args = ""
    match = _LOOP_PATTERN.match(args)
    if match is None:
        return (f"You are in a loop execution mode. The command syntax is: [interval] <prompt>\n\n"
                f"The interval specifies how frequently (in seconds) to execute the prompt. "
                f"Minimum interval is 1 second.\n\n"
                f"However, the input provided did not match the expected format. Received:\n"
                f'  "{args}"\n\n'
                f"Please ask the user to specify the command in the format: [interval] <prompt>\n"
                f"Example: [10] Run the build and report any errors\n\n"
                f"Also provide the following guidance to the user:\n"
                f"- Use shorter intervals (1-5s) for rapid feedback loops like file watching\n"
                f"- Use medium intervals (10-30s) for build/test cycles\n"
                f"- Use longer intervals (60-300s) for monitoring tasks\n"
                f"- Add [stop] or press Ctrl+C to terminate the loop\n")

    interval_str = match.group(1).strip()
    prompt_text = match.group(2).strip()

    try:
        interval_seconds = float(interval_str)
    except ValueError:
        interval_seconds = 5.0

    if interval_seconds < 1.0:
        interval_seconds = 1.0

    return _loader.load_with_context(
        "loop", category="skills",
        prompt_text=prompt_text,
        interval_seconds=str(interval_seconds),
    )
