#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

"""EntaPromptBuilder — builds prompts for teacher models in the outer loop.
"""


class EntaPromptBuilder:
    """Builds prompts for teacher models to generate training data.

    All system prompts enforce strict boundaries:
    - Benchmark datasets and test splits must NEVER be leaked or referenced
    - Identities of all models (teacher, judge, student) are NEVER disclosed
    - Topics avoid any internal terminology (model names, param scales, etc.)

    The builder wraps a topic string into a full prompt, optionally
    including a system message that describes the teacher's role.
    """

    # Default system prompt for teacher models generating training data.
    # This prompt enforces data boundary discipline and identity concealment.
    DEFAULT_TEACHER_SYSTEM: str = (
        "You are an expert instructor creating high-quality, original training material "
        "for a learning agent. Your output will be used to train the agent on reasoning, "
        "tool use, and problem-solving capabilities.\n\n"
        "Rules:\n"
        "1. Generate original content based on the topic only. Do NOT reproduce, reference, "
        "or hint at any benchmark datasets, evaluation suites, or standardized test questions.\n"
        "2. Do NOT reveal your identity, your model name, your version, or any internal "
        "system details. Refer to yourself only as \"an instructor.\"\n"
        "3. Do NOT reference the student model's name, architecture, internal identifiers, "
        "or any training framework details.\n"
        "4. Keep reasoning steps explicit but do not dump the entire chain of thought; "
        "present a clear, step-by-step explanation that is pedagogically useful.\n"
        "5. If a task involves tool usage, demonstrate the tool call format, arguments, "
        "and a plausible result. Do not fabricate sensitive data.\n"
        "6. All content must be self-contained and original. Avoid any phrasing that "
        "suggests you are reproducing an existing question or passage.\n\n"
        "Topic: {topic}"
    )

    def __init__(self) -> None:
        pass

    def build_prompt(self, topic: str, system: str | None = None) -> str:
        """Wrap a topic description into a teacher prompt.

        Args:
            topic: A topic description string (e.g. from
                :meth:`EntaTaskGenerator.generate_topic`).
            system: Optional system message.  When ``None`` a default
                system message is used.

        Returns:
            A fully formed prompt string ready for the teacher model.
        """
        if system:
            return f"{system}\n\n{topic}"
        return self.DEFAULT_TEACHER_SYSTEM.format(topic=topic)
