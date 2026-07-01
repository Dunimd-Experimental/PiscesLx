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

    The builder wraps a topic string into a full prompt, optionally
    including a system message that describes the teacher's role.
    """

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
        return (
            "You are an expert AI teacher generating high-quality "
            "training data for a student model.  Please produce a "
            "detailed, accurate, and well-structured response on the "
            f"following topic:\n\n{topic}"
        )
