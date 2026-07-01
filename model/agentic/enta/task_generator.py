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

"""EntaTaskGenerator — decides what data to generate next during outer loop.

The generator uses the ``enta_model_layout`` fields
(``dynamic_head_param_scale``, ``knowledge_field_param_scale``, etc.)
to guide topic selection for teacher-generated training data.
"""

import random
from typing import Dict

_TOPIC_TEMPLATES: Dict[str, str] = {
    "tool_use": (
        "Write a detailed guide on using {n_tools} different programming tools "
        "to accomplish a complex data-processing task."
    ),
    "reasoning": (
        "Solve a multi-step reasoning problem that requires chaining together "
        "{n_steps} logical deductions."
    ),
    "code_generation": (
        "Generate a complete {module_type} module that integrates with an "
        "external API and handles errors gracefully."
    ),
    "knowledge_retrieval": (
        "Explain how {field_scale}x scaled knowledge fields encode and retrieve "
        "information in a neural architecture."
    ),
    "dynamic_head": (
        "Describe how a dynamic head with param scale {head_scale}, hidden dim "
        "{hidden_dim}, and {num_codebooks} codebooks processes variable-length inputs."
    ),
}


class EntaTaskGenerator:
    """Decides what training data to generate next in the outer loop.

    Uses ``enta_model_layout`` fields to bias topic selection toward
    exercising the dynamic head and knowledge field at their configured scales.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def generate_topic(self, enta_model_layout: dict) -> str:
        """Return a topic description string guided by the model layout.

        Args:
            enta_model_layout: Dict with fields like
                ``dynamic_head_param_scale``, ``knowledge_field_param_scale``,
                ``dynamic_head_hidden_dim``, ``dynamic_head_num_codebooks``.

        Returns:
            A free-form topic string suitable for passing to
            :class:`EntaPromptBuilder`.
        """
        if not enta_model_layout or not isinstance(enta_model_layout, dict):
            return self._fallback_topic()

        dh_scale = enta_model_layout.get("dynamic_head_param_scale", 1.0)
        kf_scale = enta_model_layout.get("knowledge_field_param_scale", 1.0)
        hidden_dim = enta_model_layout.get("dynamic_head_hidden_dim", 4096)
        num_codebooks = enta_model_layout.get("dynamic_head_num_codebooks", 4)

        # Select topic category biased by scale.
        # Higher param scales → more emphasis on dynamic-head / knowledge-field topics.
        scale = float(dh_scale) + float(kf_scale)
        if scale > 2.0:
            candidates = ["dynamic_head", "knowledge_retrieval"]
        elif scale > 1.0:
            candidates = ["tool_use", "code_generation", "dynamic_head"]
        else:
            candidates = ["tool_use", "reasoning", "code_generation"]

        category = self._rng.choice(candidates)
        template = _TOPIC_TEMPLATES.get(category, _TOPIC_TEMPLATES["tool_use"])

        return template.format(
            n_tools=self._rng.randint(3, 6),
            n_steps=self._rng.randint(3, 5),
            module_type=self._rng.choice(["data", "network", "async"]),
            field_scale=float(kf_scale),
            head_scale=float(dh_scale),
            hidden_dim=int(hidden_dim),
            num_codebooks=int(num_codebooks),
        )

    def _fallback_topic(self) -> str:
        """Return a safe generic topic when no layout is available."""
        return (
            "Generate a multi-turn conversation that exercises tool calling, "
            "logical reasoning, and error recovery."
        )
