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

"""EntaIntake — EnTA startup questionnaire for dynamic head and knowledge field parameters."""

import os
import sys
from typing import Any, Dict, List


# Required fields for the startup questionnaire
REQUIRED_FIELDS = [
    "dynamic_head_param_scale",
    "dynamic_head_hidden_dim",
    "dynamic_head_num_codebooks",
    "knowledge_field_param_scale",
    "knowledge_field_codebook_size",
    "knowledge_field_entry_dim",
]

FIELD_LABELS = {
    "dynamic_head_param_scale": "Dynamic Head parameter scale (e.g. 0.5B, 1B, or custom)",
    "dynamic_head_hidden_dim": "Dynamic Head hidden dimension (e.g. 2048, 4096)",
    "dynamic_head_num_codebooks": "Dynamic Head number of codebooks (e.g. 4, 8, 16)",
    "knowledge_field_param_scale": "Knowledge Field parameter scale (e.g. 314B, 512B)",
    "knowledge_field_codebook_size": "Knowledge Field codebook size (e.g. 65536, 131072)",
    "knowledge_field_entry_dim": "Knowledge Field entry dimension (e.g. 128, 256)",
}


class EntaIntake:
    """Startup questionnaire for EnTA dynamic head and knowledge field parameters.

    When ``--enta`` is enabled and the required layout fields are missing from
    ``configs/teachers.yaml``, this class prompts the user interactively via CLI
    and writes the answers back to the YAML file.
    """

    def __init__(self, config_path: str = "configs/teachers.yaml") -> None:
        self.config_path = config_path

    def check_missing_fields(self) -> List[str]:
        """Return a list of required fields that are missing from the config."""
        import yaml

        if not os.path.exists(self.config_path):
            return list(REQUIRED_FIELDS)

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        layout = data.get("enta_model_layout", {})
        if not isinstance(layout, dict):
            return list(REQUIRED_FIELDS)

        missing = []
        for field in REQUIRED_FIELDS:
            val = layout.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                missing.append(field)
        return missing

    def prompt_user(self, missing_fields: List[str]) -> Dict[str, Any]:
        """Interactively prompt the user for each missing field.

        Args:
            missing_fields: List of field names to prompt for.

        Returns:
            Dict mapping field names to the user-provided values.
        """
        print("\n=== EnTA Startup Questionnaire ===")
        print("The following model layout parameters are required but missing:\n")

        values: Dict[str, Any] = {}
        for field in missing_fields:
            label = FIELD_LABELS.get(field, field)
            while True:
                raw = input(f"  {label}: ").strip()
                if raw:
                    values[field] = raw
                    break
                print("  (This field cannot be empty. Please provide a value.)")

        print("\nQuestionnaire complete.\n")
        return values

    def merge_into_config(self, values: Dict[str, Any]) -> None:
        """Merge questionnaire results into ``configs/teachers.yaml``.

        Args:
            values: Dict of field names to values from ``prompt_user()``.
        """
        import yaml

        # Load existing config or start fresh
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        # Ensure enta_model_layout section exists
        if "enta_model_layout" not in data or not isinstance(data["enta_model_layout"], dict):
            data["enta_model_layout"] = {}

        # Merge values (preserve existing fields not in values)
        data["enta_model_layout"].update(values)

        # Write back
        os.makedirs(os.path.dirname(self.config_path) or ".", exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def ensure_layout(self) -> Dict[str, Any]:
        """Check for missing fields, prompt if needed, and persist.

        This is the main entry point called by the orchestrator before
        the EnTA outer loop starts.

        Returns:
            The complete ``enta_model_layout`` dict with all required fields.
        """
        missing = self.check_missing_fields()
        if missing:
            values = self.prompt_user(missing)
            self.merge_into_config(values)
        return self._load_layout()

    def _load_layout(self) -> Dict[str, Any]:
        """Load the current enta_model_layout from config."""
        import yaml

        if not os.path.exists(self.config_path):
            return {}

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        layout = data.get("enta_model_layout", {})
        return layout if isinstance(layout, dict) else {}
