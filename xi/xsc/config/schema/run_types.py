#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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

"""
Run Types Schema Module

Dataclasses for run types configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class XiRunTypeConfig:
    """
    Configuration for a single run type.
    
    Attributes:
        name: Internal name of the run type (e.g., "train", "inference")
        label: Display label for the UI
        description: Short description of what this run type does
        icon: Icon name for the UI (Lucide icon names)
        color: Hex color code for the UI
        enabled: Whether this run type is available
        order: Display order in the UI
    """
    name: str
    label: str
    description: str = ""
    icon: str = "Play"
    color: str = "#6B7280"
    enabled: bool = True
    order: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "enabled": self.enabled,
            "order": self.order,
        }
