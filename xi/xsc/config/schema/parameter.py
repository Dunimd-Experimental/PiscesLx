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
Parameter schema definitions.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any

from .widget import XiWidgetConfig, XiValueMapping


@dataclass
class XiTabSchema:
    """
    Tab schema definition for grouping parameters.
    
    Attributes:
        name: Tab name (used as identifier)
        label: Display label for the tab
        available: Whether this tab is available
        unavailable_reason: Reason why tab is unavailable
    """
    name: str = ""
    label: str = ""
    available: bool = True
    unavailable_reason: str = ""


@dataclass
class XiParameterSchema:
    """
    Parameter schema definition for command arguments.
    
    Attributes:
        name: Parameter name
        type: Parameter type (string, integer, float, boolean, select, path)
        description: Human-readable description
        required: Whether the parameter is required
        default: Default value
        options: Available options for select type
        min: Minimum value for numeric types
        max: Maximum value for numeric types
        source: Source for dynamic options (directory path or variable)
        source_type: Type of source (directory, file, api)
        filter: Filter pattern for source (e.g., "*.yaml")
        available: Whether this parameter is available (has valid options/source)
        unavailable_reason: Reason why parameter is unavailable
        tab: Tab group this parameter belongs to
        widget: Custom widget configuration for UI rendering
        value_mapping: Value mapping for command assembly
    """
    name: str = ""
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    options: List[str] = field(default_factory=list)
    min: Optional[float] = None
    max: Optional[float] = None
    source: str = ""
    source_type: str = ""
    filter: str = ""
    available: bool = True
    unavailable_reason: str = ""
    tab: str = "basic"
    widget: Optional[XiWidgetConfig] = None
    value_mapping: Optional[XiValueMapping] = None
