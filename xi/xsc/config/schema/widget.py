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
Widget configuration schema.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


@dataclass
class XiWidgetValidation:
    """
    Widget validation configuration.
    
    Attributes:
        pattern: Regex pattern for validation
        message: Error message on validation failure
        min_length: Minimum length for string types
        max_length: Maximum length for string types
    """
    pattern: str = ""
    message: str = ""
    min_length: Optional[int] = None
    max_length: Optional[int] = None


@dataclass
class XiWidgetStyle:
    """
    Widget style configuration.
    
    Attributes:
        width: Widget width (e.g., "full", "half", "auto")
        height: Widget height in pixels (for textarea, etc.)
        placeholder: Placeholder text
        prefix: Prefix text or icon
        suffix: Suffix text or icon
        class_name: Additional CSS classes
    """
    width: str = "full"
    height: Optional[int] = None
    placeholder: str = ""
    prefix: str = ""
    suffix: str = ""
    class_name: str = ""


@dataclass
class XiWidgetConfig:
    """
    Custom widget configuration for dynamic UI rendering.
    
    This allows XSC to define custom controls that XIS will render,
    enabling extensibility without modifying XIS code.
    
    Attributes:
        type: Widget type (text, textarea, number, slider, toggle, 
              select, multiselect, file, directory, color, date, 
              time, datetime, code, markdown, keyvalue, list, custom)
        style: Widget styling configuration
        validation: Validation rules
        props: Additional widget-specific properties
        depends_on: Other parameters this widget depends on
        show_if: Condition to show this widget
        disabled_if: Condition to disable this widget
        custom_component: Custom component name (for type="custom")
        custom_props: Custom component properties
    """
    type: str = "text"
    style: 'XiWidgetStyle' = field(default_factory=XiWidgetStyle)
    validation: 'XiWidgetValidation' = field(default_factory=XiWidgetValidation)
    props: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    show_if: str = ""
    disabled_if: str = ""
    custom_component: str = ""
    custom_props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class XiValueMapping:
    """
    Value mapping configuration for command argument assembly.
    
    Defines how the frontend value should be transformed into
    command-line arguments.
    
    Attributes:
        arg_format: Format string for argument (e.g., "--{name}={value}", "-{name} {value}")
        arg_prefix: Prefix for argument name (e.g., "--", "-")
        arg_separator: Separator between name and value (e.g., "=", " ")
        skip_if: Skip this argument if value matches condition
        transform: Value transformation (lowercase, uppercase, str, int, float, json, path)
        default_if_empty: Default value if empty
        join_with: Join multiple values with this separator
        wrap_value: Wrap value in quotes
        template: Full template for complex argument generation
    """
    arg_format: str = ""
    arg_prefix: str = "--"
    arg_separator: str = " "
    skip_if: str = ""
    transform: str = ""
    default_if_empty: Any = None
    join_with: str = ","
    wrap_value: bool = False
    template: str = ""
