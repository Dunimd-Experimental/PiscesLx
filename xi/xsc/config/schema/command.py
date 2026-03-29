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
Command configuration schema.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from .parameter import XiParameterSchema, XiTabSchema


@dataclass
class XiCommandSchema:
    """
    Command schema definition.
    
    Attributes:
        description: Command description
        parameters: List of parameter schemas
        tabs: List of tab schemas for grouping parameters
        available: Whether the entire command is available
        unavailable_reason: Reason why command is unavailable
    """
    description: str = ""
    parameters: List[XiParameterSchema] = field(default_factory=list)
    tabs: List[XiTabSchema] = field(default_factory=list)
    available: bool = True
    unavailable_reason: str = ""


@dataclass
class XiCommandConfig:
    """
    Command definition configuration.
    
    Attributes:
        executable: Executable to run (e.g., "python")
        script: Script to execute (e.g., "manage.py")
        args: Default command arguments
        env: Environment variables
        cwd: Working directory (supports variable substitution)
        timeout: Command timeout in seconds
        background: Whether to run in background
        defaults: Default parameter values
        schema: Parameter schema for UI generation
    """
    executable: str = "python"
    script: str = "manage.py"
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: str = "${paths.root}"
    timeout: int = 3600
    background: bool = True
    defaults: Dict[str, Any] = field(default_factory=dict)
    schema: Optional[XiCommandSchema] = None
