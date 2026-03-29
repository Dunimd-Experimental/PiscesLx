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
Project configuration schema.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class XiProjectCommandsConfig:
    """
    Project commands configuration.
    
    Attributes:
        enabled: List of enabled command names
    """
    enabled: List[str] = field(default_factory=list)


@dataclass
class XiProjectConfig:
    """
    Project metadata configuration.
    
    Attributes:
        name: Project name
        version: Project version
        backend: Backend type (e.g., "piscesl1", "custom")
        description: Project description
        author: Project author
        first_launch: Whether this is the first launch
        commands: Command configuration
    """
    name: str = "xi-project"
    version: str = "1.0.0"
    backend: str = "piscesl1"
    description: str = ""
    author: str = ""
    first_launch: bool = True
    commands: Optional['XiProjectCommandsConfig'] = None
