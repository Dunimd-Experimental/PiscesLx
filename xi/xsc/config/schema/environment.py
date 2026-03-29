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
Environment configuration schema.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class XiRequirementConfig:
    """
    Python requirements file configuration.
    
    Attributes:
        name: Name identifier for this requirements file
        path: Path to requirements.txt file (relative to project root)
        required: Whether this file is required for the project to work
    """
    name: str = "main"
    path: str = "requirements.txt"
    required: bool = True


@dataclass
class XiVirtualenvConfig:
    """
    Python virtual environment configuration.
    
    Attributes:
        enabled: Whether virtual environment is enabled
        path: Path to virtual environment directory
        create_if_missing: Whether to create venv if it doesn't exist
    """
    enabled: bool = False
    path: str = ".venv"
    create_if_missing: bool = False


@dataclass
class XiEnvironmentConfig:
    """
    Environment configuration for Python and CUDA.
    
    Attributes:
        python_path: Path to Python executable
        cuda_required: Whether CUDA is required
        cuda_version: Required CUDA version (e.g., "11.8", "12.1")
        requirements: List of requirements file configurations
        virtualenv: Virtual environment configuration
    """
    python_path: str = "python"
    cuda_required: bool = False
    cuda_version: str = ""
    requirements: List['XiRequirementConfig'] = field(default_factory=list)
    virtualenv: Optional['XiVirtualenvConfig'] = None
