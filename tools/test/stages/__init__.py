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

"""
Test Stage Modules for PiscesL1.

This package contains all stage checker implementations for the
project health check system.

Stages:
    1. Environment Check - Python, PyTorch, CUDA, GPU, dependencies
    2. Project Structure Check - Files, directories, permissions
    3. Module Import Check - All module imports validation
    4. Configuration Check - YAML configs, parameters
    5. Model Instantiation Check - Model creation and initialization
    6. Forward Pass Check - Forward propagation tests
    7. Generation Check - Text generation functionality
    8. Optimization Check - Flash Attention, KV cache, quantization
"""

from .environment import PiscesLxEnvironmentChecker
from .structure import PiscesLxStructureChecker
from .imports import PiscesLxImportChecker
from .config import PiscesLxConfigChecker
from .model import PiscesLxModelChecker
from .forward import PiscesLxForwardChecker
from .generation import PiscesLxGenerationChecker
from .optimization import PiscesLxOptimizationChecker
from .memory_opt import PiscesLxMemoryOptChecker

__all__ = [
    "PiscesLxEnvironmentChecker",
    "PiscesLxStructureChecker",
    "PiscesLxImportChecker",
    "PiscesLxConfigChecker",
    "PiscesLxModelChecker",
    "PiscesLxForwardChecker",
    "PiscesLxGenerationChecker",
    "PiscesLxOptimizationChecker",
    "PiscesLxMemoryOptChecker",
]
