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
Welcome Service Module

Business logic for Welcome screen operations including:
- Configuration validation
- Environment setup
- Agreement handling
"""

from .validator import (
    PiscesL1WelcomeValidator,
    ValidationResult,
    get_welcome_validator,
)
from .setup import (
    PiscesL1WelcomeSetup,
    SetupResult,
    get_welcome_setup,
)
from .agreement import (
    PiscesL1WelcomeAgreement,
    get_welcome_agreement,
)

__all__ = [
    "PiscesL1WelcomeValidator",
    "ValidationResult",
    "get_welcome_validator",
    "PiscesL1WelcomeSetup",
    "SetupResult",
    "get_welcome_setup",
    "PiscesL1WelcomeAgreement",
    "get_welcome_agreement",
]
