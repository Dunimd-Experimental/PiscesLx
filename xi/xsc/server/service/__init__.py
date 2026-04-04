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
Xi Service Layer

Business logic layer for Xi Studio server operations.
This module separates business logic from communication layers (HTTP/WebSocket).
"""

from .welcome import (
    PiscesL1WelcomeValidator,
    PiscesL1WelcomeSetup,
    PiscesL1WelcomeAgreement,
    ValidationResult,
    SetupResult,
    get_welcome_validator,
    get_welcome_setup,
    get_welcome_agreement,
)

__all__ = [
    "PiscesL1WelcomeValidator",
    "PiscesL1WelcomeSetup",
    "PiscesL1WelcomeAgreement",
    "ValidationResult",
    "SetupResult",
    "get_welcome_validator",
    "get_welcome_setup",
    "get_welcome_agreement",
]
