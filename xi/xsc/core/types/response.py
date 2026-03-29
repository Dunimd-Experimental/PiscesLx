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
Response type definitions for Xi Studio.

Contains response dataclasses for command execution and control results.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class XiResponse:
    success: bool
    run_id: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


@dataclass
class XiControlResponse:
    success: bool
    run_id: str
    action: str
    message: str
    previous_status: Optional[str] = None
    new_status: Optional[str] = None
