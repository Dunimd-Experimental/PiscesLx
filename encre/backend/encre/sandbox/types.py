#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

from dataclasses import dataclass, field
from typing import Literal

NetworkPolicy = Literal["none", "limited", "host"]


@dataclass
class SandboxConfig:
    image: str = "python:3.11-slim"
    workspace_mount: str = "/workspace"
    network: NetworkPolicy = "none"
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    timeout: int = 120
    read_only: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    extra_mounts: dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    duration_ms: float = 0.0
