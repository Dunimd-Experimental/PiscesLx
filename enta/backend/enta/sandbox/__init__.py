#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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


"""EnTA Sandbox Isolation System.

Three-layer sandbox architecture:

1. **Path isolation** (``_sandbox.py``)
   Remaps file paths into a per-session sandbox directory.
   Prevents path traversal attacks and symlink escapes.

2. **Container sandbox** (``container.py``)
   Docker-based isolation with seccomp, capability drop,
   read-only rootfs, and resource limits.  Supports both
   ephemeral (one-shot ``docker run --rm``) and persistent
   (session container) modes.

3. **Permission system** (``safety.py``)
   Static analysis + ML classification of commands before
   execution.  Routes dangerous commands through the
   container sandbox when enabled.

Usage::

    from enta.sandbox.types import SandboxConfig, SandboxMode, NetworkPolicy
    from enta.sandbox.container import EncreContainerSandbox

    # Ephemeral sandbox
    with EncreContainerSandbox("/path/to/workspace") as sb:
        result = sb.execute("ls -la")
        print(result.stdout)

    # With custom config
    config = SandboxConfig(
        mode=SandboxMode.CONTAINER,
        image="python:3.11-slim",
        timeout=60,
    )
    sb = EncreContainerSandbox("/path/to/workspace", config)
    sb.run_container()
    sb.exec_in_container("npm test")
    sb.stop_container()
"""

from enta.sandbox.container import EncreContainerSandbox as EncreContainerSandbox
from enta.sandbox.types import (
    CGroupLimit as CGroupLimit,
    EnvConfig as EnvConfig,
    FileProtection as FileProtection,
    FileProtectionConfig as FileProtectionConfig,
    NetworkConfig as NetworkConfig,
    NetworkPolicy as NetworkPolicy,
    ResourceConfig as ResourceConfig,
    SandboxConfig as SandboxConfig,
    SandboxMode as SandboxMode,
    SandboxResult as SandboxResult,
    SeccompConfig as SeccompConfig,
    SeccompProfile as SeccompProfile,
)

__all__ = [
    "CGroupLimit",
    "EncreContainerSandbox",
    "EnvConfig",
    "FileProtection",
    "FileProtectionConfig",
    "NetworkConfig",
    "NetworkPolicy",
    "ResourceConfig",
    "SandboxConfig",
    "SandboxMode",
    "SandboxResult",
    "SeccompConfig",
    "SeccompProfile",
]
