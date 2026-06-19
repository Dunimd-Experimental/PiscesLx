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


from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# ── Enumerations ───────────────────────────────────────────────────

class SandboxMode(str, Enum):
    """Execution mode for sandboxed operations."""
    NONE = "none"           # Unrestricted — runs on host directly
    PATH = "path"           # Path-only isolation (file ops confined)
    CONTAINER = "container" # Docker container isolation
    HYBRID = "hybrid"       # path + container (recommended for safety)


class NetworkPolicy(str, Enum):
    """Network access policy."""
    NONE = "none"        # No network at all
    LIMITED = "limited"  # DNS-allowed domains only (no IP connect)
    HOST = "host"        # Full host network access


class FileProtection(str, Enum):
    """Level of file-system protection inside the sandbox."""
    DISABLED = "disabled"      # Standard rw access
    READ_ONLY = "read_only"    # Read-only workspace, tmpfs for writes
    STRICT = "strict"          # Strict: no shell, no symlink, no mount


class CGroupLimit(str, Enum):
    """Resource limits expressed via cgroups inside container."""
    OFF = "off"
    MEMORY_512M = "512m"
    MEMORY_1G = "1g"
    MEMORY_2G = "2g"
    MEMORY_4G = "4g"


class SeccompProfile(str, Enum):
    """Seccomp filtering profile."""
    UNPRIVILEGED = "unprivileged"  # Default: block privileged syscalls
    STRICT = "strict"              # Block most: no fork, no mount, no network


@dataclass
class NetworkConfig:
    """Fine-grained network access policy."""
    policy: NetworkPolicy = NetworkPolicy.NONE
    allowed_domains: list[str] = field(default_factory=list)
    allowed_ports: list[int] = field(default_factory=list)
    blocked_ports: list[int] = field(default_factory=list)
    dns_only: bool = True  # When True, allow DNS (port 53) but block all other outbound TCP/UDP


@dataclass
class ResourceConfig:
    """CPU / memory / process limits."""
    memory_limit: str = "512m"              # Docker --memory value
    cpu_limit: float = 1.0                   # Docker --cpus
    pids_limit: int = 64                     # Max PID count (cgroup pids.max)
    no_new_privileges: bool = True           # NO_NEW_PRIVS via seccomp
    address_space_increase: int = 0          # Max % increase over baseline RSS
    oom_score_adj: int = 1000                # Push OOM killer toward this process
    disk_quota_mb: int = 1024                # Per-container disk quota


@dataclass
class FileProtectionConfig:
    """File-system isolation policy."""
    read_only_root: bool = True              # Container --read-only
    workspace_mode: str = "rw"               # rw, ro, or "tmpfs"
    allowed_paths: list[str] = field(default_factory=list)  # Extra allowed host paths
    blocked_paths: list[str] = field(default_factory=list)  # Always-blocked paths
    symlink_protection: bool = True          # Reject symlinks pointing outside sandbox
    mount_protection: bool = True            # Reject mount / pivot_root / chroot
    tmpfs_exec: bool = True                  # /tmp and /var/tmp with exec
    no_suid: bool = True                     # No setuid/setgid binaries allowed


@dataclass
class SeccompConfig:
    """Linux seccomp-BPF syscall filtering."""
    profile: SeccompProfile = SeccompProfile.UNPRIVILEGED
    extra_blocked_syscalls: list[str] = field(default_factory=list)
    extra_allowed_syscalls: list[str] = field(default_factory=list)


@dataclass
class EnvConfig:
    """Environment variable policy for sandboxed execution."""
    inherit_env: bool = False                # Inherit host environment? (dangerous)
    allow_env_patterns: list[str] = field(default_factory=list)  # glob patterns of allowed env vars
    env_vars: dict[str, str] = field(default_factory=dict)       # Additional sandbox env vars
    deny_secret_patterns: list[str] = field(default_factory=list)  # Env var name glob patterns to strip


@dataclass
class SandboxConfig:
    """Comprehensive sandbox configuration.

    Controls isolation level for all potentially dangerous operations
    (shell execution, file I/O, network access).
    """
    # ── Core ──
    mode: SandboxMode = SandboxMode.NONE
    image: str = "python:3.11-slim"
    workspace_mount: str = "/workspace"
    workspace_container_path: str = "/workspace"

    # ── Sandboxing layers ──
    path_isolation: bool = True           # Enable path-based file isolation
    path_sandbox_dir: str = ""            # Empty = auto from config
    network: NetworkConfig = field(default_factory=NetworkConfig)
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    file_protection: FileProtectionConfig = field(default_factory=FileProtectionConfig)
    seccomp: SeccompConfig = field(default_factory=SeccompConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    timeout: int = 120                    # Default timeout in seconds

    # ── Security hardening ──
    disable_sudo: bool = True
    disable_network_tooling: bool = True  # Block curl/wget in sandbox
    disable_interactive: bool = True      # Block commands that open TUIs
    command_pattern_blocklist: list[str] = field(default_factory=list)
    max_command_length: int = 4096         # Max command string length
    max_output_bytes: int = 2 * 1024 * 1024  # 2 MB hard cap on output

    # ── Container extras ──
    container_labels: dict[str, str] = field(default_factory=dict)
    extra_mounts: dict[str, str] = field(default_factory=dict)
    extra_docker_opts: list[str] = field(default_factory=list)


@dataclass
class SandboxResult:
    """Result of a sandboxed execution."""
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    duration_ms: float = 0.0
    sandbox_violation: str = ""           # Description of any policy violation
    killed: bool = False                  # True if sandbox killed the process
    output_truncated: bool = False        # True if output was size-capped
    pid: int = 0                          # Container PID (0 if N/A)
    security_events: list[dict] = field(default_factory=list)  # Audit log
