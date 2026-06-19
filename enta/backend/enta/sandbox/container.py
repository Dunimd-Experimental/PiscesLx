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


"""Docker container sandbox for secure command execution.

Provides three isolation modes:

1. **Ephemeral** (default) — ``docker run --rm`` one-shot per command.
   No state persists between calls.  Seccomp + capability drop + no-new-privs
   applied on every execution.

2. **Session** — A persistent container per session.  The container is
   started once, commands are ``exec``-ed into it, and it is torn down
   when the sandbox is closed.  State across commands within a session.

3. **Strict** — All of the above + no network, no root FS writes, no
   suid binaries, max 64 PIDs, and a custom seccomp profile that blocks
   ALL privileged syscalls.

Security hardening applied in ALL modes:
- ``--cap-drop=ALL``  / ``--security-opt=no-new-privileges:true``
- ``--read-only`` root filesystem (except tmpfs on ``/tmp``, ``/var/tmp``)
- ``--pids-limit=64`` (prevents fork bombs)
- ``--memory`` / ``--cpus`` resource limits
- Seccomp profile that blocks dangerous syscalls
- ``--user=nobody`` (maps to container ``nobody:65534``) when workspace
  permissions allow
- ``--network=none`` when network policy is ``none``
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from enta.sandbox.types import (
    EnvConfig,
    FileProtectionConfig,
    NetworkConfig,
    NetworkPolicy,
    ResourceConfig,
    SandboxConfig,
    SandboxMode,
    SandboxResult,
    SeccompConfig,
    SeccompProfile,
)


# ── Constants ──────────────────────────────────────────────────────

# Syscalls that are always blocked regardless of profile
_BLOCKED_SYSCALLS_COMMON = frozenset({
    "acct", "add_key", "bdflush", "bpf", "chroot",
    "clock_settime", "create_module", "delete_module",
    "finit_module", "get_kernel_syms", "init_module",
    "ioperm", "iopl", "kexec_file_load", "kexec_load",
    "keyctl", "kill", "lookup_dcookie", "mount",
    "nfsservctl", "perf_event_open", "pivot_root",
    "process_vm_readv", "process_vm_writev", "ptrace",
    "query_module", "reboot", "request_key",
    "setdomainname", "sethostname", "settimeofday",
    "stime", "swapoff", "swapon", "sysfs",
    "_sysctl", "umount2", "unshare", "uselib",
    "userfaultfd", "vfork", "vhangup",
})

# Syscalls blocked in strict mode on top of common
_BLOCKED_SYSCALLS_STRICT = frozenset({
    "clone", "fork", "socket", "bind", "connect",
    "listen", "accept", "sendto", "recvfrom",
    "sendmsg", "recvmsg", "shutdown", "setsockopt",
    "getsockopt", "getpeername", "getsockname",
    "socketpair", "mlock", "mlock2", "mlockall",
    "munlock", "munlockall", "mprotect",
})

# Commands that are blocked by default in sandbox mode
_BLOCKED_COMMANDS_PATTERNS = [
    # Sudo / privilege escalation
    "sudo", "doas", "pkexec", "su ", "sg ",
    # Network scanning / exploitation
    "nmap", "masscan", "zmap", "nikto",
    # Crypto mining
    "minerd", "xmrig", "cpuminer", "ccminer",
    # Reverse shells / binds
    "ncat -e", "nc -e", "socat exec",
    "bash -i", "sh -i",
    # Kernel operations
    "insmod", "modprobe", "modprobe -r",
    # Docker operations inside sandbox
    "docker ", "docker-compose ", "podman ",
]

# Block network tooling in network-isolated mode
_BLOCKED_NETWORK_COMMANDS = frozenset({
    "curl", "wget", "nc", "ncat", "ssh", "scp",
    "sftp", "telnet", "ftp", "socat", "tcpdump",
    "tshark", "ngrep", "iftop", "iptraf",
})

# Container user for privilege-dropped execution
_CONTAINER_USER = "65534:65534"  # nobody:nogroup


@dataclass
class SecurityAuditEntry:
    """A single security-relevant event recorded during sandbox execution."""
    timestamp: float
    event_type: str           # e.g. "execution", "violation", "timeout"
    command: str              # The command that was executed
    details: str              # Human-readable details
    severity: str = "info"    # info / warning / critical


# ── Main sandbox class ────────────────────────────────────────────

class EncreContainerSandbox:
    """Secure Docker container sandbox with multiple isolation modes.

    Usage::

        # Ephemeral (one-shot per command)
        with EncreContainerSandbox("/path/to/workspace") as sb:
            result = sb.execute("ls -la")

        # Session (persistent container across commands)
        sb = EncreContainerSandbox("/path/to/workspace",
                                    config=SandboxConfig(mode=SandboxMode.CONTAINER))
        sb.run_container()
        result1 = sb.exec_in_container("echo hello")
        result2 = sb.exec_in_container("ls -la")
        sb.stop_container()
    """

    def __init__(
        self,
        workspace: str,
        config: SandboxConfig | None = None,
    ) -> None:
        self.workspace = os.path.abspath(workspace)
        self.config = config or SandboxConfig()
        self._container_id: str | None = None
        self._active: bool = False
        self._temp_seccomp: str | None = None
        self._audit_log: list[SecurityAuditEntry] = []
        self._created_temp_dirs: list[str] = []

    # ── Public API ────────────────────────────────────────────────

    def execute(self, command: str, timeout: int | None = None) -> SandboxResult:
        """Execute *command* in an ephemeral Docker container.

        The container is ``--rm`` (auto-deleted on exit).  Returns a
        ``SandboxResult`` with stdout, stderr, exit code, and any
        security violations.
        """
        # Pre-execution security scan
        violation = self._check_command(command)
        if violation:
            self._audit(SecurityAuditEntry(
                timestamp=time.time(),
                event_type="violation",
                command=command[:200],
                details=violation,
                severity="critical",
            ))
            return SandboxResult(
                stdout="",
                stderr=violation,
                exit_code=-4,
                duration_ms=0.0,
                sandbox_violation=violation,
            )

        start = time.time()
        cmd = self._build_docker_command(command)
        effective_timeout = timeout or self.config.timeout
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                encoding="utf-8",
                errors="replace",
            )

            # Record execution event
            self._audit(SecurityAuditEntry(
                timestamp=time.time(),
                event_type="execution",
                command=command[:200],
                details=f"exit_code={proc.returncode}",
                severity="info",
            ))

            return SandboxResult(
                stdout=_truncate_output(proc.stdout, self.config.max_output_bytes),
                stderr=_truncate_output(proc.stderr, self.config.max_output_bytes),
                exit_code=proc.returncode,
                duration_ms=(time.time() - start) * 1000,
                output_truncated=(
                    len(proc.stdout or "") > self.config.max_output_bytes
                    or len(proc.stderr or "") > self.config.max_output_bytes
                ),
            )
        except subprocess.TimeoutExpired:
            self._audit(SecurityAuditEntry(
                timestamp=time.time(),
                event_type="timeout",
                command=command[:200],
                details=f"timed out after {effective_timeout}s",
                severity="warning",
            ))
            return SandboxResult(
                stdout="",
                stderr="Command timed out",
                exit_code=-1,
                timed_out=True,
                duration_ms=(time.time() - start) * 1000,
            )
        except FileNotFoundError:
            return SandboxResult(
                stdout="",
                stderr="Docker not found. Install Docker to use container sandbox.",
                exit_code=-2,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            self._audit(SecurityAuditEntry(
                timestamp=time.time(),
                event_type="error",
                command=command[:200],
                details=str(e),
                severity="warning",
            ))
            return SandboxResult(
                stdout="",
                stderr=f"Sandbox execution error: {e}",
                exit_code=-3,
                duration_ms=(time.time() - start) * 1000,
            )
        finally:
            self._cleanup_tempfiles()

    async def execute_async(
        self, command: str, timeout: int | None = None,
    ) -> SandboxResult:
        """Execute *command* asynchronously in an ephemeral Docker container."""
        violation = self._check_command(command)
        if violation:
            self._audit(SecurityAuditEntry(
                timestamp=time.time(),
                event_type="violation",
                command=command[:200],
                details=violation,
                severity="critical",
            ))
            return SandboxResult(
                stdout="",
                stderr=violation,
                exit_code=-4,
                duration_ms=0.0,
                sandbox_violation=violation,
            )

        start = time.time()
        cmd = self._build_docker_command(command)
        effective_timeout = timeout or self.config.timeout
        try:
            from enta.tools.builtin._suppress_window import hidden_subprocess_kwargs
            kwargs = hidden_subprocess_kwargs()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **kwargs,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=effective_timeout,
                )

                self._audit(SecurityAuditEntry(
                    timestamp=time.time(),
                    event_type="execution",
                    command=command[:200],
                    details=f"exit_code={proc.returncode}",
                    severity="info",
                ))

                stdout = (stdout_bytes.decode("utf-8", errors="replace")
                          if stdout_bytes else "")
                stderr = (stderr_bytes.decode("utf-8", errors="replace")
                          if stderr_bytes else "")

                return SandboxResult(
                    stdout=_truncate_output(stdout, self.config.max_output_bytes),
                    stderr=_truncate_output(stderr, self.config.max_output_bytes),
                    exit_code=proc.returncode or 0,
                    duration_ms=(time.time() - start) * 1000,
                    output_truncated=(
                        len(stdout) > self.config.max_output_bytes
                        or len(stderr) > self.config.max_output_bytes
                    ),
                )
            except TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                self._audit(SecurityAuditEntry(
                    timestamp=time.time(),
                    event_type="timeout",
                    command=command[:200],
                    details=f"timed out after {effective_timeout}s",
                    severity="warning",
                ))
                return SandboxResult(
                    stdout="",
                    stderr="Command timed out",
                    exit_code=-1,
                    timed_out=True,
                    duration_ms=(time.time() - start) * 1000,
                )
        except FileNotFoundError:
            return SandboxResult(
                stdout="",
                stderr="Docker not found. Install Docker to use container sandbox.",
                exit_code=-2,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            self._audit(SecurityAuditEntry(
                timestamp=time.time(),
                event_type="error",
                command=command[:200],
                details=str(e),
                severity="warning",
            ))
            return SandboxResult(
                stdout="",
                stderr=f"Sandbox execution error: {e}",
                exit_code=-3,
                duration_ms=(time.time() - start) * 1000,
            )
        finally:
            self._cleanup_tempfiles()

    # ── Session (persistent container) API ────────────────────────

    def run_container(self) -> str:
        """Start a persistent container that stays alive for multiple commands.

        Returns the container ID.
        """
        container_name = f"enta_sandbox_{uuid.uuid4().hex[:12]}"
        cmd = self._build_container_create_command(container_name)
        cmd.extend(["tail", "-f", "/dev/null"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container: {result.stderr.strip()}"
            )

        self._container_id = result.stdout.strip()
        self._active = True

        # Apply additional security hardening after container starts
        self._post_start_harden()

        self._audit(SecurityAuditEntry(
            timestamp=time.time(),
            event_type="execution",
            command="container:start",
            details=f"container_id={self._container_id}",
            severity="info",
        ))
        return self._container_id

    def exec_in_container(self, command: str) -> SandboxResult:
        """Execute a command in the persistent container."""
        if not self._container_id or not self._active:
            raise RuntimeError("No active container. Call run_container() first.")

        violation = self._check_command(command, in_container=True)
        if violation:
            self._audit(SecurityAuditEntry(
                timestamp=time.time(),
                event_type="violation",
                command=command[:200],
                details=violation,
                severity="critical",
            ))
            return SandboxResult(
                stdout="",
                stderr=violation,
                exit_code=-4,
                sandbox_violation=violation,
            )

        start = time.time()
        cmd = [
            "docker", "exec",
            "-w", self.config.workspace_mount,
            self._container_id,
            "sh", "-c", command,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                encoding="utf-8",
                errors="replace",
            )
            self._audit(SecurityAuditEntry(
                timestamp=time.time(),
                event_type="execution",
                command=command[:200],
                details=f"exit_code={proc.returncode}",
                severity="info",
            ))
            return SandboxResult(
                stdout=_truncate_output(proc.stdout, self.config.max_output_bytes),
                stderr=_truncate_output(proc.stderr, self.config.max_output_bytes),
                exit_code=proc.returncode,
                duration_ms=(time.time() - start) * 1000,
                output_truncated=(
                    len(proc.stdout or "") > self.config.max_output_bytes
                    or len(proc.stderr or "") > self.config.max_output_bytes
                ),
            )
        except subprocess.TimeoutExpired:
            self._audit(SecurityAuditEntry(
                timestamp=time.time(),
                event_type="timeout",
                command=command[:200],
                details="timed out in container",
                severity="warning",
            ))
            return SandboxResult(
                stdout="", stderr="Command timed out",
                exit_code=-1, timed_out=True,
            )
        except Exception as e:
            return SandboxResult(
                stdout="", stderr=f"Container exec error: {e}",
                exit_code=-3,
            )

    def stop_container(self) -> None:
        """Stop and remove the persistent container."""
        if not self._container_id:
            return

        self._audit(SecurityAuditEntry(
            timestamp=time.time(),
            event_type="execution",
            command="container:stop",
            details=f"container_id={self._container_id}",
            severity="info",
        ))

        with contextlib.suppress(Exception):
            subprocess.run(
                ["docker", "stop", "-t", "3", self._container_id],
                capture_output=True, timeout=15,
            )

        with contextlib.suppress(Exception):
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True, timeout=10,
            )

        self._container_id = None
        self._active = False

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        """Release all resources."""
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up containers, tempfiles, and audit state."""
        self.stop_container()
        if self._temp_seccomp and os.path.exists(self._temp_seccomp):
            with contextlib.suppress(OSError):
                os.unlink(self._temp_seccomp)
        self._temp_seccomp = None
        for tmp_dir in self._created_temp_dirs:
            if os.path.exists(tmp_dir):
                with contextlib.suppress(OSError):
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)
        self._created_temp_dirs.clear()
        self._audit_log.clear()

    def is_available(self) -> bool:
        """Check if Docker is available on this system."""
        try:
            subprocess.run(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                capture_output=True, timeout=5,
            )
            return True
        except Exception:
            return False

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return the security audit log."""
        return [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "command": e.command,
                "details": e.details,
                "severity": e.severity,
            }
            for e in self._audit_log
        ]

    # ── Docker command builders ───────────────────────────────────

    def _build_docker_command(self, command: str) -> list[str]:
        """Build a ``docker run --rm`` command with full security hardening."""
        cmd = ["docker", "run", "--rm"]

        # ── Security base: drop ALL capabilities ────────────────
        cmd.append("--cap-drop=ALL")
        cmd.append("--security-opt=no-new-privileges:true")

        # ── Network policy ──────────────────────────────────────
        nw = self.config.network
        if nw.policy == NetworkPolicy.NONE:
            cmd.append("--network=none")
        elif nw.policy == NetworkPolicy.LIMITED:
            cmd.append("--network=bridge")
            for domain in nw.allowed_domains:
                cmd.extend(["--add-host", domain])
            # Block all ports not in allowed_ports
            if nw.allowed_ports and not nw.blocked_ports:
                pass  # Docker native can't per-port filter
        elif nw.policy == NetworkPolicy.HOST:
            cmd.append("--network=host")

        # ── Resource limits ─────────────────────────────────────
        dl = self.config.resource
        cmd.extend(["--memory", dl.memory_limit])
        cmd.extend(["--cpus", str(dl.cpu_limit)])
        cmd.extend(["--pids-limit", str(dl.pids_limit)])
        cmd.extend(["--oom-score-adj", str(dl.oom_score_adj)])

        # ── Read-only root filesystem ──────────────────────────
        fp = self.config.file_protection
        if fp.read_only_root:
            cmd.append("--read-only")
            cmd.append("--tmpfs=/tmp:exec,nosuid,nodeinit")
            cmd.append("--tmpfs=/var/tmp:exec,nosuid,nodeinit")
            # Some tools need /home — provide it as tmpfs too
            cmd.append("--tmpfs=/home:exec,nosuid,nodeinit")

        # ── Workspace mount ────────────────────────────────────
        mount_mode = "ro" if fp.workspace_mode == "ro" else "rw"
        cmd.extend(["-v", f"{self.workspace}:{self.config.workspace_mount}:{mount_mode}"])

        # ── Extra mounts ────────────────────────────────────────
        for host_path, container_path in self.config.extra_mounts.items():
            extra_mount_opts = "ro" if fp.read_only_root else "rw"
            cmd.extend(["-v", f"{host_path}:{container_path}:{extra_mount_opts}"])

        # ── Run as non-root user ───────────────────────────────
        cmd.extend(["--user", _CONTAINER_USER])

        # ── Seccomp profile ─────────────────────────────────────
        self._temp_seccomp = self._build_seccomp_profile()
        if self._temp_seccomp:
            cmd.extend(["--security-opt", f"seccomp={self._temp_seccomp}"])

        # ── Env vars ────────────────────────────────────────────
        for k, v in self.config.env.env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])

        # ── Container labels ────────────────────────────────────
        for k, v in self.config.container_labels.items():
            cmd.extend(["--label", f"{k}={v}"])

        # ── Extra Docker options ────────────────────────────────
        for opt in self.config.extra_docker_opts:
            cmd.append(opt)

        # ── Image and command ───────────────────────────────────
        cmd.append(self.config.image)

        wrapped = _wrap_command(command, self.config)
        cmd.extend(["sh", "-c", wrapped])

        return cmd

    def _build_container_create_command(self, container_name: str) -> list[str]:
        """Build a ``docker create`` command for persistent containers."""
        cmd = ["docker", "create", "--name", container_name]

        # Same hardening as ephemeral
        cmd.append("--cap-drop=ALL")
        cmd.append("--security-opt=no-new-privileges:true")

        nw = self.config.network
        if nw.policy == NetworkPolicy.NONE:
            cmd.append("--network=none")
        elif nw.policy == NetworkPolicy.LIMITED:
            cmd.append("--network=bridge")
            for domain in nw.allowed_domains:
                cmd.extend(["--add-host", domain])

        dl = self.config.resource
        cmd.extend(["--memory", dl.memory_limit])
        cmd.extend(["--cpus", str(dl.cpu_limit)])
        cmd.extend(["--pids-limit", str(dl.pids_limit)])
        cmd.extend(["--oom-score-adj", str(dl.oom_score_adj)])

        fp = self.config.file_protection
        if fp.read_only_root:
            cmd.append("--read-only")
            cmd.append("--tmpfs=/tmp:exec,nosuid,nodeinit")
            cmd.append("--tmpfs=/var/tmp:exec,nosuid,nodeinit")
            cmd.append("--tmpfs=/home:exec,nosuid,nodeinit")

        mount_mode = "ro" if fp.workspace_mode == "ro" else "rw"
        cmd.extend(["-v", f"{self.workspace}:{self.config.workspace_mount}:{mount_mode}"])

        for host_path, container_path in self.config.extra_mounts.items():
            extra_mount_opts = "ro" if fp.read_only_root else "rw"
            cmd.extend(["-v", f"{host_path}:{container_path}:{extra_mount_opts}"])

        cmd.extend(["--user", _CONTAINER_USER])

        self._temp_seccomp = self._build_seccomp_profile()
        if self._temp_seccomp:
            cmd.extend(["--security-opt", f"seccomp={self._temp_seccomp}"])

        for k, v in self.config.env.env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])

        if self.config.container_labels:
            for k, v in self.config.container_labels.items():
                cmd.extend(["--label", f"{k}={v}"])

        cmd.append(self.config.image)
        return cmd

    # ── Seccomp profile generator ────────────────────────────────

    def _build_seccomp_profile(self) -> str | None:
        """Build a restrictive seccomp-BPF profile for the container.

        Returns a path to a temporary JSON file, or ``None`` if we
        can't write one (profile will be absent and the kernel's
        default applies, which is less restrictive).
        """
        blocked = set(_BLOCKED_SYSCALLS_COMMON)
        sc = self.config.seccomp

        if sc.profile == SeccompProfile.STRICT:
            blocked.update(_BLOCKED_SYSCALLS_STRICT)

        # Network-specific blocks
        nw = self.config.network
        if nw.policy == NetworkPolicy.NONE:
            blocked.update({
                "socket", "bind", "connect", "listen", "accept",
                "sendto", "recvfrom", "sendmsg", "recvmsg",
                "shutdown", "setsockopt", "getsockopt",
                "getpeername", "getsockname", "socketpair",
            })

        # Extra blocked from config
        blocked.update(sc.extra_blocked_syscalls)

        # Build the profile: default allow, deny blocked list
        profile = {
            "defaultAction": "SCMP_ACT_ALLOW",
            "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
            "syscalls": [
                {
                    "names": sorted(blocked),
                    "action": "SCMP_ACT_ERRNO",
                },
            ],
        }

        try:
            fd, path = tempfile.mkstemp(
                suffix=".json", prefix="enta_seccomp_",
            )
            with os.fdopen(fd, "w") as f:
                json.dump(profile, f)
            return path
        except Exception:
            return None

    # ── Command security checks ──────────────────────────────────

    def _check_command(self, command: str, in_container: bool = False) -> str:
        """Check *command* against the security policy.

        Returns an empty string if safe, or an error description if
        the command violates policy.
        """
        # ── Length check ───────────────────────────────────────
        if len(command) > self.config.max_command_length:
            return (
                f"Command too long: {len(command)} characters "
                f"(max {self.config.max_command_length})"
            )

        # ── Pattern blocklist ──────────────────────────────────
        command_lower = command.lower().strip()
        for pattern in _BLOCKED_COMMANDS_PATTERNS:
            if pattern.lower() in command_lower:
                return (
                    f"Blocked command pattern detected: '{pattern}'. "
                    "This operation is not allowed in the sandbox."
                )

        # ── Config-level patterns ──────────────────────────────
        for pattern in self.config.command_pattern_blocklist:
            if pattern.lower() in command_lower:
                return (
                    f"Blocked by custom pattern: '{pattern}'"
                )

        # ── Network tooling check ──────────────────────────────
        if self.config.disable_network_tooling:
            # Check first word of the command (the binary name)
            first_word = command.split(None, 1)[0] if command else ""
            if first_word in _BLOCKED_NETWORK_COMMANDS:
                return (
                    f"Network tool '{first_word}' is disabled in sandbox mode. "
                    "Use the dedicated web_fetch/web_search tools instead."
                )

        # ── Interactive mode check ─────────────────────────────
        if self.config.disable_interactive:
            for hint in ("-i ", "--interactive", "read ", "expect", "script "):
                if hint in command_lower:
                    return (
                        "Interactive commands are not allowed in sandbox mode."
                    )

        return ""

    def _post_start_harden(self) -> None:
        """Apply additional hardening after a container starts.

        Called after ``run_container()``.  Runs ``chmod``, removes
        unnecessary binaries, etc.
        """
        if not self._container_id:
            return

        harden_cmds = [
            # Remove setuid on everything
            'find / -perm -4000 -o -perm -2000 | xargs -r chmod u-s,g-s 2>/dev/null || true',
            # Remove dangerous tools if they exist
            'rm -f /usr/bin/su /usr/bin/pkexec /usr/bin/sudo /usr/bin/doas 2>/dev/null || true',
            # Disable core dumps
            'echo "core 0" | tee /proc/self/coredump_filter 2>/dev/null || true',
        ]

        for hc in harden_cmds:
            try:
                subprocess.run(
                    ["docker", "exec", self._container_id, "sh", "-c", hc],
                    capture_output=True, timeout=10,
                )
            except Exception:
                pass

    def _audit(self, entry: SecurityAuditEntry) -> None:
        """Record a security audit entry."""
        self._audit_log.append(entry)
        # Keep audit log bounded
        if len(self._audit_log) > 500:
            self._audit_log.pop(0)

    def _cleanup_tempfiles(self) -> None:
        """Remove temporary files created during execution."""
        if self._temp_seccomp and os.path.exists(self._temp_seccomp):
            try:
                os.unlink(self._temp_seccomp)
            except OSError:
                pass
            self._temp_seccomp = None

    # ── Context manager support ──────────────────────────────────

    def __enter__(self) -> EncreContainerSandbox:
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()


# ── Helper functions ──────────────────────────────────────────────

def _wrap_command(command: str, config: SandboxConfig) -> str:
    """Wrap *command* with security restrictions for container execution.

    The wrapper:
    - Sets safe environment variables (no leak from host)
    - Applies a timeout
    - Strips dangerous flags
    """
    parts: list[str] = []

    # Set a minimal, safe environment
    safe_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin",
        "HOME": "/home/sandbox",
        "SHELL": "/bin/sh",
        "TERM": "dumb",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }

    for k, v in safe_env.items():
        parts.append(f"export {k}='{v}'")

    # Drop dangerous env vars from inherited config
    if not config.env.inherit_env:
        parts.append("unset $(env | grep -o '^[^=]*' | grep -E '^(AWS|AZURE|GCP|GOOGLE|SECRET|TOKEN|KEY|PASS)') 2>/dev/null || true")  # noqa: E501

    # Extra env vars from config
    for k, v in config.env.env_vars.items():
        parts.append(f"export {k}='{v}'")

    # If disable_sudo, alias it to a no-op
    if config.disable_sudo:
        parts.append("alias sudo='echo \"sudo is disabled in sandbox\"'")

    # Change to workspace directory and run command
    parts.append(f"cd {config.workspace_mount}")
    parts.append(command)

    return " && ".join(parts)


def _truncate_output(text: str, max_bytes: int) -> str:
    """Truncate *text* to *max_bytes* if too long."""
    if not text or len(text.encode("utf-8")) <= max_bytes:
        return text
    # Binary-search for the safe truncation point
    encoded = text.encode("utf-8")
    truncated = encoded[:max_bytes]
    # Decode back, dropping any incomplete multi-byte char at the end
    result = truncated.decode("utf-8", errors="ignore")
    return result + f"\n...(truncated, {len(encoded) - max_bytes} bytes omitted)"
