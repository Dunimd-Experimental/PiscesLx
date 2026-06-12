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

import asyncio
import json
import os
import subprocess
import tempfile
import time
import uuid

from encre.sandbox.types import SandboxConfig, SandboxResult


class EncreContainerSandbox:
    def __init__(self, workspace: str, config: SandboxConfig | None = None) -> None:
        self.workspace = os.path.abspath(workspace)
        self.config = config or SandboxConfig()
        self._container_id: str | None = None
        self._active: bool = False
        self._temp_seccomp: str | None = None

    def _build_docker_command(self, command: str) -> list[str]:
        cmd = ["docker", "run", "--rm"]

        if self.config.network == "none":
            cmd.append("--network=none")
        elif self.config.network == "host":
            cmd.append("--network=host")
        elif self.config.network == "limited":
            cmd.append("--network=bridge")
            for domain in self.config.allowed_domains:
                cmd.extend(["--add-host", domain])

        cmd.extend(["--memory", self.config.memory_limit])
        cmd.extend(["--cpus", str(self.config.cpu_limit)])

        mount_mode = "ro" if self.config.read_only else "rw"
        cmd.extend(["-v", f"{self.workspace}:{self.config.workspace_mount}:{mount_mode}"])

        for host_path, container_path in self.config.extra_mounts.items():
            mount_opts = "ro" if self.config.read_only else "rw"
            cmd.extend(["-v", f"{host_path}:{container_path}:{mount_opts}"])

        if self.config.read_only:
            cmd.append("--read-only")
            cmd.append("--tmpfs=/tmp:exec")
            cmd.append("--tmpfs=/var/tmp:exec")

        for k, v in self.config.env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])

        self._temp_seccomp = self._build_seccomp_profile()
        if self._temp_seccomp:
            cmd.extend(["--security-opt", f"seccomp={self._temp_seccomp}"])

        cmd.append(self.config.image)
        cmd.extend(["sh", "-c", f"cd {self.config.workspace_mount} && {command}"])

        return cmd

    def _build_seccomp_profile(self) -> str | None:
        profile = {
            "defaultAction": "SCMP_ACT_ALLOW",
            "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
            "syscalls": [
                {
                    "names": [
                        "clone", "fork", "vfork", "kill", "ptrace",
                        "mount", "umount2", "pivot_root", "chroot",
                        "kexec_load", "kexec_file_load", "init_module",
                        "finit_module", "delete_module", "add_key",
                        "request_key", "keyctl", "iopl", "ioperm",
                        "swapon", "swapoff", "reboot", "acct",
                        "settimeofday", "adjtimex", "clock_settime",
                        "create_module", "get_kernel_syms", "query_module",
                        "nfsservctl", "_sysctl", "bdflush", "sysfs",
                        "uselib", "syslog", "perf_event_open",
                        "fanotify_init", "bpf", "userfaultfd",
                    ],
                    "action": "SCMP_ACT_ERRNO",
                }
            ],
        }

        if self.config.network == "none":
            profile["syscalls"][0]["names"].extend([
                "socket", "bind", "connect", "listen", "accept",
                "sendto", "recvfrom", "sendmsg", "recvmsg",
                "shutdown", "setsockopt", "getsockopt",
                "getpeername", "getsockname", "socketpair",
            ])

        try:
            fd, path = tempfile.mkstemp(suffix=".json", prefix="yim_seccomp_")
            with os.fdopen(fd, "w") as f:
                json.dump(profile, f)
            return path
        except Exception:
            return None

    def execute(self, command: str, timeout: int | None = None) -> SandboxResult:
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
            return SandboxResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration_ms=(time.time() - start) * 1000,
            )
        except subprocess.TimeoutExpired:
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
            return SandboxResult(
                stdout="",
                stderr=f"Sandbox execution error: {e}",
                exit_code=-3,
                duration_ms=(time.time() - start) * 1000,
            )
        finally:
            if self._temp_seccomp and os.path.exists(self._temp_seccomp):
                try:
                    os.unlink(self._temp_seccomp)
                    self._temp_seccomp = None
                except OSError:
                    pass

    async def execute_async(self, command: str, timeout: int | None = None) -> SandboxResult:
        start = time.time()
        cmd = self._build_docker_command(command)
        effective_timeout = timeout or self.config.timeout
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=effective_timeout
                )
                return SandboxResult(
                    stdout=stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else "",
                    stderr=stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else "",
                    exit_code=proc.returncode or 0,
                    duration_ms=(time.time() - start) * 1000,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
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
            return SandboxResult(
                stdout="",
                stderr=f"Sandbox execution error: {e}",
                exit_code=-3,
                duration_ms=(time.time() - start) * 1000,
            )
        finally:
            if self._temp_seccomp and os.path.exists(self._temp_seccomp):
                try:
                    os.unlink(self._temp_seccomp)
                    self._temp_seccomp = None
                except OSError:
                    pass

    def run_container(self) -> str:
        container_name = f"yim_sandbox_{uuid.uuid4().hex[:12]}"
        cmd = ["docker", "run", "-d", "--name", container_name]

        if self.config.network == "none":
            cmd.append("--network=none")
        elif self.config.network == "host":
            cmd.append("--network=host")
        elif self.config.network == "limited":
            cmd.append("--network=bridge")
            for domain in self.config.allowed_domains:
                cmd.extend(["--add-host", domain])

        cmd.extend(["--memory", self.config.memory_limit])
        cmd.extend(["--cpus", str(self.config.cpu_limit)])

        mount_mode = "ro" if self.config.read_only else "rw"
        cmd.extend(["-v", f"{self.workspace}:{self.config.workspace_mount}:{mount_mode}"])

        for host_path, container_path in self.config.extra_mounts.items():
            mount_opts = "ro" if self.config.read_only else "rw"
            cmd.extend(["-v", f"{host_path}:{container_path}:{mount_opts}"])

        if self.config.read_only:
            cmd.append("--read-only")
            cmd.append("--tmpfs=/tmp:exec")
            cmd.append("--tmpfs=/var/tmp:exec")

        for k, v in self.config.env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])

        seccomp_path = self._build_seccomp_profile()
        if seccomp_path:
            cmd.extend(["--security-opt", f"seccomp={seccomp_path}"])

        cmd.extend([self.config.image, "tail", "-f", "/dev/null"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr.strip()}")

        self._container_id = result.stdout.strip()
        self._active = True
        return self._container_id

    def exec_in_container(self, command: str) -> SandboxResult:
        if not self._container_id or not self._active:
            raise RuntimeError("No active container. Call run_container() first.")

        start = time.time()
        cmd = ["docker", "exec", "-w", self.config.workspace_mount, self._container_id, "sh", "-c", command]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                encoding="utf-8",
                errors="replace",
            )
            return SandboxResult(
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration_ms=(time.time() - start) * 1000,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stdout="",
                stderr="Command timed out in container",
                exit_code=-1,
                timed_out=True,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return SandboxResult(
                stdout="",
                stderr=f"Container exec error: {e}",
                exit_code=-3,
                duration_ms=(time.time() - start) * 1000,
            )

    def stop_container(self) -> None:
        if not self._container_id:
            return

        try:
            subprocess.run(
                ["docker", "stop", "-t", "5", self._container_id],
                capture_output=True, timeout=15,
            )
        except Exception:
            pass

        try:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass

        self._container_id = None
        self._active = False

    def close(self) -> None:
        """Stop and remove Docker container, clean up seccomp tempfiles."""
        self.cleanup()

    def cleanup(self) -> None:
        self.stop_container()
        if self._temp_seccomp and os.path.exists(self._temp_seccomp):
            try:
                os.unlink(self._temp_seccomp)
            except OSError:
                pass
        self._temp_seccomp = None

    def is_available(self) -> bool:
        try:
            subprocess.run(["docker", "version"], capture_output=True, timeout=5)
            return True
        except Exception:
            return False

    def __enter__(self) -> EncreContainerSandbox:
        return self

    def __exit__(self, *args: object) -> None:
        self.cleanup()
