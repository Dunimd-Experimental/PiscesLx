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

"""Tests for the sandbox configuration, result types, and container sandbox."""

import os
import pytest


class TestSandboxConfig:
    def test_default_values(self):
        from encre.sandbox.types import SandboxConfig
        cfg = SandboxConfig()
        assert cfg.image == "python:3.11-slim"
        assert cfg.workspace_mount == "/workspace"
        assert cfg.network == "none"
        assert cfg.memory_limit == "512m"
        assert cfg.cpu_limit == 1.0
        assert cfg.timeout == 120
        assert cfg.read_only is False
        assert cfg.allowed_domains == []
        assert cfg.env_vars == {}
        assert cfg.extra_mounts == {}

    def test_custom_values(self):
        from encre.sandbox.types import SandboxConfig
        cfg = SandboxConfig(
            image="ubuntu:22.04",
            workspace_mount="/app",
            network="host",
            memory_limit="2g",
            cpu_limit=2.0,
            timeout=300,
            read_only=True,
            allowed_domains=["api.example.com"],
            env_vars={"DEBUG": "1"},
            extra_mounts={"/data": "/mnt/data"},
        )
        assert cfg.image == "ubuntu:22.04"
        assert cfg.workspace_mount == "/app"
        assert cfg.network == "host"
        assert cfg.memory_limit == "2g"
        assert cfg.cpu_limit == 2.0
        assert cfg.timeout == 300
        assert cfg.read_only is True
        assert cfg.allowed_domains == ["api.example.com"]
        assert cfg.env_vars == {"DEBUG": "1"}
        assert cfg.extra_mounts == {"/data": "/mnt/data"}

    def test_network_policy_values(self):
        from encre.sandbox.types import SandboxConfig
        none_cfg = SandboxConfig(network="none")
        assert none_cfg.network == "none"

        limited_cfg = SandboxConfig(network="limited")
        assert limited_cfg.network == "limited"

        host_cfg = SandboxConfig(network="host")
        assert host_cfg.network == "host"

    def test_multiple_allowed_domains(self):
        from encre.sandbox.types import SandboxConfig
        cfg = SandboxConfig(
            network="limited",
            allowed_domains=["pypi.org", "github.com", "registry.npmjs.org"],
        )
        assert len(cfg.allowed_domains) == 3
        assert "pypi.org" in cfg.allowed_domains

    def test_multiple_env_vars(self):
        from encre.sandbox.types import SandboxConfig
        cfg = SandboxConfig(
            env_vars={"PYTHONPATH": "/app", "NODE_ENV": "production", "LOG_LEVEL": "debug"},
        )
        assert cfg.env_vars["PYTHONPATH"] == "/app"
        assert cfg.env_vars["NODE_ENV"] == "production"
        assert len(cfg.env_vars) == 3

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        from encre.sandbox.types import SandboxConfig
        assert is_dataclass(SandboxConfig)


class TestSandboxResult:
    def test_basic_result(self):
        from encre.sandbox.types import SandboxResult
        result = SandboxResult(
            stdout="hello world\n",
            stderr="",
            exit_code=0,
        )
        assert result.stdout == "hello world\n"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.timed_out is False
        assert result.duration_ms == 0.0

    def test_error_result(self):
        from encre.sandbox.types import SandboxResult
        result = SandboxResult(
            stdout="",
            stderr="command not found: xxx",
            exit_code=127,
            timed_out=False,
            duration_ms=150.5,
        )
        assert result.exit_code == 127
        assert "command not found" in result.stderr
        assert result.timed_out is False
        assert result.duration_ms == 150.5

    def test_timeout_result(self):
        from encre.sandbox.types import SandboxResult
        result = SandboxResult(
            stdout="partial output",
            stderr="Command timed out",
            exit_code=-1,
            timed_out=True,
            duration_ms=120000.0,
        )
        assert result.timed_out is True
        assert result.exit_code == -1
        assert result.duration_ms == 120000.0

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        from encre.sandbox.types import SandboxResult
        assert is_dataclass(SandboxResult)


class TestEncreContainerSandbox:
    def test_construction_basic(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        assert sandbox.workspace == os.path.abspath("/tmp/test")
        assert sandbox._container_id is None
        assert sandbox._active is False

    def test_construction_with_config(self):
        from encre.sandbox.container import EncreContainerSandbox
        from encre.sandbox.types import SandboxConfig
        cfg = SandboxConfig(image="python:3.11-slim", timeout=60, memory_limit="256m")
        sandbox = EncreContainerSandbox(workspace="/tmp/test", config=cfg)
        assert sandbox.config.image == "python:3.11-slim"
        assert sandbox.config.timeout == 60
        assert sandbox.config.memory_limit == "256m"

    def test_is_available_returns_bool(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        result = sandbox.is_available()
        assert isinstance(result, bool)

    def test_context_manager_interface(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        assert hasattr(sandbox, "__enter__")
        assert hasattr(sandbox, "__exit__")

    def test_context_manager_enter_returns_self(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        with sandbox as s:
            assert s is sandbox

    def test_close_method(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        assert hasattr(sandbox, "close")
        sandbox.close()  # Should not raise even with no active container

    def test_cleanup_method(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        assert hasattr(sandbox, "cleanup")
        sandbox.cleanup()  # Should not raise even with no active container

    def test_execute_without_docker_returns_file_not_found(self):
        """When Docker is not installed, execute should return exit_code -2."""
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        result = sandbox.execute("echo hello")
        if sandbox.is_available():
            # Docker is available, test runs normally
            assert result.exit_code in (0, -2, -3)
        else:
            assert result.exit_code == -2
            assert "Docker not found" in result.stderr

    def test_execute_timeout_handling(self):
        """Test that a timeout returns exit_code -1 with timed_out=True."""
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        if sandbox.is_available():
            # Run a sleep command with a very short timeout
            result = sandbox.execute("sleep 10", timeout=1)
            assert result.timed_out is True
            assert result.exit_code == -1

    def test_stop_container_noop_when_no_container(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        sandbox.stop_container()  # Should not raise
        assert sandbox._container_id is None
        assert sandbox._active is False

    def test_exec_in_container_requires_active_container(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        with pytest.raises(RuntimeError, match="No active container"):
            sandbox.exec_in_container("echo hello")

    def test_run_container_requires_docker(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        if not sandbox.is_available():
            pytest.skip("Docker not available")
        container_id = sandbox.run_container()
        try:
            assert container_id is not None
            assert len(container_id) > 0
            assert sandbox._active is True
        finally:
            sandbox.cleanup()

    def test_exec_in_running_container(self):
        from encre.sandbox.container import EncreContainerSandbox
        sandbox = EncreContainerSandbox(workspace="/tmp/test")
        if not sandbox.is_available():
            pytest.skip("Docker not available")
        sandbox.run_container()
        try:
            result = sandbox.exec_in_container("echo hello")
            assert result.exit_code == 0
            assert "hello" in result.stdout
        finally:
            sandbox.cleanup()
