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

"""Shared pytest fixtures for the encre test suite."""

import os
import tempfile
from pathlib import Path

import pytest

from encre.config import EncreConfig
from encre.tools.builtin import (
    EncreBashTool,
    EncreFileReadTool,
    EncreFileWriteTool,
    EncreFileEditTool,
    EncreGlobTool,
    EncreGrepTool,
    EncreTaskCreateTool,
    EncreTaskGetTool,
    EncreTaskListTool,
    EncreTaskUpdateTool,
)
from encre.tools.registry import ToolRegistry


@pytest.fixture
def temp_dir():
    """Create a temporary directory with known test files, cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Readme
        (base / "README.md").write_text("# Test Project\n\nHello, world!\n", encoding="utf-8")

        # Python file
        (base / "main.py").write_text(
            "def hello():\n    return 'Hello, world!'\n\n\nif __name__ == '__main__':\n    print(hello())\n",
            encoding="utf-8",
        )

        # Nested directory with files
        (base / "src").mkdir(exist_ok=True)
        (base / "src" / "utils.py").write_text(
            "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n",
            encoding="utf-8",
        )
        (base / "src" / "config.json").write_text('{"key": "value", "debug": true}', encoding="utf-8")

        # Empty file
        (base / "empty.txt").write_text("", encoding="utf-8")

        # Hidden directory for edge-case tests
        (base / ".hidden").mkdir(exist_ok=True)
        (base / ".hidden" / "secret.txt").write_text("secret content\n", encoding="utf-8")

        # Binary-like extension
        (base / "data.bin").write_text("binary\x00content", encoding="utf-8")

        old_cwd = os.getcwd()
        os.chdir(str(base))
        try:
            yield str(base)
        finally:
            os.chdir(old_cwd)


@pytest.fixture
def sample_config():
    """Return a default EncreConfig instance usable in tests."""
    return EncreConfig(
        model="gpt-4o",
        permission_mode="default",
        max_turns=10,
        max_tokens=4096,
        log_level="WARNING",
    )


@pytest.fixture
def tool_registry():
    """Return a ToolRegistry pre-populated with common builtin tools."""
    registry = ToolRegistry()
    registry.register_many([
        EncreFileReadTool(),
        EncreFileWriteTool(),
        EncreFileEditTool(),
        EncreBashTool(),
        EncreGrepTool(),
        EncreGlobTool(),
        EncreTaskCreateTool(),
        EncreTaskGetTool(),
        EncreTaskListTool(),
        EncreTaskUpdateTool(),
    ])
    return registry
