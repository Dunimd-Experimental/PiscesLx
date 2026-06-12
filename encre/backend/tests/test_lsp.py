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

"""Tests for encre.lsp — LSP protocol dataclasses and EncreLSPClient."""

import pytest


# ===========================================================================
# Position dataclass
# ===========================================================================

class TestPosition:
    """Tests for the LSP Position dataclass."""

    def test_creation(self):
        from encre.lsp.protocol import Position
        p = Position(line=10, character=5)
        assert p.line == 10
        assert p.character == 5

    def test_zero_position(self):
        from encre.lsp.protocol import Position
        p = Position(line=0, character=0)
        assert p.line == 0
        assert p.character == 0

    def test_large_values(self):
        from encre.lsp.protocol import Position
        p = Position(line=99999, character=999)
        assert p.line == 99999
        assert p.character == 999

    def test_is_dataclass(self):
        from encre.lsp.protocol import Position
        from dataclasses import is_dataclass
        assert is_dataclass(Position)

    def test_equality(self):
        from encre.lsp.protocol import Position
        p1 = Position(line=5, character=10)
        p2 = Position(line=5, character=10)
        p3 = Position(line=5, character=11)
        assert p1 == p2
        assert p1 != p3


# ===========================================================================
# Range dataclass
# ===========================================================================

class TestRange:
    """Tests for the LSP Range dataclass."""

    def test_creation(self):
        from encre.lsp.protocol import Position, Range
        start = Position(line=0, character=0)
        end = Position(line=10, character=20)
        r = Range(start=start, end=end)
        assert r.start.line == 0
        assert r.start.character == 0
        assert r.end.line == 10
        assert r.end.character == 20

    def test_single_line_range(self):
        from encre.lsp.protocol import Position, Range
        start = Position(line=5, character=3)
        end = Position(line=5, character=15)
        r = Range(start=start, end=end)
        assert r.start.line == r.end.line
        assert r.end.character > r.start.character

    def test_is_dataclass(self):
        from encre.lsp.protocol import Range
        from dataclasses import is_dataclass
        assert is_dataclass(Range)


# ===========================================================================
# Location dataclass
# ===========================================================================

class TestLocation:
    """Tests for the LSP Location dataclass."""

    def test_creation(self):
        from encre.lsp.protocol import Position, Range, Location
        r = Range(start=Position(line=1, character=0), end=Position(line=1, character=10))
        loc = Location(uri="file:///test.py", range=r)
        assert loc.uri == "file:///test.py"
        assert loc.range.start.line == 1

    def test_file_uri(self):
        from encre.lsp.protocol import Position, Range, Location
        r = Range(start=Position(line=0, character=0), end=Position(line=0, character=5))
        loc = Location(uri="file:///home/user/project/main.py", range=r)
        assert loc.uri.startswith("file:///")

    def test_is_dataclass(self):
        from encre.lsp.protocol import Location
        from dataclasses import is_dataclass
        assert is_dataclass(Location)


# ===========================================================================
# Diagnostic dataclass
# ===========================================================================

class TestDiagnostic:
    """Tests for the LSP Diagnostic dataclass."""

    def test_creation(self):
        from encre.lsp.protocol import Position, Range, Diagnostic
        r = Range(start=Position(line=5, character=0), end=Position(line=5, character=10))
        diag = Diagnostic(
            range=r,
            message="Unused variable 'x'",
            severity=2,
            source="pyright",
        )
        assert diag.message == "Unused variable 'x'"
        assert diag.severity == 2
        assert diag.source == "pyright"

    def test_default_source(self):
        from encre.lsp.protocol import Position, Range, Diagnostic
        r = Range(start=Position(line=1, character=0), end=Position(line=1, character=5))
        diag = Diagnostic(range=r, message="Error", severity=1)
        assert diag.source == ""

    def test_severity_levels(self):
        from encre.lsp.protocol import Position, Range, Diagnostic
        r = Range(start=Position(line=0, character=0), end=Position(line=0, character=1))
        for sev in [1, 2, 3, 4]:
            diag = Diagnostic(range=r, message=f"Level {sev}", severity=sev)
            assert diag.severity == sev

    def test_is_dataclass(self):
        from encre.lsp.protocol import Diagnostic
        from dataclasses import is_dataclass
        assert is_dataclass(Diagnostic)


# ===========================================================================
# HoverResult dataclass
# ===========================================================================

class TestHoverResult:
    """Tests for the LSP HoverResult dataclass."""

    def test_creation_without_range(self):
        from encre.lsp.protocol import HoverResult
        hr = HoverResult(contents="def foo(x: int) -> str")
        assert hr.contents == "def foo(x: int) -> str"
        assert hr.range is None

    def test_creation_with_range(self):
        from encre.lsp.protocol import HoverResult, Position, Range
        r = Range(start=Position(line=1, character=0), end=Position(line=1, character=10))
        hr = HoverResult(contents="A string value", range=r)
        assert hr.contents == "A string value"
        assert hr.range is not None
        assert hr.range.start.line == 1

    def test_markdown_contents(self):
        from encre.lsp.protocol import HoverResult
        md = "```python\ndef foo() -> int: ...\n```"
        hr = HoverResult(contents=md)
        assert "```" in hr.contents

    def test_is_dataclass(self):
        from encre.lsp.protocol import HoverResult
        from dataclasses import is_dataclass
        assert is_dataclass(HoverResult)


# ===========================================================================
# LSPState dataclass
# ===========================================================================

class TestLSPState:
    """Tests for the LSPState dataclass."""

    def test_creation_running(self):
        from encre.lsp.protocol import LSPState
        state = LSPState(status="running")
        assert state.status == "running"
        assert state.error is None

    def test_creation_with_error(self):
        from encre.lsp.protocol import LSPState
        state = LSPState(status="stopped", error="connection refused")
        assert state.status == "stopped"
        assert state.error == "connection refused"

    def test_status_values(self):
        from encre.lsp.protocol import LSPState
        for status in ["starting", "running", "stopped", "error"]:
            s = LSPState(status=status)
            assert s.status == status

    def test_is_dataclass(self):
        from encre.lsp.protocol import LSPState
        from dataclasses import is_dataclass
        assert is_dataclass(LSPState)


# ===========================================================================
# EncreLSPClient
# ===========================================================================

class TestEncreLSPClient:
    """Tests for the EncreLSPClient class."""

    def test_construction(self):
        from encre.lsp.client import EncreLSPClient
        client = EncreLSPClient(server_name="pylsp")
        assert client is not None
        assert client._server_name == "pylsp"
        assert client._initialized is False
        assert client._request_id == 0

    def test_construction_different_server_names(self):
        from encre.lsp.client import EncreLSPClient
        for name in ["pylsp", "pyright", "rust-analyzer", "gopls", "typescript-language-server"]:
            client = EncreLSPClient(server_name=name)
            assert client._server_name == name

    def test_initial_state(self):
        from encre.lsp.client import EncreLSPClient
        client = EncreLSPClient(server_name="test")
        assert client._process is None
        assert client._initialized is False
        assert client._reader_task is None

    def test_close_before_start_does_not_raise(self):
        import asyncio
        from encre.lsp.client import EncreLSPClient

        async def _test():
            client = EncreLSPClient(server_name="test")
            await client.close()

        asyncio.run(_test())

    def test_public_api_exports(self):
        from encre.lsp import EncreLSPClient, EncreLSPManager, Position, Range, Location, Diagnostic, HoverResult, LSPState
        # Verify all public symbols are importable
        assert EncreLSPClient is not None
        assert EncreLSPManager is not None
        assert Position is not None
        assert Range is not None
        assert Location is not None
        assert Diagnostic is not None
        assert HoverResult is not None
        assert LSPState is not None


# ===========================================================================
# EncreLSPManager
# ===========================================================================

class TestEncreLSPManager:
    """Tests for the EncreLSPManager class."""

    def test_construction(self):
        from encre.lsp.manager import EncreLSPManager
        manager = EncreLSPManager()
        assert manager is not None
