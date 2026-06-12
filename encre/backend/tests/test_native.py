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

"""Tests for encre.native — Rust native bridge with Python fallbacks."""

import os
import tempfile
from pathlib import Path

import pytest

from encre import native


class TestNativeImport:
    """Verify the native module is importable and has the expected API."""

    def test_module_importable(self):
        """The native bridge module should always be importable."""
        assert native is not None

    def test_has_native_flag_exists(self):
        """_HAS_NATIVE is a boolean indicating whether the Rust extension loaded."""
        assert isinstance(native._HAS_NATIVE, bool)

    def test_all_functions_exist(self):
        """Every function defined in _native.pyi stubs must be present in native.py."""
        expected = [
            "read_file",
            "write_file",
            "grep",
            "glob_pattern",
            "count_tokens",
            "compute_diff",
            "apply_diff",
            "sandbox_execute",
            "sandbox_read_file",
            "sandbox_write_file",
            "search_codebase",
        ]
        for name in expected:
            assert hasattr(native, name), f"Missing function: {name}"
            assert callable(getattr(native, name)), f"Not callable: {name}"

    def test_pyi_stubs_match(self):
        """_native.pyi stub signatures should exist and be callable."""
        try:
            from encre import _native as _rust_native  # type: ignore
        except ImportError:
            pytest.skip("Rust _native extension not built (expected in dev)")

        # The Rust extension should have the functions declared in _native.pyi
        expected = [
            "search_codebase",
            "read_file",
            "write_file",
            "grep",
            "glob",
            "count_tokens",
            "compute_diff",
            "apply_diff",
            "sandbox_execute",
            "sandbox_read_file",
            "sandbox_write_file",
        ]
        for name in expected:
            assert hasattr(_rust_native, name), f"Rust _native missing: {name}"


class TestReadWriteFile:
    """Test file reading and writing with temp files."""

    def test_write_and_read_file(self, tmp_path: Path):
        filepath = str(tmp_path / "test_file.txt")
        content = "Hello, encre native tests!\nLine two.\n"

        assert native.write_file(filepath, content) is True
        result = native.read_file(filepath)
        # Native read may or may not preserve trailing newline depending on impl
        assert "Hello, encre native tests!" in result
        assert "Line two" in result

    def test_read_file_with_offset(self, tmp_path: Path):
        filepath = str(tmp_path / "offset_test.txt")
        lines = "line_1\nline_2\nline_3\nline_4\n"
        native.write_file(filepath, lines)

        result = native.read_file(filepath, offset=2)  # 1-indexed
        assert "line_2" in result

    def test_read_file_with_offset_and_limit(self, tmp_path: Path):
        filepath = str(tmp_path / "limit_test.txt")
        lines = "a\nb\nc\nd\ne\n"
        native.write_file(filepath, lines)

        result = native.read_file(filepath, offset=2, limit=2)
        parts = result.strip().splitlines()
        assert len(parts) <= 3  # offset=2 starts at line 2

    def test_read_file_not_found(self, tmp_path: Path):
        filepath = str(tmp_path / "does_not_exist.txt")
        with pytest.raises(FileNotFoundError):
            native.read_file(filepath)

    def test_write_file_creates_directories(self, tmp_path: Path):
        filepath = str(tmp_path / "deep" / "nested" / "dir" / "file.txt")
        content = "deeply nested content"
        assert native.write_file(filepath, content) is True
        assert native.read_file(filepath) == content


class TestGrep:
    """Test grep (regex search) function."""

    def test_grep_content_mode(self, tmp_path: Path):
        filepath = str(tmp_path / "grep_test.py")
        native.write_file(filepath, "def foo():\n    return 42\n\ndef bar():\n    return 99\n")
        result = native.grep(r"def \w+", filepath)
        assert "def foo" in result
        assert "def bar" in result

    def test_grep_case_insensitive(self, tmp_path: Path):
        filepath = str(tmp_path / "case_test.txt")
        native.write_file(filepath, "HELLO world\nhello WORLD\n")
        result = native.grep("hello", filepath, case_insensitive=True)
        # Should match both lines
        assert result.count("hello") + result.count("HELLO") >= 2 or "2 match" in result

    def test_grep_files_with_matches_mode(self, tmp_path: Path):
        filepath = str(tmp_path / "fwm_test.py")
        native.write_file(filepath, "def test():\n    pass\n")
        result = native.grep("def", filepath, output_mode="files_with_matches")
        assert filepath in result or "fwm_test" in result

    def test_grep_count_mode(self, tmp_path: Path):
        filepath = str(tmp_path / "count_test.py")
        native.write_file(filepath, "def a():\n    pass\n\ndef b():\n    pass\n")
        result = native.grep("def", filepath, output_mode="count")
        assert "2" in result

    def test_grep_no_match(self, tmp_path: Path):
        filepath = str(tmp_path / "no_match.txt")
        native.write_file(filepath, "just some text\n")
        result = native.grep("NOTFOUND", filepath)
        assert "No matches" in result or result == "No matches found."

    def test_grep_invalid_regex(self, tmp_path: Path):
        filepath = str(tmp_path / "bad_regex.txt")
        native.write_file(filepath, "content\n")
        result = native.grep("[invalid", filepath)
        assert "Error" in result


class TestGlobPattern:
    """Test glob pattern matching."""

    def test_glob_finds_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = native.glob_pattern("*.py", str(tmp_path))
        assert len(result) == 2
        assert any("a.py" in p for p in result)
        assert any("b.py" in p for p in result)

    def test_glob_no_match(self, tmp_path: Path):
        result = native.glob_pattern("*.xyz", str(tmp_path))
        assert isinstance(result, list)
        assert len(result) == 0

    def test_glob_default_path(self, tmp_path: Path):
        # Create files in current/working context
        (tmp_path / "hello.md").write_text("")
        result = native.glob_pattern("*.md", str(tmp_path))
        assert len(result) >= 1


class TestCountTokens:
    """Test token counting."""

    def test_count_tokens_returns_int(self):
        result = native.count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty_string(self):
        result = native.count_tokens("")
        assert result == 0

    def test_count_tokens_long_text(self):
        text = "The quick brown fox " * 100
        result = native.count_tokens(text)
        assert result > 50  # rough estimate at chars/4

    def test_count_tokens_whitespace_only(self):
        result = native.count_tokens("   \t\n  ")
        # Implementation differs: Rust may count spaces, Python strips
        assert isinstance(result, int)


class TestDiff:
    """Test compute_diff and apply_diff."""

    def test_compute_diff_identical(self):
        diff = native.compute_diff("hello\nworld\n", "hello\nworld\n")
        assert isinstance(diff, str)

    def test_compute_diff_changed(self):
        diff = native.compute_diff("hello\nworld\n", "hello\nuniverse\n")
        # Native implementations may use different diff formats
        assert isinstance(diff, str)
        assert len(diff) > 0  # changed content should produce non-empty diff

    def test_apply_diff_simple(self):
        original = "hello\nworld\n"
        diff = native.compute_diff(original, "hello\nuniverse\n")
        result = native.apply_diff(original, diff)
        assert "universe" in result

    def test_apply_diff_roundtrip(self):
        old = "line1\nline2\nline3\n"
        new = "line1\nline2_modified\nline3\nline4\n"
        diff = native.compute_diff(old, new)
        applied = native.apply_diff(old, diff)
        assert applied == new

    def test_compute_diff_empty_strings(self):
        diff = native.compute_diff("", "")
        assert isinstance(diff, str)


class TestSandboxExecute:
    """Test sandbox_execute (runs locally for development)."""

    def test_sandbox_echo(self):
        result = native.sandbox_execute("echo hello", timeout=10)
        assert isinstance(result, dict)
        assert "stdout" in result
        assert "stderr" in result
        assert "exit_code" in result
        assert "hello" in result["stdout"]

    def test_sandbox_exit_code_success(self):
        result = native.sandbox_execute("exit 0", timeout=10)
        assert result["exit_code"] == 0

    def test_sandbox_stderr(self):
        result = native.sandbox_execute("echo error >&2", timeout=10)
        assert "error" in result["stderr"] or result["exit_code"] is not None


class TestSandboxFileOps:
    """Test sandbox file read/write operations."""

    def test_sandbox_write_and_read(self, tmp_path: Path):
        filepath = str(tmp_path / "sandbox_file.txt")
        assert native.sandbox_write_file(filepath, "sandbox content") is True
        result = native.sandbox_read_file(filepath)
        assert result == "sandbox content"

    def test_sandbox_read_missing(self, tmp_path: Path):
        filepath = str(tmp_path / "sandbox_missing.txt")
        with pytest.raises(FileNotFoundError):
            native.sandbox_read_file(filepath)


class TestSearchCodebase:
    """Test search_codebase function."""

    def test_search_finds_content(self, tmp_path: Path):
        (tmp_path / "sample.py").write_text("def my_function():\n    return True\n")
        results = native.search_codebase("my_function", str(tmp_path))
        assert isinstance(results, list)
        assert len(results) > 0
        assert any("sample.py" in r.get("file_path", "") for r in results)

    def test_search_no_match(self, tmp_path: Path):
        (tmp_path / "data.txt").write_text("ordinary text here\n")
        results = native.search_codebase("XYZ-NONEXISTENT", str(tmp_path))
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_default_path(self):
        results = native.search_codebase("def")
        assert isinstance(results, list)
