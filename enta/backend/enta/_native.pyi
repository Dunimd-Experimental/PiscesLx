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

from typing import Any


def search_codebase(query: str, path: str | None = None) -> list[dict[str, Any]]:
    ...


def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    ...


def write_file(path: str, content: str) -> bool:
    ...


def grep(pattern: str, path: str, case_insensitive: bool = False) -> list[dict[str, Any]]:
    ...


def glob(pattern: str, path: str | None = None) -> list[str]:
    ...


def count_tokens(text: str) -> int:
    ...


def compute_diff(old: str, new: str) -> str:
    ...


def apply_diff(content: str, diff: str) -> str:
    ...


def sandbox_execute(
    command: str,
    timeout: int = 30,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Execute *command* in a sandboxed subprocess.

    When *workspace* is provided **and** Landlock is available (Linux
    5.13+), the child process is restricted to read/write access under
    *workspace* only — no network, no exec outside workspace.

    Falls back to a plain subprocess on non-Linux or when Landlock is
    unavailable (callers should layer Docker sandbox on top).
    """
    ...


def sandbox_read_file(path: str) -> str:
    ...


def sandbox_write_file(path: str, content: str) -> bool:
    ...


def execute_shell(command: str, cwd: str | None = None, timeout: int = 30) -> dict[str, Any]:
    ...


# ---------------------------------------------------------------------------
# New: embedding
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two f32 slices."""
    ...


def text_similarity(a: str, b: str) -> float:
    """Compute Jaccard text similarity on whitespace-delimited tokens."""
    ...


# ---------------------------------------------------------------------------
# New: simd_search
# ---------------------------------------------------------------------------

def simd_contains(haystack: str, needle: str) -> bool:
    """SIMD-accelerated substring check."""
    ...


def simd_find_all(haystack: str, needle: str) -> list[int]:
    """SIMD-accelerated find-all match positions (byte offsets)."""
    ...


def simd_memmem(haystack: bytes, needle: bytes) -> int | None:
    """SIMD-accelerated byte-level memmem."""
    ...


# ---------------------------------------------------------------------------
# New: landlock
# ---------------------------------------------------------------------------

def landlock_restrict_read_only(paths: list[str]) -> None:
    """Restrict the current thread to read-only filesystem access under paths."""
    ...


def landlock_restrict_network() -> None:
    """Restrict the current thread from making network connections."""
    ...


def landlock_full_sandbox(workspace: str) -> None:
    """Full sandbox: read-only filesystem under workspace, no network, no exec."""
    ...


def landlock_workspace_sandbox(workspace: str) -> None:
    """Workspace sandbox: read-write under workspace, no network, no exec.

    Suitable for running build tools (npm install, cargo build, etc.)
    in an isolated environment.  The calling thread can read, write,
    create, and remove files **only** under *workspace*.
    """
    ...


def landlock_available() -> bool:
    """Check whether Landlock is available on the current kernel."""
    ...


def landlock_abi_version() -> int:
    """Return the highest Landlock ABI version (0 if not available)."""
    ...


# ---------------------------------------------------------------------------
# New: lsp_proto
# ---------------------------------------------------------------------------

def parse_lsp_message(raw: str) -> str:
    """Parse a raw JSON-RPC 2.0 message string into a JSON dict string."""
    ...


def parse_diagnostics(raw: str) -> list[dict[str, Any]]:
    """Extract diagnostics from a publishDiagnostics notification params."""
    ...


def build_lsp_request(id: int, method: str, params: str) -> str:
    """Build a JSON-RPC 2.0 request string."""
    ...


def build_content_length_header(content: str) -> str:
    """Build an LSP Content-Length header for the given body."""
    ...


# ---------------------------------------------------------------------------
# BM25 indexer
# ---------------------------------------------------------------------------

class Bm25Index:
    """BM25-powered code search index.

    Usage::

        idx = Bm25Index()
        idx.build([("path.py", "def foo(): pass")])
        results = idx.search("foo", 10)
    """

    def __init__(self) -> None: ...
    def build(self, files: list[tuple[str, str]]) -> None:
        """Build or rebuild the index from (path, content) pairs."""
        ...
    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Search with BM25 ranking. Returns list of (path, score)."""
        ...
    def add_document(self, path: str, content: str) -> None:
        """Add or update a single document."""
        ...
    def remove_document(self, path: str) -> None:
        """Remove a document by path."""
        ...
    def clear(self) -> None:
        """Clear the entire index."""
        ...
    def __len__(self) -> int: ...
    def __bool__(self) -> bool: ...
