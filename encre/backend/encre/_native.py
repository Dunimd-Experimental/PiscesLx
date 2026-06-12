#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure-Python fallback for the native Rust extension (``_native``).

When the compiled ``_native.pyd`` / ``_native.so`` is not available
(e.g. on Kaggle, Windows without Rust toolchain, etc.), this module
provides functional Python equivalents for all native exports.

All functions match the interface declared in ``_native.pyi``.
"""

from __future__ import annotations

import difflib
import glob as _glob
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from typing import Any, Iterator, List, Optional, Tuple


# ── File I/O ──────────────────────────────────────────────────────────

def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if offset > 0:
            f.seek(offset)
        return f.read(limit) if limit > 0 else f.read()


def write_file(path: str, content: str) -> bool:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


# ── Search ────────────────────────────────────────────────────────────

def grep(pattern: str, path: str, case_insensitive: bool = False) -> list[dict[str, Any]]:
    flags = re.IGNORECASE if case_insensitive else 0
    results = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                if re.search(pattern, line, flags):
                    results.append({"line": i, "content": line.rstrip("\n")})
    except Exception:
        pass
    return results


def glob(pattern: str, path: str | None = None) -> list[str]:
    cwd = path or os.getcwd()
    full = os.path.join(cwd, pattern)
    return _glob.glob(full, recursive=True)


# ── Token counting ─────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Approximate token count (4 chars per token)."""
    return max(1, len(text) // 4)


# ── Diff ──────────────────────────────────────────────────────────────

def compute_diff(old: str, new: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))


def apply_diff(content: str, diff: str) -> str:
    import patch
    try:
        result = patch.fromstring(diff).apply(content)
        return result if result else content
    except Exception:
        return content


# ── Sandbox ───────────────────────────────────────────────────────────

def sandbox_execute(command: str, timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "timeout"}
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": str(e)}


def sandbox_read_file(path: str) -> str:
    return read_file(path)


def sandbox_write_file(path: str, content: str) -> bool:
    return write_file(path, content)


# ── Embedding / similarity ────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-10)


def text_similarity(a: str, b: str) -> float:
    set_a = set(a.split())
    set_b = set(b.split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── SIMD (pure Python fallbacks) ──────────────────────────────────────

def simd_contains(haystack: str, needle: str) -> bool:
    return needle in haystack


def simd_find_all(haystack: str, needle: str) -> list[int]:
    results = []
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos == -1:
            break
        results.append(pos)
        start = pos + 1
    return results


def simd_memmem(haystack: bytes, needle: bytes) -> int | None:
    pos = haystack.find(needle)
    return pos if pos >= 0 else None


# ── Landlock (no-op stubs — Linux-only kernel feature) ────────────────

def landlock_restrict_read_only(paths: list[str]) -> None:
    pass


def landlock_restrict_network() -> None:
    pass


def landlock_full_sandbox(workspace: str) -> None:
    pass


def landlock_available() -> bool:
    return False


def landlock_abi_version() -> int:
    return 0


# ── LSP helpers ──────────────────────────────────────────────────────

def parse_lsp_message(raw: str) -> str:
    import json
    try:
        return json.dumps(json.loads(raw))
    except Exception:
        return raw


def parse_diagnostics(raw: str) -> list[dict[str, Any]]:
    import json
    try:
        data = json.loads(raw)
        params = data.get("params", {})
        return params.get("diagnostics", [])
    except Exception:
        return []


def build_lsp_request(id: int, method: str, params: str) -> str:
    import json
    return json.dumps({"jsonrpc": "2.0", "id": id, "method": method, "params": json.loads(params) if params else {}})


def build_content_length_header(content: str) -> str:
    return f"Content-Length: {len(content.encode('utf-8'))}\r\n\r\n"


# ── BM25 index (pure Python) ──────────────────────────────────────────

class Bm25Index:
    """Pure-Python BM25 index for code search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[str] = []
        self._paths: list[str] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._doc_freq: Counter[str] = Counter()
        self._built = False

    def build(self, files: list[tuple[str, str]]) -> None:
        self._paths = [p for p, _ in files]
        self._docs = [c for _, c in files]
        self._avgdl = sum(len(d.split()) for d in self._docs) / max(1, len(self._docs))
        self._doc_freq = Counter()
        for doc in self._docs:
            terms = set(doc.lower().split())
            for t in terms:
                self._doc_freq[t] += 1
        n = len(self._docs)
        self._idf = {
            t: math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for t, df in self._doc_freq.items()
        }
        self._built = True

    def search(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        if not self._built:
            return []
        query_terms = query.lower().split()
        scores = []
        for i, doc in enumerate(self._docs):
            doc_terms = doc.lower().split()
            dl = len(doc_terms)
            score = 0.0
            for qt in query_terms:
                if qt in self._idf:
                    tf = doc_terms.count(qt)
                    score += self._idf[qt] * (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                    )
            scores.append((self._paths[i], score))
        scores.sort(key=lambda x: -x[1])
        return scores[:limit]

    def add_document(self, path: str, content: str) -> None:
        self._paths.append(path)
        self._docs.append(content)
        self._built = False

    def remove_document(self, path: str) -> None:
        if path in self._paths:
            idx = self._paths.index(path)
            self._paths.pop(idx)
            self._docs.pop(idx)
            self._built = False

    def clear(self) -> None:
        self._paths.clear()
        self._docs.clear()
        self._built = False

    def __len__(self) -> int:
        return len(self._docs)

    def __bool__(self) -> bool:
        return len(self._docs) > 0


# ── Backward-compatible alias ────────────────────────────────────────
glob_pattern = glob

__all__ = [
    "glob_pattern",
    "Bm25Index",
    "apply_diff",
    "build_content_length_header",
    "build_lsp_request",
    "compute_diff",
    "cosine_similarity",
    "count_tokens",
    "glob",
    "grep",
    "landlock_abi_version",
    "landlock_available",
    "landlock_full_sandbox",
    "landlock_restrict_network",
    "landlock_restrict_read_only",
    "parse_diagnostics",
    "parse_lsp_message",
    "read_file",
    "sandbox_execute",
    "sandbox_read_file",
    "sandbox_write_file",
    "search_codebase",
    "simd_contains",
    "simd_find_all",
    "simd_memmem",
    "text_similarity",
    "write_file",
]

# Re-export search_codebase from codebase module if available
try:
    from encre.codebase.indexer import search_codebase  # type: ignore
except ImportError:
    def search_codebase(query: str, path: str | None = None) -> list[dict[str, Any]]:
        return []
