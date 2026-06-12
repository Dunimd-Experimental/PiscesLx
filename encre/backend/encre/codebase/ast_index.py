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

"""
AST-based code index powered by `tree-sitter`.

This module implements :class:`EncreASTIndex`, a workspace-level AST
indexer that gives Encre Codex-class code understanding on top of the
existing BM25 keyword index.  It is a deliberately **independent**
index from :class:`encre.codebase.indexer.EncreCodeIndex`; the two are
composed by callers that want both fast keyword search and structural
queries (``goto_definition``, ``find_references``, ``get_outline``).

Features
--------
- **Multi-language parsing** — Python, JavaScript, TypeScript, TSX, Rust,
  Go, Java, C, C++, C#, Ruby, PHP, Swift, Kotlin, Scala, all via the
  `tree-sitter-language-pack` wheel of prebuilt parsers (with a
  graceful fallback to `tree-sitter-languages` if that pack is not
  installed).
- **Per-file symbol table** — every parsed file yields a flat list of
  :class:`Symbol` records for top-level definitions plus their nested
  methods / inner classes.
- **Global symbol index** — ``(name) -> [Symbol]`` so callers can do
  a workspace-wide lookup in O(1) plus a tiny filter step.
- **Cross-file references** — fast regex-based scan over indexed
  files, returning line/column tuples for every match.  The regex
  uses word boundaries and respects identifier syntax (``[A-Za-z_]
  [A-Za-z0-9_]*``).
- **Incremental scanning** — the scanner tracks per-file mtime and
  only re-parses files that have actually changed since the last
  scan, preserving the symbol tables of untouched files.
- **Persistence** — the index is serialised to
  ``{workspace}/.encre/ast_index.json`` so subsequent server starts
  can resume from disk without re-parsing.

Design notes
------------
This index runs **on demand** from the agent's perspective.  The
caller (``IndexManager``) is responsible for off-loading the scan
work to a background thread or subprocess — the methods on this class
are all synchronous and CPU-bound.  ``load()`` is intentionally fast
(it only deserialises JSON) and safe to call from the main event
loop.

If tree-sitter is not available in the active Python environment, the
index degrades to a no-op shell: :meth:`scan` records no symbols and
:meth:`available` returns ``False``.  Callers should check
:meth:`available` before relying on AST results.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("encre.codebase.ast_index")

# ---------------------------------------------------------------------------
# Optional tree-sitter import — both backends are tried in turn
# ---------------------------------------------------------------------------

_TS_BACKEND: str | None = None
_TS_GET_PARSER: Callable[[str], Any] | None = None

try:
    import tree_sitter_language_pack as _tslp  # type: ignore[import-not-found]

    def _tslp_get_parser(lang: str) -> Any:
        return _tslp.get_parser(lang)

    _TS_GET_PARSER = _tslp_get_parser
    _TS_BACKEND = "tree-sitter-language-pack"
except ImportError:
    try:
        from tree_sitter_languages import get_parser as _ts_languages_get_parser  # type: ignore[import-not-found]

        _TS_GET_PARSER = _ts_languages_get_parser
        _TS_BACKEND = "tree-sitter-languages"
    except ImportError:
        _TS_GET_PARSER = None
        _TS_BACKEND = None


def _ts_available() -> bool:
    return _TS_GET_PARSER is not None


def _ts_backend_name() -> str | None:
    return _TS_BACKEND


# ---------------------------------------------------------------------------
# Language → file-extension map.  Only languages whose parsers ship in
# `tree-sitter-language-pack` are listed.  Adding a new language is a
# single line: register the extension here and (optionally) extend
# ``DEFINITION_NODE_TYPES`` below for richer symbol extraction.
# ---------------------------------------------------------------------------

LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".pyx": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sc": "scala",
}

# tree-sitter node types that introduce a named symbol, mapped to the
# human-readable kind we expose through :class:`Symbol.kind`.  Any
# tree-sitter parser not listed here still parses the file but does
# not produce symbol records (callers fall back to BM25 search).

DEFINITION_NODE_TYPES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
    },
    "javascript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "variable_declarator": "variable",
    },
    "typescript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type_alias",
        "enum_declaration": "enum",
        "variable_declarator": "variable",
        "lexical_declaration": "variable",
    },
    "tsx": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type_alias",
        "enum_declaration": "enum",
        "variable_declarator": "variable",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "impl",
        "type_item": "type_alias",
        "const_item": "constant",
        "static_item": "constant",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "type",
        "const_declaration": "constant",
        "var_declaration": "variable",
    },
    "java": {
        "method_declaration": "method",
        "constructor_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "annotation_type_declaration": "interface",
    },
    "c": {
        "function_definition": "function",
        "struct_specifier": "struct",
        "union_specifier": "struct",
        "enum_specifier": "enum",
        "type_definition": "type_alias",
    },
    "cpp": {
        "function_definition": "function",
        "function_decl": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "union_specifier": "struct",
        "enum_specifier": "enum",
        "namespace_definition": "namespace",
    },
    "csharp": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "struct_declaration": "struct",
        "enum_declaration": "enum",
        "record_declaration": "class",
        "delegate_declaration": "type_alias",
    },
    "ruby": {
        "method": "method",
        "class": "class",
        "module": "module",
    },
    "php": {
        "function_definition": "function",
        "class_declaration": "class",
        "method_declaration": "method",
        "interface_declaration": "interface",
        "trait_declaration": "trait",
    },
    "swift": {
        "function_declaration": "function",
        "class_declaration": "class",
        "protocol_declaration": "interface",
        "enum_declaration": "enum",
        "struct_declaration": "struct",
        "extension_declaration": "extension",
    },
    "kotlin": {
        "function_declaration": "function",
        "class_declaration": "class",
        "object_declaration": "class",
        "interface_declaration": "interface",
        "property_declaration": "property",
    },
    "scala": {
        "function_definition": "function",
        "class_definition": "class",
        "object_definition": "class",
        "trait_definition": "trait",
    },
}


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class Symbol:
    """A named program element extracted from a parsed file.

    Attributes:
        name: Identifier as it appears in source (e.g. ``MyClass``,
            ``compute_hash``).
        kind: Human-readable kind — ``"function"``, ``"class"``,
            ``"method"``, ``"interface"``, ``"struct"``, ``"enum"``,
            ``"trait"``, ``"type_alias"``, ``"constant"``,
            ``"variable"``, ``"property"``, ``"namespace"``,
            ``"impl"``, ``"extension"``, ``"module"``.
        file: Workspace-relative path of the file that contains the
            definition.
        start_line: 0-based starting line.
        start_col: 0-based starting column (in bytes).
        end_line: 0-based ending line (inclusive of the closing brace
            of a block-scoped definition).
        end_col: 0-based ending column.
        parent: Name of the enclosing symbol, or ``None`` for
            top-level definitions.
        signature: First line of the definition, trimmed — useful
            for hover-style previews without re-parsing.
        docstring: Extracted docstring / leading comment, if any.
            Only Python definitions have a real docstring here; for
            other languages the field is ``None``.
    """

    name: str
    kind: str
    file: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    parent: str | None = None
    signature: str | None = None
    docstring: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Symbol:
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            file=str(data["file"]),
            start_line=int(data["start_line"]),
            start_col=int(data["start_col"]),
            end_line=int(data["end_line"]),
            end_col=int(data["end_col"]),
            parent=data.get("parent"),
            signature=data.get("signature"),
            docstring=data.get("docstring"),
        )


@dataclass
class Reference:
    """A textual reference to a name found in an indexed file.

    Attributes:
        file: Workspace-relative path of the file containing the
            reference.
        line: 0-based line.
        col: 0-based column (in bytes).
        name: The identifier that was matched.
        kind: ``"definition"`` for declarations extracted by the
            parser, ``"reference"`` for any other textual occurrence.
    """

    file: str
    line: int
    col: int
    name: str
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main index
# ---------------------------------------------------------------------------


class EncreASTIndex:
    """Tree-sitter powered AST index for a single workspace.

    The index keeps two complementary data structures in memory:

    - ``_symbols_by_file``: ``{rel_path: [Symbol, ...]}`` — the symbol
      table for each indexed file.
    - ``_global_index``: ``{name: [Symbol, ...]}`` — a name-keyed
      secondary index for O(1) workspace-wide lookup.

    Both are persisted together to ``.encre/ast_index.json`` after
    every scan / incremental update.
    """

    _SKIP_DIRS: frozenset[str] = frozenset({
        "node_modules", "__pycache__", "target", "build", "dist",
        ".git", "venv", ".venv", "env", ".tox", ".eggs",
        ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".svn", ".hg",
    })

    _MAX_FILE_SIZE: int = 2 * 1024 * 1024

    _IDENTIFIER_RE: re.Pattern[str] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    def __init__(self, workspace: str) -> None:
        self.workspace: str = workspace
        self._symbols_by_file: dict[str, list[Symbol]] = {}
        self._global_index: dict[str, list[Symbol]] = {}
        self._file_mtimes: dict[str, float] = {}
        self._parsers: dict[str, Any] = {}
        self._indexed: bool = False
        self.load()

    # ── Public properties ────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Return ``True`` if a real tree-sitter parser is importable."""
        return _ts_available()

    @property
    def backend(self) -> str | None:
        """Return the active tree-sitter backend name, or ``None``."""
        return _ts_backend_name()

    # ── Parsing helpers ───────────────────────────────────────────────

    def _get_parser(self, lang: str) -> Any:
        if _TS_GET_PARSER is None:
            return None
        if lang in self._parsers:
            return self._parsers[lang]
        try:
            parser = _TS_GET_PARSER(lang)
        except Exception:
            return None
        self._parsers[lang] = parser
        return parser

    def _parse_file(self, rel_path: str, content: str, lang: str) -> list[Symbol]:
        parser = self._get_parser(lang)
        if parser is None:
            return []
        try:
            tree = parser.parse(content.encode("utf-8"))
        except Exception:
            return []
        root = tree.root_node
        symbols: list[Symbol] = []
        self._walk(root, rel_path, content, lang, parent=None, out=symbols)
        return symbols

    def _walk(
        self,
        node: Any,
        rel_path: str,
        content: str,
        lang: str,
        parent: str | None,
        out: list[Symbol],
    ) -> None:
        def_types = DEFINITION_NODE_TYPES.get(lang, {})
        for child in node.children:
            kind = def_types.get(child.type)
            if kind is not None:
                name = self._extract_name(child, content)
                if name:
                    sig = self._extract_signature(child, content)
                    doc = self._extract_docstring(child, content, lang)
                    sym = Symbol(
                        name=name,
                        kind=kind,
                        file=rel_path,
                        start_line=int(child.start_point[0]),
                        start_col=int(child.start_point[1]),
                        end_line=int(child.end_point[0]),
                        end_col=int(child.end_point[1]),
                        parent=parent,
                        signature=sig,
                        docstring=doc,
                    )
                    out.append(sym)
                    # Recurse so nested classes / methods are picked up
                    # with the correct ``parent`` chain.
                    self._walk(child, rel_path, content, lang, parent=name, out=out)
                else:
                    self._walk(child, rel_path, content, lang, parent=parent, out=out)
            else:
                self._walk(child, rel_path, content, lang, parent=parent, out=out)

    @staticmethod
    def _extract_name(node: Any, content: str) -> str | None:
        """Extract the identifier that names this definition node."""
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return content[name_node.start_byte:name_node.end_byte]
        # JS/TS: ``const x = 1;`` -> ``variable_declarator`` is the
        # grandchild of the lexical declaration; its first identifier
        # child is the name.
        if node.type == "variable_declarator":
            for c in node.children:
                if c.type in ("identifier", "property_identifier"):
                    return content[c.start_byte:c.end_byte]
        # Rust: ``const_item`` / ``static_item`` expose their name via
        # the ``name`` field on a child ``identifier`` node.
        if node.type in ("const_item", "static_item"):
            for c in node.children:
                if c.type == "identifier":
                    return content[c.start_byte:c.end_byte]
        # C: ``type_definition { ... } X;`` — last child is the alias.
        if node.type == "type_definition":
            for c in reversed(node.children):
                if c.type == "type_identifier":
                    return content[c.start_byte:c.end_byte]
        # Go: ``const_declaration`` / ``var_declaration`` have nested
        # spec groups whose first child is the name.
        if node.type in ("const_declaration", "var_declaration"):
            for c in node.children:
                if c.type in ("const_spec", "var_spec"):
                    for cc in c.children:
                        if cc.type == "identifier":
                            return content[cc.start_byte:cc.end_byte]
        return None

    @staticmethod
    def _extract_signature(node: Any, content: str) -> str | None:
        """Return the first non-empty line of the definition, trimmed."""
        start = node.start_point[0]
        if start < 0:
            return None
        lines = content.split("\n")
        if start >= len(lines):
            return None
        first = lines[start]
        if first and first.strip():
            return first.strip()[:200]
        # If the first line is empty (rare), peek at the next one.
        if start + 1 < len(lines) and lines[start + 1].strip():
            return lines[start + 1].strip()[:200]
        return None

    @staticmethod
    def _extract_docstring(node: Any, content: str, lang: str) -> str | None:
        """Extract a Python docstring from a definition node, if present."""
        if lang != "python":
            return None
        body = node.child_by_field_name("body")
        if body is None or not body.children:
            return None
        first = body.children[0]
        if first.type != "string":
            return None
        raw = content[first.start_byte:first.end_byte]
        # Strip triple- or single-quote wrappers; keep inner text.
        for triple in ('"""', "'''"):
            if raw.startswith(triple) and raw.endswith(triple) and len(raw) >= 6:
                return raw[3:-3].strip()[:500]
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
            return raw[1:-1].strip()[:500]
        return None

    # ── Scanning ──────────────────────────────────────────────────────

    def scan(self, progress_cb: Optional[Callable[[str, int], None]] = None) -> None:
        """Perform a full AST scan of the workspace.

        Walks the workspace, parses every recognised source file with
        the appropriate tree-sitter parser, and rebuilds both the
        per-file and the global symbol index.  The result is persisted
        to ``.encre/ast_index.json``.

        Args:
            progress_cb: Optional ``(rel_path, total)`` callback
                invoked after each file is parsed.  Useful for driving
                progress bars in the desktop UI.

        Notes:
            This is CPU-bound.  Callers in the agent hot path should
            off-load it to a background thread (or to
            :class:`IndexManager` which uses a subprocess).
        """
        ws = Path(self.workspace).resolve()
        if not ws.exists():
            self._indexed = True
            return
        self._symbols_by_file.clear()
        self._global_index.clear()
        self._file_mtimes.clear()
        total = 0
        for root, dirs, files in os.walk(ws):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in self._SKIP_DIRS]
            for fname in files:
                if fname.startswith("."):
                    continue
                fpath = Path(root) / fname
                suffix = fpath.suffix.lower()
                if suffix not in LANG_BY_EXT:
                    continue
                rel = str(fpath.relative_to(ws)).replace("\\", "/")
                try:
                    if fpath.stat().st_size > self._MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                try:
                    with open(fpath, "rb") as fh:
                        raw = fh.read(self._MAX_FILE_SIZE)
                    if b"\x00" in raw[:8192]:
                        continue
                    content = raw.decode("utf-8", errors="replace")
                except Exception:
                    continue
                try:
                    mtime = fpath.stat().st_mtime
                except OSError:
                    continue
                self._file_mtimes[rel] = mtime
                symbols = self._parse_file(rel, content, LANG_BY_EXT[suffix])
                self._symbols_by_file[rel] = symbols
                for sym in symbols:
                    self._global_index.setdefault(sym.name, []).append(sym)
                total += 1
                if progress_cb is not None:
                    try:
                        progress_cb(rel, total)
                    except Exception:
                        pass
        self._indexed = True
        self.save()

    def scan_incremental(
        self, progress_cb: Optional[Callable[[str, int], None]] = None
    ) -> None:
        """Incrementally update the AST index.

        Re-parses files whose mtime has increased (or that are new)
        and removes symbols belonging to deleted files.  Falls back to
        a full :meth:`scan` if the index was never built.
        """
        ws = Path(self.workspace).resolve()
        if not ws.exists():
            self._indexed = True
            return
        if not self._indexed:
            self.scan(progress_cb=progress_cb)
            return

        current_files: set[str] = set()
        changed_files: set[str] = set()

        for root, dirs, files in os.walk(ws):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in self._SKIP_DIRS]
            for fname in files:
                if fname.startswith("."):
                    continue
                fpath = Path(root) / fname
                suffix = fpath.suffix.lower()
                if suffix not in LANG_BY_EXT:
                    continue
                rel = str(fpath.relative_to(ws)).replace("\\", "/")
                try:
                    if fpath.stat().st_size > self._MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                current_files.add(rel)
                try:
                    mtime = fpath.stat().st_mtime
                except OSError:
                    continue
                if rel not in self._file_mtimes or self._file_mtimes[rel] < mtime:
                    changed_files.add(rel)
                    self._file_mtimes[rel] = mtime

        # Remove symbols belonging to deleted files (also clean up
        # the global index).
        deleted = set(self._symbols_by_file.keys()) - current_files
        for rel in deleted:
            stale = self._symbols_by_file.pop(rel, [])
            for sym in stale:
                bucket = self._global_index.get(sym.name)
                if bucket:
                    self._global_index[sym.name] = [s for s in bucket if s.file != rel]
                    if not self._global_index[sym.name]:
                        self._global_index.pop(sym.name, None)
            self._file_mtimes.pop(rel, None)

        # Re-parse changed files.
        for rel in changed_files:
            fpath = ws / rel
            try:
                with open(fpath, "rb") as fh:
                    raw = fh.read(self._MAX_FILE_SIZE)
                if b"\x00" in raw[:8192]:
                    continue
                content = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            suffix = fpath.suffix.lower()
            lang = LANG_BY_EXT.get(suffix)
            if lang is None:
                continue
            # Drop old symbols for this file from the global index.
            old = self._symbols_by_file.get(rel, [])
            for sym in old:
                bucket = self._global_index.get(sym.name)
                if bucket:
                    self._global_index[sym.name] = [s for s in bucket if s.file != rel]
                    if not self._global_index[sym.name]:
                        self._global_index.pop(sym.name, None)
            new_symbols = self._parse_file(rel, content, lang)
            self._symbols_by_file[rel] = new_symbols
            for sym in new_symbols:
                self._global_index.setdefault(sym.name, []).append(sym)

        if changed_files or deleted:
            self.save()

    # ── Persistence ──────────────────────────────────────────────────

    def _storage_path(self) -> Path:
        return Path(self.workspace) / ".encre" / "ast_index.json"

    def save(self) -> None:
        storage = self._storage_path()
        storage.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "workspace": self.workspace,
            "file_mtimes": self._file_mtimes,
            "symbols_by_file": {
                f: [s.to_dict() for s in syms]
                for f, syms in self._symbols_by_file.items()
            },
        }
        storage.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> bool:
        """Load the AST index from disk.

        Returns:
            ``True`` if a usable index was found and loaded,
            ``False`` otherwise (in which case the in-memory state is
            empty and a future :meth:`scan` will rebuild it).
        """
        storage = self._storage_path()
        if not storage.exists():
            return False
        try:
            data = json.loads(storage.read_text(encoding="utf-8"))
        except Exception:
            return False
        if data.get("workspace") != self.workspace:
            return False
        self._file_mtimes = {str(k): float(v) for k, v in data.get("file_mtimes", {}).items()}
        self._symbols_by_file.clear()
        self._global_index.clear()
        for f, syms in data.get("symbols_by_file", {}).items():
            loaded: list[Symbol] = []
            for s in syms:
                try:
                    loaded.append(Symbol.from_dict(s))
                except Exception:
                    continue
            self._symbols_by_file[f] = loaded
            for sym in loaded:
                self._global_index.setdefault(sym.name, []).append(sym)
        self._indexed = True
        return True

    # ── Public query API ─────────────────────────────────────────────

    def get_symbol(self, name: str) -> list[Symbol]:
        """Return every symbol named ``name`` across the workspace.

        The result is a list because the same identifier can be
        defined in multiple files (or multiple times in the same
        file).  Callers that need a single match should disambiguate
        by file or by closest position.
        """
        return list(self._global_index.get(name, []))

    def get_outline(self, file: str) -> list[Symbol]:
        """Return the symbol table for ``file`` in source order.

        ``file`` is a workspace-relative path with forward slashes.
        """
        return list(self._symbols_by_file.get(file, []))

    def list_files(self) -> list[str]:
        """Return the list of indexed files (workspace-relative)."""
        return list(self._file_mtimes.keys())

    def find_references(self, name: str) -> list[Reference]:
        """Find textual references to ``name`` across the workspace.

        The reference scan is a fast regex pass that uses word
        boundaries.  It does not consult the parse tree, so it will
        include references inside string literals and comments —
        callers that need exact reference resolution should fall back
        to the LSP ``textDocument/references`` request instead.

        Args:
            name: Identifier to search for.  Must be a valid
                identifier (``[A-Za-z_][A-Za-z0-9_]*``); non-matching
                names yield an empty list.
        """
        if not name or not self._IDENTIFIER_RE.fullmatch(name):
            return []
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        refs: list[Reference] = []
        ws = Path(self.workspace)
        for rel in list(self._file_mtimes.keys()):
            fpath = ws / rel
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            offset = 0
            text_left = content
            while text_left:
                m = pattern.search(text_left)
                if m is None:
                    break
                abs_pos = offset + m.start()
                line = content.count("\n", 0, abs_pos)
                last_nl = content.rfind("\n", 0, abs_pos) + 1
                col = abs_pos - last_nl
                refs.append(
                    Reference(
                        file=rel,
                        line=line,
                        col=col,
                        name=name,
                        kind="reference",
                    )
                )
                advance = m.end()
                offset += advance
                text_left = text_left[advance:]
        return refs

    def goto_definition(self, file: str, line: int, col: int) -> Symbol | None:
        """Resolve the identifier at ``(file, line, col)`` to its definition.

        Args:
            file: Workspace-relative path.
            line: 0-based line.
            col: 0-based byte column.

        Returns:
            The matching :class:`Symbol` if found, otherwise ``None``.
        """
        ws = Path(self.workspace)
        fpath = ws / file
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError:
            return None
        if not content:
            return None
        # Locate the line in the file.
        line_start = 0
        current_line = 0
        for ch in content:
            if current_line == line:
                break
            if ch == "\n":
                current_line += 1
                line_start += 1
        else:
            return None
        if current_line != line:
            return None
        line_end = content.find("\n", line_start)
        if line_end == -1:
            line_end = len(content)
        line_text = content[line_start:line_end]
        # Find the identifier boundary at ``col``.
        if col < 0 or col >= len(line_text):
            return None
        i = col
        while i > 0 and (line_text[i - 1].isalnum() or line_text[i - 1] == "_"):
            i -= 1
        j = col
        while j < len(line_text) and (line_text[j].isalnum() or line_text[j] == "_"):
            j += 1
        if i >= j:
            return None
        name = line_text[i:j]
        candidates = self.get_symbol(name)
        if not candidates:
            return None
        # Prefer same-file definitions; among those, the one whose
        # ``start_line`` is closest to (but not after) the cursor.
        same_file = [s for s in candidates if s.file == file]
        if same_file:
            same_file.sort(key=lambda s: s.start_line)
            best = same_file[0]
            for s in same_file:
                if s.start_line <= line:
                    best = s
                else:
                    break
            return best
        return candidates[0]

    def find_relevant(self, name: str, limit: int = 10) -> list[Symbol]:
        """Return the symbols whose name contains ``name`` (case-sensitive).

        Cheap lookup helper used by the agent when the model asks
        "where is X defined?" and only has a partial identifier.
        """
        if not name:
            return []
        out: list[Symbol] = []
        for sym_name, syms in self._global_index.items():
            if name in sym_name:
                out.extend(syms)
                if len(out) >= limit:
                    return out[:limit]
        return out
