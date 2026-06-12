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

from __future__ import annotations
"""
Multi-language code indexer with BM25 search and dependency analysis.

This module implements :class:`EncreCodeIndex`, a workspace-level code
indexing engine that supports:

- **Full scan**: Walks the entire workspace, parsing every recognised source
  file into a :class:`ModuleInfo` record with imports, exports, and language.
- **Incremental scan**: Re-parses only files whose modification timestamps
  have changed since the last scan, preserving the existing index for
  unchanged files and removing deleted files.
- **Live file watcher**: Integrates with the ``watchfiles`` library to
  detect filesystem changes in real time and trigger incremental re-indexing.
- **Multi-language parsing**: Uses language-appropriate techniques for each
  supported language:
  - Python: ``ast`` module for structural import/export extraction
  - JavaScript/TypeScript: regex-based import and export matching
  - Rust: regex-based ``use`` statement and ``pub fn/struct/enum/trait`` extraction
  - Go: regex-based import block and exported function/type extraction
  - Others: generic fallback for include/import/require patterns
- **Dependency graph**: Resolves import statements to module paths, building
  both forward and reverse dependency graphs for impact analysis.
- **BM25 full-text search**: Okapi BM25 ranking with code-specific tokenisation
  and a +2.0 name-match bonus for module paths matching query tokens.
- **Context builder**: Generates a formatted string containing source code,
  imports, dependents, and exports for a given file — useful for LLM context.

Supported file extensions:
    Python (``.py``, ``.pyi``, ``.pyx``),
    JavaScript/TypeScript (``.js``, ``.jsx``, ``.ts``, ``.tsx``, ``.mjs``, ``.cjs``),
    Rust (``.rs``),
    Go (``.go``),
    Others (``.java``, ``.rb``, ``.php``, ``.c``, ``.cpp``, ``.h``, ``.hpp``,
    ``.cc``, ``.cxx``, ``.swift``, ``.kt``, ``.scala``).

Design notes:
    The index is entirely in-memory and rebuilt on each ``scan()`` call.
    For large workspaces (>10,000 files), consider using the incremental
    scan mode or the file watcher to avoid repeated full re-indexing.
"""

import ast
import asyncio
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    import pathspec
except ImportError:
    pathspec = None  # type: ignore[assignment]

from encre.native import Bm25Index

logger = logging.getLogger("encre.codebase.indexer")


@dataclass
class ModuleInfo:
    """Metadata for a single source code module in the index.

    Stores everything extracted during parsing: the file path relative to
    the workspace root, the module name, lists of imports and exports,
    the programming language, and line count.  The ``imported_by`` field is
    populated during the dependency graph build phase.

    Attributes:
        path: File path relative to the workspace root (Unix-style separators).
        name: Module name (typically the relative path or file stem).
        imports: List of module names or paths that this module imports.
        imported_by: List of module paths that import this module (populated
            during :meth:`EncreCodeIndex._build_dependencies`).
        exports: List of public symbols exported by this module (functions,
            classes, constants, types, etc.).
        language: Programming language identifier (e.g., ``"python"``,
            ``"rust"``, ``"typescript"``).
        loc: Lines of code (total line count in the source file).
    """

    path: str
    name: str
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    language: str = ""
    loc: int = 0


class EncreCodeIndex:
    """In-memory searchable index of source code files in a workspace.

    The index builds a structured representation of every source file in
    the workspace, enabling fast full-text search (BM25), dependency
    queries, and context extraction for downstream use (e.g., LLM prompts).

    The class maintains:
    - A dictionary of :class:`ModuleInfo` records keyed by relative path.
    - Forward and reverse dependency graphs for import chain analysis.
    - A BM25-weighted inverted index for relevance-ranked code search.
    - File modification timestamps for incremental re-indexing.
    - An optional ``asyncio.Task`` for live file watching via ``watchfiles``.

    Args:
        workspace: Absolute or relative path to the workspace root directory
            to index.
    """

    # ── Language extension sets ──────────────────────────────────────

    _PY_EXTS: set[str] = {".py", ".pyi", ".pyx"}
    """Python file extensions parsed via the ``ast`` module."""

    _JS_EXTS: set[str] = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
    """JavaScript/TypeScript file extensions parsed via regex."""

    _RS_EXTS: set[str] = {".rs"}
    """Rust file extensions parsed via regex."""

    _GO_EXTS: set[str] = {".go"}
    """Go file extensions parsed via regex."""

    _KNOWN_EXTS: set[str] = {
        ".py", ".pyi", ".pyx",
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        ".rs", ".go",
        ".java", ".rb", ".php",
        ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx",
        ".swift", ".kt", ".scala",
        ".sh", ".bash", ".zsh",
        ".sql",
        ".html", ".htm", ".css", ".scss", ".sass", ".less",
        ".json", ".yaml", ".yml", ".toml",
        ".md", ".rst",
    }
    """Recognised source file extensions (still used for language-specific
    parsing, but no longer gates whether a file is indexed)."""

    _MAX_FILE_SIZE: int = 2 * 1024 * 1024  # 2 MB — skip files larger than this

    _SKIP_DIRS: frozenset[str] = frozenset({
        "node_modules", "__pycache__", "target", "build", "dist",
        ".git", "venv", ".venv", "env", ".tox", ".eggs",
        ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".svn", ".hg",
    })
    """Directory names to skip entirely during scanning (names only, not paths)."""

    def __init__(self, workspace: str) -> None:
        """Initialise a new code index for the given workspace.

        Args:
            workspace: Path to the workspace root directory.  The index
                will recursively discover and parse all source files under
                this directory.
        """
        self.workspace: str = workspace
        self._modules: dict[str, ModuleInfo] = {}
        self._depgraph: dict[str, set[str]] = {}
        self._reverse_depgraph: dict[str, set[str]] = {}
        self._bm25_index: Bm25Index = Bm25Index()
        self._indexed: bool = False
        self._file_mtimes: dict[str, float] = {}
        self._content_cache: dict[str, str] = {}  # rel_path → content, cleared after _build_inverted_index
        self._watcher_task: Optional[asyncio.Task] = None
        self._has_git: bool = False
        self._has_gitignore: bool = False
        self._gitignored_count: int = 0
        self._gitignore_specs: list[tuple[str, "PathSpec"]] = []
        self._need_reindex: bool = False
        # Tracks whether dependency graphs and the BM25 inverted index
        # have been derived from the current module set.  Reset to False
        # whenever the module set changes; rebuilt lazily by
        # ``_ensure_query_ready`` on the first query.  This is what keeps
        # ``EncreCodeIndex(workspace)`` cheap enough to construct from
        # the agent's main event loop.
        self._query_ready: bool = False
        self.load()

    # ── Gitignore helpers ────────────────────────────────────────────

    def _load_gitignore(self, ws_path: str) -> None:
        """Load all .gitignore files under *ws_path* into ``_gitignore_specs``.

        Reads the root ``.gitignore`` first, then discovers any nested
        ``.gitignore`` files during the walk (loaded lazily in ``scan()``).
        """
        self._gitignore_specs.clear()
        self._has_gitignore = False
        if pathspec is None:
            return

        root_gitignore = os.path.join(ws_path, ".gitignore")
        if os.path.isfile(root_gitignore):
            self._has_gitignore = True
            try:
                with open(root_gitignore, "r", encoding="utf-8", errors="replace") as f:
                    spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
                self._gitignore_specs.append(("", spec))
            except Exception:
                pass

    def _load_subdir_gitignore(self, dir_path: str, ws_path: str) -> None:
        """Load a .gitignore found in *dir_path* (a subdirectory of *ws_path*)."""
        if pathspec is None or self._gitignore_specs is None:
            return
        rel_dir = os.path.relpath(dir_path, ws_path).replace("\\", "/")
        if rel_dir == ".":
            return  # root already handled by _load_gitignore
        # Avoid duplicates (same subdirectory reloaded)
        if any(d == rel_dir for d, _ in self._gitignore_specs):
            return
        try:
            with open(os.path.join(dir_path, ".gitignore"), "r", encoding="utf-8", errors="replace") as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
            self._gitignore_specs.append((rel_dir, spec))
            self._has_gitignore = True
        except Exception:
            pass

    def _is_gitignored(self, rel_path: str) -> bool:
        """Check if *rel_path* (workspace-relative, Unix separators) matches
        any loaded ``.gitignore`` pattern."""
        if pathspec is None:
            return False
        for gitignore_dir, spec in self._gitignore_specs:
            if gitignore_dir == "":
                rel_to_spec = rel_path
            elif rel_path.startswith(gitignore_dir + "/"):
                rel_to_spec = rel_path[len(gitignore_dir) + 1:]
            else:
                continue
            if spec.match_file(rel_to_spec):
                return True
        return False

    # ── Scanning ─────────────────────────────────────────────────────

    def scan(self, progress_cb: Optional[callable] = None) -> None:
        """Perform a full scan of the workspace, rebuilding the entire index.

        Walks the entire workspace directory tree, skipping common build
        artifact and cache directories (``node_modules``, ``__pycache__``,
        ``target``, ``build``, etc.) and dot-directories.  Files matching
        ``.gitignore`` patterns are skipped from content indexing (their
        existence is recorded in ``_gitignored_count``).  ALL non-ignored
        files are indexed regardless of extension — binary files (null bytes
        in the first 8 KB) and files larger than 2 MB are skipped.  Each
        recognised source file is parsed into a :class:`ModuleInfo` record,
        then dependency graphs and the BM25 inverted index are rebuilt.

        This is a blocking, CPU-bound operation.  For large workspaces,
        consider calling this in a thread pool executor.

        Args:
            progress_cb: Optional callback ``(rel_path, total_so_far)``
                invoked after each file is parsed, for progress reporting.
        """
        ws = Path(self.workspace).resolve()
        if not ws.exists():
            self._indexed = True
            return
        self._modules.clear()
        self._depgraph.clear()
        self._reverse_depgraph.clear()
        self._file_mtimes.clear()
        self._gitignored_count = 0
        ws_str = str(ws)
        self._has_git = os.path.isdir(os.path.join(ws_str, ".git"))
        self._load_gitignore(ws_str)
        total = 0
        for root, dirs, files in os.walk(ws_str):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in self._SKIP_DIRS]
            # Load any .gitignore in this directory
            if pathspec is not None and ".gitignore" in files:
                self._load_subdir_gitignore(root, ws_str)
            for fname in files:
                if fname == ".gitignore":
                    continue
                fpath = Path(root) / fname
                # Skip hidden files (.*)
                if fname.startswith("."):
                    continue
                rel = str(fpath.relative_to(ws)).replace("\\", "/")
                # Skip gitignored files — record count only
                if self._is_gitignored(rel):
                    self._gitignored_count += 1
                    continue
                # Skip large files
                try:
                    if fpath.stat().st_size > self._MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                # Binary sniff + read whole file (up to _MAX_FILE_SIZE)
                suffix = fpath.suffix.lower()
                try:
                    with open(fpath, "rb") as fh:
                        raw = fh.read(self._MAX_FILE_SIZE)
                    # Skip binary files (null byte in first 8 KB)
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
                self._content_cache[rel] = content
                if total % 50 == 0:
                    logger.info("[codebase] scan file #%d: %s", total + 1, rel)
                mod = self._parse_file(rel, content, suffix)
                self._modules[rel] = mod
                total += 1
                if progress_cb:
                    progress_cb(rel, total)
        if progress_cb:
            progress_cb("_build_dependencies", total)
        self._build_dependencies()
        if progress_cb:
            progress_cb("_build_inverted_index", total)
        self._build_inverted_index()
        self._content_cache.clear()
        # Scan rebuilds both the dep graph and the BM25 index in-place,
        # so the lazy-build cache is up to date.
        self._query_ready = True
        self._indexed = True
        if progress_cb:
            progress_cb("_save", total)
        self.save()
        if progress_cb:
            progress_cb("_done", total)

    def scan_incremental(self, progress_cb: Optional[callable] = None) -> None:
        """Incrementally update the index: re-parse changed/new files, remove deleted files.

        Compares the current filesystem state against the cached modification
        timestamps (``_file_mtimes``).  Only files whose mtime has increased
        (or are newly encountered) are re-parsed.  Files that no longer exist
        on disk are removed from the index, including cleanup of their
        dependency graph entries and ``imported_by`` references.

        If the index has never been built (``_indexed`` is False), falls back
        to a full :meth:`scan`.  Dependency graphs and the inverted index are
        only rebuilt if changes were actually detected.

        Args:
            progress_cb: Optional callback ``(rel_path, total_so_far)``
                for progress reporting during the initial full scan
                (only used if falling back to :meth:`scan`).
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

        ws_str = str(ws)
        self._has_git = os.path.isdir(os.path.join(ws_str, ".git"))
        self._load_gitignore(ws_str)

        for root, dirs, files in os.walk(ws_str):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in self._SKIP_DIRS]
            if pathspec is not None and ".gitignore" in files:
                self._load_subdir_gitignore(root, ws_str)
            files_to_check = [f for f in files if f != ".gitignore"]
            for fname in files_to_check:
                fpath = Path(root) / fname
                if fname.startswith("."):
                    continue
                rel = str(fpath.relative_to(ws)).replace("\\", "/")
                # Skip gitignored files entirely (they will appear as deleted
                # and be cleaned up from the index below).
                if self._is_gitignored(rel):
                    continue
                try:
                    if fpath.stat().st_size > self._MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                current_files.add(rel)
                try:
                    mtime = fpath.stat().st_mtime
                except Exception:
                    continue
                if rel not in self._file_mtimes or self._file_mtimes[rel] < mtime:
                    changed_files.add(rel)
                    self._file_mtimes[rel] = mtime
                else:
                    self._file_mtimes[rel] = mtime

        # Remove deleted files from the index, including dependency references.
        deleted_files = set(self._modules.keys()) - current_files
        for rel in deleted_files:
            self._modules.pop(rel, None)
            self._file_mtimes.pop(rel, None)
            self._depgraph.pop(rel, None)
            self._reverse_depgraph.pop(rel, None)
            for mod in self._modules.values():
                if rel in mod.imported_by:
                    mod.imported_by.remove(rel)

        # Re-parse changed and new files.
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
            self._content_cache[rel] = content
            mod = self._parse_file(rel, content, suffix=fpath.suffix.lower())
            self._modules[rel] = mod

        # Rebuild only if there were actual changes.
        if changed_files or deleted_files:
            self._build_dependencies()
            self._build_inverted_index()
            self._content_cache.clear()
            self.save()
            # Module set changed — derived structures are fresh, but
            # flag for the lazy path so any future mutation invalidates
            # the cache again.
            self._query_ready = True

    # ── File watcher ─────────────────────────────────────────────────

    async def watch(self) -> Optional[asyncio.Task]:
        """Start watching the workspace for file changes using ``watchfiles``.

        Launches an ``asyncio.Task`` that monitors the workspace directory
        for filesystem events.  On each batch of changes, it filters out
        non-code files and changes under ignored directories, then calls
        :meth:`scan_incremental` to update the index.

        Returns:
            The ``asyncio.Task`` handle for cancellation, or ``None`` if
            ``watchfiles`` is not installed.
        """
        try:
            import watchfiles
        except ImportError:
            return None

        if self._watcher_task is not None and not self._watcher_task.done():
            return self._watcher_task

        ws = Path(self.workspace).resolve()
        if not ws.exists():
            return None

        _WATCH_IGNORE_DIRS = (
            ".git", "node_modules", "__pycache__", "target", "build", "dist",
            "venv", ".venv", "env", ".tox", ".eggs", ".mypy_cache",
            ".pytest_cache", ".ruff_cache",
        )

        async def _watcher_loop() -> None:
            try:
                async for changes in watchfiles.awatch(str(ws)):
                    relevant_changes = False
                    for change_type, changed_path in changes:
                        rel_path = str(Path(changed_path).relative_to(ws))
                        parts = rel_path.replace("\\", "/").split("/")
                        if any(p in _WATCH_IGNORE_DIRS for p in parts):
                            continue
                        if Path(changed_path).suffix.lower() in self._KNOWN_EXTS:
                            relevant_changes = True
                            break
                    if relevant_changes:
                        self.scan_incremental()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self._watcher_task = asyncio.create_task(_watcher_loop())
        return self._watcher_task

    def stop_watch(self) -> None:
        """Stop the file watcher task if one is running.

        Cancels the running ``asyncio.Task`` and sets the watcher reference
        to ``None``.  Safe to call even if no watcher is active.
        """
        if self._watcher_task is not None and not self._watcher_task.done():
            self._watcher_task.cancel()
            self._watcher_task = None

    # ── Persistence ──────────────────────────────────────────────────

    def _get_storage_path(self) -> Path:
        return Path(self.workspace) / ".encre" / "code_index.json"

    def save(self) -> None:
        storage = self._get_storage_path()
        storage.parent.mkdir(parents=True, exist_ok=True)

        modules_data = {}
        for path, mod in self._modules.items():
            modules_data[path] = asdict(mod)

        data = {
            "workspace": self.workspace,
            "modules": modules_data,
            "file_mtimes": self._file_mtimes,
            "has_git": self._has_git,
            "has_gitignore": self._has_gitignore,
            "gitignored_count": self._gitignored_count,
        }

        storage.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def load(self) -> bool:
        """Load index metadata from disk without rebuilding derived structures.

        Reads ``.encre/code_index.json`` and deserialises the module list
        and file timestamps.  The dependency graphs and the BM25 inverted
        index are **not** rebuilt here — they are built lazily on the
        first query through :meth:`_ensure_query_ready`.

        This is the fix for the "workspace indexing blocks the main
        agent loop" bug: the previous implementation synchronously
        re-derived the dep graph and rebuilt the BM25 index inside
        ``__init__`` (via ``load()``), which stalled the event loop on
        the GIL.  External code (typically :class:`IndexManager`) should
        own the lifecycle of heavy indexing work; this method is the
        cheap, side-effect-free path that lets the agent inspect a
        previously-built index in milliseconds.
        """
        storage = self._get_storage_path()
        if not storage.exists():
            return False

        try:
            data = json.loads(storage.read_text(encoding="utf-8"))
            if data.get("workspace") != self.workspace:
                return False

            self._modules.clear()
            for path, mod_data in data.get("modules", {}).items():
                self._modules[path] = ModuleInfo(**mod_data)

            self._file_mtimes = data.get("file_mtimes", {})
            self._has_git = data.get("has_git", False)
            self._has_gitignore = data.get("has_gitignore", False)
            self._gitignored_count = data.get("gitignored_count", 0)

            # Metadata is loaded; mark the index as available.  The
            # BM25 inverted index and dep graph are intentionally
            # deferred until the first query — see _ensure_query_ready.
            self._indexed = True
            self._query_ready = False
            return True
        except Exception:
            self._modules.clear()
            self._file_mtimes.clear()
            return False

    def _ensure_query_ready(self) -> None:
        """Lazily build derived structures needed for query operations.

        Builds the forward and reverse dependency graphs and the BM25
        inverted index on the calling thread.  This is invoked from
        public query methods (``find_relevant``, ``build_dependency_graph``,
        ``get_importers``, ``build_context``) so the cost is paid only
        when a caller actually needs the data, and exactly once per
        index lifetime unless :meth:`scan` / :meth:`scan_incremental`
        invalidates :attr:`_query_ready`.

        The caller is responsible for running this off the main event
        loop when the workspace is large (e.g. via
        :func:`asyncio.to_thread`).  In Encre's normal flow the
        :class:`IndexManager` is the owner of the index and exposes a
        fully-prepared instance to the agent loop, so this method is
        a cheap no-op on the hot path.
        """
        if getattr(self, "_query_ready", False):
            return
        self._build_dependencies()

        # BM25 needs file content; load from disk on demand.
        self._bm25_index.clear()
        files: list[tuple[str, str]] = []
        for mod in list(self._modules.values()):
            try:
                full_path = os.path.join(self.workspace, mod.path)
                with open(full_path, "rb") as fh:
                    raw = fh.read(self._MAX_FILE_SIZE)
                if b"\x00" in raw[:8192]:
                    continue
                text = raw.decode("utf-8", errors="replace").lower()
            except Exception:
                continue
            files.append((mod.path, text))
        if files:
            self._bm25_index.build(files)
        self._query_ready = True

    # ── File parsing ─────────────────────────────────────────────────

    def _parse_file(self, rel_path: str, content: str, suffix: str) -> ModuleInfo:
        """Route file parsing to the appropriate language-specific parser.

        Dispatches based on file extension to one of the specialised parsers:
        :meth:`_parse_python`, :meth:`_parse_javascript`, :meth:`_parse_rust`,
        :meth:`_parse_go`, or :meth:`_parse_generic`.

        Args:
            rel_path: File path relative to the workspace root.
            content: UTF-8 decoded file contents.
            suffix: Lowercased file extension (e.g., ``".py"``, ``".rs"``).

        Returns:
            A :class:`ModuleInfo` instance with parsed metadata.
        """
        if suffix in self._PY_EXTS:
            return self._parse_python(rel_path, content)
        elif suffix in self._JS_EXTS:
            return self._parse_javascript(rel_path, content, suffix)
        elif suffix in self._RS_EXTS:
            return self._parse_rust(rel_path, content)
        elif suffix in self._GO_EXTS:
            return self._parse_go(rel_path, content)
        else:
            return self._parse_generic(rel_path, content, suffix)

    def _parse_python(self, rel_path: str, content: str) -> ModuleInfo:
        """Parse a Python source file using the ``ast`` module.

        Extracts:
        - Imports: ``import X`` and ``from X import Y`` statements
        - Exports: top-level functions, async functions, classes (non-underscore),
          and module-level uppercase constants (``UPPER_CASE = ...``)

        Args:
            rel_path: Relative file path.
            content: File content as string.

        Returns:
            A :class:`ModuleInfo` with language set to ``"python"``.
        """
        info = ModuleInfo(path=rel_path, name=rel_path, language="python", loc=len(content.splitlines()))
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return info
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    info.imports.append(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name and not node.name.startswith("_"):
                    info.exports.append(node.name)
            elif isinstance(node, ast.ClassDef):
                if node.name and not node.name.startswith("_"):
                    info.exports.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id and not target.id.startswith("_") and target.id.isupper():
                        info.exports.append(target.id)
        return info

    def _parse_javascript(self, rel_path: str, content: str, suffix: str) -> ModuleInfo:
        """Parse a JavaScript/TypeScript source file using regex.

        Extracts:
        - Imports: ``import ... from "module"``, ``import "module"``,
          ``require("module")`` — only for non-relative module names
        - Exports: ``export function/class/const/let/var/interface/type/enum Name``
          and ``export { name1, name2 }`` (handles ``as`` aliases)

        Args:
            rel_path: Relative file path.
            content: File content as string.
            suffix: File extension (``".js"``, ``".ts"``, ``".tsx"``, etc.).

        Returns:
            A :class:`ModuleInfo` with language set to ``"typescript"`` or
            ``"javascript"``.
        """
        lang = "typescript" if suffix in (".ts", ".tsx") else "javascript"
        info = ModuleInfo(path=rel_path, name=rel_path, language=lang, loc=len(content.splitlines()))
        import_re = re.compile(
            r'''(?:import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s*,?\s*)*from\s+['"]([^'"]+)['"])|'''
            r'''(?:import\s+['"]([^'"]+)['"])|'''
            r'''(?:require\s*\(\s*['"]([^'"]+)['"]\s*\))'''
        )
        for m in import_re.finditer(content):
            mod_name = m.group(1) or m.group(2) or m.group(3)
            if mod_name and not mod_name.startswith("."):
                info.imports.append(mod_name)
        export_re = re.compile(
            r'''(?:export\s+(?:default\s+)?(?:function|class|const|let|var|interface|type|enum)\s+(\w+))|'''
            r'''(?:export\s*\{\s*([^}]*)\s*\})'''
        )
        for m in export_re.finditer(content):
            name = m.group(1)
            if name:
                info.exports.append(name)
            elif m.group(2):
                for part in m.group(2).split(","):
                    part = part.strip()
                    if part:
                        info.exports.append(part.split(" as ")[-1].strip())
        return info

    def _parse_rust(self, rel_path: str, content: str) -> ModuleInfo:
        """Parse a Rust source file using regex.

        Extracts:
        - Imports: ``use crate::module;``, ``use std::collections::HashMap;``
        - Exports: ``pub fn``, ``pub async fn``, ``pub struct``,
          ``pub enum``, ``pub trait``

        Args:
            rel_path: Relative file path.
            content: File content as string.

        Returns:
            A :class:`ModuleInfo` with language set to ``"rust"``.
        """
        info = ModuleInfo(path=rel_path, name=rel_path, language="rust", loc=len(content.splitlines()))
        use_re = re.compile(r'use\s+((?:\w+::)*\w+)\s*;')
        for m in use_re.finditer(content):
            info.imports.append(m.group(1))
        pub_re = re.compile(r'pub\s+(?:async\s+)?fn\s+(\w+)')
        pub_struct_re = re.compile(r'pub\s+struct\s+(\w+)')
        pub_enum_re = re.compile(r'pub\s+enum\s+(\w+)')
        pub_trait_re = re.compile(r'pub\s+trait\s+(\w+)')
        for m in pub_re.finditer(content):
            info.exports.append(m.group(1))
        for m in pub_struct_re.finditer(content):
            info.exports.append(m.group(1))
        for m in pub_enum_re.finditer(content):
            info.exports.append(m.group(1))
        for m in pub_trait_re.finditer(content):
            info.exports.append(m.group(1))
        return info

    def _parse_go(self, rel_path: str, content: str) -> ModuleInfo:
        """Parse a Go source file using regex.

        Extracts:
        - Imports: both multi-line ``import ( "pkg1" "pkg2" )`` blocks and
          single-line ``import "pkg"`` statements
        - Exports: exported functions (uppercase first letter) and
          struct type definitions

        Args:
            rel_path: Relative file path.
            content: File content as string.

        Returns:
            A :class:`ModuleInfo` with language set to ``"go"``.
        """
        info = ModuleInfo(path=rel_path, name=rel_path, language="go", loc=len(content.splitlines()))
        import_block_re = re.compile(r'import\s*\(\s*((?:[^)]*?\"[^\"]+\"[^)]*?)*)\s*\)', re.DOTALL)
        for block in import_block_re.finditer(content):
            for line in block.group(1).split("\n"):
                m = re.search(r'"([^"]+)"', line)
                if m:
                    info.imports.append(m.group(1))
        single_import_re = re.compile(r'import\s+"([^"]+)"')
        for m in single_import_re.finditer(content):
            info.imports.append(m.group(1))
        func_re = re.compile(r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)')
        for m in func_re.finditer(content):
            name = m.group(1)
            if name and name[0].isupper():
                info.exports.append(name)
        type_re = re.compile(r'type\s+(\w+)\s+struct')
        for m in type_re.finditer(content):
            info.exports.append(m.group(1))
        return info

    def _parse_generic(self, rel_path: str, content: str, suffix: str) -> ModuleInfo:
        """Parse a source file in an unsupported language using a generic regex fallback.

        Attempts to extract import-like statements using a broad pattern
        matching ``import``, ``from ... import``, ``#include``, and
        ``require()`` constructs.  This provides best-effort import tracking
        for languages without a dedicated parser.

        Args:
            rel_path: Relative file path.
            content: File content as string.
            suffix: File extension used as the language identifier.

        Returns:
            A :class:`ModuleInfo` with language set to the file extension
            (without the dot).
        """
        info = ModuleInfo(path=rel_path, name=rel_path, language=suffix.lstrip("."), loc=len(content.splitlines()))
        import_re = re.compile(
            r'''(?:import\s+[\w.]+)|'''
            r'''(?:from\s+\S+\s+import\s+\S+)|'''
            r'''(?:#include\s+[<\"][^>\"]+[>\"])|'''
            r'''(?:require\s*\(\s*['"][^'"]+['"]\s*\))'''
        )
        for m in import_re.finditer(content):
            info.imports.append(m.group(0).strip())
        return info

    # ── Dependency graph ─────────────────────────────────────────────

    def _build_dependencies(self) -> None:
        """Build forward and reverse dependency graphs from module import data.

        Resolves each module's import list to actual module paths in the
        index by matching the import root package name against known module
        basenames.  Populates:
        - ``_depgraph[mod_path]``: set of module paths that *mod_path* imports
        - ``_reverse_depgraph[mod_path]``: set of module paths that import *mod_path*
        - ``ModuleInfo.imported_by`` on each module: list of importing module paths

        Resolution is heuristic: given an import ``foo.bar.baz``, only the
        root ``foo`` is used for matching — this works for flat project
        structures where module basenames are unique.
        """
        self._depgraph.clear()
        self._reverse_depgraph.clear()
        modules_by_name: dict[str, ModuleInfo] = {}
        for mod in self._modules.values():
            base = os.path.splitext(os.path.basename(mod.path))[0]
            modules_by_name[base] = mod
        for mod in self._modules.values():
            deps: set[str] = set()
            for imp in mod.imports:
                parts = imp.split(".")
                candidate = parts[0]
                if candidate in modules_by_name:
                    resolved = modules_by_name[candidate].path
                    deps.add(resolved)
            self._depgraph[mod.path] = deps
        for mod_path, deps in self._depgraph.items():
            for dep in deps:
                if dep not in self._reverse_depgraph:
                    self._reverse_depgraph[dep] = set()
                self._reverse_depgraph[dep].add(mod_path)
        for mod_path, mod in self._modules.items():
            for dep_path in self._depgraph.get(mod_path, set()):
                dep_mod = self._modules.get(dep_path)
                if dep_mod:
                    dep_mod.imported_by.append(mod_path)

    # ── Inverted index (Rust-native BM25) ─────────────────────────────

    def _build_inverted_index(self) -> None:
        """Build a BM25-weighted inverted index via the Rust native layer.

        For each module, the file content is sent to Rust for tokenization
        and BM25 indexing. This is **much faster** than pure Python for
        large codebases.
        """
        self._bm25_index.clear()
        files: list[tuple[str, str]] = []
        for mod in list(self._modules.values()):
            try:
                text = self._content_cache.get(mod.path)
                if text is None:
                    full_path = os.path.join(self.workspace, mod.path)
                    with open(full_path, "rb") as fh:
                        raw = fh.read(self._MAX_FILE_SIZE)
                    if b"\x00" in raw[:8192]:
                        continue
                    text = raw.decode("utf-8", errors="replace")
                text = text.lower()
            except Exception:
                continue
            files.append((mod.path, text))
        if files:
            self._bm25_index.build(files)

    # ── Public query API ─────────────────────────────────────────────

    def build_dependency_graph(self) -> dict[str, set[str]]:
        """Return the forward dependency graph.

        Each entry maps a module path to the set of module paths it imports.
        Triggers a full scan if the index has not been built yet.

        Returns:
            A dictionary of ``{module_path: {dependency_path, ...}}``.
        """
        if not self._indexed:
            self.scan()
        self._ensure_query_ready()
        return dict(self._depgraph)

    def get_importers(self, file_path: str) -> list[str]:
        """Return all modules that import the given file.

        Uses the reverse dependency graph to find dependents.  Triggers a
        full scan if the index has not been built yet.

        Args:
            file_path: Relative file path to query (workspace-relative).

        Returns:
            A list of module paths that import the given file.
        """
        if not self._indexed:
            self.scan()
        self._ensure_query_ready()
        return list(self._reverse_depgraph.get(file_path, set()))

    def find_relevant(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Search the codebase using BM25 ranking and return top results.

        Delegates to the Rust-native BM25 indexer for performance.  The
        BM25 inverted index is built lazily on the first call when the
        index was loaded from cache, so this method may briefly perform
        a small amount of CPU work the very first time it is invoked.

        Args:
            query: Free-text search query (e.g., ``"database connection pool"``).
            limit: Maximum number of results to return.  Defaults to 10.

        Returns:
            A list of ``(module_path, score)`` tuples sorted by descending
            relevance score.  Empty list if the query produces no tokens.
        """
        if not self._indexed:
            self.scan()
        self._ensure_query_ready()
        if len(self._bm25_index) == 0:
            return []
        return self._bm25_index.search(query, limit)

    def build_context(self, file_path: str) -> str:
        """Build a formatted context string for a given file.

        The context includes the file header (path, language, LOC), the
        full source code, and up to 30 entries each of imports, importers,
        and exports.  This is designed to produce a compact prompt context
        for LLM consumption.

        Args:
            file_path: Relative file path to build context for.

        Returns:
            A formatted string ready for inclusion in LLM prompts, or an
            empty string if the file is not in the index.
        """
        if not self._indexed:
            self.scan()
        self._ensure_query_ready()
        mod = self._modules.get(file_path)
        if mod is None:
            return ""
        parts: list[str] = []
        parts.append(f"[{file_path}] ({mod.language}, {mod.loc} loc)")
        try:
            full_path = os.path.join(self.workspace, file_path)
            parts.append(Path(full_path).read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
        if mod.imports:
            parts.append("Imports: " + ", ".join(mod.imports[:30]))
        if mod.imported_by:
            parts.append("Imported by: " + ", ".join(mod.imported_by[:30]))
        if mod.exports:
            parts.append("Exports: " + ", ".join(mod.exports[:30]))
        return "\n\n".join(parts)

    def get_module_info(self, file_path: str) -> ModuleInfo | None:
        """Retrieve the :class:`ModuleInfo` for a given file path.

        Triggers a full scan if the index has not been built yet.

        Args:
            file_path:<think> Relative file path to look up.

        Returns:
            The :class:`ModuleInfo` record, or ``None`` if the file is not
            in the index.
        """
        if not self._indexed:
            self.scan()
        return self._modules.get(file_path)

    def list_all_modules(self) -> list[ModuleInfo]:
        """Return all indexed modules.

        Triggers a full scan if the index has not been built yet.

        Returns:
            A list of all :class:`ModuleInfo` records in the index.
        """
        if not self._indexed:
            self.scan()
        return list(self._modules.values())

    def search_by_name(self, name: str) -> list[ModuleInfo]:
        """Search modules by file path or export name substring match.

        Performs a case-insensitive substring search against module paths,
        module names, and export names.  Useful for quick lookup when you
        know part of a symbol or file name.

        Args:
            name: Substring to search for (e.g., ``"database"``, ``"connect"``).

        Returns:
            A list of :class:`ModuleInfo` records whose path, name, or
            exports contain the search string.
        """
        if not self._indexed:
            self.scan()
        results: list[ModuleInfo] = []
        name_lower = name.lower()
        for mod in self._modules.values():
            if name_lower in mod.path.lower() or name_lower in mod.name.lower():
                results.append(mod)
            else:
                for exp in mod.exports:
                    if name_lower in exp.lower():
                        results.append(mod)
                        break
        return results