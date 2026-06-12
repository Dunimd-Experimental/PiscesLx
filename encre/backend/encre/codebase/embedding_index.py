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
Semantic code embedding index.

This module implements :class:`EncreEmbeddingIndex`, a workspace-level
**vector** index that complements the BM25 keyword index in
:mod:`encre.codebase.indexer` and the AST symbol table in
:mod:`encre.codebase.ast_index`.  Together, the three indices give
Encre Codex-class code search: keyword ranking for exact matches,
AST lookups for structural queries, and dense vectors for
"find me code that does X" semantic queries.

Design
------
- **Slice strategy** — the indexer reads the AST index produced by
  :class:`encre.codebase.ast_index.EncreASTIndex` and slices each file
  at function / class / method / type boundaries.  Each slice is
  embedded as a single vector, and the slice metadata (file path,
  start line, end line, symbol name, kind) is stored alongside the
  vector so search results are actionable.
- **Embedding backend** — a pluggable callable
  ``embedding_fn: list[str] -> list[list[float]]``.  The default
  implementation is :class:`OpenAICompatibleEmbedding` which POSTs to
  any OpenAI-compatible ``/v1/embeddings`` endpoint (OpenAI, Azure,
  vLLM, Ollama, OpenRouter, etc.).  Callers can swap in a local
  model by passing a different callable.
- **Storage** — slices and vectors are persisted to
  ``{workspace}/.encre/embedding_index.npz`` and the slice metadata
  to ``.encre/embedding_index.json``.  Subsequent server starts can
  resume from disk without re-embedding the entire workspace.
- **Query** — :meth:`search` embeds the query string with the same
  backend, normalises both query and document vectors, and returns
  the top-k matches by cosine similarity.  Results are returned as
  :class:`EmbeddingHit` records.
- **Incremental update** — :meth:`scan_incremental` uses the AST
  index's per-file mtimes to skip unchanged files and re-embeds
  only what changed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np

from encre.codebase.ast_index import EncreASTIndex, Symbol

logger = logging.getLogger("encre.codebase.embedding_index")

EmbeddingFn = Callable[[list[str]], list[list[float]]]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingSlice:
    """A single embeddable region of a source file.

    Attributes:
        file: Workspace-relative path.
        start_line: 0-based starting line of the slice.
        end_line: 0-based ending line (inclusive).
        symbol: Name of the symbol covered by the slice, or an empty
            string for module-level slices.
        kind: Human-readable kind — matches :attr:`Symbol.kind` where
            applicable, otherwise ``"module"``.
        text: The source text that was actually embedded (useful for
            LLM context construction).
    """

    file: str
    start_line: int
    end_line: int
    symbol: str
    kind: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingSlice:
        return cls(
            file=str(data["file"]),
            start_line=int(data["start_line"]),
            end_line=int(data["end_line"]),
            symbol=str(data.get("symbol", "")),
            kind=str(data.get("kind", "module")),
            text=str(data.get("text", "")),
        )


@dataclass
class EmbeddingHit:
    """One search result returned by :meth:`EncreEmbeddingIndex.search`.

    Attributes:
        file: Workspace-relative path of the matched slice.
        start_line: 0-based starting line of the matched slice.
        end_line: 0-based ending line.
        symbol: Symbol name, or empty string for module slices.
        kind: ``"function"`` / ``"class"`` / ``"method"`` / etc.
        score: Cosine similarity in the closed interval ``[-1, 1]``.
            Higher is more similar.
        text: Source text of the slice.  Useful for prompt construction.
    """

    file: str
    start_line: int
    end_line: int
    symbol: str
    kind: str
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Default embedding backend — OpenAI-compatible HTTP API
# ---------------------------------------------------------------------------


class OpenAICompatibleEmbedding:
    """Embedding backend that talks to any OpenAI-compatible HTTP API.

    The class is intentionally small and dependency-free apart from
    :mod:`httpx` (which is already a hard dependency of the Encre
    framework).  It works with OpenAI, Azure OpenAI, OpenRouter,
    vLLM, Ollama, LM-Studio, and any other service that exposes a
    ``POST {base_url}/embeddings`` endpoint accepting the OpenAI
    request schema.

    Args:
        api_key: Bearer token.  Falls back to the ``OPENAI_API_KEY``
            environment variable.
        base_url: Base URL of the API.  Defaults to OpenAI's public
            endpoint.  Override for Azure / vLLM / Ollama.
        model: Model identifier (e.g. ``"text-embedding-3-small"``).
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        timeout: float = 60.0,
    ) -> None:
        self.api_key: str = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAICompatibleEmbedding requires an API key. Pass "
                "api_key=... or set the OPENAI_API_KEY environment variable."
            )
        self.base_url: str = base_url.rstrip("/")
        self.model: str = model
        self.timeout: float = timeout

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Lazy import so the class can be referenced in environments
        # where ``httpx`` is not yet importable (e.g. during early
        # dependency diagnostics).
        import httpx

        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Embedding request to {url} failed: {e}"
            ) from e
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Embedding API error {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        # OpenAI returns ``{"data": [{"embedding": [...], ...}, ...]}``
        # in the order of the inputs.
        items = data.get("data", [])
        if not isinstance(items, list) or len(items) != len(texts):
            raise RuntimeError(
                f"Embedding API returned unexpected payload shape: {data!r}"
            )
        return [list(item["embedding"]) for item in items]


# ---------------------------------------------------------------------------
# Helper — slice a file into embeddable regions using an AST index
# ---------------------------------------------------------------------------


def _slice_file_with_ast(
    rel_path: str,
    content: str,
    symbols: list[Symbol],
    min_chars: int = 32,
    max_chars: int = 4000,
) -> list[EmbeddingSlice]:
    """Convert an AST symbol table into embeddable text slices.

    Each top-level symbol (function, class, method, type, etc.)
    becomes a slice spanning ``[start_line, end_line]`` of the file.
    A single module-level slice is added for files that have no
    recognisable definitions, so the index still has a representation
    of "unstructured" files.

    Very long slices are trimmed to ``max_chars`` characters at a
    whitespace boundary to keep the embedding request size bounded.
    """
    if not symbols:
        # No AST symbols: treat the entire file as a single module
        # slice (truncated if necessary).
        text = content[:max_chars] if len(content) > max_chars else content
        if text.strip():
            return [
                EmbeddingSlice(
                    file=rel_path,
                    start_line=0,
                    end_line=content.count("\n"),
                    symbol="",
                    kind="module",
                    text=text,
                )
            ]
        return []

    lines = content.splitlines()
    out: list[EmbeddingSlice] = []
    for sym in symbols:
        if sym.start_line < 0 or sym.end_line < sym.start_line:
            continue
        if sym.start_line >= len(lines):
            continue
        end_line = min(sym.end_line, len(lines) - 1)
        if end_line < sym.start_line:
            continue
        chunk = "\n".join(lines[sym.start_line : end_line + 1])
        if len(chunk) < min_chars:
            continue
        if len(chunk) > max_chars:
            # Trim at the last whitespace before max_chars so we don't
            # split an identifier or keyword.
            cut = chunk.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunk = chunk[:cut]
        out.append(
            EmbeddingSlice(
                file=rel_path,
                start_line=sym.start_line,
                end_line=end_line,
                symbol=sym.name,
                kind=sym.kind,
                text=chunk,
            )
        )
    if not out:
        # File has symbols but none met the minimum size threshold.
        # Add a module-level fallback so the file is still searchable.
        text = content[:max_chars] if len(content) > max_chars else content
        if text.strip():
            out.append(
                EmbeddingSlice(
                    file=rel_path,
                    start_line=0,
                    end_line=content.count("\n"),
                    symbol="",
                    kind="module",
                    text=text,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Main index
# ---------------------------------------------------------------------------


class EncreEmbeddingIndex:
    """Vector index over embeddable code slices.

    The index is fed by an :class:`encre.codebase.ast_index.EncreASTIndex`
    that lives in the same workspace; the AST index supplies both the
    file list and the per-file symbol tables used for slicing.

    Args:
        workspace: Absolute or relative path to the workspace root.
        ast_index: Optional pre-built AST index.  If ``None``, one is
            created lazily on the first :meth:`scan` call.
        embedding_fn: Callable that turns a list of strings into a
            list of equal-length float vectors.  Defaults to an
            :class:`OpenAICompatibleEmbedding` instance using the
            ``OPENAI_API_KEY`` environment variable.
        embedding_dim: Optional expected dimension of the embedding
            vectors.  Inferred from the first call to ``embedding_fn``
            if not supplied.
        max_text_chars: Maximum characters per slice sent to the
            embedding backend.
    """

    _NPZ_NAME: str = "embedding_index.npz"
    _META_NAME: str = "embedding_index.json"

    def __init__(
        self,
        workspace: str,
        ast_index: EncreASTIndex | None = None,
        embedding_fn: EmbeddingFn | None = None,
        embedding_dim: int | None = None,
        max_text_chars: int = 4000,
    ) -> None:
        self.workspace: str = workspace
        self._ast: EncreASTIndex = ast_index if ast_index is not None else EncreASTIndex(workspace)
        self._embedding_fn: EmbeddingFn = embedding_fn or OpenAICompatibleEmbedding()
        self._embedding_dim: int | None = embedding_dim
        self._max_text_chars: int = max_text_chars

        self._slices: list[EmbeddingSlice] = []
        self._vectors: np.ndarray | None = None  # shape (N, D), float32, row-normalised
        self._file_mtimes: dict[str, float] = {}
        self._indexed: bool = False
        self._lock = threading.Lock()
        self.load()

    # ── Public properties ────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Return ``True`` once the index has at least one vector."""
        return self._vectors is not None and len(self._slices) > 0

    @property
    def slice_count(self) -> int:
        return len(self._slices)

    # ── Embedding helper ─────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._embedding_dim or 0), dtype=np.float32)
        raw = self._embedding_fn(texts)
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Embedding function returned a non-2D array: shape={arr.shape}"
            )
        if self._embedding_dim is None:
            self._embedding_dim = int(arr.shape[1])
        elif arr.shape[1] != self._embedding_dim:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                f"got {arr.shape[1]}"
            )
        # L2-normalise so cosine similarity collapses to a dot product.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (arr / norms).astype(np.float32)

    # ── Scanning ──────────────────────────────────────────────────────

    def scan(self) -> None:
        """Build the embedding index from scratch.

        Uses the AST index to find files and slice them.  Any
        embedding error aborts the scan and the on-disk cache is
        left untouched.
        """
        if not self._ast.available:
            logger.warning(
                "[embedding_index] AST index unavailable (tree-sitter not "
                "installed) — embedding index will be empty"
            )
            self._indexed = True
            return

        if not self._ast._indexed:
            self._ast.scan()

        all_slices: list[EmbeddingSlice] = []
        files = list(self._ast._symbols_by_file.items())
        for rel, symbols in files:
            fpath = Path(self.workspace) / rel
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            all_slices.extend(
                _slice_file_with_ast(rel, content, symbols, max_chars=self._max_text_chars)
            )
        # Also cover files that have AST symbols empty (config files, etc.).
        for rel in self._ast._file_mtimes.keys():
            fpath = Path(self.workspace) / rel
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            all_slices.extend(
                _slice_file_with_ast(rel, content, [], max_chars=self._max_text_chars)
            )

        if all_slices:
            with self._lock:
                vectors = self._embed([s.text for s in all_slices])
                self._slices = all_slices
                self._vectors = vectors
                self._file_mtimes = dict(self._ast._file_mtimes)
                self._indexed = True
                self.save()
        else:
            with self._lock:
                self._slices = []
                self._vectors = np.zeros((0, self._embedding_dim or 0), dtype=np.float32)
                self._file_mtimes = dict(self._ast._file_mtimes)
                self._indexed = True
                self.save()

    def scan_incremental(self) -> None:
        """Update the index for files that have changed since the last scan."""
        if not self._indexed:
            self.scan()
            return
        if not self._ast.available:
            return
        if not self._ast._indexed:
            self._ast.scan()
        changed: set[str] = set()
        for rel, mtime in self._ast._file_mtimes.items():
            prev = self._file_mtimes.get(rel)
            if prev is None or prev < mtime:
                changed.add(rel)
        deleted = set(self._file_mtimes.keys()) - set(self._ast._file_mtimes.keys())
        if not changed and not deleted:
            return
        # Remove vectors/slices belonging to changed or deleted files.
        keep_indices: list[int] = []
        for i, sl in enumerate(self._slices):
            if sl.file in changed or sl.file in deleted:
                continue
            keep_indices.append(i)
        new_slices = [self._slices[i] for i in keep_indices]
        new_vectors = (
            self._vectors[keep_indices]
            if self._vectors is not None and len(self._vectors) > 0
            else np.zeros((0, self._embedding_dim or 0), dtype=np.float32)
        )
        # Re-slice changed files and embed.
        for rel in changed:
            symbols = self._ast._symbols_by_file.get(rel, [])
            fpath = Path(self.workspace) / rel
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue
            new_file_slices = _slice_file_with_ast(
                rel, content, symbols, max_chars=self._max_text_chars
            )
            if not new_file_slices:
                continue
            new_vecs = self._embed([s.text for s in new_file_slices])
            new_slices.extend(new_file_slices)
            new_vectors = (
                np.vstack([new_vectors, new_vecs])
                if new_vectors.size > 0
                else new_vecs
            )
        with self._lock:
            self._slices = new_slices
            self._vectors = new_vectors
            self._file_mtimes = dict(self._ast._file_mtimes)
            self.save()

    # ── Search ───────────────────────────────────────────────────────

    def search(self, query: str, k: int = 10) -> list[EmbeddingHit]:
        """Return the top-k slices most semantically similar to ``query``.

        Args:
            query: Free-form natural-language or code query.
            k: Maximum number of results to return.

        Returns:
            A list of :class:`EmbeddingHit` ordered by descending
            cosine similarity.  An empty list is returned if the
            index is empty.
        """
        if not query:
            return []
        if self._vectors is None or len(self._vectors) == 0:
            return []
        q_vec = self._embed([query])
        if q_vec.size == 0:
            return []
        # Both ``_vectors`` and ``q_vec`` are L2-normalised, so the
        # cosine similarity is a plain matrix product.
        scores = self._vectors @ q_vec[0]
        if scores.size == 0:
            return []
        # ``argpartition`` is O(N); we then sort only the top-k.
        k = max(1, min(int(k), scores.size))
        if k >= scores.size:
            top_indices = np.argsort(-scores)
        else:
            partition = np.argpartition(-scores, k - 1)[:k]
            top_indices = partition[np.argsort(-scores[partition])]
        out: list[EmbeddingHit] = []
        for idx in top_indices:
            sl = self._slices[int(idx)]
            out.append(
                EmbeddingHit(
                    file=sl.file,
                    start_line=sl.start_line,
                    end_line=sl.end_line,
                    symbol=sl.symbol,
                    kind=sl.kind,
                    score=float(scores[int(idx)]),
                    text=sl.text,
                )
            )
        return out

    # ── Persistence ──────────────────────────────────────────────────

    def _storage_dir(self) -> Path:
        return Path(self.workspace) / ".encre"

    def _npz_path(self) -> Path:
        return self._storage_dir() / self._NPZ_NAME

    def _meta_path(self) -> Path:
        return self._storage_dir() / self._META_NAME

    def save(self) -> None:
        d = self._storage_dir()
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._vectors is not None:
                np.savez(
                    self._npz_path(),
                    vectors=self._vectors,
                )
            meta: dict[str, Any] = {
                "workspace": self.workspace,
                "embedding_dim": self._embedding_dim,
                "file_mtimes": self._file_mtimes,
                "slices": [s.to_dict() for s in self._slices],
            }
            self._meta_path().write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load(self) -> bool:
        """Load the index from disk if a previous run persisted it."""
        d = self._storage_dir()
        npz = d / self._NPZ_NAME
        meta = d / self._META_NAME
        if not npz.exists() or not meta.exists():
            return False
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            return False
        if data.get("workspace") != self.workspace:
            return False
        try:
            with np.load(npz) as npz_data:
                vectors = np.asarray(npz_data["vectors"], dtype=np.float32)
        except Exception:
            return False
        try:
            slices = [EmbeddingSlice.from_dict(s) for s in data.get("slices", [])]
        except Exception:
            slices = []
        with self._lock:
            self._slices = slices
            self._vectors = vectors
            self._file_mtimes = {str(k): float(v) for k, v in data.get("file_mtimes", {}).items()}
            self._embedding_dim = int(data.get("embedding_dim") or (vectors.shape[1] if vectors.ndim == 2 else 0))
            self._indexed = True
        return True
