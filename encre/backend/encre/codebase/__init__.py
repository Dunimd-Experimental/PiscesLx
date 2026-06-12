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
Codebase indexing module — multi-language code search and dependency analysis.

This package provides the :class:`EncreCodeIndex` class, which builds an
in-memory searchable index of source code files across a workspace
directory.  It supports:

- **Multi-language parsing**: Python (AST), JavaScript/TypeScript (regex),
  Rust (regex), Go (regex), and generic fallback for other languages.
- **Full-text search with BM25 ranking**: TF-IDF style retrieval scoring
  with Okapi BM25, tuned for code search.
- **Dependency graph construction**: Resolves import statements between
  modules and builds forward and reverse dependency graphs.
- **Incremental scanning**: Re-parses only changed files since the last
  scan, preserving existing index data for unchanged files.
- **File watching**: Optional ``watchfiles``-based live file watcher that
  triggers incremental re-indexing on filesystem changes.
- **Context building**: Generates a formatted context string for a given
  file path, including source code, imports, dependents, and exports.

Typical usage::

    index = EncreCodeIndex("/path/to/workspace")
    index.scan()
    results = index.find_relevant("database connection", limit=5)
    deps = index.build_dependency_graph()
    importers = index.get_importers("src/main.py")
    context = index.build_context("src/utils.py")
"""

from encre.codebase.indexer import EncreCodeIndex, ModuleInfo
from encre.codebase.document_manager import EncreDocument, EncreDocumentManager
from encre.codebase.ast_index import EncreASTIndex, Symbol, Reference
from encre.codebase.embedding_index import (
    EmbeddingHit,
    EmbeddingSlice,
    EncreEmbeddingIndex,
    OpenAICompatibleEmbedding,
)

__all__ = [
    "EncreCodeIndex",
    "ModuleInfo",
    "EncreDocument",
    "EncreDocumentManager",
    "EncreASTIndex",
    "Symbol",
    "Reference",
    "EncreEmbeddingIndex",
    "EmbeddingSlice",
    "EmbeddingHit",
    "OpenAICompatibleEmbedding",
]