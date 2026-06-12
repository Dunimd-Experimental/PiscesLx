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
Lightweight intent classifier for dynamic prompt assembly.

Classifies user queries into intent categories (coding, research, data, etc.)
to drive conditional prompt block inclusion and tool filtering.
"""

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code", "function", "class", "import", "def ", "async ", "await ",
        "bug", "fix", "refactor", "test", "compile", "build", "deploy",
        "python", "javascript", "typescript", "rust", "go ", "golang",
        "react", "vue", "angular", "node", "express", "django", "flask",
        "api", "endpoint", "route", "component", "hook", "state",
        "css", "html", "style", "layout", "responsive",
        "database", "sql", "query", "schema", "migration",
        "docker", "container", "kubernetes", "ci/cd", "pipeline",
        "git", "commit", "branch", "merge", "rebase", "repo", "pull request", "pr",
        ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".rb",
        "script", "module", "package", "dependency", "npm", "pip",
        "lsp", "language server", "autocomplete", "diagnostics",
        "error", "exception", "traceback", "stack trace",
        "file_read", "file_edit", "file_write", "grep", "glob",
    ],
    "research": [
        "search", "find", "look up", "research", "what is",
        "how to", "explain", "tell me about", "information",
        "news", "article", "documentation", "docs", "wiki",
        "latest", "update", "trend", "compare", "difference",
        "definition", "meaning", "example", "tutorial",
        "web search", "web_search", "web fetch", "web_fetch",
        "url", "http", "https", "website", "page",
        "paper", "publication", "reference", "source",
    ],
    "data": [
        "data", "csv", "excel", "spreadsheet", ".csv", ".xlsx",
        "notebook", "jupyter", ".ipynb", "pandas", "matplotlib",
        "analysis", "analyze", "visualize", "chart", "graph",
        "statistics", "statistical", "regression", "correlation",
        "dataset", "dataframe", "column", "row", "table",
        "pdf", ".pdf", "document", "extract",
        "image", "photo", "picture", ".png", ".jpg", ".jpeg", "screenshot",
        "plot", "scatter", "histogram", "bar chart",
        "sql", "query", "database",
    ],
}


def _is_conversational(query: str) -> bool:
    """Heuristic: short queries without coding/data/research signals are conversational."""
    query_stripped = query.strip()
    if len(query_stripped) > 30:
        return False

    # Single punctuation, greetings, short questions with no domain keywords
    is_short_greeting = (
        len(query_stripped) <= 3
        or query_stripped.lower() in ("hi", "hey", "hello", "yo", "sup", "ok", "thanks", "ty", "help")
        or query_stripped in ("?", "??", "!?", "!")
    )

    if is_short_greeting:
        return True

    # Check if the query has any domain keyword
    query_lower = query.lower()
    for keywords in _DOMAIN_KEYWORDS.values():
        for kw in keywords:
            if kw in query_lower:
                return False

    # Short, no domain keywords = conversational
    if len(query_stripped) < 20:
        return True

    return False


def classify_intents(query: str) -> list[str]:
    """Classify a user query into intent labels for prompt assembly.

    Returns a list of intent strings such as ["general", "coding"] or
    ["conversation"] for purely conversational queries.
    """
    intents: set[str] = {"general"}

    if _is_conversational(query):
        intents.add("conversation")
        return list(intents)

    query_lower = query.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                intents.add(domain)
                break

    return list(intents)
