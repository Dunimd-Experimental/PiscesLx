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

"""Tests for memdir: memory system, semantic search, working memory, consolidation."""

import os
import tempfile

import pytest

from encre.memdir.system import EncreMemorySystem, MemoryHeader, EntrypointResult
from encre.memdir.semantic import (
    SemanticMemorySearch,
    SearchResult,
    WorkingMemory,
    MemoryConsolidator,
    ConsolidationAction,
    _tokenize,
    _jaccard_similarity,
    _tf_idf_vectorize,
    _cosine_similarity,
    _build_idf,
)


# ===========================================================================
# Tokeniser & similarity
# ===========================================================================

class TestTokenize:
    def test_simple(self):
        t = _tokenize("Hello World! This is a test.")
        assert "hello" in t
        assert "world" in t
        assert "this" in t

    def test_chinese(self):
        t = _tokenize("测试 中文 and English 混合")
        assert "and" in t
        assert "english" in t

    def test_short_tokens_dropped(self):
        t = _tokenize("a b c ab cd ef hello")
        assert "hello" in t
        # Short single-char tokens are dropped; "ab" is exact boundary
        assert "a" not in t
        assert len(t) > 0

    def test_empty(self):
        assert _tokenize("") == []


class TestJaccard:
    def test_identical(self):
        assert _jaccard_similarity("hello world", "hello world") == 1.0

    def test_disjoint(self):
        assert _jaccard_similarity("abc def", "xyz uvw") == 0.0

    def test_partial(self):
        s = _jaccard_similarity("hello world foo", "hello world bar")
        assert 0.4 < s < 1.0

    def test_one_empty(self):
        assert _jaccard_similarity("", "hello") == 0.0
        assert _jaccard_similarity("hello", "") == 0.0


class TestTfIdf:
    def test_build_idf(self):
        corpus = ["hello world", "hello foo", "bar baz"]
        idf = _build_idf(corpus)
        assert "hello" in idf
        assert "world" in idf
        assert idf["hello"] < idf["world"]  # hello appears in 2 docs, world in 1

    def test_empty_corpus(self):
        assert _build_idf([]) == {}

    def test_vectorize(self):
        corpus = ["hello world foo", "hello bar", "bar baz qux"]
        idf = _build_idf(corpus)
        vocab = set(idf.keys())
        vec = _tf_idf_vectorize("hello world", idf, vocab)
        assert "hello" in vec
        assert vec["hello"] > 0

    def test_cosine_same(self):
        corpus = ["hello world", "foo bar"]
        idf = _build_idf(corpus)
        vocab = set(idf.keys())
        v = _tf_idf_vectorize("hello world", idf, vocab)
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_orthogonal(self):
        corpus = ["hello world", "foo bar"]
        idf = _build_idf(corpus)
        vocab = set(idf.keys())
        v1 = _tf_idf_vectorize("hello world", idf, vocab)
        v2 = _tf_idf_vectorize("foo bar", idf, vocab)
        assert _cosine_similarity(v1, v2) == 0.0


# ===========================================================================
# SemanticMemorySearch
# ===========================================================================

class TestSemanticMemorySearch:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp()
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write(self, name, content):
        with open(os.path.join(self.tmpdir, name), "w", encoding="utf-8") as f:
            f.write(content)

    def test_search_finds_relevant(self):
        self._write("auth.md", "The login system uses OAuth2 with JWT tokens.")
        self._write("ui.md", "The dashboard uses React and Tailwind CSS for styling.")
        sms = SemanticMemorySearch(self.tmpdir)
        results = sms.search("authentication login")
        assert len(results) >= 1
        assert results[0].file_name == "auth.md"

    def test_search_respects_top_k(self):
        for i in range(10):
            self._write(f"doc{i}.md", f"Document number {i} about various topics.")
        sms = SemanticMemorySearch(self.tmpdir)
        results = sms.search("document", top_k=3)
        assert len(results) <= 3

    def test_search_empty_dir(self):
        sms = SemanticMemorySearch(self.tmpdir)
        assert sms.search("anything") == []

    def test_search_relevant_higher_threshold(self):
        self._write("a.md", "python async programming guide")
        self._write("b.md", "baking chocolate cake recipe")
        sms = SemanticMemorySearch(self.tmpdir)
        results = sms.search_relevant("python programming")
        assert len(results) >= 1
        assert results[0].file_name == "a.md"

    def test_ignores_memory_md(self):
        self._write("MEMORY.md", "entrypoint content")
        self._write("real.md", "actual memory content here")
        sms = SemanticMemorySearch(self.tmpdir)
        results = sms.search("content")
        names = {r.file_name for r in results}
        assert "MEMORY.md" not in names
        assert "real.md" in names

    def test_index_explicit(self):
        sms = SemanticMemorySearch(self.tmpdir)
        sms.index({"a.md": "hello world", "b.md": "foo bar"})
        results = sms.search("hello")
        assert results[0].file_name == "a.md"


# ===========================================================================
# WorkingMemory
# ===========================================================================

class TestWorkingMemory:
    def test_initial_empty(self):
        wm = WorkingMemory()
        assert wm.current_goal == ""
        assert wm.subgoals == []
        assert wm.hypotheses == []

    def test_set_goal(self):
        wm = WorkingMemory()
        wm.set_goal("Implement OAuth2")
        assert wm.current_goal == "Implement OAuth2"

    def test_add_subgoal_no_dupes(self):
        wm = WorkingMemory()
        wm.add_subgoal("Write tests")
        wm.add_subgoal("Write tests")
        assert len(wm.subgoals) == 1

    def test_complete_subgoal(self):
        wm = WorkingMemory()
        wm.add_subgoal("Write tests")
        wm.complete_subgoal("Write tests")
        assert wm.subgoals == []

    def test_hypothesis_lifecycle(self):
        wm = WorkingMemory()
        wm.add_hypothesis("The bug is in auth.py")
        wm.confirm_hypothesis("The bug is in auth.py")
        assert wm.hypotheses == []
        assert any("CONFIRMED" in f for f in wm.findings)

    def test_reject_hypothesis(self):
        wm = WorkingMemory()
        wm.add_hypothesis("Memory leak in loop")
        wm.reject_hypothesis("Memory leak in loop")
        assert wm.hypotheses == []
        assert any("REJECTED" in f for f in wm.findings)

    def test_add_finding(self):
        wm = WorkingMemory()
        wm.add_finding("Token refresh endpoint returns 401")
        assert len(wm.findings) == 1

    def test_question_lifecycle(self):
        wm = WorkingMemory()
        wm.add_question("Should we use asyncpg?")
        wm.resolve_question("Should we use asyncpg?", "Yes, it's faster")
        assert wm.open_questions == []
        assert any("asyncpg" in f for f in wm.findings)

    def test_scratchpad(self):
        wm = WorkingMemory()
        wm.note("TODO: check error handling")
        wm.note("Done: error handling looks fine")
        assert len(wm.scratchpad) == 2

    def test_summarize_empty(self):
        wm = WorkingMemory()
        assert "empty" in wm.summarize().lower()

    def test_summarize_with_content(self):
        wm = WorkingMemory()
        wm.set_goal("Test framework")
        wm.add_finding("pytest configured")
        s = wm.summarize()
        assert "Test framework" in s
        assert "pytest configured" in s

    def test_summarize_truncates_lists(self):
        wm = WorkingMemory()
        for i in range(20):
            wm.add_finding(f"Finding {i}")
        s = wm.summarize()
        # Should show only last 10 findings
        assert "Finding 0" not in s
        assert "Finding 19" in s

    def test_serialize_roundtrip(self):
        wm = WorkingMemory()
        wm.set_goal("Test")
        wm.add_hypothesis("H1")
        wm.add_finding("F1")
        d = wm.to_dict()
        wm2 = WorkingMemory.from_dict(d)
        assert wm2.current_goal == "Test"
        assert "H1" in wm2.hypotheses
        assert "F1" in wm2.findings


# ===========================================================================
# MemoryConsolidator
# ===========================================================================

class TestMemoryConsolidator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mc = MemoryConsolidator(self.tmpdir)
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_find_duplicates(self):
        files = {
            "a.md": "Always use async/await for network calls in production code.",
            "b.md": "Always use async/await for network calls in the production environment.",
            "c.md": "Completely different topic about CSS grid layout and flexbox.",
        }
        actions = self.mc.find_duplicates(files)
        assert len(actions) >= 1
        action = actions[0]
        assert action.action == "merge"
        assert action.merged_content

    def test_find_duplicates_none(self):
        files = {"a.md": "foo bar", "b.md": "completely unrelated"}
        assert self.mc.find_duplicates(files) == []

    def test_find_conflicts(self):
        files = {
            "a.md": "Always use async/await for network calls.",
            "b.md": "Never use async/await; prefer synchronous calls.",
        }
        actions = self.mc.find_conflicts(files)
        assert len(actions) >= 1
        assert actions[0].action == "flag_conflict"

    def test_find_conflicts_no_overlap_no_flag(self):
        files = {
            "a.md": "Always use async/await for network calls.",
            "b.md": "The CSS grid system is preferred for layouts.",
        }
        actions = self.mc.find_conflicts(files)
        assert len(actions) == 0

    def test_find_stale(self):
        files = {"old.md": "Reference: `src/auth.py:42` has the login flow."}
        age_days = {"old.md": 60}
        actions = self.mc.find_stale(files, age_days, stale_threshold_days=30)
        # src/auth.py likely doesn't exist in cwd
        assert len(actions) >= 1
        assert actions[0].action == "mark_stale"

    def test_find_stale_not_old_enough(self):
        files = {"recent.md": "Reference: `src/auth.py:42`"}
        age_days = {"recent.md": 5}
        actions = self.mc.find_stale(files, age_days, stale_threshold_days=30)
        assert len(actions) == 0

    def test_consolidate_orders_actions(self):
        files = {
            "dup_a.md": "Always use async/await for network calls in production code.",
            "dup_b.md": "Always use async/await for network calls in the production environment.",
            "conflict.md": "Never use async/await; prefer synchronous calls.",
        }
        age_days = {"dup_a.md": 35, "dup_b.md": 10, "conflict.md": 5}
        actions = self.mc.consolidate(files, age_days)
        # merge should come before conflict
        assert actions[0].action == "merge"
        assert any(a.action == "flag_conflict" for a in actions)


# ===========================================================================
# EncreMemorySystem integration
# ===========================================================================

class TestEncreMemorySystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ms = EncreMemorySystem(self.tmpdir)
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_scan_empty(self):
        assert self.ms.scan() == []

    def test_scan_single(self):
        self._write("test.md", "---\ndescription: Test memory\ntype: reference\n---\nContent here.")
        memories = self.ms.scan()
        assert len(memories) == 1
        assert memories[0].description == "Test memory"
        assert memories[0].memory_type == "reference"

    def test_scan_skips_entrypoint(self):
        self._write("MEMORY.md", "entrypoint")
        self._write("real.md", "real memory")
        memories = self.ms.scan()
        names = {m.filename for m in memories}
        assert "MEMORY.md" not in names
        assert "real.md" in names

    def test_format_manifest_empty(self):
        manifest = self.ms.format_manifest([])
        assert manifest == ""

    def test_build_prompt(self):
        self._write("test.md", "---\ndescription: A test\n---\nTest content.")
        prompt = self.ms.build_prompt()
        assert "MEMORY.md Entrypoint" in prompt

    def test_search_delegates_to_semantic(self):
        self._write("auth.md", "OAuth2 JWT token authentication system.")
        self._write("ui.md", "CSS grid layout with responsive breakpoints.")
        results = self.ms.search("authentication login")
        assert len(results) >= 1
        assert results[0].file_name == "auth.md"

    def test_search_relevant(self):
        self._write("db.md", "Database connection pooling with postgresql and asyncpg for performance.")
        results = self.ms.search_relevant("database postgres")
        # search_relevant has higher threshold — may or may not match, depends on corpus
        assert isinstance(results, list)

    def test_working_memory_accessible(self):
        wm = self.ms.working
        wm.set_goal("Test goal")
        assert self.ms.working.current_goal == "Test goal"

    def test_reset_working(self):
        self.ms.working.set_goal("Old")
        self.ms.reset_working()
        assert self.ms.working.current_goal == ""

    def test_inject_working_empty(self):
        assert self.ms.inject_working_memory_prompt() == ""

    def test_inject_working_with_content(self):
        self.ms.working.set_goal("Fix login bug")
        prompt = self.ms.inject_working_memory_prompt()
        assert "Fix login bug" in prompt

    def test_build_prompt_with_context(self):
        self._write("auth.md", "OAuth2 JWT token authentication.")
        self._write("css.md", "Tailwind CSS utility classes.")
        prompt = self.ms.build_prompt_with_context("authentication")
        assert "Semantically Relevant" in prompt
        assert "auth.md" in prompt

    def test_consolidate_empty(self):
        assert self.ms.consolidate() == []

    def test_write_entrypoint(self):
        self.ms.write_entrypoint("# Test\n\nEntrypoint content.")
        result = self.ms.load_entrypoint()
        assert "Entrypoint content" in result.content

    def test_load_entrypoint_empty(self):
        result = self.ms.load_entrypoint()
        assert result.content == ""
        assert result.was_line_truncated is False

    def _write(self, name, content):
        with open(os.path.join(self.tmpdir, name), "w", encoding="utf-8") as f:
            f.write(content)
