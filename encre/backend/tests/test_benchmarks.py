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

"""Performance benchmarks for critical paths: semantic search, tokenization,
memory scanning, Jaccard/tf-idf vectorization."""

import time
import tempfile
import os

import pytest

from encre.memdir.semantic import _tokenize, _jaccard_similarity, _build_idf, _tf_idf_vectorize, _cosine_similarity


# ===========================================================================
# Tokenizer benchmarks
# ===========================================================================

class TestTokenizerBench:
    def test_tokenize_10k_lines(self):
        text = "Hello world this is a test of the tokenizer. " * 10000
        start = time.perf_counter()
        tokens = _tokenize(text)
        elapsed = time.perf_counter() - start
        assert len(tokens) > 0
        assert elapsed < 5.0, f"tokenize 10k lines took {elapsed:.2f}s"

    def test_tokenize_chinese(self):
        text = "测试中文分词效果 这是一个测试 " * 5000
        start = time.perf_counter()
        tokens = _tokenize(text)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"tokenize Chinese took {elapsed:.2f}s"


# ===========================================================================
# Jaccard benchmarks
# ===========================================================================

class TestJaccardBench:
    def test_jaccard_1000_pairs(self):
        docs = [f"This is document number {i} about various topics including python, rust, typescript, and more." for i in range(100)]
        start = time.perf_counter()
        for i in range(len(docs) - 1):
            _jaccard_similarity(docs[i], docs[i + 1])
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"1000 Jaccard pairs took {elapsed:.2f}s"

    def test_jaccard_empty(self):
        start = time.perf_counter()
        for _ in range(10000):
            _jaccard_similarity("", "")
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"10000 empty Jaccard took {elapsed:.2f}s"


# ===========================================================================
# TF-IDF benchmarks
# ===========================================================================

class TestTfIdfBench:
    def test_build_idf_large_corpus(self):
        corpus = [f"Document {i}: contains words about various topics like python programming and async rust development." for i in range(1000)]
        start = time.perf_counter()
        idf = _build_idf(corpus)
        elapsed = time.perf_counter() - start
        assert len(idf) > 0
        assert elapsed < 2.0, f"build_idf 1000 docs took {elapsed:.2f}s"

    def test_vectorize_and_cosine(self):
        corpus = [f"Document {i}: python rust typescript async programming patterns" for i in range(500)]
        idf = _build_idf(corpus)
        vocab = set(idf.keys())
        vecs = [_tf_idf_vectorize(doc, idf, vocab) for doc in corpus]
        start = time.perf_counter()
        for i in range(len(vecs) - 1):
            _cosine_similarity(vecs[i], vecs[i + 1])
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"500 cosine pairs took {elapsed:.2f}s"


# ===========================================================================
# Memory scan benchmarks
# ===========================================================================

class TestMemoryScanBench:
    def test_scan_large_memory_dir(self):
        tmpdir = tempfile.mkdtemp()
        # Create 200 memory files
        for i in range(200):
            with open(os.path.join(tmpdir, f"memory_{i:04d}.md"), "w", encoding="utf-8") as f:
                f.write(f"---\ndescription: Memory {i}\ntype: reference\n---\n\nContent for memory {i}.\n" * 10)

        from encre.memdir.system import EncreMemorySystem
        ms = EncreMemorySystem(tmpdir)
        start = time.perf_counter()
        memories = ms.scan()
        elapsed = time.perf_counter() - start
        assert len(memories) > 0
        assert elapsed < 3.0, f"scan 200 files took {elapsed:.2f}s"

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_semantic_search_performance(self):
        tmpdir = tempfile.mkdtemp()
        for i in range(100):
            with open(os.path.join(tmpdir, f"doc_{i:04d}.md"), "w", encoding="utf-8") as f:
                f.write(f"Document {i} about python programming and async patterns.\n" * 5)

        from encre.memdir.semantic import SemanticMemorySearch
        sms = SemanticMemorySearch(tmpdir)
        start = time.perf_counter()
        results = sms.search("python async programming", top_k=10)
        elapsed = time.perf_counter() - start
        assert len(results) > 0
        assert elapsed < 2.0, f"semantic search 100 docs took {elapsed:.2f}s"

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Consolidation benchmarks
# ===========================================================================

class TestConsolidationBench:
    def test_consolidate_many_files(self):
        tmpdir = tempfile.mkdtemp()
        from encre.memdir.semantic import MemoryConsolidator

        files = {}
        for i in range(50):
            files[f"doc_{i:04d}.md"] = f"Document {i} about {'async programming' if i % 2 == 0 else 'CSS styling'} patterns and best practices.\n" * 3

        mc = MemoryConsolidator(tmpdir)
        start = time.perf_counter()
        actions = mc.consolidate(files, {})
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"consolidate 50 files took {elapsed:.2f}s"

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
