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

"""Tests for encre.codebase.indexer — EncreCodeIndex and ModuleInfo."""

import os
import tempfile
import textwrap

import pytest


# ===========================================================================
# ModuleInfo dataclass
# ===========================================================================

class TestModuleInfo:
    """Tests for the ModuleInfo dataclass."""

    def test_creation_with_all_fields(self):
        from encre.codebase.indexer import ModuleInfo
        mi = ModuleInfo(
            path="src/my_module.py",
            name="my_module",
            imports=["os", "json", "typing"],
            imported_by=["main.py", "test_module.py"],
            exports=["public_func", "MyClass", "CONSTANT"],
            language="python",
            loc=150,
        )
        assert mi.path == "src/my_module.py"
        assert mi.name == "my_module"
        assert len(mi.imports) == 3
        assert "os" in mi.imports
        assert len(mi.imported_by) == 2
        assert "main.py" in mi.imported_by
        assert len(mi.exports) == 3
        assert "MyClass" in mi.exports
        assert mi.language == "python"
        assert mi.loc == 150

    def test_default_values(self):
        from encre.codebase.indexer import ModuleInfo
        mi = ModuleInfo(path="test.py", name="test")
        assert mi.imports == []
        assert mi.imported_by == []
        assert mi.exports == []
        assert mi.language == ""
        assert mi.loc == 0

    def test_is_dataclass(self):
        from encre.codebase.indexer import ModuleInfo
        from dataclasses import is_dataclass
        assert is_dataclass(ModuleInfo)

    def test_language_variants(self):
        from encre.codebase.indexer import ModuleInfo
        for lang in ["python", "rust", "go", "javascript", "typescript", "java"]:
            mi = ModuleInfo(path=f"src/module.{lang[:2]}", name="mod", language=lang)
            assert mi.language == lang

    def test_windows_path_normalization(self):
        from encre.codebase.indexer import ModuleInfo
        mi = ModuleInfo(path="src\\subdir\\module.py", name="module")
        assert "src" in mi.path


# ===========================================================================
# EncreCodeIndex construction
# ===========================================================================

class TestEncreCodeIndexConstruction:
    """Tests for EncreCodeIndex construction and initial state."""

    def test_construction(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        assert ci is not None
        assert ci.workspace == "."
        assert ci._indexed is False

    def test_construction_absolute_path(self):
        from encre.codebase.indexer import EncreCodeIndex
        abs_path = os.path.abspath(".")
        ci = EncreCodeIndex(workspace=abs_path)
        assert ci.workspace == abs_path

    def test_initial_state_empty(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        assert ci._modules == {}
        assert ci._depgraph == {}
        assert ci._reverse_depgraph == {}
        assert ci._inverted_index == {}
        assert ci._total_docs == 0
        assert ci._indexed is False

    def test_known_extensions_set(self):
        from encre.codebase.indexer import EncreCodeIndex
        assert ".py" in EncreCodeIndex._KNOWN_EXTS
        assert ".rs" in EncreCodeIndex._KNOWN_EXTS
        assert ".go" in EncreCodeIndex._KNOWN_EXTS
        assert ".js" in EncreCodeIndex._KNOWN_EXTS
        assert ".ts" in EncreCodeIndex._KNOWN_EXTS
        assert ".java" in EncreCodeIndex._KNOWN_EXTS


# ===========================================================================
# EncreCodeIndex scan and search
# ===========================================================================

class TestEncreCodeIndexScan:
    """Tests for scanning a real codebase."""

    def test_scan_runs_without_error(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        ci.scan()
        assert ci._indexed is True

    def test_scan_indexes_modules(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        ci.scan()
        assert len(ci._modules) > 0

    def test_scan_modules_have_paths(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        ci.scan()
        for path, mod in ci._modules.items():
            assert isinstance(mod.path, str)
            assert len(mod.path) > 0

    def test_scan_finds_python_files(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        ci.scan()
        python_modules = [m for m in ci._modules.values() if m.language == "python"]
        assert len(python_modules) > 0

    def test_scan_empty_directory(self):
        from encre.codebase.indexer import EncreCodeIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            ci = EncreCodeIndex(workspace=tmpdir)
            ci.scan()
            assert ci._indexed is True
            assert len(ci._modules) == 0

    def test_scan_nonexistent_directory(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace="/nonexistent/path/for/testing")
        ci.scan()
        assert ci._indexed is True
        assert len(ci._modules) == 0


class TestEncreCodeIndexWithFiles:
    """Tests that scan a temporary directory with known files."""

    def test_scan_python_file_parses_imports(self):
        from encre.codebase.indexer import EncreCodeIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "test_mod.py")
            with open(src, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent("""\
                    import os
                    import json
                    from collections import defaultdict

                    def public_function():
                        return 42

                    class MyClass:
                        pass

                    CONSTANT = 3.14
                """))
            ci = EncreCodeIndex(workspace=tmpdir)
            ci.scan()
            assert ci._indexed is True
            assert len(ci._modules) == 1
            mod_key = list(ci._modules.keys())[0]
            mod = ci._modules[mod_key]
            assert mod.language == "python"
            assert "os" in mod.imports
            assert "json" in mod.imports
            assert "public_function" in mod.exports
            assert "MyClass" in mod.exports
            assert "CONSTANT" in mod.exports


# ===========================================================================
# EncreCodeIndex public query API
# ===========================================================================

class TestEncreCodeIndexQueries:
    """Tests for the public query methods."""

    def test_build_dependency_graph(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        graph = ci.build_dependency_graph()
        assert isinstance(graph, dict)

    def test_get_importers(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        ci.scan()
        # Pick any module and query its importers
        if ci._modules:
            first_path = list(ci._modules.keys())[0]
            importers = ci.get_importers(first_path)
            assert isinstance(importers, list)

    def test_get_importers_nonexistent(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        importers = ci.get_importers("nonexistent_file.py")
        assert importers == []

    def test_find_relevant_returns_list(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        results = ci.find_relevant("python class")
        assert isinstance(results, list)

    def test_find_relevant_empty_query(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        results = ci.find_relevant("")
        assert results == []

    def test_find_relevant_returns_tuples(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        results = ci.find_relevant("import")
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_find_relevant_sorted_descending(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        results = ci.find_relevant("def class")
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    def test_build_context_returns_str(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        ci.scan()
        if ci._modules:
            first_path = list(ci._modules.keys())[0]
            context = ci.build_context(first_path)
            assert isinstance(context, str)
            assert len(context) > 0

    def test_build_context_nonexistent(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        context = ci.build_context("no_such_file.py")
        assert context == ""

    def test_get_module_info(self):
        from encre.codebase.indexer import EncreCodeIndex, ModuleInfo
        ci = EncreCodeIndex(workspace=".")
        ci.scan()
        if ci._modules:
            first_path = list(ci._modules.keys())[0]
            mod = ci.get_module_info(first_path)
            assert isinstance(mod, ModuleInfo)

    def test_get_module_info_nonexistent(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        mod = ci.get_module_info("nonexistent.py")
        assert mod is None

    def test_list_all_modules_returns_list(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        modules = ci.list_all_modules()
        assert isinstance(modules, list)
        from encre.codebase.indexer import ModuleInfo
        for mod in modules:
            assert isinstance(mod, ModuleInfo)

    def test_search_by_name_returns_list(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        results = ci.search_by_name("agent")
        assert isinstance(results, list)

    def test_search_by_name_case_insensitive(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        upper = ci.search_by_name("AGENT")
        lower = ci.search_by_name("agent")
        assert len(upper) == len(lower)


# ===========================================================================
# EncreCodeIndex incremental scan
# ===========================================================================

class TestEncreCodeIndexIncremental:
    """Tests for incremental scanning."""

    def test_scan_incremental_on_fresh_index(self):
        from encre.codebase.indexer import EncreCodeIndex
        ci = EncreCodeIndex(workspace=".")
        ci.scan_incremental()
        assert ci._indexed is True

    def test_scan_incremental_after_full_scan(self):
        from encre.codebase.indexer import EncreCodeIndex
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file
            src = os.path.join(tmpdir, "hello.py")
            with open(src, "w", encoding="utf-8") as f:
                f.write("import os\n\ndef greet():\n    return 'hello'\n")

            ci = EncreCodeIndex(workspace=tmpdir)
            ci.scan()
            assert len(ci._modules) == 1

            # Create a new file
            src2 = os.path.join(tmpdir, "world.py")
            with open(src2, "w", encoding="utf-8") as f:
                f.write("import sys\n\ndef farewell():\n    return 'bye'\n")

            ci.scan_incremental()
            assert len(ci._modules) == 2
