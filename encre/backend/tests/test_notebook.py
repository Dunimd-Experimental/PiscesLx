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

"""Tests for encre.notebook.session — EncreNotebookSession."""

import asyncio
import uuid

import pytest


# ===========================================================================
# EncreNotebookSession construction
# ===========================================================================

class TestEncreNotebookSessionConstruction:
    """Tests for creating EncreNotebookSession instances."""

    def test_construction_default_kernel(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        assert sess is not None
        assert sess.kernel_name == "python3"

    def test_construction_custom_kernel(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession(kernel_name="python3.12")
        assert sess.kernel_name == "python3.12"

    def test_construction_other_python_versions(self):
        from encre.notebook.session import EncreNotebookSession
        for kname in ["python3.9", "python3.10", "python3.11", "python3.13"]:
            sess = EncreNotebookSession(kernel_name=kname)
            assert sess.kernel_name == kname

    def test_session_id_is_uuid_string(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        assert isinstance(sess.session_id, str)
        # Should be a valid UUID
        uuid.UUID(sess.session_id)

    def test_each_session_has_unique_id(self):
        from encre.notebook.session import EncreNotebookSession
        a = EncreNotebookSession()
        b = EncreNotebookSession()
        assert a.session_id != b.session_id

    def test_initial_state_not_started(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        assert sess._started is False
        assert sess._process is None

    def test_initial_cells_empty(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        assert sess._cells == {}
        assert sess._cell_order == []

    def test_kernel_script_is_python_code(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        assert "import sys" in sess._kernel_script
        assert "exec(" in sess._kernel_script
        assert "__SHUTDOWN__" in sess._kernel_script


# ===========================================================================
# EncreNotebookSession cell management
# ===========================================================================

class TestEncreNotebookSessionCells:
    """Tests for cell CRUD operations (no kernel needed)."""

    def test_create_cell_returns_id(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        cell_id = sess.create_cell(code="print('hello')")
        assert isinstance(cell_id, str)
        assert len(cell_id) == 8

    def test_create_cell_default_type_code(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        cell_id = sess.create_cell(code="x = 1")
        state = sess.get_state()
        cells = state["cells"]
        assert len(cells) == 1
        assert cells[0]["cell_type"] == "code"

    def test_create_cell_markdown_type(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        cell_id = sess.create_cell(code="# Title", cell_type="markdown")
        state = sess.get_state()
        cells = state["cells"]
        assert cells[0]["cell_type"] == "markdown"

    def test_create_multiple_cells_preserves_order(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        id1 = sess.create_cell(code="a = 1")
        id2 = sess.create_cell(code="b = 2")
        id3 = sess.create_cell(code="c = 3")
        state = sess.get_state()
        cell_ids = [c["id"] for c in state["cells"]]
        assert cell_ids == [id1, id2, id3]

    def test_edit_cell_changes_code(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        cell_id = sess.create_cell(code="original")
        result = sess.edit_cell(cell_id, code="modified")
        assert result is True
        state = sess.get_state()
        assert state["cells"][0]["code"] == "modified"

    def test_edit_nonexistent_cell_returns_false(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        result = sess.edit_cell("nonexistent", code="x = 1")
        assert result is False

    def test_edit_cell_resets_status_and_outputs(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        cell_id = sess.create_cell(code="print('hi')")
        sess.edit_cell(cell_id, code="print('hello')")
        state = sess.get_state()
        cell = state["cells"][0]
        assert cell["status"] == "idle"
        assert cell["output"] == ""
        assert cell["error"] == ""

    def test_delete_cell_removes_from_state(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        cell_id = sess.create_cell(code="x = 1")
        assert sess.delete_cell(cell_id) is True
        state = sess.get_state()
        assert state["cell_count"] == 0
        assert state["cells"] == []

    def test_delete_nonexistent_cell_returns_false(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        result = sess.delete_cell("no_such_cell")
        assert result is False

    def test_delete_cell_preserves_order(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        id1 = sess.create_cell(code="a = 1")
        id2 = sess.create_cell(code="b = 2")
        id3 = sess.create_cell(code="c = 3")
        sess.delete_cell(id2)
        state = sess.get_state()
        cell_ids = [c["id"] for c in state["cells"]]
        assert cell_ids == [id1, id3]

    def test_get_output_nonexistent_returns_empty(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        result = sess.get_output("no_such_cell")
        assert result == ""

    def test_get_error_nonexistent_returns_empty(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        result = sess.get_error("no_such_cell")
        assert result == ""

    def test_get_output_for_existing_cell(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        cell_id = sess.create_cell(code="x = 1")
        # Cell hasn't been executed, so output is empty
        assert sess.get_output(cell_id) == ""


# ===========================================================================
# EncreNotebookSession state
# ===========================================================================

class TestEncreNotebookSessionState:
    """Tests for get_state."""

    def test_get_state_initial(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        state = sess.get_state()
        assert state["session_id"] == sess.session_id
        assert state["kernel_name"] == "python3"
        assert state["cells"] == []
        assert state["cell_count"] == 0

    def test_get_state_after_creating_cells(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        sess.create_cell(code="x = 1")
        sess.create_cell(code="y = 2")
        state = sess.get_state()
        assert state["cell_count"] == 2
        assert len(state["cells"]) == 2
        assert state["cells"][0]["code"] == "x = 1"
        assert state["cells"][1]["code"] == "y = 2"

    def test_get_state_keys(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        state = sess.get_state()
        for key in ["session_id", "kernel_name", "cells", "cell_count"]:
            assert key in state

    def test_cell_state_keys(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        sess.create_cell(code="print('hi')")
        state = sess.get_state()
        cell = state["cells"][0]
        for key in ["id", "code", "cell_type", "output", "error", "status", "execution_time"]:
            assert key in cell

    def test_cell_initial_status_idle(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        cell_id = sess.create_cell(code="print('hi')")
        state = sess.get_state()
        assert state["cells"][0]["status"] == "idle"
        assert state["cells"][0]["execution_time"] == 0.0


# ===========================================================================
# EncreNotebookSession close
# ===========================================================================

class TestEncreNotebookSessionClose:
    """Tests for session close behavior."""

    def test_close_before_kernel_started_does_not_raise(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        sess.close()
        assert sess._started is False
        assert sess._process is None

    def test_close_is_idempotent(self):
        from encre.notebook.session import EncreNotebookSession
        sess = EncreNotebookSession()
        sess.close()
        sess.close()
        assert sess._started is False
