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

import json
from typing import Any

from encre.notebook.session import EncreNotebookSession
from encre.tools.base import build_tool

_session: EncreNotebookSession | None = None


def _get_session() -> EncreNotebookSession:
    global _session
    if _session is None:
        _session = EncreNotebookSession()
    return _session


async def _notebook_execute(**kwargs: Any) -> str:
    action = kwargs.get("action", "")
    session = _get_session()

    if action == "create_cell":
        code = kwargs.get("code", "")
        cell_type = kwargs.get("cell_type", "code")
        cell_id = session.create_cell(code, cell_type)
        return json.dumps({"ok": True, "cell_id": cell_id})

    elif action == "edit_cell":
        cell_id = kwargs.get("cell_id", "")
        code = kwargs.get("code", "")
        ok = session.edit_cell(cell_id, code)
        return json.dumps({"ok": ok, "cell_id": cell_id})

    elif action == "execute_cell":
        cell_id = kwargs.get("cell_id", "")
        timeout = kwargs.get("timeout", 60)
        result = await session.execute_cell(cell_id, timeout)
        return json.dumps({"ok": True, **result}, ensure_ascii=False)

    elif action == "execute_all":
        timeout = kwargs.get("timeout", 300)
        results = await session.execute_all(timeout)
        return json.dumps({"ok": True, "results": results}, ensure_ascii=False)

    elif action == "get_output":
        cell_id = kwargs.get("cell_id", "")
        output = session.get_output(cell_id)
        error = session.get_error(cell_id)
        return json.dumps({"ok": True, "output": output, "error": error}, ensure_ascii=False)

    elif action == "get_state":
        state = session.get_state()
        return json.dumps(state, ensure_ascii=False)

    elif action == "delete_cell":
        cell_id = kwargs.get("cell_id", "")
        ok = session.delete_cell(cell_id)
        return json.dumps({"ok": ok})

    elif action == "reset":
        global _session
        if _session is not None:
            _session.close()
        kernel_name = kwargs.get("kernel_name", "python3")
        _session = EncreNotebookSession(kernel_name)
        return json.dumps({"ok": True, "message": "Notebook session reset"})

    else:
        return json.dumps({"ok": False, "error": f"Unknown action: {action}"})


EncreNotebookTool = build_tool(
    name="notebook",
    description=(
        "Manage an interactive Jupyter-style notebook session. "
        "Supports creating cells, editing cells, executing code, and inspecting results. "
        "Use this for iterative data exploration, visualization, or long-running computations."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create_cell",
                    "edit_cell",
                    "execute_cell",
                    "execute_all",
                    "get_output",
                    "get_state",
                    "delete_cell",
                    "reset",
                ],
                "description": "The notebook action to perform",
            },
            "code": {
                "type": "string",
                "description": "Python code for the cell (used with create_cell, edit_cell)",
            },
            "cell_id": {
                "type": "string",
                "description": "Cell ID (used with edit_cell, execute_cell, get_output, delete_cell)",
            },
            "cell_type": {
                "type": "string",
                "enum": ["code", "markdown"],
                "description": "Type of cell (default: code)",
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 60 for single, 300 for all)",
            },
            "kernel_name": {
                "type": "string",
                "description": "Kernel name for reset action (default: python3)",
            },
        },
        "required": ["action"],
    },
    execute=_notebook_execute,
    intents=["data"],
)
