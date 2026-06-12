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

import asyncio
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass


@dataclass
class _CellData:
    code: str
    cell_type: str
    output: str = ""
    error: str = ""
    status: str = "idle"
    execution_time: float = 0.0


class EncreNotebookSession:
    def __init__(self, kernel_name: str = "python3") -> None:
        self.kernel_name = kernel_name
        self.session_id: str = str(uuid.uuid4())
        self._cells: dict[str, _CellData] = {}
        self._cell_order: list[str] = []
        self._process: subprocess.Popen[bytes] | None = None
        self._started: bool = False
        self._kernel_script: str = (
            "import sys, json, traceback\n"
            "import builtins as _bi\n"
            "_ns = {}\n"
            'sys.stdout.write("READY\\n")\n'
            "sys.stdout.flush()\n"
            "for _line in sys.stdin:\n"
            "    _line = _line.strip()\n"
            "    if not _line:\n"
            "        continue\n"
            "    try:\n"
            "        _req = json.loads(_line)\n"
            "    except Exception:\n"
            "        continue\n"
            "    _code = _req.get('code', '')\n"
            "    if _code == '__SHUTDOWN__':\n"
            "        sys.stdout.write(json.dumps({'ok': True}))\n"
            "        sys.stdout.write('\\n')\n"
            "        sys.stdout.flush()\n"
            "        break\n"
            "    from io import StringIO\n"
            "    _out_io = StringIO()\n"
            "    _err_io = StringIO()\n"
            "    _old_stdout = sys.stdout\n"
            "    _old_stderr = sys.stderr\n"
            "    sys.stdout = _out_io\n"
            "    sys.stderr = _err_io\n"
            "    _ok = True\n"
            "    try:\n"
            "        exec(_code, _ns)\n"
            "    except Exception:\n"
            "        _ok = False\n"
            "        _err_io.write(traceback.format_exc())\n"
            "    finally:\n"
            "        sys.stdout = _old_stdout\n"
            "        sys.stderr = _old_stderr\n"
            "    _result = {'ok': _ok, 'output': _out_io.getvalue(), 'error': _err_io.getvalue()}\n"
            "    sys.stdout.write(json.dumps(_result) + '\\n')\n"
            "    sys.stdout.flush()\n"
        )

    def _ensure_kernel(self) -> None:
        if self._started and self._process is not None and self._process.poll() is None:
            return
        python_exe = self.kernel_name if self.kernel_name else sys.executable
        self._process = subprocess.Popen(
            [python_exe, "-u", "-c", self._kernel_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        ready_line = self._process.stdout.readline().decode("utf-8", errors="replace").strip() if self._process.stdout else ""
        if ready_line != "READY":
            self._started = False
            raise RuntimeError("Kernel failed to start")
        self._started = True

    def create_cell(self, code: str, cell_type: str = "code") -> str:
        cell_id = str(uuid.uuid4())[:8]
        self._cells[cell_id] = _CellData(code=code, cell_type=cell_type, status="idle")
        self._cell_order.append(cell_id)
        return cell_id

    def edit_cell(self, cell_id: str, code: str) -> bool:
        if cell_id not in self._cells:
            return False
        self._cells[cell_id].code = code
        self._cells[cell_id].status = "idle"
        self._cells[cell_id].output = ""
        self._cells[cell_id].error = ""
        return True

    async def execute_cell(self, cell_id: str, timeout: int = 60) -> dict[str, str | float]:
        if cell_id not in self._cells:
            return {"output": "", "error": f"Cell {cell_id} not found", "execution_time": 0.0}
        cell = self._cells[cell_id]
        self._ensure_kernel()
        t0 = time.time()
        request = json.dumps({"code": cell.code}) + "\n"
        try:
            proc = self._process
            if proc is None or proc.stdin is None or proc.stdout is None:
                cell.status = "error"
                cell.error = "Kernel process not available"
                return {"output": "", "error": "Kernel process not available", "execution_time": 0.0}
            proc.stdin.write(request.encode("utf-8"))
            proc.stdin.flush()
            loop = asyncio.get_running_loop()
            try:
                line_bytes = await asyncio.wait_for(
                    loop.run_in_executor(None, proc.stdout.readline),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                cell.status = "error"
                cell.error = f"Execution timed out after {timeout}s"
                cell.execution_time = time.time() - t0
                return {"output": "", "error": cell.error, "execution_time": cell.execution_time}
            result = json.loads(line_bytes.decode("utf-8", errors="replace"))
            elapsed = time.time() - t0
            cell.output = result.get("output", "")
            cell.error = result.get("error", "")
            cell.status = "completed" if result.get("ok", True) else "error"
            cell.execution_time = elapsed
            return {"output": cell.output, "error": cell.error, "execution_time": elapsed}
        except Exception as e:
            cell.status = "error"
            cell.error = str(e)
            cell.execution_time = time.time() - t0
            return {"output": "", "error": str(e), "execution_time": cell.execution_time}

    async def execute_all(self, timeout: int = 300) -> list[dict[str, str | float]]:
        results: list[dict[str, str | float]] = []
        for cell_id in self._cell_order:
            result = await self.execute_cell(cell_id, timeout=max(timeout // max(len(self._cell_order), 1), 10))
            results.append({"cell_id": cell_id, **result})
        return results

    def get_state(self) -> dict:
        cells = []
        for cell_id in self._cell_order:
            cell = self._cells.get(cell_id)
            if cell is None:
                continue
            cells.append({
                "id": cell_id,
                "code": cell.code,
                "cell_type": cell.cell_type,
                "output": cell.output,
                "error": cell.error,
                "status": cell.status,
                "execution_time": cell.execution_time,
            })
        return {
            "session_id": self.session_id,
            "kernel_name": self.kernel_name,
            "cells": cells,
            "cell_count": len(cells),
        }

    def get_output(self, cell_id: str) -> str:
        cell = self._cells.get(cell_id)
        if cell is None:
            return ""
        return cell.output

    def get_error(self, cell_id: str) -> str:
        cell = self._cells.get(cell_id)
        if cell is None:
            return ""
        return cell.error

    def delete_cell(self, cell_id: str) -> bool:
        if cell_id not in self._cells:
            return False
        del self._cells[cell_id]
        self._cell_order.remove(cell_id)
        return True

    def close(self) -> None:
        if self._process is not None and self._process.stdin is not None:
            try:
                request = json.dumps({"code": "__SHUTDOWN__"}) + "\n"
                self._process.stdin.write(request.encode("utf-8"))
                self._process.stdin.flush()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
        self._started = False
        self._process = None
