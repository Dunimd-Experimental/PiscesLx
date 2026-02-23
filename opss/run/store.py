#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

"""
Run Store - State File Management for Action System.

This module provides the core storage layer for action system state files.
All state files are protected by POPSSStateGuard to ensure only authorized
action system components can modify them.

State Files:
    - spec.json: Run specification
    - state.json: Run state (status, phase, pid, etc.)
    - events.jsonl: Event log
    - metrics.jsonl: Metrics log
    - control.jsonl: Control commands
    - stdout.log: Standard output log
    - artifacts.json: Artifact records

Note:
    External modules should use POPSSStateReader for read-only access.
    Direct write access is restricted to action system components only.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .lock import POPSSFileLock


class POPSSRunStore:
    """
    Run Store - State File Management for Action System.
    
    This class manages all state files for a single run. All write operations
    are protected by the action system authorization mechanism.
    
    Attributes:
        _run_id: Unique run identifier.
        _run_dir: Directory containing run state files.
        _paths: Dictionary mapping file keys to paths.
    
    Example:
        >>> store = POPSSRunStore("train_001")
        >>> store.write_state({"status": "running", "pid": 12345})
        >>> state = store.read_state()
        {'status': 'running', 'pid': 12345}
    """
    
    def __init__(self, run_id: str, run_dir: Optional[str] = None):
        """
        Initialize the run store.
        
        Args:
            run_id: Unique run identifier.
            run_dir: Optional custom run directory.
            
        Raises:
            ValueError: If run_id is empty.
        """
        self._run_id = str(run_id).strip()
        if not self._run_id:
            raise ValueError("run_id_required")
        if run_dir is None:
            root = Path(".pisceslx") / "runs"
            os.makedirs(str(root), exist_ok=True)
            self._run_dir = str(root / self._run_id)
        else:
            self._run_dir = str(Path(run_dir))

        self._paths = {
            "spec": str(Path(self._run_dir) / "spec.json"),
            "state": str(Path(self._run_dir) / "state.json"),
            "events": str(Path(self._run_dir) / "events.jsonl"),
            "metrics": str(Path(self._run_dir) / "metrics.jsonl"),
            "control": str(Path(self._run_dir) / "control.jsonl"),
            "stdout": str(Path(self._run_dir) / "stdout.log"),
            "artifacts": str(Path(self._run_dir) / "artifacts.json"),
            "lock": str(Path(self._run_dir) / ".lock"),
        }

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> str:
        return self._run_dir

    def ensure(self) -> None:
        os.makedirs(self._run_dir, exist_ok=True)

    def path(self, key: str) -> str:
        if key not in self._paths:
            raise KeyError(f"unknown_path_key:{key}")
        return self._paths[key]

    def _atomic_write_json(self, file_path: str, obj: Dict[str, Any]) -> None:
        self.ensure()
        tmp = f"{file_path}.tmp.{int(time.time() * 1000)}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, file_path)

    def write_spec(self, spec: Dict[str, Any]) -> None:
        with POPSSFileLock(self._paths["lock"]):
            self._atomic_write_json(self._paths["spec"], spec)

    def read_spec(self) -> Dict[str, Any]:
        p = self._paths["spec"]
        if not os.path.exists(p):
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}

    def write_state(self, state: Dict[str, Any]) -> None:
        with POPSSFileLock(self._paths["lock"]):
            self._atomic_write_json(self._paths["state"], state)

    def read_state(self) -> Dict[str, Any]:
        p = self._paths["state"]
        if not os.path.exists(p):
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}

    def merge_state(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        with POPSSFileLock(self._paths["lock"]):
            cur = self.read_state()
            cur.update(patch or {})
            self._atomic_write_json(self._paths["state"], cur)
            return cur

    def append_jsonl(self, key: str, obj: Dict[str, Any]) -> None:
        p = self._paths[key]
        self.ensure()
        line = json.dumps(obj, ensure_ascii=False)
        with POPSSFileLock(self._paths["lock"]):
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

    def append_event(self, obj: Dict[str, Any]) -> None:
        self.append_jsonl("events", obj)

    def append_metric(self, obj: Dict[str, Any]) -> None:
        self.append_jsonl("metrics", obj)

    def append_control(self, obj: Dict[str, Any]) -> None:
        self.append_jsonl("control", obj)

    def read_artifacts(self) -> Dict[str, Any]:
        p = self._paths["artifacts"]
        if not os.path.exists(p):
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}

    def merge_artifacts(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        with POPSSFileLock(self._paths["lock"]):
            cur = self.read_artifacts()
            cur.update(patch or {})
            self._atomic_write_json(self._paths["artifacts"], cur)
            return cur

    def tail_jsonl(self, key: str, offset_bytes: int = 0, max_bytes: int = 256 * 1024) -> Tuple[str, int]:
        p = self._paths[key]
        if not os.path.exists(p):
            return "", 0
        size = os.path.getsize(p)
        if offset_bytes < 0:
            offset_bytes = 0
        if offset_bytes > size:
            offset_bytes = size
        with open(p, "rb") as f:
            f.seek(offset_bytes)
            chunk = f.read(max_bytes)
            new_off = f.tell()
        try:
            return chunk.decode("utf-8", errors="replace"), new_off
        except Exception:
            return "", new_off

    def delete(self) -> bool:
        if os.path.exists(self._run_dir):
            shutil.rmtree(self._run_dir)
            return True
        return False

