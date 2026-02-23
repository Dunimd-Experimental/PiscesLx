#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
State File Guard for Action System Isolation.

This module provides write protection for action system state files,
ensuring only authorized action components can modify them while
allowing read-only access to external modules.

Key Features:
    - Write authorization control
    - Read-only access for external modules
    - State file integrity protection
    - Audit logging for unauthorized access attempts

State Files Protected:
    - .pisceslx/runs/<run_id>/state.json
    - .pisceslx/runs/<run_id>/spec.json
    - .pisceslx/runs/<run_id>/metrics.json
    - .pisceslx/gpu_scheduler/gpu_state.json
    - .pisceslx/queue/queue_state.json
    - .pisceslx/resources/resource_state.json
    - .pisceslx/recovery/recovery_state.json

Usage:
    # In action system (write access):
    from opss.run.state_guard import POPSSStateGuard, write_authorized
    
    @write_authorized
    def update_state(run_id, data):
        guard = POPSSStateGuard()
        guard.write_state(run_id, data)
    
    # In external modules (read-only access):
    from opss.run.state_guard import POPSSStateGuard
    
    guard = POPSSStateGuard()
    state = guard.read_state(run_id)  # OK - read allowed
    guard.write_state(run_id, data)   # Raises UnauthorizedWriteError
"""

import functools
import hashlib
import inspect
import json
import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


class POPSSUnauthorizedWriteError(PermissionError):
    """Raised when an unauthorized module attempts to write state files."""
    pass


class POPSSStateFileCorruptedError(Exception):
    """Raised when a state file is corrupted or tampered with."""
    pass


class POPSSStateGuard:
    """
    State File Guard for Action System Isolation.
    
    This class controls write access to action system state files,
    ensuring only authorized components within the action system can
    modify them. External modules can only read.
    
    Authorization Rules:
        1. Modules in AUTHORIZED_MODULES can write
        2. Functions decorated with @write_authorized can write
        3. All other modules/functions can only read
    
    Attributes:
        _authorized_modules: Set of module names allowed to write.
        _authorized_functions: Set of function names allowed to write.
        _write_log: Log of all write operations.
        _lock: Thread lock for concurrent access.
    
    Example:
        >>> guard = POPSSStateGuard()
        >>> state = guard.read_state("train_001")  # Always allowed
        >>> guard.write_state("train_001", {"status": "running"})  # Only if authorized
    """
    
    AUTHORIZED_MODULES: Set[str] = {
        "opss.run.cli",
        "opss.run.controller",
        "opss.run.store",
        "opss.run.gpu_scheduler",
        "opss.run.queue",
        "opss.run.resources",
        "opss.run.recovery",
        "opss.run.monitor",
        "opss.run.attacher",
    }
    
    AUTHORIZED_FUNCTIONS: Set[str] = {
        "_spawn_and_maybe_attach",
        "_handle_submit",
        "_handle_serve",
        "_handle_control",
        "_handle_gpu",
        "_handle_queue",
        "_handle_resources",
        "_handle_recover",
        "_handle_worker_dataset",
        "_handle_status",
        "_handle_list",
        "init_run",
        "update_state",
        "append_event",
        "append_metric",
        "record_artifact",
        "enqueue_control",
        "poll_controls",
        "allocate",
        "release",
        "enqueue",
        "dequeue",
        "complete",
        "fail",
        "cancel",
        "register_task",
        "restart_task",
        "check_and_recover",
    }
    
    STATE_DIR = ".pisceslx"
    
    PROTECTED_PATTERNS = [
        "runs/*/state.json",
        "runs/*/spec.json",
        "runs/*/metrics.json",
        "runs/*/events.jsonl",
        "runs/*/controls.jsonl",
        "runs/*/artifacts.json",
        "gpu_scheduler/gpu_state.json",
        "queue/queue_state.json",
        "resources/resource_state.json",
        "recovery/recovery_state.json",
    ]
    
    def __init__(self, state_dir: Optional[str] = None):
        """
        Initialize the state guard.
        
        Args:
            state_dir: Base directory for state files.
        """
        self._state_dir = state_dir or self.STATE_DIR
        self._write_log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._session_token: Optional[str] = None
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about the calling module/function."""
        stack = inspect.stack()
        caller_info = {
            "module": "",
            "function": "",
            "file": "",
            "line": 0,
            "authorized": False,
        }
        
        for i, frame_info in enumerate(stack):
            if i < 2:
                continue
            
            frame = frame_info.frame
            module = frame.f_globals.get("__name__", "")
            function = frame_info.function
            
            if module.startswith("opss.run"):
                caller_info["module"] = module
                caller_info["function"] = function
                caller_info["file"] = frame_info.filename
                caller_info["line"] = frame_info.lineno
                
                if module in self.AUTHORIZED_MODULES:
                    caller_info["authorized"] = True
                    break
                
                if function in self.AUTHORIZED_FUNCTIONS:
                    caller_info["authorized"] = True
                    break
        
        return caller_info
    
    def _is_write_authorized(self) -> bool:
        """Check if the current caller is authorized to write."""
        caller_info = self._get_caller_info()
        return caller_info.get("authorized", False)
    
    def _log_access(self, operation: str, path: str, authorized: bool, caller_info: Dict[str, Any]) -> None:
        """Log an access attempt."""
        with self._lock:
            self._write_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operation": operation,
                "path": path,
                "authorized": authorized,
                "caller_module": caller_info.get("module", ""),
                "caller_function": caller_info.get("function", ""),
                "caller_file": caller_info.get("file", ""),
                "caller_line": caller_info.get("line", 0),
            })
    
    def _is_protected_path(self, path: str) -> bool:
        """Check if a path is a protected state file."""
        normalized = str(Path(path).as_posix())
        
        for pattern in self.PROTECTED_PATTERNS:
            pattern_parts = pattern.replace("*", "").split("/")
            path_parts = normalized.split("/")
            
            if len(pattern_parts) > len(path_parts):
                continue
            
            match = True
            for i, pp in enumerate(pattern_parts):
                if pp and path_parts[-(len(pattern_parts) - i)] != pp:
                    match = False
                    break
            
            if match:
                return True
        
        return False
    
    def read_state(self, run_id: str, filename: str = "state.json") -> Optional[Dict[str, Any]]:
        """
        Read state file (always allowed).
        
        Args:
            run_id: Run identifier.
            filename: State file name.
            
        Returns:
            State dictionary or None if not found.
        """
        path = os.path.join(self._state_dir, "runs", run_id, filename)
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def read_gpu_state(self) -> Optional[Dict[str, Any]]:
        """Read GPU scheduler state (read-only for external modules)."""
        path = os.path.join(self._state_dir, "gpu_scheduler", "gpu_state.json")
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def read_queue_state(self) -> Optional[Dict[str, Any]]:
        """Read task queue state (read-only for external modules)."""
        path = os.path.join(self._state_dir, "queue", "queue_state.json")
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def read_resources_state(self) -> Optional[Dict[str, Any]]:
        """Read resources state (read-only for external modules)."""
        path = os.path.join(self._state_dir, "resources", "resource_state.json")
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def read_recovery_state(self) -> Optional[Dict[str, Any]]:
        """Read recovery state (read-only for external modules)."""
        path = os.path.join(self._state_dir, "recovery", "recovery_state.json")
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def write_state(
        self,
        run_id: str,
        data: Dict[str, Any],
        filename: str = "state.json",
    ) -> str:
        """
        Write state file (authorized only).
        
        Args:
            run_id: Run identifier.
            data: State data to write.
            filename: State file name.
            
        Returns:
            Path to written file.
            
        Raises:
            POPSSUnauthorizedWriteError: If caller is not authorized.
        """
        path = os.path.join(self._state_dir, "runs", run_id, filename)
        
        caller_info = self._get_caller_info()
        
        if not self._is_write_authorized():
            self._log_access("write", path, False, caller_info)
            raise POPSSUnauthorizedWriteError(
                f"Unauthorized write attempt to {path} by {caller_info.get('module', 'unknown')}.{caller_info.get('function', 'unknown')}. "
                f"Only action system modules can modify state files."
            )
        
        self._log_access("write", path, True, caller_info)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data["_guard_signature"] = self._sign_data(data)
        data["_guard_updated_at"] = datetime.now(timezone.utc).isoformat()
        data["_guard_updated_by"] = caller_info.get("module", "unknown")
        
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        
        return path
    
    def write_gpu_state(self, data: Dict[str, Any]) -> str:
        """Write GPU scheduler state (authorized only)."""
        path = os.path.join(self._state_dir, "gpu_scheduler", "gpu_state.json")
        return self._write_protected_state(path, data)
    
    def write_queue_state(self, data: Dict[str, Any]) -> str:
        """Write task queue state (authorized only)."""
        path = os.path.join(self._state_dir, "queue", "queue_state.json")
        return self._write_protected_state(path, data)
    
    def write_resources_state(self, data: Dict[str, Any]) -> str:
        """Write resources state (authorized only)."""
        path = os.path.join(self._state_dir, "resources", "resource_state.json")
        return self._write_protected_state(path, data)
    
    def write_recovery_state(self, data: Dict[str, Any]) -> str:
        """Write recovery state (authorized only)."""
        path = os.path.join(self._state_dir, "recovery", "recovery_state.json")
        return self._write_protected_state(path, data)
    
    def _write_protected_state(self, path: str, data: Dict[str, Any]) -> str:
        """Write to a protected state file with authorization check."""
        caller_info = self._get_caller_info()
        
        if not self._is_write_authorized():
            self._log_access("write", path, False, caller_info)
            raise POPSSUnauthorizedWriteError(
                f"Unauthorized write attempt to {path} by {caller_info.get('module', 'unknown')}.{caller_info.get('function', 'unknown')}. "
                f"Only action system modules can modify state files."
            )
        
        self._log_access("write", path, True, caller_info)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data["_guard_signature"] = self._sign_data(data)
        data["_guard_updated_at"] = datetime.now(timezone.utc).isoformat()
        data["_guard_updated_by"] = caller_info.get("module", "unknown")
        
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        
        return path
    
    def _sign_data(self, data: Dict[str, Any]) -> str:
        """Create a signature for data integrity verification."""
        sign_data = {k: v for k, v in data.items() if not k.startswith("_guard_")}
        content = json.dumps(sign_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    
    def verify_integrity(self, path: str) -> bool:
        """
        Verify the integrity of a state file.
        
        Args:
            path: Path to state file.
            
        Returns:
            True if file integrity is valid.
        """
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            stored_signature = data.pop("_guard_signature", None)
            if not stored_signature:
                return False
            
            computed_signature = self._sign_data(data)
            return stored_signature == computed_signature
        except (json.JSONDecodeError, IOError):
            return False
    
    def get_access_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent access log entries.
        
        Args:
            limit: Maximum number of entries to return.
            
        Returns:
            List of access log entries.
        """
        with self._lock:
            return self._write_log[-limit:]
    
    def get_unauthorized_attempts(self) -> List[Dict[str, Any]]:
        """Get all unauthorized access attempts."""
        with self._lock:
            return [entry for entry in self._write_log if not entry.get("authorized", True)]
    
    def clear_access_log(self) -> None:
        """Clear the access log."""
        with self._lock:
            self._write_log = []


def write_authorized(func: Callable) -> Callable:
    """
    Decorator to mark a function as authorized to write state files.
    
    Functions decorated with this decorator will be allowed to write
    to protected state files regardless of their module.
    
    Example:
        @write_authorized
        def my_custom_writer(run_id, data):
            guard = POPSSStateGuard()
            return guard.write_state(run_id, data)
    """
    POPSSStateGuard.AUTHORIZED_FUNCTIONS.add(func.__name__)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper


def readonly_access(func: Callable) -> Callable:
    """
    Decorator to enforce read-only access to state files.
    
    Functions decorated with this decorator will only be able to
    read state files, even if they are in an authorized module.
    
    Example:
        @readonly_access
        def get_task_status(run_id):
            guard = POPSSStateGuard()
            return guard.read_state(run_id)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        guard = POPSSStateGuard()
        caller_info = guard._get_caller_info()
        caller_info["authorized"] = False
        
        original_authorized = POPSSStateGuard.AUTHORIZED_FUNCTIONS.copy()
        try:
            if func.__name__ in POPSSStateGuard.AUTHORIZED_FUNCTIONS:
                POPSSStateGuard.AUTHORIZED_FUNCTIONS.discard(func.__name__)
            return func(*args, **kwargs)
        finally:
            POPSSStateGuard.AUTHORIZED_FUNCTIONS.update(original_authorized)
    
    return wrapper


class POPSSStateReader:
    """
    Read-only state reader for external modules.
    
    This class provides a safe, read-only interface for external modules
    to access action system state files. All write methods are disabled.
    
    Example:
        >>> reader = POPSSStateReader()
        >>> state = reader.get_run_state("train_001")
        >>> gpu_info = reader.get_gpu_allocations()
    """
    
    def __init__(self, state_dir: Optional[str] = None):
        self._guard = POPSSStateGuard(state_dir=state_dir)
    
    def get_run_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run state (read-only)."""
        return self._guard.read_state(run_id, "state.json")
    
    def get_run_spec(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run specification (read-only)."""
        return self._guard.read_state(run_id, "spec.json")
    
    def get_run_metrics(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run metrics (read-only)."""
        return self._guard.read_state(run_id, "metrics.json")
    
    def get_gpu_allocations(self) -> Optional[Dict[str, Any]]:
        """Get GPU allocation state (read-only)."""
        return self._guard.read_gpu_state()
    
    def get_queue_state(self) -> Optional[Dict[str, Any]]:
        """Get task queue state (read-only)."""
        return self._guard.read_queue_state()
    
    def get_resources_state(self) -> Optional[Dict[str, Any]]:
        """Get resources state (read-only)."""
        return self._guard.read_resources_state()
    
    def get_recovery_state(self) -> Optional[Dict[str, Any]]:
        """Get recovery state (read-only)."""
        return self._guard.read_recovery_state()
    
    def list_runs(self) -> List[str]:
        """List all run IDs (read-only)."""
        runs_dir = os.path.join(self._guard._state_dir, "runs")
        if not os.path.exists(runs_dir):
            return []
        
        run_ids = []
        for name in os.listdir(runs_dir):
            if os.path.isdir(os.path.join(runs_dir, name)):
                run_ids.append(name)
        
        return sorted(run_ids)
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all run states (read-only)."""
        result = {}
        for run_id in self.list_runs():
            state = self.get_run_state(run_id)
            if state:
                result[run_id] = state
        return result
