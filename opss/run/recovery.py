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
Auto-Recovery System for LLM Training Tasks.

This module provides automatic crash detection and recovery for LLM training
tasks, including checkpoint discovery, automatic restart, and restart limits.

Key Features:
    - Crash detection via process monitoring
    - Checkpoint discovery and validation
    - Automatic restart with resume_ckpt
    - Configurable restart limits
    - Recovery state persistence

Usage:
    from opss.run.recovery import POPSSTaskRecovery
    
    recovery = POPSSTaskRecovery(controller)
    
    if recovery.check_crashed("train_001"):
        checkpoint = recovery.find_latest_checkpoint("train_001")
        recovery.restart_task("train_001", checkpoint)
"""

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .controller import POPSSRunController
from .store import POPSSRunStore


@dataclass
class POPSSCheckpointInfo:
    """Checkpoint information structure."""
    checkpoint_path: str
    step: int
    epoch: int
    loss: float
    created_at: str
    size_mb: float
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class POPSSRecoveryState:
    """Recovery state for a task."""
    task_id: str
    original_run_id: str
    restart_count: int = 0
    max_restarts: int = 3
    last_checkpoint: Optional[str] = None
    last_restart_at: Optional[str] = None
    recovery_enabled: bool = True
    recovery_history: List[Dict[str, Any]] = field(default_factory=list)


class POPSSTaskRecovery:
    """
    Auto-Recovery System for LLM Training Tasks.
    
    This class provides automatic crash detection and recovery capabilities
    for LLM training workloads, ensuring minimal progress loss.
    
    Attributes:
        _controller: Run controller instance.
        _recovery_states: Dictionary mapping task ID to recovery state.
        _lock: Thread lock for concurrent access.
        _state_file: Path to persistent state file.
        _checkpoint_patterns: Regex patterns for checkpoint discovery.
    
    Example:
        >>> recovery = POPSSTaskRecovery(controller)
        >>> if recovery.check_and_recover("train_001"):
        ...     print("Task recovered successfully")
    """
    
    DEFAULT_CHECKPOINT_PATTERNS = [
        r"checkpoint[-_]?(\d+)",
        r"ckpt[-_]?(\d+)",
        r"step[-_]?(\d+)",
        r"epoch[-_]?(\d+)",
        r"model[-_]?(\d+)",
    ]
    
    def __init__(
        self,
        controller: Optional[POPSSRunController] = None,
        state_dir: Optional[str] = None,
        checkpoint_dirs: Optional[List[str]] = None,
    ):
        """
        Initialize the recovery system.
        
        Args:
            controller: Run controller instance.
            state_dir: Directory for persistent state storage.
            checkpoint_dirs: List of checkpoint directories to search.
        """
        self._controller = controller
        self._recovery_states: Dict[str, POPSSRecoveryState] = {}
        self._lock = threading.RLock()
        
        if state_dir is None:
            state_dir = str(Path(".pisceslx") / "recovery")
        self._state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self._state_file = str(Path(state_dir) / "recovery_state.json")
        
        if checkpoint_dirs is None:
            checkpoint_dirs = [
                ".pisceslx/checkpoints",
                "checkpoints",
                "outputs/checkpoints",
            ]
        self._checkpoint_dirs = checkpoint_dirs
        self._checkpoint_patterns = self.DEFAULT_CHECKPOINT_PATTERNS
        
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persistent state from disk."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for task_id, state_data in data.get("recovery_states", {}).items():
                self._recovery_states[task_id] = POPSSRecoveryState(
                    task_id=state_data["task_id"],
                    original_run_id=state_data["original_run_id"],
                    restart_count=state_data.get("restart_count", 0),
                    max_restarts=state_data.get("max_restarts", 3),
                    last_checkpoint=state_data.get("last_checkpoint"),
                    last_restart_at=state_data.get("last_restart_at"),
                    recovery_enabled=state_data.get("recovery_enabled", True),
                    recovery_history=state_data.get("recovery_history", []),
                )
        except Exception:
            pass
    
    def _save_state(self) -> None:
        """Save persistent state to disk."""
        data = {
            "recovery_states": {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        for task_id, state in self._recovery_states.items():
            data["recovery_states"][task_id] = {
                "task_id": state.task_id,
                "original_run_id": state.original_run_id,
                "restart_count": state.restart_count,
                "max_restarts": state.max_restarts,
                "last_checkpoint": state.last_checkpoint,
                "last_restart_at": state.last_restart_at,
                "recovery_enabled": state.recovery_enabled,
                "recovery_history": state.recovery_history,
            }
        
        tmp_file = f"{self._state_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, self._state_file)
    
    def register_task(
        self,
        task_id: str,
        run_id: str,
        max_restarts: int = 3,
        recovery_enabled: bool = True,
    ) -> POPSSRecoveryState:
        """
        Register a task for recovery tracking.
        
        Args:
            task_id: Task identifier.
            run_id: Run ID for the task.
            max_restarts: Maximum number of restart attempts.
            recovery_enabled: Whether auto-recovery is enabled.
            
        Returns:
            Recovery state for the task.
        """
        with self._lock:
            state = POPSSRecoveryState(
                task_id=task_id,
                original_run_id=run_id,
                max_restarts=max_restarts,
                recovery_enabled=recovery_enabled,
            )
            self._recovery_states[task_id] = state
            self._save_state()
            return state
    
    def unregister_task(self, task_id: str) -> bool:
        """
        Unregister a task from recovery tracking.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            True if task was unregistered.
        """
        with self._lock:
            if task_id in self._recovery_states:
                del self._recovery_states[task_id]
                self._save_state()
                return True
            return False
    
    def get_recovery_state(self, task_id: str) -> Optional[POPSSRecoveryState]:
        """
        Get recovery state for a task.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            Recovery state or None if not registered.
        """
        with self._lock:
            return self._recovery_states.get(task_id)
    
    def check_crashed(self, task_id: str, store: Optional[POPSSRunStore] = None) -> bool:
        """
        Check if a task has crashed.
        
        Args:
            task_id: Task identifier.
            store: Optional store to check state.
            
        Returns:
            True if task has crashed.
        """
        if store is None and self._controller:
            store = self._controller.store
        
        if store is None:
            return False
        
        state = store.read_state()
        if not state:
            return False
        
        status = state.get("status", "")
        phase = state.get("phase", "")
        pid = state.get("pid")
        
        if status == "dead" or phase == "process_lost":
            return True
        
        if status == "running" and pid:
            try:
                os.kill(int(pid), 0)
            except (OSError, ProcessLookupError):
                return True
        
        return False
    
    def find_latest_checkpoint(
        self,
        task_id: str,
        checkpoint_dir: Optional[str] = None,
        validate: bool = True,
    ) -> Optional[POPSSCheckpointInfo]:
        """
        Find the latest valid checkpoint for a task.
        
        Args:
            task_id: Task identifier.
            checkpoint_dir: Specific checkpoint directory to search.
            validate: Whether to validate checkpoint integrity.
            
        Returns:
            Checkpoint info or None if not found.
        """
        search_dirs = [checkpoint_dir] if checkpoint_dir else self._checkpoint_dirs
        
        checkpoints = []
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            
            task_checkpoint_dir = os.path.join(search_dir, task_id)
            if os.path.exists(task_checkpoint_dir):
                checkpoints.extend(self._scan_checkpoint_dir(task_checkpoint_dir, validate))
            
            checkpoints.extend(self._scan_checkpoint_dir(search_dir, validate, task_id))
        
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda c: c.step, reverse=True)
        return checkpoints[0]
    
    def _scan_checkpoint_dir(
        self,
        directory: str,
        validate: bool,
        task_filter: Optional[str] = None,
    ) -> List[POPSSCheckpointInfo]:
        """Scan a directory for checkpoints."""
        checkpoints = []
        
        try:
            for name in os.listdir(directory):
                if task_filter and task_filter not in name:
                    continue
                
                path = os.path.join(directory, name)
                if not os.path.isdir(path):
                    continue
                
                step = self._extract_step(name)
                if step is None:
                    continue
                
                checkpoint_info = self._create_checkpoint_info(path, step, validate)
                if checkpoint_info:
                    checkpoints.append(checkpoint_info)
        except Exception:
            pass
        
        return checkpoints
    
    def _extract_step(self, name: str) -> Optional[int]:
        """Extract step number from checkpoint name."""
        for pattern in self._checkpoint_patterns:
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None
    
    def _create_checkpoint_info(
        self,
        path: str,
        step: int,
        validate: bool,
    ) -> Optional[POPSSCheckpointInfo]:
        """Create checkpoint info from path."""
        try:
            stat = os.stat(path)
            size_mb = stat.st_size / (1024 * 1024) if os.path.isfile(path) else 0
            
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for f in files:
                        size_mb += os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
            
            metadata = {}
            meta_file = os.path.join(path, "metadata.json")
            if os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            
            is_valid = True
            if validate:
                is_valid = self._validate_checkpoint(path)
            
            return POPSSCheckpointInfo(
                checkpoint_path=path,
                step=step,
                epoch=metadata.get("epoch", 0),
                loss=metadata.get("loss", 0.0),
                created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
                size_mb=size_mb,
                is_valid=is_valid,
                metadata=metadata,
            )
        except Exception:
            return None
    
    def _validate_checkpoint(self, path: str) -> bool:
        """Validate checkpoint integrity."""
        required_files = ["pytorch_model.bin", "model.safetensors", "config.json"]
        index_files = ["pytorch_model.bin.index.json", "model.safetensors.index.json"]
        
        if os.path.isfile(path):
            return path.endswith((".bin", ".safetensors", ".pt", ".pth"))
        
        for f in required_files + index_files:
            if os.path.exists(os.path.join(path, f)):
                return True
        
        if os.path.exists(os.path.join(path, "checkpoint.pt")):
            return True
        
        return False
    
    def can_restart(self, task_id: str) -> Tuple[bool, str]:
        """
        Check if a task can be restarted.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            Tuple of (can_restart, reason).
        """
        with self._lock:
            state = self._recovery_states.get(task_id)
            if not state:
                return False, "Task not registered for recovery"
            
            if not state.recovery_enabled:
                return False, "Recovery disabled for this task"
            
            if state.restart_count >= state.max_restarts:
                return False, f"Max restarts ({state.max_restarts}) exceeded"
            
            return True, "Ready to restart"
    
    def restart_task(
        self,
        task_id: str,
        checkpoint: Optional[POPSSCheckpointInfo] = None,
        store: Optional[POPSSRunStore] = None,
    ) -> Tuple[bool, str]:
        """
        Restart a crashed task from checkpoint.
        
        Args:
            task_id: Task identifier.
            checkpoint: Checkpoint to resume from.
            store: Optional store for the task.
            
        Returns:
            Tuple of (success, message).
        """
        with self._lock:
            can_restart, reason = self.can_restart(task_id)
            if not can_restart:
                return False, reason
            
            state = self._recovery_states[task_id]
            
            if checkpoint is None:
                checkpoint = self.find_latest_checkpoint(task_id)
            
            if checkpoint is None:
                return False, "No valid checkpoint found"
            
            if not checkpoint.is_valid:
                return False, "Checkpoint validation failed"
            
            recovery_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "checkpoint": checkpoint.checkpoint_path,
                "step": checkpoint.step,
                "restart_number": state.restart_count + 1,
            }
            
            state.restart_count += 1
            state.last_checkpoint = checkpoint.checkpoint_path
            state.last_restart_at = recovery_record["timestamp"]
            state.recovery_history.append(recovery_record)
            
            self._save_state()
            
            return True, f"Task ready to restart from step {checkpoint.step}"
    
    def check_and_recover(
        self,
        task_id: str,
        store: Optional[POPSSRunStore] = None,
    ) -> Tuple[bool, str]:
        """
        Check if task crashed and attempt recovery.
        
        Args:
            task_id: Task identifier.
            store: Optional store for the task.
            
        Returns:
            Tuple of (recovered, message).
        """
        if not self.check_crashed(task_id, store):
            return False, "Task is not crashed"
        
        checkpoint = self.find_latest_checkpoint(task_id)
        if checkpoint is None:
            return False, "No checkpoint available for recovery"
        
        return self.restart_task(task_id, checkpoint, store)
    
    def update_checkpoint(
        self,
        task_id: str,
        checkpoint_path: str,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update the last known checkpoint for a task.
        
        Args:
            task_id: Task identifier.
            checkpoint_path: Path to the checkpoint.
            step: Training step number.
            metadata: Optional checkpoint metadata.
        """
        with self._lock:
            state = self._recovery_states.get(task_id)
            if state:
                state.last_checkpoint = checkpoint_path
                self._save_state()
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        Get recovery statistics.
        
        Returns:
            Dictionary with recovery statistics.
        """
        with self._lock:
            total_tasks = len(self._recovery_states)
            total_restarts = sum(s.restart_count for s in self._recovery_states.values())
            tasks_at_limit = sum(
                1 for s in self._recovery_states.values()
                if s.restart_count >= s.max_restarts
            )
            
            return {
                "total_tasks": total_tasks,
                "total_restarts": total_restarts,
                "tasks_at_limit": tasks_at_limit,
                "recovery_enabled_tasks": sum(
                    1 for s in self._recovery_states.values() if s.recovery_enabled
                ),
            }
    
    def get_all_recovery_states(self) -> Dict[str, POPSSRecoveryState]:
        """
        Get all recovery states.
        
        Returns:
            Dictionary mapping task ID to recovery state.
        """
        with self._lock:
            return dict(self._recovery_states)
    
    def reset_restart_count(self, task_id: str) -> bool:
        """
        Reset the restart count for a task.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            True if reset was successful.
        """
        with self._lock:
            state = self._recovery_states.get(task_id)
            if state:
                state.restart_count = 0
                self._save_state()
                return True
            return False
    
    def set_recovery_enabled(self, task_id: str, enabled: bool) -> bool:
        """
        Enable or disable recovery for a task.
        
        Args:
            task_id: Task identifier.
            enabled: Whether recovery is enabled.
            
        Returns:
            True if setting was changed.
        """
        with self._lock:
            state = self._recovery_states.get(task_id)
            if state:
                state.recovery_enabled = enabled
                self._save_state()
                return True
            return False
    
    def cleanup_old_states(self, active_task_ids: List[str]) -> int:
        """
        Remove recovery states for inactive tasks.
        
        Args:
            active_task_ids: List of currently active task IDs.
            
        Returns:
            Number of states removed.
        """
        with self._lock:
            to_remove = [
                task_id for task_id in self._recovery_states
                if task_id not in active_task_ids
            ]
            for task_id in to_remove:
                del self._recovery_states[task_id]
            
            if to_remove:
                self._save_state()
            
            return len(to_remove)
    
    def scan_all_checkpoints(self) -> Dict[str, List[POPSSCheckpointInfo]]:
        """
        Scan all checkpoint directories for available checkpoints.
        
        Returns:
            Dictionary mapping task ID to list of checkpoints.
        """
        result = {}
        
        for search_dir in self._checkpoint_dirs:
            if not os.path.exists(search_dir):
                continue
            
            for name in os.listdir(search_dir):
                path = os.path.join(search_dir, name)
                if not os.path.isdir(path):
                    continue
                
                step = self._extract_step(name)
                if step is None:
                    continue
                
                task_id = name.split("_")[0] if "_" in name else name
                
                checkpoint_info = self._create_checkpoint_info(path, step, validate=True)
                if checkpoint_info:
                    if task_id not in result:
                        result[task_id] = []
                    result[task_id].append(checkpoint_info)
        
        for task_id in result:
            result[task_id].sort(key=lambda c: c.step, reverse=True)
        
        return result
