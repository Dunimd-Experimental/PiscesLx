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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
Run Controller - Process Lifecycle and State Management

This module provides a controller for managing run lifecycle, state transitions,
event logging, and process spawning in the PiscesL1 execution framework.

Key Components:
    - POPSSRunController: Main controller class for run management

Features:
    - Run initialization with spec and state
    - Event and metric logging
    - State updates with timestamps
    - Control queue for external commands
    - Process spawning (daemon and foreground)
    - Process lifecycle management (kill, check)

Usage:
    from opss.run.store import POPSSRunStore
    from opss.run.controller import POPSSRunController
    
    store = POPSSRunStore(run_id="run_123", base_dir="./runs")
    controller = POPSSRunController(store)
    
    # Initialize run
    controller.init_run({"model": "ruchbah", "task": "train"})
    
    # Log events and metrics
    controller.append_event("training_started", level="info")
    controller.append_metric({"loss": 0.5, "step": 100})
    
    # Update state
    controller.update_state({"status": "running", "phase": "training"})
    
    # Spawn process
    pid = controller.spawn(["python", "train.py"], daemon=True)

Thread Safety:
    File operations are atomic within single calls. For concurrent access,
    external synchronization may be required.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .store import POPSSRunStore


class POPSSRunController:
    """
    Controller for managing run lifecycle, state, and process execution.
    
    This class provides comprehensive control over run execution, including
    initialization, state management, event logging, and process spawning.
    
    Attributes:
        _store: POPSSRunStore instance for persistent storage.
    
    State Structure:
        - run_id: Unique identifier for the run
        - status: Current status (pending, running, completed, failed)
        - phase: Current execution phase (init, spawned, training, etc.)
        - pid: Process ID of spawned process
        - created_at: ISO timestamp of run creation
        - updated_at: ISO timestamp of last update
        - last_control_seq: Sequence number of last processed control
    
    Example:
        >>> store = POPSSRunStore(run_id="run_abc", base_dir="./runs")
        >>> controller = POPSSRunController(store)
        >>> controller.init_run({"model": "ruchbah"})
        >>> controller.append_event("started", level="info")
        >>> controller.update_state({"status": "running"})
    """
    
    def __init__(self, store: POPSSRunStore):
        """
        Initialize the run controller.
        
        Args:
            store: POPSSRunStore instance for persistent storage.
        """
        self._store = store
        self._gpu_scheduler = None
        self._resource_limits = None

    @property
    def store(self) -> POPSSRunStore:
        """
        Get the underlying store instance.
        
        Returns:
            POPSSRunStore: The store instance.
        """
        return self._store

    def _get_gpu_scheduler(self):
        """Get or create GPU scheduler instance."""
        if self._gpu_scheduler is None:
            from .gpu_scheduler import POPSSGPUScheduler
            self._gpu_scheduler = POPSSGPUScheduler()
        return self._gpu_scheduler

    def _get_resource_limits(self):
        """Get or create resource limits instance."""
        if self._resource_limits is None:
            from .resources import POPSSResourceLimits
            self._resource_limits = POPSSResourceLimits()
        return self._resource_limits

    def allocate_gpus(
        self,
        gpu_count: int = 1,
        min_memory_mb: int = 0,
        priority: str = "normal",
    ) -> List[int]:
        """
        Allocate GPUs for this run.
        
        Args:
            gpu_count: Number of GPUs to allocate.
            min_memory_mb: Minimum memory per GPU.
            priority: Allocation priority.
            
        Returns:
            List of allocated GPU IDs.
        """
        scheduler = self._get_gpu_scheduler()
        return scheduler.allocate(
            self._store.run_id,
            count=gpu_count,
            min_memory_mb=min_memory_mb,
            priority=priority,
        )

    def release_gpus(self) -> List[int]:
        """
        Release GPUs allocated to this run.
        
        Returns:
            List of released GPU IDs.
        """
        scheduler = self._get_gpu_scheduler()
        return scheduler.release(self._store.run_id)

    def get_allocated_gpus(self) -> List[int]:
        """
        Get GPUs allocated to this run.
        
        Returns:
            List of allocated GPU IDs.
        """
        scheduler = self._get_gpu_scheduler()
        return scheduler.get_task_gpus(self._store.run_id)

    @staticmethod
    def now_iso() -> str:
        """
        Get current timestamp in ISO 8601 format with UTC timezone.
        
        Returns:
            str: ISO formatted timestamp string.
        """
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def new_seq() -> int:
        """
        Generate a new sequence number using nanosecond timestamp.
        
        Returns:
            int: Unique sequence number.
        """
        return time.time_ns()

    def init_run(self, spec: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a new run with specification and initial state.
        
        Args:
            spec: Run specification dictionary (model, task, config, etc.).
            state: Optional initial state overrides.
        """
        base_state = {
            "run_id": self._store.run_id,
            "status": "pending",
            "phase": "init",
            "pid": None,
            "created_at": self.now_iso(),
            "updated_at": self.now_iso(),
            "last_control_seq": 0,
        }
        if state:
            base_state.update(state)
        self._store.write_spec(spec or {"run_id": self._store.run_id})
        self._store.write_state(base_state)
        self.append_event("run_initialized", payload={"spec": spec or {}})

    def append_event(self, event_type: str, level: str = "info", payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Append an event to the run's event log.
        
        Args:
            event_type: Type of event (e.g., "training_started", "checkpoint_saved").
            level: Log level (debug, info, warning, error).
            payload: Optional dictionary with event details.
        """
        obj = {
            "ts": self.now_iso(),
            "run_id": self._store.run_id,
            "type": str(event_type),
            "level": str(level),
            "payload": payload or {},
        }
        self._store.append_event(obj)

    def append_metric(self, payload: Dict[str, Any]) -> None:
        """
        Append a metric record to the run's metric log.
        
        Args:
            payload: Dictionary of metric values (e.g., {"loss": 0.5, "accuracy": 0.95}).
        """
        obj = {
            "ts": self.now_iso(),
            "run_id": self._store.run_id,
            **(payload or {}),
        }
        self._store.append_metric(obj)

    def update_state(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update run state with a partial patch.
        
        Args:
            patch: Dictionary of state fields to update.
        
        Returns:
            Dict[str, Any]: The updated state dictionary.
        """
        p = dict(patch or {})
        p["updated_at"] = self.now_iso()
        return self._store.merge_state(p)

    def record_artifact(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Record an artifact in the run's artifact store.
        
        Args:
            key: Artifact identifier.
            value: Artifact value (will be JSON serialized).
        
        Returns:
            Dict[str, Any]: The updated artifacts dictionary.
        """
        return self._store.merge_artifacts({str(key): value})

    def enqueue_control(self, action: str, payload: Optional[Dict[str, Any]] = None) -> int:
        """
        Enqueue a control command for the run.
        
        Args:
            action: Control action (e.g., "stop", "pause", "resume").
            payload: Optional dictionary with action parameters.
        
        Returns:
            int: Sequence number of the enqueued control.
        """
        seq = self.new_seq()
        obj = {
            "ts": self.now_iso(),
            "run_id": self._store.run_id,
            "seq": seq,
            "action": str(action),
            "payload": payload or {},
        }
        self._store.append_control(obj)
        self.append_event("control_enqueued", payload={"action": action, "seq": seq})
        return seq

    def poll_controls(self, last_seq: int) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Poll for control commands since a given sequence number.
        
        Args:
            last_seq: Last processed sequence number.
        
        Returns:
            Tuple[int, List[Dict]]: Tuple of (max_seq, list of control commands).
        """
        p = self._store.path("control")
        if not os.path.exists(p):
            return last_seq, []
        out: List[Dict[str, Any]] = []
        max_seq = int(last_seq or 0)
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    seq = int(obj.get("seq") or 0)
                    if seq <= max_seq:
                        continue
                    out.append(obj)
                    if seq > max_seq:
                        max_seq = seq
        except Exception:
            return last_seq, []
        return max_seq, out

    def spawn(
        self,
        cmd: List[str],
        daemon: bool,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Spawn a new process for the run.
        
        Args:
            cmd: Command and arguments to execute.
            daemon: If True, spawn as detached daemon process.
            cwd: Working directory for the process.
            env: Environment variables for the process.
        
        Returns:
            int: Process ID of the spawned process.
        """
        self._store.ensure()
        stdout_path = self._store.path("stdout")
        with open(stdout_path, "a", encoding="utf-8") as log:
            if daemon:
                # Daemon process - detached from parent
                if sys.platform == "win32":
                    p = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=log,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                        cwd=cwd or os.getcwd(),
                        env=env,
                    )
                else:
                    p = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                        cwd=cwd or os.getcwd(),
                        env=env,
                    )
                pid = int(p.pid)
                self.update_state({"pid": pid, "status": "running", "phase": "spawned"})
                self.append_event("process_spawned", payload={"pid": pid, "cmd": cmd, "daemon": True})
                return pid
            # Foreground process - same PID as current
            pid = int(os.getpid())
            self.update_state({"pid": pid, "status": "running", "phase": "foreground"})
            self.append_event("process_foreground", payload={"pid": pid, "cmd": cmd, "daemon": False})
            return pid

    def pid_exists(self, pid: int) -> bool:
        """
        Check if a process with given PID exists.
        
        Args:
            pid: Process ID to check.
        
        Returns:
            bool: True if process exists, False otherwise.
        """
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False

    def kill_pid(self, pid: int, force: bool = True) -> bool:
        """
        Terminate a process by PID.
        
        Args:
            pid: Process ID to terminate.
            force: If True, force kill after graceful termination fails.
        
        Returns:
            bool: True if termination succeeded, False otherwise.
        """
        try:
            pid = int(pid)
            if pid <= 0:
                return False
            if sys.platform == "win32":
                import ctypes
                try:
                    kernel32 = ctypes.windll.kernel32
                    PROCESS_TERMINATE = 0x0001
                    handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                    if handle:
                        result = kernel32.TerminateProcess(handle, 1)
                        kernel32.CloseHandle(handle)
                        return bool(result)
                except Exception:
                    pass
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/F"],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                    )
                except Exception:
                    pass
                return True
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
            if not force:
                return True
            for _ in range(10):
                if not self.pid_exists(pid):
                    return True
                time.sleep(0.2)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
            return True
        except Exception:
            return False

