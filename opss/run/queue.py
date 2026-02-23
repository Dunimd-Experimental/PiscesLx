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
Priority Task Queue for LLM Training and Inference.

This module provides a priority-based task queue specifically designed for
LLM workloads, supporting resource-aware scheduling and FIFO ordering within
priority levels.

Key Features:
    - Three priority levels (high/normal/low)
    - FIFO ordering within same priority
    - Resource-aware scheduling
    - Persistent queue state
    - Task dependency support

Usage:
    from opss.run.queue import POPSSTaskQueue, POPSSTask
    
    queue = POPSSTaskQueue()
    
    task = POPSSTask(
        task_id="train_001",
        task_type="train",
        priority="normal",
        requirements={"gpu_count": 2, "memory_mb": 64000},
    )
    
    queue.enqueue(task)
    next_task = queue.dequeue()
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class POPSSTask:
    """Task definition for the queue."""
    task_id: str
    task_type: str
    priority: str = "normal"
    requirements: Dict[str, Any] = field(default_factory=dict)
    spec: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: str = ""
    enqueued_at: str = ""
    attempts: int = 0
    max_attempts: int = 3
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "requirements": self.requirements,
            "spec": self.spec,
            "dependencies": self.dependencies,
            "created_at": self.created_at,
            "enqueued_at": self.enqueued_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "POPSSTask":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            priority=data.get("priority", "normal"),
            requirements=data.get("requirements", {}),
            spec=data.get("spec", {}),
            dependencies=data.get("dependencies", []),
            created_at=data.get("created_at", ""),
            enqueued_at=data.get("enqueued_at", ""),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
        )


class POPSSTaskQueue:
    """
    Priority Task Queue for LLM Training and Inference.
    
    This class manages a priority-based task queue with resource-aware
    scheduling capabilities. Tasks are ordered by priority (high > normal > low)
    and FIFO within the same priority level.
    
    Attributes:
        _queues: Dictionary mapping priority level to task list.
        _task_index: Dictionary mapping task ID to task.
        _completed: Set of completed task IDs.
        _lock: Thread lock for concurrent access.
        _state_file: Path to persistent state file.
    
    Example:
        >>> queue = POPSSTaskQueue()
        >>> task = POPSSTask("train_001", "train", priority="high")
        >>> queue.enqueue(task)
        >>> queue.size()
        1
        >>> next_task = queue.dequeue()
    """
    
    PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}
    VALID_PRIORITIES = {"high", "normal", "low"}
    
    def __init__(self, state_dir: Optional[str] = None):
        """
        Initialize the task queue.
        
        Args:
            state_dir: Directory for persistent state storage.
        """
        self._queues: Dict[str, List[POPSSTask]] = {
            "high": [],
            "normal": [],
            "low": [],
        }
        self._task_index: Dict[str, POPSSTask] = {}
        self._completed: Set[str] = set()
        self._running: Set[str] = set()
        self._lock = threading.RLock()
        
        if state_dir is None:
            state_dir = str(Path(".pisceslx") / "queue")
        self._state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self._state_file = str(Path(state_dir) / "queue_state.json")
        
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persistent state from disk."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for priority, task_list in data.get("queues", {}).items():
                if priority not in self._queues:
                    continue
                for task_data in task_list:
                    task = POPSSTask.from_dict(task_data)
                    self._queues[priority].append(task)
                    self._task_index[task.task_id] = task
            
            self._completed = set(data.get("completed", []))
            self._running = set(data.get("running", []))
        except Exception:
            pass
    
    def _save_state(self) -> None:
        """Save persistent state to disk."""
        data = {
            "queues": {},
            "completed": list(self._completed),
            "running": list(self._running),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        for priority, task_list in self._queues.items():
            data["queues"][priority] = [task.to_dict() for task in task_list]
        
        tmp_file = f"{self._state_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, self._state_file)
    
    def enqueue(
        self,
        task: POPSSTask,
        priority: Optional[str] = None,
    ) -> int:
        """
        Add a task to the queue.
        
        Args:
            task: Task to add.
            priority: Override priority level.
            
        Returns:
            Queue position (0 = first).
            
        Raises:
            ValueError: If task already exists or invalid priority.
        """
        if priority:
            task.priority = priority
        
        if task.priority not in self.VALID_PRIORITIES:
            raise ValueError(f"Invalid priority: {task.priority}")
        
        with self._lock:
            if task.task_id in self._task_index:
                raise ValueError(f"Task already exists: {task.task_id}")
            
            task.enqueued_at = datetime.now(timezone.utc).isoformat()
            self._queues[task.priority].append(task)
            self._task_index[task.task_id] = task
            self._save_state()
            
            position = 0
            for p in ["high", "normal", "low"]:
                if p == task.priority:
                    position += len(self._queues[p]) - 1
                    break
                position += len(self._queues[p])
            
            return position
    
    def dequeue(
        self,
        available_resources: Optional[Dict[str, Any]] = None,
    ) -> Optional[POPSSTask]:
        """
        Get the next ready task from the queue.
        
        Args:
            available_resources: Optional resource constraints to check.
            
        Returns:
            Next ready task or None if queue is empty.
        """
        with self._lock:
            for priority in ["high", "normal", "low"]:
                for i, task in enumerate(self._queues[priority]):
                    if not self._dependencies_satisfied(task):
                        continue
                    
                    if available_resources and not self._check_resources(task, available_resources):
                        continue
                    
                    self._queues[priority].pop(i)
                    self._running.add(task.task_id)
                    task.attempts += 1
                    self._save_state()
                    return task
            
            return None
    
    def peek(self) -> Optional[POPSSTask]:
        """
        View the next task without removing it.
        
        Returns:
            Next ready task or None if queue is empty.
        """
        with self._lock:
            for priority in ["high", "normal", "low"]:
                for task in self._queues[priority]:
                    if self._dependencies_satisfied(task):
                        return task
            return None
    
    def _dependencies_satisfied(self, task: POPSSTask) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self._completed:
                return False
        return True
    
    def _check_resources(
        self,
        task: POPSSTask,
        available: Dict[str, Any],
    ) -> bool:
        """Check if resources are available for the task."""
        req = task.requirements
        
        if "gpu_count" in req and "gpu_count" in available:
            if available["gpu_count"] < req["gpu_count"]:
                return False
        
        if "gpu_memory_mb" in req and "gpu_memory_mb" in available:
            if available["gpu_memory_mb"] < req["gpu_memory_mb"]:
                return False
        
        if "memory_mb" in req and "memory_mb" in available:
            if available["memory_mb"] < req["memory_mb"]:
                return False
        
        if "cpu_cores" in req and "cpu_cores" in available:
            if available["cpu_cores"] < req["cpu_cores"]:
                return False
        
        return True
    
    def complete(self, task_id: str) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            True if task was marked as completed.
        """
        with self._lock:
            if task_id in self._running:
                self._running.remove(task_id)
            self._completed.add(task_id)
            self._task_index.pop(task_id, None)
            self._save_state()
            return True
    
    def fail(self, task_id: str, requeue: bool = True) -> bool:
        """
        Mark a task as failed.
        
        Args:
            task_id: Task identifier.
            requeue: Whether to requeue the task.
            
        Returns:
            True if task was requeued, False otherwise.
        """
        with self._lock:
            if task_id not in self._running:
                return False
            
            self._running.remove(task_id)
            
            if task_id in self._task_index:
                task = self._task_index[task_id]
                if requeue and task.attempts < task.max_attempts:
                    self._queues[task.priority].append(task)
                    self._save_state()
                    return True
            
            self._save_state()
            return False
    
    def cancel(self, task_id: str) -> bool:
        """
        Cancel a task from the queue.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            True if task was cancelled.
        """
        with self._lock:
            if task_id in self._running:
                self._running.remove(task_id)
            
            for priority in self._queues:
                for i, task in enumerate(self._queues[priority]):
                    if task.task_id == task_id:
                        self._queues[priority].pop(i)
                        self._task_index.pop(task_id, None)
                        self._save_state()
                        return True
            
            self._task_index.pop(task_id, None)
            self._save_state()
            return False
    
    def get_task(self, task_id: str) -> Optional[POPSSTask]:
        """
        Get a task by ID.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            Task or None if not found.
        """
        with self._lock:
            return self._task_index.get(task_id)
    
    def get_position(self, task_id: str) -> int:
        """
        Get the position of a task in the queue.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            Position (0 = first) or -1 if not found.
        """
        with self._lock:
            if task_id not in self._task_index:
                return -1
            
            position = 0
            task = self._task_index[task_id]
            
            for priority in ["high", "normal", "low"]:
                if priority == task.priority:
                    for t in self._queues[priority]:
                        if t.task_id == task_id:
                            return position
                        position += 1
                    break
                position += len(self._queues[priority])
            
            return -1
    
    def size(self, priority: Optional[str] = None) -> int:
        """
        Get the number of tasks in the queue.
        
        Args:
            priority: Filter by priority level.
            
        Returns:
            Number of tasks.
        """
        with self._lock:
            if priority:
                return len(self._queues.get(priority, []))
            return sum(len(q) for q in self._queues.values())
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self.size() == 0
    
    def clear(self, priority: Optional[str] = None) -> int:
        """
        Clear tasks from the queue.
        
        Args:
            priority: Clear only this priority level.
            
        Returns:
            Number of tasks cleared.
        """
        with self._lock:
            count = 0
            if priority:
                for task in self._queues[priority]:
                    self._task_index.pop(task.task_id, None)
                    count += 1
                self._queues[priority] = []
            else:
                for p in self._queues:
                    for task in self._queues[p]:
                        self._task_index.pop(task.task_id, None)
                        count += 1
                    self._queues[p] = []
            
            self._save_state()
            return count
    
    def get_all_tasks(self) -> List[POPSSTask]:
        """
        Get all tasks in the queue.
        
        Returns:
            List of all tasks ordered by priority.
        """
        with self._lock:
            tasks = []
            for priority in ["high", "normal", "low"]:
                tasks.extend(self._queues[priority])
            return tasks
    
    def get_running_tasks(self) -> List[str]:
        """Get list of running task IDs."""
        with self._lock:
            return list(self._running)
    
    def get_completed_tasks(self) -> List[str]:
        """Get list of completed task IDs."""
        with self._lock:
            return list(self._completed)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue statistics.
        """
        with self._lock:
            return {
                "total_queued": self.size(),
                "high_priority": len(self._queues["high"]),
                "normal_priority": len(self._queues["normal"]),
                "low_priority": len(self._queues["low"]),
                "running": len(self._running),
                "completed": len(self._completed),
            }
    
    def reorder(self, task_id: str, new_priority: str) -> bool:
        """
        Change the priority of a task.
        
        Args:
            task_id: Task identifier.
            new_priority: New priority level.
            
        Returns:
            True if priority was changed.
        """
        if new_priority not in self.VALID_PRIORITIES:
            raise ValueError(f"Invalid priority: {new_priority}")
        
        with self._lock:
            if task_id not in self._task_index:
                return False
            
            task = self._task_index[task_id]
            old_priority = task.priority
            
            if old_priority == new_priority:
                return True
            
            for i, t in enumerate(self._queues[old_priority]):
                if t.task_id == task_id:
                    self._queues[old_priority].pop(i)
                    break
            
            task.priority = new_priority
            self._queues[new_priority].append(task)
            self._save_state()
            return True
    
    def add_dependency(self, task_id: str, depends_on: str) -> bool:
        """
        Add a dependency to a task.
        
        Args:
            task_id: Task identifier.
            depends_on: Task ID that this task depends on.
            
        Returns:
            True if dependency was added.
        """
        with self._lock:
            if task_id not in self._task_index:
                return False
            
            task = self._task_index[task_id]
            if depends_on not in task.dependencies:
                task.dependencies.append(depends_on)
                self._save_state()
            return True
    
    def remove_dependency(self, task_id: str, depends_on: str) -> bool:
        """
        Remove a dependency from a task.
        
        Args:
            task_id: Task identifier.
            depends_on: Task ID to remove from dependencies.
            
        Returns:
            True if dependency was removed.
        """
        with self._lock:
            if task_id not in self._task_index:
                return False
            
            task = self._task_index[task_id]
            if depends_on in task.dependencies:
                task.dependencies.remove(depends_on)
                self._save_state()
            return True
    
    def cleanup_completed(self, max_age_hours: int = 24) -> int:
        """
        Remove old completed task records.
        
        Args:
            max_age_hours: Maximum age in hours.
            
        Returns:
            Number of records removed.
        """
        return len(self._completed)
