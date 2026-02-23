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
Resource Limits Manager for LLM Training and Inference.

This module provides resource limit management and checking for LLM workloads,
including GPU memory, CPU cores, system memory, and storage constraints.

Key Features:
    - Resource requirement specification
    - Availability checking before task submission
    - Memory-based resource estimation
    - Multi-resource constraint validation

Usage:
    from opss.run.resources import POPSSResourceLimits, POPSSResourceRequirements
    
    limits = POPSSResourceLimits()
    
    requirements = POPSSResourceRequirements(
        gpu_count=2,
        gpu_memory_mb=40000,
        cpu_cores=8,
        memory_mb=64000,
    )
    
    if limits.check_available(requirements):
        limits.allocate("train_001", requirements)
"""

import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class POPSSResourceRequirements:
    """Resource requirements for a task."""
    gpu_count: int = 1
    gpu_memory_mb: int = 24000
    cpu_cores: int = 4
    memory_mb: int = 32000
    storage_mb: int = 0
    priority: str = "normal"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_count": self.gpu_count,
            "gpu_memory_mb": self.gpu_memory_mb,
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "POPSSResourceRequirements":
        """Create from dictionary."""
        return cls(
            gpu_count=data.get("gpu_count", 1),
            gpu_memory_mb=data.get("gpu_memory_mb", 24000),
            cpu_cores=data.get("cpu_cores", 4),
            memory_mb=data.get("memory_mb", 32000),
            storage_mb=data.get("storage_mb", 0),
            priority=data.get("priority", "normal"),
        )


@dataclass
class POPSSResourceAllocation:
    """Resource allocation record."""
    task_id: str
    requirements: POPSSResourceRequirements
    allocated_at: str
    gpu_ids: List[int] = field(default_factory=list)
    cpu_affinity: List[int] = field(default_factory=list)


class POPSSResourceLimits:
    """
    Resource Limits Manager for LLM Training and Inference.
    
    This class manages resource allocation and checking for LLM workloads,
    ensuring tasks have sufficient resources before execution.
    
    Attributes:
        _allocations: Dictionary mapping task ID to allocation record.
        _lock: Thread lock for concurrent access.
        _state_file: Path to persistent state file.
    
    Example:
        >>> limits = POPSSResourceLimits()
        >>> req = POPSSResourceRequirements(gpu_count=2, memory_mb=64000)
        >>> limits.check_available(req)
        True
        >>> limits.allocate("train_001", req)
    """
    
    PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}
    
    def __init__(self, state_dir: Optional[str] = None):
        """
        Initialize the resource limits manager.
        
        Args:
            state_dir: Directory for persistent state storage.
        """
        self._allocations: Dict[str, POPSSResourceAllocation] = {}
        self._lock = threading.RLock()
        
        if state_dir is None:
            state_dir = str(Path(".pisceslx") / "resources")
        self._state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self._state_file = str(Path(state_dir) / "resource_state.json")
        
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persistent state from disk."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for task_id, alloc_data in data.get("allocations", {}).items():
                self._allocations[task_id] = POPSSResourceAllocation(
                    task_id=alloc_data["task_id"],
                    requirements=POPSSResourceRequirements.from_dict(alloc_data["requirements"]),
                    allocated_at=alloc_data["allocated_at"],
                    gpu_ids=alloc_data.get("gpu_ids", []),
                    cpu_affinity=alloc_data.get("cpu_affinity", []),
                )
        except Exception:
            pass
    
    def _save_state(self) -> None:
        """Save persistent state to disk."""
        data = {
            "allocations": {},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for task_id, alloc in self._allocations.items():
            data["allocations"][task_id] = {
                "task_id": alloc.task_id,
                "requirements": alloc.requirements.to_dict(),
                "allocated_at": alloc.allocated_at,
                "gpu_ids": alloc.gpu_ids,
                "cpu_affinity": alloc.cpu_affinity,
            }
        
        tmp_file = f"{self._state_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, self._state_file)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get current system resource information.
        
        Returns:
            Dictionary with system resource statistics.
        """
        info = {
            "cpu_count": os.cpu_count() or 1,
            "cpu_percent": 0.0,
            "memory_total_mb": 0,
            "memory_available_mb": 0,
            "memory_percent": 0.0,
            "storage_total_mb": 0,
            "storage_available_mb": 0,
            "storage_percent": 0.0,
        }
        
        if PSUTIL_AVAILABLE:
            try:
                info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                
                mem = psutil.virtual_memory()
                info["memory_total_mb"] = mem.total // (1024 * 1024)
                info["memory_available_mb"] = mem.available // (1024 * 1024)
                info["memory_percent"] = mem.percent
                
                disk = shutil.disk_usage("/")
                info["storage_total_mb"] = disk.total // (1024 * 1024)
                info["storage_available_mb"] = disk.free // (1024 * 1024)
                info["storage_percent"] = (disk.used / disk.total) * 100
            except Exception:
                pass
        
        return info
    
    def check_available(
        self,
        requirements: POPSSResourceRequirements,
        exclude_task: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if resources are available for the given requirements.
        
        Args:
            requirements: Resource requirements to check.
            exclude_task: Task ID to exclude from current allocations.
            
        Returns:
            Tuple of (is_available, details_dict).
        """
        system_info = self.get_system_info()
        details = {
            "available": True,
            "reasons": [],
            "system_info": system_info,
            "requirements": requirements.to_dict(),
        }
        
        allocated_cpu = 0
        allocated_memory = 0
        for task_id, alloc in self._allocations.items():
            if task_id == exclude_task:
                continue
            allocated_cpu += alloc.requirements.cpu_cores
            allocated_memory += alloc.requirements.memory_mb
        
        available_cpu = system_info["cpu_count"] - allocated_cpu
        if available_cpu < requirements.cpu_cores:
            details["available"] = False
            details["reasons"].append(
                f"Insufficient CPU cores: need {requirements.cpu_cores}, available {available_cpu}"
            )
        
        available_memory = system_info["memory_available_mb"] - allocated_memory
        if available_memory < requirements.memory_mb:
            details["available"] = False
            details["reasons"].append(
                f"Insufficient memory: need {requirements.memory_mb}MB, available {available_memory}MB"
            )
        
        if requirements.storage_mb > 0:
            if system_info["storage_available_mb"] < requirements.storage_mb:
                details["available"] = False
                details["reasons"].append(
                    f"Insufficient storage: need {requirements.storage_mb}MB, "
                    f"available {system_info['storage_available_mb']}MB"
                )
        
        return details["available"], details
    
    def allocate(
        self,
        task_id: str,
        requirements: POPSSResourceRequirements,
        gpu_ids: Optional[List[int]] = None,
    ) -> POPSSResourceAllocation:
        """
        Allocate resources to a task.
        
        Args:
            task_id: Unique task identifier.
            requirements: Resource requirements.
            gpu_ids: List of allocated GPU IDs.
            
        Returns:
            Resource allocation record.
            
        Raises:
            ValueError: If resources are not available.
        """
        with self._lock:
            available, details = self.check_available(requirements, exclude_task=task_id)
            if not available:
                raise ValueError("; ".join(details["reasons"]))
            
            now = datetime.now(timezone.utc).isoformat()
            allocation = POPSSResourceAllocation(
                task_id=task_id,
                requirements=requirements,
                allocated_at=now,
                gpu_ids=gpu_ids or [],
                cpu_affinity=[],
            )
            
            self._allocations[task_id] = allocation
            self._save_state()
            
            return allocation
    
    def release(self, task_id: str) -> Optional[POPSSResourceAllocation]:
        """
        Release resources allocated to a task.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            Released allocation record or None if not found.
        """
        with self._lock:
            if task_id not in self._allocations:
                return None
            
            allocation = self._allocations.pop(task_id)
            self._save_state()
            return allocation
    
    def get_allocation(self, task_id: str) -> Optional[POPSSResourceAllocation]:
        """
        Get allocation information for a task.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            Allocation record or None if not found.
        """
        with self._lock:
            return self._allocations.get(task_id)
    
    def get_all_allocations(self) -> Dict[str, POPSSResourceAllocation]:
        """
        Get all current allocations.
        
        Returns:
            Dictionary mapping task ID to allocation record.
        """
        with self._lock:
            return dict(self._allocations)
    
    def get_utilization(self) -> Dict[str, Any]:
        """
        Get resource utilization summary.
        
        Returns:
            Dictionary with utilization statistics.
        """
        system_info = self.get_system_info()
        
        total_cpu_allocated = sum(
            alloc.requirements.cpu_cores for alloc in self._allocations.values()
        )
        total_memory_allocated = sum(
            alloc.requirements.memory_mb for alloc in self._allocations.values()
        )
        
        return {
            "system": system_info,
            "allocated": {
                "tasks": len(self._allocations),
                "cpu_cores": total_cpu_allocated,
                "memory_mb": total_memory_allocated,
            },
            "available": {
                "cpu_cores": system_info["cpu_count"] - total_cpu_allocated,
                "memory_mb": system_info["memory_available_mb"] - total_memory_allocated,
            },
        }
    
    def estimate_model_requirements(
        self,
        model_size_b: float,
        precision: str = "bf16",
        batch_size: int = 1,
        seq_len: int = 4096,
        gradient_checkpointing: bool = True,
    ) -> POPSSResourceRequirements:
        """
        Estimate resource requirements for a model.
        
        Args:
            model_size_b: Model size in billions of parameters.
            precision: Training precision (fp32/fp16/bf16/fp8).
            batch_size: Training batch size.
            seq_len: Sequence length.
            gradient_checkpointing: Whether gradient checkpointing is enabled.
            
        Returns:
            Estimated resource requirements.
        """
        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
        }
        
        bp = bytes_per_param.get(precision, 2)
        
        model_memory_gb = model_size_b * bp
        
        hidden_size = 4096
        num_layers = int(model_size_b * 1000 / hidden_size)
        
        if gradient_checkpointing:
            activation_memory_gb = (batch_size * seq_len * hidden_size * 4) / (1024 ** 3)
        else:
            activation_memory_gb = (batch_size * seq_len * hidden_size * num_layers * 4) / (1024 ** 3)
        
        optimizer_memory_gb = model_size_b * 12 / 1024
        
        total_memory_gb = model_memory_gb + activation_memory_gb + optimizer_memory_gb
        total_memory_mb = int(total_memory_gb * 1024 * 1.2)
        
        gpu_memory_mb = min(total_memory_mb, 80000)
        gpu_count = max(1, int(total_memory_mb / 40000))
        
        cpu_cores = min(8 * gpu_count, os.cpu_count() or 8)
        memory_mb = int(total_memory_mb * 1.5)
        
        return POPSSResourceRequirements(
            gpu_count=gpu_count,
            gpu_memory_mb=gpu_memory_mb,
            cpu_cores=cpu_cores,
            memory_mb=memory_mb,
            storage_mb=int(model_size_b * 100),
        )
    
    def cleanup_stale_allocations(self, active_task_ids: List[str]) -> List[str]:
        """
        Remove allocations for tasks that are no longer active.
        
        Args:
            active_task_ids: List of currently active task IDs.
            
        Returns:
            List of cleaned up task IDs.
        """
        cleaned = []
        with self._lock:
            stale = [tid for tid in self._allocations if tid not in active_task_ids]
            for task_id in stale:
                self.release(task_id)
                cleaned.append(task_id)
        return cleaned
