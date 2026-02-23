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
GPU Resource Scheduler for LLM Training and Inference.

This module provides a comprehensive GPU resource management system specifically
designed for large language model training and inference workloads.

Key Features:
    - GPU availability detection via nvidia-smi
    - Memory-based allocation (24GB/40GB/80GB tiers)
    - Exclusive allocation (one task per GPU)
    - Auto-release on task completion
    - Multi-GPU allocation support
    - Memory utilization tracking

Usage:
    from opss.run.gpu_scheduler import POPSSGPUScheduler
    
    scheduler = POPSSGPUScheduler()
    
    # Get available GPUs
    available = scheduler.get_available_gpus()
    
    # Allocate GPUs for a task
    allocated = scheduler.allocate("train_001", [0, 1])
    
    # Release GPUs when task completes
    scheduler.release("train_001")
"""

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class POPSSGPUInfo:
    """GPU information structure."""
    gpu_id: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    utilization_percent: int
    temperature_c: int
    power_usage_w: int
    power_limit_w: int
    is_available: bool = True
    allocated_to: Optional[str] = None
    allocated_at: Optional[str] = None


@dataclass
class POPSSGPUAllocation:
    """GPU allocation record."""
    task_id: str
    gpu_ids: List[int]
    allocated_at: str
    memory_required_mb: int = 0
    priority: str = "normal"


class POPSSGPUScheduler:
    """
    GPU Resource Scheduler for LLM Training and Inference.
    
    This class manages GPU allocation, deallocation, and monitoring for
    large language model workloads. It provides exclusive GPU allocation
    to ensure training stability and optimal resource utilization.
    
    Attributes:
        _gpu_pool: Dictionary mapping GPU ID to allocation info.
        _allocations: Dictionary mapping task ID to allocation record.
        _gpu_info: Dictionary mapping GPU ID to GPU info.
        _lock: Thread lock for concurrent access.
        _state_file: Path to persistent state file.
    
    Example:
        >>> scheduler = POPSSGPUScheduler()
        >>> gpus = scheduler.get_available_gpus(min_memory_mb=24000)
        >>> scheduler.allocate("train_001", gpus[:2])
        [0, 1]
        >>> scheduler.release("train_001")
        [0, 1]
    """
    
    MEMORY_TIERS = {
        "small": 12 * 1024,
        "medium": 24 * 1024,
        "large": 40 * 1024,
        "xlarge": 80 * 1024,
    }
    
    PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}
    
    def __init__(self, state_dir: Optional[str] = None):
        """
        Initialize the GPU scheduler.
        
        Args:
            state_dir: Directory for persistent state storage.
        """
        self._gpu_pool: Dict[int, POPSSGPUAllocation] = {}
        self._allocations: Dict[str, POPSSGPUAllocation] = {}
        self._gpu_info: Dict[int, POPSSGPUInfo] = {}
        self._lock = threading.RLock()
        
        if state_dir is None:
            state_dir = str(Path(".pisceslx") / "gpu_scheduler")
        self._state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self._state_file = str(Path(state_dir) / "gpu_state.json")
        
        self._load_state()
        self._refresh_gpu_info()
    
    def _load_state(self) -> None:
        """Load persistent state from disk."""
        if not os.path.exists(self._state_file):
            return
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for task_id, alloc_data in data.get("allocations", {}).items():
                self._allocations[task_id] = POPSSGPUAllocation(
                    task_id=alloc_data["task_id"],
                    gpu_ids=alloc_data["gpu_ids"],
                    allocated_at=alloc_data["allocated_at"],
                    memory_required_mb=alloc_data.get("memory_required_mb", 0),
                    priority=alloc_data.get("priority", "normal"),
                )
                for gpu_id in alloc_data["gpu_ids"]:
                    self._gpu_pool[gpu_id] = self._allocations[task_id]
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
                "gpu_ids": alloc.gpu_ids,
                "allocated_at": alloc.allocated_at,
                "memory_required_mb": alloc.memory_required_mb,
                "priority": alloc.priority,
            }
        
        tmp_file = f"{self._state_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, self._state_file)
    
    def _run_nvidia_smi(self, args: List[str]) -> Optional[str]:
        """
        Run nvidia-smi command and return output.
        
        Args:
            args: Additional arguments for nvidia-smi.
            
        Returns:
            Command output string or None on failure.
        """
        try:
            cmd = ["nvidia-smi", "--query-gpu=" + ",".join(args), "--format=csv,noheader,nounits"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    def _refresh_gpu_info(self) -> None:
        """Refresh GPU information from nvidia-smi."""
        output = self._run_nvidia_smi([
            "index",
            "name",
            "memory.total",
            "memory.free",
            "memory.used",
            "utilization.gpu",
            "temperature.gpu",
            "power.draw",
            "power.limit",
        ])
        
        if not output:
            return
        
        with self._lock:
            for line in output.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 9:
                    continue
                
                try:
                    gpu_id = int(parts[0])
                    gpu_info = POPSSGPUInfo(
                        gpu_id=gpu_id,
                        name=parts[1],
                        total_memory_mb=int(float(parts[2])),
                        free_memory_mb=int(float(parts[3])),
                        used_memory_mb=int(float(parts[4])),
                        utilization_percent=int(float(parts[5])),
                        temperature_c=int(float(parts[6])),
                        power_usage_w=int(float(parts[7])),
                        power_limit_w=int(float(parts[8])),
                    )
                    
                    if gpu_id in self._gpu_pool:
                        alloc = self._gpu_pool[gpu_id]
                        gpu_info.is_available = False
                        gpu_info.allocated_to = alloc.task_id
                        gpu_info.allocated_at = alloc.allocated_at
                    
                    self._gpu_info[gpu_id] = gpu_info
                except (ValueError, IndexError):
                    continue
    
    def get_gpu_count(self) -> int:
        """
        Get total number of GPUs in the system.
        
        Returns:
            Number of GPUs available.
        """
        self._refresh_gpu_info()
        return len(self._gpu_info)
    
    def get_gpu_info(self, gpu_id: Optional[int] = None) -> Dict[int, POPSSGPUInfo]:
        """
        Get GPU information.
        
        Args:
            gpu_id: Specific GPU ID, or None for all GPUs.
            
        Returns:
            Dictionary mapping GPU ID to GPU info.
        """
        self._refresh_gpu_info()
        if gpu_id is not None:
            if gpu_id in self._gpu_info:
                return {gpu_id: self._gpu_info[gpu_id]}
            return {}
        return dict(self._gpu_info)
    
    def get_available_gpus(
        self,
        count: int = 1,
        min_memory_mb: int = 0,
        exclude: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Find available GPUs meeting the requirements.
        
        Args:
            count: Number of GPUs required.
            min_memory_mb: Minimum free memory per GPU in MB.
            exclude: List of GPU IDs to exclude.
            
        Returns:
            List of available GPU IDs.
        """
        self._refresh_gpu_info()
        exclude = exclude or []
        
        available = []
        for gpu_id, info in sorted(self._gpu_info.items()):
            if gpu_id in exclude:
                continue
            if not info.is_available:
                continue
            if info.free_memory_mb < min_memory_mb:
                continue
            available.append(gpu_id)
        
        return available[:count]
    
    def allocate(
        self,
        task_id: str,
        gpu_ids: Optional[List[int]] = None,
        count: int = 1,
        min_memory_mb: int = 0,
        priority: str = "normal",
    ) -> List[int]:
        """
        Allocate GPUs to a task.
        
        Args:
            task_id: Unique task identifier.
            gpu_ids: Specific GPU IDs to allocate, or None for auto-selection.
            count: Number of GPUs to allocate (used if gpu_ids is None).
            min_memory_mb: Minimum memory requirement per GPU.
            priority: Task priority (high/normal/low).
            
        Returns:
            List of allocated GPU IDs.
            
        Raises:
            ValueError: If requested GPUs are not available.
        """
        with self._lock:
            if task_id in self._allocations:
                return self._allocations[task_id].gpu_ids
            
            if gpu_ids is None:
                gpu_ids = self.get_available_gpus(count=count, min_memory_mb=min_memory_mb)
                if len(gpu_ids) < count:
                    raise ValueError(f"Insufficient available GPUs: need {count}, have {len(gpu_ids)}")
            else:
                available = self.get_available_gpus(count=len(gpu_ids), min_memory_mb=min_memory_mb)
                for gid in gpu_ids:
                    if gid not in available:
                        raise ValueError(f"GPU {gid} is not available")
            
            now = datetime.now(timezone.utc).isoformat()
            allocation = POPSSGPUAllocation(
                task_id=task_id,
                gpu_ids=gpu_ids,
                allocated_at=now,
                memory_required_mb=min_memory_mb,
                priority=priority,
            )
            
            for gpu_id in gpu_ids:
                self._gpu_pool[gpu_id] = allocation
                if gpu_id in self._gpu_info:
                    self._gpu_info[gpu_id].is_available = False
                    self._gpu_info[gpu_id].allocated_to = task_id
                    self._gpu_info[gpu_id].allocated_at = now
            
            self._allocations[task_id] = allocation
            self._save_state()
            
            return gpu_ids
    
    def release(self, task_id: str) -> List[int]:
        """
        Release GPUs allocated to a task.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            List of released GPU IDs.
        """
        with self._lock:
            if task_id not in self._allocations:
                return []
            
            allocation = self._allocations.pop(task_id)
            released_gpus = allocation.gpu_ids
            
            for gpu_id in released_gpus:
                self._gpu_pool.pop(gpu_id, None)
                if gpu_id in self._gpu_info:
                    self._gpu_info[gpu_id].is_available = True
                    self._gpu_info[gpu_id].allocated_to = None
                    self._gpu_info[gpu_id].allocated_at = None
            
            self._save_state()
            return released_gpus
    
    def get_task_gpus(self, task_id: str) -> List[int]:
        """
        Get GPUs allocated to a task.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            List of allocated GPU IDs.
        """
        with self._lock:
            if task_id in self._allocations:
                return self._allocations[task_id].gpu_ids
            return []
    
    def get_allocation_info(self, task_id: str) -> Optional[POPSSGPUAllocation]:
        """
        Get allocation information for a task.
        
        Args:
            task_id: Task identifier.
            
        Returns:
            Allocation record or None if not allocated.
        """
        with self._lock:
            return self._allocations.get(task_id)
    
    def get_all_allocations(self) -> Dict[str, POPSSGPUAllocation]:
        """
        Get all current allocations.
        
        Returns:
            Dictionary mapping task ID to allocation record.
        """
        with self._lock:
            return dict(self._allocations)
    
    def is_gpu_available(self, gpu_id: int) -> bool:
        """
        Check if a specific GPU is available.
        
        Args:
            gpu_id: GPU ID to check.
            
        Returns:
            True if available, False otherwise.
        """
        self._refresh_gpu_info()
        if gpu_id not in self._gpu_info:
            return False
        return self._gpu_info[gpu_id].is_available
    
    def get_utilization(self) -> Dict[str, Any]:
        """
        Get overall GPU utilization summary.
        
        Returns:
            Dictionary with utilization statistics.
        """
        self._refresh_gpu_info()
        
        total_gpus = len(self._gpu_info)
        available_gpus = sum(1 for info in self._gpu_info.values() if info.is_available)
        allocated_gpus = total_gpus - available_gpus
        
        total_memory = sum(info.total_memory_mb for info in self._gpu_info.values())
        free_memory = sum(info.free_memory_mb for info in self._gpu_info.values())
        used_memory = total_memory - free_memory
        
        avg_utilization = 0.0
        if self._gpu_info:
            avg_utilization = sum(info.utilization_percent for info in self._gpu_info.values()) / len(self._gpu_info)
        
        return {
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "allocated_gpus": allocated_gpus,
            "total_memory_mb": total_memory,
            "free_memory_mb": free_memory,
            "used_memory_mb": used_memory,
            "memory_utilization_percent": (used_memory / total_memory * 100) if total_memory > 0 else 0,
            "avg_gpu_utilization_percent": avg_utilization,
            "allocations": len(self._allocations),
        }
    
    def suggest_gpus(
        self,
        model_size_b: float,
        precision: str = "bf16",
        batch_size: int = 1,
        seq_len: int = 4096,
    ) -> Dict[str, Any]:
        """
        Suggest GPU configuration for a model.
        
        Args:
            model_size_b: Model size in billions of parameters.
            precision: Training precision (fp32/fp16/bf16/fp8).
            batch_size: Training batch size.
            seq_len: Sequence length.
            
        Returns:
            Dictionary with GPU suggestions.
        """
        bytes_per_param = {
            "fp32": 4,
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
        }
        
        bp = bytes_per_param.get(precision, 2)
        model_memory_gb = model_size_b * bp
        
        activation_memory_gb = (batch_size * seq_len * 4096 * bp) / (1024 ** 3)
        
        optimizer_memory_gb = model_size_b * 8 / 1024
        
        total_memory_gb = model_memory_gb + activation_memory_gb + optimizer_memory_gb
        total_memory_mb = int(total_memory_gb * 1024)
        
        suitable_gpus = []
        for gpu_id, info in self._gpu_info.items():
            if info.total_memory_mb >= total_memory_mb * 0.8:
                suitable_gpus.append(gpu_id)
        
        return {
            "estimated_memory_mb": total_memory_mb,
            "model_memory_gb": model_memory_gb,
            "activation_memory_gb": activation_memory_gb,
            "optimizer_memory_gb": optimizer_memory_gb,
            "recommended_gpu_count": max(1, int(total_memory_mb / 40000) + 1),
            "suitable_gpus": suitable_gpus,
            "min_gpu_memory_mb": total_memory_mb,
        }
    
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
