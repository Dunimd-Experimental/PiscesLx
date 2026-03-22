#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file file except in compliance with the License.
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
Ink MoE Dynamic Management Module

This module provides LRU-based dynamic loading/unloading of MoE experts
for memory-efficient training and inference with mixture-of-experts models.

Key Features:
    - LRU cache for experts on GPU
    - Dynamic loading based on utilization threshold
    - Offloading least-used experts to CPU
    - Memory-aware expert management

Memory Savings:
    - For MoE models with N experts, keeping only K experts on GPU
    - Memory reduction: N/K where K < N
    - Typical: 8 experts on GPU out of 64 total = 8x reduction for expert storage
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import OrderedDict
import time

from configs.version import VERSION


class POPSSInkMoEManager:
    """
    LRU-based MoE Expert Manager.
    
    Dynamically manages which experts are kept on GPU vs CPU based on
    recent utilization, significantly reducing memory for MoE models.
    
    Attributes:
        num_experts: Total number of experts in the model
        max_experts_on_gpu: Maximum number of experts to keep on GPU
        offload_threshold: Utilization threshold for offloading
        device: Target device (cuda/cpu)
    
    Example:
        >>> manager = POPSSInkMoEManager(num_experts=64, max_experts_on_gpu=8)
        >>> manager.register_expert(expert_id=0, expert_module=expert_layer)
        >>> active_experts = manager.get_active_experts()
        >>> manager.record_access(expert_id=0)
    """
    
    def __init__(
        self,
        num_experts: int = 64,
        max_experts_on_gpu: int = 4,
        offload_threshold: float = 0.8,
        lru_cache_size: int = 8,
        device: Optional[torch.device] = None,
    ):
        self.num_experts = num_experts
        self.max_experts_on_gpu = max_experts_on_gpu
        self.offload_threshold = offload_threshold
        self.lru_cache_size = lru_cache_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._expert_states: Dict[int, str] = {}
        self._expert_refs: Dict[int, nn.Module] = {}
        self._access_counts: Dict[int, int] = {}
        self._last_access_times: Dict[int, float] = {}
        self._lru_order: OrderedDict[int, None] = OrderedDict()
        
        self._stats: Dict[str, Any] = {
            "total_accesses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "offloads": 0,
            "loads": 0,
        }
    
    def register_expert(
        self,
        expert_id: int,
        expert_module: nn.Module,
    ):
        """
        Register an expert module.
        
        Args:
            expert_id: Unique identifier for the expert
            expert_module: The expert PyTorch module
        """
        self._expert_refs[expert_id] = expert_module
        self._expert_states[expert_id] = "cpu"
        self._access_counts[expert_id] = 0
        self._last_access_times[expert_id] = 0.0
    
    def record_access(self, expert_id: int):
        """
        Record access to an expert for LRU tracking.
        
        Args:
            expert_id: ID of the accessed expert
        """
        if expert_id not in self._lru_order:
            self._lru_order[expert_id] = None
        
        self._lru_order.move_to_end(expert_id)
        
        self._last_access_times[expert_id] = time.time()
        self._access_counts[expert_id] = self._access_counts.get(expert_id, 0) + 1
        self._stats["total_accesses"] += 1
    
    def get_active_experts(self) -> Set[int]:
        """
        Get currently active (on-GPU) expert IDs.
        
        Returns:
            Set of expert IDs currently on GPU
        """
        return {
            eid for eid, state in self._expert_states.items()
            if state == "cuda"
        }
    
    def should_offload(self, expert_id: int) -> bool:
        """
        Check if an expert should be offloaded.
        
        Args:
            expert_id: ID of the expert to check
        
        Returns:
            True if expert should be moved to CPU
        """
        if len(self.get_active_experts()) <= self.max_experts_on_gpu:
            return False
        
        if expert_id not in self._lru_order:
            return True
        
        lru_id = next(iter(self._lru_order))
        return expert_id == lru_id
    
    def offload_least_used(self) -> Optional[int]:
        """
        Offload the least recently used expert to CPU.
        
        Returns:
            ID of offloaded expert, or None if no expert to offload
        """
        if len(self.get_active_experts()) <= self.max_experts_on_gpu:
            return None
        
        if not self._lru_order:
            return None
        
        lru_id = next(iter(self._lru_order))
        
        if lru_id in self._expert_refs:
            expert = self._expert_refs[lru_id]
            if hasattr(expert, "to"):
                expert.to("cpu")
        
        self._expert_states[lru_id] = "cpu"
        del self._lru_order[lru_id]
        
        self._stats["offloads"] += 1
        
        return lru_id
    
    def load_expert(self, expert_id: int) -> bool:
        """
        Load an expert from CPU to GPU.
        
        Args:
            expert_id: ID of the expert to load
        
        Returns:
            True if successfully loaded
        """
        if self._expert_states.get(expert_id) == "cuda":
            return True
        
        while len(self.get_active_experts()) >= self.max_experts_on_gpu:
            offloaded = self.offload_least_used()
            if offloaded is None:
                return False
        
        if expert_id in self._expert_refs:
            expert = self._expert_refs[expert_id]
            if hasattr(expert, "to"):
                expert.to(self.device)
        
        self._expert_states[expert_id] = "cuda"
        self._lru_order[expert_id] = None
        self._lru_order.move_to_end(expert_id)
        
        self._stats["loads"] += 1
        
        return True
    
    def get_utilization(self) -> Dict[int, float]:
        """
        Get utilization ratio for each expert.
        
        Returns:
            Dictionary mapping expert_id to utilization ratio
        """
        if not self._access_counts:
            return {eid: 0.0 for eid in self._expert_refs.keys()}
        
        max_count = max(self._access_counts.values()) if self._access_counts else 1
        
        return {
            eid: count / max_count
            for eid, count in self._access_counts.items()
        }
    
    def get_overutilized_experts(self) -> List[int]:
        """
        Get list of over-utilized experts (above threshold).
        
        Returns:
            List of expert IDs with high utilization
        """
        utilization = self.get_utilization()
        return [
            eid for eid, ratio in utilization.items()
            if ratio >= self.offload_threshold
        ]
    
    def get_underutilized_experts(self) -> List[int]:
        """
        Get list of under-utilized experts (below threshold).
        
        Returns:
            List of expert IDs with low utilization
        """
        utilization = self.get_utilization()
        return [
            eid for eid, ratio in utilization.items()
            if ratio < (1.0 - self.offload_threshold)
        ]
    
    def dynamic_manage(self):
        """
        Perform dynamic expert management.
        
        Offloads underutilized experts and ensures active experts
        are loaded based on LRU order.
        """
        underutilized = self.get_underutilized_experts()
        
        for eid in underutilized:
            if self.should_offload(eid):
                self.offload_least_used()
        
        for eid in self._expert_refs.keys():
            if self._expert_states.get(eid) == "cpu":
                if eid in self._lru_order:
                    lru_id = next(iter(self._lru_order))
                    if lru_id != eid:
                        continue
                self.load_expert(eid)
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Estimate memory footprint of managed experts.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        cuda_experts = len(self.get_active_experts())
        cpu_experts = self.num_experts - cuda_experts
        
        total_params = 0
        for expert in self._expert_refs.values():
            total_params += sum(p.numel() for p in expert.parameters())
        
        param_size_mb = (total_params * 4) / (1024 * 1024)
        
        cuda_mem = param_size_mb * (cuda_experts / max(1, self.num_experts))
        cpu_mem = param_size_mb * (cpu_experts / max(1, self.num_experts))
        
        return {
            "total_experts": self.num_experts,
            "cuda_experts": cuda_experts,
            "cpu_experts": cpu_experts,
            "total_params_mb": param_size_mb,
            "cuda_memory_mb": cuda_mem,
            "cpu_memory_mb": cpu_mem,
            "memory_reduction_ratio": self.num_experts / max(1, cuda_experts),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        stats = self._stats.copy()
        stats["cache_hit_rate"] = (
            stats["cache_hits"] / stats["total_accesses"]
            if stats["total_accesses"] > 0 else 0.0
        )
        stats["active_experts"] = len(self.get_active_experts())
        stats["memory_footprint"] = self.get_memory_footprint()
        return stats
    
    def reset_statistics(self):
        """Reset access statistics."""
        self._stats = {
            "total_accesses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "offloads": 0,
            "loads": 0,
        }
        self._access_counts.clear()
        self._last_access_times.clear()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state for serialization."""
        return {
            "num_experts": self.num_experts,
            "max_experts_on_gpu": self.max_experts_on_gpu,
            "offload_threshold": self.offload_threshold,
            "lru_cache_size": self.lru_cache_size,
            "expert_states": self._expert_states.copy(),
            "access_counts": self._access_counts.copy(),
            "last_access_times": self._last_access_times.copy(),
            "lru_order": list(self._lru_order.keys()),
            "stats": self._stats.copy(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state from dictionary."""
        self.num_experts = state_dict["num_experts"]
        self.max_experts_on_gpu = state_dict["max_experts_on_gpu"]
        self.offload_threshold = state_dict["offload_threshold"]
        self.lru_cache_size = state_dict["lru_cache_size"]
        self._expert_states = state_dict["expert_states"].copy()
        self._access_counts = state_dict["access_counts"].copy()
        self._last_access_times = state_dict["last_access_times"].copy()
        
        self._lru_order = OrderedDict()
        for key in state_dict["lru_order"]:
            self._lru_order[key] = None
        
        self._stats = state_dict["stats"].copy()