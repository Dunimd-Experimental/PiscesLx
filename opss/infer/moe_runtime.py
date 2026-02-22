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
MoE Runtime Inference Operator Implementation

High-performance Mixture of Experts (MoE) runtime optimization for inference,
implementing adaptive routing, expert caching, load balancing, and dynamic
capacity management for optimal inference throughput and latency.

Key Features:
    - Adaptive Top-K routing with temperature scaling
    - Expert load balancing with auxiliary loss
    - Dynamic capacity management
    - Expert caching for token reuse
    - Cross-request batch expert sharing
    - Inference-aware scheduling with pipeline parallelism
    - Real-time routing statistics and monitoring

This operator integrates with existing MoE layers (YvMoELayer, YvDynamicMoELayer)
to provide inference-specific optimizations that are distinct from training-time considerations.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    MoERuntimeOperator                        │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
    │  │   Router    │  │  ExpertPool  │  │  CapacityMgr    │   │
    │  │ (Top-K +    │  │ (Caching &  │  │  (Dynamic       │   │
    │  │  Temp)      │  │   Sharing)   │  │   Scheduling)  │   │
    │  └─────────────┘  └──────────────┘  └──────────────────┘   │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
    │  │ LoadBalancer│  │  StatsCollector│ │  BatchScheduler │   │
    │  │ (Aux Loss)  │  │  (Monitoring) │ │  (Optimization) │   │
    │  └─────────────┘  └──────────────┘  └──────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
"""

import os
import sys
import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from pathlib import Path
from collections import OrderedDict, deque
from enum import Enum
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus

from configs.version import VERSION

class POPSSMoERoutingStrategy(Enum):
    """MoE routing strategy enumeration."""
    TOP_K = "top_k"
    EXPERT_CHOICE = "expert_choice"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"


@dataclass
class POPSSMoERuntimeConfig:
    """
    MoE runtime inference configuration.
    
    This configuration controls runtime behavior of MoE layers during inference,
    distinct from training configuration. All parameters are designed to optimize
    inference throughput and latency while maintaining output quality.
    
    Attributes:
        routing_temp: Temperature for softmax routing, higher values increase 
            randomness in expert selection (default: 1.12)
        top_k: Number of experts to route each token to (default: 2)
        capacity_factor: Multiplier for expert capacity (default: 1.0)
        min_capacity: Minimum tokens per expert (default: 4)
        noise_std: Standard deviation of noise for load balancing (default: 0.1)
        enable_load_balancing: Whether to compute load balancing auxiliary loss
        load_balance_alpha: Weight for load balancing loss (default: 0.01)
        enable_expert_caching: Whether to cache expert computations for token reuse
        expert_cache_size: Maximum number of cached expert states (default: 10000)
        enable_batch_expert_sharing: Whether to share experts across batch requests
        batch_sharing_threshold: Tokens threshold for batch expert sharing
        dynamic_capacity: Whether to dynamically adjust expert capacity
        capacity_update_interval: Steps between capacity updates (default: 16)
        capacity_margin: Extra capacity percentage (default: 0.1)
        enable_adaptive_temp: Whether to adaptively adjust routing temperature
        adaptive_temp_step: Temperature adjustment step size (default: 0.03)
        adaptive_temp_interval: Steps between temperature adjustments (default: 16)
        adaptive_temp_cap: Maximum routing temperature (default: 1.30)
        enable_prefix_caching: Whether to cache KV caches for prefix tokens
        prefix_cache_size: Maximum prefix cache size (default: 1000)
        priority_routing: Enable priority-based expert routing for critical tokens
        priority_threshold: Token importance threshold for priority routing
        drop_tokens: Whether to drop tokens when exceeding capacity
        expert_selection_strategy: Strategy for expert selection (default: TOP_K)
    """
    routing_temp: float = 1.12
    top_k: int = 2
    capacity_factor: float = 1.0
    min_capacity: int = 4
    noise_std: float = 0.1
    
    enable_load_balancing: bool = True
    load_balance_alpha: float = 0.01
    
    enable_expert_caching: bool = True
    expert_cache_size: int = 10000
    enable_batch_expert_sharing: bool = True
    batch_sharing_threshold: int = 32
    
    dynamic_capacity: bool = True
    capacity_update_interval: int = 16
    capacity_margin: float = 0.1
    
    enable_adaptive_temp: bool = True
    adaptive_temp_step: float = 0.03
    adaptive_temp_interval: int = 16
    adaptive_temp_cap: float = 1.30
    
    enable_prefix_caching: bool = True
    prefix_cache_size: int = 1000
    priority_routing: bool = False
    priority_threshold: float = 0.5
    drop_tokens: bool = True
    
    expert_selection_strategy: POPSSMoERoutingStrategy = POPSSMoERoutingStrategy.TOP_K
    
    @classmethod
    def from_model_config(cls, model_config: Any) -> "POPSSMoERuntimeConfig":
        """
        Create runtime config from model configuration.
        
        Args:
            model_config: Model configuration object (YvConfig or dict from JSON)
        
        Returns:
            MoERuntimeConfig with values from model config
        """
        if hasattr(model_config, '__dict__'):
            model_dict = vars(model_config)
        elif isinstance(model_config, dict):
            model_dict = model_config
        else:
            return cls()
        
        config_dict = {}
        
        config_dict['routing_temp'] = model_dict.get('routing_temp', 1.12)
        config_dict['top_k'] = getattr(model_config, 'moe_top_k', 
                                       model_dict.get('moe_top_k', 2))
        config_dict['capacity_factor'] = getattr(model_config, 'moe_capacity_factor',
                                                  model_dict.get('moe_capacity_factor', 1.0))
        config_dict['min_capacity'] = getattr(model_config, 'moe_min_capacity',
                                              model_dict.get('moe_min_capacity', 4))
        config_dict['noise_std'] = getattr(model_config, 'moe_noise_std',
                                           model_dict.get('moe_noise_std', 0.1))
        config_dict['load_balance_alpha'] = getattr(model_config, 'moe_load_balance_alpha',
                                                     model_dict.get('moe_load_balance_alpha', 0.01))
        config_dict['enable_load_balancing'] = model_dict.get('enable_load_balancing', True)
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, config_path: str) -> "POPSSMoERuntimeConfig":
        """
        Load runtime config from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
        
        Returns:
            POPSSMoERuntimeConfig instance
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for field_name, field_value in self.__dataclass_fields__.items():
            if field_value.type == POPSSMoERoutingStrategy:
                result[field_name] = field_value.value
            else:
                result[field_name] = getattr(self, field_name)
        return result


@dataclass
class ExpertStats:
    """Statistics for a single expert."""
    expert_id: int
    call_count: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    last_accessed: float = 0.0
    load_ratio: float = 0.0
    
    def update(self, token_count: int, latency_ms: float):
        """Update expert statistics."""
        self.call_count += 1
        self.total_tokens += token_count
        self.last_accessed = time.time()
        
        if self.call_count > 1:
            alpha = 0.3
            self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms


class ExpertCacheEntry:
    """Cache entry for expert computation reuse."""
    
    def __init__(
        self,
        input_hash: str,
        expert_indices: Tuple[int, ...],
        outputs: Tuple[torch.Tensor, ...],
        created_at: float = 0.0,
    ):
        self.input_hash = input_hash
        self.expert_indices = expert_indices
        self.outputs = outputs
        self.created_at = created_at or time.time()
        self.last_accessed = created_at or time.time()
        self.access_count = 0
        self.hit_count = 0
    
    def hit(self) -> None:
        """Record a cache hit."""
        self.last_accessed = time.time()
        self.hit_count += 1
        self.access_count += 1
    
    def miss(self) -> None:
        """Record a cache miss."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'input_hash': self.input_hash,
            'expert_indices': list(self.expert_indices),
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'hit_count': self.hit_count,
        }


class ExpertCacheManager:
    """
    Expert computation cache manager for MoE inference.
    
    Caches intermediate expert computations to avoid redundant forward passes
    for repeated or similar inputs. Uses LRU eviction policy with optional
    LFU optimization for frequently accessed entries.
    
    Features:
        - LRU eviction with access frequency weighting
        - Input hash-based matching
        - Automatic cache size management
        - Hit rate statistics tracking
        - Thread-safe operations
    """
    
    def __init__(
        self,
        max_cache_size: int = 10000,
        enable_frequency_weighting: bool = True,
        eviction_policy: str = "lru",
    ):
        self.cache: OrderedDict[str, ExpertCacheEntry] = OrderedDict()
        self.frequency_map: Dict[str, int] = {}
        self.max_cache_size = max_cache_size
        self.enable_frequency_weighting = enable_frequency_weighting
        self.eviction_policy = eviction_policy
        self.lock = threading.Lock()
        
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'insertions': 0,
            'total_accesses': 0,
        }
        
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Infer",file_path=get_log_file("PiscesLx.Opss.Infer"), enable_file=True)
    
    def _compute_input_hash(self, hidden_states: torch.Tensor) -> str:
        """
        Compute hash for hidden states input.
        
        Uses a combination of tensor statistics and truncated data hash
        for efficient approximate matching.
        """
        tensor_bytes = hidden_states.detach().cpu().numpy().tobytes()
        stats_bytes = f"{hidden_states.shape}_{hidden_states.dtype}".encode()
        combined = stats_bytes + tensor_bytes[:1024]
        return hashlib.sha256(combined).hexdigest()[:16]
    
    def get(
        self, 
        hidden_states: torch.Tensor, 
        expert_indices: Tuple[int, ...]
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Retrieve cached expert computation.
        
        Args:
            hidden_states: Input tensor for experts
            expert_indices: Indices of selected experts
        
        Returns:
            Cached outputs if hit, None otherwise
        """
        with self.lock:
            input_hash = self._compute_input_hash(hidden_states)
            cache_key = f"{input_hash}_{expert_indices}"
            
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.hit()
                self.cache.move_to_end(cache_key)
                self._update_frequency(cache_key, 1)
                
                self.stats['hits'] += 1
                self.stats['total_accesses'] += 1
                
                self._LOG.debug(f"Expert cache hit: {cache_key[:8]}...")
                return entry.outputs
            
            self.stats['misses'] += 1
            self.stats['total_accesses'] += 1
            
            self._LOG.debug(f"Expert cache miss: {cache_key[:8]}...")
            return None
    
    def insert(
        self,
        hidden_states: torch.Tensor,
        expert_indices: Tuple[int, ...],
        outputs: Tuple[torch.Tensor, ...],
    ) -> bool:
        """
        Insert expert computation into cache.
        
        Args:
            hidden_states: Input tensor for experts
            expert_indices: Indices of selected experts
            outputs: Expert computation outputs
        
        Returns:
            True if inserted successfully, False if cache full
        """
        with self.lock:
            input_hash = self._compute_input_hash(hidden_states)
            cache_key = f"{input_hash}_{expert_indices}"
            
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.hit()
                self._update_frequency(cache_key, 1)
                return True
            
            entry = ExpertCacheEntry(
                input_hash=input_hash,
                expert_indices=expert_indices,
                outputs=outputs,
            )
            
            self.cache[cache_key] = entry
            self._update_frequency(cache_key, 1)
            self.stats['insertions'] += 1
            
            self._enforce_cache_limit()
            
            self._LOG.debug(f"Inserted expert cache: {cache_key[:8]}...")
            return True
    
    def _update_frequency(self, cache_key: str, delta: int = 1):
        """Update access frequency for cache entry."""
        if self.enable_frequency_weighting:
            self.frequency_map[cache_key] = self.frequency_map.get(cache_key, 0) + delta
    
    def _enforce_cache_limit(self):
        """Evict entries to stay within cache size limit."""
        while len(self.cache) > self.max_cache_size and self.cache:
            if self.eviction_policy == "lfu":
                evicted_key = min(self.frequency_map, key=self.frequency_map.get)
                del self.frequency_map[evicted_key]
            else:
                evicted_key = next(iter(self.cache))
            
            if evicted_key in self.cache:
                del self.cache[evicted_key]
                self.stats['evictions'] += 1
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.stats['total_accesses']
            return {
                **self.stats,
                'hit_rate': self.get_hit_rate(),
                'cache_size': len(self.cache),
                'max_cache_size': self.max_cache_size,
            }
    
    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.frequency_map.clear()
            self._LOG.info("Expert cache cleared")


class BatchExpertSharing:
    """
    Batch expert sharing manager for cross-request optimization.
    
    Enables efficient expert computation sharing across batch requests
    when inputs are similar or have overlapping prefixes. Reduces redundant
    computations and improves throughput for batch inference.
    
    Features:
        - Similarity-based token grouping
        - Prefix-based KV cache sharing
        - Dynamic batch formation
        - Overlap detection and optimization
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        prefix_overlap_threshold: float = 0.7,
        max_batch_size: int = 16,
    ):
        self.similarity_threshold = similarity_threshold
        self.prefix_overlap_threshold = prefix_overlap_threshold
        self.max_batch_size = max_batch_size
        
        self.batch_groups: Dict[str, List[Dict[str, Any]]] = {}
        self.shared_prefixes: Dict[str, Tuple[int, ...]] = {}
        self.lock = threading.Lock()
        
        self.stats = {
            'total_batches': 0,
            'shared_batches': 0,
            'tokens_shared': 0,
            'computations_saved': 0,
        }
        
        self._LOG = PiscesLxLogger("popss.ops.infer.batch_sharing")
    
    def compute_similarity(self, input_ids_1: Tuple[int, ...], input_ids_2: Tuple[int, ...]) -> float:
        """
        Compute similarity between two token sequences.
        
        Uses Jaccard similarity with position weighting.
        """
        set1 = set(input_ids_1)
        set2 = set(input_ids_2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        jaccard = intersection / union if union > 0 else 0.0
        
        position_weight = 0.0
        min_len = min(len(input_ids_1), len(input_ids_2))
        for i in range(min_len):
            if input_ids_1[i] == input_ids_2[i]:
                position_weight += 1.0 / (i + 1)
        
        position_weight /= min_len
        
        return 0.4 * jaccard + 0.6 * position_weight
    
    def find_similar_requests(
        self, 
        requests: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group similar requests for batch processing.
        
        Args:
            requests: List of request dictionaries with 'input_ids' key
        
        Returns:
            List of grouped requests suitable for batch processing
        """
        with self.lock:
            if len(requests) <= 1:
                return [requests]
            
            groups = []
            processed = set()
            
            for i, req in enumerate(requests):
                if i in processed:
                    continue
                
                group = [req]
                processed.add(i)
                
                for j, other_req in enumerate(requests):
                    if j in processed:
                        continue
                    
                    input_ids_i = tuple(req.get('input_ids', []))
                    input_ids_j = tuple(other_req.get('input_ids', []))
                    
                    similarity = self.compute_similarity(input_ids_i, input_ids_j)
                    
                    if similarity >= self.similarity_threshold:
                        group.append(other_req)
                        processed.add(j)
                
                if len(group) > 1:
                    self.stats['shared_batches'] += 1
                
                groups.append(group)
            
            self.stats['total_batches'] += len(groups)
            
            return groups
    
    def share_prefix_kv(
        self,
        requests: List[Dict[str, Any]],
        shared_kv_cache: Dict[str, Any],
    ) -> Dict[int, List[int]]:
        """
        Identify which requests can share prefix KV caches.
        
        Returns:
            Mapping from request index to list of shared prefix indices
        """
        sharing_map: Dict[int, List[int]] = {}
        
        for i, req in enumerate(requests):
            input_ids = tuple(req.get('input_ids', []))
            if not input_ids:
                continue
            
            shared_prefixes: List[int] = []
            
            for prefix_hash, prefix_ids in self.shared_prefixes.items():
                if input_ids[:len(prefix_ids)] == prefix_ids:
                    shared_prefixes.append(hash(prefix_hash))
            
            if shared_prefixes:
                sharing_map[i] = shared_prefixes
                self.stats['tokens_shared'] += len(shared_prefixes)
        
        return sharing_map
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sharing statistics."""
        with self.lock:
            return {
                **self.stats,
                'shared_ratio': (
                    self.stats['shared_batches'] / self.stats['total_batches']
                    if self.stats['total_batches'] > 0 else 0.0
                ),
            }


class LoadBalancer:
    """
    MoE load balancer for expert utilization optimization.
    
    Computes auxiliary load balancing losses and tracks expert utilization
    to enable runtime optimizations. Provides real-time feedback for adaptive
    routing temperature adjustment.
    
    Features:
        - Per-token expert selection tracking
        - Expert utilization statistics
        - Load balancing auxiliary loss computation
        - Utilization-based routing adjustments
    """
    
    def __init__(
        self,
        num_experts: int,
        loss_weight: float = 0.01,
        utilization_threshold: float = 0.15,
        enable_adaptive_weighting: bool = True,
    ):
        self.num_experts = num_experts
        self.loss_weight = loss_weight
        self.utilization_threshold = utilization_threshold
        self.enable_adaptive_weighting = enable_adaptive_weighting
        
        self.expert_stats: Dict[int, ExpertStats] = {
            i: ExpertStats(expert_id=i) for i in range(num_experts)
        }
        self.expert_calls: List[int] = []
        self.lock = threading.Lock()
        
        self._LOG = PiscesLxLogger("popss.ops.infer.load_balancer")
    
    def record_routing(
        self,
        expert_indices: torch.Tensor,
        token_weights: Optional[torch.Tensor] = None,
    ):
        """
        Record expert routing decisions for load balancing.
        
        Args:
            expert_indices: Tensor of shape [num_tokens, top_k] with expert indices
            token_weights: Optional tensor of weights for each token
        """
        with self.lock:
            for i in range(expert_indices.shape[0]):
                for j in range(expert_indices.shape[1]):
                    expert_id = expert_indices[i, j].item()
                    if expert_id in self.expert_stats:
                        weight = token_weights[i].item() if token_weights is not None else 1.0
                        self.expert_stats[expert_id].call_count += 1
                        self.expert_stats[expert_id].total_tokens += int(weight > 0)
    
    def compute_load_balance_loss(
        self, 
        routing_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        
        The loss encourages uniform expert utilization by penalizing
        deviation from uniform distribution.
        
        Args:
            routing_probs: Tensor of shape [num_experts] with routing probabilities
        
        Returns:
            Load balancing loss scalar
        """
        target_distribution = torch.ones_like(routing_probs) / routing_probs.size(0)
        
        loss = F.mse_loss(routing_probs, target_distribution)
        
        return self.loss_weight * loss
    
    def get_utilization_ratios(self) -> torch.Tensor:
        """
        Get current expert utilization ratios.
        
        Returns:
            Tensor of shape [num_experts] with utilization ratios
        """
        with self.lock:
            total_calls = sum(
                stats.call_count for stats in self.expert_stats.values()
            )
            
            if total_calls == 0:
                return torch.zeros(self.num_experts)
            
            utilization = torch.zeros(self.num_experts)
            for expert_id, stats in self.expert_stats.items():
                utilization[expert_id] = stats.call_count / total_calls
            
            return utilization
    
    def get_overutilized_experts(self, threshold: Optional[float] = None) -> List[int]:
        """
        Get list of over-utilized expert IDs.
        
        Args:
            threshold: Custom threshold (defaults to utilization_threshold)
        
        Returns:
            List of over-utilized expert IDs
        """
        if threshold is None:
            threshold = self.utilization_threshold
        
        with self.lock:
            utilization = self.get_utilization_ratios()
            
            target = 1.0 / self.num_experts
            overutilized = [
                i for i in range(self.num_experts)
                if utilization[i].item() > target * (1 + threshold)
            ]
            
            return overutilized
    
    def get_underutilized_experts(self, threshold: Optional[float] = None) -> List[int]:
        """
        Get list of under-utilized expert IDs.
        
        Args:
            threshold: Custom threshold (defaults to utilization_threshold)
        
        Returns:
            List of under-utilized expert IDs
        """
        if threshold is None:
            threshold = self.utilization_threshold
        
        with self.lock:
            utilization = self.get_utilization_ratios()
            
            target = 1.0 / self.num_experts
            underutilized = [
                i for i in range(self.num_experts)
                if utilization[i].item() < target * (1 - threshold)
            ]
            
            return underutilized
    
    def reset(self):
        """Reset load balancing statistics."""
        with self.lock:
            for stats in self.expert_stats.values():
                stats.call_count = 0
                stats.total_tokens = 0
            self.expert_calls.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            utilization = self.get_utilization_ratios()
            
            overutilized = self.get_overutilized_experts()
            underutilized = self.get_underutilized_experts()
            
            return {
                'utilization': utilization.tolist(),
                'utilization_std': float(utilization.std()),
                'overutilized_count': len(overutilized),
                'underutilized_count': len(underutilized),
                'overutilized_experts': overutilized,
                'underutilized_experts': underutilized,
                'balance_score': 1.0 - float(utilization.std()) * 2,
            }


class POPSSMoERuntimeOperator(PiscesLxOperatorInterface):
    """
    MoE Runtime Inference Operator
    
    High-performance operator for Mixture of Experts inference optimization.
    Provides adaptive routing, expert caching, load balancing, and dynamic
    capacity management for optimal inference performance.
    
    This operator integrates with existing MoE model implementations to provide
    inference-specific optimizations that are distinct from training-time concerns.
    
    Attributes:
        config: MoERuntimeConfig for runtime behavior
        expert_cache: ExpertCacheManager for computation caching
        load_balancer: LoadBalancer for utilization tracking
        batch_sharing: BatchExpertSharing for cross-request optimization
        
    Example:
        >>> operator = POPSSMoERuntimeOperator()
        >>> result = operator.execute({
        ...     'hidden_states': hidden_states,
        ...     'action': 'route',
        ...     'expert_indices': selected_experts,
        ... })
        
    See Also:
        - YvMoELayer: Base MoE layer implementation
        - YvDynamicMoELayer: Dynamic MoE with expert migration
        - model.config.YvConfig: Model configuration with MoE parameters
    """
    
    def __init__(self, config: Optional[POPSSMoERuntimeConfig] = None):
        """
        Initialize the MoE runtime operator.
        
        Args:
            config: Optional POPSSMoERuntimeConfig instance (uses default if not provided)
        """
        super().__init__()
        self.name = "infer.moe_runtime"
        self.version = VERSION
        self.type = "inference"
        self._LOG = PiscesLxLogger("pisceslx.ops.infer.moe_runtime")
        
        self.config = config or POPSSMoERuntimeConfig()
        self.num_experts = 64
        self.expert_capacity = 64
        
        self._adaptive_temp = self.config.routing_temp
        self._adaptive_step_count = 0
        self._model_gate = None
        self._model = None
        
        self._initialize_components()
        
        self._LOG.info(
            f"MoE Runtime Operator initialized: "
            f"top_k={self.config.top_k}, "
            f"temp={self.config.routing_temp}, "
            f"cache_size={self.config.expert_cache_size}"
        )
    
    def _initialize_components(self):
        """Initialize internal components based on configuration."""
        self.expert_cache = ExpertCacheManager(
            max_cache_size=self.config.expert_cache_size,
            eviction_policy="lru",
        )
        
        self.batch_sharing = BatchExpertSharing(
            similarity_threshold=0.85,
            prefix_overlap_threshold=0.7,
            max_batch_size=16,
        )
        
        self.load_balancer = None
        
        self._request_queue = deque()
        self._batch_buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def _initialize_load_balancer(self, num_experts: int):
        """Initialize load balancer with expert count."""
        if self.load_balancer is None:
            self.load_balancer = LoadBalancer(
                num_experts=num_experts,
                loss_weight=self.config.load_balance_alpha,
                utilization_threshold=0.15,
                enable_adaptive_weighting=self.config.enable_adaptive_temp,
            )
            self.num_experts = num_experts
    
    def set_model(self, model: nn.Module):
        """
        Set the model for MoE operations.
        
        Extracts MoE gates and layers from the model for runtime optimization.
        
        Args:
            model: PyTorch model containing MoE layers
        """
        self._model = model
        
        gate_found = False
        for module in model.modules():
            if hasattr(module, 'gate') and hasattr(module, 'num_experts'):
                self._model_gate = module.gate
                self.num_experts = getattr(module, 'num_experts', 64)
                self.expert_capacity = getattr(module, 'expert_capacity', 64)
                gate_found = True
                self._LOG.info(
                    f"Found MoE gate with {self.num_experts} experts, "
                    f"capacity={self.expert_capacity}"
                )
                break
        
        if not gate_found:
            self._LOG.warning("No MoE gate found in model")
        
        self._initialize_load_balancer(self.num_experts)
    
    @property
    def description(self) -> str:
        return "MoE runtime inference operator with adaptive routing and caching"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "hidden_states": {"type": "Tensor", "required": True, 
                             "description": "Input hidden states for MoE routing"},
            "action": {"type": "str", "required": False, 
                      "default": "route",
                      "description": "Operation action: route, balance, cache, stats"},
            "expert_indices": {"type": "tuple", "required": False,
                              "description": "Pre-computed expert indices for routing"},
            "token_weights": {"type": "Tensor", "required": False,
                             "description": "Optional weights for each token"},
            "return_aux_loss": {"type": "bool", "required": False,
                               "default": False,
                               "description": "Whether to return load balancing loss"},
            "temperature": {"type": "float", "required": False,
                          "description": "Override routing temperature"},
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "expert_indices": {"type": "Tensor", "description": "Selected expert indices"},
            "routing_probs": {"type": "Tensor", "description": "Routing probabilities"},
            "load_balance_loss": {"type": "Tensor", "description": "Auxiliary loss if requested"},
            "routing_temp": {"type": "float", "description": "Current routing temperature"},
            "cache_hit": {"type": "bool", "description": "Whether cache was hit"},
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        if "hidden_states" not in inputs:
            self._LOG.error("Missing required parameter: hidden_states")
            return False
        return True
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute MoE runtime inference operations.
        
        Args:
            inputs: Dictionary containing:
                - hidden_states: Input tensor for routing
                - action: Operation type (route, balance, cache, stats)
                - expert_indices: Optional pre-computed expert indices
                - token_weights: Optional token weights
                - return_aux_loss: Whether to compute load balancing loss
                - temperature: Optional temperature override
        
        Returns:
            PiscesLxOperatorResult with routing outputs and statistics
        """
        start_time = time.time()
        
        try:
            if not self.validate_inputs(inputs):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Invalid input parameters",
                    execution_time=time.time() - start_time,
                )
            
            action = inputs.get("action", "route")
            
            if action == "route":
                return self._route(hidden_states=inputs["hidden_states"],
                                  expert_indices=inputs.get("expert_indices"),
                                  token_weights=inputs.get("token_weights"),
                                  return_aux_loss=inputs.get("return_aux_loss", False),
                                  temperature=inputs.get("temperature"),
                                  start_time=start_time)
            
            elif action == "balance":
                return self._compute_load_balance(inputs.get("routing_probs"),
                                                 start_time)
            
            elif action == "cache":
                return self._handle_cache(inputs, start_time)
            
            elif action == "stats":
                return self._get_stats(start_time)
            
            elif action == "batch":
                return self._handle_batch(inputs, start_time)
            
            elif action == "adaptive":
                return self._adaptive_adjust(inputs, start_time)
            
            else:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error=f"Unknown action: {action}",
                    execution_time=time.time() - start_time,
                )
                
        except Exception as e:
            self._LOG.error(f"MoE runtime operation failed: {str(e)}", exc_info=True)
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def _route(
        self,
        hidden_states: torch.Tensor,
        expert_indices: Optional[Tuple[int, ...]] = None,
        token_weights: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False,
        temperature: Optional[float] = None,
        start_time: float = 0.0,
    ) -> PiscesLxOperatorResult:
        """
        Route tokens to top-k experts.
        
        Performs adaptive routing with temperature scaling, optionally using
        pre-computed expert indices for consistency across generations.
        """
        try:
            current_temp = temperature or self._adaptive_temp
            
            batch_size, hidden_dim = hidden_states.shape[:2]
            num_tokens = batch_size
            
            if self._model_gate is not None:
                routing_logits = self._model_gate(hidden_states)
                
                if routing_logits.shape[-1] != self.num_experts:
                    self._LOG.warning(
                        f"Gate output shape {routing_logits.shape[-1]} != "
                        f"num_experts {self.num_experts}"
                    )
                
                routing_temp = torch.tensor(current_temp, device=routing_logits.device)
                routing_probs = F.softmax(routing_logits / routing_temp, dim=-1)
                
                routing_probs_flat = routing_probs.reshape(-1, routing_probs.size(-1))
                
                top_k_probs, top_k_indices = torch.topk(
                    routing_probs_flat, 
                    k=min(self.config.top_k, routing_probs_flat.size(-1)),
                    dim=-1
                )
                
                if self.load_balancer is not None:
                    self.load_balancer.record_routing(top_k_indices, token_weights)
                
                if self._should_adapt_temp():
                    self._adaptive_temp = self._update_routing_temperature()
                
                routing_probs_output = routing_probs
                load_balance_loss = None
                
                if return_aux_loss:
                    expert_probs = routing_probs_flat.mean(dim=0)
                    load_balance_loss = self.load_balancer.compute_load_balance_loss(expert_probs)
                
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output={
                        "expert_indices": top_k_indices,
                        "routing_probs": top_k_probs,
                        "routing_temp": current_temp,
                        "load_balance_loss": load_balance_loss,
                    },
                    execution_time=time.time() - start_time,
                )
            else:
                if expert_indices is not None:
                    fallback_indices = torch.tensor(
                        [list(expert_indices) for _ in range(num_tokens)],
                        dtype=torch.long,
                        device=hidden_states.device
                    )
                    fallback_probs = torch.ones(
                        num_tokens, self.config.top_k,
                        device=hidden_states.device
                    ) / self.config.top_k
                    
                    return PiscesLxOperatorResult(
                        operator_name=self.name,
                        status=PiscesLxOperatorStatus.SUCCESS,
                        output={
                            "expert_indices": fallback_indices,
                            "routing_probs": fallback_probs,
                            "routing_temp": current_temp,
                            "load_balance_loss": None,
                        },
                        execution_time=time.time() - start_time,
                    )
                
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="No MoE gate found and no expert indices provided",
                    execution_time=time.time() - start_time,
                )
                
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Routing failed: {str(e)}",
                execution_time=time.time() - start_time,
            )
    
    def _should_adapt_temp(self) -> bool:
        """Check if temperature adaptation is needed."""
        if not self.config.enable_adaptive_temp:
            return False
        
        self._adaptive_step_count += 1
        
        return (self._adaptive_step_count % self.config.adaptive_temp_interval == 0)
    
    def _update_routing_temperature(self) -> float:
        """Update routing temperature based on load balancing."""
        if self.load_balancer is None:
            return self._adaptive_temp
        
        overutilized = self.load_balancer.get_overutilized_experts()
        underutilized = self.load_balancer.get_underutilized_experts()
        
        if len(overutilized) > 0 or len(underutilized) > 0:
            new_temp = self._adaptive_temp + self.config.adaptive_temp_step
            
            new_temp = min(new_temp, self.config.adaptive_temp_cap)
            new_temp = max(new_temp, 0.5)
            
            self._LOG.debug(
                f"Adaptive temp update: {self._adaptive_temp:.4f} -> {new_temp:.4f}, "
                f"over={len(overutilized)}, under={len(underutilized)}"
            )
            
            return new_temp
        
        return self._adaptive_temp
    
    def _compute_load_balance(
        self,
        routing_probs: Optional[torch.Tensor] = None,
        start_time: float = 0.0,
    ) -> PiscesLxOperatorResult:
        """Compute load balancing loss and statistics."""
        if self.load_balancer is None:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Load balancer not initialized",
                execution_time=time.time() - start_time,
            )
        
        loss = None
        if routing_probs is not None:
            expert_probs = routing_probs.mean(dim=0)
            loss = self.load_balancer.compute_load_balance_loss(expert_probs)
        
        utilization = self.load_balancer.get_utilization_ratios()
        balance_score = 1.0 - float(utilization.std()) * 2
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                "utilization": utilization.tolist(),
                "balance_score": balance_score,
                "load_balance_loss": loss,
            },
            execution_time=time.time() - start_time,
            metadata=self.load_balancer.get_stats(),
        )
    
    def _handle_cache(
        self,
        inputs: Dict[str, Any],
        start_time: float,
    ) -> PiscesLxOperatorResult:
        """Handle expert cache operations."""
        cache_action = inputs.get("cache_action", "lookup")
        
        if cache_action == "lookup":
            hidden_states = inputs.get("hidden_states")
            expert_indices = tuple(inputs.get("expert_indices", []))
            
            if hidden_states is not None:
                cached = self.expert_cache.get(hidden_states, expert_indices)
                
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output={
                        "cached_outputs": cached,
                        "cache_hit": cached is not None,
                    },
                    execution_time=time.time() - start_time,
                    metadata=self.expert_cache.get_stats(),
                )
        
        elif cache_action == "insert":
            hidden_states = inputs.get("hidden_states")
            expert_indices = tuple(inputs.get("expert_indices", []))
            outputs = inputs.get("outputs")
            
            if hidden_states is not None and outputs is not None:
                self.expert_cache.insert(hidden_states, expert_indices, outputs)
                
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output={"inserted": True},
                    execution_time=time.time() - start_time,
                )
        
        elif cache_action == "clear":
            self.expert_cache.clear()
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"cleared": True},
                execution_time=time.time() - start_time,
            )
        
        elif cache_action == "stats":
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=self.expert_cache.get_stats(),
                execution_time=time.time() - start_time,
            )
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.FAILED,
            error=f"Unknown cache action: {cache_action}",
            execution_time=time.time() - start_time,
        )
    
    def _handle_batch(
        self,
        inputs: Dict[str, Any],
        start_time: float,
    ) -> PiscesLxOperatorResult:
        """Handle batch processing with expert sharing."""
        requests = inputs.get("requests", [])
        
        if not requests:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"batched_requests": [], "shared_count": 0},
                execution_time=time.time() - start_time,
            )
        
        groups = self.batch_sharing.find_similar_requests(requests)
        
        shared_count = sum(len(g) - 1 for g in groups if len(g) > 1)
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                "batched_requests": groups,
                "shared_count": shared_count,
            },
            execution_time=time.time() - start_time,
            metadata=self.batch_sharing.get_stats(),
        )
    
    def _adaptive_adjust(
        self,
        inputs: Dict[str, Any],
        start_time: float,
    ) -> PiscesLxOperatorResult:
        """Perform adaptive adjustments based on current state."""
        adjustment_type = inputs.get("adjustment_type", "temperature")
        
        if adjustment_type == "temperature":
            new_temp = self._update_routing_temperature()
            old_temp = self._adaptive_temp
            self._adaptive_temp = new_temp
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "old_temperature": old_temp,
                    "new_temperature": new_temp,
                    "adjustment_type": "temperature",
                },
                execution_time=time.time() - start_time,
            )
        
        elif adjustment_type == "capacity":
            if self.load_balancer:
                overutilized = self.load_balancer.get_overutilized_experts()
                underutilized = self.load_balancer.get_underutilized_experts()
                
                capacity_adjustment = 0
                if len(overutilized) > len(underutilized):
                    capacity_adjustment = int(self.expert_capacity * self.config.capacity_margin)
                elif len(underutilized) > len(overutilized):
                    capacity_adjustment = -int(self.expert_capacity * self.config.capacity_margin)
                
                new_capacity = max(
                    self.config.min_capacity,
                    self.expert_capacity + capacity_adjustment
                )
                
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output={
                        "old_capacity": self.expert_capacity,
                        "new_capacity": new_capacity,
                        "overutilized_count": len(overutilized),
                        "underutilized_count": len(underutilized),
                        "adjustment_type": "capacity",
                    },
                    execution_time=time.time() - start_time,
                )
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Load balancer not initialized",
                execution_time=time.time() - start_time,
            )
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.FAILED,
            error=f"Unknown adjustment type: {adjustment_type}",
            execution_time=time.time() - start_time,
        )
    
    def _get_stats(self, start_time: float) -> PiscesLxOperatorResult:
        """Get comprehensive operator statistics."""
        stats = {
            "config": self.config.to_dict(),
            "adaptive_temperature": self._adaptive_temp,
            "step_count": self._adaptive_step_count,
        }
        
        if self.load_balancer:
            stats["load_balancer"] = self.load_balancer.get_stats()
        
        stats["expert_cache"] = self.expert_cache.get_stats()
        stats["batch_sharing"] = self.batch_sharing.get_stats()
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.SUCCESS,
            output=stats,
            execution_time=time.time() - start_time,
        )
    
    def reset(self):
        """Reset operator state."""
        self._adaptive_temp = self.config.routing_temp
        self._adaptive_step_count = 0
        
        if self.load_balancer:
            self.load_balancer.reset()
        
        self.expert_cache.clear()
        
        self._LOG.info("MoE Runtime Operator state reset")
    
    def cleanup(self):
        """Cleanup resources."""
        self.reset()
        self.expert_cache.clear()
        
        self._LOG.info("MoE Runtime Operator cleaned up")


def create_moe_runtime_operator(config: Optional[POPSSMoERuntimeConfig] = None) -> 'POPSSMoERuntimeOperator':
    """
    Factory function to create MoE runtime operator.
    
    Args:
        config: Optional POPSSMoERuntimeConfig instance
    
    Returns:
        Configured POPSSMoERuntimeOperator instance
    """
    return POPSSMoERuntimeOperator(config=config)


__all__ = [
    'POPSSMoERuntimeOperator',
    'POPSSMoERuntimeConfig',
    'POPSSMoERoutingStrategy',
]
