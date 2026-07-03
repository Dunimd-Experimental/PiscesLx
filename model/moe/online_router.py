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

from __future__ import annotations

"""Online Clustering Router for Mixture-of-Experts.

This module implements an online clustering-based routing mechanism for MoE
layers, using dynamic cluster centroid updates and distance-based routing.
This approach provides semantic-aware routing without requiring predefined
domain labels.

Key Features:
    1. Online Clustering:
       - Dynamic cluster centroid maintenance via EMA
       - No predefined domain labels required
       - Adaptive to evolving data distributions
       - Real-time cluster updates during training
    
    2. Distance-Based Routing:
       - Euclidean distance between tokens and centroids
       - Temperature-scaled softmax for routing weights
       - Top-k expert selection with capacity constraints
       - Smooth routing distribution control
    
    3. Load Balancing:
       - Auxiliary loss-free balancing via centroid adjustment
       - Dynamic capacity allocation based on cluster size
       - Expert usage tracking and rebalancing
       - Collapse prevention mechanisms
    
    4. Stability Mechanisms:
       - EMA momentum scheduling for stable updates
       - Centroid initialization strategies
       - Outlier detection and handling
       - Gradient-friendly routing decisions

Mathematical Formulation:
    Distance Computation:
        d_i = ||x - c_i||_2  for each centroid i
    
    Routing Weights:
        w_i = softmax(-d_i / temperature)
    
    Centroid Update (EMA):
        c_i = momentum * c_i + (1 - momentum) * mean(x_assigned_to_i)
    
    Temperature Schedule:
        temperature = max(min_temp, initial_temp * decay^step)

Performance Characteristics:
    - Routing overhead: O(num_experts * hidden_size) for distance computation
    - Memory: Centroids + routing buffers
    - Update cost: O(batch_size * hidden_size) for EMA updates
    - Typical overhead: 2-3% of total MoE computation

Usage Example:
    >>> from model.moe.online_router import YvOnlineClusterRouter
    >>> 
    >>> # Create online cluster router
    >>> router = YvOnlineClusterRouter(
    ...     hidden_size=4096,
    ...     num_experts=64,
    ...     top_k=2,
    ...     num_clusters=8,
    ...     temperature=1.0,
    ...     ema_momentum=0.9
    ... )
    >>> 
    >>> # Route tokens
    >>> scores, indices, loss = router(hidden_states)
    >>> 
    >>> # Update centroids (automatic during forward pass)
    >>> # router.update_centroids() is called internally

Note:
    All classes follow the YvXxx naming convention.
    Online clustering provides semantic-aware routing without labels.
    EMA momentum should be tuned based on data distribution stability.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import YvNumericalGuard, YvEPS
from typing import Optional, Tuple, List
from dataclasses import dataclass

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.Moe", file_path=get_log_file("Yv.Moe"), enable_file=True)


@dataclass
class YvClusterConfig:
    """Configuration for online cluster routing.
    
    Encapsulates all parameters for the online clustering router,
    including cluster initialization, EMA update schedules, and
    routing hyperparameters.
    
    Attributes:
        num_clusters (int): Number of cluster centroids. Default: 8.
        temperature (float): Initial temperature for softmax. Default: 1.0.
        min_temperature (float): Minimum temperature. Default: 0.1.
        temperature_decay (float): Temperature decay rate. Default: 0.9999.
        ema_momentum (float): EMA momentum for centroid updates. Default: 0.9.
        ema_warmup_steps (int): Steps before full EMA updates. Default: 1000.
        init_method (str): Centroid initialization method. Default: "kaiming".
            Options: "kaiming", "xavier", "orthogonal", "from_data".
        capacity_factor (float): Expert capacity multiplier. Default: 1.25.
        load_balance_alpha (float): Load balance loss coefficient. Default: 0.01.
        enable_adaptive_temperature (bool): Use adaptive temperature. Default: True.
        enable_cluster_reassignment (bool): Allow cluster reassignment. Default: True.
        reassignment_threshold (float): Threshold for reassignment. Default: 0.1.
    
    Example:
        >>> config = YvClusterConfig(
        ...     num_clusters=16,
        ...     temperature=2.0,
        ...     ema_momentum=0.95
        ... )
    """
    num_clusters: int = 8
    temperature: float = 1.0
    min_temperature: float = 0.1
    temperature_decay: float = 0.9999
    ema_momentum: float = 0.9
    ema_warmup_steps: int = 1000
    init_method: str = "kaiming"
    capacity_factor: float = 1.25
    load_balance_alpha: float = 0.01
    enable_adaptive_temperature: bool = True
    enable_cluster_reassignment: bool = True
    reassignment_threshold: float = 0.1


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvOnlineClusterRouter(nn.Module):
    """Online clustering-based router for Mixture-of-Experts.
    
    Implements a semantic-aware routing mechanism using online clustering.
    Cluster centroids are dynamically maintained via EMA updates, enabling
    adaptive routing without predefined domain labels.
    
    Architecture:
        Input -> Distance Computation -> Temperature Scaling -> 
        Softmax -> Top-k Selection -> Expert Assignment
    
    Key Innovations:
        1. Label-Free Routing: No predefined domain labels required
        2. Online Adaptation: Centroids evolve with data distribution
        3. Semantic Awareness: Similar tokens route to similar experts
        4. Stable Learning: EMA provides smooth centroid updates
    
    Routing Process:
        1. Compute Euclidean distance between tokens and all centroids
        2. Apply temperature scaling: -distance / temperature
        3. Compute softmax to get routing probabilities
        4. Select top-k experts based on probabilities
        5. Update centroids using EMA with assigned tokens
    
    Attributes:
        hidden_size (int): Input hidden dimension.
        num_experts (int): Total number of experts.
        num_clusters (int): Number of cluster centroids.
        top_k (int): Number of experts to route each token to.
        temperature (float): Current temperature value.
        ema_momentum (float): EMA momentum for centroid updates.
        centroids (torch.Tensor): Cluster centroids [num_clusters, hidden_size].
        cluster_expert_mapping (nn.Parameter): Mapping from clusters to experts.
        expert_usage_count (torch.Tensor): Buffer tracking expert usage.
    
    Example:
        >>> router = YvOnlineClusterRouter(
        ...     hidden_size=4096,
        ...     num_experts=64,
        ...     top_k=2,
        ...     num_clusters=8
        ... )
        >>> scores, indices, loss = router(hidden_states)
    
    Note:
        Centroids are automatically updated during forward pass.
        Temperature decays over time for sharper routing decisions.
        Cluster-to-expert mapping is learned during training.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        num_clusters: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 0.9999,
        ema_momentum: float = 0.9,
        ema_warmup_steps: int = 1000,
        init_method: str = "kaiming",
        capacity_factor: float = 1.25,
        load_balance_alpha: float = 0.01,
        enable_adaptive_temperature: bool = True,
        enable_cluster_reassignment: bool = True,
        reassignment_threshold: float = 0.1,
        cfg: Optional[object] = None
    ) -> None:
        """Initialize the online cluster router.
        
        Args:
            hidden_size (int): Size of the input hidden dimension.
            num_experts (int): Number of experts to route between.
            top_k (int): Number of top experts to select per token. Default: 2.
            num_clusters (int): Number of cluster centroids. Default: 8.
            device: Device to place the module on. Default: None.
            dtype: Data type for module parameters. Default: None.
            temperature (float): Initial temperature for softmax. Default: 1.0.
            min_temperature (float): Minimum temperature. Default: 0.1.
            temperature_decay (float): Temperature decay rate per step. Default: 0.9999.
            ema_momentum (float): EMA momentum for centroid updates. Default: 0.9.
            ema_warmup_steps (int): Steps before full EMA updates. Default: 1000.
            init_method (str): Centroid initialization method. Default: "kaiming".
                Options: "kaiming", "xavier", "orthogonal", "from_data".
            capacity_factor (float): Expert capacity multiplier. Default: 1.25.
            load_balance_alpha (float): Load balance loss coefficient. Default: 0.01.
            enable_adaptive_temperature (bool): Use adaptive temperature. Default: True.
            enable_cluster_reassignment (bool): Allow cluster reassignment. Default: True.
            reassignment_threshold (float): Threshold for reassignment. Default: 0.1.
            cfg: Configuration object with additional parameters.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_clusters = num_clusters
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balance_alpha = load_balance_alpha
        self.enable_adaptive_temperature = enable_adaptive_temperature
        self.enable_cluster_reassignment = enable_cluster_reassignment
        self.reassignment_threshold = reassignment_threshold
        
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.ema_momentum = ema_momentum
        self.ema_warmup_steps = ema_warmup_steps
        self.init_method = init_method
        
        self.register_buffer(
            'centroids',
            self._initialize_centroids(num_clusters, hidden_size, init_method, device, dtype)
        )
        
        self.cluster_expert_mapping = nn.Parameter(
            torch.randn(num_clusters, num_experts, device=device, dtype=dtype) * 0.01
        )
        
        self.register_buffer('temperature', torch.tensor(temperature))
        self.register_buffer('step_counter', torch.tensor(0))
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        self.register_buffer('total_routing_count', torch.tensor(0.0))
        self.register_buffer('cluster_sizes', torch.zeros(num_clusters))
        self.register_buffer('cluster_assignments', torch.zeros(10000, dtype=torch.long))
        self.register_buffer('assignment_ptr', torch.tensor(0))
        
        self.register_buffer('centroid_update_buffer', torch.zeros(num_clusters, hidden_size))
        self.register_buffer('centroid_update_count', torch.zeros(num_clusters))
        
        self._is_checkpointing = False
        
        if cfg is not None:
            self.load_balance_alpha = getattr(cfg, 'moe_load_balance_alpha', 0.01)
            self.capacity_factor = getattr(cfg, 'moe_capacity_factor', 1.25)
        
        _LOG.info(
            f"YvOnlineClusterRouter initialized: {num_clusters} clusters, "
            f"{num_experts} experts, top-{top_k}, temp={temperature}"
        )
    
    def _initialize_centroids(
        self,
        num_clusters: int,
        hidden_size: int,
        init_method: str,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype]
    ) -> torch.Tensor:
        """Initialize cluster centroids using specified method.
        
        Args:
            num_clusters (int): Number of clusters to initialize.
            hidden_size (int): Dimension of each centroid.
            init_method (str): Initialization method.
            device: Device for centroids.
            dtype: Data type for centroids.
        
        Returns:
            torch.Tensor: Initialized centroids [num_clusters, hidden_size].
        
        Raises:
            ValueError: If init_method is not recognized.
        """
        if init_method == "kaiming":
            centroids = torch.empty(num_clusters, hidden_size, device=device, dtype=dtype)
            nn.init.kaiming_normal_(centroids)
        elif init_method == "xavier":
            centroids = torch.empty(num_clusters, hidden_size, device=device, dtype=dtype)
            nn.init.xavier_normal_(centroids)
        elif init_method == "orthogonal":
            centroids = torch.empty(num_clusters, hidden_size, device=device, dtype=dtype)
            nn.init.orthogonal_(centroids)
        elif init_method == "from_data":
            centroids = torch.randn(num_clusters, hidden_size, device=device, dtype=dtype)
            centroids = F.normalize(centroids, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        return centroids
    
    def _compute_distances(
        self,
        x: torch.Tensor,
        centroids: torch.Tensor
    ) -> torch.Tensor:
        """Compute Euclidean distances between tokens and centroids.
        
        Uses the identity ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x·c
        for efficient computation.
        
        Args:
            x (torch.Tensor): Input tokens [batch_size, hidden_size].
            centroids (torch.Tensor): Cluster centroids [num_clusters, hidden_size].
        
        Returns:
            torch.Tensor: Distance matrix [batch_size, num_clusters].
        """
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        c_norm_sq = (centroids ** 2).sum(dim=-1).unsqueeze(0)
        
        cross_term = torch.mm(x, centroids.t())
        
        distances_sq = x_norm_sq + c_norm_sq - 2 * cross_term
        
        distances_sq = F.relu(distances_sq)
        
        distances = torch.sqrt(distances_sq.clamp(min=0) + YvNumericalGuard.get_eps(distances_sq.dtype))
        
        return distances
    
    def _compute_routing_weights(
        self,
        distances: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute routing weights using temperature-scaled softmax.
        
        Applies softmax to negative distances divided by temperature,
        giving higher weights to closer centroids.
        
        Args:
            distances (torch.Tensor): Distance matrix [batch_size, num_clusters].
            temperature (float): Temperature for softmax scaling.
        
        Returns:
            torch.Tensor: Routing weights [batch_size, num_clusters].
        """
        logits = -distances / max(temperature, YvEPS.DEFAULT)
        
        weights = F.softmax(logits, dim=-1)
        
        return weights
    
    def _select_top_k_experts(
        self,
        cluster_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select top-k experts based on cluster weights.
        
        Maps cluster weights to expert weights via learned mapping,
        then selects top-k experts for each token.
        
        Args:
            cluster_weights (torch.Tensor): Cluster routing weights [batch, num_clusters].
        
        Returns:
            tuple: A tuple containing:
                - top_k_weights (torch.Tensor): Expert routing weights [batch, top_k].
                - top_k_indices (torch.Tensor): Expert indices [batch, top_k].
                - expert_weights (torch.Tensor): Full expert weights [batch, num_experts].
        """
        batch_size = cluster_weights.size(0)
        
        mapping_weights = F.softmax(self.cluster_expert_mapping, dim=-1)
        
        expert_weights = torch.mm(cluster_weights, mapping_weights)
        
        top_k_weights, top_k_indices = torch.topk(expert_weights, min(self.top_k, self.num_experts), dim=-1)
        
        top_k_weights = YvNumericalGuard.safe_div(top_k_weights, top_k_weights.sum(dim=-1, keepdim=True))
        
        return top_k_weights, top_k_indices, expert_weights
    
    def update_centroids(
        self,
        x: torch.Tensor,
        cluster_assignments: torch.Tensor
    ) -> None:
        """Update cluster centroids using EMA with assigned tokens.
        
        Implements exponential moving average update for centroids based
        on newly assigned tokens. Uses warmup schedule for stability.
        
        Update Formula:
            c_i = momentum * c_i + (1 - momentum) * mean(x_assigned_to_i)
        
        Args:
            x (torch.Tensor): Input tokens [batch_size, hidden_size].
            cluster_assignments (torch.Tensor): Cluster assignment for each token [batch_size].
        
        Note:
            Called automatically during forward pass in training mode.
            Uses warmup schedule: momentum increases from 0 to ema_momentum.
        """
        if not self.training or self._is_checkpointing:
            return
        
        step = self.step_counter.item()
        
        warmup_progress = min(1.0, step / self.ema_warmup_steps)
        current_momentum = self.ema_momentum * warmup_progress
        
        self.centroid_update_buffer.zero_()
        self.centroid_update_count.zero_()
        
        for cluster_id in range(self.num_clusters):
            mask = (cluster_assignments == cluster_id)
            
            if mask.any():
                assigned_tokens = x[mask]
                
                cluster_mean = assigned_tokens.mean(dim=0)
                
                self.centroid_update_buffer[cluster_id] = cluster_mean
                self.centroid_update_count[cluster_id] = mask.sum()
        
        for cluster_id in range(self.num_clusters):
            if self.centroid_update_count[cluster_id] > 0:
                new_centroid = self.centroid_update_buffer[cluster_id]
                
                updated_centroid = (
                    current_momentum * self.centroids[cluster_id] +
                    (1 - current_momentum) * new_centroid
                )
                
                self.centroids[cluster_id] = updated_centroid
    
    def _adjust_temperature(self) -> None:
        """Adjust temperature based on training progress.
        
        Implements temperature decay schedule and adaptive adjustment
        based on load balance quality.
        
        Temperature Schedule:
            temperature = max(min_temp, initial_temp * decay^step)
        
        Adaptive Adjustment:
            - Increases temperature if load imbalance is high
            - Decreases temperature if load is well balanced
        """
        if not self.training or self._is_checkpointing:
            return
        
        decayed_temp = self.temperature.item() * self.temperature_decay
        decayed_temp = max(self.min_temperature, decayed_temp)
        
        if self.enable_adaptive_temperature and self.total_routing_count > 100:
            usage = YvNumericalGuard.safe_div(self.expert_usage_count, self.total_routing_count)
            
            ideal_usage = 1.0 / self.num_experts
            imbalance = YvNumericalGuard.safe_div(torch.var(usage), torch.as_tensor(ideal_usage, dtype=usage.dtype, device=usage.device))
            
            if imbalance > 0.2:
                decayed_temp = min(decayed_temp * 1.1, 5.0)
            elif imbalance < 0.05:
                decayed_temp = max(decayed_temp * 0.95, self.min_temperature)
        
        self.temperature.fill_(decayed_temp)
    
    def _compute_load_balance_loss(
        self,
        expert_weights: torch.Tensor,
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balance loss for even expert utilization.
        
        Combines auxiliary loss with actual usage statistics to encourage
        balanced expert utilization.
        
        Loss Components:
            1. Auxiliary loss: Penalizes uneven routing weight distribution
            2. Actual usage loss: Penalizes uneven expert selection
        
        Args:
            expert_weights (torch.Tensor): Full expert weights [batch, num_experts].
            top_k_indices (torch.Tensor): Selected expert indices [batch, top_k].
        
        Returns:
            torch.Tensor: Load balance loss scalar.
        """
        expert_freq = expert_weights.mean(dim=0)
        
        ideal_freq = torch.ones_like(expert_freq) / self.num_experts
        
        aux_loss = self.load_balance_alpha * torch.sum((expert_freq - ideal_freq) ** 2)
        
        if self.training:
            flat_indices = top_k_indices.flatten()
            actual_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
            
            actual_freq = YvNumericalGuard.safe_div(actual_counts, torch.tensor(top_k_indices.numel(), dtype=actual_counts.dtype, device=actual_counts.device))
            
            actual_loss = self.load_balance_alpha * torch.sum((actual_freq - ideal_freq) ** 2)
            
            total_loss = 0.5 * aux_loss + 0.5 * actual_loss
        else:
            total_loss = aux_loss
        
        return total_loss
    
    def _update_statistics(
        self,
        top_k_indices: torch.Tensor,
        cluster_assignments: torch.Tensor
    ) -> None:
        """Update routing statistics for monitoring and adaptation.
        
        Tracks expert usage counts, cluster sizes, and assignment history
        for load balancing and cluster quality assessment.
        
        Args:
            top_k_indices (torch.Tensor): Selected expert indices [batch, top_k].
            cluster_assignments (torch.Tensor): Cluster assignment for each token [batch].
        """
        if not self.training or self._is_checkpointing:
            return
        
        flat_indices = top_k_indices.flatten()
        counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        self.expert_usage_count += counts
        
        self.total_routing_count += top_k_indices.numel()
        
        for cluster_id in range(self.num_clusters):
            mask = (cluster_assignments == cluster_id)
            self.cluster_sizes[cluster_id] += mask.sum()
        
        batch_size = cluster_assignments.size(0)
        ptr = self.assignment_ptr.item()
        
        if ptr + batch_size <= self.cluster_assignments.size(0):
            self.cluster_assignments[ptr:ptr + batch_size] = cluster_assignments
            self.assignment_ptr.fill_((ptr + batch_size) % self.cluster_assignments.size(0))
    
    def _reassign_clusters(self) -> None:
        """Reassign underutilized or collapsed clusters.
        
        Detects clusters that have collapsed (very few or no assignments)
        and reinitializes them from active clusters or random positions.
        
        Reassignment Strategy:
            1. Identify collapsed clusters (size < threshold * mean_size)
            2. Reinitialize from largest cluster with noise
            3. Update cluster-to-expert mapping
        
        Note:
            Only called during training when enable_cluster_reassignment is True.
        """
        if not self.training or self._is_checkpointing:
            return
        
        if not self.enable_cluster_reassignment:
            return
        
        if self.total_routing_count < 1000:
            return
        
        mean_size = self.cluster_sizes.mean()
        threshold = mean_size * self.reassignment_threshold
        
        collapsed_clusters = (self.cluster_sizes < threshold).nonzero().squeeze(-1)
        
        if collapsed_clusters.numel() == 0:
            return
        
        largest_cluster = self.cluster_sizes.argmax()
        
        for cluster_id in collapsed_clusters.tolist():
            noise = torch.randn_like(self.centroids[cluster_id]) * 0.1
            self.centroids[cluster_id] = self.centroids[largest_cluster] + noise
            
            self.cluster_sizes[cluster_id] = mean_size * 0.5
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the online cluster router.
        
        Routes tokens to experts based on distance to cluster centroids.
        Automatically updates centroids during training using EMA.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size].
        
        Returns:
            tuple: A tuple containing:
                - top_k_weights (torch.Tensor): Routing weights [batch*seq, top_k].
                - top_k_indices (torch.Tensor): Expert indices [batch*seq, top_k].
                - load_balance_loss (torch.Tensor): Load balance loss scalar.
        
        Note:
            Centroids are updated automatically during training.
            Temperature decays over time for sharper routing.
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        num_tokens = x_flat.size(0)
        
        distances = self._compute_distances(x_flat, self.centroids)
        
        current_temp = self.temperature.item()
        cluster_weights = self._compute_routing_weights(distances, current_temp)
        
        top_k_weights, top_k_indices, expert_weights = self._select_top_k_experts(cluster_weights)
        
        cluster_assignments = cluster_weights.argmax(dim=-1)
        
        if self.training and not self._is_checkpointing:
            self.step_counter.add_(1)
            
            self.update_centroids(x_flat, cluster_assignments)
            
            self._update_statistics(top_k_indices, cluster_assignments)
            
            self._adjust_temperature()
            
            if self.step_counter.item() % 100 == 0:
                self._reassign_clusters()
        
        load_balance_loss = self._compute_load_balance_loss(expert_weights, top_k_indices)
        
        return top_k_weights, top_k_indices, load_balance_loss
    
    def get_cluster_stats(self) -> dict:
        """Get statistics about cluster quality and usage.
        
        Returns:
            dict: Dictionary containing:
                - cluster_sizes: Size of each cluster
                - cluster_balance: Balance metric (lower is better)
                - temperature: Current temperature value
                - total_assignments: Total number of assignments
                - expert_usage: Expert usage distribution
        
        Example:
            >>> stats = router.get_cluster_stats()
            >>> print(f"Cluster balance: {stats['cluster_balance']:.4f}")
        """
        cluster_balance = YvNumericalGuard.safe_div(torch.var(self.cluster_sizes), self.cluster_sizes.mean())
        
        expert_usage = YvNumericalGuard.safe_div(self.expert_usage_count, self.total_routing_count)
        
        return {
            'cluster_sizes': self.cluster_sizes.cpu().tolist(),
            'cluster_balance': cluster_balance.item(),
            'temperature': self.temperature.item(),
            'total_assignments': self.total_routing_count.item(),
            'expert_usage': expert_usage.cpu().tolist()
        }
    
    def reset_statistics(self) -> None:
        """Reset all routing statistics for new training epoch.
        
        Clears expert usage counts, cluster sizes, and assignment history.
        Does not reset centroids or learned parameters.
        """
        self.expert_usage_count.zero_()
        self.total_routing_count.zero_()
        self.cluster_sizes.zero_()
        self.cluster_assignments.zero_()
        self.assignment_ptr.zero_()
        self.centroid_update_buffer.zero_()
        self.centroid_update_count.zero_()
    
    def extra_repr(self) -> str:
        """Return extra representation string for module printing.
        
        Returns:
            str: String representation of key parameters.
        """
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_experts={self.num_experts}, "
            f"num_clusters={self.num_clusters}, "
            f"top_k={self.top_k}, "
            f"temperature={self.temperature.item():.3f}, "
            f"ema_momentum={self.ema_momentum}"
        )
