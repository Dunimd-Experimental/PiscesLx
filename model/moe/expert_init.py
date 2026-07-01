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

"""Gradient Clustering Expert Initialization for Mixture-of-Experts.

This module implements gradient-based clustering for expert initialization,
using gradient statistics to discover natural parameter groupings and
initialize experts with specialized weights.

Key Components:
    1. Gradient Statistics Collection:
       - Mean: Average gradient magnitude per parameter
       - Std: Gradient variance across training samples
       - Norm: L2 norm of gradient vectors
       - Direction: Normalized gradient direction vectors

    2. K-means Clustering:
       - Discover natural parameter groupings
       - Automatic cluster number detection via elbow method
       - Initialize experts from cluster centroids

    3. Expert Initialization:
       - Weight initialization based on gradient clusters
       - Specialized initialization for each expert
       - Preserve learned representations from pre-training

Mathematical Foundation:
    Gradient Statistics:
        - mean_i = (1/N) * sum(g_i) for parameter i
        - std_i = sqrt((1/N) * sum((g_i - mean_i)^2))
        - norm_i = ||g_i||_2
        - direction_i = g_i / ||g_i||_2

    K-means Clustering:
        - Objective: minimize sum(||x - mu_c||^2)
        - Convergence: iterate assignment and update steps
        - Initialization: k-means++ for better convergence

    Elbow Method:
        - Compute within-cluster sum of squares (WCSS)
        - Find elbow point where WCSS decrease slows
        - Use second derivative to detect elbow

Usage Example:
    >>> from model.moe.expert_init import YvGradientClusterInitializer
    >>> 
    >>> # Initialize gradient cluster initializer
    >>> initializer = YvGradientClusterInitializer(
    ...     hidden_size=4096,
    ...     intermediate_size=11008,
    ...     num_experts=64,
    ...     min_clusters=4,
    ...     max_clusters=16
    ... )
    >>> 
    >>> # Collect gradient statistics during training
    >>> gradient_stats = initializer.collect_gradients(model, dataloader)
    >>> 
    >>> # Discover expert specializations via clustering
    >>> clusters = initializer.discover_expert_specializations(
    ...     gradient_stats,
    ...     auto_detect_k=True
    ... )
    >>> 
    >>> # Initialize experts from clusters
    >>> initializer.initialize_experts_from_clusters(
    ...     model.moe_layer,
    ...     clusters,
    ...     gradient_stats
    ... )

Note:
    All classes follow the YvXxx naming convention.
    Gradient clustering provides better expert specialization than random init.
    Recommended for transfer learning and fine-tuning scenarios.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.Moe", file_path=get_log_file("Yv.Moe"), enable_file=True)


@dataclass
class GradientStatistics:
    """Dataclass for storing gradient statistics per parameter.
    
    Encapsulates gradient statistics collected during forward/backward passes,
    including mean, variance, norm, and direction information for each parameter.
    
    Attributes:
        mean (torch.Tensor): Mean gradient values per parameter.
        std (torch.Tensor): Standard deviation of gradients per parameter.
        norm (torch.Tensor): L2 norm of gradient vectors.
        direction (torch.Tensor): Normalized gradient direction vectors.
        count (int): Number of gradient samples collected.
        param_shape (Tuple[int, ...]): Shape of the original parameter.
        param_name (str): Name of the parameter in the model.
    
    Example:
        >>> stats = GradientStatistics(
        ...     mean=torch.zeros(4096, 11008),
        ...     std=torch.ones(4096, 11008),
        ...     norm=torch.ones(4096, 11008),
        ...     direction=torch.zeros(4096, 11008),
        ...     count=1000,
        ...     param_shape=(4096, 11008),
        ...     param_name="expert.0.weight"
        ... )
    """
    mean: torch.Tensor
    std: torch.Tensor
    norm: torch.Tensor
    direction: torch.Tensor
    count: int
    param_shape: Tuple[int, ...]
    param_name: str


@dataclass
class ClusterInfo:
    """Dataclass for storing cluster information.
    
    Encapsulates information about a single cluster discovered during
    gradient clustering, including centroid, member indices, and statistics.
    
    Attributes:
        cluster_id (int): Unique identifier for the cluster.
        centroid (torch.Tensor): Centroid vector of the cluster.
        member_indices (List[int]): Indices of parameters in this cluster.
        member_names (List[str]): Names of parameters in this cluster.
        intra_cluster_variance (float): Variance within the cluster.
        size (int): Number of members in the cluster.
    
    Example:
        >>> cluster = ClusterInfo(
        ...     cluster_id=0,
        ...     centroid=torch.randn(128),
        ...     member_indices=[0, 5, 12, 45],
        ...     member_names=["layer.0.weight", "layer.2.weight", ...],
        ...     intra_cluster_variance=0.0234,
        ...     size=4
        ... )
    """
    cluster_id: int
    centroid: torch.Tensor
    member_indices: List[int]
    member_names: List[str]
    intra_cluster_variance: float
    size: int


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvGradientClusterInitializer:
    """Gradient-based clustering initializer for MoE experts.
    
    Implements gradient clustering to discover natural parameter groupings
    and initialize experts with specialized weights. This approach leverages
    gradient information from pre-training or fine-tuning to create
    specialized experts that capture different aspects of the data.
    
    Key Features:
        - Gradient statistics collection (mean, std, norm, direction)
        - K-means clustering for parameter grouping
        - Elbow method for automatic cluster detection
        - Expert initialization from cluster centroids
        - Support for hierarchical clustering
    
    Initialization Process:
        1. Collect gradient statistics during training
        2. Extract gradient features for each parameter
        3. Apply K-means clustering to discover groupings
        4. Initialize each expert from cluster centroid
        5. Fine-tune experts with specialized data
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        intermediate_size (int): Expert intermediate dimension.
        num_experts (int): Total number of experts to initialize.
        min_clusters (int): Minimum number of clusters for auto-detection.
        max_clusters (int): Maximum number of clusters for auto-detection.
        gradient_features (Dict[str, GradientStatistics]): Collected gradient stats.
        clusters (List[ClusterInfo]): Discovered parameter clusters.
        device (torch.device): Device for computation.
    
    Example:
        >>> initializer = YvGradientClusterInitializer(
        ...     hidden_size=4096,
        ...     intermediate_size=11008,
        ...     num_experts=64
        ... )
        >>> 
        >>> # During training, collect gradients
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     initializer.collect_gradients(model)
        ...     optimizer.step()
        >>> 
        >>> # After training, discover clusters
        >>> clusters = initializer.discover_expert_specializations()
        >>> 
        >>> # Initialize experts
        >>> initializer.initialize_experts_from_clusters(model.moe_layer)
    
    Note:
        Gradient clustering provides better expert specialization than random init.
        Best used when transferring from a pre-trained dense model.
        Collect gradients from diverse data samples for best results.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        min_clusters: int = 4,
        max_clusters: int = 16,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize the gradient cluster initializer.
        
        Args:
            hidden_size (int): Model hidden dimension.
            intermediate_size (int): Expert intermediate dimension.
            num_experts (int): Total number of experts to initialize.
            min_clusters (int): Minimum clusters for auto-detection. Default: 4.
            max_clusters (int): Maximum clusters for auto-detection. Default: 16.
            device (torch.device, optional): Device for computation. Default: None.
            dtype (torch.dtype, optional): Data type for tensors. Default: None.
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32
        
        self.gradient_features: Dict[str, GradientStatistics] = {}
        self.clusters: List[ClusterInfo] = []
        self._gradient_buffer: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._collection_count = 0
        
        _LOG.info(
            f"YvGradientClusterInitializer initialized: "
            f"hidden={hidden_size}, intermediate={intermediate_size}, "
            f"experts={num_experts}, clusters=[{min_clusters}, {max_clusters}]"
        )
    
    def collect_gradients(
        self,
        model: nn.Module,
        param_filter: Optional[str] = None
    ) -> Dict[str, GradientStatistics]:
        """Collect gradient statistics from model parameters.
        
        Analyzes gradients accumulated during backward pass and computes
        statistical features including mean, standard deviation, L2 norm,
        and direction vectors for each parameter.
        
        Gradient Statistics:
            - Mean: Average gradient magnitude, indicates parameter importance
            - Std: Gradient variance, indicates parameter stability
            - Norm: L2 norm of gradient, indicates overall gradient magnitude
            - Direction: Normalized gradient, indicates optimization direction
        
        Args:
            model (nn.Module): Model with accumulated gradients.
            param_filter (str, optional): Regex pattern to filter parameters.
                If None, collects gradients from all parameters with requires_grad.
                Default: None.
        
        Returns:
            Dict[str, GradientStatistics]: Dictionary mapping parameter names
                to their gradient statistics.
        
        Example:
            >>> loss = model(input_ids)
            >>> loss.backward()
            >>> stats = initializer.collect_gradients(model, param_filter="expert.*weight")
        
        Note:
            Should be called after backward() and before optimizer.step().
            Multiple calls accumulate statistics across training steps.
        """
        import re
        
        self._collection_count += 1
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            if param_filter is not None and not re.search(param_filter, name):
                continue
            
            grad = param.grad.detach().clone().to(device=self.device, dtype=self.dtype)
            grad_flat = grad.view(-1)
            
            if name not in self._gradient_buffer:
                self._gradient_buffer[name] = []
            
            self._gradient_buffer[name].append(grad_flat)
        
        self.gradient_features = self._compute_gradient_statistics()
        
        if self._collection_count % 100 == 0:
            _LOG.info(f"Collected gradients from {self._collection_count} steps")
        
        return self.gradient_features
    
    def _compute_gradient_statistics(self) -> Dict[str, GradientStatistics]:
        """Compute gradient statistics from collected gradient buffer.
        
        Processes accumulated gradients to compute mean, standard deviation,
        L2 norm, and direction vectors for each parameter.
        
        Returns:
            Dict[str, GradientStatistics]: Computed gradient statistics.
        """
        stats_dict = {}
        
        for name, grad_list in self._gradient_buffer.items():
            if len(grad_list) == 0:
                continue
            
            grads = torch.stack(grad_list, dim=0)
            
            mean = grads.mean(dim=0)
            std = grads.std(dim=0)
            norm = torch.norm(grads, p=2, dim=0)
            
            grad_norms = torch.norm(grads, p=2, dim=1, keepdim=True)
            grad_norms = grad_norms.clamp(min=1e-8)
            normalized_grads = grads / grad_norms
            direction = normalized_grads.mean(dim=0)
            direction = direction / (direction.norm(p=2) + 1e-8)
            
            param_shape = tuple(grad_list[0].shape)
            
            stats_dict[name] = GradientStatistics(
                mean=mean,
                std=std,
                norm=norm,
                direction=direction,
                count=len(grad_list),
                param_shape=param_shape,
                param_name=name
            )
        
        return stats_dict
    
    def discover_expert_specializations(
        self,
        gradient_stats: Optional[Dict[str, GradientStatistics]] = None,
        auto_detect_k: bool = True,
        num_clusters: Optional[int] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
        seed: Optional[int] = None
    ) -> List[ClusterInfo]:
        """Discover expert specializations using K-means clustering.
        
        Applies K-means clustering to gradient statistics to discover natural
        parameter groupings. Each cluster represents a potential expert
        specialization based on gradient patterns.
        
        Clustering Features:
            - Concatenated gradient statistics (mean, std, norm, direction)
            - Normalized features for fair comparison
            - K-means++ initialization for better convergence
        
        Args:
            gradient_stats (Dict[str, GradientStatistics], optional): Pre-computed
                gradient statistics. If None, uses self.gradient_features.
                Default: None.
            auto_detect_k (bool): Whether to automatically detect optimal number
                of clusters using elbow method. Default: True.
            num_clusters (int, optional): Number of clusters. Required if
                auto_detect_k is False. Default: None.
            max_iterations (int): Maximum K-means iterations. Default: 100.
            tolerance (float): Convergence tolerance. Default: 1e-4.
            seed (int, optional): Random seed for reproducibility. Default: None.
        
        Returns:
            List[ClusterInfo]: List of discovered clusters with member information.
        
        Raises:
            ValueError: If no gradient statistics available or invalid num_clusters.
        
        Example:
            >>> clusters = initializer.discover_expert_specializations(
            ...     auto_detect_k=True,
            ...     max_iterations=200
            ... )
            >>> print(f"Discovered {len(clusters)} expert specializations")
        
        Note:
            Uses elbow method to find optimal cluster count automatically.
            K-means++ initialization ensures good cluster quality.
        """
        if gradient_stats is None:
            gradient_stats = self.gradient_features
        
        if len(gradient_stats) == 0:
            raise ValueError("No gradient statistics available. Call collect_gradients first.")
        
        features, param_names = self._extract_clustering_features(gradient_stats)
        
        if auto_detect_k:
            num_clusters = self.auto_detect_clusters(
                features,
                min_k=self.min_clusters,
                max_k=min(self.max_clusters, features.shape[0])
            )
            _LOG.info(f"Auto-detected optimal cluster count: {num_clusters}")
        elif num_clusters is None:
            raise ValueError("num_clusters must be specified when auto_detect_k is False")
        
        centroids, labels = self._kmeans_clustering(
            features,
            num_clusters,
            max_iterations,
            tolerance,
            seed
        )
        
        self.clusters = self._build_cluster_info(
            centroids,
            labels,
            param_names,
            features
        )
        
        _LOG.info(
            f"Discovered {len(self.clusters)} clusters: "
            f"sizes = {[c.size for c in self.clusters]}"
        )
        
        return self.clusters
    
    def _extract_clustering_features(
        self,
        gradient_stats: Dict[str, GradientStatistics]
    ) -> Tuple[torch.Tensor, List[str]]:
        """Extract clustering features from gradient statistics.
        
        Concatenates and normalizes gradient statistics to create feature
        vectors suitable for clustering.
        
        Args:
            gradient_stats: Dictionary of gradient statistics.
        
        Returns:
            Tuple[torch.Tensor, List[str]]: Feature matrix and parameter names.
        """
        features_list = []
        param_names = []
        
        for name, stats in gradient_stats.items():
            mean_feat = stats.mean
            std_feat = stats.std
            norm_feat = stats.norm
            dir_feat = stats.direction
            
            mean_norm = mean_feat.norm(p=2)
            std_norm = std_feat.norm(p=2) + 1e-8
            norm_norm = norm_feat.norm(p=2) + 1e-8
            
            normalized_mean = mean_feat / (mean_norm + 1e-8)
            normalized_std = std_feat / std_norm
            normalized_norm = norm_feat / norm_norm
            
            feature = torch.cat([
                normalized_mean,
                normalized_std,
                normalized_norm,
                dir_feat
            ])
            
            features_list.append(feature)
            param_names.append(name)
        
        features = torch.stack(features_list, dim=0)
        
        return features, param_names
    
    def _kmeans_clustering(
        self,
        features: torch.Tensor,
        num_clusters: int,
        max_iterations: int,
        tolerance: float,
        seed: Optional[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform K-means clustering with K-means++ initialization.
        
        Implements the K-means algorithm with K-means++ initialization
        for better convergence and cluster quality.
        
        Args:
            features: Feature matrix [num_samples, feature_dim].
            num_clusters: Number of clusters.
            max_iterations: Maximum iterations.
            tolerance: Convergence tolerance.
            seed: Random seed.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Centroids and cluster labels.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        num_samples, feature_dim = features.shape
        
        centroids = self._kmeans_plusplus_init(features, num_clusters)
        
        labels = torch.zeros(num_samples, dtype=torch.long, device=self.device)
        
        for iteration in range(max_iterations):
            distances = torch.cdist(features, centroids, p=2)
            new_labels = distances.argmin(dim=1)
            
            new_centroids = torch.zeros_like(centroids)
            for k in range(num_clusters):
                mask = new_labels == k
                if mask.sum() > 0:
                    new_centroids[k] = features[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]
            
            centroid_shift = torch.norm(new_centroids - centroids, p=2).max()
            centroids = new_centroids
            labels = new_labels
            
            if centroid_shift < tolerance:
                _LOG.debug(f"K-means converged at iteration {iteration}")
                break
        
        return centroids, labels
    
    def _kmeans_plusplus_init(
        self,
        features: torch.Tensor,
        num_clusters: int
    ) -> torch.Tensor:
        """Initialize centroids using K-means++ algorithm.
        
        K-means++ provides better initial centroid selection by choosing
        centroids that are far apart, leading to better convergence.
        
        Args:
            features: Feature matrix [num_samples, feature_dim].
            num_clusters: Number of clusters.
        
        Returns:
            torch.Tensor: Initial centroids [num_clusters, feature_dim].
        """
        num_samples, feature_dim = features.shape
        centroids = torch.zeros(num_clusters, feature_dim, device=self.device)
        
        first_idx = torch.randint(0, num_samples, (1,), device=self.device)
        centroids[0] = features[first_idx]
        
        for k in range(1, num_clusters):
            distances = torch.cdist(features, centroids[:k], p=2)
            min_distances = distances.min(dim=1)[0]
            
            probabilities = min_distances ** 2
            probabilities = probabilities / probabilities.sum()
            
            cumulative_probs = torch.cumsum(probabilities, dim=0)
            r = torch.rand(1, device=self.device)
            next_idx = (cumulative_probs >= r).nonzero()[0]
            
            if next_idx.numel() > 0:
                centroids[k] = features[next_idx[0]]
            else:
                centroids[k] = features[torch.randint(0, num_samples, (1,), device=self.device)]
        
        return centroids
    
    def auto_detect_clusters(
        self,
        features: torch.Tensor,
        min_k: int = 2,
        max_k: int = 16
    ) -> int:
        """Automatically detect optimal number of clusters using elbow method.
        
        Computes within-cluster sum of squares (WCSS) for different k values
        and finds the elbow point where WCSS decrease slows significantly.
        
        Elbow Detection:
            1. Compute WCSS for k in [min_k, max_k]
            2. Calculate second derivative of WCSS curve
            3. Find point with maximum curvature (elbow)
            4. Return optimal k value
        
        Args:
            features: Feature matrix [num_samples, feature_dim].
            min_k (int): Minimum number of clusters to test. Default: 2.
            max_k (int): Maximum number of clusters to test. Default: 16.
        
        Returns:
            int: Optimal number of clusters.
        
        Example:
            >>> optimal_k = initializer.auto_detect_clusters(features, min_k=4, max_k=16)
            >>> print(f"Optimal cluster count: {optimal_k}")
        
        Note:
            Uses second derivative method for elbow detection.
            Ensures returned k is within [min_k, max_k] range.
        """
        if features.shape[0] < min_k:
            return max(1, features.shape[0])
        
        max_k = min(max_k, features.shape[0])
        
        wcss_values = []
        k_values = list(range(min_k, max_k + 1))
        
        for k in k_values:
            centroids, labels = self._kmeans_clustering(
                features, k,
                max_iterations=50,
                tolerance=1e-3,
                seed=42
            )
            
            wcss = 0.0
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    cluster_points = features[mask]
                    wcss += torch.norm(cluster_points - centroids[i], p=2).item() ** 2
            
            wcss_values.append(wcss)
        
        if len(wcss_values) < 3:
            return min_k
        
        wcss_tensor = torch.tensor(wcss_values, device=self.device)
        
        diffs = wcss_tensor[1:] - wcss_tensor[:-1]
        second_diffs = diffs[1:] - diffs[:-1]
        
        elbow_idx = torch.argmax(torch.abs(second_diffs)).item()
        optimal_k = k_values[min(elbow_idx + 1, len(k_values) - 1)]
        
        _LOG.debug(
            f"Elbow detection: WCSS values = {wcss_values[:5]}..., "
            f"optimal k = {optimal_k}"
        )
        
        return optimal_k
    
    def _build_cluster_info(
        self,
        centroids: torch.Tensor,
        labels: torch.Tensor,
        param_names: List[str],
        features: torch.Tensor
    ) -> List[ClusterInfo]:
        """Build cluster information from clustering results.
        
        Creates ClusterInfo objects containing centroid, member indices,
        and intra-cluster variance for each cluster.
        
        Args:
            centroids: Cluster centroids [num_clusters, feature_dim].
            labels: Cluster labels for each sample [num_samples].
            param_names: List of parameter names.
            features: Feature matrix [num_samples, feature_dim].
        
        Returns:
            List[ClusterInfo]: List of cluster information objects.
        """
        clusters = []
        num_clusters = centroids.shape[0]
        
        for k in range(num_clusters):
            mask = labels == k
            member_indices = mask.nonzero(as_tuple=True)[0].tolist()
            member_names = [param_names[i] for i in member_indices]
            
            if len(member_indices) > 0:
                cluster_features = features[mask]
                centroid = centroids[k]
                variance = torch.mean(torch.norm(cluster_features - centroid, p=2, dim=1) ** 2).item()
            else:
                variance = 0.0
            
            clusters.append(ClusterInfo(
                cluster_id=k,
                centroid=centroids[k],
                member_indices=member_indices,
                member_names=member_names,
                intra_cluster_variance=variance,
                size=len(member_indices)
            ))
        
        return clusters
    
    def initialize_experts_from_clusters(
        self,
        experts: nn.ModuleList,
        clusters: Optional[List[ClusterInfo]] = None,
        gradient_stats: Optional[Dict[str, GradientStatistics]] = None,
        init_scale: float = 0.02,
        preserve_structure: bool = True
    ) -> None:
        """Initialize experts from discovered clusters.
        
        Uses cluster centroids and member statistics to initialize expert
        weights. Each expert receives initialization based on its assigned
        cluster, providing specialized starting points.
        
        Initialization Strategy:
            1. Assign clusters to experts (round-robin or by size)
            2. For each expert, use cluster centroid as initialization guide
            3. Add small random perturbation for diversity
            4. Preserve layer structure if preserve_structure is True
        
        Args:
            experts (nn.ModuleList): List of expert modules to initialize.
            clusters (List[ClusterInfo], optional): Pre-computed clusters.
                If None, uses self.clusters. Default: None.
            gradient_stats (Dict[str, GradientStatistics], optional): Gradient
                statistics. If None, uses self.gradient_features. Default: None.
            init_scale (float): Scale for random perturbation. Default: 0.02.
            preserve_structure (bool): Whether to preserve parameter structure.
                Default: True.
        
        Example:
            >>> initializer.initialize_experts_from_clusters(
            ...     model.moe_layer.experts,
            ...     init_scale=0.01
            ... )
        
        Note:
            Experts should be nn.Linear or nn.Sequential modules.
            Initialization preserves the original parameter shapes.
        """
        if clusters is None:
            clusters = self.clusters
        
        if gradient_stats is None:
            gradient_stats = self.gradient_features
        
        if len(clusters) == 0:
            _LOG.warning("No clusters available, using default initialization")
            self._default_expert_init(experts, init_scale)
            return
        
        num_experts = len(experts)
        num_clusters = len(clusters)
        
        cluster_assignments = self._assign_clusters_to_experts(
            num_experts,
            num_clusters,
            clusters
        )
        
        for expert_idx, cluster_idx in enumerate(cluster_assignments):
            cluster = clusters[cluster_idx]
            expert = experts[expert_idx]
            
            self._initialize_single_expert(
                expert,
                cluster,
                gradient_stats,
                init_scale,
                preserve_structure
            )
        
        _LOG.info(
            f"Initialized {num_experts} experts from {num_clusters} clusters"
        )
    
    def _assign_clusters_to_experts(
        self,
        num_experts: int,
        num_clusters: int,
        clusters: List[ClusterInfo]
    ) -> List[int]:
        """Assign clusters to experts based on cluster sizes.
        
        Distributes clusters to experts proportionally based on cluster size,
        ensuring larger clusters get more experts assigned.
        
        Args:
            num_experts: Total number of experts.
            num_clusters: Total number of clusters.
            clusters: List of cluster information.
        
        Returns:
            List[int]: Cluster index for each expert.
        """
        assignments = []
        
        total_size = sum(c.size for c in clusters)
        expert_per_cluster = []
        
        for cluster in clusters:
            proportion = cluster.size / total_size
            num_assigned = max(1, int(proportion * num_experts))
            expert_per_cluster.append(num_assigned)
        
        while sum(expert_per_cluster) < num_experts:
            max_idx = max(range(num_clusters), key=lambda i: clusters[i].size / (expert_per_cluster[i] + 1))
            expert_per_cluster[max_idx] += 1
        
        while sum(expert_per_cluster) > num_experts:
            min_idx = min(range(num_clusters), key=lambda i: expert_per_cluster[i])
            if expert_per_cluster[min_idx] > 1:
                expert_per_cluster[min_idx] -= 1
        
        for cluster_idx, count in enumerate(expert_per_cluster):
            assignments.extend([cluster_idx] * count)
        
        return assignments[:num_experts]
    
    def _initialize_single_expert(
        self,
        expert: nn.Module,
        cluster: ClusterInfo,
        gradient_stats: Dict[str, GradientStatistics],
        init_scale: float,
        preserve_structure: bool
    ) -> None:
        """Initialize a single expert from cluster information.
        
        Uses cluster centroid and member gradient statistics to initialize
        expert weights with specialized patterns.
        
        Args:
            expert: Expert module to initialize.
            cluster: Cluster information.
            gradient_stats: Gradient statistics dictionary.
            init_scale: Scale for random perturbation.
            preserve_structure: Whether to preserve parameter structure.
        """
        for name, param in expert.named_parameters():
            if not param.requires_grad:
                continue
            
            if len(cluster.member_names) > 0:
                ref_name = cluster.member_names[0]
                if ref_name in gradient_stats:
                    stats = gradient_stats[ref_name]
                    
                    if preserve_structure:
                        target_shape = param.shape
                        source_shape = stats.mean.shape
                        
                        if target_shape == source_shape:
                            init_mean = stats.mean.view(target_shape)
                            init_std = stats.std.view(target_shape)
                        else:
                            init_mean = self._reshape_stats(stats.mean, target_shape)
                            init_std = self._reshape_stats(stats.std, target_shape)
                        
                        noise = torch.randn_like(param) * init_scale
                        param.data = init_mean + init_std * noise
                    else:
                        param.data = stats.mean[:param.numel()].view(param.shape)
            else:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    
    def _reshape_stats(
        self,
        source: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Reshape statistics to match target parameter shape.
        
        Handles shape mismatches between source statistics and target
        parameter by using adaptive pooling or replication.
        
        Args:
            source: Source statistics tensor.
            target_shape: Target shape to match.
        
        Returns:
            torch.Tensor: Reshaped statistics.
        """
        target_numel = 1
        for dim in target_shape:
            target_numel *= dim
        
        if source.numel() >= target_numel:
            return source[:target_numel].view(target_shape)
        else:
            repeat_factor = (target_numel + source.numel() - 1) // source.numel()
            expanded = source.repeat(repeat_factor)[:target_numel]
            return expanded.view(target_shape)
    
    def _default_expert_init(
        self,
        experts: nn.ModuleList,
        init_scale: float
    ) -> None:
        """Apply default Kaiming initialization to experts.
        
        Used as fallback when no cluster information is available.
        
        Args:
            experts: List of expert modules.
            init_scale: Initialization scale.
        """
        for expert in experts:
            for param in expert.parameters():
                if param.requires_grad:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    
    def reset_statistics(self) -> None:
        """Reset collected gradient statistics.
        
        Clears all accumulated gradient buffers and statistics.
        Should be called before starting a new gradient collection phase.
        """
        self._gradient_buffer.clear()
        self.gradient_features.clear()
        self._collection_count = 0
        _LOG.info("Gradient statistics reset")
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of discovered clusters.
        
        Returns a dictionary containing cluster statistics and member
        information for analysis and debugging.
        
        Returns:
            Dict[str, Any]: Cluster summary including:
                - num_clusters: Number of clusters
                - cluster_sizes: List of cluster sizes
                - avg_variance: Average intra-cluster variance
                - total_parameters: Total parameters clustered
        
        Example:
            >>> summary = initializer.get_cluster_summary()
            >>> print(f"Found {summary['num_clusters']} clusters")
        """
        if len(self.clusters) == 0:
            return {
                "num_clusters": 0,
                "cluster_sizes": [],
                "avg_variance": 0.0,
                "total_parameters": 0
            }
        
        return {
            "num_clusters": len(self.clusters),
            "cluster_sizes": [c.size for c in self.clusters],
            "avg_variance": sum(c.intra_cluster_variance for c in self.clusters) / len(self.clusters),
            "total_parameters": sum(c.size for c in self.clusters)
        }
