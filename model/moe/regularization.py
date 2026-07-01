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

"""Expert Diversity Regularization for Mixture-of-Experts Models.

This module implements comprehensive diversity regularization techniques
to ensure experts learn distinct and complementary representations,
preventing expert collapse and improving model generalization.

Key Components:
    1. YvExpertDiversityRegularizer:
       - Orthogonality loss for expert weight decorrelation
       - Routing entropy loss for balanced expert utilization
       - Activation variance loss for diverse expert outputs
       - Unified diversity loss combining all components

Mathematical Foundations:

    Orthogonality Loss:
        L_ortho = sum_{i<j} |W_i · W_j^T|^2 / (||W_i||^2 ||W_j||^2)
        
        Encourages expert weight matrices to be orthogonal, ensuring
        experts learn independent transformations.

    Routing Entropy Loss:
        L_entropy = -sum_e p_e log(p_e)
        
        Maximizes entropy of routing distribution to ensure uniform
        expert utilization across the input distribution.

    Activation Variance Loss:
        L_var = -var(expert_outputs)
        
        Maximizes variance of expert outputs to encourage diverse
        expert specializations.

Performance Characteristics:
    - Orthogonality: O(num_experts^2 * hidden^2) for pairwise computation
    - Entropy: O(num_experts) for distribution entropy
    - Variance: O(batch_size * num_experts * hidden) for output variance
    - Memory: Expert weight matrices + routing statistics

Usage Example:
    >>> from model.moe.regularization import YvExpertDiversityRegularizer
    >>> 
    >>> regularizer = YvExpertDiversityRegularizer(
    ...     num_experts=64,
    ...     hidden_size=4096,
    ...     ortho_weight=0.01,
    ...     entropy_weight=0.01,
    ...     variance_weight=0.01
    ... )
    >>> 
    >>> # Compute diversity loss
    >>> diversity_loss, metrics = regularizer.compute_diversity_loss(
    ...     expert_weights=expert.weight,
    ...     routing_probs=routing_weights,
    ...     expert_outputs=expert_activations
    ... )

Note:
    All classes follow the YvXxx naming convention.
    Diversity regularization should be balanced with task performance.
    Recommended to use with load balancing for optimal results.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.Moe.Regularization", file_path=get_log_file("Yv.Moe.Regularization"), enable_file=True)


@dataclass
class YvRegularizationConfig:
    """Configuration for expert diversity regularization.
    
    Encapsulates all parameters for orthogonality, entropy, and
    variance-based diversity regularization.
    
    Attributes:
        num_experts (int): Number of experts in the MoE layer.
        hidden_size (int): Hidden dimension of expert weights.
        ortho_weight (float): Weight for orthogonality loss.
        entropy_weight (float): Weight for routing entropy loss.
        variance_weight (float): Weight for activation variance loss.
        target_entropy (float): Target entropy for routing distribution.
        temperature (float): Temperature for entropy scaling.
        use_adaptive_weights (bool): Use adaptive loss weighting.
        moving_average_momentum (float): Momentum for moving averages.
    """
    num_experts: int = 64
    hidden_size: int = 4096
    ortho_weight: float = 0.01
    entropy_weight: float = 0.01
    variance_weight: float = 0.01
    target_entropy: float = None
    temperature: float = 1.0
    use_adaptive_weights: bool = True
    moving_average_momentum: float = 0.9
    
    def __post_init__(self):
        if self.target_entropy is None:
            self.target_entropy = math.log(self.num_experts)


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvExpertDiversityRegularizer(nn.Module):
    """Comprehensive diversity regularizer for MoE expert networks.
    
    Implements multiple diversity regularization techniques to ensure
    experts learn distinct and complementary representations. Combines
    orthogonality, entropy, and variance-based regularization into a
    unified framework.
    
    Key Features:
        - Orthogonality loss for expert weight decorrelation
        - Routing entropy loss for balanced expert utilization
        - Activation variance loss for diverse expert outputs
        - Adaptive loss weighting based on training dynamics
        - Moving average statistics for stable optimization
    
    Mathematical Formulation:
        L_div = w_ortho * L_ortho + w_entropy * L_entropy + w_var * L_var
        
        Where:
        - L_ortho: Orthogonality loss for weight decorrelation
        - L_entropy: Entropy loss for routing distribution
        - L_var: Variance loss for output diversity
    
    Attributes:
        config (YvRegularizationConfig): Configuration parameters.
        num_experts (int): Number of experts.
        hidden_size (int): Hidden dimension.
        ortho_weight (nn.Parameter): Learnable orthogonality weight.
        entropy_weight (nn.Parameter): Learnable entropy weight.
        variance_weight (nn.Parameter): Learnable variance weight.
    
    Example:
        >>> regularizer = YvExpertDiversityRegularizer(
        ...     num_experts=64,
        ...     hidden_size=4096
        ... )
        >>> loss, metrics = regularizer.compute_diversity_loss(
        ...     expert_weights, routing_probs, expert_outputs
        ... )
    """
    
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        ortho_weight: float = 0.01,
        entropy_weight: float = 0.01,
        variance_weight: float = 0.01,
        target_entropy: Optional[float] = None,
        temperature: float = 1.0,
        use_adaptive_weights: bool = True,
        moving_average_momentum: float = 0.9,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize the expert diversity regularizer.
        
        Args:
            num_experts: Number of experts in the MoE layer.
            hidden_size: Hidden dimension of expert weights.
            ortho_weight: Weight for orthogonality loss. Default: 0.01.
            entropy_weight: Weight for routing entropy loss. Default: 0.01.
            variance_weight: Weight for activation variance loss. Default: 0.01.
            target_entropy: Target entropy for routing distribution.
                Default: log(num_experts) for uniform distribution.
            temperature: Temperature for entropy scaling. Default: 1.0.
            use_adaptive_weights: Use adaptive loss weighting. Default: True.
            moving_average_momentum: Momentum for moving averages. Default: 0.9.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.use_adaptive_weights = use_adaptive_weights
        self.moving_average_momentum = moving_average_momentum
        
        if target_entropy is None:
            self.target_entropy = math.log(num_experts)
        else:
            self.target_entropy = target_entropy
        
        if use_adaptive_weights:
            self.ortho_weight = nn.Parameter(
                torch.tensor(ortho_weight, device=device, dtype=dtype)
            )
            self.entropy_weight = nn.Parameter(
                torch.tensor(entropy_weight, device=device, dtype=dtype)
            )
            self.variance_weight = nn.Parameter(
                torch.tensor(variance_weight, device=device, dtype=dtype)
            )
        else:
            self.register_buffer(
                'ortho_weight',
                torch.tensor(ortho_weight, device=device, dtype=dtype)
            )
            self.register_buffer(
                'entropy_weight',
                torch.tensor(entropy_weight, device=device, dtype=dtype)
            )
            self.register_buffer(
                'variance_weight',
                torch.tensor(variance_weight, device=device, dtype=dtype)
            )
        
        self.register_buffer(
            'moving_ortho_loss',
            torch.tensor(0.0, device=device, dtype=dtype)
        )
        self.register_buffer(
            'moving_entropy_loss',
            torch.tensor(0.0, device=device, dtype=dtype)
        )
        self.register_buffer(
            'moving_variance_loss',
            torch.tensor(0.0, device=device, dtype=dtype)
        )
        
        self.register_buffer(
            'expert_usage_stats',
            torch.zeros(num_experts, device=device, dtype=dtype)
        )
        self.register_buffer(
            'total_samples',
            torch.tensor(0.0, device=device, dtype=dtype)
        )
    
    def orthogonality_loss(
        self,
        expert_weights: torch.Tensor,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute orthogonality loss for expert weight matrices.
        
        Encourages expert weight matrices to be orthogonal to each other,
        ensuring experts learn independent transformations. Uses normalized
        Frobenius inner product for scale-invariant regularization.
        
        Mathematical Formulation:
            L_ortho = sum_{i<j} |W_i · W_j^T|^2 / (||W_i||^2 ||W_j||^2)
            
            For normalized weights:
            L_ortho = sum_{i<j} |W_i_norm · W_j_norm^T|^2
        
        Args:
            expert_weights: Expert weight tensor of shape
                [num_experts, hidden_size, intermediate_size] or
                [num_experts, hidden_size] for flattened weights.
            normalize: Whether to normalize weights before computing
                orthogonality. Default: True for scale invariance.
        
        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): Orthogonality loss scalar.
                - metrics (Dict[str, float]): Metrics for logging.
        
        Example:
            >>> weights = torch.randn(64, 4096, 11008)
            >>> loss, metrics = regularizer.orthogonality_loss(weights)
        
        Note:
            Normalization ensures the loss is scale-invariant.
            For large expert counts, consider chunked computation.
        """
        if expert_weights.dim() == 3:
            num_experts, hidden_size, intermediate_size = expert_weights.shape
            expert_weights_flat = expert_weights.view(num_experts, -1)
        elif expert_weights.dim() == 2:
            expert_weights_flat = expert_weights
            num_experts = expert_weights_flat.size(0)
        else:
            raise ValueError(
                f"Expected expert_weights to have 2 or 3 dimensions, "
                f"got {expert_weights.dim()}"
            )
        
        if normalize:
            weight_norms = torch.norm(expert_weights_flat, p=2, dim=1, keepdim=True)
            weight_norms = torch.clamp(weight_norms, min=1e-8)
            expert_weights_norm = expert_weights_flat / weight_norms
        else:
            expert_weights_norm = expert_weights_flat
        
        similarity_matrix = torch.mm(expert_weights_norm, expert_weights_norm.t())
        
        mask = ~torch.eye(num_experts, dtype=torch.bool, device=expert_weights.device)
        
        upper_tri_mask = torch.triu(mask, diagonal=1)
        
        off_diagonal_similarities = similarity_matrix[upper_tri_mask]
        
        ortho_loss = (off_diagonal_similarities ** 2).mean()
        
        max_similarity = off_diagonal_similarities.abs().max().item() if off_diagonal_similarities.numel() > 0 else 0.0
        mean_similarity = off_diagonal_similarities.abs().mean().item() if off_diagonal_similarities.numel() > 0 else 0.0
        
        metrics = {
            'orthogonality_loss': ortho_loss.item(),
            'max_weight_similarity': max_similarity,
            'mean_weight_similarity': mean_similarity,
            'num_expert_pairs': off_diagonal_similarities.numel()
        }
        
        return ortho_loss, metrics
    
    def routing_entropy_loss(
        self,
        routing_probs: torch.Tensor,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute entropy-based loss for routing distribution.
        
        Encourages uniform routing distribution by maximizing the entropy
        of expert selection probabilities. This ensures balanced expert
        utilization and prevents routing collapse.
        
        Mathematical Formulation:
            L_entropy = -sum_e p_e log(p_e)
            
            For batch routing:
            L_entropy = -mean_batch(sum_e p_e log(p_e))
            
            Scaled by temperature:
            L_entropy_scaled = L_entropy / temperature
        
        Args:
            routing_probs: Routing probability tensor of shape
                [batch_size, num_experts] or [batch_size, seq_len, num_experts].
                Should be normalized probabilities (sum to 1 along last dim).
            reduction: Reduction method for batch dimension.
                Options: 'mean', 'sum', 'none'. Default: 'mean'.
        
        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): Negative entropy loss scalar
                    (minimizing this maximizes entropy).
                - metrics (Dict[str, float]): Metrics for logging.
        
        Example:
            >>> routing_probs = F.softmax(logits, dim=-1)
            >>> loss, metrics = regularizer.routing_entropy_loss(routing_probs)
        
        Note:
            Returns negative entropy so minimizing the loss maximizes entropy.
            Target entropy is log(num_experts) for uniform distribution.
        """
        if routing_probs.dim() == 3:
            batch_size, seq_len, num_experts = routing_probs.shape
            routing_probs_flat = routing_probs.view(-1, num_experts)
        elif routing_probs.dim() == 2:
            routing_probs_flat = routing_probs
        else:
            raise ValueError(
                f"Expected routing_probs to have 2 or 3 dimensions, "
                f"got {routing_probs.dim()}"
            )
        
        routing_probs_clamped = torch.clamp(routing_probs_flat, min=1e-10, max=1.0)
        
        entropy = -torch.sum(
            routing_probs_clamped * torch.log(routing_probs_clamped),
            dim=-1
        )
        
        if reduction == 'mean':
            entropy_mean = entropy.mean()
        elif reduction == 'sum':
            entropy_mean = entropy.sum()
        else:
            entropy_mean = entropy
        
        max_entropy = math.log(self.num_experts)
        
        entropy_loss = max_entropy - entropy_mean / self.temperature
        
        entropy_ratio = entropy_mean.item() / max_entropy if isinstance(entropy_mean, torch.Tensor) else entropy_mean / max_entropy
        
        expert_usage = routing_probs_clamped.mean(dim=0)
        usage_variance = torch.var(expert_usage).item()
        
        metrics = {
            'routing_entropy': entropy_mean.item() if isinstance(entropy_mean, torch.Tensor) else entropy_mean,
            'entropy_ratio': entropy_ratio,
            'max_entropy': max_entropy,
            'usage_variance': usage_variance,
            'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss
        }
        
        return entropy_loss, metrics
    
    def activation_variance_loss(
        self,
        expert_outputs: torch.Tensor,
        normalize: bool = True,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute variance-based loss for expert output diversity.
        
        Encourages diverse expert outputs by maximizing the variance
        of expert activations. This ensures experts produce distinct
        representations for the same input.
        
        Mathematical Formulation:
            L_var = -var(expert_outputs)
            
            For batched outputs:
            L_var = -mean_batch(var_across_experts(expert_outputs))
            
            With normalization:
            L_var = -var(expert_outputs_normalized)
        
        Args:
            expert_outputs: Expert output tensor of shape
                [batch_size, num_experts, hidden_size] or
                [num_experts, hidden_size] for single batch.
            normalize: Whether to normalize outputs before computing
                variance. Default: True for scale invariance.
            reduction: Reduction method for batch dimension.
                Options: 'mean', 'sum', 'none'. Default: 'mean'.
        
        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): Negative variance loss scalar
                    (minimizing this maximizes variance).
                - metrics (Dict[str, float]): Metrics for logging.
        
        Example:
            >>> expert_outputs = model(inputs)  # [batch, num_experts, hidden]
            >>> loss, metrics = regularizer.activation_variance_loss(expert_outputs)
        
        Note:
            Returns negative variance so minimizing the loss maximizes variance.
            Normalization ensures the loss is scale-invariant.
        """
        if expert_outputs.dim() == 3:
            batch_size, num_experts, hidden_size = expert_outputs.shape
        elif expert_outputs.dim() == 2:
            expert_outputs = expert_outputs.unsqueeze(0)
            batch_size, num_experts, hidden_size = expert_outputs.shape
        else:
            raise ValueError(
                f"Expected expert_outputs to have 2 or 3 dimensions, "
                f"got {expert_outputs.dim()}"
            )
        
        if normalize:
            output_norms = torch.norm(expert_outputs, p=2, dim=-1, keepdim=True)
            output_norms = torch.clamp(output_norms, min=1e-8)
            expert_outputs_norm = expert_outputs / output_norms
        else:
            expert_outputs_norm = expert_outputs
        
        variance = torch.var(expert_outputs_norm, dim=1)
        
        variance_mean = variance.mean(dim=-1)
        
        if reduction == 'mean':
            variance_final = variance_mean.mean()
        elif reduction == 'sum':
            variance_final = variance_mean.sum()
        else:
            variance_final = variance_mean
        
        variance_loss = -variance_final
        
        pairwise_diff = expert_outputs_norm.unsqueeze(2) - expert_outputs_norm.unsqueeze(1)
        pairwise_distance = torch.norm(pairwise_diff, p=2, dim=-1)
        
        mask = ~torch.eye(num_experts, dtype=torch.bool, device=expert_outputs.device)
        mean_pairwise_distance = pairwise_distance[:, mask].mean().item()
        
        output_norms_mean = torch.norm(expert_outputs, p=2, dim=-1).mean().item()
        
        metrics = {
            'activation_variance': -variance_loss.item() if isinstance(variance_loss, torch.Tensor) else -variance_loss,
            'variance_loss': variance_loss.item() if isinstance(variance_loss, torch.Tensor) else variance_loss,
            'mean_pairwise_distance': mean_pairwise_distance,
            'mean_output_norm': output_norms_mean
        }
        
        return variance_loss, metrics
    
    def compute_diversity_loss(
        self,
        expert_weights: Optional[torch.Tensor] = None,
        routing_probs: Optional[torch.Tensor] = None,
        expert_outputs: Optional[torch.Tensor] = None,
        update_stats: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute comprehensive diversity loss combining all components.
        
        Combines orthogonality, entropy, and variance losses into a unified
        diversity regularization objective. Automatically handles missing
        components and applies adaptive weighting if enabled.
        
        Mathematical Formulation:
            L_div = w_ortho * L_ortho + w_entropy * L_entropy + w_var * L_var
            
            With adaptive weighting:
            w_i = w_i_base / (moving_avg_i + epsilon)
        
        Args:
            expert_weights: Expert weight tensor for orthogonality loss.
                Shape: [num_experts, hidden_size, intermediate_size] or
                [num_experts, hidden_size]. Optional.
            routing_probs: Routing probability tensor for entropy loss.
                Shape: [batch_size, num_experts] or
                [batch_size, seq_len, num_experts]. Optional.
            expert_outputs: Expert output tensor for variance loss.
                Shape: [batch_size, num_experts, hidden_size]. Optional.
            update_stats: Whether to update moving average statistics.
                Default: True.
        
        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): Combined diversity loss scalar.
                - metrics (Dict[str, float]): All metrics for logging.
        
        Example:
            >>> loss, metrics = regularizer.compute_diversity_loss(
            ...     expert_weights=expert.weight,
            ...     routing_probs=routing_weights,
            ...     expert_outputs=expert_activations
            ... )
        
        Note:
            At least one of expert_weights, routing_probs, or expert_outputs
            should be provided for meaningful regularization.
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device if hasattr(self, 'ortho_weight') else 'cpu')
        metrics = {}
        
        if expert_weights is not None:
            ortho_loss, ortho_metrics = self.orthogonality_loss(expert_weights)
            
            if self.use_adaptive_weights and self.training:
                weight_scale = 1.0 / (self.moving_ortho_loss.abs() + 1e-8)
                scaled_ortho_loss = self.ortho_weight * weight_scale * ortho_loss
            else:
                scaled_ortho_loss = self.ortho_weight * ortho_loss
            
            total_loss = total_loss + scaled_ortho_loss
            metrics.update({f'ortho_{k}': v for k, v in ortho_metrics.items()})
            
            if update_stats and self.training:
                with torch.no_grad():
                    self.moving_ortho_loss.mul_(self.moving_average_momentum).add_(
                        ortho_loss.item() * (1 - self.moving_average_momentum)
                    )
        
        if routing_probs is not None:
            entropy_loss, entropy_metrics = self.routing_entropy_loss(routing_probs)
            
            if self.use_adaptive_weights and self.training:
                weight_scale = 1.0 / (self.moving_entropy_loss.abs() + 1e-8)
                scaled_entropy_loss = self.entropy_weight * weight_scale * entropy_loss
            else:
                scaled_entropy_loss = self.entropy_weight * entropy_loss
            
            total_loss = total_loss + scaled_entropy_loss
            metrics.update({f'entropy_{k}': v for k, v in entropy_metrics.items()})
            
            if update_stats and self.training:
                with torch.no_grad():
                    self.moving_entropy_loss.mul_(self.moving_average_momentum).add_(
                        entropy_loss.item() * (1 - self.moving_average_momentum)
                    )
                
                if routing_probs.dim() == 3:
                    expert_usage = routing_probs.mean(dim=[0, 1])
                else:
                    expert_usage = routing_probs.mean(dim=0)
                
                self.expert_usage_stats.mul_(self.moving_average_momentum).add_(
                    expert_usage.detach() * (1 - self.moving_average_momentum)
                )
                self.total_samples.add_(routing_probs.size(0))
        
        if expert_outputs is not None:
            var_loss, var_metrics = self.activation_variance_loss(expert_outputs)
            
            if self.use_adaptive_weights and self.training:
                weight_scale = 1.0 / (self.moving_variance_loss.abs() + 1e-8)
                scaled_var_loss = self.variance_weight * weight_scale * var_loss
            else:
                scaled_var_loss = self.variance_weight * var_loss
            
            total_loss = total_loss + scaled_var_loss
            metrics.update({f'variance_{k}': v for k, v in var_metrics.items()})
            
            if update_stats and self.training:
                with torch.no_grad():
                    self.moving_variance_loss.mul_(self.moving_average_momentum).add_(
                        var_loss.item() * (1 - self.moving_average_momentum)
                    )
        
        metrics['total_diversity_loss'] = total_loss.item()
        
        if self.use_adaptive_weights:
            metrics['adaptive_ortho_weight'] = self.ortho_weight.item()
            metrics['adaptive_entropy_weight'] = self.entropy_weight.item()
            metrics['adaptive_variance_weight'] = self.variance_weight.item()
        
        return total_loss, metrics
    
    def get_expert_utilization(self) -> torch.Tensor:
        """Get the current expert utilization distribution.
        
        Returns the moving average of expert usage statistics,
        representing how frequently each expert is selected.
        
        Returns:
            torch.Tensor: Expert utilization distribution [num_experts].
        
        Example:
            >>> utilization = regularizer.get_expert_utilization()
            >>> print(f"Most used expert: {utilization.argmax()}")
        """
        if self.total_samples > 0:
            return self.expert_usage_stats / self.total_samples
        else:
            return torch.ones(self.num_experts, device=self.expert_usage_stats.device) / self.num_experts
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """Get current diversity regularization metrics.
        
        Returns the moving average statistics for all diversity
        loss components, useful for monitoring training dynamics.
        
        Returns:
            Dict[str, float]: Dictionary of diversity metrics.
        
        Example:
            >>> metrics = regularizer.get_diversity_metrics()
            >>> print(f"Moving ortho loss: {metrics['moving_ortho_loss']}")
        """
        return {
            'moving_ortho_loss': self.moving_ortho_loss.item(),
            'moving_entropy_loss': self.moving_entropy_loss.item(),
            'moving_variance_loss': self.moving_variance_loss.item(),
            'total_samples': self.total_samples.item()
        }
    
    def reset_statistics(self):
        """Reset all moving average statistics.
        
        Useful when starting a new training phase or after
        significant model changes.
        """
        self.moving_ortho_loss.zero_()
        self.moving_entropy_loss.zero_()
        self.moving_variance_loss.zero_()
        self.expert_usage_stats.zero_()
        self.total_samples.zero_()
    
    def forward(
        self,
        expert_weights: Optional[torch.Tensor] = None,
        routing_probs: Optional[torch.Tensor] = None,
        expert_outputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass computing diversity loss.
        
        Convenience method that calls compute_diversity_loss.
        
        Args:
            expert_weights: Expert weight tensor for orthogonality loss.
            routing_probs: Routing probability tensor for entropy loss.
            expert_outputs: Expert output tensor for variance loss.
        
        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): Combined diversity loss.
                - metrics (Dict[str, float]): All metrics for logging.
        """
        return self.compute_diversity_loss(
            expert_weights=expert_weights,
            routing_probs=routing_probs,
            expert_outputs=expert_outputs
        )


__all__ = [
    "YvRegularizationConfig",
    "YvExpertDiversityRegularizer",
]
