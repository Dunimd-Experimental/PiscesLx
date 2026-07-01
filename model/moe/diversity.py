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

"""Information Bottleneck and Contrastive Diversification for MoE Expert Diversity.

This module implements state-of-the-art diversity regularization techniques
for Mixture-of-Experts models, ensuring experts learn distinct and complementary
representations while maintaining task performance.

Key Components:
    1. YvInformationBottleneckExpert:
       - MINE-based mutual information estimation
       - Information bottleneck regularization
       - Prevents expert collapse and redundancy
    
    2. YvContrastiveDiversification:
       - Contrastive learning for expert differentiation
       - Temperature-scaled similarity penalty
       - Adaptive threshold for diversity control

Mathematical Foundations:

    MINE (Mutual Information Neural Estimation):
        I(X;Y) ≈ E[T(x,y)] - log(E[exp(T(x,y'))])
        
        Where:
        - T: Statistics network (neural estimator)
        - (x,y): Joint distribution samples
        - (x,y'): Marginal distribution samples (y' shuffled from batch)
    
    Information Bottleneck:
        L_IB = I(X;Z) - β * I(Z;Y)
        
        Minimizes mutual information between input X and expert representation Z,
        while preserving information about target Y.
    
    Contrastive Diversification:
        L_div = Σ max(0, sim(z_i, z_j) - τ) for all expert pairs
        
        Penalizes expert pairs with similarity above threshold τ,
        encouraging diverse expert specializations.

Performance Characteristics:
    - MINE estimation: O(batch_size * hidden_size) per forward pass
    - Contrastive loss: O(num_experts^2) pairwise comparisons
    - Memory: Statistics network + expert embeddings

Usage Example:
    >>> from model.moe.diversity import (
    ...     YvInformationBottleneckExpert,
    ...     YvContrastiveDiversification
    ... )
    >>> 
    >>> # Information bottleneck for expert regularization
    >>> ib_expert = YvInformationBottleneckExpert(
    ...     hidden_size=4096,
    ...     bottleneck_dim=512,
    ...     num_experts=64
    ... )
    >>> mi_loss = ib_expert.compute_mi_loss(expert_outputs, inputs)
    >>> 
    >>> # Contrastive diversification
    >>> diversification = YvContrastiveDiversification(
    ...     num_experts=64,
    ...     temperature=0.1,
    ...     similarity_threshold=0.5
    ... )
    >>> div_loss = diversification.compute_diversity_loss(expert_embeddings)

Note:
    All classes follow the YvXxx naming convention.
    These components are designed to work with the MoE layer architecture.
    Diversity regularization should be balanced with task performance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.Moe.Diversity", file_path=get_log_file("Yv.Moe.Diversity"), enable_file=True)


@dataclass
class YvDiversityConfig:
    """Configuration for diversity regularization components.
    
    Encapsulates all parameters for information bottleneck and
    contrastive diversification settings.
    
    Attributes:
        hidden_size (int): Input hidden dimension.
        bottleneck_dim (int): Bottleneck dimension for information compression.
        num_experts (int): Number of experts in the MoE layer.
        mine_hidden_dim (int): Hidden dimension for MINE statistics network.
        temperature (float): Temperature for contrastive similarity scaling.
        similarity_threshold (float): Threshold for penalizing similar experts.
        ib_beta (float): Beta coefficient for information bottleneck trade-off.
        diversity_weight (float): Weight for diversity loss in total loss.
        use_moving_average (bool): Use moving average for MI estimation stability.
        momentum (float): Momentum for moving average updates.
    """
    hidden_size: int = 4096
    bottleneck_dim: int = 512
    num_experts: int = 64
    mine_hidden_dim: int = 256
    temperature: float = 0.1
    similarity_threshold: float = 0.5
    ib_beta: float = 1.0
    diversity_weight: float = 0.01
    use_moving_average: bool = True
    momentum: float = 0.9


class YvMineStatisticsNetwork(nn.Module):
    """Statistics network for MINE mutual information estimation.
    
    Implements the neural network T(x,y) used in MINE algorithm to
    estimate mutual information between input X and representation Y.
    
    Architecture:
        Input [x; y] -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Scalar
    
    The network learns to distinguish between joint distribution samples
    (x,y) and marginal distribution samples (x,y').
    
    Attributes:
        network (nn.Sequential): Multi-layer perceptron for statistics estimation.
    
    Note:
        Based on "Mutual Information Neural Estimation" (Belghazi et al., 2018)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        """Initialize the MINE statistics network.
        
        Args:
            input_dim: Dimension of concatenated input (x_dim + y_dim).
            hidden_dim: Hidden layer dimension. Default: 256.
            num_layers: Number of hidden layers. Default: 3.
        """
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute statistics T(x,y) for MINE estimation.
        
        Args:
            x: Input tensor [batch_size, x_dim].
            y: Representation tensor [batch_size, y_dim].
        
        Returns:
            torch.Tensor: Statistics values [batch_size, 1].
        """
        joint = torch.cat([x, y], dim=-1)
        return self.network(joint)


# Paper: Tishby et al., "The Information Bottleneck Method", arXiv:physics/0004057, 2000; Belghazi et al., "Mutual Information Neural Estimation", ICML 2018, arXiv:1801.04062
class YvInformationBottleneckExpert(nn.Module):
    """Information Bottleneck regularization for MoE experts.
    
    Implements information bottleneck principle for expert diversity,
    using MINE algorithm for mutual information estimation. This ensures
    experts learn compressed representations that preserve task-relevant
    information while discarding redundant information.
    
    Key Features:
        - MINE-based mutual information estimation
        - Information bottleneck regularization
        - Moving average for stable MI estimation
        - Expert-specific bottleneck dimensions
    
    Mathematical Formulation:
        L_IB = I(X;Z) - β * I(Z;Y)
        
        Where:
        - I(X;Z): Mutual information between input and expert representation
        - I(Z;Y): Mutual information between representation and output
        - β: Trade-off parameter
    
    The MINE estimator computes:
        I(X;Y) ≈ E[T(x,y)] - log(E[exp(T(x,y'))])
    
    Attributes:
        config (YvDiversityConfig): Configuration parameters.
        encoder (nn.Linear): Encoder for bottleneck representation.
        decoder (nn.Linear): Decoder from bottleneck to output.
        mine_net (YvMineStatisticsNetwork): MINE statistics network.
        mine_net_output (YvMineStatisticsNetwork): MINE for output MI.
        moving_average_mi (torch.Tensor): Buffer for stable MI estimation.
    
    Example:
        >>> ib_expert = YvInformationBottleneckExpert(
        ...     hidden_size=4096,
        ...     bottleneck_dim=512,
        ...     num_experts=64
        ... )
        >>> output, mi_loss = ib_expert(inputs, targets)
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 512,
        num_experts: int = 64,
        mine_hidden_dim: int = 256,
        ib_beta: float = 1.0,
        use_moving_average: bool = True,
        momentum: float = 0.9,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize the information bottleneck expert.
        
        Args:
            hidden_size: Input and output hidden dimension.
            bottleneck_dim: Dimension of bottleneck representation.
            num_experts: Number of experts (for scaling).
            mine_hidden_dim: Hidden dimension for MINE network.
            ib_beta: Beta coefficient for IB trade-off.
            use_moving_average: Use moving average for MI stability.
            momentum: Momentum for moving average updates.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.num_experts = num_experts
        self.ib_beta = ib_beta
        self.use_moving_average = use_moving_average
        self.momentum = momentum
        
        self.encoder = nn.Linear(
            hidden_size, bottleneck_dim, bias=False,
            device=device, dtype=dtype
        )
        self.decoder = nn.Linear(
            bottleneck_dim, hidden_size, bias=False,
            device=device, dtype=dtype
        )
        
        self.mine_net = YvMineStatisticsNetwork(
            input_dim=hidden_size + bottleneck_dim,
            hidden_dim=mine_hidden_dim
        )
        
        self.mine_net_output = YvMineStatisticsNetwork(
            input_dim=bottleneck_dim + hidden_size,
            hidden_dim=mine_hidden_dim
        )
        
        if use_moving_average:
            self.register_buffer('moving_mi_input', torch.tensor(0.0))
            self.register_buffer('moving_mi_output', torch.tensor(0.0))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to bottleneck representation.
        
        Args:
            x: Input tensor [batch_size, hidden_size].
        
        Returns:
            torch.Tensor: Bottleneck representation [batch_size, bottleneck_dim].
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck representation to output.
        
        Args:
            z: Bottleneck tensor [batch_size, bottleneck_dim].
        
        Returns:
            torch.Tensor: Output tensor [batch_size, hidden_size].
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through information bottleneck.
        
        Args:
            x: Input tensor [batch_size, hidden_size].
        
        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): Reconstructed output [batch, hidden].
                - z (torch.Tensor): Bottleneck representation [batch, bottleneck].
        """
        z = self.encode(x)
        output = self.decode(z)
        return output, z
    
    def estimate_mi(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mine_net: nn.Module
    ) -> torch.Tensor:
        """Estimate mutual information using MINE algorithm.
        
        Implements the MINE estimator:
            I(X;Y) ≈ E[T(x,y)] - log(E[exp(T(x,y'))])
        
        Where y' is obtained by shuffling y along the batch dimension,
        creating samples from the marginal distribution.
        
        Args:
            x: First variable tensor [batch_size, x_dim].
            y: Second variable tensor [batch_size, y_dim].
            mine_net: Statistics network T(x,y).
        
        Returns:
            torch.Tensor: Estimated mutual information scalar.
        """
        batch_size = x.size(0)
        
        joint_statistics = mine_net(x, y)
        expectation_joint = joint_statistics.mean()
        
        y_shuffle = y[torch.randperm(batch_size, device=y.device)]
        marginal_statistics = mine_net(x, y_shuffle)
        
        expectation_marginal = torch.logsumexp(marginal_statistics, dim=0) - math.log(batch_size)
        
        mi_estimate = expectation_joint - expectation_marginal
        
        return mi_estimate
    
    def compute_mi_loss(
        self,
        expert_outputs: torch.Tensor,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute information bottleneck loss for expert diversity.
        
        Computes the information bottleneck objective:
            L_IB = I(X;Z) - β * I(Z;Y)
        
        Where Z is the bottleneck representation. This encourages
        experts to learn compressed, task-relevant representations.
        
        Args:
            expert_outputs: Expert output tensor [batch, hidden] or
                [batch, num_experts, hidden].
            inputs: Input tensor [batch, hidden].
            targets: Optional target tensor [batch, hidden] for
                output mutual information. If None, uses expert_outputs.
        
        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): Information bottleneck loss scalar.
                - metrics (Dict[str, float]): MI estimates for logging.
        """
        if expert_outputs.dim() == 3:
            expert_outputs_flat = expert_outputs.view(-1, expert_outputs.size(-1))
        else:
            expert_outputs_flat = expert_outputs
        
        z = self.encode(expert_outputs_flat)
        
        mi_input = self.estimate_mi(expert_outputs_flat, z, self.mine_net)
        
        if targets is not None:
            if targets.dim() == 3:
                targets_flat = targets.view(-1, targets.size(-1))
            else:
                targets_flat = targets
            mi_output = self.estimate_mi(z, targets_flat, self.mine_net_output)
        else:
            reconstructed = self.decode(z)
            mi_output = self.estimate_mi(z, reconstructed.detach(), self.mine_net_output)
        
        if self.use_moving_average and self.training:
            with torch.no_grad():
                self.moving_mi_input.mul_(self.momentum).add_(
                    mi_input.item() * (1 - self.momentum)
                )
                self.moving_mi_output.mul_(self.momentum).add_(
                    mi_output.item() * (1 - self.momentum)
                )
        
        ib_loss = mi_input - self.ib_beta * mi_output
        
        metrics = {
            'mi_input': mi_input.item(),
            'mi_output': mi_output.item(),
            'ib_loss': ib_loss.item()
        }
        
        return ib_loss, metrics
    
    def get_bottleneck_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get bottleneck representation without computing loss.
        
        Args:
            x: Input tensor [batch_size, hidden_size].
        
        Returns:
            torch.Tensor: Bottleneck representation [batch_size, bottleneck_dim].
        """
        with torch.no_grad():
            return self.encode(x)


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvExpertEmbedding(nn.Module):
    """Learnable embedding for expert representation in diversity space.
    
    Creates a shared embedding space where expert specializations
    can be compared for diversity regularization.
    
    Attributes:
        embeddings (nn.Parameter): Learnable expert embeddings.
        projection (nn.Linear): Projection layer for input embeddings.
    
    Note:
        Embeddings are initialized with orthogonal vectors for
        maximum initial diversity.
    """
    
    def __init__(
        self,
        num_experts: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize expert embeddings.
        
        Args:
            num_experts: Number of experts.
            embedding_dim: Dimension of embedding space.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        
        self.embeddings = nn.Parameter(
            torch.empty(num_experts, embedding_dim, device=device, dtype=dtype)
        )
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with orthogonal vectors."""
        if self.num_experts <= self.embedding_dim:
            nn.init.orthogonal_(self.embeddings)
        else:
            nn.init.xavier_uniform_(self.embeddings)
    
    def forward(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings for specified experts.
        
        Args:
            expert_indices: Expert indices [batch_size] or [batch, top_k].
        
        Returns:
            torch.Tensor: Expert embeddings [batch, embedding_dim] or
                [batch, top_k, embedding_dim].
        """
        return F.embedding(expert_indices, self.embeddings)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all expert embeddings.
        
        Returns:
            torch.Tensor: All embeddings [num_experts, embedding_dim].
        """
        return self.embeddings


# Paper: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020, arXiv:2002.05709
class YvContrastiveDiversification(nn.Module):
    """Contrastive diversification for MoE expert specialization.
    
    Implements contrastive learning to encourage diverse expert
    specializations by penalizing expert pairs with high similarity.
    This prevents expert collapse where multiple experts learn
    identical functions.
    
    Key Features:
        - Temperature-scaled similarity computation
        - Adaptive similarity threshold
        - Hard negative mining for efficient training
        - Expert embedding regularization
    
    Mathematical Formulation:
        L_div = (1/K) * Σ max(0, sim(z_i, z_j) - τ)
        
        Where:
        - sim(z_i, z_j): Cosine similarity between expert embeddings
        - τ: Similarity threshold (experts above this are penalized)
        - K: Number of expert pairs
    
    Temperature Scaling:
        sim_scaled = cos(z_i, z_j) / temperature
        
        Lower temperature makes similarity more discriminative,
        encouraging stronger differentiation between experts.
    
    Attributes:
        num_experts (int): Number of experts.
        embedding_dim (int): Dimension of expert embedding space.
        temperature (float): Temperature for similarity scaling.
        similarity_threshold (float): Threshold for penalizing similarity.
        expert_embeddings (YvExpertEmbedding): Learnable expert embeddings.
        hard_negative_ratio (float): Ratio of hard negatives to use.
    
    Example:
        >>> diversification = YvContrastiveDiversification(
        ...     num_experts=64,
        ...     embedding_dim=128,
        ...     temperature=0.1,
        ...     similarity_threshold=0.5
        ... )
        >>> div_loss = diversification.compute_diversity_loss(expert_outputs)
    """
    
    def __init__(
        self,
        num_experts: int,
        embedding_dim: int = 128,
        temperature: float = 0.1,
        similarity_threshold: float = 0.5,
        hard_negative_ratio: float = 0.5,
        adaptive_threshold: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize contrastive diversification.
        
        Args:
            num_experts: Number of experts in MoE layer.
            embedding_dim: Dimension of expert embedding space.
            temperature: Temperature for similarity scaling. Lower values
                make similarity more discriminative. Default: 0.1.
            similarity_threshold: Threshold for penalizing similar experts.
                Experts with similarity above this are penalized. Default: 0.5.
            hard_negative_ratio: Ratio of hard negatives (most similar pairs)
                to focus on during training. Default: 0.5.
            adaptive_threshold: Use adaptive threshold based on training
                progress. Default: True.
            device: Device to place parameters on.
            dtype: Data type for parameters.
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        self.hard_negative_ratio = hard_negative_ratio
        self.adaptive_threshold = adaptive_threshold
        
        self.expert_embeddings = YvExpertEmbedding(
            num_experts, embedding_dim, device, dtype
        )
        
        self.input_projection = nn.Linear(
            embedding_dim, embedding_dim, bias=False,
            device=device, dtype=dtype
        )
        
        if adaptive_threshold:
            self.register_buffer('threshold_history', torch.zeros(100))
            self.register_buffer('history_ptr', torch.tensor(0))
            self.register_buffer('adaptive_threshold_value', torch.tensor(similarity_threshold))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.xavier_uniform_(self.input_projection.weight)
    
    def compute_similarity_matrix(
        self,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise similarity matrix for expert embeddings.
        
        Uses temperature-scaled cosine similarity:
            sim(z_i, z_j) = cos(z_i, z_j) / temperature
        
        Args:
            embeddings: Expert embeddings [num_experts, embedding_dim].
        
        Returns:
            torch.Tensor: Similarity matrix [num_experts, num_experts].
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        similarity_matrix = similarity_matrix / self.temperature
        
        return similarity_matrix
    
    def compute_diversity_loss(
        self,
        expert_outputs: Optional[torch.Tensor] = None,
        expert_indices: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute contrastive diversification loss.
        
        Penalizes expert pairs with similarity above threshold,
        encouraging diverse expert specializations.
        
        Loss formulation:
            L_div = (1/K) * Σ max(0, sim(z_i, z_j) - τ)
        
        Where K is the number of expert pairs with similarity above τ.
        
        Args:
            expert_outputs: Optional expert output tensor for computing
                dynamic embeddings. Shape: [batch, num_experts, hidden].
            expert_indices: Optional indices of active experts.
                Shape: [batch, top_k].
            routing_weights: Optional routing weights for weighted loss.
                Shape: [batch, top_k].
        
        Returns:
            tuple: A tuple containing:
                - loss (torch.Tensor): Diversity loss scalar.
                - metrics (Dict[str, float]): Diversity metrics for logging.
        """
        embeddings = self.expert_embeddings.get_all_embeddings()
        
        similarity_matrix = self.compute_similarity_matrix(embeddings)
        
        threshold = self._get_current_threshold()
        
        mask = ~torch.eye(self.num_experts, dtype=torch.bool, device=embeddings.device)
        
        upper_tri_mask = torch.triu(mask, diagonal=1)
        
        similarities = similarity_matrix[upper_tri_mask]
        
        violation_loss = F.relu(similarities - threshold)
        
        if self.training and self.hard_negative_ratio < 1.0:
            num_pairs = similarities.size(0)
            num_hard = max(1, int(num_pairs * self.hard_negative_ratio))
            
            _, hard_indices = torch.topk(similarities, num_hard)
            
            violation_loss = violation_loss[hard_indices]
        
        diversity_loss = violation_loss.mean()
        
        if expert_indices is not None and routing_weights is not None:
            active_loss = self._compute_active_expert_diversity(
                expert_indices, routing_weights, embeddings
            )
            diversity_loss = diversity_loss + 0.5 * active_loss
        
        if self.adaptive_threshold and self.training:
            self._update_adaptive_threshold(similarities.mean().item())
        
        metrics = {
            'diversity_loss': diversity_loss.item(),
            'mean_similarity': similarities.mean().item(),
            'max_similarity': similarities.max().item(),
            'threshold': threshold,
            'num_violations': (similarities > threshold).sum().item()
        }
        
        return diversity_loss, metrics
    
    def _compute_active_expert_diversity(
        self,
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute diversity loss for actively routed experts.
        
        Focuses diversity regularization on experts that are actually
        being used in the current batch, weighted by routing importance.
        
        Args:
            expert_indices: Active expert indices [batch, top_k].
            routing_weights: Routing weights [batch, top_k].
            embeddings: Expert embeddings [num_experts, embedding_dim].
        
        Returns:
            torch.Tensor: Weighted diversity loss for active experts.
        """
        batch_size, top_k = expert_indices.shape
        
        active_embeddings = embeddings[expert_indices]
        
        active_embeddings_norm = F.normalize(active_embeddings, p=2, dim=-1)
        
        similarity = torch.bmm(
            active_embeddings_norm,
            active_embeddings_norm.transpose(1, 2)
        ) / self.temperature
        
        mask = ~torch.eye(top_k, dtype=torch.bool, device=similarity.device)
        
        weight_matrix = routing_weights.unsqueeze(2) * routing_weights.unsqueeze(1)
        
        violation = F.relu(similarity - self.similarity_threshold) * mask
        
        weighted_loss = (violation * weight_matrix).sum() / (mask.sum() + 1e-8)
        
        return weighted_loss
    
    def _get_current_threshold(self) -> float:
        """Get current similarity threshold.
        
        Returns adaptive threshold if enabled, otherwise returns
        fixed threshold.
        
        Returns:
            float: Current similarity threshold.
        """
        if self.adaptive_threshold:
            return self.adaptive_threshold_value.item()
        return self.similarity_threshold
    
    def _update_adaptive_threshold(self, current_similarity: float):
        """Update adaptive threshold based on current similarity.
        
        Adjusts threshold to maintain a target level of diversity:
        - If similarity is too high, increase threshold to be more strict
        - If similarity is too low, decrease threshold to be more lenient
        
        Args:
            current_similarity: Current mean similarity between experts.
        """
        self.threshold_history[self.history_ptr] = current_similarity
        self.history_ptr = (self.history_ptr + 1) % 100
        
        if self.history_ptr > 10:
            recent_sim = self.threshold_history[:self.history_ptr].mean()
            
            target_sim = self.similarity_threshold * 0.8
            
            adjustment = (recent_sim - target_sim) * 0.1
            
            new_threshold = self.adaptive_threshold_value + adjustment
            new_threshold = torch.clamp(
                new_threshold,
                min=self.similarity_threshold * 0.5,
                max=self.similarity_threshold * 1.5
            )
            
            self.adaptive_threshold_value = new_threshold
    
    def get_expert_similarity_matrix(self) -> torch.Tensor:
        """Get the current expert similarity matrix.
        
        Useful for visualization and analysis of expert diversity.
        
        Returns:
            torch.Tensor: Similarity matrix [num_experts, num_experts].
        """
        with torch.no_grad():
            embeddings = self.expert_embeddings.get_all_embeddings()
            return self.compute_similarity_matrix(embeddings)
    
    def get_most_similar_pairs(self, k: int = 5) -> List[Tuple[int, int, float]]:
        """Get the k most similar expert pairs.
        
        Useful for identifying potential expert collapse.
        
        Args:
            k: Number of pairs to return. Default: 5.
        
        Returns:
            List of tuples (expert_i, expert_j, similarity).
        """
        with torch.no_grad():
            similarity_matrix = self.get_expert_similarity_matrix()
            
            mask = ~torch.eye(self.num_experts, dtype=torch.bool)
            
            similarities = similarity_matrix[mask]
            
            indices = torch.argsort(similarities, descending=True)[:k]
            
            pairs = []
            all_pairs = [(i, j) for i in range(self.num_experts) 
                         for j in range(i+1, self.num_experts)]
            
            for idx in indices:
                i, j = all_pairs[idx]
                sim = similarity_matrix[i, j].item()
                pairs.append((i, j, sim))
            
            return pairs


# Paper: Original contribution by Dunimd Team (Yv Architecture)
class YvDiversityRegularizer(nn.Module):
    """Combined diversity regularizer for MoE models.
    
    Combines information bottleneck and contrastive diversification
    into a unified regularization framework for comprehensive expert
    diversity management.
    
    Key Features:
        - Unified interface for all diversity losses
        - Configurable loss weights
        - Automatic loss balancing
        - Comprehensive metrics logging
    
    Attributes:
        info_bottleneck (YvInformationBottleneckExpert): IB regularizer.
        contrastive_div (YvContrastiveDiversification): Contrastive regularizer.
        ib_weight (float): Weight for information bottleneck loss.
        contrastive_weight (float): Weight for contrastive loss.
    
    Example:
        >>> regularizer = YvDiversityRegularizer(
        ...     hidden_size=4096,
        ...     num_experts=64,
        ...     bottleneck_dim=512
        ... )
        >>> total_loss, metrics = regularizer(expert_outputs, inputs)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        bottleneck_dim: int = 512,
        embedding_dim: int = 128,
        ib_beta: float = 1.0,
        temperature: float = 0.1,
        similarity_threshold: float = 0.5,
        ib_weight: float = 0.01,
        contrastive_weight: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize the combined diversity regularizer.
        
        Args:
            hidden_size: Input hidden dimension.
            num_experts: Number of experts.
            bottleneck_dim: Bottleneck dimension for IB.
            embedding_dim: Embedding dimension for contrastive.
            ib_beta: Beta for information bottleneck.
            temperature: Temperature for contrastive similarity.
            similarity_threshold: Threshold for contrastive penalty.
            ib_weight: Weight for IB loss in total loss.
            contrastive_weight: Weight for contrastive loss.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        
        self.ib_weight = ib_weight
        self.contrastive_weight = contrastive_weight
        
        self.info_bottleneck = YvInformationBottleneckExpert(
            hidden_size=hidden_size,
            bottleneck_dim=bottleneck_dim,
            num_experts=num_experts,
            ib_beta=ib_beta,
            device=device,
            dtype=dtype
        )
        
        self.contrastive_div = YvContrastiveDiversification(
            num_experts=num_experts,
            embedding_dim=embedding_dim,
            temperature=temperature,
            similarity_threshold=similarity_threshold,
            device=device,
            dtype=dtype
        )
    
    def forward(
        self,
        expert_outputs: torch.Tensor,
        inputs: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined diversity regularization loss.
        
        Args:
            expert_outputs: Expert output tensor.
            inputs: Input tensor.
            expert_indices: Optional active expert indices.
            routing_weights: Optional routing weights.
            targets: Optional target tensor for IB.
        
        Returns:
            tuple: A tuple containing:
                - total_loss (torch.Tensor): Combined diversity loss.
                - metrics (Dict[str, float]): All metrics for logging.
        """
        ib_loss, ib_metrics = self.info_bottleneck.compute_mi_loss(
            expert_outputs, inputs, targets
        )
        
        div_loss, div_metrics = self.contrastive_div.compute_diversity_loss(
            expert_outputs=expert_outputs,
            expert_indices=expert_indices,
            routing_weights=routing_weights
        )
        
        total_loss = self.ib_weight * ib_loss + self.contrastive_weight * div_loss
        
        metrics = {
            'total_diversity_loss': total_loss.item(),
            **{f'ib_{k}': v for k, v in ib_metrics.items()},
            **{f'div_{k}': v for k, v in div_metrics.items()}
        }
        
        return total_loss, metrics


__all__ = [
    "YvDiversityConfig",
    "YvMineStatisticsNetwork",
    "YvInformationBottleneckExpert",
    "YvExpertEmbedding",
    "YvContrastiveDiversification",
    "YvDiversityRegularizer",
]
