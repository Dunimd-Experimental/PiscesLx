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

"""Expert Evolution Module for Yv MoE.

Implements per-expert adaptive learning and domain specialization:
- Per-expert adaptive learning rates based on usage frequency
- Domain prototype vectors for expert specialization
- Hebbian local weight updates for online learning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class YvExpertEvolution(nn.Module):
    """Expert evolution with Hebbian learning and domain specialization.

    Each expert maintains its own learning dynamics based on:
    1. Usage frequency: frequently-used experts get higher learning rates
    2. Domain prototypes: experts specialize to different input domains
    3. Hebbian updates: local weight updates strengthen active connections

    Attributes:
        num_experts (int): Number of experts.
        hidden_size (int): Hidden dimension.
        base_lr (float): Base learning rate.
        decay (float): Weight decay for Hebbian updates.
        expert_usage (torch.Tensor): Usage frequency per expert.
        expert_lr (torch.Tensor): Adaptive learning rate per expert.
        domain_prototypes (torch.Tensor): Domain prototype per expert.

    Example:
        >>> evo = YvExpertEvolution(num_experts=8, hidden_size=4096)
        >>> evo.update_expert_usage(routing_weights, expert_indices)
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        base_lr: float = 1e-5,
        decay: float = 0.99,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.base_lr = base_lr
        self.decay = decay

        # Expert usage statistics
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("total_tokens", torch.tensor(0.0))

        # Adaptive learning rates per expert
        self.expert_lr = nn.Parameter(torch.ones(num_experts) * base_lr)

        # Domain prototypes for specialization
        self.domain_prototypes = nn.Parameter(
            torch.randn(num_experts, hidden_size, device=device, dtype=dtype) * 0.02
        )

        # Hebbian gate for selective updates
        self.hebbian_gate = nn.Parameter(torch.ones(hidden_size) * 0.1)

    def update_expert_usage(
        self,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> None:
        """Update expert usage frequency statistics.

        Args:
            routing_weights: Routing weights [batch, top_k].
            expert_indices: Selected expert indices [batch, top_k].
        """
        for idx in range(self.num_experts):
            usage = (expert_indices == idx).float().sum()
            self.expert_usage[idx] += usage.item()

        self.total_tokens += routing_weights.shape[0]

        # Update adaptive learning rates based on usage
        if self.total_tokens.item() > 0:
            usage_norm = self.expert_usage / self.expert_usage.sum().clamp(min=1.0)
            # Frequently used experts: higher LR (more specialization)
            # Rarely used experts: lower LR (preserve general knowledge)
            self.expert_lr.data = self.base_lr * (1.0 + 2.0 * usage_norm)

    def adapt_expert_weights(
        self,
        expert_idx: int,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        expert_weight: torch.Tensor
    ) -> torch.Tensor:
        """Adapt specific expert weights with local gradient update.

        Args:
            expert_idx: Expert index to update.
            input_batch: Input activations [batch, hidden_size].
            target_batch: Target outputs [batch, hidden_size].
            expert_weight: Current expert weight matrix [hidden_size, hidden_size].

        Returns:
            Updated expert weight matrix.
        """
        lr = self.expert_lr[expert_idx].item()

        # Simple gradient: minimize ||W @ x - target||^2
        pred = torch.matmul(input_batch, expert_weight.t())
        grad = torch.matmul((pred - target_batch).t(), input_batch) / input_batch.shape[0]

        # Hebbian update
        delta_W = lr * grad

        # Apply update with decay
        updated_weight = expert_weight - delta_W

        return updated_weight

    def hebbian_update(
        self,
        W: torch.Tensor,
        pre_activation: torch.Tensor,
        post_activation: torch.Tensor
    ) -> torch.Tensor:
        """Apply Hebbian learning rule: delta_W = lr * (pre * post - decay * W).

        Args:
            W: Weight matrix [hidden_size, hidden_size].
            pre_activation: Pre-synaptic activation [batch, hidden_size].
            post_activation: Post-synaptic activation [batch, hidden_size].

        Returns:
            Updated weight matrix.
        """
        # Outer product: pre and post co-activation strengthens connection
        hebbian_term = torch.matmul(pre_activation.t(), post_activation) / pre_activation.shape[0]

        # Gate the Hebbian update
        gated_hebbian = hebbian_term * self.hebbian_gate.unsqueeze(1)

        # Update rule with decay
        delta_W = self.base_lr * (gated_hebbian - (1.0 - self.decay) * W)

        return W + delta_W

    def domain_specialization(
        self,
        input_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Find best expert for input based on domain similarity.

        Args:
            input_embedding: Input feature [batch, hidden_size].

        Returns:
            Expert indices [batch] most similar to input.
        """
        similarities = F.cosine_similarity(
            input_embedding.unsqueeze(1),
            self.domain_prototypes.unsqueeze(0),
            dim=-1
        )
        best_expert = similarities.argmax(dim=-1)

        # Update domain prototypes toward assigned inputs
        with torch.no_grad():
            for i in range(self.num_experts):
                mask = (best_expert == i)
                if mask.any():
                    assigned = input_embedding[mask].mean(dim=0)
                    self.domain_prototypes[i] = 0.9 * self.domain_prototypes[i] + 0.1 * assigned

        return best_expert
