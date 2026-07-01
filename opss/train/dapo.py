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

"""DAPO: Decoupled Clipping and Dynamic Sampling Policy Optimization.

Based on 2025 DAPO paper. Upgrades GRPO with:
1. Decoupled clipping: separate epsilon_low and epsilon_high bounds
2. Dynamic sampling: adjust sample count based on group diversity
3. Asymmetric clipping for better stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# Paper: Yu et al., "DAPO: An Open-Source RL System from the Ground Up", arXiv:2503.14476
class YvDAPO:
    """DAPO policy optimization trainer.

    Extends GRPO with decoupled clipping and dynamic sampling.
    Uses asymmetric clipping bounds for more stable training.

    Attributes:
        epsilon_low (float): Lower clipping bound.
        epsilon_high (float): Upper clipping bound.
        beta (float): KL penalty coefficient.
        diversity_threshold (float): Threshold for dynamic sampling.

    Example:
        >>> dapo = YvDAPO(epsilon_low=0.2, epsilon_high=0.4)
        >>> loss = dapo.compute_policy_loss(log_probs, ref_log_probs, advantages)
    """

    def __init__(
        self,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.4,
        beta: float = 0.01,
        diversity_threshold: float = 0.3
    ):
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.beta = beta
        self.diversity_threshold = diversity_threshold

    def compute_decoupled_advantages(
        self,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """Compute group-relative advantages with decoupled normalization.

        Args:
            rewards: Reward tensor [group_size].

        Returns:
            Normalized advantages [group_size].
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        return advantages

    def dynamic_sampling(
        self,
        prompts: List,
        base_samples: int = 4
    ) -> int:
        """Determine number of samples based on group diversity.

        Higher diversity -> more samples needed.
        Lower diversity -> fewer samples sufficient.

        Args:
            prompts: List of prompts.
            base_samples: Base number of samples.

        Returns:
            Adjusted number of samples.
        """
        # Simple heuristic: longer/complex prompts need more samples
        avg_length = sum(len(p) for p in prompts) / max(len(prompts), 1)
        diversity_factor = min(avg_length / 100.0, 2.0)

        if diversity_factor > self.diversity_threshold:
            return int(base_samples * 1.5)
        return base_samples

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute DAPO policy loss with decoupled clipping.

        Args:
            log_probs: Current policy log probs [batch, seq_len].
            ref_log_probs: Reference policy log probs [batch, seq_len].
            advantages: Advantage values [batch].
            mask: Optional mask [batch, seq_len].

        Returns:
            Tuple of (loss, metrics_dict).
        """
        # Compute probability ratio
        ratio = torch.exp(log_probs - ref_log_probs)

        # Expand advantages to match sequence dimension
        advantages_expanded = advantages.unsqueeze(-1)

        # Decoupled clipping: asymmetric bounds
        clipped_ratio_low = torch.clamp(ratio, min=1.0 - self.epsilon_low)
        clipped_ratio_high = torch.clamp(ratio, max=1.0 + self.epsilon_high)
        clipped_ratio = torch.min(clipped_ratio_high, torch.max(clipped_ratio_low, ratio))

        # Policy gradient objective
        unclipped_objective = ratio * advantages_expanded
        clipped_objective = clipped_ratio * advantages_expanded

        policy_loss = -torch.min(unclipped_objective, clipped_objective)

        if mask is not None:
            policy_loss = policy_loss * mask
            policy_loss = policy_loss.sum() / mask.sum().clamp(min=1)
        else:
            policy_loss = policy_loss.mean()

        # KL penalty
        kl_div = (ref_log_probs - log_probs).mean()
        total_loss = policy_loss + self.beta * kl_div

        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
        }

        return total_loss, metrics
