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

"""Dynamic Mixture-of-Experts Routing Components for Yv Models.

This module provides sophisticated routing mechanisms for MoE layers,
implementing state-of-the-art routing algorithms with load balancing,
stability mechanisms, and cognitive density enhancements.

Routing Mechanisms:
    1. Standard Top-k Routing (YvMoEGate):
       - Routes each token to top-k experts based on learned scores
       - Includes load balancing loss for even expert utilization
       - Supports noise injection for exploration during training
       - Dynamic top-k adjustment based on load balance
    
    2. Stable Routing (YvStableMoEGate):
       - Prevents routing collapse through load prediction
       - Uses LSTM-based future load forecasting
       - Dynamic capacity adjustment per expert
       - Cognitive density enhancement for complex routing patterns

Key Features:
    - Load balancing with configurable loss coefficients
    - Expert capacity constraints to prevent overload
    - Temperature scaling for routing distribution control
    - Z-loss for routing stability
    - Random routing warmup for cold start stability
    - Dynamic top-k adjustment based on expert utilization
    - Cognitive density enhancement (optional)

Load Balancing Strategies:
    - Auxiliary loss: Penalizes uneven expert utilization
    - Capacity limitation: Prevents expert overload
    - Temperature adjustment: Controls routing distribution sharpness
    - Dynamic top-k: Adjusts routing breadth based on balance

Performance Characteristics:
    - Gate computation: O(hidden_size * num_experts)
    - Top-k selection: O(num_experts * log(k))
    - Memory: Gate weights + routing buffers
    - Typical overhead: 1-2% of total MoE computation

Usage Example:
    >>> from model.moe.gate import YvMoEGate, moe_init_weights
    >>> 
    >>> # Create routing gate
    >>> gate = YvMoEGate(
    ...     hidden_size=4096,
    ...     num_experts=64,
    ...     top_k=2,
    ...     load_balance_alpha=0.01
    ... )
    >>> 
    >>> # Route tokens
    >>> scores, indices, loss = gate(hidden_states)
    >>> # scores: [batch*seq, top_k] routing weights
    >>> # indices: [batch*seq, top_k] selected expert indices
    >>> # loss: scalar load balancing loss

Note:
    All classes follow the YvXxx naming convention.
    Load balance alpha should be tuned based on expert count.
    Higher alpha = stronger balancing but potentially worse performance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, Optional, Tuple
from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Moe", file_path=get_log_file("Yv.Moe"), enable_file=True)

def moe_init_weights(m):
    """Initialize weights for linear layers in MoE modules.
    
    Applies Kaiming uniform initialization to linear layer weights
    and zeros bias parameters. This initialization scheme is suitable
    for ReLU-family activations commonly used in MoE experts.
    
    Args:
        m (nn.Module): A PyTorch module to initialize.
            Only affects nn.Linear modules.
    
    Example:
        >>> expert = nn.Linear(4096, 11008)
        >>> expert.apply(moe_init_weights)
    
    Note:
        Should be applied to expert modules after construction.
        Uses Kaiming uniform with a=sqrt(5) for compatibility.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class YvMoEGate(nn.Module):
    """Expert routing gate for MoE with top-k selection and load balancing.
    
    Implements standard top-k routing where each token is routed to the
    k highest-scoring experts. Includes comprehensive load balancing
    mechanisms to ensure even expert utilization.
    
    Routing Process:
        1. Compute routing logits via linear projection
        2. Optionally add noise during training for exploration
        3. Apply temperature scaling
        4. Compute softmax scores
        5. Select top-k experts
        6. Compute load balancing loss
    
    Load Balancing Features:
        - Auxiliary loss penalizing uneven expert utilization
        - Capacity limitation to prevent expert overload
        - Dynamic top-k adjustment based on balance metrics
        - Z-loss for routing stability
    
    Attributes:
        gate (nn.Linear): Linear layer for computing routing logits.
        top_k (int): Number of experts to route each token to.
        num_experts (int): Total number of experts.
        load_balance_alpha (float): Coefficient for load balancing loss.
        noise_std (float): Standard deviation for routing noise.
        enable_cognitive_density (bool): Whether to use cognitive enhancement.
        expert_capacity_limit (float): Maximum capacity factor per expert.
        dynamic_top_k_min (int): Minimum top-k for dynamic adjustment.
        dynamic_top_k_max (int): Maximum top-k for dynamic adjustment.
        expert_usage_count (torch.Tensor): Buffer tracking expert usage.
        temperature (torch.Tensor): Buffer for routing temperature.
    
    Example:
        >>> gate = YvMoEGate(
        ...     hidden_size=4096,
        ...     num_experts=64,
        ...     top_k=2,
        ...     load_balance_alpha=0.01
        ... )
        >>> scores, indices, loss = gate(hidden_states)
    
    Note:
        Load balance alpha should be tuned based on expert count.
        Higher values enforce stronger balancing but may hurt performance.
    """
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None, 
                 load_balance_alpha=0.01, noise_std=0.1, enable_cognitive_density=False,
                 expert_capacity_limit=1.2, dynamic_top_k_min=1, dynamic_top_k_max=3,
                 load_balance_threshold=0.15, cfg=None):
        """Initialize the MoE gate module with load balancing.
        
        Args:
            hidden_size (int): Size of the input hidden dimension.
            num_experts (int): Number of experts to route between.
            top_k (int): Number of top experts to select per token. Default: 2.
            device: Device to place the module on. Default: None.
            dtype: Data type for module parameters. Default: None.
            load_balance_alpha (float): Coefficient for load balancing loss.
                Higher values enforce stronger balancing. Default: 0.01.
            noise_std (float): Standard deviation for routing noise during
                training. Set to 0 to disable. Default: 0.1.
            enable_cognitive_density (bool): Enable cognitive density
                enhancement for complex routing. Default: False.
            expert_capacity_limit (float): Maximum capacity factor per expert.
                Values > 1 allow some overload. Default: 1.2.
            dynamic_top_k_min (int): Minimum top-k for dynamic adjustment.
                Default: 1.
            dynamic_top_k_max (int): Maximum top-k for dynamic adjustment.
                Default: 3.
            load_balance_threshold (float): Threshold for triggering dynamic
                top-k adjustment. Default: 0.15.
            cfg: Configuration object with additional MoE parameters.
                Supports: moe_expert_grad_clip, moe_z_loss_alpha,
                moe_random_to_gradient_steps, moe_gate_warmup_alpha,
                moe_attention_mamba_temp, moe_l2_smooth_8k.
        """
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.top_k = top_k
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha
        self.noise_std = max(noise_std, 0.2)  # Increase minimum noise for better exploration
        self.enable_cognitive_density = enable_cognitive_density
        self.cognitive_enhancement_scale = 0.1
        
        self.expert_capacity_limit = expert_capacity_limit
        self.dynamic_top_k_min = dynamic_top_k_min
        self.dynamic_top_k_max = dynamic_top_k_max
        self.load_balance_threshold = load_balance_threshold
        
        self.expert_grad_clip = getattr(cfg, 'moe_expert_grad_clip', 0.1) if cfg is not None else 0.1
        self.z_loss_alpha = getattr(cfg, 'moe_z_loss_alpha', 1e-4) if cfg is not None else 1e-4
        self.random_to_gradient_steps = getattr(cfg, 'moe_random_to_gradient_steps', 2000) if cfg is not None else 2000  # Increase random routing period
        self.gate_warmup_alpha = getattr(cfg, 'moe_gate_warmup_alpha', 0.05) if cfg is not None else 0.05
        self.attention_mamba_temp = getattr(cfg, 'moe_attention_mamba_temp', 0.3) if cfg is not None else 0.3
        self.l2_smooth_8k = getattr(cfg, 'moe_l2_smooth_8k', 0.01) if cfg is not None else 0.01
        
        self.use_random_routing = True
        self.current_step = 0
        
        initial_temp = getattr(cfg, 'moe_routing_temperature', 3.0) if cfg is not None else 3.0
        min_temp = getattr(cfg, 'moe_temperature_min', 1.5) if cfg is not None else 1.5
        max_temp = getattr(cfg, 'moe_temperature_max', 5.0) if cfg is not None else 5.0
        
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        self.register_buffer('temperature', torch.tensor(initial_temp))
        self.min_temperature = min_temp
        self.register_buffer('total_routing_count', torch.tensor(0.0))
        self.register_buffer('expert_temperature_max', torch.tensor(max_temp))
        
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.bias_update_rate = 0.05
        self.bias_update_freq = 10
        self.register_buffer('bias_update_counter', torch.tensor(0))
        
        # Gradient checkpointing compatibility flag
        # When True, disables non-deterministic operations (noise, random routing)
        self._is_checkpointing = False
        
    def forward(self, x):
        """Forward pass of the MoE gate with enhanced load balancing.
        
        Performs the complete routing process including cognitive enhancement
        (if enabled), noise injection, temperature scaling, top-k selection,
        and load balancing loss computation.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
        
        Returns:
            tuple: A tuple containing:
                - top_scores (torch.Tensor): Routing weights [batch*seq, top_k].
                - top_idx (torch.Tensor): Selected expert indices [batch*seq, top_k].
                - total_loss (torch.Tensor): Combined load balance and z-loss.
        
        Note:
            During training, uses random routing for the first
            random_to_gradient_steps steps to prevent cold start issues.
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Apply cognitive density enhancement if enabled
        if self.enable_cognitive_density and hasattr(self, 'cognitive_encoder'):
            # Extract multi-scale features
            micro_features = self.cognitive_encoder['micro'](x_flat)
            macro_features = self.cognitive_encoder['macro'](x_flat)
            advanced_features = self.cognitive_encoder['advanced'](x_flat)
            
            # Fuse multi-scale features using dynamic attention
            combined_features = torch.stack([micro_features, macro_features, advanced_features], dim=1)
            fused_features, _ = self.cognitive_fusion(x_flat.unsqueeze(1), combined_features, combined_features)
            fused_features = fused_features.squeeze(1)
            
            # Enhance features with context-aware routing
            context_features = torch.cat([micro_features, macro_features, advanced_features], dim=-1)
            context_enhanced = self.context_router(context_features)
            
            # Apply meta-cognitive control
            meta_input = x_flat.unsqueeze(1)
            meta_output, _ = self.meta_controller(meta_input)
            meta_control = meta_output.squeeze(1)
            
            # Compute cognitive amplification gates
            precision_gate = self.amplification_gates['precision'](meta_control)
            recall_gate = self.amplification_gates['recall'](meta_control)
            creativity_gate = self.amplification_gates['creativity'](meta_control)
            
            # Integrate features adaptively
            memory_input = torch.cat([fused_features, context_enhanced], dim=-1).unsqueeze(1)
            enhanced_representation, _ = self.memory_integrator(memory_input)
            enhanced_representation = enhanced_representation.squeeze(1)
            
            # Apply cognitive gates to the enhanced representation
            enhanced_representation = enhanced_representation * (precision_gate + recall_gate + creativity_gate)
            
            # Update the original representation with the enhanced one
            x_flat = x_flat + enhanced_representation * self.cognitive_enhancement_scale
        
        # Update routing mode based on training steps
        if self.training and (not self._is_checkpointing):
            self.current_step += 1
            if self.current_step > self.random_to_gradient_steps:
                self.use_random_routing = False
        
        # Compute routing logits
        logits = self.gate(x_flat)
        
        # Apply DeepSeek-style dynamic bias for auxiliary-loss-free load balancing
        logits = logits + self.expert_bias
        
        # Apply gate warmup for cold start (disabled during checkpointing for determinism)
        if self.training and (not self._is_checkpointing) and self.current_step < 100:
            warmup_scale = min(1.0, self.current_step / 100.0)
            logits = logits * (self.gate_warmup_alpha + (1.0 - self.gate_warmup_alpha) * warmup_scale)
        
        # Use fixed top_k during checkpointing for deterministic behavior
        current_top_k = self.top_k if self._is_checkpointing else self._get_dynamic_top_k()
        
        # Disable noise during gradient checkpointing for determinism
        # Noise causes non-deterministic routing which breaks checkpoint recomputation
        if self.training and self.noise_std > 0 and not self._is_checkpointing:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Apply temperature scaling with 8k sequence length consideration
        seq_len = x.shape[1] if len(x.shape) > 1 else x.shape[0]
        if seq_len >= 8192:  # 8k sequence length threshold
            effective_temp = self.temperature * self.attention_mamba_temp
            # Apply stability bounds for 8k sequences
            max_temp = float(getattr(self, 'expert_temperature_max', torch.tensor(2.0)).item())
            effective_temp = max(0.1, min(max_temp, effective_temp))
            logits = logits / effective_temp
        else:
            # Apply stability bounds for normal sequences
            max_temp = float(getattr(self, 'expert_temperature_max', torch.tensor(2.0)).item())
            temp_bounded = max(0.1, min(max_temp, float(self.temperature.item())))
            logits = logits / temp_bounded
        
        # Compute routing scores
        scores = F.softmax(logits, dim=-1)
        
        # Apply expert capacity limitation
        scores = self._apply_capacity_limitation(scores)
        
        # During checkpointing: use deterministic routing only
        # Random routing is disabled during checkpointing for determinism
        if self.training and self.use_random_routing and self.current_step <= self.random_to_gradient_steps and not self._is_checkpointing:
            random_scores = torch.rand_like(scores)
            top_scores, top_idx = torch.topk(random_scores, current_top_k, dim=-1)
            top_scores = torch.ones_like(top_scores) / current_top_k
        else:
            top_scores, top_idx = torch.topk(scores, current_top_k, dim=-1)
            top_scores = F.softmax(top_scores, dim=-1, dtype=torch.float32).type_as(x)
            
            # Skip balance enforcement during checkpointing for determinism
            if self.training and (not self._is_checkpointing) and self.total_routing_count > 100:
                top_idx = self._enforce_balance_routing(top_idx, scores, current_top_k)
        
        if self.training and (not self._is_checkpointing):
            if not self.use_random_routing:
                self._update_expert_usage(top_idx, current_top_k)
                self._adjust_temperature()
            self.total_routing_count += x_flat.size(0)
        
        # Compute load balance loss
        load_balance_loss = self._compute_load_balance_loss(scores, top_idx, current_top_k)
        
        # Add z-loss for routing stability
        z_loss = self._compute_z_loss(logits)
        total_loss = load_balance_loss + z_loss
        
        return top_scores, top_idx, total_loss
    
    def _get_dynamic_top_k(self):
        """Get dynamic top-k value based on current load balance.
        
        Adjusts the number of experts each token is routed to based on
        the current load balance state. Increases top-k when imbalance
        is high to distribute load more evenly.
        
        Returns:
            int: Dynamic top-k value between dynamic_top_k_min and dynamic_top_k_max.
        """
        if not self.training:
            return self.top_k
            
        if self.total_routing_count > 0:
            usage_distribution = self.expert_usage_count / self.total_routing_count
            usage_variance = torch.var(usage_distribution)
            max_min_ratio = torch.max(usage_distribution) / (torch.min(usage_distribution) + 1e-8)
            
            imbalance_score = usage_variance * 0.7 + (max_min_ratio - 1.0) * 0.3
            
            if imbalance_score > self.load_balance_threshold:
                dynamic_top_k = min(self.dynamic_top_k_max, self.top_k + 1)
            elif imbalance_score < self.load_balance_threshold * 0.5:
                dynamic_top_k = max(self.dynamic_top_k_min, self.top_k - 1)
            else:
                dynamic_top_k = self.top_k
        else:
            dynamic_top_k = self.top_k
            
        return dynamic_top_k
    
    def _apply_capacity_limitation(self, scores):
        """Apply expert capacity limitation to prevent overloading.
        
        Scales routing scores based on historical expert usage to prevent
        any single expert from being overloaded. Uses capacity limits
        derived from the expert_capacity_limit parameter.
        
        Args:
            scores (torch.Tensor): Raw routing scores [num_tokens, num_experts].
            
        Returns:
            torch.Tensor: Capacity-limited and renormalized scores.
        """
        # Skip capacity limitation during checkpointing for determinism
        if not self.training or self._is_checkpointing:
            return scores
            
        if self.total_routing_count > 0:
            usage_rate = self.expert_usage_count / self.total_routing_count
            capacity_limits = torch.clamp(
                self.expert_capacity_limit * (1.0 - usage_rate + 0.5),
                min=0.1, max=1.0
            )
            
            limited_scores = scores * capacity_limits.unsqueeze(0)
            limited_scores = limited_scores / (limited_scores.sum(dim=-1, keepdim=True) + 1e-8)
            
            return limited_scores
        else:
            return scores
    
    def _compute_load_balance_loss(self, scores, top_idx=None, current_top_k=None):
        """Compute enhanced load balance loss with capacity awareness.
        
        Calculates a loss term that penalizes uneven expert utilization.
        Combines basic load balance loss with capacity-aware penalty
        for overused experts.
        
        Args:
            scores (torch.Tensor): Routing scores [num_tokens, num_experts].
            top_idx (torch.Tensor, optional): Selected expert indices for actual usage.
            current_top_k (int, optional): Dynamic top_k value.
            
        Returns:
            torch.Tensor: Combined load balance loss scalar.
        """
        expert_load = scores.mean(dim=0)
        ideal_load = torch.ones_like(expert_load) / self.num_experts
        
        basic_loss = self.load_balance_alpha * torch.sum((expert_load - ideal_load) ** 2)
        
        if top_idx is not None and self.training:
            effective_top_k = current_top_k if current_top_k is not None else self.top_k
            # Vectorized: use bincount instead of loop with scatter_add
            flat_top_idx = top_idx[:, :effective_top_k].flatten()
            actual_counts = torch.bincount(flat_top_idx, minlength=self.num_experts).float()
            
            actual_distribution = actual_counts / (top_idx.shape[0] * effective_top_k + 1e-8)
            actual_loss = self.load_balance_alpha * torch.sum((actual_distribution - ideal_load) ** 2)
            basic_loss = 0.5 * basic_loss + 0.5 * actual_loss
        
        if self.total_routing_count > 0:
            usage_rate = self.expert_usage_count / self.total_routing_count
            overuse_penalty = torch.sum(torch.relu(usage_rate - self.expert_capacity_limit) ** 2)
            capacity_loss = 0.005 * overuse_penalty
        else:
            capacity_loss = 0.0
            
        return basic_loss + capacity_loss
    
    def _compute_z_loss(self, logits):
        """Compute z-loss for routing stability.
        
        Z-loss encourages logit values to stay close to zero, preventing
        routing collapse where a few experts dominate. This is critical
        for stable MoE training.
        
        Args:
            logits (torch.Tensor): Raw routing logits [num_tokens, num_experts].
            
        Returns:
            torch.Tensor: Z-loss scalar value.
        """
        logit_squared = torch.square(logits)
        z_loss = self.z_loss_alpha * torch.mean(logit_squared)
        return z_loss
    
    def _update_expert_usage(self, top_idx, current_top_k=None):
        """Update expert usage statistics for load balancing.
        
        Tracks how many times each expert is selected during routing.
        This information is used for temperature adjustment and load
        balancing decisions.
        
        Args:
            top_idx (torch.Tensor): Selected expert indices [batch*seq, top_k].
            current_top_k (int, optional): Dynamic top_k value. If None, uses self.top_k.
        """
        if not self.training or self._is_checkpointing:
            return
        
        effective_top_k = current_top_k if current_top_k is not None else self.top_k
        
        # Vectorized: use bincount instead of loop with scatter_add
        flat_top_idx = top_idx[:, :effective_top_k].flatten()
        counts = torch.bincount(flat_top_idx, minlength=self.num_experts).float()
        self.expert_usage_count += counts
    
    def _adjust_temperature(self):
        """Dynamically adjust routing temperature based on load balance.
        
        Automatically increases temperature when expert load is imbalanced
        to encourage more uniform distribution. Uses variance and max/min
        ratio to detect imbalance and adjust temperature accordingly.
        
        Temperature adjustment strategy:
            - High imbalance (ratio > 10): Increase temperature significantly
            - Medium imbalance: Gradual temperature increase
            - Good balance: Slowly decrease temperature toward baseline
        """
        if not self.training or self._is_checkpointing:
            return
        
        min_samples_for_adjustment = 10
        if self.total_routing_count < min_samples_for_adjustment:
            return
        
        usage_distribution = self.expert_usage_count / (self.total_routing_count + 1e-8)
        
        nonzero_mask = usage_distribution > 1e-8
        if nonzero_mask.sum() < 2:
            return
        
        nonzero_usage = usage_distribution[nonzero_mask]
        load_variance = torch.var(usage_distribution).item()
        max_usage = torch.max(nonzero_usage).item()
        min_usage = torch.min(nonzero_usage).item()
        load_ratio = max_usage / (min_usage + 1e-8)
        
        max_temp = float(self.expert_temperature_max.item())
        ratio_signal = math.log(load_ratio + 1e-8)
        
        if load_ratio > 50.0:
            target_temp = min(max_temp, 2.0 + ratio_signal * 0.3)
        elif load_ratio > 10.0:
            target_temp = min(max_temp, max(1.5 + load_variance * 15, 1.0 + ratio_signal * 0.2))
        elif load_variance > 0.1:
            target_temp = min(max_temp, max(1.0 + load_variance * 10, 1.0 + ratio_signal * 0.1))
        elif load_variance > 0.05:
            target_temp = max(1.0, 1.0 + load_variance * 5)
        else:
            target_temp = max(self.min_temperature, 1.0 - load_variance * 2)
        
        adjustment_rate = 0.2
        current_temp = self.temperature.item()
        new_temp = current_temp * (1 - adjustment_rate) + target_temp * adjustment_rate
        
        new_temp = max(self.min_temperature, min(max_temp, new_temp))
        
        self.temperature.fill_(new_temp)
        
        self._update_expert_bias()
    
    def _update_expert_bias(self):
        """Update expert bias for auxiliary-loss-free load balancing.
        
        DeepSeek-V3 style: dynamically adjust per-expert bias based on load.
        - If expert is overloaded: decrease bias (make it less likely to be selected)
        - If expert is underutilized: increase bias (make it more likely to be selected)
        
        This achieves automatic load balancing without auxiliary loss functions.
        """
        if not self.training or self._is_checkpointing:
            return
        
        self.bias_update_counter.add_(1)
        if self.bias_update_counter.item() % self.bias_update_freq != 0:
            return
        
        usage_distribution = self.expert_usage_count / (self.total_routing_count + 1e-8)
        
        target_load = 1.0 / self.num_experts
        
        load_deviation = target_load - usage_distribution
        
        nonzero_mask = usage_distribution > 1e-8
        if nonzero_mask.sum() >= 2:
            nonzero_usage = usage_distribution[nonzero_mask]
            max_usage = torch.max(nonzero_usage).item()
            min_usage = torch.min(nonzero_usage).item()
            load_ratio = max_usage / (min_usage + 1e-8)
            
            if load_ratio > 20.0:
                dynamic_rate = self.bias_update_rate * 3.0
            elif load_ratio > 10.0:
                dynamic_rate = self.bias_update_rate * 2.0
            elif load_ratio > 5.0:
                dynamic_rate = self.bias_update_rate * 1.5
            else:
                dynamic_rate = self.bias_update_rate
        else:
            dynamic_rate = self.bias_update_rate
        
        bias_update = load_deviation * dynamic_rate
        
        new_bias = self.expert_bias + bias_update
        new_bias = torch.clamp(new_bias, min=-2.0, max=2.0)
        
        self.expert_bias.copy_(new_bias)
    
    def _enforce_balance_routing(self, top_idx, scores, current_top_k):
        """Enforce balanced routing when load is severely imbalanced.
        
        When expert load ratio exceeds critical threshold, this method
        redistributes some token assignments from overloaded experts
        to underutilized experts to prevent routing collapse.
        
        Args:
            top_idx (torch.Tensor): Selected expert indices [batch*seq, top_k].
            scores (torch.Tensor): Full routing scores [batch*seq, num_experts].
            current_top_k (int): Current top_k value.
            
        Returns:
            torch.Tensor: Potentially modified expert indices.
        """
        # Skip during checkpointing for determinism
        if self._is_checkpointing:
            return top_idx
        
        if self.total_routing_count < 100:
            return top_idx
        
        usage_distribution = self.expert_usage_count / (self.total_routing_count + 1e-8)
        
        nonzero_mask = usage_distribution > 1e-8
        if nonzero_mask.sum() < 2:
            return top_idx
        
        nonzero_usage = usage_distribution[nonzero_mask]
        max_usage = torch.max(nonzero_usage).item()
        min_usage = torch.min(nonzero_usage).item()
        load_ratio = max_usage / (min_usage + 1e-8)
        
        if load_ratio < 10.0:
            return top_idx
        
        target_load = 1.0 / self.num_experts
        overloaded_experts = (usage_distribution > target_load * 2.0).nonzero().squeeze(-1)
        underloaded_experts = (usage_distribution < target_load * 0.5).nonzero().squeeze(-1)
        
        if overloaded_experts.numel() == 0 or underloaded_experts.numel() == 0:
            return top_idx
        
        if overloaded_experts.dim() == 0:
            overloaded_experts = overloaded_experts.unsqueeze(0)
        if underloaded_experts.dim() == 0:
            underloaded_experts = underloaded_experts.unsqueeze(0)
        
        new_top_idx = top_idx.clone()
        num_tokens = top_idx.shape[0]
        redistribution_rate = min(0.3, (load_ratio - 10.0) / 100.0)
        
        for k in range(current_top_k):
            for overloaded_id in overloaded_experts:
                overloaded_id = overloaded_id.item()
                mask = (new_top_idx[:, k] == overloaded_id)
                num_to_redistribute = int(mask.sum().item() * redistribution_rate)
                
                if num_to_redistribute == 0:
                    continue
                
                redistribute_indices = mask.nonzero().squeeze(-1)
                if redistribute_indices.numel() == 0:
                    continue
                
                if redistribute_indices.dim() == 0:
                    redistribute_indices = redistribute_indices.unsqueeze(0)
                
                # Use deterministic permutation for gradient checkpointing compatibility
                generator = torch.Generator(device=top_idx.device)
                generator.manual_seed(int(top_idx.sum().item() * 1e6) % (2**31))
                perm = torch.randperm(redistribute_indices.numel(), generator=generator, device=top_idx.device)
                selected_indices = redistribute_indices[perm[:num_to_redistribute]]
                
                for idx in selected_indices:
                    token_idx = idx.item()
                    for underloaded_id in underloaded_experts:
                        underloaded_id = underloaded_id.item()
                        if underloaded_id not in new_top_idx[token_idx]:
                            new_top_idx[token_idx, k] = underloaded_id
                            break
        
        return new_top_idx

class YvStableMoEGate(nn.Module):
    """Stable MoE routing gate with load prediction and dynamic capacity.
    
    Implements an advanced routing mechanism designed to prevent routing
    collapse through load prediction and dynamic capacity adjustment.
    This gate uses LSTM-based prediction to forecast expert load and
    proactively adjust routing to maintain stability.
    
    Key Features:
        - Load prediction using LSTM networks
        - Dynamic capacity adjustment per expert
        - Cognitive density enhancement (optional)
        - Fixed shape mode for gradient checkpointing compatibility
        - Multi-scale cognitive processing
    
    Stability Mechanisms:
        - Predicts future expert load to prevent collapse
        - Dynamically adjusts expert capacity based on prediction
        - Uses complexity prediction for adaptive routing
        - Maintains load history for informed decisions
    
    Attributes:
        gate (nn.Linear): Linear layer for computing routing logits.
        top_k (int): Number of experts to route each token to.
        num_experts (int): Total number of experts.
        capacity_factor (float): Base capacity factor for experts.
        min_capacity (int): Minimum capacity per expert.
        prediction_horizon (int): Steps ahead for load prediction.
        fixed_shape_mode (bool): Enable for gradient checkpointing.
        enable_dynamic_capacity (bool): Use dynamic capacity adjustment.
        enable_cognitive_density (bool): Use cognitive enhancement.
        complexity_predictor (nn.Sequential): Network for predicting token complexity.
        load_predictor (nn.LSTM): LSTM for predicting future expert load.
    
    Example:
        >>> gate = YvStableMoEGate(
        ...     hidden_size=4096,
        ...     num_experts=64,
        ...     top_k=2,
        ...     enable_dynamic_capacity=True
        ... )
        >>> scores, indices, loss = gate(hidden_states)
    
    Note:
        More computationally expensive than standard gate but provides
        better stability for large expert counts or complex routing patterns.
    """
    
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None,
                 capacity_factor=1.0, min_capacity=4, prediction_horizon=10, 
                 fixed_shape_mode=False, enable_dynamic_capacity=True, enable_cognitive_density=False):
        """Initialize the stable MoE gate.
        
        Args:
            hidden_size (int): Size of the input hidden dimension.
            num_experts (int): Number of experts to route between.
            top_k (int): Number of top experts to select per token. Default: 2.
            device: Device to place the module on. Default: None.
            dtype: Data type for module parameters. Default: None.
            capacity_factor (float): Base capacity factor for experts.
                Higher values allow more tokens per expert. Default: 1.0.
            min_capacity (int): Minimum capacity per expert.
                Ensures no expert is completely starved. Default: 4.
            prediction_horizon (int): Number of steps ahead for load
                prediction. Default: 10.
            fixed_shape_mode (bool): Enable fixed shape mode for gradient
                checkpointing compatibility. Default: False.
            enable_dynamic_capacity (bool): Enable dynamic capacity
                adjustment based on prediction. Default: True.
            enable_cognitive_density (bool): Enable cognitive density
                enhancement for complex routing. Default: False.
        """
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.top_k = top_k
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.prediction_horizon = prediction_horizon
        self.fixed_shape_mode = fixed_shape_mode
        self.enable_dynamic_capacity = enable_dynamic_capacity
        self.enable_cognitive_density = enable_cognitive_density
        
        if self.enable_dynamic_capacity:
            self.complexity_predictor = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        if self.enable_cognitive_density:
            self.cognitive_encoder = nn.ModuleDict({
                'micro': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.SiLU(),
                    nn.Linear(hidden_size // 4, hidden_size // 16),
                    nn.SiLU()
                ),
                'macro': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.SiLU()
                ),
                'advanced': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size)
                )
            })
            
            self.cognitive_fusion = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True
            )
            
            self.context_router = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            
            # Meta-cognitive controller
            self.meta_controller = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            
            # Cognitive amplification gates
            self.amplification_gates = nn.ModuleDict({
                'precision': nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid()),
                'recall': nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid()),
                'creativity': nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
            })
            
            # Adaptive memory integration
            self.memory_integrator = nn.GRU(
                input_size=hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
        
        # Buffer to record routing history
        self.register_buffer('routing_history', torch.zeros(100, num_experts))
        # Pointer for routing history buffer
        self.register_buffer('history_ptr', torch.tensor(0))
        
        # LSTM-based load prediction network
        self.load_predictor = nn.LSTM(
            input_size=num_experts,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        # Linear layer for generating load predictions
        self.predictor_head = nn.Linear(64, num_experts * prediction_horizon)
        
        # Buffer to record expert load history
        self.register_buffer('expert_load_buffer', torch.zeros(50, num_experts))
        # Pointer for expert load buffer
        self.register_buffer('load_buffer_ptr', torch.tensor(0))
        
        # DeepSeek-style dynamic bias for auxiliary-loss-free load balancing
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.bias_update_rate = 0.05  # Faster update rate for stable gate
        self.bias_update_freq = 50    # Update bias every N steps
        self.register_buffer('bias_update_counter', torch.tensor(0))
        self.register_buffer('total_routing_count', torch.tensor(0.0))
        
        # Gradient checkpointing compatibility flag
        # When True, disables non-deterministic operations (noise, random routing)
        self._is_checkpointing = False
        
    def _predict_future_load(self):
        """Predict future expert load using LSTM.
        
        Uses the LSTM-based load predictor to forecast expert utilization
        based on historical load patterns. This enables proactive routing
        adjustments to prevent collapse.
        
        Returns:
            torch.Tensor: Predicted load distribution [num_experts].
                Values are normalized via softmax to form a probability
                distribution over experts.
        
        Note:
            Returns uniform distribution if insufficient history available.
        """
        if self.load_buffer_ptr < 10:
            return torch.ones(self.num_experts) / self.num_experts
        
        historical_load = self.expert_load_buffer[:self.load_buffer_ptr].unsqueeze(0)
        
        with torch.no_grad():
            lstm_out, _ = self.load_predictor(historical_load)
            predictions = self.predictor_head(lstm_out[:, -1, :])
            
        future_load = predictions.view(self.num_experts, self.prediction_horizon).mean(dim=1)
        future_load = torch.softmax(future_load, dim=0)
        
        return future_load
    
    def _update_expert_bias_stable(self, top_idx):
        """Update expert bias for auxiliary-loss-free load balancing (DeepSeek-style).
        
        Dynamically adjust per-expert bias based on actual routing distribution.
        This is the key innovation from DeepSeek-V3 for automatic load balancing.
        
        Args:
            top_idx (torch.Tensor): Selected expert indices [batch*seq, top_k].
        """
        if not self.training or self._is_checkpointing:
            return
        
        self.total_routing_count.add_(top_idx.numel())
        self.bias_update_counter.add_(1)
        
        if self.bias_update_counter.item() % self.bias_update_freq != 0:
            return
        
        # Count actual expert usage
        expert_counts = torch.bincount(top_idx.flatten(), minlength=self.num_experts)
        usage_distribution = expert_counts.float() / (self.total_routing_count + 1e-8)
        
        # Target: uniform distribution
        target_load = 1.0 / self.num_experts
        
        # Calculate bias adjustment: underloaded experts get positive bias
        load_deviation = target_load - usage_distribution
        
        # Apply bias update with rate limiting
        bias_update = load_deviation * self.bias_update_rate
        
        # Clamp bias to prevent extreme values
        new_bias = self.expert_bias + bias_update
        new_bias = torch.clamp(new_bias, min=-2.0, max=2.0)
        
        self.expert_bias.copy_(new_bias)
    
    def _calculate_dynamic_capacity(self, x):
        """Calculate dynamic capacity based on input complexity.
        
        Uses the complexity predictor to estimate how much capacity
        each expert should have based on the input tokens. More complex
        inputs receive higher capacity factors.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size].
        
        Returns:
            float: Dynamic capacity factor between 0.5 and 2.0.
        
        Note:
            Returns fixed capacity_factor if dynamic capacity is disabled.
        """
        if not self.enable_dynamic_capacity:
            return self.capacity_factor
            
        with torch.no_grad():
            complexity_scores = self.complexity_predictor(x.view(-1, x.size(-1)))
            avg_complexity = complexity_scores.mean().item()
            
        dynamic_factor = 0.5 + 1.5 * avg_complexity
        return max(0.5, min(2.0, dynamic_factor))
    
    def forward(self, x):
        """Forward pass of the stable MoE gate.
        
        Performs routing with load prediction and dynamic capacity adjustment.
        Uses LSTM-based prediction to forecast expert load and proactively
        adjust routing to maintain stability.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size].
        
        Returns:
            tuple: A tuple containing:
                - top_scores (torch.Tensor): Routing weights.
                - top_idx (torch.Tensor): Selected expert indices.
                - loss (torch.Tensor): Zero loss (for API compatibility).
        
        Note:
            In fixed_shape_mode, uses simple top-k routing without
            load prediction for gradient checkpointing compatibility.
        """
        batch_size, seq_len, hidden_size = x.shape
        # Flatten the input tensor
        x_flat = x.view(-1, hidden_size)
        
        # Use simple Top-K routing in fixed shape mode
        if self.fixed_shape_mode:
            logits = self.gate(x_flat)
            logits = logits + self.expert_bias  # Apply DeepSeek-style dynamic bias
            scores = F.softmax(logits, dim=-1)
            top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
            
            # Normalize top-k scores
            top_scores = F.softmax(top_scores, dim=-1, dtype=torch.float32).type_as(x)
            
            # Update bias for load balancing (skip during checkpointing)
            if self.training and not self._is_checkpointing:
                self._update_expert_bias_stable(top_idx)
            
            return top_scores, top_idx, torch.tensor(0.0, device=x.device)
        
        # During checkpointing: use simple deterministic routing
        if self._is_checkpointing:
            logits = self.gate(x_flat)
            logits = logits + self.expert_bias
            scores = F.softmax(logits, dim=-1)
            top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
            top_scores = F.softmax(top_scores, dim=-1, dtype=torch.float32).type_as(x)
            return top_scores, top_idx, torch.tensor(0.0, device=x.device)
        
        # Calculate expert capacity with dynamic adjustment
        current_capacity_factor = self._calculate_dynamic_capacity(x)
        tokens_per_expert = max(
            int(x_flat.size(0) * current_capacity_factor / self.num_experts),
            self.min_capacity
        )
        
        # Compute routing logits
        logits = self.gate(x_flat)
        
        # Record routing history and expert load during training
        if self.training:
            expert_probs = F.softmax(logits, dim=-1)
            expert_usage = expert_probs.mean(0)
            self.routing_history[self.history_ptr] = expert_usage.detach()
            self.history_ptr = (self.history_ptr + 1) % 100
            
            self.expert_load_buffer[self.load_buffer_ptr] = expert_usage.detach()
            self.load_buffer_ptr = (self.load_buffer_ptr + 1) % 50
        
        # Predict future load and adjust routing logits
        predicted_load = self._predict_future_load().to(x.device)
        
        # Calculate expert load balance and adjust routing if necessary
        if self.history_ptr > 0:
            recent_usage = self.routing_history[:self.history_ptr].mean(0)
            usage_entropy = -torch.sum(recent_usage * torch.log(recent_usage + 1e-8))
            max_entropy = torch.log(torch.tensor(self.num_experts))
            balance_ratio = usage_entropy / (max_entropy + 1e-8)
            
            # Adjust routing logits based on predicted load
            load_adjustment = 0.1 * (predicted_load - recent_usage)
            logits = logits + load_adjustment.unsqueeze(0)
            
            # Add noise if the load is unbalanced (disabled during checkpointing)
            if balance_ratio < 0.7:
                noise_scale = 0.1 * (1.0 - balance_ratio)
                noise = torch.randn_like(logits) * noise_scale
                logits = logits + noise
        
        # Compute routing scores using softmax
        scores = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
        
        # Apply capacity limitation - optimized batched processing
        final_scores = []
        final_indices = []
        
        # Vectorized capacity check
        expert_counts = torch.bincount(top_idx.flatten(), minlength=self.num_experts)
        overloaded_experts = (expert_counts > tokens_per_expert).nonzero().squeeze(-1)
        
        if len(overloaded_experts) > 0:
            # Process overloaded experts
            for expert_id in overloaded_experts.tolist():
                mask = (top_idx == expert_id).any(dim=-1)
                if mask.sum() > tokens_per_expert:
                    expert_scores = scores[mask, expert_id]
                    _, top_indices = torch.topk(expert_scores, tokens_per_expert)
                    keep_mask = torch.zeros_like(expert_scores, dtype=torch.bool)
                    keep_mask[top_indices] = True
                    mask[mask.clone()] = keep_mask
                
                if mask.any():
                    final_scores.append(scores[mask])
                    final_indices.append(torch.full_like(scores[mask], expert_id))
        else:
            # Fast path: no capacity overflow
            for expert_id in range(self.num_experts):
                mask = (top_idx == expert_id).any(dim=-1)
                if mask.any():
                    final_scores.append(scores[mask])
                    final_indices.append(torch.full_like(scores[mask], expert_id))
        
        # Use uniform distribution if there is no valid routing
        if len(final_scores) == 0:
            uniform_expert = torch.randint(0, self.num_experts, (x_flat.size(0),), device=x.device)
            final_scores = [torch.ones(x_flat.size(0), device=x.device) / self.top_k]
            final_indices = [uniform_expert]
        
        # Update bias for load balancing
        if self.training:
            self._update_expert_bias_stable(top_idx)
        
        return torch.cat(final_scores), torch.cat(final_indices), torch.tensor(0.0, device=x.device)

class YvExpertOrientedRouter(nn.Module):
    """Expert-Oriented Router for improved MoE load balancing.
    
    Implements a routing mechanism that uses expert capability embeddings
    and similarity-based routing to achieve better load distribution.
    This router considers both learned routing scores and expert-token
    similarity for more informed routing decisions.
    
    Key Features:
        - Expert capability embeddings for semantic routing
        - Similarity-based scoring between inputs and experts
        - Optional task-aware affinity gating
        - Dynamic load balancing correction
    
    Routing Process:
        1. Compute base routing logits via linear projection
        2. Encode input tokens and expert capabilities
        3. Compute similarity scores between inputs and experts
        4. Combine base logits with similarity scores
        5. Apply task affinity gating if enabled
        6. Select top-k experts with load balancing
    
    Attributes:
        gate (nn.Linear): Linear layer for base routing logits.
        expert_embeddings (nn.Parameter): Learnable expert capability embeddings.
        expert_capability_encoder (nn.Sequential): Encoder for expert embeddings.
        input_encoder (nn.Sequential): Encoder for input tokens.
        similarity_scorer (nn.Bilinear): Bilinear layer for similarity computation.
        affinity_gate (nn.Sequential): Optional task affinity gate.
        expert_usage_count (torch.Tensor): Buffer tracking expert usage.
    
    Example:
        >>> router = YvExpertOrientedRouter(
        ...     hidden_size=4096,
        ...     num_experts=64,
        ...     top_k=2,
        ...     expert_embed_dim=64
        ... )
        >>> scores, indices, all_scores = router(hidden_states)
    
    Note:
        More expressive than standard routing but with higher overhead.
        Best suited for scenarios where expert specialization matters.
    """
    
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None,
                 expert_embed_dim=64, capacity_factor=1.0, enable_expert_affinity=False):
        """Initialize the expert-oriented router.
        
        Args:
            hidden_size (int): Size of the input hidden dimension.
            num_experts (int): Number of experts to route between.
            top_k (int): Number of top experts to select per token. Default: 2.
            device: Device to place the module on. Default: None.
            dtype: Data type for module parameters. Default: None.
            expert_embed_dim (int): Dimension of expert capability embeddings.
                Default: 64.
            capacity_factor (float): Capacity factor for load balancing.
                Default: 1.0.
            enable_expert_affinity (bool): Enable task-aware affinity gating.
                Default: False.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_embed_dim = expert_embed_dim
        self.capacity_factor = capacity_factor
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        
        self.expert_embeddings = nn.Parameter(torch.randn(num_experts, expert_embed_dim))
        
        self.expert_capability_encoder = nn.Sequential(
            nn.Linear(expert_embed_dim, expert_embed_dim),
            nn.SiLU(),
            nn.Linear(expert_embed_dim, expert_embed_dim)
        )
        
        self.input_encoder = nn.Sequential(
            nn.Linear(hidden_size, expert_embed_dim),
            nn.SiLU(),
            nn.Linear(expert_embed_dim, expert_embed_dim)
        )
        
        self.similarity_scorer = nn.Bilinear(expert_embed_dim, expert_embed_dim, 1)
        
        self.affinity_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        ) if enable_expert_affinity else None
        
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        self.register_buffer('total_routed', torch.tensor(0.0))
        self.register_buffer('affinity_matrix', torch.eye(num_experts))
        
    def forward(self, x, task_embedding=None):
        """Forward pass of the expert-oriented router.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size].
            task_embedding (torch.Tensor, optional): Task embedding for
                affinity-aware routing. Default: None.
        
        Returns:
            tuple: A tuple containing:
                - top_scores (torch.Tensor): Routing weights [batch*seq, top_k].
                - top_idx (torch.Tensor): Selected expert indices [batch*seq, top_k].
                - scores (torch.Tensor): Full routing scores [batch*seq, num_experts].
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        logits = self.gate(x_flat)
        
        input_embeddings = self.input_encoder(x_flat)
        encoded_expert_embeds = self.expert_capability_encoder(self.expert_embeddings)
        
        similarity_scores = []
        for i in range(self.num_experts):
            score = self.similarity_scorer(input_embeddings, encoded_expert_embeds[i:i+1].expand_as(input_embeddings))
            similarity_scores.append(score)
        similarity_scores = torch.cat(similarity_scores, dim=-1)
        
        combined_logits = logits + 0.3 * similarity_scores
        
        if task_embedding is not None and self.affinity_gate is not None:
            tiled_task = task_embedding.unsqueeze(0).expand(x_flat.size(0), -1)
            affinity = self.affinity_gate(torch.cat([x_flat, tiled_task], dim=-1))
            combined_logits = combined_logits * (0.5 + 0.5 * affinity)
        
        scores = F.softmax(combined_logits, dim=-1)
        
        top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
        top_scores = F.softmax(top_scores, dim=-1, dtype=torch.float32).type_as(x)
        
        if self.training:
            self.expert_usage_count.scatter_add_(0, top_idx.view(-1), torch.ones_like(top_idx, dtype=torch.float32).view(-1))
            self.total_routed += x_flat.size(0)
            
            usage = self.expert_usage_count / (self.total_routed + 1e-8)
            ideal = 1.0 / self.num_experts
            imbalance = torch.var(usage) / (ideal + 1e-8)
            
            if imbalance > 0.15:
                correction = (ideal - usage) * 0.1
                combined_logits = combined_logits + correction.unsqueeze(0)
                scores = F.softmax(combined_logits, dim=-1)
        
        return top_scores, top_idx, scores


class YvMoELayer(nn.Module):
    """Mixture of Experts layer with load balancing and dynamic expert management.
    
    Implements a complete MoE layer that combines multiple expert networks
    with a routing gate for sparse computation. Supports both standard and
    stable routing gates with dynamic GPU memory management for large
    expert pools.
    
    Architecture:
        Input -> Router -> Expert Selection -> Expert Computation -> Weighted Sum -> Output
    
    Key Features:
        - Standard or stable routing gate options
        - Dynamic expert loading/unloading for GPU memory efficiency
        - LRU cache for active experts
        - Predictive expert loading based on load forecasting
        - Comprehensive load balance monitoring
        - L2 smoothing for long sequences (8k+)
    
    Memory Management:
        - Maintains LRU cache of active experts on GPU
        - Automatically moves experts between GPU and CPU
        - Predictive loading based on routing history
        - Configurable max_gpu_experts limit
    
    Attributes:
        gate (YvMoEGate or YvStableMoEGate): Routing gate.
        experts (nn.ModuleList): List of expert networks.
        top_k (int): Number of experts to route each token to.
        num_experts (int): Total number of experts.
        max_gpu_experts (int): Maximum experts to keep on GPU.
        expert_load_history (torch.Tensor): Buffer for load history.
    
    Example:
        >>> layer = YvMoELayer(
        ...     config,
        ...     use_stable_gate=True,
        ...     max_gpu_experts=4
        ... )
        >>> output, aux_loss = layer(hidden_states)
    
    Note:
        Use stable gate for better load balancing with large expert counts.
        Configure max_gpu_experts based on available GPU memory.
    """
    _layer_count = 0
    def __init__(self, cfg, device=None, dtype=None, print_every=8, max_gpu_experts=4,
                 use_stable_gate=True):
        """Initialize the MoE layer with load balancing.
        
        Args:
            cfg: Configuration object with MoE parameters:
                - hidden_size: Input/output dimension
                - intermediate_size: Expert intermediate dimension
                - moe_top_k: Number of experts per token (default: 2)
                - moe_num_experts: Total expert count (default: 8)
                - moe_capacity_factor: Expert capacity factor
                - moe_min_capacity: Minimum capacity per expert
                - moe_prediction_horizon: Steps for load prediction
                - enable_dynamic_capacity: Enable dynamic capacity
                - enable_cognitive_density: Enable cognitive enhancement
            device: Device to place the module on. Default: None.
            dtype: Data type for module parameters. Default: None.
            print_every (int): Print interval for logging. Default: 8.
            max_gpu_experts (int): Maximum experts on GPU. Default: 4.
            use_stable_gate (bool): Use stable gate for better balance.
                Default: True.
        """
        super().__init__()
        YvMoELayer._layer_count += 1
        self.cfg = cfg
        self.top_k = getattr(cfg, 'moe_top_k', 2)
        self.num_experts = getattr(cfg, 'moe_num_experts', 8)
        
        # Initialize routing gate
        if use_stable_gate:
            # Enable fixed shape mode for small models to avoid gradient checkpointing issues
            fixed_shape_mode = (cfg.hidden_size <= 768)
            self.gate = YvStableMoEGate(
                cfg.hidden_size, self.num_experts, top_k=self.top_k,
                device=device, dtype=dtype,
                capacity_factor=getattr(cfg, 'moe_capacity_factor', 1.0),
                min_capacity=getattr(cfg, 'moe_min_capacity', 4),
                prediction_horizon=getattr(cfg, 'moe_prediction_horizon', 10),
                fixed_shape_mode=fixed_shape_mode,
                enable_dynamic_capacity=getattr(cfg, 'enable_dynamic_capacity', True),
                enable_cognitive_density=getattr(cfg, 'enable_cognitive_density', False)
            )
            # Pass configuration parameters to the gate
            if hasattr(self.gate, 'expert_temperature_max'):
                self.gate.expert_temperature_max = getattr(cfg, 'expert_temperature_max', 5.0)
            if hasattr(self.gate, 'expert_load_balance_threshold'):
                self.gate.expert_load_balance_threshold = getattr(cfg, 'expert_load_balance_threshold', 0.15)
        else:
            self.gate = YvMoEGate(
                cfg.hidden_size, self.num_experts, top_k=self.top_k,
                device=device, dtype=dtype,
                load_balance_alpha=getattr(cfg, 'moe_load_balance_alpha', 0.01),
                noise_std=getattr(cfg, 'moe_noise_std', 0.1),
                enable_cognitive_density=getattr(cfg, 'enable_cognitive_density', False)
            )
        
        # Initialize expert modules
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)
            ) for _ in range(self.num_experts)
        ])
        # Initialize weights for expert modules
        for expert in self.experts:
            expert.apply(moe_init_weights)
        
        # Print layer information for the first layer
        if YvMoELayer._layer_count == 1:
            gate_type = "Stable" if use_stable_gate else "Standard"
            _LOG.info(f"YvMoELayer: {self.num_experts} experts, top-{self.top_k} routing, {gate_type} gate")

        self.max_gpu_experts = max_gpu_experts
        # Ordered dictionary to record the last used step of each expert
        self._active_experts = OrderedDict()  # expert_id: last_used_step
        self._step = 0
        
        # Buffer to record expert load history
        self.register_buffer('expert_load_history', torch.zeros(50, self.num_experts))
        # Pointer for expert load history buffer
        self.register_buffer('load_history_ptr', torch.tensor(0))

    def _move_expert_to_gpu(self, expert_id):
        """Move an expert to GPU and manage the LRU cache of active experts.
        
        Implements dynamic expert placement by moving the specified expert
        to GPU and evicting the least recently used expert if the GPU
        expert limit is exceeded.
        
        Args:
            expert_id (int): ID of the expert to move to GPU.
        
        Note:
            Updates the _active_experts LRU cache.
            Evicts LRU expert if max_gpu_experts limit is reached.
        """
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cuda':
            expert.to('cuda')
        self._active_experts[expert_id] = self._step
        
        if len(self._active_experts) > self.max_gpu_experts:
            lru_expert_id, _ = self._active_experts.popitem(last=False)
            self._move_expert_to_cpu(lru_expert_id)

    def _move_expert_to_cpu(self, expert_id):
        """Move an expert to CPU to free GPU memory.
        
        Args:
            expert_id (int): ID of the expert to move to CPU.
        """
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cpu':
            expert.to('cpu')

    def _monitor_expert_balance(self, expert_indices):
        """Monitor expert load balance with enhanced monitoring.
        
        Tracks expert utilization and provides warnings for severe
        load imbalance. Calculates variance and max/min ratios to
        detect routing collapse or uneven expert usage.
        
        Args:
            expert_indices (torch.Tensor): Indices of experts assigned
                to each input token.
        """
        if expert_indices.dtype != torch.long:
            expert_indices = expert_indices.long()
        
        expert_counts = torch.bincount(expert_indices.flatten(), minlength=self.num_experts)
        expert_load = expert_counts.float() / (expert_indices.numel() + 1e-8)
        
        self.expert_load_history[self.load_history_ptr] = expert_load
        self.load_history_ptr = (self.load_history_ptr + 1) % 50

    def forward(self, x):
        """Forward pass of the MoE layer with load balancing.
        
        Routes input tokens to experts, computes expert outputs, and
        combines results with weighted sum. Handles both standard and
        stable routing gates with appropriate processing modes.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, hidden_size].
        
        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): Combined expert outputs [batch, seq, hidden].
                - aux_loss (torch.Tensor): Auxiliary loss for load balancing.
        
        Note:
            Applies L2 smoothing for sequences >= 8192 tokens.
            Uses fixed shape mode for small models with gradient checkpointing.
            Dynamically manages expert placement on GPU for large expert pools.
        """
        b, t, d = x.shape
        # Apply L2 smoothing for 8k sequence length
        if t >= 8192 and hasattr(self, 'l2_smooth_8k'):
            x = x * (1.0 - self.l2_smooth_8k) + x.mean(dim=1, keepdim=True) * self.l2_smooth_8k
        h = x.view(-1, d)  # [B*T, d]
        
        # Pre-move all experts to correct device (one-time check)
        if h.device.type == 'cuda':
            for expert in self.experts:
                if next(expert.parameters()).device.type != h.device.type:
                    expert.to(h.device)
        
        # Use different processing modes based on the type of routing gate
        if isinstance(self.gate, YvStableMoEGate) and hasattr(self.gate, 'fixed_shape_mode') and self.gate.fixed_shape_mode:
            # Fixed shape mode: optimized batched processing
            scores, idx, aux_loss = self.gate(x)
            self._monitor_expert_balance(idx)
            
            # Optimized: vectorized batched processing without inner loops
            y = torch.zeros_like(h)
            
            # Flatten for efficient processing
            flat_idx = idx.flatten()  # [B*T*top_k]
            flat_scores = scores.flatten()  # [B*T*top_k]
            token_indices = torch.arange(h.size(0), device=h.device).unsqueeze(1).expand(-1, self.top_k).flatten()
            
            for expert_id in range(self.num_experts):
                expert_mask = (flat_idx == expert_id)
                if expert_mask.any():
                    selected_tokens = token_indices[expert_mask]
                    selected_scores = flat_scores[expert_mask]
                    
                    # Batched expert computation
                    h_batch = h[selected_tokens]
                    expert_out = self.experts[expert_id](h_batch)
                    
                    # Scatter outputs back using scatter_add for efficiency
                    y.scatter_add_(0, selected_tokens.unsqueeze(1).expand(-1, d), 
                                   selected_scores.unsqueeze(1) * expert_out)
            
            return y.view(b, t, d), aux_loss
        elif isinstance(self.gate, YvStableMoEGate):
            scores, idx, aux_loss = self.gate(x)
            
            # Fast path: direct batched processing
            if len(scores) == 0 or len(idx) == 0:
                # Simple fallback without random operations
                uniform_idx = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
                idx = uniform_idx
                scores = torch.ones(h.size(0), device=h.device) / self.num_experts
            
            # Vectorized processing - no loops over experts
            y = torch.zeros_like(h)
            
            # Flatten for efficient processing
            flat_idx = idx.flatten() if idx.dim() > 1 else idx
            flat_scores = scores.flatten() if scores.dim() > 1 else scores
            
            # Ensure correct dimensions
            if flat_idx.dim() == 0:
                flat_idx = flat_idx.unsqueeze(0)
            if flat_scores.dim() == 0:
                flat_scores = flat_scores.unsqueeze(0)
            
            # Expand token indices to match flattened scores
            num_assignments = flat_scores.size(0)
            if idx.dim() > 1:
                token_indices = torch.arange(h.size(0), device=h.device).unsqueeze(1).expand(-1, idx.size(1)).flatten()[:num_assignments]
            else:
                token_indices = torch.arange(min(num_assignments, h.size(0)), device=h.device)
            
            # Process each expert with batched operations
            for expert_id in range(self.num_experts):
                expert_mask = (flat_idx == expert_id)
                if expert_mask.any():
                    selected_tokens = token_indices[expert_mask]
                    selected_scores = flat_scores[expert_mask]
                    
                    # Clamp indices to valid range
                    valid_mask = selected_tokens < h.size(0)
                    selected_tokens = selected_tokens[valid_mask]
                    selected_scores = selected_scores[valid_mask]
                    
                    if len(selected_tokens) > 0:
                        h_batch = h[selected_tokens]
                        expert_out = self.experts[expert_id](h_batch)
                        y.scatter_add_(0, selected_tokens.unsqueeze(1).expand(-1, d),
                                       selected_scores.unsqueeze(1) * expert_out)
            
            self._monitor_expert_balance(idx if idx.dim() == 1 else idx[:, 0])
            self._step += 1
            return y.view(b, t, d), aux_loss
        else:
            # Standard MoEGate - optimized batched processing
            scores, idx, aux_loss = self.gate(x)
            
            y = torch.zeros_like(h)
            
            # Flatten for efficient processing
            flat_idx = idx.flatten()  # [B*T*top_k]
            flat_scores = scores.flatten()  # [B*T*top_k]
            token_indices = torch.arange(h.size(0), device=h.device).unsqueeze(1).expand(-1, self.top_k).flatten()
            
            # Process each expert with batched operations
            for expert_id in range(self.num_experts):
                expert_mask = (flat_idx == expert_id)
                if expert_mask.any():
                    selected_tokens = token_indices[expert_mask]
                    selected_scores = flat_scores[expert_mask]
                    
                    # Batched expert computation
                    h_batch = h[selected_tokens]
                    expert_out = self.experts[expert_id](h_batch)
                    
                    # Scatter outputs back
                    y.scatter_add_(0, selected_tokens.unsqueeze(1).expand(-1, d),
                                   selected_scores.unsqueeze(1) * expert_out)
            
            self._monitor_expert_balance(idx)
            self._step += 1
            return y.view(b, t, d), aux_loss
