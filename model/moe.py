#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

"""Dynamic Mixture-of-Experts routing components used across Ruchbah models."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, Optional, Tuple
# Use dms_core logging exclusively
import dms_core
PiscesLxCoreLog = dms_core.log.get_logger

logger = PiscesLxCoreLog("Ruchbah.Core.MoE", file_path="logs/RuchbahCore.log")

def moe_init_weights(m):
    """
    Initialize weights for linear layers in MoE.

    Args:
        m (torch.nn.Module): A PyTorch module.
    """
    if isinstance(m, nn.Linear):
        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            # Initialize bias to zero
            nn.init.zeros_(m.bias)

class RuchbahMoEGate(nn.Module):
    """
    Expert routing gate for MoE (top-k configurable) with load balancing.
    This gate is responsible for routing inputs to experts and maintaining load balance.
    """
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None, 
                 load_balance_alpha=0.01, noise_std=0.1, enable_cognitive_density=False,
                 expert_capacity_limit=1.2, dynamic_top_k_min=1, dynamic_top_k_max=3,
                 load_balance_threshold=0.15):
        """
        Initialize the MoE gate module with load balancing.

        Args:
            hidden_size (int): Size of the hidden layer.
            num_experts (int): Number of experts.
            top_k (int, optional): Number of top experts to select. Defaults to 2.
            device (torch.device, optional): Device to place the module on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the module. Defaults to None.
            load_balance_alpha (float, optional): Load balancing loss coefficient. Defaults to 0.01.
            noise_std (float, optional): Standard deviation for routing noise. Defaults to 0.1.
            enable_cognitive_density (bool, optional): Enable cognitive density enhancement. Defaults to False.
            expert_capacity_limit (float, optional): Expert capacity limit factor. Defaults to 1.2.
            dynamic_top_k_min (int, optional): Minimum top-k for dynamic adjustment. Defaults to 1.
            dynamic_top_k_max (int, optional): Maximum top-k for dynamic adjustment. Defaults to 3.
            load_balance_threshold (float, optional): Load balance threshold. Defaults to 0.15.
        """
        super().__init__()
        # Linear layer for computing routing logits
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.top_k = top_k
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha
        self.noise_std = noise_std
        self.enable_cognitive_density = enable_cognitive_density
        # Scale factor for cognitive enhancement
        self.cognitive_enhancement_scale = 0.1
        
        # Expert load balancing optimizations
        self.expert_capacity_limit = expert_capacity_limit
        self.dynamic_top_k_min = dynamic_top_k_min
        self.dynamic_top_k_max = dynamic_top_k_max
        self.load_balance_threshold = load_balance_threshold
        
        # Stability configurations for 3rd-order mixture
        self.expert_grad_clip = getattr(cfg, 'moe_expert_grad_clip', 0.1) if cfg else 0.1
        self.z_loss_alpha = getattr(cfg, 'moe_z_loss_alpha', 1e-4) if cfg else 1e-4
        self.random_to_gradient_steps = getattr(cfg, 'moe_random_to_gradient_steps', 500) if cfg else 500
        self.gate_warmup_alpha = getattr(cfg, 'moe_gate_warmup_alpha', 0.05) if cfg else 0.05
        self.attention_mamba_temp = getattr(cfg, 'moe_attention_mamba_temp', 0.3) if cfg else 0.3
        self.l2_smooth_8k = getattr(cfg, 'moe_l2_smooth_8k', 0.01) if cfg else 0.01
        
        # Routing mode and step counter
        self.use_random_routing = True  # Start with random routing
        self.current_step = 0
        
        # Buffer to record expert usage count
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        # Temperature parameter for routing
        self.register_buffer('temperature', torch.tensor(1.0))
        self.min_temperature = 0.1
        self.register_buffer('total_routing_count', torch.tensor(0.0))
        self.register_buffer('expert_temperature_max', torch.tensor(5.0))
        
    def forward(self, x):
        """
        Forward pass of the MoE gate with enhanced load balancing.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing routing scores, expert indices, and load balance loss.
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
        if self.training:
            self.current_step += 1
            if self.current_step > self.random_to_gradient_steps:
                self.use_random_routing = False
        
        # Compute routing logits
        logits = self.gate(x_flat)
        
        # Apply gate warmup for cold start
        if self.training and self.current_step < 100:  # First 100 steps warmup
            warmup_scale = min(1.0, self.current_step / 100.0)
            logits = logits * (self.gate_warmup_alpha + (1.0 - self.gate_warmup_alpha) * warmup_scale)
        
        # Dynamic top-k adjustment based on load balance
        current_top_k = self._get_dynamic_top_k()
        
        # Add noise during training to encourage exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Apply temperature scaling with 8k sequence length consideration
        seq_len = x.shape[1] if len(x.shape) > 1 else x.shape[0]
        if seq_len >= 8192:  # 8k sequence length threshold
            effective_temp = self.temperature * self.attention_mamba_temp
            # Apply stability bounds for 8k sequences
            effective_temp = max(0.1, min(2.0, effective_temp))
            logits = logits / effective_temp
        else:
            # Apply stability bounds for normal sequences
            temp_bounded = max(0.1, min(2.0, self.temperature))
            logits = logits / temp_bounded
        
        # Compute routing scores
        scores = F.softmax(logits, dim=-1)
        
        # Apply expert capacity limitation
        scores = self._apply_capacity_limitation(scores)
        
        # Select top-k experts with dynamic adjustment
        if self.training and self.use_random_routing and self.current_step <= self.random_to_gradient_steps:
            # Random routing for cold start
            random_scores = torch.rand_like(scores)
            top_scores, top_idx = torch.topk(random_scores, current_top_k, dim=-1)
            # Use uniform scores for random routing
            top_scores = torch.ones_like(top_scores) / current_top_k
        else:
            # Gradient-based routing after cold start
            top_scores, top_idx = torch.topk(scores, current_top_k, dim=-1)
            # Normalize top-k scores
            top_scores = F.softmax(top_scores, dim=-1, dtype=torch.float32).type_as(x)
        
        # Update expert usage count and temperature
        if self.training:
            self._update_expert_usage(top_idx)
            self._adjust_temperature()
            self.total_routing_count += x_flat.size(0)
        
        # Compute load balance loss
        load_balance_loss = self._compute_load_balance_loss(scores)
        
        # Add z-loss for routing stability
        z_loss = self._compute_z_loss(logits)
        total_loss = load_balance_loss + z_loss
        
        return top_scores, top_idx, total_loss
    
    def _get_dynamic_top_k(self):
        """
        Get dynamic top-k value based on current load balance.
        
        Returns:
            int: Dynamic top-k value.
        """
        if not self.training:
            return self.top_k
            
        # Calculate load balance score based on usage distribution
        if self.total_routing_count > 0:
            usage_distribution = self.expert_usage_count / self.total_routing_count
            usage_variance = torch.var(usage_distribution)
            max_min_ratio = torch.max(usage_distribution) / (torch.min(usage_distribution) + 1e-8)
            
            # Calculate imbalance score
            imbalance_score = usage_variance * 0.7 + (max_min_ratio - 1.0) * 0.3
            
            # Adjust top-k based on imbalance
            if imbalance_score > self.load_balance_threshold:
                # Increase top-k to distribute load more evenly
                dynamic_top_k = min(self.dynamic_top_k_max, self.top_k + 1)
            elif imbalance_score < self.load_balance_threshold * 0.5:
                # Decrease top-k when balance is good
                dynamic_top_k = max(self.dynamic_top_k_min, self.top_k - 1)
            else:
                dynamic_top_k = self.top_k
        else:
            dynamic_top_k = self.top_k
            
        return dynamic_top_k
    
    def _apply_capacity_limitation(self, scores):
        """
        Apply expert capacity limitation to prevent overloading.
        
        Args:
            scores (torch.Tensor): Raw routing scores.
            
        Returns:
            torch.Tensor: Capacity-limited scores.
        """
        if not self.training:
            return scores
            
        # Calculate capacity limit based on usage history
        if self.total_routing_count > 0:
            usage_rate = self.expert_usage_count / self.total_routing_count
            capacity_limits = torch.clamp(
                self.expert_capacity_limit * (1.0 - usage_rate + 0.5),
                min=0.1, max=1.0
            )
            
            # Apply capacity limitation by scaling scores
            limited_scores = scores * capacity_limits.unsqueeze(0)
            
            # Renormalize to maintain probability distribution
            limited_scores = limited_scores / (limited_scores.sum(dim=-1, keepdim=True) + 1e-8)
            
            return limited_scores
        else:
            return scores
    
    def _compute_load_balance_loss(self, scores):
        """
        Compute enhanced load balance loss with capacity awareness.
        
        Args:
            scores (torch.Tensor): Routing scores.
            
        Returns:
            torch.Tensor: Load balance loss.
        """
        # Calculate expert load from scores
        expert_load = scores.mean(dim=0)
        ideal_load = torch.ones_like(expert_load) / self.num_experts
        
        # Basic load balance loss
        basic_loss = self.load_balance_alpha * torch.sum((expert_load - ideal_load) ** 2)
        
        # Add capacity-aware penalty for overused experts
        if self.total_routing_count > 0:
            usage_rate = self.expert_usage_count / self.total_routing_count
            overuse_penalty = torch.sum(torch.relu(usage_rate - self.expert_capacity_limit) ** 2)
            capacity_loss = 0.005 * overuse_penalty  # Smaller weight for capacity loss
        else:
            capacity_loss = 0.0
            
        return basic_loss + capacity_loss
    
    def _compute_z_loss(self, logits):
        """
        Compute z-loss for routing stability.
        
        Args:
            logits (torch.Tensor): Routing logits.
            
        Returns:
            torch.Tensor: Z-loss value.
        """
        # Z-loss encourages logit values to stay close to zero
        logit_squared = torch.square(logits)
        z_loss = self.z_loss_alpha * torch.mean(logit_squared)
        return z_loss

class RuchbahStableMoEGate(nn.Module):
    """
    Stable MoE routing gate to prevent routing collapse with load prediction.
    This gate uses load prediction and dynamic capacity adjustment to improve stability.
    """
    
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None,
                 capacity_factor=1.0, min_capacity=4, prediction_horizon=10, 
                 fixed_shape_mode=False, enable_dynamic_capacity=True, enable_cognitive_density=False):
        super().__init__()
        # Linear layer for computing routing logits
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.top_k = top_k
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.prediction_horizon = prediction_horizon
        # Enable fixed shape mode for gradient checkpointing compatibility
        self.fixed_shape_mode = fixed_shape_mode
        self.enable_dynamic_capacity = enable_dynamic_capacity
        self.enable_cognitive_density = enable_cognitive_density
        
        # Initialize dynamic capacity prediction network if enabled
        if self.enable_dynamic_capacity:
            self.complexity_predictor = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        # Initialize cognitive density enhancement modules if enabled
        if self.enable_cognitive_density:
            # Multi-scale cognitive processing
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
            
            # Dynamic attention fusion
            self.cognitive_fusion = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True
            )
            
            # Context-aware routing
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
        
    def _predict_future_load(self):
        """
        Predict future expert load using LSTM.

        Returns:
            torch.Tensor: Predicted load distribution for each expert.
        """
        if self.load_buffer_ptr < 10:
            return torch.ones(self.num_experts) / self.num_experts
        
        # Get historical load data
        historical_load = self.expert_load_buffer[:self.load_buffer_ptr].unsqueeze(0)  # [1, seq_len, num_experts]
        
        # Predict future load without updating gradients
        with torch.no_grad():
            lstm_out, _ = self.load_predictor(historical_load)
            predictions = self.predictor_head(lstm_out[:, -1, :])  # [1, num_experts * horizon]
            
        # Reshape predictions to get the load for the next step
        future_load = predictions.view(self.num_experts, self.prediction_horizon).mean(dim=1)
        future_load = torch.softmax(future_load, dim=0)
        
        return future_load
    
    def _calculate_dynamic_capacity(self, x):
        """
        Calculate dynamic capacity based on input complexity.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            float: Dynamic capacity factor.
        """
        if not self.enable_dynamic_capacity:
            return self.capacity_factor
            
        # Calculate the average complexity score of input samples
        with torch.no_grad():
            complexity_scores = self.complexity_predictor(x.view(-1, x.size(-1)))
            avg_complexity = complexity_scores.mean().item()
            
        # Compute dynamic capacity factor based on complexity
        dynamic_factor = 0.5 + 1.5 * avg_complexity
        return max(0.5, min(2.0, dynamic_factor))
    
    def forward(self, x):
        """
        Forward pass of the stable MoE gate.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing routing scores, expert indices, and a zero loss tensor.
        """
        batch_size, seq_len, hidden_size = x.shape
        # Flatten the input tensor
        x_flat = x.view(-1, hidden_size)
        
        # Use simple Top-K routing in fixed shape mode
        if self.fixed_shape_mode:
            logits = self.gate(x_flat)
            scores = F.softmax(logits, dim=-1)
            top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
            
            # Normalize top-k scores
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
            
            # Add noise if the load is unbalanced
            if balance_ratio < 0.7:
                noise_scale = 0.1 * (1.0 - balance_ratio)
                logits = logits + torch.randn_like(logits) * noise_scale
        
        # Compute routing scores using softmax
        scores = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
        
        # Apply capacity limitation
        final_scores = []
        final_indices = []
        
        for expert_id in range(self.num_experts):
            mask = (top_idx == expert_id).any(dim=-1)
            if mask.sum() > tokens_per_expert:
                # Re-route if capacity is exceeded
                expert_scores = scores[mask, expert_id]
                _, top_indices = torch.topk(expert_scores, tokens_per_expert)
                keep_mask = torch.zeros_like(expert_scores, dtype=torch.bool)
                keep_mask[top_indices] = True
                mask[mask.clone()] = keep_mask
            
            if mask.any():
                final_scores.append(scores[mask])
                final_indices.append(torch.full_like(scores[mask], expert_id))
        
        # Use uniform distribution if there is no valid routing
        if len(final_scores) == 0:
            uniform_expert = torch.randint(0, self.num_experts, (x_flat.size(0),), device=x.device)
            final_scores = [torch.ones(x_flat.size(0), device=x.device) / self.top_k]
            final_indices = [uniform_expert]
        
        return torch.cat(final_scores), torch.cat(final_indices), torch.tensor(0.0, device=x.device)

class RuchbahExpertOrientedRouter(nn.Module):
    """
    Expert-Oriented Router for improved MoE load balancing.
    
    This router uses expert capability embeddings and similarity-based routing
    to achieve better load distribution compared to LSTM-based prediction.
    """
    
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None,
                 expert_embed_dim=64, capacity_factor=1.0, enable_expert_affinity=False):
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


class RuchbahMoELayer(nn.Module):
    """
    Mixture of Experts layer with improved load balancing and stability.
    This layer combines multiple experts and uses a routing gate to distribute inputs.
    """
    _layer_count = 0
    def __init__(self, cfg, device=None, dtype=None, print_every=8, max_gpu_experts=4,
                 use_stable_gate=True):
        """
        Initialize the MoE layer with load balancing.

        Args:
            cfg: Configuration object.
            device (torch.device, optional): Device to place the module on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the module. Defaults to None.
            print_every (int, optional): Print interval. Defaults to 8.
            max_gpu_experts (int, optional): Maximum number of experts on GPU. Defaults to 4.
            use_stable_gate (bool, optional): Whether to use stable gate. Defaults to True.
        """
        super().__init__()
        RuchbahMoELayer._layer_count += 1
        self.cfg = cfg
        self.top_k = getattr(cfg, 'moe_top_k', 2)
        self.num_experts = getattr(cfg, 'moe_num_experts', 8)
        
        # Initialize routing gate
        if use_stable_gate:
            # Enable fixed shape mode for small models to avoid gradient checkpointing issues
            fixed_shape_mode = (cfg.hidden_size <= 768)
            self.gate = RuchbahStableMoEGate(
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
            self.gate = RuchbahMoEGate(
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
        if RuchbahMoELayer._layer_count == 1:
            gate_type = "Stable" if use_stable_gate else "Standard"
            logger.info(f"RuchbahMoELayer: {self.num_experts} experts, top-{self.top_k} routing, {gate_type} gate")

        self.max_gpu_experts = max_gpu_experts
        # Ordered dictionary to record the last used step of each expert
        self._active_experts = OrderedDict()  # expert_id: last_used_step
        self._step = 0
        
        # Buffer to record expert load history
        self.register_buffer('expert_load_history', torch.zeros(50, self.num_experts))
        # Pointer for expert load history buffer
        self.register_buffer('load_history_ptr', torch.tensor(0))

    def _move_expert_to_gpu(self, expert_id):
        """
        Move an expert to GPU and manage the LRU cache of active experts.

        Args:
            expert_id (int): ID of the expert to move.
        """
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cuda':
            expert.to('cuda')
        self._active_experts[expert_id] = self._step
        
        # Move the least recently used expert to CPU if the number of active experts exceeds the limit
        if len(self._active_experts) > self.max_gpu_experts:
            lru_expert_id, _ = self._active_experts.popitem(last=False)
            self._move_expert_to_cpu(lru_expert_id)

    def _move_expert_to_cpu(self, expert_id):
        """
        Move an expert to CPU.

        Args:
            expert_id (int): ID of the expert to move.
        """
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cpu':
            expert.to('cpu')

    def _monitor_expert_balance(self, expert_indices):
        """
        Monitor expert load balance with enhanced monitoring for both training and inference.

        Args:
            expert_indices (torch.Tensor): Indices of experts assigned to each input.
        """
        # Ensure expert_indices is of integer type for bincount
        if expert_indices.dtype != torch.long:
            expert_indices = expert_indices.long()
        
        # Calculate the usage frequency of each expert
        expert_counts = torch.bincount(expert_indices.flatten(), minlength=self.num_experts)
        expert_load = expert_counts.float() / expert_indices.numel()
        
        # Record expert load history
        self.expert_load_history[self.load_history_ptr] = expert_load
        self.load_history_ptr = (self.load_history_ptr + 1) % 50
        
        # Calculate load imbalance and provide detailed monitoring
        if self.load_history_ptr > 10:
            recent_load = self.expert_load_history[:self.load_history_ptr].mean(0)
            load_variance = torch.var(recent_load)
            
            # Compute detailed statistics
            max_load_idx = torch.argmax(recent_load)
            min_load_idx = torch.argmin(recent_load)
            load_ratio = recent_load[max_load_idx] / (recent_load[min_load_idx] + 1e-8)
            
            # Issue warnings for severe imbalance
            balance_threshold = getattr(self, 'expert_load_balance_threshold', 0.15)
            if load_variance > balance_threshold or load_ratio > 8.0:
                if load_ratio > 10.0:  # Critical imbalance
                    # 使用新的日志系统替换旧的日志调用
                    logger.error(f"CRITICAL: Expert load severely imbalanced - max/min ratio: {load_ratio:.2f}, variance: {load_variance:.4f}")
                elif load_variance > balance_threshold * 1.33:  # High variance
                    # 使用新的日志系统替换旧的日志调用
                    logger.warning(f"WARNING: Expert load distribution showing high variance - ratio: {load_ratio:.2f}, variance: {load_variance:.4f}")
            
            # Provide adaptive load balancing suggestions
            suggestion_threshold = balance_threshold * 0.67
            if load_variance > suggestion_threshold or load_ratio > 5.0:
                suggested_temp = min(2.0, 1.0 + load_variance * 10)
                mode = "Training" if self.training else "Inference"
                # 使用新的日志系统替换旧的日志调用
                logger.info(f"SUGGESTION ({mode}): Consider increasing routing temperature to {suggested_temp:.2f} for better load distribution")

    def forward(self, x):
        """
        Forward pass of the MoE layer with load balancing.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the output tensor and auxiliary loss.
        """
        b, t, d = x.shape
        # Apply L2 smoothing for 8k sequence length
        if t >= 8192 and hasattr(self, 'l2_smooth_8k'):
            x = x * (1.0 - self.l2_smooth_8k) + x.mean(dim=1, keepdim=True) * self.l2_smooth_8k
        h = x.view(-1, d)  # [B*T, d]
        
        # Use different processing modes based on the type of routing gate
        if isinstance(self.gate, RuchbahStableMoEGate) and hasattr(self.gate, 'fixed_shape_mode') and self.gate.fixed_shape_mode:
            # Fixed shape mode: simpler processing
            scores, idx, aux_loss = self.gate(x)
            self._monitor_expert_balance(idx)
            
            # Compute outputs for each expert
            y = torch.zeros_like(h)
            for k in range(self.top_k):
                for expert_id in range(self.num_experts):
                    mask = (idx[:, k] == expert_id)
                    if mask.any():
                        expert = self.experts[expert_id]
                        if next(expert.parameters()).device.type != h.device.type:
                            expert.to(h.device)
                        
                        s_sel = scores[mask, k]
                        h_sel = h[mask]
                        y[mask] += s_sel.unsqueeze(1) * expert(h_sel)
            
            return y.view(b, t, d), aux_loss
        elif isinstance(self.gate, RuchbahStableMoEGate):
            scores, idx, aux_loss = self.gate(x)
            # Handle the output of StableMoEGate with capacity limitation
            
            # Create a mapping from original indices to expert assignments
            if len(scores) == 0 or len(idx) == 0:
                # Fallback to uniform expert assignment
                uniform_scores = torch.ones(h.size(0), device=h.device) / self.num_experts
                uniform_idx = torch.randint(0, self.num_experts, (h.size(0),), device=h.device)
                expert_assignment = [(uniform_scores, uniform_idx)]
            else:
                # Group scores and indices by expert
                expert_assignment = []
                current_pos = 0
                for expert_id in range(self.num_experts):
                    expert_mask = (idx == expert_id)
                    if expert_mask.any():
                        expert_scores = scores[expert_mask]
                        expert_indices = torch.full((expert_scores.size(0),), expert_id, device=h.device)
                        expert_assignment.append((expert_scores, expert_indices, expert_mask.nonzero().squeeze(-1)))
        else:
            # Standard MoEGate
            scores, idx, aux_loss = self.gate(x)
            expert_assignment = [(scores, idx)]
        
        # Monitor expert load balance
        if isinstance(self.gate, RuchbahStableMoEGate):
            # Create a representative idx for monitoring
            monitor_idx = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            for expert_id in range(self.num_experts):
                if len(expert_assignment) > expert_id and len(expert_assignment[expert_id]) > 2:
                    _, _, indices = expert_assignment[expert_id]
                    if len(indices) > 0:
                        monitor_idx[indices] = expert_id
            self._monitor_expert_balance(monitor_idx)
        else:
            self._monitor_expert_balance(idx)
        
        # Manage expert placement on GPU with predictive loading
        if self.num_experts > 8 and h.device.type == 'cuda':
            if isinstance(self.gate, RuchbahStableMoEGate):
                needed_experts = set(range(len(expert_assignment)))
                # Predict future expert needs
                if hasattr(self.gate, '_predict_future_load') and self.training:
                    try:
                        future_load_pred = self.gate._predict_future_load()
                        top_future_experts = torch.topk(future_load_pred, min(3, self.num_experts)).indices.cpu().numpy()
                        needed_experts.update(top_future_experts)
                    except Exception:
                        pass
            else:
                needed_experts = set(idx.cpu().numpy().flatten().tolist())
            
            # Calculate expert priority with temporal decay
            current_time = self._step if hasattr(self, '_step') else 0
            expert_priority = {}
            for expert_id in needed_experts:
                if self.load_history_ptr > 0:
                    historical_usage = self.expert_load_history[:self.load_history_ptr, expert_id].mean().item()
                    temporal_score = 1.0 / (1.0 + 0.01 * current_time)
                    expert_priority[expert_id] = historical_usage * 0.8 + temporal_score * 0.2
                else:
                    expert_priority[expert_id] = 0.5
            
            # Sort experts by priority and load them to GPU
            sorted_experts = sorted(expert_priority.items(), key=lambda x: x[1], reverse=True)
            experts_to_load = min(len(sorted_experts), self.max_gpu_experts)
            
            for expert_id, priority in sorted_experts[:experts_to_load]:
                if priority > 0.1:
                    self._move_expert_to_gpu(expert_id)
        
        # Compute outputs for each expert
        y = torch.zeros_like(h)
        expert_counts = torch.zeros(self.num_experts, device=h.device)
        
        if isinstance(self.gate, RuchbahStableMoEGate):
            # Handle StableMoEGate output
            for expert_id in range(self.num_experts):
                expert_found = False
                for assignment in expert_assignment:
                    if len(assignment) == 3:
                        expert_scores, expert_indices, original_indices = assignment
                        if len(expert_indices) > 0 and expert_indices[0].item() == expert_id:
                            expert = self.experts[expert_id]
                            if next(expert.parameters()).device.type != h.device.type:
                                expert.to(h.device)
                            
                            # Ensure indices are within bounds
                            valid_indices = original_indices[original_indices < h.size(0)]
                            if len(valid_indices) > 0:
                                h_sel = h[valid_indices]
                                expert_output = expert(h_sel)
                                
                                if len(expert_scores) == len(valid_indices):
                                    expert_output = expert_scores.unsqueeze(1) * expert_output
                                
                                y[valid_indices] += expert_output
                                expert_counts[expert_id] = len(valid_indices)
                                expert_found = True
                            break
                
                if not expert_found:
                    expert_counts[expert_id] = 0
        else:
            # Handle standard MoEGate output
            for expert_id in range(self.num_experts):
                mask = (idx == expert_id).any(dim=-1)
                if mask.any():
                    expert = self.experts[expert_id]
                    if next(expert.parameters()).device.type != h.device.type:
                        expert.to(h.device)
                    
                    # Process each top-k selection
                    for k in range(self.top_k):
                        sel_k = (idx[:, k] == expert_id)
                        if sel_k.any():
                            s_sel = scores[sel_k, k]
                            h_k = h[sel_k]
                            y[sel_k] += s_sel.unsqueeze(1) * expert(h_k)
                    
                    expert_counts[expert_id] = mask.sum()
        
        self._step += 1
        
        # Add additional load balancing loss during training
        if self.training:
            expert_distribution = expert_counts / (expert_counts.sum() + 1e-8)
            uniform_distribution = torch.ones_like(expert_distribution) / self.num_experts
            distribution_loss = 0.01 * torch.sum((expert_distribution - uniform_distribution) ** 2)
            aux_loss = aux_loss + distribution_loss
        
        return y.view(b, t, d), aux_loss