#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from torch import nn
from utils.log import RIGHT
import torch.nn.functional as F
from collections import OrderedDict

# Initialize weights for MoE (Mixture of Experts) related linear layers
def moe_init_weights(m):
    """
    Initialize weights for linear layers in MoE.

    Args:
        m (torch.nn.Module): A PyTorch module.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MoEGate(nn.Module):
    """Expert routing gate for MoE (top-k configurable) with load balancing"""
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None, 
                 load_balance_alpha=0.01, noise_std=0.1):
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
        """
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.top_k = top_k
        self.num_experts = num_experts
        self.load_balance_alpha = load_balance_alpha
        self.noise_std = noise_std
        
        # Routing stability parameters
        self.register_buffer('expert_usage_count', torch.zeros(num_experts))
        self.register_buffer('temperature', torch.tensor(1.0))
        self.min_temperature = 0.1
        
    def forward(self, x):
        """
        Forward pass of the MoE gate with load balancing and stability.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing scores, indices, and load balancing loss.
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Compute routing logits
        logits = self.gate(x_flat)  # [N, num_experts]
        
        # Add routing noise to improve exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        
        # Temperature annealing
        logits = logits / self.temperature
        
        # Top-k routing
        scores, idx = torch.topk(logits, self.top_k, dim=-1)  # [N, top_k]
        
        # Calculate expert usage frequency
        expert_mask = torch.zeros(x_flat.size(0), self.num_experts, device=x.device)
        expert_mask.scatter_(1, idx, 1)
        current_usage = expert_mask.sum(0)
        
        # Update expert usage count
        if self.training:
            self.expert_usage_count += current_usage.detach()
            
            # Dynamically adjust temperature
            usage_variance = torch.var(self.expert_usage_count)
            if usage_variance > 0:
                self.temperature = torch.clamp(
                    self.temperature * 0.999, 
                    min=self.min_temperature
                )
        
        # Calculate load balancing loss
        expert_load = current_usage.float() / x_flat.size(0)
        ideal_load = torch.ones_like(expert_load) / self.num_experts
        load_balance_loss = self.load_balance_alpha * torch.sum(
            (expert_load - ideal_load) ** 2
        )
        
        # Calculate routing scores
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(x)
        
        return scores, idx, load_balance_loss

class StableMoEGate(nn.Module):
    """Stable MoE routing gate to prevent routing collapse with load prediction"""
    
    def __init__(self, hidden_size, num_experts, top_k=2, device=None, dtype=None,
                 capacity_factor=1.0, min_capacity=4, prediction_horizon=10, 
                 fixed_shape_mode=False):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.top_k = top_k
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.min_capacity = min_capacity
        self.prediction_horizon = prediction_horizon
        self.fixed_shape_mode = fixed_shape_mode  # For gradient checkpointing compatibility
        
        # Routing stability parameters
        self.register_buffer('routing_history', torch.zeros(100, num_experts))
        self.register_buffer('history_ptr', torch.tensor(0))
        
        # Load prediction network using LSTM
        self.load_predictor = nn.LSTM(
            input_size=num_experts,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.predictor_head = nn.Linear(64, num_experts * prediction_horizon)
        
        # Expert load tracking
        self.register_buffer('expert_load_buffer', torch.zeros(50, num_experts))
        self.register_buffer('load_buffer_ptr', torch.tensor(0))
        
    def _predict_future_load(self):
        """Predict future expert load using LSTM"""
        if self.load_buffer_ptr < 10:
            return torch.ones(self.num_experts) / self.num_experts
        
        # Get historical load data
        historical_load = self.expert_load_buffer[:self.load_buffer_ptr].unsqueeze(0)  # [1, seq_len, num_experts]
        
        # Predict future load
        with torch.no_grad():
            lstm_out, _ = self.load_predictor(historical_load)
            predictions = self.predictor_head(lstm_out[:, -1, :])  # [1, num_experts * horizon]
            
        # Reshape to get predictions for next step
        future_load = predictions.view(self.num_experts, self.prediction_horizon).mean(dim=1)
        future_load = torch.softmax(future_load, dim=0)
        
        return future_load
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Fixed shape mode for gradient checkpointing compatibility
        if self.fixed_shape_mode:
            # Simple Top-K routing without capacity limitations
            logits = self.gate(x_flat)
            scores = F.softmax(logits, dim=-1)
            top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
            
            # Normalize scores
            top_scores = F.softmax(top_scores, dim=-1, dtype=torch.float32).type_as(x)
            
            return top_scores, top_idx, torch.tensor(0.0, device=x.device)
        
        # Original dynamic routing logic
        # Calculate expert capacity
        tokens_per_expert = max(
            int(x_flat.size(0) * self.capacity_factor / self.num_experts),
            self.min_capacity
        )
        
        # Compute routing logits
        logits = self.gate(x_flat)
        
        # Record routing history
        if self.training:
            expert_probs = F.softmax(logits, dim=-1)
            expert_usage = expert_probs.mean(0)
            self.routing_history[self.history_ptr] = expert_usage.detach()
            self.history_ptr = (self.history_ptr + 1) % 100
            
            # Record expert load for prediction
            self.expert_load_buffer[self.load_buffer_ptr] = expert_usage.detach()
            self.load_buffer_ptr = (self.load_buffer_ptr + 1) % 50
        
        # Predict future load and adjust routing
        predicted_load = self._predict_future_load().to(x.device)
        
        # Calculate expert load balance
        if self.history_ptr > 0:
            recent_usage = self.routing_history[:self.history_ptr].mean(0)
            usage_entropy = -torch.sum(recent_usage * torch.log(recent_usage + 1e-8))
            max_entropy = torch.log(torch.tensor(self.num_experts))
            balance_ratio = usage_entropy / (max_entropy + 1e-8)
            
            # Use predicted load to adjust routing strategy
            load_adjustment = 0.1 * (predicted_load - recent_usage)
            logits = logits + load_adjustment.unsqueeze(0)
            
            # Add noise if the load is unbalanced
            if balance_ratio < 0.7:
                noise_scale = 0.1 * (1.0 - balance_ratio)
                logits = logits + torch.randn_like(logits) * noise_scale
        
        # Use sinkhorn routing algorithm to improve stability
        scores = F.softmax(logits, dim=-1)
        
        # Top-k selection
        top_scores, top_idx = torch.topk(scores, self.top_k, dim=-1)
        
        # Capacity limitation
        final_scores = []
        final_indices = []
        
        for expert_id in range(self.num_experts):
            mask = (top_idx == expert_id).any(dim=-1)
            if mask.sum() > tokens_per_expert:
                # Exceed capacity, re-route
                expert_scores = scores[mask, expert_id]
                _, top_indices = torch.topk(expert_scores, tokens_per_expert)
                keep_mask = torch.zeros_like(expert_scores, dtype=torch.bool)
                keep_mask[top_indices] = True
                mask[mask.clone()] = keep_mask
            
            if mask.any():
                final_scores.append(scores[mask])
                final_indices.append(torch.full_like(scores[mask], expert_id))
        
        # If there is no valid routing, use uniform distribution
        if len(final_scores) == 0:
            uniform_expert = torch.randint(0, self.num_experts, (x_flat.size(0),), device=x.device)
            final_scores = [torch.ones(x_flat.size(0), device=x.device) / self.top_k]
            final_indices = [uniform_expert]
        
        return torch.cat(final_scores), torch.cat(final_indices), torch.tensor(0.0, device=x.device)

class MoELayer(nn.Module):
    """Mixture of Experts layer with improved load balancing and stability"""
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
        MoELayer._layer_count += 1
        self.cfg = cfg
        self.top_k = getattr(cfg, 'moe_top_k', 2)
        self.num_experts = getattr(cfg, 'moe_num_experts', 8)
        
        # Use stable routing gate
        if use_stable_gate:
            # Enable fixed shape mode for 0.5B model to avoid gradient checkpointing issues
            fixed_shape_mode = (cfg.hidden_size <= 768)  # 0.5B and smaller models
            self.gate = StableMoEGate(
                cfg.hidden_size, self.num_experts, top_k=self.top_k,
                device=device, dtype=dtype,
                capacity_factor=getattr(cfg, 'moe_capacity_factor', 1.0),
                min_capacity=getattr(cfg, 'moe_min_capacity', 4),
                prediction_horizon=getattr(cfg, 'moe_prediction_horizon', 10),
                fixed_shape_mode=fixed_shape_mode
            )
        else:
            self.gate = MoEGate(
                cfg.hidden_size, self.num_experts, top_k=self.top_k,
                device=device, dtype=dtype,
                load_balance_alpha=getattr(cfg, 'moe_load_balance_alpha', 0.01),
                noise_std=getattr(cfg, 'moe_noise_std', 0.1)
            )
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)
            ) for _ in range(self.num_experts)
        ])
        for expert in self.experts:
            expert.apply(moe_init_weights)
        
        if MoELayer._layer_count == 1:
            gate_type = "Stable" if use_stable_gate else "Standard"
            RIGHT(f"MoELayer: {self.num_experts} experts, top-{self.top_k} routing, {gate_type} gate")

        self.max_gpu_experts = max_gpu_experts
        self._active_experts = OrderedDict()  # expert_id: last_used_step
        self._step = 0
        
        # Load balancing monitoring
        self.register_buffer('expert_load_history', torch.zeros(50, self.num_experts))
        self.register_buffer('load_history_ptr', torch.tensor(0))

    def _move_expert_to_gpu(self, expert_id):
        """
        Move an expert to GPU and manage the LRU cache of active experts.
        """
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cuda':
            expert.to('cuda')
        self._active_experts[expert_id] = self._step
        
        if len(self._active_experts) > self.max_gpu_experts:
            lru_expert_id, _ = self._active_experts.popitem(last=False)
            self._move_expert_to_cpu(lru_expert_id)

    def _move_expert_to_cpu(self, expert_id):
        """
        Move an expert to CPU.
        """
        expert = self.experts[expert_id]
        if next(expert.parameters()).device.type != 'cpu':
            expert.to('cpu')

    def _monitor_expert_balance(self, expert_indices):
        """
        Monitor expert load balance.
        """
        if self.training:
            # Ensure expert_indices is integer type for bincount
            if expert_indices.dtype != torch.long:
                expert_indices = expert_indices.long()
            
            # Calculate the usage frequency of each expert
            expert_counts = torch.bincount(expert_indices.flatten(), minlength=self.num_experts)
            expert_load = expert_counts.float() / expert_indices.numel()
            
            # Record load history
            self.expert_load_history[self.load_history_ptr] = expert_load
            self.load_history_ptr = (self.load_history_ptr + 1) % 50
            
            # Calculate load imbalance
            if self.load_history_ptr > 10:
                recent_load = self.expert_load_history[:self.load_history_ptr].mean(0)
                load_variance = torch.var(recent_load)
                
                # Issue a warning if the load is severely unbalanced
                if load_variance > 0.1:
                    max_load_idx = torch.argmax(recent_load)
                    min_load_idx = torch.argmin(recent_load)
                    load_ratio = recent_load[max_load_idx] / (recent_load[min_load_idx] + 1e-8)
                    if load_ratio > 5.0:
                        RIGHT(f"Warning: Expert load imbalance detected - max/min ratio: {load_ratio:.2f}")

    def forward(self, x):
        """
        Forward pass of the MoE layer with load balancing.
        """
        b, t, d = x.shape
        h = x.view(-1, d)  # [B*T, d]
        
        # Use the new routing mechanism
        if isinstance(self.gate, StableMoEGate) and hasattr(self.gate, 'fixed_shape_mode') and self.gate.fixed_shape_mode:
            # Fixed shape mode: simpler processing
            scores, idx, aux_loss = self.gate(x)
            self._monitor_expert_balance(idx)
            
            # Simple expert computation for fixed shape mode
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
        elif isinstance(self.gate, StableMoEGate):
            scores, idx, aux_loss = self.gate(x)
            # StableMoEGate returns concatenated scores and indices from capacity limitation
            # scores and idx may have different sizes than h due to capacity constraints
            # We need to handle this carefully
            
            # Create a mapping from original indices to expert assignments
            if len(scores) == 0 or len(idx) == 0:
                # Fallback: uniform expert assignment
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
        
        # Monitor expert load
        if isinstance(self.gate, StableMoEGate):
            # For stable gate, create a representative idx for monitoring
            monitor_idx = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            for expert_id in range(self.num_experts):
                if len(expert_assignment) > expert_id and len(expert_assignment[expert_id]) > 2:
                    _, _, indices = expert_assignment[expert_id]
                    if len(indices) > 0:
                        monitor_idx[indices] = expert_id
            self._monitor_expert_balance(monitor_idx)
        else:
            self._monitor_expert_balance(idx)
        
        # GPU expert management
        if self.num_experts > 8 and h.device.type == 'cuda':
            if isinstance(self.gate, StableMoEGate):
                needed_experts = set(range(len(expert_assignment)))
            else:
                needed_experts = set(idx.cpu().numpy().flatten().tolist())
            for expert_id in needed_experts:
                self._move_expert_to_gpu(expert_id)
        
        # Efficient expert computation
        y = torch.zeros_like(h)
        expert_counts = torch.zeros(self.num_experts, device=h.device)
        
        if isinstance(self.gate, StableMoEGate):
            # Handle StableMoEGate output
            for expert_id in range(self.num_experts):
                # Find assignments for this expert
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
                                
                                # Apply scores if available and matching size
                                if len(expert_scores) == len(valid_indices):
                                    expert_output = expert_scores.unsqueeze(1) * expert_output
                                
                                y[valid_indices] += expert_output
                                expert_counts[expert_id] = len(valid_indices)
                                expert_found = True
                            break
                
                if not expert_found:
                    # No tokens assigned to this expert
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
        
        # Add additional load balancing loss
        if self.training:
            expert_distribution = expert_counts / (expert_counts.sum() + 1e-8)
            uniform_distribution = torch.ones_like(expert_distribution) / self.num_experts
            distribution_loss = 0.01 * torch.sum((expert_distribution - uniform_distribution) ** 2)
            aux_loss = aux_loss + distribution_loss
        
        return y.view(b, t, d), aux_loss