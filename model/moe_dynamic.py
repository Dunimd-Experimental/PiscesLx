#!/usr/bin/env python3

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
import torch.nn as nn
from utils import RIGHT
import torch.nn.functional as F
from collections import OrderedDict

def moe_init_weights(m):
    """
    Initialize weights for MoE (Mixture of Experts) layers.
    Specifically, it initializes the weights of linear layers using Kaiming uniform initialization
    and sets the bias to zero if it exists.

    Args:
        m (nn.Module): The module to initialize weights for.
    """
    if isinstance(m, nn.Linear):
        # Initialize weights using Kaiming uniform initialization with a=sqrt(5)
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            # Initialize bias to zero
            nn.init.zeros_(m.bias)

class ExpertChoiceRouter(nn.Module):
    """
    A router module that implements expert choice routing strategy for MoE.
    Each expert selects top-k tokens based on routing scores.
    """
    def __init__(self, hidden_size, num_experts, capacity_factor=1.25, top_k=2):
        """
        Initialize the ExpertChoiceRouter.

        Args:
            hidden_size (int): The size of the hidden layer.
            num_experts (int): The number of experts.
            capacity_factor (float, optional): The capacity factor for experts. Defaults to 1.25.
            top_k (int, optional): The number of top tokens each expert selects. Defaults to 2.
        """
        super().__init__()
        # Linear layer to calculate routing scores
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        self.top_k = top_k
        
    def forward(self, x):
        """
        Perform forward pass of the expert choice router.

        Args:
            x (torch.Tensor): Input tensor with shape (batch*seq, hidden_size).

        Returns:
            tuple: A tuple containing expert indices, dispatch mask, and load balancing loss.
        """
        # Calculate the maximum number of tokens each expert can handle
        tokens_per_expert = int(x.shape[0] * self.capacity_factor / self.num_experts)
        
        # Calculate routing scores for each token and expert
        logits = self.gate(x)  # Shape: (tokens, experts)
        
        # Each expert selects the top tokens_per_expert tokens with the highest scores
        expert_indices = torch.topk(logits.T, tokens_per_expert, dim=1).indices  # Shape: (experts, capacity)
        
        # Initialize a dispatch mask to indicate which tokens are routed to which experts
        dispatch_mask = torch.zeros_like(logits)
        # Set the corresponding positions in the dispatch mask to 1
        dispatch_mask[expert_indices.T.flatten(), torch.arange(self.num_experts).repeat(tokens_per_expert)] = 1
        
        # Calculate the number of tokens assigned to each expert
        expert_load = dispatch_mask.sum(dim=0)
        
        # Calculate the load balancing loss to ensure uniform distribution of tokens among experts
        load_balancing_loss = (expert_load * torch.log(expert_load / expert_load.mean())).mean()
        
        return expert_indices, dispatch_mask, load_balancing_loss

class DynamicMoELayer(nn.Module):
    """
    A dynamic Mixture of Experts (MoE) layer with expert device management.
    This layer can dynamically manage the placement of experts on different devices (CPU/GPU).
    """
    _layer_count = 0
    def __init__(self, cfg, device=None, dtype=None):
        """
        Initialize the DynamicMoELayer.

        Args:
            cfg: Configuration object containing parameters for the layer.
            device (torch.device, optional): The device to place the model on. Defaults to None.
            dtype (torch.dtype, optional): The data type of the model. Defaults to None.
        """
        super().__init__()
        DynamicMoELayer._layer_count += 1
        self.cfg = cfg
        # Get the number of top experts to select for each token
        self.top_k = getattr(cfg, 'moe_top_k', 2)
        # Get the total number of experts
        self.num_experts = getattr(cfg, 'moe_num_experts', 8)
        # Initialize the expert choice router
        self.router = ExpertChoiceRouter(
            cfg.hidden_size, 
            self.num_experts, 
            capacity_factor=getattr(cfg, 'moe_capacity_factor', 1.25),
            top_k=self.top_k
        )
        # Initialize the expert modules
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)
            ) for _ in range(self.num_experts)
        ])
        # Initialize the weights of each expert
        for expert in self.experts:
            expert.apply(moe_init_weights)
        
        # Get the maximum number of experts that can be placed on GPU
        self.max_gpu_experts = getattr(cfg, 'max_gpu_experts', 4)
        # Ordered dictionary to record the last used step of each active expert
        self._active_experts = OrderedDict()  # expert_id: last_used_step
        self._step = 0
        
        if DynamicMoELayer._layer_count == 1:
            try:
                RIGHT(f"DynamicMoELayer: {self.num_experts} experts, top-{self.top_k} routing, capacity_factor={self.router.capacity_factor}")
            except UnicodeEncodeError:
                print(f"[OK] DynamicMoELayer: {self.num_experts} experts, top-{self.top_k} routing, capacity_factor={self.router.capacity_factor}")
    
    def _move_expert_to_gpu(self, expert_id):
        """
        Move an expert to GPU and manage the active experts list.
        If the number of active experts exceeds the limit, move the least recently used expert to CPU.

        Args:
            expert_id (int): The ID of the expert to move to GPU.
        """
        expert = self.experts[expert_id]
        # Move the expert to GPU if it's not already on GPU
        if next(expert.parameters()).device.type != 'cuda':
            expert.to('cuda')
        # Update the last used step of the expert
        self._active_experts[expert_id] = self._step
        
        # If the number of active experts exceeds the limit, move the least recently used expert to CPU
        if len(self._active_experts) > self.max_gpu_experts:
            lru_expert_id, _ = self._active_experts.popitem(last=False)
            self._move_expert_to_cpu(lru_expert_id)
    
    def _move_expert_to_cpu(self, expert_id):
        """
        Move an expert to CPU.

        Args:
            expert_id (int): The ID of the expert to move to CPU.
        """
        expert = self.experts[expert_id]
        # Move the expert to CPU if it's not already on CPU
        if next(expert.parameters()).device.type != 'cpu':
            expert.to('cpu')
    
    def forward(self, x):
        """
        Perform forward pass of the dynamic MoE layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, hidden_size).

        Returns:
            tuple: A tuple containing output tensor and load balancing loss.
        """
        batch_size, seq_len, hidden = x.shape
        # Flatten the input tensor
        x_flat = x.view(-1, hidden)
        
        # Get expert indices, dispatch mask, and load balancing loss from the router
        expert_indices, dispatch_mask, load_balancing_loss = self.router(x_flat)
        
        # Initialize the output tensor
        outputs = torch.zeros_like(x_flat)
        
        # Dynamically manage expert devices if there are more than 8 experts and the input is on GPU
        if self.num_experts > 8 and x.device.type == 'cuda':
            needed_experts = set()
            # Find all experts that need to process tokens
            for expert_id in range(self.num_experts):
                if expert_indices[expert_id].numel() > 0:
                    needed_experts.add(expert_id)
            # Move needed experts to GPU
            for expert_id in needed_experts:
                self._move_expert_to_gpu(expert_id)
        
        # Process tokens through each expert
        for expert_id, expert in enumerate(self.experts):
            tokens = x_flat[expert_indices[expert_id]]
            if tokens.shape[0] > 0:
                expert_out = expert(tokens)
                outputs[expert_indices[expert_id]] += expert_out
        
        self._step += 1
        return outputs.view(batch_size, seq_len, hidden), load_balancing_loss