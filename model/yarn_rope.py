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

import math
import torch
import torch.nn as nn
from typing import Optional

class RuchbahYaRNRotaryEmbedding(nn.Module):
    """
    A module implementing YaRN (Yet Another RoPE Extension) with DynamicNTK for long-context positional encoding.
    Based on RoPE (Rotary Position Embedding), it supports dynamic scaling for handling different sequence lengths.
    """
    def __init__(self,
                 dim: int,
                 max_position_embeddings: int = 10485760,
                 base: int = 10000,
                 scale: float = 32.0,
                 original_max_position_embeddings: int = 4096,
                 device: Optional[torch.device] = None):
        """
        Initialize the RuchbahYaRNRotaryEmbedding module.

        Args:
            dim (int): Dimension of the embeddings.
            max_position_embeddings (int, optional): Maximum number of position embeddings. Defaults to 10485760.
            base (int, optional): Base value for frequency calculation. Defaults to 10000.
            scale (float, optional): Scaling factor for YaRN extension. Defaults to 32.0.
            original_max_position_embeddings (int, optional): Original maximum number of position embeddings. Defaults to 4096.
            device (Optional[torch.device], optional): Device to place the tensors on. Defaults to None.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        
        # Calculate the frequency factors for position encoding
        self.freq_factors = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        
        # Cache the inverse frequency for efficient computation
        self.register_buffer("inv_freq", self.freq_factors, persistent=False)
        
        # Register buffers for dynamic NTK parameters
        self.register_buffer("dynamic_base", torch.tensor(float(base), device=device), persistent=False)
        self.register_buffer("max_seq_len_seen", torch.tensor(0, device=device), persistent=False)
        
    def _compute_dynamic_ntk_scale(self, seq_len: int) -> float:
        """
        Compute the dynamic NTK scaling factor based on the input sequence length.

        Args:
            seq_len (int): Current sequence length.

        Returns:
            float: Dynamic scaling factor.
        """
        if seq_len <= self.original_max_position_embeddings:
            return 1.0
        
        # Compute the ratio of current sequence length to original maximum position embeddings
        ratio = seq_len / self.original_max_position_embeddings
        
        if seq_len > 1000000:
            # Apply logarithmic scaling for sequences longer than 1 million tokens
            log_scale = math.log(ratio) / math.log(10) + 1.0
            scale = ratio ** (self.dim / (self.dim - 2)) * log_scale
        else:
            # Apply standard NTK scaling for shorter sequences
            scale = ratio ** (self.dim / (self.dim - 2))
        
        # Ensure the scaling factor is at least 1.0
        scale = max(scale, 1.0)
        # Limit the scaling factor to twice the predefined scale
        return min(scale, self.scale * 2)
    
    def _compute_scale_factors(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Compute the YaRN scaling factors with dynamic NTK adjustment.

        Args:
            seq_len (int): Current sequence length.
            device (Optional[torch.device], optional): Device to place the tensors on. Defaults to None.

        Returns:
            torch.Tensor: YaRN scaling factors with dynamic NTK.
        """
        # Update the maximum sequence length seen so far
        if seq_len > self.max_seq_len_seen:
            self.max_seq_len_seen = torch.tensor(seq_len, device=device)
        
        # Compute the dynamic NTK scaling factor
        ntk_scale = self._compute_dynamic_ntk_scale(seq_len)
        
        # Generate position indices from 0 to seq_len - 1
        positions = torch.arange(seq_len, device=device)
        
        # Initialize scale factors with ones
        scale_factors = torch.ones(seq_len, device=device)
        
        if ntk_scale > 1.0:
            # Calculate the crossover point for applying YaRN extension
            crossover = int(math.sqrt(self.original_max_position_embeddings))
            if seq_len > crossover:
                # Apply YaRN extension with NTK scaling to positions beyond the crossover point
                high_positions = positions[crossover:]
                scale_factors[crossover:] = crossover * (high_positions / crossover) ** (1.0 / (ntk_scale * self.scale))
        
        return scale_factors
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Perform the forward pass of the YaRNRotaryEmbedding module with dynamic NTK.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, n_head, seq_len, head_dim] or [batch, seq_len, dim].
            seq_len (int, optional): Length of the sequence. If None, uses x.shape[-2].

        Returns:
            torch.Tensor: Tensor with YaRN rotary embeddings applied with dynamic NTK scaling.

        Raises:
            ValueError: If the input tensor is not 3D or 4D.
        """
        # Get the device of the input tensor
        device = x.device
        
        if x.dim() == 4:  # [batch, n_head, seq_len, head_dim]
            actual_seq_len = seq_len or x.shape[2]
            head_dim = x.shape[3]
            # Use head_dim as embedding dimension for 4D tensor
            embedding_dim = head_dim  
        elif x.dim() == 3:  # [batch, seq_len, dim]
            actual_seq_len = seq_len or x.shape[1]
            embedding_dim = x.shape[2]
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {x.dim()}D")
        
        # Compute dynamic scale factors based on the actual sequence length
        scale_factors = self._compute_scale_factors(actual_seq_len, device)
        
        # Generate position indices for the actual sequence
        t = torch.arange(actual_seq_len, device=device, dtype=torch.float32)
        
        # Apply dynamic YaRN and NTK scaling to position indices
        t = t * scale_factors
        
        # Calculate dynamic frequencies for the appropriate dimension
        dynamic_freq = 1.0 / (self.dynamic_base ** (torch.arange(0, embedding_dim, 2).float().to(device) / embedding_dim))
        freqs = torch.outer(t, dynamic_freq)
        
        # Compute cosine and sine values for the frequencies
        cos = freqs.cos()  # [actual_seq_len, embedding_dim//2]
        sin = freqs.sin()  # [actual_seq_len, embedding_dim//2]
        
        # Trim cosine and sine values to match the input sequence length
        if x.dim() == 4:
            cos = cos[:x.shape[2], :]
            sin = sin[:x.shape[2], :]
        else:
            cos = cos[:x.shape[1], :]
            sin = sin[:x.shape[1], :]
        
        # Apply rotation to the input tensor
        return self._rotate_half(x, cos, sin)
    
    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Rotate the input tensor by half using the provided cosine and sine values.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, n_head, seq_len, head_dim] or [batch, seq_len, dim].
            cos (torch.Tensor): Cosine values with shape [seq_len, embedding_dim//2].
            sin (torch.Tensor): Sine values with shape [seq_len, embedding_dim//2].

        Returns:
            torch.Tensor: Rotated tensor.
        """
        if x.dim() == 4:  # [batch, n_head, seq_len, head_dim]
            # Add batch and head dimensions for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:  # [batch, seq_len, dim]
            # Add batch dimension for broadcasting
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        # Split the input tensor into two halves
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        
        # Apply rotation operation and concatenate the results
        rotated = torch.cat((-x2 * sin + x1 * cos, x1 * sin + x2 * cos), dim=-1)
        
        return rotated
    
    def extra_repr(self) -> str:
        """
        Return extra representation information of the module.

        Returns:
            str: String containing module parameters.
        """
        return (f"dim={self.dim}, max_position_embeddings={self.max_position_embeddings}, "
                f"base={self.base}, scale={self.scale}, "
                f"original_max_position_embeddings={self.original_max_position_embeddings}")

class RuchbahDynamicYaRNRotaryEmbedding(nn.Module):
    """
    Dynamic YaRN RoPE with learned scaling factors for improved long-context extrapolation.
    
    Enhancements over base YaRN:
    - Learned scaling factors instead of fixed parameters
    - Task-aware position scaling
    - Improved NTK interpolation for ultra-long sequences
    """
    
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 10485760,
        base: int = 10000,
        scale: float = 32.0,
        original_max_position_embeddings: int = 4096,
        device: Optional[torch.device] = None,
        enable_learned_scaling: bool = True,
        enable_task_aware: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.enable_learned_scaling = enable_learned_scaling
        self.enable_task_aware = enable_task_aware
        
        self.freq_factors = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", self.freq_factors, persistent=False)
        self.register_buffer("dynamic_base", torch.tensor(float(base), device=device), persistent=False)
        self.register_buffer("max_seq_len_seen", torch.tensor(0, device=device), persistent=False)
        
        if enable_learned_scaling:
            self.learned_scale = nn.Parameter(torch.tensor(1.0))
            self.ntk_scale_factor = nn.Parameter(torch.tensor(1.0))
            self.log_scale_factor = nn.Parameter(torch.tensor(0.0))
        
        if enable_task_aware:
            self.task_scale_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, 2)
            )
        
    def _compute_dynamic_ntk_scale(self, seq_len: int) -> float:
        if seq_len <= self.original_max_position_embeddings:
            return 1.0
        
        ratio = seq_len / self.original_max_position_embeddings
        
        base_scale = ratio ** (self.dim / (self.dim - 2))
        
        if hasattr(self, 'learned_scale'):
            learned_multiplier = torch.sigmoid(self.learned_scale)
            ntk_multiplier = torch.sigmoid(self.ntk_scale_factor)
            log_factor = torch.exp(self.log_scale_factor)
            
            if seq_len > 1000000:
                log_scale = math.log(ratio) / math.log(10) + 1.0
                scale = base_scale * log_scale * log_factor
            else:
                scale = base_scale * ntk_multiplier
            
            scale = scale * learned_multiplier
        else:
            if seq_len > 1000000:
                log_scale = math.log(ratio) / math.log(10) + 1.0
                scale = ratio ** (self.dim / (self.dim - 2)) * log_scale
            else:
                scale = ratio ** (self.dim / (self.dim - 2))
        
        scale = max(scale, 1.0)
        return min(scale, self.scale * 2)
    
    def _compute_scale_factors(self, seq_len: int, task_embedding: torch.Tensor = None) -> torch.Tensor:
        if seq_len > self.max_seq_len_seen:
            self.max_seq_len_seen = torch.tensor(seq_len, device=task_embedding.device if task_embedding is not None else self.dynamic_base.device)
        
        ntk_scale = self._compute_dynamic_ntk_scale(seq_len)
        
        positions = torch.arange(seq_len, device=self.dynamic_base.device)
        scale_factors = torch.ones(seq_len, device=self.dynamic_base.device)
        
        if ntk_scale > 1.0:
            crossover = int(math.sqrt(self.original_max_position_embeddings))
            
            task_modulation = torch.ones(seq_len, device=self.dynamic_base.device)
            if task_embedding is not None and self.enable_task_aware:
                task_weights = self.task_scale_net(task_embedding.mean(dim=0))
                task_modulation = 1.0 + 0.1 * task_weights[0] * torch.arange(seq_len, device=self.dynamic_base.device) / seq_len
            
            if seq_len > crossover:
                high_positions = positions[crossover:]
                base_scaling = (high_positions / crossover) ** (1.0 / (ntk_scale * self.scale))
                scale_factors[crossover:] = base_scaling * task_modulation[crossover:]
        
        return scale_factors
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: int = None,
        task_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        device = x.device
        
        if x.dim() == 4:
            actual_seq_len = seq_len or x.shape[2]
            head_dim = x.shape[3]
            embedding_dim = head_dim
        elif x.dim() == 3:
            actual_seq_len = seq_len or x.shape[1]
            embedding_dim = x.shape[2]
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {x.dim()}D")
        
        scale_factors = self._compute_scale_factors(actual_seq_len, task_embedding)
        
        t = torch.arange(actual_seq_len, device=device, dtype=torch.float32)
        t = t * scale_factors
        
        dynamic_freq = 1.0 / (self.dynamic_base ** (torch.arange(0, embedding_dim, 2).float().to(device) / embedding_dim))
        freqs = torch.outer(t, dynamic_freq)
        
        cos = freqs.cos()
        sin = freqs.sin()
        
        if x.dim() == 4:
            cos = cos[:x.shape[2], :]
            sin = sin[:x.shape[2], :]
        else:
            cos = cos[:x.shape[1], :]
            sin = sin[:x.shape[1], :]
        
        return self._rotate_half(x, cos, sin)
    
    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        rotated = torch.cat((-x2 * sin + x1 * cos, x1 * sin + x2 * cos), dim=-1)
        return rotated