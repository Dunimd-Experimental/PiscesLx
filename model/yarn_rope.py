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
from typing import Optional

class YaRNRotaryEmbedding(nn.Module):
    """Implementation of YaRN long-context positional encoding with DynamicNTK.
    Based on RoPE (Rotary Position Embedding) and extended with YaRN (Yet Another RoPE Extension) method.
    Supports ultra-long context lengths up to 10M tokens with dynamic NTK scaling.
    """
    def __init__(self,
                 dim: int,
                 max_position_embeddings: int = 10485760,
                 base: int = 10000,
                 scale: float = 32.0,
                 original_max_position_embeddings: int = 4096,
                 device: Optional[torch.device] = None):
        """Initialize the YaRNRotaryEmbedding module.

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
        
        # Calculate frequency factors
        self.freq_factors = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        
        # Cache position embeddings
        self.register_buffer("inv_freq", self.freq_factors, persistent=False)
        
        # Dynamic NTK parameters
        self.register_buffer("dynamic_base", torch.tensor(float(base), device=device), persistent=False)
        self.register_buffer("max_seq_len_seen", torch.tensor(0, device=device), persistent=False)
        
    def _compute_dynamic_ntk_scale(self, seq_len: int) -> float:
        """Compute dynamic NTK scaling factor based on sequence length.
        
        Args:
            seq_len (int): Current sequence length.
            
        Returns:
            float: Dynamic scaling factor.
        """
        if seq_len <= self.original_max_position_embeddings:
            return 1.0
        
        # Ultra-long context scaling with adaptive NTK
        ratio = seq_len / self.original_max_position_embeddings
        
        # Use logarithmic scaling for ultra-long contexts (>1M)
        if seq_len > 1000000:
            # Logarithmic scaling for 1M-10M range
            log_scale = math.log(ratio) / math.log(10) + 1.0
            scale = ratio ** (self.dim / (self.dim - 2)) * log_scale
        else:
            # Standard NTK scaling for shorter contexts
            scale = ratio ** (self.dim / (self.dim - 2))
        
        # Ensure minimum resolution for ultra-long contexts
        scale = max(scale, 1.0)
        return min(scale, self.scale * 2)  # Allow 2x scale for 10M context
    
    def _compute_scale_factors(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute YaRN scaling factors with dynamic NTK adjustment.

        Args:
            seq_len (int): Current sequence length.
            device (Optional[torch.device], optional): Device to place the tensors on. Defaults to None.

        Returns:
            torch.Tensor: YaRN scaling factors with dynamic NTK.
        """
        # Update max sequence length seen
        if seq_len > self.max_seq_len_seen:
            self.max_seq_len_seen = torch.tensor(seq_len, device=device)
        
        # Calculate dynamic NTK scaling
        ntk_scale = self._compute_dynamic_ntk_scale(seq_len)
        
        # Generate position indices
        positions = torch.arange(seq_len, device=device)
        
        # Apply combined YaRN + DynamicNTK scaling
        scale_factors = torch.ones(seq_len, device=device)
        
        if ntk_scale > 1.0:
            # Calculate crossover point (similar to original YaRN but with NTK adjustment)
            crossover = int(math.sqrt(self.original_max_position_embeddings))
            if seq_len > crossover:
                # Apply YaRN extension with NTK scaling
                high_positions = positions[crossover:]
                scale_factors[crossover:] = crossover * (high_positions / crossover) ** (1.0 / (ntk_scale * self.scale))
        
        return scale_factors
    
    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """Forward pass of the YaRNRotaryEmbedding module with dynamic NTK.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, n_head, seq_len, head_dim] or [batch, seq_len, dim].
            seq_len (int, optional): Length of the sequence. If None, uses x.shape[-2].

        Returns:
            torch.Tensor: Tensor with YaRN rotary embeddings applied with dynamic NTK scaling.
        """
        # Get device and determine dimensions based on input shape
        device = x.device
        
        if x.dim() == 4:  # [batch, n_head, seq_len, head_dim]
            actual_seq_len = seq_len or x.shape[2]
            head_dim = x.shape[3]
            embedding_dim = head_dim  # Use head_dim for 4D tensor
        elif x.dim() == 3:  # [batch, seq_len, dim]
            actual_seq_len = seq_len or x.shape[1]
            embedding_dim = x.shape[2]
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {x.dim()}D")
        
        # Compute dynamic scale factors based on current sequence length
        scale_factors = self._compute_scale_factors(actual_seq_len, device)
        
        # Get position indices for the actual sequence
        t = torch.arange(actual_seq_len, device=device, dtype=torch.float32)
        
        # Apply dynamic YaRN + NTK scaling
        t = t * scale_factors
        
        # Calculate frequencies for the appropriate dimension
        # Use embedding_dim instead of self.dim for proper dimension matching
        dynamic_freq = 1.0 / (self.dynamic_base ** (torch.arange(0, embedding_dim, 2).float().to(device) / embedding_dim))
        freqs = torch.outer(t, dynamic_freq)
        
        # Get cos and sin for the actual input length and dimension
        cos = freqs.cos()  # [actual_seq_len, embedding_dim//2]
        sin = freqs.sin()  # [actual_seq_len, embedding_dim//2]
        
        # Trim to actual sequence length of input
        if x.dim() == 4:
            cos = cos[:x.shape[2], :]
            sin = sin[:x.shape[2], :]
        else:
            cos = cos[:x.shape[1], :]
            sin = sin[:x.shape[1], :]
        
        # Apply rotation
        return self._rotate_half(x, cos, sin)
    
    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate the input tensor by half.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, n_head, seq_len, head_dim] or [batch, seq_len, dim].
            cos (torch.Tensor): Cosine values with shape [seq_len, embedding_dim//2].
            sin (torch.Tensor): Sine values with shape [seq_len, embedding_dim//2].

        Returns:
            torch.Tensor: Rotated tensor.
        """
        # Handle different input shapes
        if x.dim() == 4:  # [batch, n_head, seq_len, head_dim]
            # Add batch and head dimensions for broadcasting: [1, 1, seq_len, head_dim//2]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            
        elif x.dim() == 3:  # [batch, seq_len, dim]
            # Add batch dimension for broadcasting: [1, seq_len, dim//2]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        # Split the tensor into two halves
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        
        # Apply rotation with proper broadcasting
        rotated = torch.cat((-x2 * sin + x1 * cos, x1 * sin + x2 * cos), dim=-1)
        
        return rotated
    
    def extra_repr(self) -> str:
        """Return extra representation information of the module.

        Returns:
            str: String containing module parameters.
        """
        return (f"dim={self.dim}, max_position_embeddings={self.max_position_embeddings}, "
                f"base={self.base}, scale={self.scale}, "
                f"original_max_position_embeddings={self.original_max_position_embeddings}")