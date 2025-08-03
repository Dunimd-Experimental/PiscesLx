#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import math
import torch
import torch.nn as nn
from typing import Optional

class YaRNRotaryEmbedding(nn.Module):
    """Implementation of YaRN long-context positional encoding.
    Based on RoPE (Rotary Position Embedding) and extended with YaRN (Yet Another RoPE Extension) method.
    Supports ultra-long context lengths up to 10M tokens.
    """
    def __init__(self,
                 dim: int,
                 max_position_embeddings: int = 10485760,
                 base: int = 10000,
                 scale: float = 32.0,
                 original_max_position_embeddings: int = 4096,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        
        # Calculate frequency factors
        self.freq_factors = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        
        # Precompute YaRN scaling factors
        scale_factors = self._compute_scale_factors(device)
        
        # Cache position embeddings
        self.register_buffer("inv_freq", self.freq_factors, persistent=False)
        self.register_buffer("scale_factors", scale_factors, persistent=False)
        
    def _compute_scale_factors(self, device: Optional[torch.device] = None) -> torch.Tensor:
        # Generate original position indices
        original_positions = torch.arange(self.original_max_position_embeddings, device=device)
        
        # Calculate YaRN scaling factors
        scale_factors = torch.ones(self.max_position_embeddings, device=device)
        
        # Apply YaRN extension formula
        if self.scale > 1.0:
            # Calculate crossover point
            crossover = int(math.sqrt(self.original_max_position_embeddings))
            # Calculate scaling region
            high_positions = torch.arange(crossover, self.max_position_embeddings, device=device)
            # Apply logarithmic scaling
            scale_factors[crossover:] = crossover * (high_positions / crossover) ** (1.0 / self.scale)
        
        return scale_factors
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # Get device
        device = x.device
        
        # Get position indices
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # Apply YaRN scaling
        t = t * self.scale_factors[:seq_len].to(device)
        
        # Calculate frequencies
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        
        # Apply rotation
        return self._rotate_half(x, cos, sin)
    
    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Split the tensor into two halves
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        
        # Apply rotation
        rotated = torch.cat((-x2 * sin + x1 * cos, x1 * sin + x2 * cos), dim=-1)
        
        return rotated
    
    def extra_repr(self) -> str:
        return (f"dim={self.dim}, max_position_embeddings={self.max_position_embeddings}, "
                f"base={self.base}, scale={self.scale}, "
                f"original_max_position_embeddings={self.original_max_position_embeddings}")