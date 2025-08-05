#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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
        
        # Precompute YaRN scaling factors
        scale_factors = self._compute_scale_factors(device)
        
        # Cache position embeddings
        self.register_buffer("inv_freq", self.freq_factors, persistent=False)
        self.register_buffer("scale_factors", scale_factors, persistent=False)
        
    def _compute_scale_factors(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute the YaRN scaling factors.

        Args:
            device (Optional[torch.device], optional): Device to place the tensors on. Defaults to None.

        Returns:
            torch.Tensor: YaRN scaling factors.
        """
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
        """Forward pass of the YaRNRotaryEmbedding module.

        Args:
            x (torch.Tensor): Input tensor.
            seq_len (int): Length of the sequence.

        Returns:
            torch.Tensor: Tensor with YaRN rotary embeddings applied.
        """
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
        """Rotate the input tensor by half.

        Args:
            x (torch.Tensor): Input tensor.
            cos (torch.Tensor): Cosine values.
            sin (torch.Tensor): Sine values.

        Returns:
            torch.Tensor: Rotated tensor.
        """
        # Split the tensor into two halves
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        
        # Apply rotation
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