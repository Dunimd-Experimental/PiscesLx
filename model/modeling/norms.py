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

"""
Normalization and embedding modules for Arctic model.

This module provides RMS normalization and rotary position embedding
implementations used in the transformer architecture.
"""

import math
import torch
from torch import nn

def _arctic_init_weights(m):
    """
    Initialize weights for Arctic model modules.

    Applies Kaiming uniform initialization for linear layers and normal
    initialization for embedding layers.

    Args:
        m (nn.Module): Module to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)

class ArcticRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes the input by dividing by the root mean square of the
    features, then scales by a learned parameter.
    """

    def __init__(self, dim, eps=1e-6):
        """
        Initialize RMS normalization layer.

        Args:
            dim (int): Dimension to normalize over.
            eps (float): Small epsilon value for numerical stability.
                Defaults to 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        """
        Apply RMS normalization.

        Args:
            x (torch.Tensor): Input tensor of shape [..., dim].

        Returns:
            torch.Tensor: Normalized and scaled tensor of same shape as input.
        """
        # Compute root mean square: sqrt(mean(x^2) + eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * rms

class ArcticRotaryEmbedding(nn.Module):
    """
    Rotary position embedding (RoPE).

    Applies rotary position embeddings to input tensors by rotating
    pairs of features based on their position in the sequence.
    """

    def __init__(self, dim, max_seq_len=8192, base=1e6, device=None, dtype=None):
        """
        Initialize rotary embedding module.

        Precomputes cosine and sine values for all positions up to max_seq_len.

        Args:
            dim (int): Dimension of the embedding (must be even).
            max_seq_len (int): Maximum sequence length to precompute.
                Defaults to 8192.
            base (float): Base frequency for computing inverse frequencies.
                Defaults to 1e6.
            device (torch.device, optional): Device to create buffers on.
            dtype (torch.dtype, optional): Data type for buffers.
        """
        super().__init__()
        # Compute inverse frequencies: 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # Compute frequencies: outer product of positions and inverse frequencies
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x, seq_len):
        """
        Apply rotary position embedding.

        Rotates pairs of features using precomputed cosine and sine values.

        Args:
            x (torch.Tensor): Input tensor of shape [..., dim].
            seq_len (int): Current sequence length (must be <= max_seq_len).

        Returns:
            torch.Tensor: Rotated tensor of same shape as input.
        """
        # Get cosine and sine values for current sequence length
        cos, sin = self.cos[:seq_len], self.sin[:seq_len]

        # Split features into pairs: [x0, x1, x2, x3, ...] -> [x0, x2, ...], [x1, x3, ...]
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.flatten(-2)
