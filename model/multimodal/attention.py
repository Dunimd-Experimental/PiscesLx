#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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
from torch import nn
import torch.nn.functional as F

class ArcticCrossModalAttention(nn.Module):
    """
    A cross-modal attention module that fuses information from different modalities using multi-head attention.
    Supports both xFormers memory-efficient attention and standard scaled dot-product attention.
    """
    def __init__(self, cfg):
        """
        Initialize the cross-modal attention module.

        Args:
            cfg: Configuration object containing attention parameters.
                Expected attributes:
                - n_head (int): Number of attention heads.
                - hidden_size (int): Hidden size of the input features.
        """
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.n_head
        self.hidden_size = cfg.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Linear projection layers for query, key, value, and output
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Layer normalization layers
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        """
        Perform cross-modal attention.

        Args:
            query (torch.Tensor): Query tensor of shape [B, T_q, hidden_size].
            key (torch.Tensor): Key tensor of shape [B, T_k, hidden_size].
            value (torch.Tensor): Value tensor of shape [B, T_k, hidden_size].
            mask (torch.Tensor, optional): Mask tensor of shape [B, T_q, T_k]. 
                Zero values indicate positions to be masked. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [B, T_q, hidden_size].
        """
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape

        # Normalize and project input tensors to query, key, and value space
        # Then reshape and transpose to [B, num_heads, T, head_dim]
        Q = self.q_proj(self.norm1(query)).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(self.norm2(key)).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Try to import xFormers for memory-efficient attention
        try:
            from xformers.ops import memory_efficient_attention  # type: ignore
            _use_xformers = True
        except ImportError:
            _use_xformers = False

        if _use_xformers and mask is None:
            # Use xFormers memory-efficient attention if available and no mask is provided
            out = memory_efficient_attention(Q, K, V)  # Output shape: [B, num_heads, T_q, head_dim]
            out = out.transpose(1, 2).contiguous().view(B, T_q, self.hidden_size)
        else:
            # Use standard scaled dot-product attention as fallback
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, V)
            out = out.transpose(1, 2).contiguous().view(B, T_q, self.hidden_size)

        return self.o_proj(out)