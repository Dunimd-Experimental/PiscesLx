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

"""Memory Cross-Attention for Knowledge Injection.

Implements the YvMemoryCrossAttention layer that injects retrieved
knowledge embeddings into the model's hidden flow via cross-attention
with gated fusion, enabling the model to access external knowledge
without storing it in its weights.

Architecture inspired by:
    Liang Wenfeng et al., "Engram: Conditional Memory via Scalable
    Lookup", arXiv:2601.07372, 2026.

Key Design:
    - Query: model hidden states (what the model "wants to know")
    - Key/Value: retrieved knowledge embeddings (what the store provides)
    - Multi-head cross-attention over top-K knowledge slots
    - Learnable gate controlling knowledge injection strength
    - Gated fusion: output = hidden + gate * cross_attn(hidden, knowledge)
    - RMSNorm pre/post for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .norms import YvRMSNorm
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("Yv.MemoryAttention", file_path=get_log_file("Yv.MemoryAttention"), enable_file=True)


class YvMemoryCrossAttention(nn.Module):
    """Cross-attention layer for knowledge injection into hidden flow.

    Takes model hidden states as queries and retrieved knowledge
    embeddings as keys/values, computes cross-attention, and fuses
    the result back through a gated residual connection.

    Architecture:
        pre_norm: RMSNorm on hidden states
        q_proj: hidden -> n_heads * head_dim (query from hidden states)
        k_proj: knowledge_dim -> n_heads * head_dim (key from knowledge)
        v_proj: knowledge_dim -> n_heads * head_dim (value from knowledge)
        o_proj: n_heads * head_dim -> hidden
        post_norm: RMSNorm on output
        gate: sigmoid(gate_param) scaling factor

    Math:
        Q = q_proj(norm(hidden))       # [B, T, n_heads, head_dim]
        K = k_proj(knowledge)          # [B, T, top_k, n_heads, head_dim]
        V = v_proj(knowledge)          # [B, T, top_k, n_heads, head_dim]
        scores = Q @ K^T / sqrt(d)     # [B, T, n_heads, 1, top_k]
        attn = softmax(scores) @ V     # [B, T, n_heads, head_dim]
        output = o_proj(attn)          # [B, T, hidden]
        fused = hidden + sigmoid(gate) * output

    Attributes:
        pre_norm (YvRMSNorm): Input normalization.
        q_proj (nn.Linear): Query projection from hidden states.
        k_proj (nn.Linear): Key projection from knowledge embeddings.
        v_proj (nn.Linear): Value projection from knowledge embeddings.
        o_proj (nn.Linear): Output projection back to hidden space.
        post_norm (YvRMSNorm): Output normalization.
        gate (nn.Parameter): Learnable fusion gate.
    """

    def __init__(
        self,
        hidden_size: int,
        knowledge_dim: int = 256,
        n_heads: int = 4,
        head_dim: Optional[int] = None,
        gate_init: float = 0.0,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize memory cross-attention layer.

        Args:
            hidden_size: Model hidden dimension.
            knowledge_dim: Knowledge slot embedding dimension (default 256).
            n_heads: Number of cross-attention heads (default 4).
            head_dim: Per-head dimension. Auto-computed if None.
            gate_init: Initial gate value (0 = start with no injection).
            dropout: Attention dropout rate.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.knowledge_dim = knowledge_dim
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // n_heads
        self.scale = self.head_dim ** -0.5

        # Ensure head_dim * n_heads will produce valid projections
        inner_dim = self.n_heads * self.head_dim

        # Normalization
        self.pre_norm = YvRMSNorm(hidden_size, device=device, dtype=dtype)
        self.post_norm = YvRMSNorm(hidden_size, device=device, dtype=dtype)

        # Query from hidden states, Key/Value from knowledge embeddings
        self.q_proj = nn.Linear(hidden_size, inner_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(knowledge_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(knowledge_dim, inner_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(inner_dim, hidden_size, bias=False, device=device, dtype=dtype)

        # Learnable gate: controls how much knowledge is injected
        self.gate = nn.Parameter(
            torch.tensor(gate_init, device=device, dtype=dtype)
        )

        # Dropout for attention weights
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Optional ablations
        self.use_qk_norm = True

        _LOG.info(
            f"YvMemoryCrossAttention initialized: hidden={hidden_size}, "
            f"knowledge_dim={knowledge_dim}, n_heads={n_heads}, "
            f"head_dim={self.head_dim}"
        )

    def _reshape_for_attention(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
        num_slots: int = 1,
    ) -> torch.Tensor:
        """Reshape tensor for multi-head attention computation.

        Args:
            x: Input tensor [B, T, K, inner_dim] or [B, T, inner_dim].
            batch_size: Batch size.
            seq_len: Sequence length.
            num_slots: Number of knowledge slots (1 for queries).

        Returns:
            Reshaped tensor [B, n_heads, T, K, head_dim] or [B, n_heads, T, head_dim].
        """
        if num_slots == 1:
            # Query reshape: [B, T, inner_dim] -> [B, T, n_heads, head_dim] -> [B, n_heads, T, head_dim]
            x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
            return x.transpose(1, 2).contiguous()
        else:
            # Knowledge reshape: [B, T, K, inner_dim] -> [B, T, K, n_heads, head_dim] -> [B, n_heads, T, K, head_dim]
            x = x.view(batch_size, seq_len, num_slots, self.n_heads, self.head_dim)
            return x.permute(0, 3, 1, 2, 4).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        knowledge: torch.Tensor,
        gate_override: Optional[float] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inject knowledge into hidden states via cross-attention.

        Args:
            hidden_states: Model hidden states [B, T, hidden_size].
            knowledge: Retrieved knowledge embeddings [B, T, top_k, knowledge_dim].
            gate_override: Optional override for the gate value.
            attention_mask: Optional mask for cross-attention [B, 1, T, top_k].

        Returns:
            Knowledge-injected hidden states [B, T, hidden_size].
        """
        residual = hidden_states
        batch_size, seq_len, _ = hidden_states.shape
        top_k = knowledge.shape[2]

        # Pre-normalize hidden states
        hidden_norm = self.pre_norm(hidden_states)

        # Project queries from hidden states
        q = self.q_proj(hidden_norm)  # [B, T, inner_dim]
        q = self._reshape_for_attention(q, batch_size, seq_len, num_slots=1)
        # q: [B, n_heads, T, head_dim]

        # Project keys and values from knowledge embeddings
        # knowledge: [B, T, top_k, knowledge_dim]
        k = self.k_proj(knowledge)  # [B, T, top_k, inner_dim]
        v = self.v_proj(knowledge)  # [B, T, top_k, inner_dim]

        k = self._reshape_for_attention(k, batch_size, seq_len, num_slots=top_k)
        v = self._reshape_for_attention(v, batch_size, seq_len, num_slots=top_k)
        # k, v: [B, n_heads, T, top_k, head_dim]

        # Optional QK normalization for training stability
        if self.use_qk_norm:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        # Compute attention scores: einsum over head_dim
        # q: [B, n_heads, T, head_dim], k: [B, n_heads, T, top_k, head_dim]
        # scores: [B, n_heads, T, top_k]
        scores = torch.einsum('bhtd,bhtkd->bhtk', q, k) * self.scale

        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax over knowledge slots dimension
        attn_weights = F.softmax(scores, dim=-1)  # [B, n_heads, T, top_k]
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of value vectors
        # attn_weights: [B, n_heads, T, top_k], v: [B, n_heads, T, top_k, head_dim]
        # output: [B, n_heads, T, head_dim]
        attn_output = torch.einsum('bhtk,bhtkd->bhtd', attn_weights, v)

        # Merge heads: [B, n_heads, T, head_dim] -> [B, T, n_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.n_heads * self.head_dim)

        # Output projection
        output = self.o_proj(attn_output)  # [B, T, hidden_size]
        output = self.post_norm(output)
        output = self.residual_dropout(output)

        # Gated fusion: hidden = hidden + sigmoid(gate) * cross_attn_output
        gate_value = torch.sigmoid(self.gate) if gate_override is None else gate_override
        fused = residual + gate_value * output

        return fused

    def get_gate_value(self) -> float:
        """Get current gate value.

        Returns:
            Current sigmoid(gate) value.
        """
        return torch.sigmoid(self.gate).item()


class YvMemoryFusionGate(nn.Module):
    """Adaptive fusion gate for knowledge injection scheduling.

    Complements YvMemoryCrossAttention by providing layer-wise
    adaptive gating that can vary injection strength based on
    layer depth and input complexity.

    Deeper layers get progressively more knowledge injection
    (following U-shaped sparsity allocation), while early layers
    focus on structural processing.

    Attributes:
        depth_scale (nn.Parameter): Per-layer depth scaling factor.
        base_gate (nn.Parameter): Base gate value shared across layers.
    """

    def __init__(
        self,
        n_layers: int,
        gate_init: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize fusion gate scheduler.

        Args:
            n_layers: Total number of transformer layers.
            gate_init: Initial gate value.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.n_layers = n_layers

        # Per-layer depth scaling (learned)
        self.depth_scale = nn.Parameter(
            torch.zeros(n_layers, device=device, dtype=dtype)
        )

        # Base gate value
        self.base_gate = nn.Parameter(
            torch.tensor(gate_init, device=device, dtype=dtype)
        )

    def forward(self, layer_idx: int) -> torch.Tensor:
        """Get gate value for a specific layer.

        Args:
            layer_idx: 0-based layer index.

        Returns:
            Gated value in [0, 1].
        """
        scale = torch.sigmoid(self.depth_scale[layer_idx])
        base = torch.sigmoid(self.base_gate)
        return base * scale