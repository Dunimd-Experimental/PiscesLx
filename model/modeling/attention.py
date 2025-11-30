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
Attention mechanism implementation for Ruchbah model.

This module defines the RuchbahAttention class, which provides a multi-head attention
mechanism with rotary position embeddings and optional optimizations including:
- Standard attention with optional fused QKV projections and sliding-window masking
- H2O attention backend for very long contexts or when explicitly enabled

The implementation supports KV caching for autoregressive decoding and integrates
with an external cache manager when provided.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .norms import _arctic_init_weights
# Use dms_core logging exclusively
import dms_core
PiscesLxCoreLog = dms_core.log.get_logger
from ..h2o_attention import RuchbahH2OAttention
from ..yarn_rope import RuchbahYaRNRotaryEmbedding

logger = PiscesLxCoreLog("Ruchbah.Core.Modeling.Attention", file_path="logs/RuchbahCore.log")

class RuchbahAttention(nn.Module):
    """
    Multi-head attention with optional H2O backend and sliding-window masking.

    Implements standard multi-head attention with rotary position embeddings (RoPE),
    grouped-query attention support, and optional H2O attention backend for long contexts.

    Attributes:
        cfg: Configuration object containing model hyperparameters and flags.
        n_head (int): Number of attention heads.
        n_kv_head (int): Number of key/value heads (for grouped-query attention).
        head_dim (int): Per-head hidden dimension.
        scale (float): Scaling factor for attention computation (1/sqrt(head_dim)).
        use_h2o (bool): Whether to use H2O attention backend.
    """

    def __init__(self, cfg, device=None, dtype=None):
        """
        Initialize the RuchbahAttention module.

        Args:
            cfg: Configuration object containing:
                - hidden_size (int): Model hidden dimension
                - n_head (int): Number of attention heads
                - n_kv_head (int): Number of key/value heads
                - max_position_embeddings (int): Maximum sequence length
                - rope_theta (float): Base frequency for RoPE
                - fused_qkv (bool, optional): Whether to use fused QKV projection
                - attention_dropout (float, optional): Dropout rate for attention
                - use_h2o_attention (bool, optional): Whether to use H2O backend
                - compression_ratio (int, optional): H2O compression ratio
                - streaming_window (int, optional): Sliding window size
                - sdpa_prefer_flash (bool, optional): Prefer Flash attention
                - use_sliding_window (bool, optional): Enable sliding window
            device (torch.device, optional): Device for parameter initialization.
            dtype (torch.dtype, optional): Data type for parameter initialization.
        """
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.hidden_size // cfg.n_head
        self.scale = self.head_dim ** -0.5

        # Enable H2O when configured or when max positions exceed threshold
        self.use_h2o = bool(getattr(cfg, 'use_h2o_attention', False)) or (cfg.max_position_embeddings > 1000000)

        if self.use_h2o:
            # Initialize H2O attention backend for long contexts
            self.h2o_attention = RuchbahH2OAttention(
                hidden_size=cfg.hidden_size,
                num_attention_heads=cfg.n_head,
                max_position_embeddings=cfg.max_position_embeddings,
                compression_ratio=getattr(cfg, 'compression_ratio', 8),
                streaming_window=getattr(cfg, 'streaming_window', 16384),
                dropout=getattr(cfg, 'attention_dropout', 0.0),
            )
        else:
            # Standard attention path with optional fused QKV projection
            self.fused_qkv = bool(getattr(cfg, 'fused_qkv', False))

            if self.fused_qkv:
                # Fused QKV projection: compute Q, K, V in a single linear layer
                qkv_out = (cfg.n_head + 2 * cfg.n_kv_head) * self.head_dim
                self.qkv_proj = nn.Linear(
                    cfg.hidden_size,
                    qkv_out,
                    bias=False,
                    device=device,
                    dtype=dtype
                )
            else:
                # Separate linear projections for Q, K, V
                self.q_proj = nn.Linear(
                    cfg.hidden_size,
                    cfg.n_head * self.head_dim,
                    bias=False,
                    device=device,
                    dtype=dtype
                )
                self.k_proj = nn.Linear(
                    cfg.hidden_size,
                    cfg.n_kv_head * self.head_dim,
                    bias=False,
                    device=device,
                    dtype=dtype
                )
                self.v_proj = nn.Linear(
                    cfg.hidden_size,
                    cfg.n_kv_head * self.head_dim,
                    bias=False,
                    device=device,
                    dtype=dtype
                )

            # Output projection and rotary position embedding (YaRN-scaled RoPE)
            self.o_proj = nn.Linear(
                cfg.n_head * self.head_dim,
                cfg.hidden_size,
                bias=False,
                device=device,
                dtype=dtype
            )
            self.rope = RuchbahYaRNRotaryEmbedding(
                self.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                scale=32,
                device=device,
            )
            self.attn_dropout = nn.Dropout(getattr(cfg, 'attention_dropout', 0.0))

        # Apply Ruchbah-specific parameter initialization
        self.apply(_arctic_init_weights)

    def forward(self, x, mask, past_key_values=None, use_cache=False, cache_manager=None):
        """
        Run the attention forward pass.

        Supports both H2O backend and standard attention path with KV caching
        for autoregressive decoding.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size].
            mask (torch.Tensor, optional): Attention mask broadcastable to
                [batch_size, n_head, seq_len_q, seq_len_k]. Ignored for H2O path.
            past_key_values (tuple, optional): Cached (key, value) tensors for
                extending the current sequence.
            use_cache (bool): Whether to return present key/value tensors for caching.
                Defaults to False.
            cache_manager: Optional external cache manager used by H2O backend.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If use_cache=False: attention output of shape [batch_size, seq_len, hidden_size]
                - If use_cache=True: tuple (output, (present_k, present_v)) where present_k
                  and present_v are cached key/value tensors

        Note:
            For grouped-query attention (n_kv_head != n_head), K/V heads are repeated
            to match Q head count. Sliding window is applied when cfg.use_sliding_window is True.
        """
        if self.use_h2o:
            # H2O backend: mask is not forwarded to avoid mismatches with streaming lengths
            attn_out, present_kv = self.h2o_attention(
                hidden_states=x,
                attention_mask=None,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_manager=cache_manager,
            )
            if use_cache:
                return attn_out, present_kv
            return attn_out

        # Standard attention path
        b, t, _ = x.shape

        # Project to queries, keys, and values (fused or separate)
        if getattr(self, 'fused_qkv', False):
            qkv = self.qkv_proj(x)
            q_end = self.n_head * self.head_dim
            kv_each = self.n_kv_head * self.head_dim
            # Split fused QKV output into Q, K, V
            q_lin = qkv[:, :, :q_end]
            k_lin = qkv[:, :, q_end:q_end + kv_each]
            v_lin = qkv[:, :, q_end + kv_each:]
            # Reshape to [batch, n_head, seq_len, head_dim]
            q = q_lin.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            k = k_lin.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = v_lin.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        else:
            # Separate projections for Q, K, V
            q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings to queries and keys
        q, k = self.rope(q, t), self.rope(k, t)

        # Concatenate with cached K/V for autoregressive decoding
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        seq_len = k.size(-2)

        # Grouped-query attention: repeat K/V heads to match Q head count
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Ensure dtype consistency for attention computation
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        # Merge batch and head dimensions for efficient attention computation
        q_ = q.reshape(b * self.n_head, t, self.head_dim)
        k_ = k.reshape(b * self.n_head, seq_len, self.head_dim)
        v_ = v.reshape(b * self.n_head, seq_len, self.head_dim)

        # Build causal mask (and optionally sliding window)
        base = seq_len - t
        row_pos = base + torch.arange(t, device=q.device)  # Positions of current query tokens
        key_pos = torch.arange(seq_len, device=q.device)  # Positions of all keys
        # Causal constraint: keys cannot attend to future positions
        allowed2d = key_pos.view(1, -1) <= row_pos.view(-1, 1)

        # Apply sliding window constraint when enabled
        if bool(getattr(self.cfg, 'use_sliding_window', False)):
            win = int(getattr(self.cfg, 'streaming_window', 16384))
            if win > 0 and win < seq_len:
                # Only allow attention within window size
                lower_bound = (row_pos.view(-1, 1) - (win - 1))
                local_allowed = key_pos.view(1, -1) >= lower_bound
                allowed2d = allowed2d & local_allowed

        disallow2d = ~allowed2d

        # Merge causal/sliding-window mask with provided attention mask
        if mask is not None:
            mask_slice = mask[:, :, -t:, :seq_len]
            if mask_slice.dtype == torch.bool:
                extra_disallow = ~mask_slice
            else:
                # Treat values < -1e4 as masked positions
                extra_disallow = mask_slice < -1e4
            disallow = disallow2d.view(1, 1, t, seq_len) | extra_disallow
            attn_mask = disallow.reshape(b * self.n_head, t, seq_len)
        else:
            attn_mask = disallow2d

        # Try to enable Flash SDP kernel when available and preferred
        try:
            from torch.backends.cuda import sdp_kernel as _sdp
            use_flash = torch.cuda.is_available() and bool(getattr(self.cfg, 'sdpa_prefer_flash', True))
        except Exception:
            use_flash = False

        # Compute scaled dot-product attention (Flash when possible, otherwise standard)
        if use_flash:
            with _sdp(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                out_ = F.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False,
                )
        else:
            out_ = F.scaled_dot_product_attention(
                q_,
                k_,
                v_,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False,
            )

        # Restore original dimensions: [batch, n_head, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        out = out_.reshape(b, self.n_head, t, self.head_dim).transpose(1, 2).contiguous().view(b, t, -1)
        out = self.attn_dropout(out)
        out = self.o_proj(out)

        # Return updated KV cache for autoregressive decoding if requested
        if use_cache:
            # For grouped-query attention, only cache original K/V heads
            k_cache = k[:, :self.n_kv_head] if self.n_kv_head != self.n_head else k
            v_cache = v[:, :self.n_kv_head] if self.n_kv_head != self.n_head else v
            return out, (k_cache, v_cache)

        return out
