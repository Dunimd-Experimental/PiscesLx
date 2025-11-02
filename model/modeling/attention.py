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

import torch
from torch import nn
import torch.nn.functional as F
from .norms import _arctic_init_weights
from utils.log.core import PiscesLxCoreLog
from ..h2o_attention import ArcticH2OAttention
from ..yarn_rope import ArcticYaRNRotaryEmbedding

logger = PiscesLxCoreLog("Arctic.Core.Modeling.Attention", file_path="logs/ArcticCore.log")

class ArcticAttention(nn.Module):
    """Multi-head attention module with optional H2O optimization and sliding window support.

    This class implements multi-head attention with RoPE positional embeddings,
    supporting both standard and optimized(H2O) variants. It handles KV caching
    for autoregressive decoding and applies proper masking strategies.

    Attributes:
        cfg: Configuration object containing model hyperparameters.
        n_head: Number of attention heads.
        n_kv_head: Number of key/value heads (for grouped-query attention).
        head_dim: Dimension of each attention head.
        scale: Scaling factor for attention scores.
        use_h2o: Flag indicating whether to use H2O attention optimization.
    """

    def __init__(self, cfg, device=None, dtype=None):
        """Initialize the ArcticAttention module.

        Args:
            cfg: Configuration object with model parameters.
            device: Device to place the module on.
            dtype: Data type for the module parameters.
        """
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.hidden_size // cfg.n_head
        self.scale = self.head_dim ** -0.5
        self.use_h2o = bool(getattr(cfg, 'use_h2o_attention', False)) or (cfg.max_position_embeddings > 1000000)
        
        if self.use_h2o:
            # Initialize H2O attention module for long-sequence optimization
            self.h2o_attention = ArcticH2OAttention(
                hidden_size=cfg.hidden_size,
                num_attention_heads=cfg.n_head,
                max_position_embeddings=cfg.max_position_embeddings,
                compression_ratio=getattr(cfg, 'compression_ratio', 8),
                streaming_window=getattr(cfg, 'streaming_window', 16384),
                dropout=getattr(cfg, 'attention_dropout', 0.0)
            )
        else:
            # Standard attention implementation
            self.fused_qkv = bool(getattr(cfg, 'fused_qkv', False))
            if self.fused_qkv:
                # Use fused QKV projection for efficiency
                qkv_out = (cfg.n_head + 2 * cfg.n_kv_head) * self.head_dim
                self.qkv_proj = nn.Linear(cfg.hidden_size, qkv_out, bias=False, device=device, dtype=dtype)
            else:
                # Separate projections for query, key, and value
                self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.head_dim, bias=False, device=device, dtype=dtype)
                self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
                self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
            
            # Output projection and positional encoding
            self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.hidden_size, bias=False, device=device, dtype=dtype)
            self.rope = ArcticYaRNRotaryEmbedding(self.head_dim, cfg.max_position_embeddings, cfg.rope_theta, scale=32, device=device)
            self.attn_dropout = nn.Dropout(getattr(cfg, 'attention_dropout', 0.0))
        
        # Initialize weights using Arctic-specific initialization
        self.apply(_arctic_init_weights)

    def forward(self, x, mask, past_key_values=None, use_cache=False, cache_manager=None):
        """Forward pass through the attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, hidden_size).
            mask: Attention mask tensor.
            past_key_values: Cached key/value tensors from previous forward passes.
            use_cache: Whether to return updated key/value cache.
            cache_manager: Optional cache manager for handling KV cache.

        Returns:
            Output tensor of shape (batch_size, sequence_length, hidden_size), 
            optionally with updated key/value cache when use_cache is True.
        """
        if self.use_h2o:
            # Delegate to H2O attention implementation
            attn_out, present_kv = self.h2o_attention(
                hidden_states=x,
                attention_mask=mask,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_manager=cache_manager
            )
            if use_cache:
                return attn_out, present_kv
            return attn_out

        # Standard attention computation
        b, t, _ = x.shape
        
        # Project input to query, key, and value
        if getattr(self, 'fused_qkv', False):
            qkv = self.qkv_proj(x)
            q_end = self.n_head * self.head_dim
            kv_each = self.n_kv_head * self.head_dim
            q_lin = qkv[:, :, :q_end]
            k_lin = qkv[:, :, q_end:q_end + kv_each]
            v_lin = qkv[:, :, q_end + kv_each:]
            q = q_lin.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            k = k_lin.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = v_lin.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply rotary positional embeddings
        q, k = self.rope(q, t), self.rope(k, t)

        # Concatenate with cached keys/values if available
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        seq_len = k.size(-2)
        
        # Handle grouped-query attention by repeating KV heads
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        
        # Ensure value tensor has same dtype as query
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        # Reshape tensors for attention computation
        q_ = q.reshape(b * self.n_head, t, self.head_dim)
        k_ = k.reshape(b * self.n_head, seq_len, self.head_dim)
        v_ = v.reshape(b * self.n_head, seq_len, self.head_dim)

        # Create causal and sliding window masks
        base = seq_len - t
        row_pos = base + torch.arange(t, device=q.device)
        key_pos = torch.arange(seq_len, device=q.device)
        allowed2d = key_pos.view(1, -1) <= row_pos.view(-1, 1)

        # Apply sliding window constraint if enabled
        if bool(getattr(self.cfg, 'use_sliding_window', False)):
            win = int(getattr(self.cfg, 'streaming_window', 16384))
            if win > 0 and win < seq_len:
                lower_bound = (row_pos.view(-1, 1) - (win - 1))
                local_allowed = key_pos.view(1, -1) >= lower_bound
                allowed2d = allowed2d & local_allowed
        disallow2d = ~allowed2d

        # Combine causal/sliding window mask with provided attention mask
        if mask is not None:
            mask_slice = mask[:, :, -t:, :seq_len]
            if mask_slice.dtype == torch.bool:
                extra_disallow = ~mask_slice
            else:
                extra_disallow = mask_slice < -1e4
            disallow = disallow2d.view(1, 1, t, seq_len) | extra_disallow
            attn_mask = disallow.reshape(b * self.n_head, t, seq_len)
        else:
            attn_mask = disallow2d

        # Attempt to use Flash Attention if available
        try:
            from torch.backends.cuda import sdp_kernel as _sdp
            use_flash = torch.cuda.is_available() and bool(getattr(self.cfg, 'sdpa_prefer_flash', True))
        except Exception:
            use_flash = False
            
        # Compute scaled dot-product attention
        if use_flash:
            with _sdp(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                out_ = F.scaled_dot_product_attention(
                    q_, k_, v_, 
                    attn_mask=attn_mask, 
                    dropout_p=self.attn_dropout.p if self.training else 0.0, 
                    is_causal=False
                )
        else:
            out_ = F.scaled_dot_product_attention(
                q_, k_, v_, 
                attn_mask=attn_mask, 
                dropout_p=self.attn_dropout.p if self.training else 0.0, 
                is_causal=False
            )

        # Reshape output and apply final projection
        out = out_.reshape(b, self.n_head, t, self.head_dim).transpose(1, 2).contiguous().view(b, t, -1)
        out = self.attn_dropout(out)
        out = self.o_proj(out)

        # Return output with updated cache if requested
        if use_cache:
            k_cache = k[:, :self.n_kv_head] if self.n_kv_head != self.n_head else k
            v_cache = v[:, :self.n_kv_head] if self.n_kv_head != self.n_head else v
            return out, (k_cache, v_cache)
        return out
