#!/usr/bin/env/python3

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

logger = PiscesLxCoreLog("Arctic.Core.Modeling.Attention")

class ArcticAttention(nn.Module):
    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.hidden_size // cfg.n_head
        self.scale = self.head_dim ** -0.5
        self.use_h2o = bool(getattr(cfg, 'use_h2o_attention', False)) or (cfg.max_position_embeddings > 1000000)
        if self.use_h2o:
            self.h2o_attention = ArcticH2OAttention(
                hidden_size=cfg.hidden_size,
                num_attention_heads=cfg.n_head,
                max_position_embeddings=cfg.max_position_embeddings,
                compression_ratio=getattr(cfg, 'compression_ratio', 8),
                streaming_window=getattr(cfg, 'streaming_window', 16384),
                dropout=getattr(cfg, 'attention_dropout', 0.0)
            )
        else:
            self.fused_qkv = bool(getattr(cfg, 'fused_qkv', False))
            if self.fused_qkv:
                qkv_out = (cfg.n_head + 2 * cfg.n_kv_head) * self.head_dim
                self.qkv_proj = nn.Linear(cfg.hidden_size, qkv_out, bias=False, device=device, dtype=dtype)
            else:
                self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.head_dim, bias=False, device=device, dtype=dtype)
                self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
                self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
            self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.hidden_size, bias=False, device=device, dtype=dtype)
            self.rope = ArcticYaRNRotaryEmbedding(self.head_dim, cfg.max_position_embeddings, cfg.rope_theta, scale=32, device=device)
            self.attn_dropout = nn.Dropout(getattr(cfg, 'attention_dropout', 0.0))
        self.apply(_arctic_init_weights)

    def forward(self, x, mask, past_key_values=None, use_cache=False, cache_manager=None):
        if self.use_h2o:
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

        b, t, _ = x.shape
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

        q, k = self.rope(q, t), self.rope(k, t)

        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        seq_len = k.size(-2)
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        q_ = q.reshape(b * self.n_head, t, self.head_dim)
        k_ = k.reshape(b * self.n_head, seq_len, self.head_dim)
        v_ = v.reshape(b * self.n_head, seq_len, self.head_dim)

        base = seq_len - t
        row_pos = base + torch.arange(t, device=q.device)
        key_pos = torch.arange(seq_len, device=q.device)
        allowed2d = key_pos.view(1, -1) <= row_pos.view(-1, 1)

        if bool(getattr(self.cfg, 'use_sliding_window', False)):
            win = int(getattr(self.cfg, 'streaming_window', 16384))
            if win > 0 and win < seq_len:
                lower_bound = (row_pos.view(-1, 1) - (win - 1))
                local_allowed = key_pos.view(1, -1) >= lower_bound
                allowed2d = allowed2d & local_allowed
        disallow2d = ~allowed2d

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

        try:
            from torch.backends.cuda import sdp_kernel as _sdp
            use_flash = torch.cuda.is_available() and bool(getattr(self.cfg, 'sdpa_prefer_flash', True))
        except Exception:
            use_flash = False
        if use_flash:
            with _sdp(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                out_ = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=False)
        else:
            out_ = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=False)

        out = out_.reshape(b, self.n_head, t, self.head_dim).transpose(1, 2).contiguous().view(b, t, -1)
        out = self.attn_dropout(out)
        out = self.o_proj(out)

        if use_cache:
            k_cache = k[:, :self.n_kv_head] if self.n_kv_head != self.n_head else k
            v_cache = v[:, :self.n_kv_head] if self.n_kv_head != self.n_head else v
            return out, (k_cache, v_cache)
        return out