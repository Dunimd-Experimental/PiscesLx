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

import math
import torch
from torch import nn
import torch.nn.functional as F
from ..config import ArcticConfig
from utils.log.core import PiscesLxCoreLog
from typing import Optional, Tuple, Dict, Any
from ..h2o_attention import ArcticH2OAttention

logger = PiscesLxCoreLog("Arctic.Core.Modeling.Cache")

class ArcticUnifiedCacheManager:
    """
    Unified cache manager that consolidates all caching mechanisms:
    - KV-Cache for transformer layers
    - H2O sliding window cache
    - Multimodal generation cache
    - Speculative decoding cache
    """
    def __init__(self, config: ArcticConfig):
        """Unified cache manager.
        Accepts a PiscesConfig or a plain dict of cache options for flexibility.
        """
        self.config = config
        if isinstance(config, dict):
            self.max_cache_size = config.get('kv_cache_max_size', 8192)
            self.cache_quantization = config.get('quantization_enabled', True)
            self.cache_window_size = config.get('streaming_window', 2048)
            self.block_size = config.get('kv_cache_block_size', 512)
            self.paging_enabled = bool(config.get('kv_paged_enabled', False))
            self.soft_cap_factor = float(config.get('kv_soft_cap_factor', 1.5))
        else:
            self.max_cache_size = getattr(config, 'max_cache_size', 8192)
            self.cache_quantization = getattr(config, 'cache_quantization', True)
            self.cache_window_size = getattr(config, 'streaming_window', 2048)
            self.block_size = getattr(config, 'kv_cache_block_size', 512)
            self.paging_enabled = bool(getattr(config, 'kv_paged_enabled', False))
            self.soft_cap_factor = float(getattr(config, 'kv_soft_cap_factor', 1.5))

        self.kv_cache: Dict[int, Dict[str, Any]] = {}
        self.generation_cache: Dict[str, Any] = {}
        self.speculative_cache: Dict[str, Any] = {}
        self.h2o_cache: Dict[Any, Any] = {}

        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def get_kv_cache(self, layer_idx: int, past_key_values: Optional[Tuple[torch.Tensor]] = None):
        entry = self.kv_cache.get(layer_idx, None)
        if entry is None:
            if past_key_values is not None:
                k, v = past_key_values
                self.kv_cache[layer_idx] = {'blocks': [(k, v)], 'total_len': k.shape[-2]}
                self.cache_misses += 1
                return past_key_values
            self.cache_misses += 1
            return None
        self.cache_hits += 1
        return self._concat_recent(layer_idx)

    def update_kv_cache(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, 
                        current_pos: int, use_h2o: bool = True):
        if use_h2o and hasattr(self.config, 'use_h2o_attention') and self.config.use_h2o_attention:
            key_states, value_states = self._apply_h2o_cache_selection(key_states, value_states, current_pos)

        entry = self.kv_cache.get(layer_idx)
        if entry is None:
            entry = {'blocks': [], 'total_len': 0}
            self.kv_cache[layer_idx] = entry

        new_total = key_states.shape[-2]
        delta = new_total - entry['total_len']
        if delta > 0:
            tail_k = key_states[:, :, -delta:, :]
            tail_v = value_states[:, :, -delta:, :]
            bs = self.block_size
            num_blocks = (delta + bs - 1) // bs
            for i in range(num_blocks):
                s = i * bs
                e = min(delta, (i + 1) * bs)
                kb = tail_k[:, :, s:e, :]
                vb = tail_v[:, :, s:e, :]
                if self.cache_quantization and kb.shape[2] >= min(bs, 256):
                    kb, vb = self._quantize_cache(kb, vb)
                entry['blocks'].append((kb, vb))
                entry['total_len'] += (e - s)

        soft_cap = int(self.max_cache_size * (self.soft_cap_factor if hasattr(self, 'soft_cap_factor') else 1.5))
        while entry['total_len'] > soft_cap and entry['blocks']:
            kb, vb = entry['blocks'].pop(0)
            entry['total_len'] -= kb.shape[2]
            self.cache_evictions += 1

        if entry['total_len'] > self.max_cache_size and len(entry['blocks']) >= 1:
            self._compact_blocks(entry)

        while entry['total_len'] > self.max_cache_size and entry['blocks']:
            kb, vb = entry['blocks'].pop(0)
            entry['total_len'] -= kb.shape[2]
            self.cache_evictions += 1

        return self._concat_recent(layer_idx)

    def _concat_recent(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        entry = self.kv_cache.get(layer_idx, None)
        if entry is None or not entry['blocks']:
            return None
        ks = [b[0] for b in entry['blocks']]
        vs = [b[1] for b in entry['blocks']]
        k = torch.cat(ks, dim=2)
        v = torch.cat(vs, dim=2)
        return (k, v)

    def _compact_blocks(self, entry: Dict[str, Any]):
        try:
            if not entry['blocks']:
                return
            target_total = int(self.max_cache_size * 1.1)
            idx = 0
            while entry['total_len'] > target_total and idx < len(entry['blocks']):
                kb, vb = entry['blocks'][idx]
                B, H, T, D = kb.shape
                if T < 64:
                    idx += 1
                    continue
                keep_recent = max(32, T // 4)
                tail_k = kb[:, :, -keep_recent:, :]
                tail_v = vb[:, :, -keep_recent:, :]
                head_len = T - keep_recent
                if head_len <= 0:
                    idx += 1
                    continue
                head_k = kb[:, :, :head_len, :]
                head_v = vb[:, :, :head_len, :]
                imp_k = torch.norm(head_k, dim=-1)
                imp_v = torch.norm(head_v, dim=-1)
                imp = (imp_k + imp_v).mean(dim=1)
                topk = max(keep_recent, head_len // 2)
                topk = min(topk, head_len)
                _, idx_sel = torch.topk(imp, k=topk, dim=-1)
                idx_sel, _ = torch.sort(idx_sel, dim=-1)
                idx_sel_exp = idx_sel.unsqueeze(1).unsqueeze(-1).expand(B, H, topk, D)
                head_k_sel = torch.gather(head_k, 2, idx_sel_exp)
                head_v_sel = torch.gather(head_v, 2, idx_sel_exp)
                new_k = torch.cat([head_k_sel, tail_k], dim=2)
                new_v = torch.cat([head_v_sel, tail_v], dim=2)
                delta = T - new_k.shape[2]
                if delta > 0:
                    entry['blocks'][idx] = (new_k, new_v)
                    entry['total_len'] -= delta
                idx += 1
        except Exception:
            pass

    def get_generation_cache(self, modality: str):
        return self.generation_cache.get(modality, None)

    def set_generation_cache(self, modality: str, cache_data: torch.Tensor):
        self.generation_cache[modality] = cache_data

    def get_speculative_cache(self, draft_length: int):
        cache_key = f"draft_{draft_length}"
        return self.speculative_cache.get(cache_key, None)

    def set_speculative_cache(self, draft_length: int, cache_data: Dict):
        cache_key = f"draft_{draft_length}"
        self.speculative_cache[cache_key] = cache_data

    def get_h2o_cache(self, key_states: torch.Tensor, current_pos: int, max_cache_size: int):
        cache_key = (current_pos // max(1, max_cache_size))
        return self.h2o_cache.get(cache_key, (None, None))

    def set_h2o_cache(self, key_states: torch.Tensor, current_pos: int, max_cache_size: int,
                      selected_keys: torch.Tensor, selected_values: torch.Tensor):
        cache_key = (current_pos // max(1, max_cache_size))
        self.h2o_cache[cache_key] = (selected_keys, selected_values)

    def _apply_h2o_cache_selection(self, key_states: torch.Tensor, value_states: torch.Tensor, current_pos: int):
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        if seq_len <= self.cache_window_size:
            return key_states, value_states
        importance_scores = self._calculate_importance_scores(key_states, value_states)
        cache_start = max(0, current_pos - self.cache_window_size)
        cache_end = current_pos
        if cache_end - cache_start > self.cache_window_size:
            cache_importance = importance_scores[:, :, cache_start:cache_end]
            _, top_indices = torch.topk(cache_importance, self.cache_window_size, dim=-1)
            selected_keys = torch.gather(key_states, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            selected_values = torch.gather(value_states, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            return selected_keys, selected_values
        return key_states, value_states

    def _calculate_importance_scores(self, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        attention_scores = torch.matmul(key_states, value_states.transpose(-2, -1)) / math.sqrt(head_dim)
        importance = attention_scores.diagonal(dim1=-2, dim2=-1)
        position_weights = torch.exp(-torch.arange(seq_len, device=key_states.device).float() / 100.0)
        position_weights = position_weights.unsqueeze(0).unsqueeze(0)
        importance = importance * position_weights
        return F.softmax(importance, dim=-1)

    def _quantize_cache(self, key_states: torch.Tensor, value_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = key_states.shape[2]
        if seq_len <= self.cache_window_size:
            return key_states, value_states
        try:
            use_fp8_like = torch.cuda.is_available()
        except Exception:
            use_fp8_like = False
        if use_fp8_like:
            def fake_fp8_quant(t: torch.Tensor) -> torch.Tensor:
                b, h, tlen, d = t.shape
                t_ = t.reshape(b * h, tlen, d)
                scale = t_.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
                q = torch.clamp(torch.round((t_ / scale) * 240.0), -240, 240)
                deq = (q / 240.0) * scale
                return deq.reshape(b, h, tlen, d).to(t.dtype)
            key_states = fake_fp8_quant(key_states)
            value_states = fake_fp8_quant(value_states)
            return key_states, value_states
        quant_bits = 4 if seq_len > (self.cache_window_size * 2) else 8
        key_states = self._quantize_tensor(key_states, quant_bits)
        value_states = self._quantize_tensor(value_states, quant_bits)
        return key_states, value_states

    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        if bits >= 16:
            return tensor
        max_val = tensor.abs().max()
        scale = (max_val / (2**(bits - 1) - 1)).clamp(min=1e-8)
        q = torch.clamp(torch.round(tensor / scale), min=-(2**(bits - 1)), max=(2**(bits - 1) - 1))
        return (q * scale).to(tensor.dtype)

    def clear_cache(self):
        self.kv_cache.clear()
        self.generation_cache.clear()
        self.speculative_cache.clear()
        self.h2o_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        kv_lengths = {}
        total_kv_tokens = 0
        for layer_idx, entry in self.kv_cache.items():
            try:
                if entry is None or 'total_len' not in entry:
                    kv_lengths[layer_idx] = 0
                    continue
                kv_lengths[layer_idx] = int(entry['total_len'])
                total_kv_tokens += int(entry['total_len'])
            except Exception:
                kv_lengths[layer_idx] = -1
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_evictions': self.cache_evictions,
            'kv_cache_layers': len(self.kv_cache),
            'kv_cache_lengths': kv_lengths,
            'kv_cache_total_tokens': total_kv_tokens,
            'generation_cache_size': len(self.generation_cache),
            'speculative_cache_size': len(self.speculative_cache),
            'h2o_cache_entries': len(self.h2o_cache),
            'cache_window_size': self.cache_window_size,
            'quantization_enabled': bool(self.cache_quantization),
            'paging_enabled': bool(getattr(self, 'paging_enabled', False)),
            'kv_page_size': int(getattr(self, 'block_size', 512)),
            'soft_cap_factor': float(getattr(self, 'soft_cap_factor', 1.5))
        }