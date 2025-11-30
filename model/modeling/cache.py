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
Unified cache manager module for Ruchbah model.

This module provides a unified cache manager that handles multiple caching
mechanisms including KV cache for transformer layers, H2O sliding window cache,
multimodal generation cache, and speculative decoding cache.
"""

import math
import torch
import torch.nn.functional as F
from ..config import RuchbahConfig
from typing import Optional, Tuple, Dict, Any
# Use dms_core logging exclusively
import dms_core
PiscesLxCoreLog = dms_core.log.get_logger

logger = PiscesLxCoreLog("Ruchbah.Core.Modeling.Cache", file_path="logs/RuchbahCore.log")

class RuchbahUnifiedCacheManager:
    """
    Unified cache manager for transformer model inference.

    Manages multiple types of caches including KV cache for attention layers,
    H2O attention-based cache selection, generation cache for multimodal tasks,
    and speculative decoding cache. Supports cache quantization and eviction
    policies to manage memory usage.
    """

    def __init__(self, config: RuchbahConfig):
        """
        Initialize the unified cache manager.

        Args:
            config (RuchbahConfig or dict): Configuration object or dictionary
                containing cache settings. Supports both RuchbahConfig objects
                and plain dictionaries for flexibility.
        """
        self.config = config
        if isinstance(config, dict):
            self.max_cache_size = config.get('kv_cache_max_size', 8192)
            self.cache_quantization = config.get('quantization_enabled', True)
            self.cache_window_size = config.get('streaming_window', 2048)
            self.block_size = config.get('kv_cache_block_size', 512)
        else:
            self.max_cache_size = getattr(config, 'max_cache_size', 8192)
            self.cache_quantization = getattr(config, 'cache_quantization', True)
            self.cache_window_size = getattr(config, 'streaming_window', 2048)
            self.block_size = getattr(config, 'kv_cache_block_size', 512)

        # Initialize cache storage dictionaries
        self.kv_cache: Dict[int, Dict[str, Any]] = {}
        self.generation_cache: Dict[str, Any] = {}
        self.speculative_cache: Dict[str, Any] = {}
        self.h2o_cache: Dict[Any, Any] = {}

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def get_kv_cache(self, layer_idx: int, past_key_values: Optional[Tuple[torch.Tensor]] = None):
        """
        Retrieve KV cache for a given layer index.

        If cache entry does not exist and past_key_values are provided,
        initializes the cache with those values.

        Args:
            layer_idx (int): Index of the transformer layer.
            past_key_values (Optional[Tuple[torch.Tensor]]): Previously computed
                key and value tensors to initialize cache if entry is missing.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]: Concatenated key and value
                tensors from cache, or None if not found and no past_key_values provided.
        """
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

    def update_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        current_pos: int,
        use_h2o: bool = True
    ):
        """
        Update KV cache for a given layer with new key and value states.

        Applies H2O cache selection if enabled, splits new tokens into blocks,
        quantizes if configured, and manages cache eviction when size limits
        are exceeded.

        Args:
            layer_idx (int): Index of the transformer layer.
            key_states (torch.Tensor): New key states to cache.
            value_states (torch.Tensor): New value states to cache.
            current_pos (int): Current position in the sequence.
            use_h2o (bool): Whether to apply H2O attention-based cache selection.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]: Updated concatenated
                key and value tensors from cache.
        """
        # Apply H2O cache selection if enabled
        if use_h2o and hasattr(self.config, 'use_h2o_attention') and self.config.use_h2o_attention:
            key_states, value_states = self._apply_h2o_cache_selection(key_states, value_states, current_pos)

        # Get or create cache entry for this layer
        entry = self.kv_cache.get(layer_idx)
        if entry is None:
            entry = {'blocks': [], 'total_len': 0}
            self.kv_cache[layer_idx] = entry

        # Calculate new tokens to add
        new_total = key_states.shape[-2]
        delta = new_total - entry['total_len']
        if delta > 0:
            # Extract only the new tokens (tail of sequence)
            tail_k = key_states[:, :, -delta:, :]
            tail_v = value_states[:, :, -delta:, :]
            bs = self.block_size
            num_blocks = (delta + bs - 1) // bs

            # Split new tokens into blocks
            for i in range(num_blocks):
                s = i * bs
                e = min(delta, (i + 1) * bs)
                kb = tail_k[:, :, s:e, :]
                vb = tail_v[:, :, s:e, :]

                # Quantize block if enabled and size is sufficient
                if self.cache_quantization and kb.shape[2] >= min(bs, 256):
                    kb, vb = self._quantize_cache(kb, vb)

                entry['blocks'].append((kb, vb))
                entry['total_len'] += (e - s)

        # Evict blocks if cache exceeds soft cap (1.5x max size)
        soft_cap = int(self.max_cache_size * 1.5)
        while entry['total_len'] > soft_cap and entry['blocks']:
            kb, vb = entry['blocks'].pop(0)
            entry['total_len'] -= kb.shape[2]
            self.cache_evictions += 1

        # Compact blocks if cache exceeds max size
        if entry['total_len'] > self.max_cache_size and len(entry['blocks']) >= 1:
            self._compact_blocks(entry)

        # Final eviction pass if still over limit
        while entry['total_len'] > self.max_cache_size and entry['blocks']:
            kb, vb = entry['blocks'].pop(0)
            entry['total_len'] -= kb.shape[2]
            self.cache_evictions += 1

        return self._concat_recent(layer_idx)

    def _concat_recent(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Concatenate all cached blocks for a given layer.

        Args:
            layer_idx (int): Index of the transformer layer.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor]]: Concatenated key and value
                tensors, or None if no blocks exist for this layer.
        """
        entry = self.kv_cache.get(layer_idx, None)
        if entry is None or not entry['blocks']:
            return None
        ks = [b[0] for b in entry['blocks']]
        vs = [b[1] for b in entry['blocks']]
        k = torch.cat(ks, dim=2)
        v = torch.cat(vs, dim=2)
        return (k, v)

    def _compact_blocks(self, entry: Dict[str, Any]):
        """
        Compact cached blocks by selecting important tokens.

        Reduces cache size by keeping recent tokens and selecting important
        tokens from older blocks based on tensor norm importance scores.

        Args:
            entry (Dict[str, Any]): Cache entry containing blocks to compact.
        """
        try:
            if not entry['blocks']:
                return

            target_total = int(self.max_cache_size * 1.1)
            idx = 0

            while entry['total_len'] > target_total and idx < len(entry['blocks']):
                kb, vb = entry['blocks'][idx]
                B, H, T, D = kb.shape

                # Skip small blocks
                if T < 64:
                    idx += 1
                    continue

                # Keep recent tokens (last 25% or at least 32 tokens)
                keep_recent = max(32, T // 4)
                tail_k = kb[:, :, -keep_recent:, :]
                tail_v = vb[:, :, -keep_recent:, :]
                head_len = T - keep_recent

                if head_len <= 0:
                    idx += 1
                    continue

                # Calculate importance scores for older tokens
                head_k = kb[:, :, :head_len, :]
                head_v = vb[:, :, :head_len, :]
                imp_k = torch.norm(head_k, dim=-1)
                imp_v = torch.norm(head_v, dim=-1)
                imp = (imp_k + imp_v).mean(dim=1)

                # Select top-k important tokens from head
                topk = max(keep_recent, head_len // 2)
                topk = min(topk, head_len)
                _, idx_sel = torch.topk(imp, k=topk, dim=-1)
                idx_sel, _ = torch.sort(idx_sel, dim=-1)

                # Gather selected tokens
                idx_sel_exp = idx_sel.unsqueeze(1).unsqueeze(-1).expand(B, H, topk, D)
                head_k_sel = torch.gather(head_k, 2, idx_sel_exp)
                head_v_sel = torch.gather(head_v, 2, idx_sel_exp)

                # Combine selected head tokens with tail tokens
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
        """
        Retrieve generation cache for a specific modality.

        Args:
            modality (str): Modality type (e.g., 'text', 'image').

        Returns:
            Any: Cached data for the modality, or None if not found.
        """
        return self.generation_cache.get(modality, None)

    def set_generation_cache(self, modality: str, cache_data: torch.Tensor):
        """
        Set generation cache for a specific modality.

        Args:
            modality (str): Modality type (e.g., 'text', 'image').
            cache_data (torch.Tensor): Data to cache for the modality.
        """
        self.generation_cache[modality] = cache_data

    def get_speculative_cache(self, draft_length: int):
        """
        Retrieve speculative decoding cache for a given draft length.

        Args:
            draft_length (int): Length of the draft sequence.

        Returns:
            Any: Cached data for the draft, or None if not found.
        """
        cache_key = f"draft_{draft_length}"
        return self.speculative_cache.get(cache_key, None)

    def set_speculative_cache(self, draft_length: int, cache_data: Dict):
        """
        Set speculative decoding cache for a given draft length.

        Args:
            draft_length (int): Length of the draft sequence.
            cache_data (Dict): Data to cache for the draft sequence.
        """
        cache_key = f"draft_{draft_length}"
        self.speculative_cache[cache_key] = cache_data

    def get_h2o_cache(self, key_states: torch.Tensor, current_pos: int, max_cache_size: int):
        """
        Retrieve H2O attention-based cache entry.

        Args:
            key_states (torch.Tensor): Key states used for cache key generation.
            current_pos (int): Current position in the sequence.
            max_cache_size (int): Maximum size of the cache window.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]: Selected key
                and value tensors, or (None, None) if not found.
        """
        cache_key = (current_pos // max(1, max_cache_size))
        return self.h2o_cache.get(cache_key, (None, None))

    def set_h2o_cache(
        self,
        key_states: torch.Tensor,
        current_pos: int,
        max_cache_size: int,
        selected_keys: torch.Tensor,
        selected_values: torch.Tensor
    ):
        """
        Set H2O attention-based cache entry.

        Args:
            key_states (torch.Tensor): Key states used for cache key generation.
            current_pos (int): Current position in the sequence.
            max_cache_size (int): Maximum size of the cache window.
            selected_keys (torch.Tensor): Selected key tensors to cache.
            selected_values (torch.Tensor): Selected value tensors to cache.
        """
        cache_key = (current_pos // max(1, max_cache_size))
        self.h2o_cache[cache_key] = (selected_keys, selected_values)

    def _apply_h2o_cache_selection(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        current_pos: int
    ):
        """
        Apply H2O attention-based cache selection to reduce cached tokens.

        Selects important tokens from the cache window based on importance scores
        calculated from attention weights and position decay.

        Args:
            key_states (torch.Tensor): Key states to select from.
            value_states (torch.Tensor): Value states to select from.
            current_pos (int): Current position in the sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Selected key and value tensors.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        # Return original if sequence is within cache window
        if seq_len <= self.cache_window_size:
            return key_states, value_states

        # Calculate importance scores for token selection
        importance_scores = self._calculate_importance_scores(key_states, value_states)
        cache_start = max(0, current_pos - self.cache_window_size)
        cache_end = current_pos

        # Select top-k important tokens from cache window
        if cache_end - cache_start > self.cache_window_size:
            cache_importance = importance_scores[:, :, cache_start:cache_end]
            _, top_indices = torch.topk(cache_importance, self.cache_window_size, dim=-1)
            selected_keys = torch.gather(
                key_states,
                2,
                top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            )
            selected_values = torch.gather(
                value_states,
                2,
                top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            )
            return selected_keys, selected_values

        return key_states, value_states

    def _calculate_importance_scores(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate importance scores for key-value pairs.

        Computes scores based on attention weights (diagonal of key-value
        attention matrix) and position decay (exponential decay with distance).

        Args:
            key_states (torch.Tensor): Key states to calculate importance for.
            value_states (torch.Tensor): Value states to calculate importance for.

        Returns:
            torch.Tensor: Importance scores for each token position, shape
                [batch_size, num_heads, seq_len].
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        # Compute attention scores: K @ V^T / sqrt(d)
        attention_scores = torch.matmul(key_states, value_states.transpose(-2, -1)) / math.sqrt(head_dim)

        # Extract diagonal (self-attention importance)
        importance = attention_scores.diagonal(dim1=-2, dim2=-1)

        # Apply position decay: more recent positions have higher weight
        position_weights = torch.exp(-torch.arange(seq_len, device=key_states.device).float() / 100.0)
        position_weights = position_weights.unsqueeze(0).unsqueeze(0)
        importance = importance * position_weights

        return F.softmax(importance, dim=-1)

    def _quantize_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize key and value states to reduce memory usage.

        Applies FP8-like quantization if CUDA is available, otherwise uses
        standard quantization (4-bit or 8-bit based on sequence length).

        Args:
            key_states (torch.Tensor): Key states to quantize.
            value_states (torch.Tensor): Value states to quantize.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Quantized key and value tensors.
        """
        seq_len = key_states.shape[2]

        # Return original if sequence is within cache window
        if seq_len <= self.cache_window_size:
            return key_states, value_states

        # Try FP8-like quantization if CUDA is available
        try:
            use_fp8_like = torch.cuda.is_available()
        except Exception:
            use_fp8_like = False

        if use_fp8_like:
            def fake_fp8_quant(t: torch.Tensor) -> torch.Tensor:
                """
                Apply FP8-like quantization to a tensor.

                Args:
                    t (torch.Tensor): Tensor to quantize.

                Returns:
                    torch.Tensor: Quantized tensor.
                """
                b, h, tlen, d = t.shape
                t_ = t.reshape(b * h, tlen, d)
                scale = t_.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
                q = torch.clamp(torch.round((t_ / scale) * 240.0), -240, 240)
                deq = (q / 240.0) * scale
                return deq.reshape(b, h, tlen, d).to(t.dtype)

            key_states = fake_fp8_quant(key_states)
            value_states = fake_fp8_quant(value_states)
            return key_states, value_states

        # Use standard quantization: 4-bit for long sequences, 8-bit otherwise
        quant_bits = 4 if seq_len > (self.cache_window_size * 2) else 8
        key_states = self._quantize_tensor(key_states, quant_bits)
        value_states = self._quantize_tensor(value_states, quant_bits)
        return key_states, value_states

    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Quantize a tensor to a specified bit width.

        Uses symmetric quantization with scale factor based on maximum absolute value.

        Args:
            tensor (torch.Tensor): Tensor to quantize.
            bits (int): Number of bits to quantize to (e.g., 4, 8). Values >= 16
                return the original tensor unchanged.

        Returns:
            torch.Tensor: Quantized tensor with same dtype as input.
        """
        if bits >= 16:
            return tensor

        max_val = tensor.abs().max()
        scale = (max_val / (2**(bits - 1) - 1)).clamp(min=1e-8)
        q = torch.clamp(
            torch.round(tensor / scale),
            min=-(2**(bits - 1)),
            max=(2**(bits - 1) - 1)
        )
        return (q * scale).to(tensor.dtype)

    def clear_cache(self):
        """
        Clear all cached data and reset statistics.

        Removes all entries from all cache dictionaries and resets hit/miss
        counters to zero.
        """
        self.kv_cache.clear()
        self.generation_cache.clear()
        self.speculative_cache.clear()
        self.h2o_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Return statistics about cache usage and performance.

        Returns:
            Dict[str, Any]: Dictionary containing cache statistics including:
                - cache_hits: Number of cache hits
                - cache_misses: Number of cache misses
                - hit_rate: Cache hit rate (hits / total requests)
                - cache_evictions: Number of cache evictions
                - kv_cache_layers: Number of layers with KV cache
                - kv_cache_lengths: Dictionary mapping layer indices to cache lengths
                - kv_cache_total_tokens: Total number of cached tokens across all layers
                - generation_cache_size: Number of entries in generation cache
                - speculative_cache_size: Number of entries in speculative cache
                - h2o_cache_entries: Number of entries in H2O cache
                - cache_window_size: Configured cache window size
                - quantization_enabled: Whether quantization is enabled
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        # Calculate KV cache statistics
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
            'quantization_enabled': bool(self.cache_quantization)
        }
