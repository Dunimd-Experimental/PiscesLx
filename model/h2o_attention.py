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

"""Heavy-Hitter Oracle (H2O) attention blocks for ultra-long context processing."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RuchbahDynamicH2OAttention(nn.Module):
    """
    Dynamic H2O Attention with adaptive compression and hierarchical caching.
    
    Enhances the base H2O attention with:
    - Dynamic compression ratio based on sequence complexity
    - Hierarchical cache levels (recent, compressed, archived)
    - PagedAttention integration support
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int = 10485760,
        compression_ratio: int = 8,
        heavy_hitter_ratio: float = 0.1,
        streaming_window: int = 16384,
        dropout: float = 0.1,
        num_cache_levels: int = 3,
        enable_paged_attention: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.compression_ratio = compression_ratio
        self.heavy_hitter_ratio = heavy_hitter_ratio
        self.streaming_window = streaming_window
        self.num_cache_levels = num_cache_levels
        self.enable_paged_attention = enable_paged_attention
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.complexity_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.dynamic_compressor = nn.ModuleDict({
            'recent': nn.Identity(),
            'compressed': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            ),
            'archived': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, hidden_size)
            )
        })
        
        self.level_attention = nn.ModuleDict({
            'recent': nn.MultiheadAttention(hidden_size, num_attention_heads // 3, batch_first=True),
            'compressed': nn.MultiheadAttention(hidden_size, num_attention_heads // 3, batch_first=True),
            'archived': nn.MultiheadAttention(hidden_size, num_attention_heads // 3, batch_first=True)
        })
        
        self.level_fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_cache_levels),
            nn.Softmax(dim=-1)
        )
        
        self.register_buffer('cache_sizes', torch.tensor([streaming_window // 2, streaming_window * 2, max_position_embeddings // 16]))
        
    def _predict_sequence_complexity(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        pooled = hidden_states.mean(dim=1)
        complexity = self.complexity_predictor(pooled)
        return complexity.squeeze(-1)
    
    def _compute_dynamic_compression(self, hidden_states: torch.Tensor, complexity: torch.Tensor) -> int:
        base_ratio = self.compression_ratio
        adaptive_ratio = int(base_ratio * (1 + complexity.item()))
        adaptive_ratio = max(2, min(16, adaptive_ratio))
        return adaptive_ratio
    
    def _build_hierarchical_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        importance_scores: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        sorted_indices = torch.argsort(importance_scores, dim=-1, descending=True)
        
        recent_size = min(int(self.cache_sizes[0].item()), seq_len)
        compressed_size = min(int(self.cache_sizes[1].item()), seq_len)
        archived_size = min(int(self.cache_sizes[2].item()), seq_len)
        
        recent_indices = sorted_indices[:, :, :recent_size]
        compressed_indices = sorted_indices[:, :, recent_size:recent_size + compressed_size]
        archived_indices = sorted_indices[:, :, recent_size + compressed_size:recent_size + compressed_size + archived_size]
        
        recent_keys = torch.gather(key_states, 2, recent_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        recent_values = torch.gather(value_states, 2, recent_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        
        compressed_keys = torch.gather(key_states, 2, compressed_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        compressed_values = torch.gather(value_states, 2, compressed_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        
        if archived_indices.shape[-1] > 0:
            archived_keys = torch.gather(key_states, 2, archived_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            archived_values = torch.gather(value_states, 2, archived_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
        else:
            archived_keys = torch.zeros(batch_size, num_heads, 0, head_dim, device=key_states.device)
            archived_values = torch.zeros(batch_size, num_heads, 0, head_dim, device=value_states.device)
        
        return {
            'recent': (recent_keys, recent_values),
            'compressed': (compressed_keys, compressed_values),
            'archived': (archived_keys, archived_values)
        }
    
    def _fuse_hierarchical_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        query_states: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        query_pooled = query_states.mean(dim=2, keepdim=True)
        level_weights = self.level_fusion(query_pooled.transpose(1, 2).transpose(2, 3))
        level_weights = level_weights.transpose(1, 3).squeeze(0)
        
        fused_output = torch.zeros_like(outputs['recent'])
        for i, level_name in enumerate(['recent', 'compressed', 'archived']):
            weight = level_weights[:, :, i:i+1, :].unsqueeze(-1)
            fused_output = fused_output + weight * outputs[level_name]
        
        return fused_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_manager=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
        
        complexity = self._predict_sequence_complexity(hidden_states)
        dynamic_compression = self._compute_dynamic_compression(hidden_states, complexity)
        
        if seq_len > self.streaming_window:
            compressed_key = self._compress_states(key_states, dynamic_compression)
            compressed_value = self._compress_states(value_states, dynamic_compression)
            attention_key = compressed_key
            attention_value = compressed_value
        else:
            attention_key = key_states
            attention_value = value_states
        
        attention_output = self._streaming_attention(
            query_states, attention_key, attention_value,
            attention_mask, cache_manager
        )
        
        attention_output = self.o_proj(attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size))
        attention_output = self.dropout(attention_output)
        
        return attention_output, (key_states, value_states)
    
    def _compress_states(
        self,
        states: torch.Tensor,
        compression_ratio: int = None
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = states.shape
        ratio = compression_ratio or self.compression_ratio
        
        if seq_len <= self.streaming_window:
            return states
        
        compressed_length = (seq_len + ratio - 1) // ratio
        flat = states.view(batch_size * num_heads, seq_len, head_dim)
        token_importance = torch.norm(flat, dim=-1)
        token_importance = F.softmax(token_importance, dim=-1)
        
        pad_len = compressed_length * ratio - seq_len
        if pad_len > 0:
            pad_states = torch.zeros(batch_size * num_heads, pad_len, head_dim, device=states.device, dtype=states.dtype)
            pad_weights = torch.zeros(batch_size * num_heads, pad_len, device=states.device, dtype=token_importance.dtype)
            flat = torch.cat([flat, pad_states], dim=1)
            token_importance = torch.cat([token_importance, pad_weights], dim=1)
        
        flat = flat.view(batch_size * num_heads, compressed_length, ratio, head_dim)
        w = token_importance.view(batch_size * num_heads, compressed_length, ratio)
        w_sum = w.sum(dim=2, keepdim=True) + 1e-8
        pooled = (flat * w.unsqueeze(-1)).sum(dim=2) / w_sum
        
        return pooled.view(batch_size, num_heads, compressed_length, head_dim)
    
    def _streaming_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_manager=None
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        device = query_states.device
        
        if seq_len <= self.streaming_window:
            attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            return torch.matmul(attention_weights, value_states)
        
        output_states = torch.zeros_like(query_states)
        
        importance_scores = self._calculate_importance_scores(key_states, value_states)
        cache_dict = self._build_hierarchical_cache(key_states, value_states, importance_scores)
        
        for start_idx in range(0, seq_len, self.streaming_window):
            end_idx = min(start_idx + self.streaming_window, seq_len)
            window_size = end_idx - start_idx
            
            window_query = query_states[:, :, start_idx:end_idx, :]
            
            level_outputs = {}
            for level_name, (level_keys, level_values) in cache_dict.items():
                if level_keys.shape[2] > 0:
                    level_q = window_query.reshape(batch_size * num_heads, window_size, head_dim)
                    level_k = level_keys.reshape(batch_size * num_heads, -1, head_dim)
                    level_v = level_values.reshape(batch_size * num_heads, -1, head_dim)
                    
                    row_pos = torch.arange(start_idx, end_idx, device=device)
                    pos_expanded = torch.arange(level_keys.shape[2], device=device).view(1, 1, -1)
                    pos_expanded = pos_expanded.expand(batch_size, num_heads, -1)
                    allowed = pos_expanded.unsqueeze(2) <= row_pos.view(1, 1, window_size, 1)
                    
                    disallow = ~allowed
                    attn_mask = disallow.reshape(batch_size * num_heads, window_size, -1)
                    
                    level_out = F.scaled_dot_product_attention(
                        level_q, level_k, level_v,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=False
                    )
                    level_outputs[level_name] = level_out.reshape(batch_size, num_heads, window_size, head_dim)
            
            if level_outputs:
                window_output = self._fuse_hierarchical_outputs(level_outputs, window_query)
                output_states[:, :, start_idx:end_idx, :] = window_output
        
        return output_states
    
    def _calculate_importance_scores(self, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        key_magnitude = torch.norm(key_states, dim=-1)
        value_magnitude = torch.norm(value_states, dim=-1)
        importance = key_magnitude + value_magnitude
        position_weights = torch.exp(-torch.arange(seq_len, device=key_states.device).float() / 100.0)
        position_weights = position_weights.unsqueeze(0).unsqueeze(0)
        importance = importance * position_weights
        importance = F.softmax(importance, dim=-1)
        return importance


class RuchbahH2OAttention(nn.Module):
    """Implement H2O attention with heavy-hitter retention and streaming support."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int = 10485760,
        compression_ratio: int = 8,
        heavy_hitter_ratio: float = 0.1,
        streaming_window: int = 16384,
        dropout: float = 0.1
    ):
        """Initialize projection layers and H2O routing thresholds.

        Args:
            hidden_size (int): Model hidden dimensionality.
            num_attention_heads (int): Number of attention heads for the block.
            max_position_embeddings (int): Upper bound for absolute positional bias generation.
            compression_ratio (int): Baseline ratio used when compressing long sequences.
            heavy_hitter_ratio (float): Fraction of tokens considered heavy hitters per step.
            streaming_window (int): Window size preserved verbatim during streaming mode.
            dropout (float): Dropout probability applied to attention outputs.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.compression_ratio = compression_ratio
        self.heavy_hitter_ratio = heavy_hitter_ratio
        self.streaming_window = streaming_window
        
        # Ensure the hidden size is divisible by the number of attention heads
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        
        # Initialize projection layers for query, key, value, and output
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize H2O-specific parameter
        self.heavy_hitter_threshold = None
        
    def _create_position_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return exponential decay position bias promoting locality.

        Args:
            seq_len (int): Current sequence length.
            device (torch.device): Device used to allocate the bias tensor.

        Returns:
            torch.Tensor: Position bias tensor shaped ``[seq_len, seq_len]``.
        """
        # Generate position indices.
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        # Compute the position bias with exponential decay.
        bias = torch.exp(-torch.abs(positions - positions.T) / 1000.0)
        return bias
        
    def _compute_heavy_hitters(
        self, 
        attention_scores: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """Select token indices with highest combined importance scores.

        Args:
            attention_scores (torch.Tensor): Attention weights shaped ``[B, H, T, T]``.
            seq_len (int): Sequence length used for determining heavy-hitter count.

        Returns:
            torch.Tensor: Heavy-hitter indices shaped ``[B, max_heavy_hitters]``.
        """
        batch_size, num_heads, _, _ = attention_scores.shape
        
        # Compute multi-dimensional token importance scores
        # Calculate attention entropy
        attention_entropy = -torch.sum(
            attention_scores * torch.log(attention_scores.clamp(min=1e-8)), 
            dim=-1
        )  # [batch, heads, seq_len]
        
        # Get the maximum attention score for each token
        max_attention = torch.max(attention_scores, dim=-1)[0]  # [batch, heads, seq_len]
        
        # Calculate the influence of each token
        attention_influence = torch.sum(attention_scores, dim=-2)  # [batch, heads, seq_len]
        
        # Combine multiple importance signals
        # Lower entropy indicates higher importance
        entropy_importance = -attention_entropy.mean(dim=1)
        # Higher max attention indicates higher importance
        max_importance = max_attention.mean(dim=1)
        # Higher influence indicates higher importance
        influence_importance = attention_influence.mean(dim=1)
        
        # Calculate the final importance scores
        importance_scores = (
            entropy_importance * 0.4 + 
            max_importance * 0.3 + 
            influence_importance * 0.3
        )  # [batch, seq_len]
        
        # Determine the number of heavy-hitter tokens to prevent memory issues
        max_heavy_hitters = min(1024, int(seq_len * self.heavy_hitter_ratio))
        
        # Get the indices of top-k important tokens
        _, top_k_indices = torch.topk(importance_scores, max_heavy_hitters, dim=-1)
        
        return top_k_indices
        
    def _compress_states(
        self, 
        states: torch.Tensor,
        compression_ratio: int = None
    ) -> torch.Tensor:
        """Compress key/value states using adaptive importance-weighted pooling.

        Args:
            states (torch.Tensor): Input tensor shaped ``[B, H, T, D]``.
            compression_ratio (int, optional): Override for baseline compression ratio.

        Returns:
            torch.Tensor: Compressed states preserving salient information.
        """
        batch_size, num_heads, seq_len, head_dim = states.shape
        device = states.device
        ratio = compression_ratio or self.compression_ratio
        
        # Skip compression for short sequences
        if seq_len <= self.streaming_window:
            return states
            
        # Calculate adaptive compression ratio based on sequence characteristics
        seq_complexity = torch.std(states) / (torch.mean(torch.abs(states)) + 1e-8)
        adaptive_ratio = max(1, min(ratio, int(seq_complexity * ratio)))
        # Prevent over-compression
        actual_ratio = min(adaptive_ratio, max(1, seq_len // 512))
        
        # Compute the length of the compressed sequence
        compressed_length = (seq_len + actual_ratio - 1) // actual_ratio

        # Perform vectorized importance-weighted pooling
        # Reshape the states tensor
        flat = states.view(batch_size * num_heads, seq_len, head_dim)  # [B*H, T, D]
        # Calculate token importance
        token_importance = torch.norm(flat, dim=-1)  # [B*H, T]
        # Normalize token importance
        token_importance = F.softmax(token_importance, dim=-1)

        # Pad the sequence to make its length divisible by actual_ratio
        pad_len = compressed_length * actual_ratio - seq_len
        if pad_len > 0:
            pad_states = torch.zeros(batch_size * num_heads, pad_len, head_dim, device=device, dtype=states.dtype)
            pad_weights = torch.zeros(batch_size * num_heads, pad_len, device=device, dtype=token_importance.dtype)
            flat = torch.cat([flat, pad_states], dim=1)
            token_importance = torch.cat([token_importance, pad_weights], dim=1)

        # Reshape the tensor for weighted average calculation
        flat = flat.view(batch_size * num_heads, compressed_length, actual_ratio, head_dim)
        w = token_importance.view(batch_size * num_heads, compressed_length, actual_ratio)

        # Compute weighted average for each bucket
        w_sum = w.sum(dim=2, keepdim=True) + 1e-8
        pooled = (flat * w.unsqueeze(-1)).sum(dim=2) / w_sum  # [B*H, compressed_length, D]

        # Reshape the pooled tensor back to the original format
        compressed_states = pooled.view(batch_size, num_heads, compressed_length, head_dim)

        return compressed_states
        
    def _calculate_importance_scores(self, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        """Compute per-token importance for KV cache eviction policies.

        Args:
            key_states (torch.Tensor): Key tensor shaped ``[B, H, T, D]``.
            value_states (torch.Tensor): Value tensor shaped ``[B, H, T, D]``.

        Returns:
            torch.Tensor: Normalized importance scores shaped ``[B, H, T]``.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Calculate importance based on key and value magnitudes
        # Calculate key magnitude
        key_magnitude = torch.norm(key_states, dim=-1)  # [batch, heads, seq_len]
        # Calculate value magnitude
        value_magnitude = torch.norm(value_states, dim=-1)  # [batch, heads, seq_len]
        
        # Combine key and value magnitudes
        importance = key_magnitude + value_magnitude
        
        # Incorporate position-based importance (recent tokens are more important)
        position_weights = torch.exp(-torch.arange(seq_len, device=key_states.device).float() / 100.0)
        position_weights = position_weights.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]
        importance = importance * position_weights
        
        # Normalize importance scores
        importance = F.softmax(importance, dim=-1)
        
        return importance
        
    def _select_important_cache(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        importance_scores: torch.Tensor,
        current_pos: int,
        max_cache_size: int,
        cache_manager=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retain high-importance KV entries under cache budget constraints.

        Args:
            key_states (torch.Tensor): Key tensor shaped ``[B, H, T, D]``.
            value_states (torch.Tensor): Value tensor shaped ``[B, H, T, D]``.
            importance_scores (torch.Tensor): Importance weights shaped ``[B, H, T]``.
            current_pos (int): Current decoding position.
            max_cache_size (int): Maximum retained KV entries per head.
            cache_manager (Optional[Any]): Optional cache manager implementing get/set hooks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Selected keys, values, and their positions.
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        # Determine the cache window region up to the current position
        cache_end = current_pos
        cache_start = 0
        if cache_end <= 0:
            cache_end = min(seq_len, self.streaming_window)

        # Define the recent window to preserve
        recent_keep = min(max_cache_size // 4, self.streaming_window // 2)
        recent_start = max(0, cache_end - recent_keep)

        # Check the cache manager first
        if cache_manager is not None:
            cached_keys, cached_values = cache_manager.get_h2o_cache(key_states, current_pos, max_cache_size)
            if cached_keys is not None and cached_values is not None:
                return cached_keys, cached_values, None

        # Slice the candidate pool up to the current position
        pool_keys = key_states[:, :, :cache_end, :]
        pool_values = value_states[:, :, :cache_end, :]
        pool_importance = importance_scores[:, :, :cache_end]

        # Always include the recent window
        recent_keys = pool_keys[:, :, recent_start:cache_end, :]
        recent_values = pool_values[:, :, recent_start:cache_end, :]
        recent_len = cache_end - recent_start
        recent_pos = torch.arange(recent_start, cache_end, device=key_states.device)
        recent_pos = recent_pos.view(1, 1, recent_len).expand(batch_size, num_heads, recent_len)

        # Calculate the remaining cache budget after including the recent window
        remaining = max(0, max_cache_size - (cache_end - recent_start))
        if remaining == 0:
            pos = recent_pos[:, :, -max_cache_size:]
            return recent_keys[:, :, -max_cache_size:, :], recent_values[:, :, -max_cache_size:, :], pos

        # Exclude the recent region from importance selection to avoid duplicates
        imp_region = pool_importance[:, :, :recent_start] if recent_start > 0 else None
        if imp_region is None or imp_region.shape[2] == 0:
            # No earlier tokens to select from
            selected_keys = recent_keys
            selected_values = recent_values
            pos = recent_pos
        else:
            # Allocate cache budget proportionally to each head based on importance
            head_importance = imp_region.sum(dim=-1)  # [B, H]
            alloc = head_importance / (head_importance.sum(dim=1, keepdim=True) + 1e-8)  # [B, H]
            alloc = alloc.mean(dim=0)  # [H]
            
            quotas = (alloc * remaining).round().to(torch.long)
            diff = int(remaining - quotas.sum().item())
            if diff != 0:
                # Distribute the difference to heads with the highest residuals
                order = torch.argsort(alloc, descending=True)
                for i in range(min(abs(diff), num_heads)):
                    idx = order[i].item()
                    quotas[idx] = max(0, quotas[idx] + (1 if diff > 0 else -1))
            
            # Fallback to equal split if the quota distribution is degenerate
            if quotas.sum().item() <= 0:
                quotas = torch.full((num_heads,), max(1, remaining // max(1, num_heads)), dtype=torch.long, device=imp_region.device)

            # Select top-k important tokens for each head
            sel_keys = []
            sel_vals = []
            sel_pos_list = []
            head_space = imp_region.shape[2]
            for h in range(num_heads):
                k_h = int(min(max(0, quotas[h].item()), head_space))
                if k_h <= 0:
                    continue
                imp_h = imp_region[:, h:h+1, :]  # [B,1,T]
                _, idx_h = torch.topk(imp_h, k=k_h, dim=-1)
                idx_h = torch.sort(idx_h, dim=-1).values
                k_src = pool_keys[:, h:h+1, :recent_start, :]
                v_src = pool_values[:, h:h+1, :recent_start, :]
                k_sel_h = torch.gather(k_src, 2, idx_h.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                v_sel_h = torch.gather(v_src, 2, idx_h.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                sel_keys.append(k_sel_h)
                sel_vals.append(v_sel_h)
                sel_pos_h = idx_h.expand(-1, -1, -1).clone()  # [B,1,k_h]
                sel_pos_list.append(sel_pos_h)

            if sel_keys:
                keys_sel = torch.cat(sel_keys, dim=1)  # Concatenate selected keys across heads
                vals_sel = torch.cat(sel_vals, dim=1)
                pos_sel = torch.cat(sel_pos_list, dim=1)
            else:
                # No earlier tokens selected
                keys_sel = pool_keys[:, :, :0, :]
                vals_sel = pool_values[:, :, :0, :]
                pos_sel = pool_values.new_zeros((batch_size, num_heads, 0), dtype=torch.long)

            # Concatenate selected tokens with the recent window
            selected_keys = torch.cat([keys_sel, recent_keys], dim=2)
            selected_values = torch.cat([vals_sel, recent_values], dim=2)
            pos = torch.cat([pos_sel.to(torch.long), recent_pos.to(torch.long)], dim=2)

            # Truncate if the selected states exceed the cache budget
            if selected_keys.shape[2] > max_cache_size:
                selected_keys = selected_keys[:, :, -max_cache_size:, :]
                selected_values = selected_values[:, :, -max_cache_size:, :]
                pos = pos[:, :, -max_cache_size:]

        # Cache the selected states if a cache manager is available
        if cache_manager is not None:
            cache_manager.set_h2o_cache(key_states, current_pos, max_cache_size, selected_keys, selected_values)

        return selected_keys, selected_values, pos
        
    def _streaming_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_manager=None
    ) -> torch.Tensor:
        """Execute streaming attention using sliding windows and H2O compression.

        Args:
            query_states (torch.Tensor): Query tensor shaped ``[B, H, T, D]``.
            key_states (torch.Tensor): Key tensor shaped ``[B, H, T, D]``.
            value_states (torch.Tensor): Value tensor shaped ``[B, H, T, D]``.
            attention_mask (Optional[torch.Tensor]): Optional additive mask.
            cache_manager (Optional[Any]): Optional cache manager providing H2O caches.

        Returns:
            torch.Tensor: Attention outputs shaped ``[B, H, T, D]``.
        """
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        device = query_states.device
        
        if seq_len <= self.streaming_window:
            # Perform standard attention for short sequences
            attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
                
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            return torch.matmul(attention_weights, value_states)
        
        # Initialize output states
        output_states = torch.zeros_like(query_states)
        
        # Calculate importance scores for key-value cache management
        importance_scores = self._calculate_importance_scores(key_states, value_states)
        
        # Process the sequence in streaming windows
        for start_idx in range(0, seq_len, self.streaming_window):
            end_idx = min(start_idx + self.streaming_window, seq_len)
            window_size = end_idx - start_idx
            
            # Extract query states for the current window
            window_query = query_states[:, :, start_idx:end_idx, :]
            
            # Select important key-value pairs for the cache
            if seq_len > self.streaming_window * 2:
                cache_budget = min(self.streaming_window * 2, end_idx)
                cached_key, cached_value, cached_pos = self._select_important_cache(
                    key_states, value_states, importance_scores,
                    end_idx, cache_budget, cache_manager=cache_manager
                )
            else:
                cached_key = key_states[:, :, :end_idx, :]
                cached_value = value_states[:, :, :end_idx, :]
                cached_pos = torch.arange(end_idx, device=device).view(1, 1, end_idx).expand(batch_size, num_heads, end_idx)
            
            # Build a causal mask for each head
            row_pos = torch.arange(start_idx, end_idx, device=device)  # [W]
            allowed = (cached_pos.unsqueeze(2) <= row_pos.view(1, 1, window_size, 1))  # [B,H,W,K]
            
            # Reshape tensors for scaled dot-product attention
            q = window_query.reshape(batch_size * num_heads, window_size, head_dim)
            k = cached_key.reshape(batch_size * num_heads, -1, head_dim)
            v = cached_value.reshape(batch_size * num_heads, -1, head_dim)
            
            # Convert the allowed mask to a disallow mask
            disallow = (~allowed)  # [B,H,W,K]
            if attention_mask is not None:
                mask_slice = attention_mask[:, :, start_idx:end_idx, :cached_key.shape[2]]
                if mask_slice.dtype == torch.bool:
                    extra_disallow = ~mask_slice
                else:
                    extra_disallow = mask_slice < -1e4
                disallow = disallow | extra_disallow
            attn_mask = disallow.reshape(batch_size * num_heads, window_size, -1)
            
            # Use the flash attention backend if available
            try:
                from torch.backends.cuda import sdp_kernel as _sdp
                use_ctx = torch.cuda.is_available()
            except Exception:
                use_ctx = False
            if use_ctx:
                with _sdp(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                    window_output = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=False
                    )  # [B*H, W, D]
            else:
                window_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False
                )  # [B*H, W, D]
            window_output = window_output.reshape(batch_size, num_heads, window_size, head_dim)

            output_states[:, :, start_idx:end_idx, :] = window_output
            
        return output_states
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_manager=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Perform the forward pass of the H2O attention module.

        Args:
            hidden_states (torch.Tensor): Input hidden states with shape [batch, seq_len, hidden_size].
            attention_mask (torch.Tensor, optional): Optional attention mask.
            past_key_value (Tuple[torch.Tensor], optional): Optional cached key-value states.
            output_attentions (bool, optional): Whether to output attention weights.
            use_cache (bool, optional): Whether to use key-value caching.
            cache_manager: Optional cache manager object.

        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]: 
                attention_output: Output from H2O attention with shape [batch, seq_len, hidden_size].
                present_key_value: Updated key-value cache.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project input hidden states to query, key, and value states
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Incorporate past key-value states if available
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            
        # Compress key and value states for long sequences
        if seq_len > self.streaming_window * 2:
            compressed_key = self._compress_states(key_states)
            compressed_value = self._compress_states(value_states)
            attention_key = compressed_key
            attention_value = compressed_value
        else:
            attention_key = key_states
            attention_value = value_states
        
        # Perform streaming attention
        attention_output = self._streaming_attention(
            query_states, attention_key, attention_value, attention_mask,
            cache_manager=cache_manager
        )
        
        # Reshape the attention output to the original shape
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Apply the final projection
        attention_output = self.o_proj(attention_output)
        
        # Prepare the key-value cache for the next step
        present_key_value = None
        if use_cache:
            present_key_value = (key_states, value_states)
            
        return attention_output, present_key_value
