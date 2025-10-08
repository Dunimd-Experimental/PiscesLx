#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
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
import torch.nn.functional as F
from typing import Optional, Tuple

class H2OAttention(nn.Module):
    """
    H2O: Heavy-Hitter Oracle for Efficient Ultra-Long Context Attention
    
    This module implements the H2O attention mechanism that enables processing
    of ultra-long sequences (up to 10M tokens) by selectively retaining
    important tokens (heavy-hitters) while compressing less important ones.
    """
    
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
        """
        Initialize the H2OAttention module.

        Args:
            hidden_size (int): Dimensionality of the input hidden states.
            num_attention_heads (int): Number of attention heads.
            max_position_embeddings (int, optional): Maximum number of position embeddings, defaults to 10485760.
            compression_ratio (int, optional): Compression ratio for state compression, defaults to 8.
            heavy_hitter_ratio (float, optional): Ratio of heavy-hitter tokens, defaults to 0.1.
            streaming_window (int, optional): Size of the streaming window, defaults to 16384.
            dropout (float, optional): Dropout probability, defaults to 0.1.
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
        assert hidden_size % num_attention_heads == 0
        
        # Projection layers for query, key, value, and output
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # H2O-specific parameters
        self.heavy_hitter_threshold = None
        
    def _create_position_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create position bias for attention scores in a memory-efficient manner.

        Args:
            seq_len (int): Length of the sequence.
            device (torch.device): Device to place the tensors on.

        Returns:
            torch.Tensor: Position bias tensor.
        """
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        bias = torch.exp(-torch.abs(positions - positions.T) / 1000.0)
        return bias
        
    def _compute_heavy_hitters(
        self, 
        attention_scores: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Identify heavy-hitter tokens based on attention scores (simplified).
        
        Args:
            attention_scores (torch.Tensor): Attention score matrix [batch, heads, seq_len, seq_len]
            seq_len (int): Sequence length
            
        Returns:
            torch.Tensor: Indices of heavy-hitter tokens
        """
        batch_size, num_heads, _, _ = attention_scores.shape
        
        # Compute token importance scores using advanced attention analysis
        # Multi-dimensional importance scoring considering attention patterns
        attention_entropy = -torch.sum(
            attention_scores * torch.log(attention_scores.clamp(min=1e-8)), 
            dim=-1
        )  # [batch, heads, seq_len]
        
        # Attention concentration measure
        max_attention = torch.max(attention_scores, dim=-1)[0]  # [batch, heads, seq_len]
        
        # Cross-token attention influence
        attention_influence = torch.sum(attention_scores, dim=-2)  # [batch, heads, seq_len]
        
        # Combine multiple importance signals
        entropy_importance = -attention_entropy.mean(dim=1)  # Lower entropy = higher importance
        max_importance = max_attention.mean(dim=1)  # Higher max attention = higher importance
        influence_importance = attention_influence.mean(dim=1)  # Higher influence = higher importance
        
        # Weighted combination with learned importance factors
        importance_scores = (
            entropy_importance * 0.4 + 
            max_importance * 0.3 + 
            influence_importance * 0.3
        )  # [batch, seq_len]
        
        # Determine heavy-hitter count (limit to prevent memory issues)
        max_heavy_hitters = min(1024, int(seq_len * self.heavy_hitter_ratio))
        
        # Get top-k indices
        _, top_k_indices = torch.topk(importance_scores, max_heavy_hitters, dim=-1)
        
        return top_k_indices
        
    def _compress_states(
        self, 
        states: torch.Tensor,
        compression_ratio: int = None
    ) -> torch.Tensor:
        """
        Perform advanced state compression using attention-weighted pooling and importance-based retention.
        
        Args:
            states (torch.Tensor): States to compress [batch, heads, seq_len, head_dim]
            compression_ratio (int, optional): Compression ratio, defaults to self.compression_ratio
            
        Returns:
            torch.Tensor: Compressed states with preserved semantic information
        """
        batch_size, num_heads, seq_len, head_dim = states.shape
        device = states.device
        ratio = compression_ratio or self.compression_ratio
        
        # No compression needed for short sequences
        if seq_len <= self.streaming_window:
            return states
            
        # Adaptive compression ratio based on sequence characteristics
        seq_complexity = torch.std(states) / (torch.mean(torch.abs(states)) + 1e-8)
        adaptive_ratio = max(1, min(ratio, int(seq_complexity * ratio)))
        actual_ratio = min(adaptive_ratio, max(1, seq_len // 512))  # Prevent over-compression
        
        # Calculate compressed length (ceil division)
        compressed_length = (seq_len + actual_ratio - 1) // actual_ratio

        # Vectorized importance-weighted pooling
        flat = states.view(batch_size * num_heads, seq_len, head_dim)  # [B*H, T, D]
        token_importance = torch.norm(flat, dim=-1)  # [B*H, T]
        token_importance = F.softmax(token_importance, dim=-1)

        # Pad to multiple of actual_ratio for clean reshape
        pad_len = compressed_length * actual_ratio - seq_len
        if pad_len > 0:
            pad_states = torch.zeros(batch_size * num_heads, pad_len, head_dim, device=device, dtype=states.dtype)
            pad_weights = torch.zeros(batch_size * num_heads, pad_len, device=device, dtype=token_importance.dtype)
            flat = torch.cat([flat, pad_states], dim=1)
            token_importance = torch.cat([token_importance, pad_weights], dim=1)

        # Reshape into [B*H, compressed_length, actual_ratio, D]
        flat = flat.view(batch_size * num_heads, compressed_length, actual_ratio, head_dim)
        w = token_importance.view(batch_size * num_heads, compressed_length, actual_ratio)

        # Weighted average per bucket
        w_sum = w.sum(dim=2, keepdim=True) + 1e-8
        pooled = (flat * w.unsqueeze(-1)).sum(dim=2) / w_sum  # [B*H, compressed_length, D]

        # Reshape back
        compressed_states = pooled.view(batch_size, num_heads, compressed_length, head_dim)

        return compressed_states
        
    def _calculate_importance_scores(self, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        """Calculate importance scores for key-value cache management"""
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Calculate importance based on key-value magnitude and diversity
        key_magnitude = torch.norm(key_states, dim=-1)  # [batch, heads, seq_len]
        value_magnitude = torch.norm(value_states, dim=-1)  # [batch, heads, seq_len]
        
        # Combine magnitudes
        importance = key_magnitude + value_magnitude
        
        # Add position-based importance (recent tokens are more important)
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
        """Select important key-value pairs for cache based on importance scores.
        Strategy:
        - Always preserve a recent window (recency prior)
        - Fill remaining budget with per-head Top-K (diversity across heads)
        - Optionally reuse from cache_manager if present
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        # Determine cache window region up to current
        cache_end = current_pos
        cache_start = 0
        if cache_end <= 0:
            cache_end = min(seq_len, self.streaming_window)

        # Define recency preservation
        recent_keep = min(max_cache_size // 4, self.streaming_window // 2)
        recent_start = max(0, cache_end - recent_keep)

        # Check cache manager first
        if cache_manager is not None:
            cached_keys, cached_values = cache_manager.get_h2o_cache(key_states, current_pos, max_cache_size)
            if cached_keys is not None and cached_values is not None:
                return cached_keys, cached_values

        # Slice candidate pool up to current_pos
        pool_keys = key_states[:, :, :cache_end, :]
        pool_values = value_states[:, :, :cache_end, :]
        pool_importance = importance_scores[:, :, :cache_end]

        # Always include recent window
        recent_keys = pool_keys[:, :, recent_start:cache_end, :]
        recent_values = pool_values[:, :, recent_start:cache_end, :]
        recent_len = cache_end - recent_start
        # Positions for recent window
        recent_pos = torch.arange(recent_start, cache_end, device=key_states.device)
        recent_pos = recent_pos.view(1, 1, recent_len).expand(batch_size, num_heads, recent_len)

        # Remaining budget after recent window
        remaining = max(0, max_cache_size - (cache_end - recent_start))
        if remaining == 0:
            pos = recent_pos[:, :, -max_cache_size:]
            return recent_keys[:, :, -max_cache_size:, :], recent_values[:, :, -max_cache_size:, :], pos

        # Exclude recent region from importance selection to avoid duplicates
        imp_region = pool_importance[:, :, :recent_start] if recent_start > 0 else None
        if imp_region is None or imp_region.shape[2] == 0:
            # No earlier tokens to select from
            selected_keys = recent_keys
            selected_values = recent_values
            pos = recent_pos
        else:
            # Head-wise budget based on summed importance (proportional allocation)
            # importance per head: [B, H, T]
            head_importance = imp_region.sum(dim=-1)  # [B, H]
            # Avoid div-by-zero; compute allocation per batch then average across batch
            alloc = head_importance / (head_importance.sum(dim=1, keepdim=True) + 1e-8)  # [B, H]
            alloc = alloc.mean(dim=0)  # [H]
            # Convert to integer quotas, ensure at least 1 when remaining >= H
            quotas = (alloc * remaining).round().to(torch.long)
            # Adjust quotas to sum to remaining
            diff = int(remaining - quotas.sum().item())
            if diff != 0:
                # Distribute the difference to heads with highest residuals or clip
                order = torch.argsort(alloc, descending=True)
                for i in range(min(abs(diff), num_heads)):
                    idx = order[i].item()
                    quotas[idx] = max(0, quotas[idx] + (1 if diff > 0 else -1))
            # Fallback to equal split if degenerate
            if quotas.sum().item() <= 0:
                quotas = torch.full((num_heads,), max(1, remaining // max(1, num_heads)), dtype=torch.long, device=imp_region.device)

            # Per-head top-k according to quotas
            # Build selections per head and concatenate along time dim
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
                # sort indices to maintain chronological order
                idx_h = torch.sort(idx_h, dim=-1).values
                k_src = pool_keys[:, h:h+1, :recent_start, :]
                v_src = pool_values[:, h:h+1, :recent_start, :]
                k_sel_h = torch.gather(k_src, 2, idx_h.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                v_sel_h = torch.gather(v_src, 2, idx_h.unsqueeze(-1).expand(-1, -1, -1, head_dim))
                sel_keys.append(k_sel_h)
                sel_vals.append(v_sel_h)
                # positions for selected indices
                sel_pos_h = idx_h.expand(-1, -1, -1).clone()  # [B,1,k_h]
                sel_pos_list.append(sel_pos_h)

            if sel_keys:
                keys_sel = torch.cat(sel_keys, dim=1)  # concat across heads
                vals_sel = torch.cat(sel_vals, dim=1)
                pos_sel = torch.cat(sel_pos_list, dim=1)
            else:
                # No earlier tokens selected
                keys_sel = pool_keys[:, :, :0, :]
                vals_sel = pool_values[:, :, :0, :]
                pos_sel = pool_values.new_zeros((batch_size, num_heads, 0), dtype=torch.long)

            # Concatenate recent + selected
            selected_keys = torch.cat([keys_sel, recent_keys], dim=2)
            selected_values = torch.cat([vals_sel, recent_values], dim=2)
            pos = torch.cat([pos_sel.to(torch.long), recent_pos.to(torch.long)], dim=2)

            # If still beyond budget (due to rounding), keep latest portion
            if selected_keys.shape[2] > max_cache_size:
                selected_keys = selected_keys[:, :, -max_cache_size:, :]
                selected_values = selected_values[:, :, -max_cache_size:, :]
                pos = pos[:, :, -max_cache_size:]

        # Cache the result if cache manager is available
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
        """
        Perform streaming attention with H2O compression and sliding window cache.
        
        Args:
            query_states (torch.Tensor): Query states [batch, heads, seq_len, head_dim]
            key_states (torch.Tensor): Key states [batch, heads, seq_len, head_dim]
            value_states (torch.Tensor): Value states [batch, heads, seq_len, head_dim]
            attention_mask (torch.Tensor, optional): Optional attention mask
            
        Returns:
            torch.Tensor: Output from streaming attention
        """
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        device = query_states.device
        
        if seq_len <= self.streaming_window:
            # Standard attention for short sequences
            attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
                
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            return torch.matmul(attention_weights, value_states)
        
        # Enhanced streaming attention for long sequences with importance sampling
        output_states = torch.zeros_like(query_states)
        
        # Importance-based sliding window cache
        importance_scores = self._calculate_importance_scores(key_states, value_states)
        
        # Process in streaming windows with importance-aware caching
        for start_idx in range(0, seq_len, self.streaming_window):
            end_idx = min(start_idx + self.streaming_window, seq_len)
            window_size = end_idx - start_idx
            
            # Extract window states
            window_query = query_states[:, :, start_idx:end_idx, :]
            
            # Apply enhanced importance-based cache selection for long sequences
            if seq_len > self.streaming_window * 2:
                cache_budget = min(self.streaming_window * 2, end_idx)
                cached_key, cached_value, cached_pos = self._select_important_cache(
                    key_states, value_states, importance_scores,
                    end_idx, cache_budget, cache_manager=cache_manager
                )
            else:
                cached_key = key_states[:, :, :end_idx, :]
                cached_value = value_states[:, :, :end_idx, :]
                # Positions 0..end_idx-1 for all heads
                cached_pos = torch.arange(end_idx, device=device).view(1, 1, end_idx).expand(batch_size, num_heads, end_idx)
            
            # Build per-head causal mask using absolute positions
            row_pos = torch.arange(start_idx, end_idx, device=device)  # [W]
            # cached_pos: [B, H, K]
            allowed = (cached_pos.unsqueeze(2) <= row_pos.view(1, 1, window_size, 1))  # [B,H,W,K]
            # Prepare tensors for SDPA: [B*H, W, D] x [B*H, K, D] -> [B*H, W, D]
            q = window_query.reshape(batch_size * num_heads, window_size, head_dim)
            k = cached_key.reshape(batch_size * num_heads, -1, head_dim)
            v = cached_value.reshape(batch_size * num_heads, -1, head_dim)
            # Boolean attn_mask where True means mask (disallow)
            disallow = (~allowed)  # [B,H,W,K]
            if attention_mask is not None:
                mask_slice = attention_mask[:, :, start_idx:end_idx, :cached_key.shape[2]]  # expected additive mask
                # Convert additive mask to boolean disallow (values << 0 mean disallow)
                if mask_slice.dtype == torch.bool:
                    extra_disallow = ~mask_slice
                else:
                    extra_disallow = mask_slice < -1e4
                disallow = disallow | extra_disallow
            attn_mask = disallow.reshape(batch_size * num_heads, window_size, -1)
            # Prefer flash backend when available
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
        Forward pass of H2O attention.
        
        Args:
            hidden_states (torch.Tensor): Input hidden states [batch, seq_len, hidden_size]
            attention_mask (torch.Tensor, optional): Optional attention mask
            past_key_value (Tuple[torch.Tensor], optional): Optional cached key-value states
            output_attentions (bool, optional): Whether to output attention weights
            use_cache (bool, optional): Whether to use key-value caching
            
        Returns:
            Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]: 
                attention_output: Output from H2O attention
                present_key_value: Updated key-value cache
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
        
        # Handle past key-value states
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)
            
        # Apply compression for long sequences
        if seq_len > self.streaming_window * 2:
            # Compress keys and values for memory efficiency
            compressed_key = self._compress_states(key_states)
            compressed_value = self._compress_states(value_states)
            
            # Use compressed states for attention
            attention_key = compressed_key
            attention_value = compressed_value
        else:
            # Use original states for shorter sequences
            attention_key = key_states
            attention_value = value_states
        
        # Perform streaming attention
        attention_output = self._streaming_attention(
            query_states, attention_key, attention_value, attention_mask,
            cache_manager=cache_manager
        )
        
        # Reshape output to the original shape
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        # Apply final projection
        attention_output = self.o_proj(attention_output)
        
        # Prepare key-value cache
        present_key_value = None
        if use_cache:
            present_key_value = (key_states, value_states)
            
        return attention_output, present_key_value