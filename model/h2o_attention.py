#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

class ArcticH2OAttention(nn.Module):
    """
    Module implementing the H2O attention mechanism for processing ultra-long sequences.
    This mechanism selectively retains important tokens (heavy-hitters) and compresses less important ones.
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
        Initialize the ArcticH2OAttention module.

        Args:
            hidden_size (int): Dimensionality of the input hidden states.
            num_attention_heads (int): Number of attention heads.
            max_position_embeddings (int, optional): Maximum number of position embeddings. Defaults to 10485760.
            compression_ratio (int, optional): Compression ratio for state compression. Defaults to 8.
            heavy_hitter_ratio (float, optional): Ratio of heavy-hitter tokens. Defaults to 0.1.
            streaming_window (int, optional): Size of the streaming window. Defaults to 16384.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
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
        """
        Create a position bias tensor for attention scores in a memory-efficient manner.

        Args:
            seq_len (int): Length of the sequence.
            device (torch.device): Device to place the tensors on.

        Returns:
            torch.Tensor: Position bias tensor.
        """
        # Generate position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        # Compute the position bias
        bias = torch.exp(-torch.abs(positions - positions.T) / 1000.0)
        return bias
        
    def _compute_heavy_hitters(
        self, 
        attention_scores: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Identify the indices of heavy-hitter tokens based on attention scores.

        Args:
            attention_scores (torch.Tensor): Attention score matrix with shape [batch, heads, seq_len, seq_len].
            seq_len (int): Length of the sequence.

        Returns:
            torch.Tensor: Indices of heavy-hitter tokens with shape [batch, max_heavy_hitters].
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
        """
        Compress states using attention-weighted pooling and importance-based retention.

        Args:
            states (torch.Tensor): States to compress with shape [batch, heads, seq_len, head_dim].
            compression_ratio (int, optional): Compression ratio. Defaults to self.compression_ratio.

        Returns:
            torch.Tensor: Compressed states with preserved semantic information.
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
        """
        Calculate importance scores for key-value cache management.

        Args:
            key_states (torch.Tensor): Key states with shape [batch, heads, seq_len, head_dim].
            value_states (torch.Tensor): Value states with shape [batch, heads, seq_len, head_dim].

        Returns:
            torch.Tensor: Importance scores with shape [batch, heads, seq_len].
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
        """
        Select important key-value pairs for the cache based on importance scores.

        Args:
            key_states (torch.Tensor): Key states with shape [batch, heads, seq_len, head_dim].
            value_states (torch.Tensor): Value states with shape [batch, heads, seq_len, head_dim].
            importance_scores (torch.Tensor): Importance scores with shape [batch, heads, seq_len].
            current_pos (int): Current position in the sequence.
            max_cache_size (int): Maximum size of the cache.
            cache_manager: Optional cache manager object.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                selected_keys: Selected key states.
                selected_values: Selected value states.
                pos: Positions of the selected states.
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
        """
        Perform streaming attention with H2O compression and sliding window cache.

        Args:
            query_states (torch.Tensor): Query states with shape [batch, heads, seq_len, head_dim].
            key_states (torch.Tensor): Key states with shape [batch, heads, seq_len, head_dim].
            value_states (torch.Tensor): Value states with shape [batch, heads, seq_len, head_dim].
            attention_mask (torch.Tensor, optional): Optional attention mask.
            cache_manager: Optional cache manager object.

        Returns:
            torch.Tensor: Output from streaming attention with shape [batch, heads, seq_len, head_dim].
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