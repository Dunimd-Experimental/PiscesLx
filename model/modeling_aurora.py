#!/usr/bin/env/python3

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
from torch import nn
from .moe import MoELayer
from utils import DEBUG, ERROR
import torch.nn.functional as F
from .config import PiscesConfig
from .multimodal import PiscesReasoner
from model.h2o_attention import H2OAttention
from model.moe_dynamic import DynamicMoELayer
from typing import Optional, Tuple, Dict, Any
from model.yarn_rope import YaRNRotaryEmbedding
from .reasoner import MultiModalReasoningEnhancer
from .multimodal import VisionEncoder, AudioEncoder, DocEncoder, VideoEncoder, AgentEncoder, DynamicModalFusion, CrossModalAttention
from model.speculative_decoder import SpeculativeDecoder, AdaptiveSpeculativeDecoder, SpeculativeConfig

class UnifiedCacheManager:
    """
    Unified cache manager that consolidates all caching mechanisms:
    - KV-Cache for transformer layers
    - H2O sliding window cache
    - Multimodal generation cache
    - Speculative decoding cache
    """
    
    def __init__(self, config: PiscesConfig):
        """Unified cache manager.
        Accepts a PiscesConfig or a plain dict of cache options for flexibility.
        """
        self.config = config
        # Allow dict-based options too
        if isinstance(config, dict):
            self.max_cache_size = config.get('kv_cache_max_size', 8192)
            self.cache_quantization = config.get('quantization_enabled', True)
            self.cache_window_size = config.get('streaming_window', 2048)
        else:
            self.max_cache_size = getattr(config, 'max_cache_size', 8192)
            self.cache_quantization = getattr(config, 'cache_quantization', True)
            self.cache_window_size = getattr(config, 'streaming_window', 2048)
        
        # Unified cache storage
        # KV cache schema (paged):
        #   kv_cache[layer_idx] = {
        #       'blocks': [(k_block, v_block), ...],
        #       'total_len': int
        #   }
        self.kv_cache = {}
        self.generation_cache = {}  # Multimodal generation cache
        self.speculative_cache = {}  # Speculative decoding cache
        self.h2o_cache = {}  # H2O importance-based cache
        # Paged KV parameters
        self.block_size = getattr(config, 'kv_cache_block_size', 512) if isinstance(config, object) else config.get('kv_cache_block_size', 512)
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
    
    def get_kv_cache(self, layer_idx: int, past_key_values: Optional[Tuple[torch.Tensor]] = None):
        """Get concatenated KV cache view for a specific layer (paged storage internally)."""
        entry = self.kv_cache.get(layer_idx, None)
        if entry is None:
            # Initialize from provided past_key_values if any
            if past_key_values is not None:
                k, v = past_key_values
                self.kv_cache[layer_idx] = {
                    'blocks': [(k, v)],
                    'total_len': k.shape[-2]
                }
                self.cache_misses += 1
                return past_key_values
            self.cache_misses += 1
            return None
        self.cache_hits += 1
        return self._concat_recent(layer_idx)
    
    def update_kv_cache(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, 
                       current_pos: int, use_h2o: bool = True):
        """Update paged KV cache; returns a concatenated recent view within max_cache_size."""
        # Optional H2O selection on the incoming tensors (when extremely long)
        if use_h2o and hasattr(self.config, 'use_h2o_attention') and self.config.use_h2o_attention:
            key_states, value_states = self._apply_h2o_cache_selection(
                key_states, value_states, current_pos
            )

        entry = self.kv_cache.get(layer_idx, None)
        if entry is None:
            entry = {'blocks': [], 'total_len': 0}
            self.kv_cache[layer_idx] = entry

        # Determine new tail to append as blocks
        new_total = key_states.shape[-2]
        delta = new_total - entry['total_len']
        if delta > 0:
            tail_k = key_states[:, :, -delta:, :]
            tail_v = value_states[:, :, -delta:, :]
            # Split tail into blocks of block_size
            bs = self.block_size
            num_blocks = (delta + bs - 1) // bs
            for i in range(num_blocks):
                s = i * bs
                e = min(delta, (i + 1) * bs)
                kb = tail_k[:, :, s:e, :]
                vb = tail_v[:, :, s:e, :]
                # Quantize per block if enabled and sufficiently long
                if self.cache_quantization and kb.shape[2] >= min(bs, 256):
                    kb, vb = self._quantize_cache(kb, vb)
                entry['blocks'].append((kb, vb))
                entry['total_len'] += (e - s)

        # Evict oldest blocks until within soft budget (1.5x) to allow compaction window
        soft_cap = int(self.max_cache_size * 1.5)
        while entry['total_len'] > soft_cap and entry['blocks']:
            kb, vb = entry['blocks'].pop(0)
            entry['total_len'] -= kb.shape[2]
            self.cache_evictions += 1

        # If still larger than max_cache_size, perform importance-based compaction on oldest blocks
        if entry['total_len'] > self.max_cache_size and len(entry['blocks']) >= 1:
            self._compact_blocks(entry)

        # Final hard cap eviction in case compaction is insufficient
        while entry['total_len'] > self.max_cache_size and entry['blocks']:
            kb, vb = entry['blocks'].pop(0)
            entry['total_len'] -= kb.shape[2]
            self.cache_evictions += 1

        # Return concatenated recent view
        return self._concat_recent(layer_idx)

    def _concat_recent(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate blocks into a recent-window KV up to max_cache_size."""
        entry = self.kv_cache.get(layer_idx, None)
        if entry is None or not entry['blocks']:
            return None
        # Concatenate all blocks (already evicted oldest beyond budget)
        ks = [b[0] for b in entry['blocks']]
        vs = [b[1] for b in entry['blocks']]
        k = torch.cat(ks, dim=2)
        v = torch.cat(vs, dim=2)
        return (k, v)

    def _compact_blocks(self, entry: Dict[str, Any]):
        """Compact oldest blocks by importance to reduce total_len near max_cache_size.
        Strategy per oldest block:
          - Always keep a recent tail region (recency prior)
          - From the remaining part, select tokens by per-token importance (L2 norm of K+V)
          - Target to shrink each compacted block by ~50%
        """
        try:
            if not entry['blocks']:
                return
            target_total = int(self.max_cache_size * 1.1)
            # Compact from oldest toward newest until under target_total
            idx = 0
            while entry['total_len'] > target_total and idx < len(entry['blocks']):
                kb, vb = entry['blocks'][idx]
                B, H, T, D = kb.shape
                if T < 64:
                    idx += 1
                    continue
                # Keep recent tail tokens
                keep_recent = max(32, T // 4)
                tail_k = kb[:, :, -keep_recent:, :]
                tail_v = vb[:, :, -keep_recent:, :]
                head_len = T - keep_recent
                if head_len <= 0:
                    idx += 1
                    continue
                head_k = kb[:, :, :head_len, :]
                head_v = vb[:, :, :head_len, :]
                # Importance: per-token L2 of K and V, averaged over heads
                imp_k = torch.norm(head_k, dim=-1)  # [B,H,head_len]
                imp_v = torch.norm(head_v, dim=-1)  # [B,H,head_len]
                imp = (imp_k + imp_v).mean(dim=1)   # [B, head_len]
                # Select top-k tokens from head
                topk = max(keep_recent, head_len // 2)
                topk = min(topk, head_len)
                _, idx_sel = torch.topk(imp, k=topk, dim=-1)
                # Sort to maintain chronological order
                idx_sel, _ = torch.sort(idx_sel, dim=-1)
                idx_sel_exp = idx_sel.unsqueeze(1).unsqueeze(-1).expand(B, H, topk, D)
                head_k_sel = torch.gather(head_k, 2, idx_sel_exp)
                head_v_sel = torch.gather(head_v, 2, idx_sel_exp)
                # Concatenate selected head + recent tail
                new_k = torch.cat([head_k_sel, tail_k], dim=2)
                new_v = torch.cat([head_v_sel, tail_v], dim=2)
                # Update block & total_len delta
                delta = T - new_k.shape[2]
                if delta > 0:
                    entry['blocks'][idx] = (new_k, new_v)
                    entry['total_len'] -= delta
                idx += 1
        except Exception:
            # If compaction fails for any reason, leave blocks as-is
            pass
    
    def get_generation_cache(self, modality: str):
        """Get generation cache for specific modality"""
        return self.generation_cache.get(modality, None)
    
    def set_generation_cache(self, modality: str, cache_data: torch.Tensor):
        """Set generation cache for specific modality"""
        self.generation_cache[modality] = cache_data
    
    def get_speculative_cache(self, draft_length: int):
        """Get speculative decoding cache"""
        cache_key = f"draft_{draft_length}"
        return self.speculative_cache.get(cache_key, None)
    
    def set_speculative_cache(self, draft_length: int, cache_data: Dict):
        """Set speculative decoding cache"""
        cache_key = f"draft_{draft_length}"
        self.speculative_cache[cache_key] = cache_data

    # --- H2O cache helpers used by H2OAttention ---
    def get_h2o_cache(self, key_states: torch.Tensor, current_pos: int, max_cache_size: int):
        cache_key = (current_pos // max(1, max_cache_size))
        return self.h2o_cache.get(cache_key, (None, None))

    def set_h2o_cache(self, key_states: torch.Tensor, current_pos: int, max_cache_size: int,
                      selected_keys: torch.Tensor, selected_values: torch.Tensor):
        cache_key = (current_pos // max(1, max_cache_size))
        self.h2o_cache[cache_key] = (selected_keys, selected_values)
    
    def _apply_h2o_cache_selection(self, key_states: torch.Tensor, value_states: torch.Tensor, 
                                   current_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply H2O importance-based cache selection"""
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if seq_len <= self.cache_window_size:
            return key_states, value_states
        
        # Calculate importance scores based on attention patterns
        importance_scores = self._calculate_importance_scores(key_states, value_states)
        
        # Select important cache entries
        cache_start = max(0, current_pos - self.cache_window_size)
        cache_end = current_pos
        
        if cache_end - cache_start > self.cache_window_size:
            # Select most important tokens within window
            cache_importance = importance_scores[:, :, cache_start:cache_end]
            _, top_indices = torch.topk(cache_importance, self.cache_window_size, dim=-1)
            
            # Gather selected cache entries
            selected_keys = torch.gather(key_states, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            selected_values = torch.gather(value_states, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            
            return selected_keys, selected_values
        
        return key_states, value_states
    
    def _calculate_importance_scores(self, key_states: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
        """Calculate importance scores for H2O cache selection"""
        # Simple importance based on key-value similarity and position
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Calculate attention scores as importance measure
        attention_scores = torch.matmul(key_states, value_states.transpose(-2, -1)) / math.sqrt(head_dim)
        importance = attention_scores.diagonal(dim1=-2, dim2=-1)  # Self-attention scores
        
        # Add positional weighting (more recent tokens are more important)
        position_weights = torch.exp(-torch.arange(seq_len, device=key_states.device).float() / 100.0)
        position_weights = position_weights.unsqueeze(0).unsqueeze(0)
        importance = importance * position_weights
        
        return F.softmax(importance, dim=-1)
    
    def _quantize_cache(self, key_states: torch.Tensor, value_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply dynamic quantization to cache (FP8 preferred on CUDA)."""
        seq_len = key_states.shape[2]

        # Skip short sequences
        if seq_len <= self.cache_window_size:
            return key_states, value_states

        # Prefer FP8-like fake quantization on CUDA (keep original dtype for ops compatibility)
        try:
            use_fp8_like = torch.cuda.is_available()
        except Exception:
            use_fp8_like = False

        if use_fp8_like:
            def fake_fp8_quant(t: torch.Tensor) -> torch.Tensor:
                # t: [B, H, T, D]
                b, h, tlen, d = t.shape
                t_ = t.reshape(b * h, tlen, d)
                # Per-(head, channel) absmax scale
                scale = t_.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)  # [B*H,1,D]
                # Map to E4M3 dynamic range approx: use 240 levels (~8-bit) symmetric
                q = torch.clamp(torch.round((t_ / scale) * 240.0), -240, 240)
                deq = (q / 240.0) * scale
                return deq.reshape(b, h, tlen, d).to(t.dtype)

            key_states = fake_fp8_quant(key_states)
            value_states = fake_fp8_quant(value_states)
            return key_states, value_states

        # Fallback to 8/4-bit linear quantization per-tensor
        quant_bits = 4 if seq_len > (self.cache_window_size * 2) else 8
        key_states = self._quantize_tensor(key_states, quant_bits)
        value_states = self._quantize_tensor(value_states, quant_bits)
        return key_states, value_states
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Linear quantization per-tensor with symmetric range."""
        if bits >= 16:
            return tensor
        max_val = tensor.abs().max()
        scale = (max_val / (2**(bits - 1) - 1)).clamp(min=1e-8)
        q = torch.clamp(torch.round(tensor / scale), min=-(2**(bits - 1)), max=(2**(bits - 1) - 1))
        return (q * scale).to(tensor.dtype)
    
    def clear_cache(self):
        """Clear all caches"""
        self.kv_cache.clear()
        self.generation_cache.clear()
        self.speculative_cache.clear()
        self.h2o_cache.clear()
        
        # Reset statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        # Per-layer KV statistics
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

def pisces_init_weights(m):
    """
    Initialize weights for PyTorch modules.
    Specifically, it initializes the weights of nn.Linear using Kaiming uniform initialization
    and the weights of nn.Embedding using normal distribution.

    Args:
        m (torch.nn.Module): PyTorch module to initialize weights for.
    """
    if isinstance(m, nn.Linear):
        # Initialize linear layer weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            # Initialize bias to zero if it exists
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        # Initialize embedding layer weights using normal distribution
        nn.init.normal_(m.weight, mean=0, std=0.02)

class RMSNorm(nn.Module):
    """
    RMS normalization layer. Normalizes the input using Root Mean Square normalization.
    This normalization method helps stabilize the training process by normalizing the input features.
    """
    def __init__(self, dim, eps=1e-6):
        """
        Initialize the RMSNorm layer.

        Args:
            dim (int): Dimension of the input tensor.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        # Learnable weight parameter
        self.weight = nn.Parameter(torch.ones(dim))
        # Small value to avoid division by zero
        self.eps = eps

    def forward(self, x):
        """
        Forward pass of the RMSNorm layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Compute the root mean square of the input
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * rms

class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding module. Adds rotary positional embeddings to the input.
    Rotary embeddings help the model capture positional information in sequences.
    """
    def __init__(self, dim, max_seq_len=8192, base=1e6, device=None, dtype=None):
        """
        Initialize the RotaryEmbedding layer.

        Args:
            dim (int): Dimension of the input tensor.
            max_seq_len (int, optional): Maximum sequence length. Defaults to 8192.
            base (float, optional): Base value for frequency calculation. Defaults to 1e6.
            device (torch.device, optional): Device to place the tensors on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to None.
        """
        super().__init__()
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        # Create sequence positions
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Register cosine and sine values as buffers
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x, seq_len):
        """
        Forward pass of the RotaryEmbedding layer.

        Args:
            x (torch.Tensor): Input tensor.
            seq_len (int): Current sequence length.

        Returns:
            torch.Tensor: Tensor with rotary positional embeddings applied.
        """
        # Get cosine and sine values for the current sequence length
        cos, sin = self.cos[:seq_len], self.sin[:seq_len]
        # Split the input tensor into two parts
        x1, x2 = x[..., ::2], x[..., 1::2]
        # Apply rotary embeddings
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class Attention(nn.Module):
    """
    Multi-head attention module with H2O support for ultra-long contexts.
    Supports both standard attention and H2O attention based on sequence length.
    """
    def __init__(self, cfg, device=None, dtype=None):
        """
        Initialize the Attention layer with H2O support.

        Args:
            cfg (PiscesConfig): Configuration object.
            device (torch.device, optional): Device to place the tensors on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to None.
        """
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.hidden_size // cfg.n_head
        self.scale = self.head_dim ** -0.5
        
        # Use H2O attention for ultra-long contexts (explicit flag or very long context)
        self.use_h2o = bool(getattr(cfg, 'use_h2o_attention', False)) or (cfg.max_position_embeddings > 1000000)
        
        if self.use_h2o:
            self.h2o_attention = H2OAttention(
                hidden_size=cfg.hidden_size,
                num_attention_heads=cfg.n_head,
                max_position_embeddings=cfg.max_position_embeddings,
                compression_ratio=getattr(cfg, 'compression_ratio', 8),
                streaming_window=getattr(cfg, 'streaming_window', 16384),
                dropout=getattr(cfg, 'attention_dropout', 0.0)
            )
        else:
            # Standard attention components
            # Optional fused QKV projection for better kernel efficiency
            self.fused_qkv = bool(getattr(cfg, 'fused_qkv', False))
            if self.fused_qkv:
                qkv_out = (cfg.n_head + 2 * cfg.n_kv_head) * self.head_dim
                self.qkv_proj = nn.Linear(cfg.hidden_size, qkv_out, bias=False, device=device, dtype=dtype)
            else:
                self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.head_dim, bias=False, device=device, dtype=dtype)
                self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
                self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
            self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.hidden_size, bias=False, device=device, dtype=dtype)
            self.rope = YaRNRotaryEmbedding(self.head_dim, cfg.max_position_embeddings, cfg.rope_theta, scale=32, device=device)
            self.attn_dropout = nn.Dropout(getattr(cfg, 'attention_dropout', 0.0))
            
        # Apply weight initialization
        self.apply(pisces_init_weights)

    def forward(self, x, mask, past_key_values=None, use_cache=False, cache_manager=None):
        """
        Forward pass of the Attention layer with H2O support.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Attention mask.
            past_key_values (tuple, optional): Cached key and value tensors from previous steps.
            use_cache (bool, optional): Whether to return cached key and value tensors. Defaults to False.

        Returns:
            torch.Tensor: Output tensor after attention operation.
            tuple: Cached key and value tensors (if use_cache=True).
        """
        if self.use_h2o:
            # Use H2O attention for ultra-long contexts
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
        
        # Standard attention（非 H2O）：使用 SDPA + 绝对索引因果遮罩，兼容 KV-Cache
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

        # Rotary embeddings
        q, k = self.rope(q, t), self.rope(k, t)

        # Append past KV if provided
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        seq_len = k.size(-2)

        # GQA broadcast
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Type alignment
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        # SDPA expects [B*H, T, D] x [B*H, S, D]
        q_ = q.reshape(b * self.n_head, t, self.head_dim)
        k_ = k.reshape(b * self.n_head, seq_len, self.head_dim)
        v_ = v.reshape(b * self.n_head, seq_len, self.head_dim)

        # 绝对索引因果：行位置 = (seq_len - t) + arange(t)
        base = seq_len - t
        row_pos = base + torch.arange(t, device=q.device)  # [t]
        key_pos = torch.arange(seq_len, device=q.device)    # [S]
        allowed2d = key_pos.view(1, -1) <= row_pos.view(-1, 1)  # [t, S]
        
        # 可选：滑动窗口局部注意力，在非 H2O 模式下限制注意力范围以提升长序列效率
        if bool(getattr(self.cfg, 'use_sliding_window', False)):
            win = int(getattr(self.cfg, 'streaming_window', 16384))
            if win > 0 and win < seq_len:
                # 仅允许关注最近 win 个 key（含当前）
                # 条件：key_pos >= row_pos - (win-1)
                lower_bound = (row_pos.view(-1, 1) - (win - 1))  # [t,1]
                local_allowed = key_pos.view(1, -1) >= lower_bound
                allowed2d = allowed2d & local_allowed
        disallow2d = ~allowed2d  # True means mask

        # 合并外部 attention mask（若提供）。支持 bool 或加性遮罩（-inf）
        if mask is not None:
            # 期望形状 [..., T, S]，截取对齐到当前窗口与全键长
            mask_slice = mask[:, :, -t:, :seq_len]
            if mask_slice.dtype == torch.bool:
                extra_disallow = ~mask_slice  # True=mask
            else:
                extra_disallow = mask_slice < -1e4
            # 广播 OR 合并
            disallow = disallow2d.view(1, 1, t, seq_len) | extra_disallow
            attn_mask = disallow.reshape(b * self.n_head, t, seq_len)
        else:
            attn_mask = disallow2d  # [t, S] 可被 SDPA 广播

        # Prefer flash backend when available
        try:
            from torch.backends.cuda import sdp_kernel as _sdp
            use_flash = torch.cuda.is_available() and bool(getattr(self.cfg, 'sdpa_prefer_flash', True))
        except Exception:
            use_flash = False
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
        out = out_.reshape(b, self.n_head, t, self.head_dim).transpose(1, 2).contiguous().view(b, t, -1)

        # Apply attention dropout
        out = self.attn_dropout(out)
        # Project output to the original feature dimension
        out = self.o_proj(out)
        
        if use_cache:
            # Return cached key and value tensors (before broadcasting)
            k_cache = k[:, :self.n_kv_head] if self.n_kv_head != self.n_head else k
            v_cache = v[:, :self.n_kv_head] if self.n_kv_head != self.n_head else v
            return out, (k_cache, v_cache)
        
        return out

class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MoE MLP, featuring Pre-Norm and residual scaling.
    This block combines multi-head attention and MoE MLP with pre-normalization and residual scaling
    to improve training stability and performance.
    """
    def __init__(self, cfg, device=None, dtype=None, quantization_config=None):
        """
        Initialize the TransformerBlock.

        Args:
            cfg (PiscesConfig): Configuration object.
            device (torch.device, optional): Device to place the tensors on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to None.
            quantization_config (object, optional): Configuration for quantization. Defaults to None.
        """
        super().__init__()
        # Initialize attention module
        self.attn = Attention(cfg, device=device, dtype=dtype)
        # Will be set by model after construction
        self.cache_manager = None
        self.layer_idx = -1
        
        # Use the new stable MoE configuration
        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            # Use the stable routing gate
            from .moe import MoELayer
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype, 
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            # Fall back to the dynamic MoE layer
            from model.moe_dynamic import DynamicMoELayer
            self.mlp = DynamicMoELayer(cfg, device=device, dtype=dtype)
            
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.norm2 = RMSNorm(cfg.hidden_size)
        # Pre-Norm layers for stability
        self.pre_norm1 = RMSNorm(cfg.hidden_size)
        self.pre_norm2 = RMSNorm(cfg.hidden_size)
        # Residual scaling for deep networks
        self.residual_scale = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)
        # Dropout for residual connections
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))
        # Gradient checkpointing flag
        self.use_checkpoint = getattr(cfg, 'use_gradient_checkpointing', True)
        # Adaptive gradient checkpointing configuration
        self.adaptive_checkpointing = getattr(cfg, 'adaptive_gradient_checkpointing', True)
        self.memory_threshold_high = getattr(cfg, 'memory_threshold_high', 0.85)  # 85% memory usage
        self.memory_threshold_low = getattr(cfg, 'memory_threshold_low', 0.60)   # 60% memory usage
        self.checkpoint_frequency = getattr(cfg, 'checkpoint_frequency', 1)  # Checkpoint every N layers initially
        self.current_checkpoint_freq = self.checkpoint_frequency
        # Store quantization config
        self.quantization_config = quantization_config

        # Apply mixed precision quantization based on layer importance
        if self.quantization_config is not None:
            try:
                import bitsandbytes as bnb
                
                # Determine quantization strategy based on layer position and importance
                layer_importance = self._get_layer_importance()
                
                def convert_linear_to_mixed_precision(module, layer_type='standard'):
                    """
                    Convert nn.Linear layers to mixed precision quantization.
                    
                    Args:
                        module (torch.nn.Module): The module to be converted.
                        layer_type (str): Type of layer for precision selection.
                    """
                    for name, child in module.named_children():
                        if isinstance(child, nn.Linear):
                            # Select quantization precision based on layer importance
                            if layer_importance == 'critical':
                                # Critical layers: 8-bit for better accuracy
                                new_mod = bnb.nn.Linear8bit(
                                    child.in_features,
                                    child.out_features,
                                    bias=child.bias is not None,
                                    threshold=getattr(self.quantization_config, 'bnb_8bit_threshold', 6.0),
                                )
                            elif layer_importance == 'important':
                                # Important layers: 4-bit with better compute dtype
                                new_mod = bnb.nn.Linear4bit(
                                    child.in_features,
                                    child.out_features,
                                    bias=child.bias is not None,
                                    quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                                    compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.float16),  # Better precision
                                    compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                                )
                            else:
                                # Standard layers: 4-bit for memory efficiency
                                new_mod = bnb.nn.Linear4bit(
                                    child.in_features,
                                    child.out_features,
                                    bias=child.bias is not None,
                                    quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                                    compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.bfloat16),
                                    compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                                )
                            setattr(module, name, new_mod)
                        else:
                            # Recursively process child modules with appropriate layer type
                            child_layer_type = self._get_child_layer_type(name, layer_type)
                            convert_linear_to_mixed_precision(child, child_layer_type)
                
                convert_linear_to_mixed_precision(self)
                
            except Exception as e:
                ERROR(f"Mixed precision quantization failed: {e}")
                # Fallback to original 4-bit quantization
                self._fallback_to_4bit_quantization()

    def forward(self, x, mask, past_key_values=None, use_cache=False):
        """
        Forward pass of the TransformerBlock with Pre-Norm, residual scaling, gradient checkpointing, and KV-Cache.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Attention mask.
            past_key_values (tuple, optional): Cached key and value tensors from previous steps.
            use_cache (bool, optional): Whether to return cached key and value tensors. Defaults to False.

        Returns:
            tuple: Output tensor, auxiliary loss, and cached key and value tensors (if use_cache=True).
        """
        # Delegate to checkpointing helper to ensure a single return path
        out = self._apply_with_checkpoint(x, mask, past_key_values, use_cache)
        return out
    
    def _get_layer_importance(self):
        """
        Determine layer importance based on position and configuration.
        
        Returns:
            str: Layer importance level ('critical', 'important', 'standard').
        """
        # Default to standard importance
        return getattr(self.quantization_config, 'layer_importance', 'standard')
    
    def _get_child_layer_type(self, child_name, parent_type):
        """
        Determine child layer type based on name and parent type.
        
        Args:
            child_name (str): Name of the child module.
            parent_type (str): Type of the parent layer.
            
        Returns:
            str: Child layer type.
        """
        # Special handling for attention and MLP layers
        if 'attn' in child_name.lower() or 'attention' in child_name.lower():
            return 'critical'
        elif 'mlp' in child_name.lower() or 'feedforward' in child_name.lower():
            return 'important'
        else:
            return parent_type
    
    def _fallback_to_4bit_quantization(self):
        """
        Fallback to original 4-bit quantization when mixed precision fails.
        """
        try:
            import bitsandbytes as bnb
            
            def convert_linear_to_4bit(module):
                """
                Convert all nn.Linear layers in the module to 4-bit quantization layers.
                """
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        new_mod = bnb.nn.Linear4bit(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                            compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.bfloat16),
                            compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                        )
                        setattr(module, name, new_mod)
                    else:
                        convert_linear_to_4bit(child)
            
            convert_linear_to_4bit(self)
            try:
                from utils import RIGHT
                RIGHT("Fallback to 4-bit quantization successful")
            except Exception:
                pass
        except Exception as e:
            ERROR(f"Fallback 4-bit quantization also failed: {e}")
        # No forward logic here; this method only converts modules.
    
    def _should_use_checkpoint(self):
        """
        Determine whether to use gradient checkpointing based on memory usage.
        
        Returns:
            bool: Whether to use gradient checkpointing.
        """
        if not self.use_checkpoint or not self.adaptive_checkpointing:
            return self.use_checkpoint
            
        try:
            # Get current GPU memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                
                # Calculate memory usage percentage
                memory_usage = allocated / total_memory
                
                # Adjust checkpoint frequency based on memory usage
                if memory_usage > self.memory_threshold_high:
                    # High memory usage - increase checkpoint frequency
                    self.current_checkpoint_freq = max(1, self.checkpoint_frequency // 2)
                    return True
                elif memory_usage < self.memory_threshold_low:
                    # Low memory usage - decrease checkpoint frequency
                    self.current_checkpoint_freq = self.checkpoint_frequency * 2
                    return False
                else:
                    # Medium memory usage - use default frequency
                    self.current_checkpoint_freq = self.checkpoint_frequency
                    return (self.checkpoint_frequency <= 1) or (torch.randint(0, self.checkpoint_frequency, (1,)).item() == 0)
            else:
                # CPU training - use default checkpoint setting
                return self.use_checkpoint
                
        except Exception as e:
            # Fallback to default checkpoint setting if memory check fails
            ERROR(f"Adaptive checkpointing memory check failed: {e}")
            return self.use_checkpoint

    # Ensure the forward applies checkpointing/caching using the local helper
    # (This block was previously misplaced)
    def _apply_with_checkpoint(self, x, mask, past_key_values=None, use_cache=False):
        import torch.utils.checkpoint as cp
        attn_past_key_values = past_key_values if past_key_values is not None else None
        should_checkpoint = self._should_use_checkpoint()

        def _inner(xc, kv=None):
            return self._forward_core(xc, mask, kv, use_cache)

        if should_checkpoint and self.training:
            out = cp.checkpoint(_inner, x, attn_past_key_values, use_reentrant=False)
        else:
            out = _inner(x, attn_past_key_values)
        return out

    def _forward_core(self, x, mask, attn_past_key_values=None, use_cache=False):
        # Use the same logic as _forward_pass defined in forward
        residual = x
        x_norm = self.pre_norm1(x)
        attn_cache = None
        # Fetch past from cache manager if enabled
        past_for_attn = attn_past_key_values
        if use_cache and self.cache_manager is not None and self.layer_idx >= 0:
            got = self.cache_manager.get_kv_cache(self.layer_idx, attn_past_key_values)
            if got is not None:
                past_for_attn = got
        if use_cache:
            attn_out, present_kv = self.attn(
                x_norm,
                mask,
                past_key_values=past_for_attn,
                use_cache=True,
                cache_manager=self.cache_manager
            )
            # Update paged KV in manager
            if self.cache_manager is not None and self.layer_idx >= 0 and present_kv is not None:
                self.cache_manager.update_kv_cache(
                    self.layer_idx, present_kv[0], present_kv[1], current_pos=x_norm.shape[1], use_h2o=getattr(self.attn, 'use_h2o', False)
                )
            attn_cache = present_kv
        else:
            attn_out = self.attn(
                x_norm,
                mask,
                past_key_values=past_for_attn,
                use_cache=False,
                cache_manager=self.cache_manager
            )
        x_out = residual + self.residual_dropout(self.residual_scale * attn_out)
        x_out = self.norm1(x_out)
        residual = x_out
        x_norm = self.pre_norm2(x_out)
        mlp_out, aux_loss = self.mlp(x_norm)
        x_out = residual + self.residual_dropout(self.residual_scale * mlp_out)
        x_out = self.norm2(x_out)
        if use_cache:
            return x_out, aux_loss, attn_cache
        return x_out, aux_loss

class PiscesModel(nn.Module):
    """Main Pisces L1 multimodal model (Arctic architecture) with enhanced stability mechanisms.
    
    Features:
    - Pre-Norm architecture for improved training stability
    - Residual scaling for deep networks
    - Gradient checkpointing for memory efficiency
    - Attention dropout and residual dropout
    - Dual normalization approach (Pre-Norm + Post-Norm)
    """

    # Exclude the nested PiscesAgent from PyTorch module traversal to avoid
    # infinite recursion when calling `.to()`, `.cuda()`, `.state_dict()`, etc.
    def named_children(self):
        for name, module in super().named_children():
            if name == "agent":
                continue
            yield name, module

    def __init__(self, cfg, device=None, dtype=None, quantization_config=None, lora_config=None):
        """
        Initialize the PiscesModel.

        Args:
            cfg (PiscesConfig): Configuration object.
            device (torch.device, optional): Device to place the tensors on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to None.
            quantization_config (object, optional): Configuration for quantization. Defaults to None.
            lora_config (object, optional): Configuration for LoRA. Defaults to None.
        """
        super().__init__()
        DEBUG("PiscesModel: __init__ start")
        self.cfg = cfg
        # Expose a transformers-style alias for downstream utilities
        self.config = cfg
        # Normalize common field names expected by helper modules
        if not hasattr(self.config, 'num_layers'):
            setattr(self.config, 'num_layers', getattr(self.config, 'n_layer', 0))
        if not hasattr(self.config, 'num_heads'):
            setattr(self.config, 'num_heads', getattr(self.config, 'n_head', 0))
        if not hasattr(self.config, 'n_kv_head'):
            setattr(self.config, 'n_kv_head', getattr(self.config, 'n_kv_head', getattr(self.config, 'n_head', 0)))
        # Prefer explicit long-context flag when context is extremely large
        if getattr(self.config, 'max_position_embeddings', 0) >= 1_048_576 and not hasattr(self.config, 'use_h2o_attention'):
            setattr(self.config, 'use_h2o_attention', True)
        self.quantization_config = quantization_config
        self.lora_config = lora_config
        
        # Initialize unified cache manager with configuration
        cache_config = getattr(cfg, 'cache_config', {
            "enabled": True,
            "kv_cache_max_size": 2048,
            "h2o_cache_max_size": 1024,
            "generation_cache_max_size": 512,
            "speculative_cache_max_size": 256,
            "quantization_enabled": True,
            "dynamic_quantization": True,
            "cache_eviction_policy": "lru"
        })
        self.cache_manager = UnifiedCacheManager(cache_config)
        
        DEBUG("PiscesModel: initializing embedding...")
        # Initialize token embedding layer
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=dtype)
        DEBUG(f"PiscesModel: initializing {cfg.n_layer} transformer layers...")
        self.layers = nn.ModuleList([])
        for i in range(cfg.n_layer):
            if (i % 4 == 0) or (i == cfg.n_layer-1):
                DEBUG(f"PiscesModel: initializing TransformerBlock {i+1}/{cfg.n_layer}")
            # Add transformer blocks to the module list
            block = TransformerBlock(cfg, device=device, dtype=dtype, quantization_config=self.quantization_config)
            block.cache_manager = self.cache_manager
            block.layer_idx = i
            self.layers.append(block)
        DEBUG("PiscesModel: initializing norm...")
        # Initialize final normalization layer
        self.norm = RMSNorm(cfg.hidden_size)
        DEBUG("PiscesModel: initializing multimodal encoders...")
        # Use unified VisionEncoder with NaViT native resolution support
        self.vision = VisionEncoder(cfg)
        self.video = VideoEncoder(cfg)
        self.audio = AudioEncoder(cfg)
        self.doc = DocEncoder(cfg)
        
        # Agent encoder for behavior/policy modality - now using unified AgentEncoder
        self.agent_encoder = AgentEncoder(cfg)
        
        # Initialize dynamic modal fusion layer (share unified cache manager for consistent memory behavior)
        self.modal_fusion = DynamicModalFusion(cfg)
        DEBUG("PiscesModel: initializing output heads...")
        # Initialize language model head
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=device, dtype=dtype)
        # Initialize task head
        self.task_head = nn.Linear(cfg.hidden_size, cfg.task_classes, device=device, dtype=dtype)
        # Initialize evaluation head
        self.eval_head = nn.Linear(cfg.hidden_size, cfg.eval_dims, device=device, dtype=dtype)

        # Projection for shaping fused multimodal features into token space
        self.modal_token_count = getattr(cfg, 'modal_token_count', 8)
        self.fusion_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)
        
        DEBUG("PiscesModel: initializing reasoner...")
        # Initialize reasoner
        self.reasoner = PiscesReasoner(cfg)
        # Initialize reasoning tokens for multi-path reasoning
        self.reasoner.initialize_reasoning_tokens(None)
        
        # Multi-modal reasoning enhancer for Arctic architecture
        DEBUG("PiscesModel: initializing multi-modal reasoning enhancer...")
        self.mm_reasoning_enhancer = MultiModalReasoningEnhancer(cfg)
        
        DEBUG("PiscesModel: initializing agent...")
        from .multimodal import PiscesAgent
        self.agent = PiscesAgent(cfg, model=self)
        
        # Skipped global weight initialization to avoid lengthy CPU-bound stall.
        # self.apply(pisces_init_weights)
        DEBUG("PiscesModel: skipped duplicate global weight initialization")
        
        # Initialize speculative decoder for efficient generation
        DEBUG("PiscesModel: initializing speculative decoder...")
        self.speculative_config = SpeculativeConfig(
            num_candidates=getattr(cfg, 'speculative_candidates', 4),
            draft_length=getattr(cfg, 'speculative_draft_length', 5),
            acceptance_threshold=getattr(cfg, 'speculative_acceptance_threshold', 0.8),
            temperature=getattr(cfg, 'speculative_temperature', 0.7),
            top_k=getattr(cfg, 'speculative_top_k', 50),
            top_p=getattr(cfg, 'speculative_top_p', 0.9)
        )
        self.speculative_decoder = AdaptiveSpeculativeDecoder(
            self.speculative_config, self, None
        )
        DEBUG("PiscesModel: speculative decoder initialized")
        
        if lora_config is not None:
            try:
                from peft import get_peft_model
                self = get_peft_model(self, lora_config)
                DEBUG("PiscesModel: LoRA adapters injected (peft)")
            except Exception as e:
                ERROR(f"LoRA injection failed: {e}")
        total_params = sum(p.numel() for p in self.parameters())
        DEBUG(f"PiscesModel: total parameters = {total_params/1e6:.2f}M")
        DEBUG("PiscesModel: __init__ end")

    def set_gradient_checkpointing(self, enabled: bool = True):
        """
        Enable or disable gradient checkpointing for memory efficiency.
        
        Args:
            enabled (bool): Whether to enable gradient checkpointing. Defaults to True.
        """
        for layer in self.layers:
            layer.use_checkpoint = enabled
        
    def resize_token_embeddings(self, new_num_tokens):
        """
        Resizes token embeddings and associated heads to accommodate a new vocabulary size.

        Args:
            new_num_tokens (int): New vocabulary size.
        """
        # 1. Resize main token embedding
        old_embed = self.embed
        new_embed = nn.Embedding(new_num_tokens, self.cfg.hidden_size, device=old_embed.weight.device, dtype=old_embed.weight.dtype)
        
        # Copy old weights
        num_to_copy = min(old_embed.num_embeddings, new_num_tokens)
        new_embed.weight.data[:num_to_copy, :] = old_embed.weight.data[:num_to_copy, :]
        self.embed = new_embed

        # 2. Resize LM head
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(self.cfg.hidden_size, new_num_tokens, bias=False, device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        new_lm_head.weight.data[:num_to_copy, :] = old_lm_head.weight.data[:num_to_copy, :]
        self.lm_head = new_lm_head

        # 3. Resize reasoner's thinking head
        self.reasoner.resize_vocab(new_num_tokens)
        
        # 4. Update config
        self.cfg.vocab_size = new_num_tokens
        
        # 5. Reinitialize reasoner tokens with new vocabulary size
        self.reasoner.initialize_reasoning_tokens(None)
        
        # Note: The 'RIGHT' function is not defined, assuming it's a logging function
        try:
            from utils.log import RIGHT
            RIGHT(f"Resized token embeddings to {new_num_tokens}. Remember to update special token IDs in the reasoner.")
        except ImportError:
            pass

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True, **kwargs):
        """
        Prepare inputs for text generation with KV-Cache support, compatible with PEF/Transformers generation interface.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.
            past_key_values (list, optional): Cached key and value tensors from previous steps.
            use_cache (bool, optional): Whether to use and return KV-Cache. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary containing model inputs for generation with KV-Cache support.
        """
        model_inputs = {"input_ids": input_ids}
        
        # Add attention_mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        model_inputs["attention_mask"] = attention_mask
        
        # Add position_ids if not provided
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        model_inputs["position_ids"] = position_ids
        
        # Add KV-Cache support
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            
        model_inputs["use_cache"] = use_cache
        
        # Include other kwargs
        model_inputs.update(kwargs)
        return model_inputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_speculative: bool = True,
        mode: str = 'auto',  # 'auto' | 'fast' | 'thinking'
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Unified generation interface supporting both speculative and standard decoding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            use_speculative: Whether to use speculative decoding
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs and generation statistics
        """
        # Simple thinking router
        routing = 'fast'
        if mode == 'thinking':
            routing = 'thinking'
        elif mode == 'auto':
            seq_len = input_ids.shape[1]
            # Heuristic: long prompts or high top_k/p use thinking; else fast
            if seq_len > 256 or top_k >= 50 or top_p >= 0.9:
                routing = 'thinking'
        else:
            routing = 'fast'

        # Adjust decoding params for thinking mode
        use_speculative_final = use_speculative
        temperature_final = temperature
        top_k_final = top_k
        top_p_final = top_p
        if routing == 'thinking':
            use_speculative_final = True
            temperature_final = max(0.6, temperature * 0.9)
            top_k_final = max(50, top_k)
            top_p_final = max(0.9, top_p)

        if use_speculative_final and hasattr(self, 'speculative_decoder'):
            # Update speculative config with current parameters
            self.speculative_config.temperature = temperature_final
            self.speculative_config.top_k = top_k_final
            self.speculative_config.top_p = top_p_final
            
            # Use speculative decoder for efficient generation
            out_ids, stats = self.speculative_decoder.speculative_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                cache_manager=self.cache_manager if hasattr(self, 'cache_manager') else None,
                **kwargs
            )
            stats['routing'] = routing
            return out_ids, stats
        else:
            # Standard autoregressive generation
            out_ids, stats = self._standard_generate(
                input_ids, attention_mask, max_length,
                temperature_final, top_k_final, top_p_final, **kwargs
            )
            stats['routing'] = routing
            return out_ids, stats
            
    def _standard_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Standard autoregressive generation (fallback method)
        """
        device = input_ids.device
        generated_ids = input_ids.clone()
        
        stats = {
            'total_draft_tokens': 0,
            'accepted_tokens': 0,
            'rejected_tokens': 0,
            'draft_acceptance_rate': 0.0,
            'speedup': 1.0,
            'method': 'standard'
        }
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Prepare inputs
                model_inputs = self.prepare_inputs_for_generation(
                    generated_ids, attention_mask, **kwargs
                )
                
                # Forward pass
                outputs = self(**model_inputs)
                logits = outputs.get('logits', outputs) if isinstance(outputs, dict) else outputs
                
                # Sample next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
                    
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask, 
                        torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                    ], dim=-1)
                
                # Check for EOS token
                if next_token.item() == getattr(self.cfg, 'eos_token_id', 2):
                    break
                    
        return generated_ids, stats

    def forward(self, input_ids, images=None, audio=None, video=None, docs=None, labels=None, agent_mode=False, task=None, max_steps=None, agent_obs=None, agent_embed=None, past_key_values=None, use_cache=False):
        """
        Forward pass of the PiscesModel with full multimodal support including Agent and KV-Cache.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len].
            images (torch.Tensor, optional): Input images [batch_size, channels, height, width]. Defaults to None.
            audio (torch.Tensor, optional): Input audio [batch_size, channels, time_steps]. Defaults to None.
            video (torch.Tensor, optional): Input video [batch_size, channels, frames, height, width]. Defaults to None.
            docs (torch.Tensor, optional): Input documents [batch_size, seq_len, hidden_size]. Defaults to None.
            labels (torch.Tensor, optional): Ground truth labels. Defaults to None.
            agent_mode (bool, optional): Enable agent mode. Defaults to False.
            task (str, optional): Task description for agent mode. Defaults to None.
            max_steps (int, optional): Maximum steps for agent execution. Defaults to None.
            agent_obs (torch.Tensor, optional): Agent observations [batch_size, seq_len, hidden_size]. Defaults to None.
            agent_embed (torch.Tensor, optional): Pre-computed agent embeddings [batch_size, seq_len, hidden_size]. Defaults to None.
            past_key_values (list, optional): List of cached key and value tensors for each layer. Defaults to None.
            use_cache (bool, optional): Whether to return cached key and value tensors. Defaults to False.

        Returns:
            dict: Dictionary containing model outputs and optionally cached key and value tensors.
        """
        import torch.utils.checkpoint as cp
        import torch
        
        # Agent mode: delegate to agent
        if agent_mode:
            return self.agent.run(
                input_ids=input_ids,
                images=images,
                audio=audio,
                docs=docs,
                task=task,
                max_steps=max_steps
            )
            
        # Get batch size and sequence length
        b, t = input_ids.shape
        
        # Process text embeddings
        text_emb = self.embed(input_ids)
        
        # Process multimodal features
        modal_features = {}
        modal_features['text'] = text_emb
        
        if images is not None:
            img_out = self.vision(images)
            modal_features['image'] = img_out['features'] if isinstance(img_out, dict) and 'features' in img_out else img_out
        if audio is not None:
            aud_out = self.audio(audio)
            modal_features['audio'] = aud_out['features'] if isinstance(aud_out, dict) and 'features' in aud_out else aud_out
        if video is not None:
            vid_out = self.video(video)
            modal_features['video'] = vid_out['features'] if isinstance(vid_out, dict) and 'features' in vid_out else vid_out
        if docs is not None:
            doc_out = self.doc(docs)
            modal_features['doc'] = doc_out['features'] if isinstance(doc_out, dict) and 'features' in doc_out else doc_out
            
        # Process Agent modality if provided
        if agent_embed is not None:
            # Prepare comprehensive agent input for AgentEncoder
            agent_input = {
                'observations': agent_embed.get('observations', []),
                'actions': agent_embed.get('actions', []),
                'reflections': agent_embed.get('reflections', []),
                'current_state': agent_embed.get('current_state', None),
                'task_context': agent_embed.get('task_context', None)
            }
            modal_features['agent'] = self.agent_encoder(agent_input)
            
        if agent_obs is not None:
            # Prepare comprehensive agent observation input
            agent_obs_input = {
                'observations': agent_obs.get('observations', []),
                'actions': agent_obs.get('actions', []),
                'reflections': agent_obs.get('reflections', []),
                'current_state': agent_obs.get('current_state', None),
                'task_context': agent_obs.get('task_context', None)
            }
            agent_feat = self.agent_encoder(agent_obs_input)
            modal_features['agent'] = agent_feat

        # Enhanced multimodal fusion
        if len(modal_features) > 1:
            # Use dynamic fusion for multiple modalities
            fused_features = self.modal_fusion(modal_features)
            # Shape fused features into token sequence if necessary
            if fused_features is None:
                x = text_emb
            elif fused_features.dim() == 3:
                # Ensure dtype/device consistency before concat
                if fused_features.dtype != text_emb.dtype:
                    fused_features = fused_features.to(text_emb.dtype)
                if fused_features.device != text_emb.device:
                    fused_features = fused_features.to(text_emb.device)
                x = torch.cat([fused_features, text_emb], dim=1)
            elif fused_features.dim() == 2:
                # [B, H] -> [B, N, H]
                B, H = fused_features.shape
                # Project and expand with dtype/device aligned to text embeddings
                ff = fused_features.to(device=text_emb.device, dtype=text_emb.dtype)
                proj = self.fusion_proj(ff)
                tokens = proj.unsqueeze(1).expand(B, self.modal_token_count, H).contiguous()
                x = torch.cat([tokens, text_emb], dim=1)
            else:
                # Fallback to text only if unexpected shape
                x = text_emb
        else:
            # Single modality, use text only
            x = text_emb
            
        t = x.shape[1]
        
        # Original sequence length for LM loss calculation
        lm_seq_len = x.shape[1]

        # Create causal mask
        mask = torch.full((t, t), float('-inf'), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        total_aux_loss = 0.0
        
        # Use unified cache manager for cache handling
        chunk_size = min(getattr(self.cfg, 'max_position_embeddings', 2048), 8192)
        outputs = []
        
        # Configure cache manager based on sequence length
        if use_cache:
            seq_len = x.shape[1]
            if seq_len > 1024:  # Use 4-bit quantization for long sequences
                cache_dtype = torch.float16
                cache_quant_bits = 4
            elif seq_len > 512:  # Use 8-bit quantization for medium sequences
                cache_dtype = torch.float16  
                cache_quant_bits = 8
            else:  # Use full precision for short sequences
                cache_dtype = torch.float32
                cache_quant_bits = 16
        else:
            cache_dtype = torch.float32
            cache_quant_bits = 16
            
        autocast_ctx = torch.amp.autocast("cuda", dtype=cache_dtype)
        with autocast_ctx:
            # Initialize KV-Cache storage
            next_cache = [] if use_cache else None
            
            for i in range(0, x.shape[1], chunk_size):
                x_chunk = x[:, i:i+chunk_size, ...]
                mask_chunk = mask[i:i+chunk_size, i:i+chunk_size]
                
                def block_fn(xc, msk, layer_past_key_values=None):
                    """
                    Helper function for gradient checkpointing. Applies all transformer layers with KV-Cache support.

                    Args:
                        xc (torch.Tensor): Chunk of the input tensor.
                        msk (torch.Tensor): Chunk of the attention mask.
                        layer_past_key_values (list, optional): List of KV-Cache for each transformer layer.

                    Returns:
                        tuple: Output tensor, accumulated auxiliary loss, and updated KV-Cache.
                    """
                    h = xc
                    aux = 0.0
                    new_caches = []
                    
                    for layer_idx, layer in enumerate(self.layers):
                        # Use unified cache manager for KV cache
                        past_kv = self.cache_manager.get_kv_cache(layer_idx, layer_past_key_values[layer_idx] if layer_past_key_values is not None else None)
                        
                        # Dynamically dequantize past_key_value
                        if past_kv is not None and cache_quant_bits < 16:
                            past_kv = tuple(
                                tensor.to(cache_dtype) if tensor is not None else None 
                                for tensor in past_kv
                            )
                        
                        if use_cache:
                            h, aux_loss, cache = layer(h, msk, past_key_values=past_kv, use_cache=True)
                            
                            # Update cache manager with new cache
                            if cache is not None:
                                key_states, value_states = cache
                                updated_key, updated_value = self.cache_manager.update_kv_cache(
                                    layer_idx, key_states, value_states, i + xc.shape[1], 
                                    use_h2o=getattr(self.cfg, 'use_h2o_attention', False)
                                )
                                cache = (updated_key, updated_value)
                                
                                # Dynamically quantize the new KV-Cache
                                if cache_quant_bits < 16:
                                    cache = tuple(
                                        tensor.to(torch.float16) if tensor is not None else None
                                        for tensor in cache
                                    )
                            new_caches.append(cache)
                        else:
                            h, aux_loss = layer(h, msk, past_key_values=past_kv, use_cache=False)
                            
                        aux = aux + aux_loss if aux_loss is not None else aux
                    
                    if use_cache:
                        return h, aux, new_caches
                    return h, aux, None
                
                if use_cache:
                    # Avoid torch.checkpoint here because past_key_values is non-tensor
                    h_chunk, aux_chunk, cache_chunk = block_fn(x_chunk, mask_chunk, past_key_values)
                    if next_cache is not None and cache_chunk is not None:
                        next_cache.extend(cache_chunk)
                else:
                    # Safe to checkpoint when we don't thread cache state
                    h_chunk, aux_chunk, _ = cp.checkpoint(block_fn, x_chunk, mask_chunk, None, use_reentrant=False)
                    
                outputs.append(h_chunk)
                total_aux_loss = total_aux_loss + aux_chunk
            if outputs:
                x = torch.cat(outputs, dim=1)
            
            if x.shape[1] == 0:
                # Handle empty sequences gracefully to prevent indexing errors in heads.
                return {
                    "logits": self.lm_head(x),
                    "loss": torch.tensor(0.0, device=x.device, requires_grad=True),
                    "task_logits": torch.zeros(x.shape[0], self.cfg.task_classes, device=x.device),
                    "eval_score": torch.zeros(x.shape[0], self.cfg.eval_dims, device=x.device),
                    "aux_loss": total_aux_loss,
                    "reasoner_out": {"loss": torch.tensor(0.0, device=x.device, requires_grad=True)}
                }

            # Apply final normalization
            x = self.norm(x)
            
            # Main model outputs
            logits = self.lm_head(x)
            
            # Reasoner outputs - align input_ids length with x
            reasoner_input_ids = input_ids[:, :x.shape[1]] if input_ids.shape[1] > x.shape[1] else input_ids
            reasoner_labels = labels[:, :x.shape[1]] if labels is not None and labels.shape[1] > x.shape[1] else labels
            reasoner_out = self.reasoner(x, reasoner_input_ids, reasoner_labels)

            loss = None
            if labels is not None:
                # Standard language modeling loss
                lm_loss = F.cross_entropy(
                    logits[:, :lm_seq_len, :].reshape(-1, logits.size(-1)), 
                    labels.view(-1)
                )
                
                # Combine with reasoner loss
                reasoner_loss = reasoner_out.get("loss", torch.tensor(0.0, device=x.device))
                loss = lm_loss + reasoner_loss

            # Compute MCP-ready tool trigger hints based on reasoning signals
            tool_trigger = None
            try:
                unc = reasoner_out.get("uncertainty_scores", None)
                fac = reasoner_out.get("fact_consistency", None)
                if unc is not None:
                    # Reduce across sequence if needed
                    unc_val = unc.mean().item() if unc.numel() > 0 else 0.0
                else:
                    unc_val = 0.0
                if fac is not None:
                    fac_val = fac.mean().item() if fac.numel() > 0 else 1.0
                else:
                    fac_val = 1.0
                unc_th = getattr(self.cfg, 'tool_uncertainty_threshold', 0.7)
                fac_th = getattr(self.cfg, 'tool_fact_consistency_threshold', 0.6)
                should_tool = (unc_val >= unc_th) or (fac_val <= fac_th)
                tool_trigger = {
                    'should_tool': bool(should_tool),
                    'uncertainty': float(unc_val),
                    'fact_consistency': float(fac_val),
                    'suggested_tools': ['search'] if should_tool else [],
                }
            except Exception:
                tool_trigger = None

            # Get task logits
            task_logits = self.task_head(x[:, 0])
            # Get evaluation score
            eval_score = self.eval_head(x.mean(1))

        # Assemble standardized MCP tool intent object (for MCP layer consumption)
        tool_intent = None
        try:
            if tool_trigger is not None and tool_trigger.get('should_tool', False):
                # A structured intent object providing context for MCP router
                tool_intent = {
                    'type': 'tool_intent',
                    'version': '1.0',
                    'triggers': {
                        'uncertainty': tool_trigger.get('uncertainty', 0.0),
                        'fact_consistency': tool_trigger.get('fact_consistency', 1.0),
                        'thresholds': {
                            'uncertainty': getattr(self.cfg, 'tool_uncertainty_threshold', 0.7),
                            'fact_consistency': getattr(self.cfg, 'tool_fact_consistency_threshold', 0.6)
                        }
                    },
                    'suggested_tools': tool_trigger.get('suggested_tools', []),
                    'confidence': float(min(1.0, max(0.0, tool_trigger.get('uncertainty', 0.0)))),
                    'reason': 'High uncertainty or low fact consistency detected by reasoner',
                }
        except Exception:
            tool_intent = None

        # Optional debug section
        debug_section = None
        if getattr(self.cfg, 'enable_debug_outputs', False):
            try:
                debug_section = {
                    'x_shape': tuple(x.shape),
                    'x_dtype': str(x.dtype),
                    'text_emb_shape': tuple(text_emb.shape),
                    'text_emb_dtype': str(text_emb.dtype),
                }
                if getattr(self.cfg, 'debug_verbose', False):
                    try:
                        # Check modality presence
                        present = {k: (v is not None) for k, v in modal_features.items()}
                        debug_section['modal_present'] = present
                    except Exception:
                        pass
            except Exception:
                debug_section = None

        # Assemble final outputs once
        result = {
            "logits": logits,
            "loss": loss,
            "task_logits": task_logits,
            "eval_score": eval_score,
            "aux_loss": total_aux_loss,
            "reasoner_out": reasoner_out,
            "tool_trigger": tool_trigger,
            "tool_intent": tool_intent
        }
        if debug_section is not None:
            result['debug'] = debug_section

        if use_cache:
            result["past_key_values"] = next_cache
        # Attach cache stats if available
        if hasattr(self, 'cache_manager') and self.cache_manager is not None:
            try:
                result["cache_stats"] = self.cache_manager.get_cache_stats()
            except Exception:
                pass
        
        return result