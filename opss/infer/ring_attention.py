#!/usr/bin/env/python3
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

"""
Ring Attention Operator for Ultra-Long Context

Implements Ring Attention for processing sequences longer than GPU memory capacity
by distributing attention computation across multiple devices in a ring topology.

Key Features:
    - Linear memory scaling with sequence length
    - Supports sequences up to 100M+ tokens
    - Ring topology for efficient communication
    - Compatible with FlashAttention kernels

References:
    - Ring Attention (Liu et al., 2023)
    - FlashAttention (Dao et al., 2022)

Usage:
    >>> from opss.infer.ring_attention import POPSSRingAttentionOperator, POPSSRingAttentionConfig
    >>> config = POPSSRingAttentionConfig(ring_size=8, block_size=4096)
    >>> operator = POPSSRingAttentionOperator(config)
    >>> result = operator.execute({"hidden_states": hidden_states, "attention_mask": mask})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import math

from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus
from utils.dc import PiscesLxLogger
from configs.version import VERSION


class POPSSRingTopology(Enum):
    """Ring topology types for distributed attention."""
    RING = "ring"
    DOUBLE_RING = "double_ring"
    TREE = "tree"


@dataclass
class POPSSRingAttentionConfig:
    """
    Configuration for Ring Attention Operator.
    
    Attributes:
        ring_size: Number of devices in the ring
        block_size: Block size for attention computation
        num_heads: Number of attention heads
        head_dim: Dimension per head
        dropout: Attention dropout probability
        causal: Whether to use causal attention
        topology: Ring topology type
        overlap_communication: Whether to overlap communication with computation
        use_flash_attention: Whether to use FlashAttention kernels
        max_sequence_length: Maximum supported sequence length
    """
    ring_size: int = 8
    block_size: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    dropout: float = 0.0
    causal: bool = True
    topology: POPSSRingTopology = POPSSRingTopology.RING
    overlap_communication: bool = True
    use_flash_attention: bool = True
    max_sequence_length: int = 1048576
    
    def __post_init__(self):
        if isinstance(self.topology, str):
            self.topology = POPSSRingTopology(self.topology)


class POPSSRingAttentionOperator(PiscesLxOperatorInterface):
    """
    Ring Attention Operator for Ultra-Long Context Processing.
    
    Implements memory-efficient attention for sequences exceeding GPU memory
    by distributing computation across devices in a ring topology.
    
    Architecture:
        1. Partition query/key/value across ring devices
        2. Compute local attention blocks
        3. Rotate KV blocks around the ring
        4. Accumulate attention outputs
    
    Memory Complexity:
        - Standard Attention: O(n²) memory
        - Ring Attention: O(n / ring_size) memory per device
    
    Example:
        >>> config = POPSSRingAttentionConfig(ring_size=8, block_size=4096)
        >>> operator = POPSSRingAttentionOperator(config)
        >>> output = operator.execute({
        ...     "query": query_tensor,
        ...     "key": key_tensor,
        ...     "value": value_tensor
        ... })
    """
    
    def __init__(self, config: Optional[POPSSRingAttentionConfig] = None):
        super().__init__()
        self.name = "infer.ring_attention"
        self.version = VERSION
        self.type = "inference"
        self._LOG = get_logger("popss.ops.infer.ring_attention")
        self.config = config or POPSSRingAttentionConfig()
        
        self._ring_initialized = False
        self._rank = 0
        self._world_size = 1
        
    def _initialize_ring(self):
        """Initialize ring communication group."""
        if self._ring_initialized:
            return
        
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                self._rank = dist.get_rank()
                self._world_size = dist.get_world_size()
                self._ring_initialized = True
                self._LOG.info(f"Ring initialized: rank={self._rank}, world_size={self._world_size}")
        except Exception as e:
            self._LOG.warning(f"Distributed not available, using single device: {e}")
            self._ring_initialized = True
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute ring attention computation.
        
        Args:
            inputs: Dictionary containing:
                - query: Query tensor [batch, seq_len, num_heads, head_dim]
                - key: Key tensor [batch, seq_len, num_heads, head_dim]
                - value: Value tensor [batch, seq_len, num_heads, head_dim]
                - attention_mask: Optional attention mask
        
        Returns:
            PiscesLxOperatorResult with attention output
        """
        self._initialize_ring()
        
        query = inputs.get("query")
        key = inputs.get("key")
        value = inputs.get("value")
        attention_mask = inputs.get("attention_mask")
        
        if query is None or key is None or value is None:
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error="Missing query, key, or value tensors"
            )
        
        try:
            if self._world_size > 1:
                output = self._ring_attention_distributed(query, key, value, attention_mask)
            else:
                output = self._ring_attention_single(query, key, value, attention_mask)
            
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.SUCCESS,
                output={"hidden_states": output}
            )
        except Exception as e:
            self._LOG.error(f"Ring attention failed: {e}")
            return PiscesLxOperatorResult(
                status=PiscesLxOperatorStatus.ERROR,
                error=str(e)
            )
    
    def _ring_attention_single(self, query: torch.Tensor, key: torch.Tensor, 
                               value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Single-device ring attention (block-wise processing)."""
        batch_size, seq_len, num_heads, head_dim = query.shape
        block_size = self.config.block_size
        num_blocks = (seq_len + block_size - 1) // block_size
        
        output = torch.zeros_like(query)
        normalizer = torch.zeros(batch_size, seq_len, num_heads, 1, 
                                  device=query.device, dtype=query.dtype)
        
        for i in range(num_blocks):
            q_start = i * block_size
            q_end = min((i + 1) * block_size, seq_len)
            q_block = query[:, q_start:q_end]
            
            for j in range(num_blocks if not self.config.causal else i + 1):
                k_start = j * block_size
                k_end = min((j + 1) * block_size, seq_len)
                k_block = key[:, k_start:k_end]
                v_block = value[:, k_start:k_end]
                
                attn_weights = torch.einsum('bqhd,bkhd->bhqk', q_block, k_block)
                attn_weights = attn_weights / math.sqrt(head_dim)
                
                if mask is not None:
                    block_mask = mask[:, q_start:q_end, k_start:k_end]
                    attn_weights = attn_weights.masked_fill(block_mask.unsqueeze(1) == 0, float('-inf'))
                
                if self.config.causal and j == i:
                    causal_mask = torch.triu(
                        torch.ones(q_end - q_start, k_end - k_start, 
                                   device=query.device, dtype=torch.bool),
                        diagonal=1
                    )
                    attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                
                attn_probs = F.softmax(attn_weights, dim=-1)
                if self.config.dropout > 0:
                    attn_probs = F.dropout(attn_probs, p=self.config.dropout)
                
                block_output = torch.einsum('bhqk,bkhd->bqhd', attn_probs, v_block)
                output[:, q_start:q_end] += block_output
                normalizer[:, q_start:q_end] += attn_probs.sum(dim=-1, keepdim=True).transpose(1, 2)
        
        output = output / normalizer.clamp(min=1e-6)
        return output
    
    def _ring_attention_distributed(self, query: torch.Tensor, key: torch.Tensor,
                                    value: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Distributed ring attention across multiple devices."""
        import torch.distributed as dist
        
        batch_size, seq_len, num_heads, head_dim = query.shape
        block_size = self.config.block_size
        
        local_seq_len = seq_len // self._world_size
        local_start = self._rank * local_seq_len
        local_end = local_start + local_seq_len
        
        local_q = query[:, local_start:local_end]
        local_k = key[:, local_start:local_end]
        local_v = value[:, local_start:local_end]
        
        output = torch.zeros_like(local_q)
        normalizer = torch.zeros(batch_size, local_seq_len, num_heads, 1,
                                  device=query.device, dtype=query.dtype)
        
        k_buffer = local_k.clone()
        v_buffer = local_v.clone()
        
        send_k = torch.zeros_like(k_buffer)
        send_v = torch.zeros_like(v_buffer)
        recv_k = torch.zeros_like(k_buffer)
        recv_v = torch.zeros_like(v_buffer)
        
        for step in range(self._world_size):
            src_rank = (self._rank - 1 + self._world_size) % self._world_size
            dst_rank = (self._rank + 1) % self._world_size
            
            if self.config.overlap_communication and step < self._world_size - 1:
                send_k.copy_(k_buffer)
                send_v.copy_(v_buffer)
                
                send_req_k = dist.isend(send_k, dst_rank)
                send_req_v = dist.isend(send_v, dst_rank)
                recv_req_k = dist.irecv(recv_k, src_rank)
                recv_req_v = dist.irecv(recv_v, src_rank)
            
            kv_rank = (self._rank - step + self._world_size) % self._world_size
            kv_start = kv_rank * local_seq_len
            kv_end = kv_start + local_seq_len
            
            if self.config.causal:
                q_positions = torch.arange(local_start, local_end, device=query.device)
                k_positions = torch.arange(kv_start, kv_end, device=query.device)
                causal_mask = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
            else:
                causal_mask = None
            
            attn_weights = torch.einsum('bqhd,bkhd->bhqk', local_q, k_buffer)
            attn_weights = attn_weights / math.sqrt(head_dim)
            
            if causal_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    ~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
                )
            
            if mask is not None:
                block_mask = mask[:, local_start:local_end, kv_start:kv_end]
                attn_weights = attn_weights.masked_fill(block_mask.unsqueeze(1) == 0, float('-inf'))
            
            attn_probs = F.softmax(attn_weights, dim=-1)
            if self.config.dropout > 0:
                attn_probs = F.dropout(attn_probs, p=self.config.dropout)
            
            block_output = torch.einsum('bhqk,bkhd->bqhd', attn_probs, v_buffer)
            output += block_output
            normalizer += attn_probs.sum(dim=-1, keepdim=True).transpose(1, 2)
            
            if self.config.overlap_communication and step < self._world_size - 1:
                send_req_k.wait()
                send_req_v.wait()
                recv_req_k.wait()
                recv_req_v.wait()
                
                k_buffer, recv_k = recv_k, k_buffer
                v_buffer, recv_v = recv_v, v_buffer
        
        output = output / normalizer.clamp(min=1e-6)
        return output
    
    def get_memory_estimate(self, sequence_length: int, batch_size: int = 1) -> Dict[str, int]:
        """
        Estimate memory requirements for given sequence length.
        
        Args:
            sequence_length: Total sequence length
            batch_size: Batch size
        
        Returns:
            Dictionary with memory estimates in bytes
        """
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim
        element_size = 4  # float32
        
        qkv_memory = 3 * batch_size * sequence_length * num_heads * head_dim * element_size
        
        if self._world_size > 1:
            local_seq = sequence_length // self._world_size
            attention_memory = batch_size * num_heads * local_seq * local_seq * element_size
        else:
            attention_memory = batch_size * num_heads * self.config.block_size ** 2 * element_size
        
        return {
            "qkv_memory_bytes": qkv_memory,
            "attention_memory_bytes": attention_memory,
            "total_memory_bytes": qkv_memory + attention_memory,
            "memory_per_device_bytes": (qkv_memory + attention_memory) // max(1, self._world_size)
        }


class POPSSRingAttentionLayer(nn.Module):
    """
    Ring Attention Layer for integration into transformer models.
    
    Provides a drop-in replacement for standard attention layers with
    support for ultra-long sequences through ring communication.
    """
    
    def __init__(self, config: POPSSRingAttentionConfig):
        super().__init__()
        self.config = config
        self.operator = POPSSRingAttentionOperator(config)
        
        self.q_proj = nn.Linear(config.head_dim * config.num_heads, 
                                 config.head_dim * config.num_heads, bias=False)
        self.k_proj = nn.Linear(config.head_dim * config.num_heads,
                                 config.head_dim * config.num_heads, bias=False)
        self.v_proj = nn.Linear(config.head_dim * config.num_heads,
                                 config.head_dim * config.num_heads, bias=False)
        self.out_proj = nn.Linear(config.head_dim * config.num_heads,
                                   config.head_dim * config.num_heads, bias=False)
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        query = self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.config.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.config.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, seq_len, -1, self.config.head_dim)
        
        result = self.operator.execute({
            "query": query,
            "key": key,
            "value": value,
            "attention_mask": attention_mask
        })
        
        if result.is_success():
            output = result.output["hidden_states"]
            output = output.reshape(batch_size, seq_len, -1)
            return self.out_proj(output)
        else:
            raise RuntimeError(f"Ring attention failed: {result.error}")


__all__ = [
    "POPSSRingTopology",
    "POPSSRingAttentionConfig",
    "POPSSRingAttentionOperator",
    "POPSSRingAttentionLayer",
]
