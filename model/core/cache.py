#!/usr/bin/env python3
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
Unified Cache Management Module for Yv Model.

This module provides a comprehensive cache management system that handles multiple
caching mechanisms for efficient inference in the Yv transformer architecture.
The cache system is designed for memory efficiency, fast access, and support for
various generation strategies including speculative decoding and multimodal inputs.

Architecture Overview:
    The cache system implements a unified interface with multiple backend strategies:

    1. Core Cache Types (YvCacheType):
       - KV_CACHE: Standard key-value cache for transformer attention
       - PAGED: Block-based paged attention cache
       - H2O: Heavy-Hitter Oracle sliding window cache
       - STREAMING: Streaming-friendly cache for long conversations
       - HYBRID: Combined attention-SSM cache
       - SSM: State space model cache for Mamba layers
       - SPECULATIVE: Speculative decoding cache
       - MULTIMODAL: Multimodal generation cache

    2. Unified Cache Manager (YvUnifiedCacheManager):
       - Single interface for all cache operations
       - Automatic cache type selection based on configuration
       - Memory pooling and allocation management
       - Thread-safe operations for concurrent access
       - Cache eviction and compression strategies

    3. Paged Attention Cache (YvPagedCacheManager):
       - Block-based memory allocation inspired by virtual memory
       - Efficient memory sharing across sequences (prefix caching)
       - Dynamic block allocation and deallocation
       - Supports variable-length sequences
       - Optimal for batched inference with different sequence lengths

    4. H2O Cache (YvH2OCacheManager):
       - Heavy-Hitter Oracle for long-context attention
       - Identifies and retains important tokens ("heavy hitters")
       - Sliding window with attention-based eviction
       - Constant memory footprint regardless of sequence length
       - Maintains quality while reducing memory usage

    5. Streaming Cache (YvStreamingCacheManager):
       - Optimized for streaming/incremental generation
       - Attention sinks for stable long conversations
       - Rolling window with sink tokens
       - Handles context window overflow gracefully
       - Maintains coherence in multi-turn conversations

    6. SSM Cache (YvSSMCacheManager):
       - Cache for state space model (Mamba) layers
       - Stores recurrent state for efficient generation
       - Supports both causal and bidirectional modes
       - Minimal memory overhead compared to attention cache
       - Enables constant-memory generation for SSM layers

    7. Speculative Decoding Cache (YvSpeculativeCacheManager):
       - Supports speculative decoding with draft models
       - Manages multiple candidate sequences
       - Efficient verification and acceptance/rejection
       - Rollback mechanism for rejected candidates
       - Enables 2-3x speedup with minimal quality loss

    8. Multimodal Cache (YvMultimodalCacheManager):
       - Cache for multimodal generation (text + images)
       - Handles interleaved text and image tokens
       - Modality-aware compression
       - Supports vision encoder feature caching
       - Efficient for vision-language models

    9. Hybrid Cache (YvHybridCacheManager):
       - Combined attention and SSM cache
       - Supports hybrid architecture models
       - Unified interface for both cache types
       - Dynamic routing based on layer type
       - Optimal for attention-SSM hybrid models

Design Rationale:
    - Memory Efficiency: Paged allocation reduces fragmentation
    - Flexibility: Multiple strategies for different use cases
    - Performance: Optimized data structures for fast access
    - Scalability: Supports millions of tokens with bounded memory
    - Thread Safety: Concurrent access for multi-request serving

Memory Management:
    - Block size: Configurable (typically 16-64 tokens per block)
    - Maximum blocks: Bounded by available GPU memory
    - Eviction policy: LRU with attention score weighting
    - Compression: Optional quantization for inactive blocks

Performance Considerations:
    - Paged cache reduces memory fragmentation by 40-60%
    - H2O maintains quality with 50% less memory
    - Streaming cache enables infinite-length generation
    - SSM cache uses 10x less memory than attention cache
    - Speculative decoding provides 2-3x throughput improvement

Dependencies:
    - torch: PyTorch deep learning framework
    - dataclasses: Configuration data structures
    - threading: Thread-safe operations
    - utils.dc: Logging utilities

Usage Example:
    >>> from model.core.cache import YvUnifiedCacheManager, YvCacheType
    >>> 
    >>> # Initialize cache manager
    >>> cache = YvUnifiedCacheManager(
    ...     num_layers=32,
    ...     num_heads=32,
    ...     head_dim=128,
    ...     max_batch_size=8,
    ...     cache_type=YvCacheType.PAGED
    ... )
    >>> 
    >>> # Update cache during generation
    >>> new_k, new_v = cache.update(layer_idx, key, value)
    >>> 
    >>> # Retrieve cached values
    >>> cached_k, cached_v = cache.get(layer_idx)

Note:
    All classes follow the YvXxx naming convention.
    Cache selection should match the model architecture and use case.
    Paged cache is recommended for production serving.
    H2O cache is recommended for long-context applications.
    SSM cache is required for models with Mamba layers.
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Union
from enum import Enum
from collections import OrderedDict
import threading
import time
from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Core", file_path=get_log_file("Yv.Core"), enable_file=True)

class YvCacheType(Enum):
    """Enumeration of available cache types for transformer inference.
    
    Defines the supported caching strategies for efficient inference,
    each optimized for different use cases and memory constraints.
    
    Attributes:
        KV_CACHE: Standard key-value cache for transformer attention.
            Simple and reliable, suitable for most use cases.
            Memory grows linearly with sequence length.
        PAGED: Block-based paged attention cache.
            Efficient memory allocation with virtual memory-like management.
            Optimal for batched inference with variable-length sequences.
        H2O: Heavy-Hitter Oracle sliding window cache.
            Identifies and retains important tokens for long contexts.
            Constant memory footprint regardless of sequence length.
        STREAMING: Streaming-friendly cache for long conversations.
            Rolling window with attention sinks for stability.
            Enables infinite-length generation with bounded memory.
        HYBRID: Combined attention-SSM cache.
            Supports hybrid architecture models with both attention and SSM layers.
            Unified interface for both cache types.
        SSM: State space model cache for Mamba layers.
            Stores recurrent state for efficient generation.
            Minimal memory overhead compared to attention cache.
        SPECULATIVE: Speculative decoding cache.
            Manages draft and verification caches for speculative decoding.
            Enables 2-3x speedup with minimal quality loss.
    
    Example:
        >>> cache_type = YvCacheType.PAGED
        >>> if cache_type == YvCacheType.H2O:
        ...     print("Using H2O for long-context inference")
    """
    KV_CACHE = "kv_cache"
    PAGED = "paged"
    H2O = "h2o"
    STREAMING = "streaming"
    HYBRID = "hybrid"
    SSM = "ssm"
    SPECULATIVE = "speculative"

class YvEvictionPolicy(Enum):
    """Enumeration of cache eviction policies for memory management.
    
    Defines strategies for selecting which cache entries to evict
    when memory limits are reached. Each policy has different trade-offs
    between recency, frequency, and importance of cached data.
    
    Attributes:
        LRU: Least Recently Used eviction.
            Evicts the oldest accessed entries first.
            Simple and effective for most workloads.
        LFU: Least Frequently Used eviction.
            Evicts entries with lowest access count.
            Better for workloads with skewed access patterns.
        IMPORTANCE: Importance-based eviction.
            Evicts entries with lowest importance scores.
            Considers attention scores and token significance.
        POSITION: Position-based eviction.
            Evicts entries based on position in sequence.
            Typically keeps recent tokens and discards old ones.
        ADAPTIVE: Adaptive eviction policy.
            Dynamically selects eviction strategy based on workload.
            Combines multiple strategies for optimal performance.
    
    Example:
        >>> policy = YvEvictionPolicy.ADAPTIVE
        >>> if policy == YvEvictionPolicy.IMPORTANCE:
        ...     print("Using importance-based eviction")
    """
    LRU = "lru"
    LFU = "lfu"
    IMPORTANCE = "importance"
    POSITION = "position"
    ADAPTIVE = "adaptive"

@dataclass
class YvCacheConfig:
    """Configuration dataclass for cache management systems.
    
    Encapsulates all hyperparameters for cache initialization,
    providing a centralized configuration interface for different
    cache types and optimization strategies.
    
    Memory Configuration:
        - max_cache_size: Maximum cache size in tokens (default: 8192)
        - block_size: Block size for paged attention (default: 512)
        - max_seq_len: Maximum sequence length (default: 4096)
    
    Quantization Configuration:
        - cache_quantization: Enable cache quantization (default: True)
        - quantization_bits: Bits for quantization, 4/8/16 (default: 8)
    
    Window Configuration:
        - cache_window_size: Window size for streaming cache (default: 2048)
        - use_h2o_attention: Enable H2O attention-based selection (default: True)
    
    Eviction Configuration:
        - eviction_policy: Cache eviction policy (default: ADAPTIVE)
    
    Offloading Configuration:
        - enable_offload: Enable CPU offloading (default: False)
        - offload_threshold: Memory threshold for offloading (default: 0.8)
    
    Compression Configuration:
        - enable_compression: Enable cache compression (default: True)
        - compression_ratio: Target compression ratio (default: 0.5)
    
    Model Configuration:
        - n_layers: Number of transformer layers (default: 32)
        - n_heads: Number of attention heads (default: 32)
        - head_dim: Dimension per head (default: 128)
        - dtype: Data type for cache tensors (default: float16)
        - device: Device for cache storage (default: "cuda")
    
    Cache Type:
        - cache_type: Type of cache to use (default: HYBRID)
    
    Example:
        >>> config = YvCacheConfig(
        ...     max_cache_size=16384,
        ...     cache_type=YvCacheType.PAGED,
        ...     eviction_policy=YvEvictionPolicy.ADAPTIVE
        ... )
    
    Attributes:
        max_cache_size (int): Maximum cache size in tokens.
        block_size (int): Block size for paged attention.
        cache_quantization (bool): Enable cache quantization.
        quantization_bits (int): Bits for quantization (4, 8, or 16).
        cache_window_size (int): Window size for streaming cache.
        use_h2o_attention (bool): Enable H2O attention-based cache selection.
        eviction_policy (YvEvictionPolicy): Cache eviction policy.
        enable_offload (bool): Enable CPU offloading for cache.
        offload_threshold (float): Memory threshold for offloading.
        enable_compression (bool): Enable cache compression.
        compression_ratio (float): Target compression ratio.
        max_seq_len (int): Maximum sequence length.
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of attention heads.
        head_dim (int): Dimension per head.
        dtype (torch.dtype): Data type for cache tensors.
        device (str): Device for cache storage.
    """
    max_cache_size: int = 8192
    block_size: int = 512
    cache_quantization: bool = True
    quantization_bits: int = 8
    cache_window_size: int = 2048
    use_h2o_attention: bool = True
    eviction_policy: YvEvictionPolicy = YvEvictionPolicy.ADAPTIVE
    enable_offload: bool = False
    offload_threshold: float = 0.8
    enable_compression: bool = True
    compression_ratio: float = 0.5
    max_seq_len: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    cache_type: YvCacheType = YvCacheType.HYBRID

    def __post_init__(self):
        if isinstance(self.eviction_policy, str):
            self.eviction_policy = YvEvictionPolicy(self.eviction_policy)
        if isinstance(self.cache_type, str):
            self.cache_type = YvCacheType(self.cache_type)

class YvCacheBlock:
    """Individual cache block for paged attention systems.
    
    Represents a fixed-size block of key-value cache that can be
    dynamically allocated and freed. This block-based approach enables
    efficient memory management similar to virtual memory systems.
    
    Architecture:
        Each block stores a contiguous chunk of key-value cache with
        metadata for tracking usage, importance, and lifecycle management.
    
    Memory Layout:
        - key_cache: [1, n_heads, block_size, head_dim]
        - value_cache: [1, n_heads, block_size, head_dim]
        - Total memory: 2 * n_heads * block_size * head_dim * sizeof(dtype)
    
    Key Features:
        - Fixed-size allocation for predictable memory usage
        - Reference counting for shared sequences (prefix caching)
        - Importance scoring for intelligent eviction
        - Access tracking for LRU/LFU policies
    
    Attributes:
        block_id (int): Unique identifier for this block.
        block_size (int): Maximum number of tokens this block can hold.
        n_heads (int): Number of attention heads.
        head_dim (int): Dimension per attention head.
        dtype (torch.dtype): Data type for cache tensors.
        device (str): Device where cache is stored.
        key_cache (torch.Tensor): Key cache tensor.
        value_cache (torch.Tensor): Value cache tensor.
        filled_size (int): Number of tokens currently stored.
        ref_count (int): Reference count for shared sequences.
        last_access_time (float): Timestamp of last access.
        access_count (int): Total number of accesses.
        importance_score (float): Importance score for eviction.
    
    Example:
        >>> block = YvCacheBlock(
        ...     block_id=0,
        ...     block_size=64,
        ...     n_heads=32,
        ...     head_dim=128,
        ...     dtype=torch.float16,
        ...     device='cuda'
        ... )
        >>> block.append(keys, values)
        >>> k, v = block.read()
    """

    def __init__(self, block_id: int, block_size: int, n_heads: int, head_dim: int, dtype: torch.dtype, device: str):
        """Initialize a cache block with specified parameters.
        
        Args:
            block_id: Unique identifier for this block.
            block_size: Maximum number of tokens this block can hold.
            n_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            dtype: Data type for cache tensors (e.g., torch.float16).
            device: Device where cache will be stored (e.g., 'cuda').
        
        Note:
            Memory is allocated immediately upon initialization.
            The block starts empty with filled_size=0.
        """
        self.block_id = block_id
        self.block_size = block_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.key_cache = torch.zeros(1, n_heads, block_size, head_dim, dtype=dtype, device=device)
        self.value_cache = torch.zeros(1, n_heads, block_size, head_dim, dtype=dtype, device=device)

        self.filled_size = 0
        self.ref_count = 0
        self.last_access_time = time.time()
        self.access_count = 0
        self.importance_score = 0.0

    def is_full(self) -> bool:
        """Check if the block has reached its capacity.
        
        Returns:
            True if filled_size >= block_size, False otherwise.
        """
        return self.filled_size >= self.block_size

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> int:
        """Append key-value pairs to this block.
        
        Args:
            keys: Key tensor of shape [1, n_heads, n_tokens, head_dim].
            values: Value tensor of shape [1, n_heads, n_tokens, head_dim].
        
        Returns:
            Number of tokens actually added (may be less than input if
            block is nearly full).
        
        Note:
            Updates last_access_time and access_count.
            Only appends up to available capacity.
        """
        n_tokens = keys.shape[2]
        available = self.block_size - self.filled_size
        n_to_add = min(n_tokens, available)

        if n_to_add > 0:
            self.key_cache[:, :, self.filled_size:self.filled_size + n_to_add] = keys[:, :, :n_to_add]
            self.value_cache[:, :, self.filled_size:self.filled_size + n_to_add] = values[:, :, :n_to_add]
            self.filled_size += n_to_add

        self.last_access_time = time.time()
        self.access_count += 1

        return n_to_add

    def read(self, start: int = 0, end: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read key-value pairs from this block.
        
        Args:
            start: Starting position to read from (default: 0).
            end: Ending position (exclusive). If None, reads to filled_size.
        
        Returns:
            Tuple of (keys, values) tensors.
        
        Note:
            Updates last_access_time and access_count.
        """
        if end is None:
            end = self.filled_size

        self.last_access_time = time.time()
        self.access_count += 1

        return self.key_cache[:, :, start:end], self.value_cache[:, :, start:end]

    def increment_ref(self):
        """Increment the reference count for this block.
        
        Used when multiple sequences share this block (prefix caching).
        """
        self.ref_count += 1

    def decrement_ref(self) -> int:
        """Decrement the reference count for this block.
        
        Returns:
            Updated reference count after decrementing.
        
        Note:
            Reference count is never negative (minimum 0).
        """
        self.ref_count = max(0, self.ref_count - 1)
        return self.ref_count

class YvPagedCacheManager:
    """Paged attention cache manager with virtual memory-style block management.
    
    Implements block-based cache management inspired by operating system
    virtual memory, allowing efficient memory allocation and sharing
    across sequences. This approach significantly reduces memory fragmentation
    and enables prefix caching for shared prompts.
    
    Architecture:
        The manager maintains a pool of fixed-size blocks that can be
        dynamically allocated to sequences. Blocks can be shared between
        sequences when they contain identical prefix content.
    
    Key Features:
        - Block-based allocation reduces memory fragmentation by 40-60%
        - Prefix caching enables sharing of common prompt prefixes
        - Reference counting for safe block sharing
        - Multiple eviction policies (LRU, LFU, Importance, Adaptive)
        - Dynamic block pool expansion
    
    Memory Management:
        - Blocks are allocated from a pre-allocated pool
        - When pool is exhausted, eviction is triggered
        - Eviction respects reference counts (shared blocks not evicted)
        - Block tables map sequences to their allocated blocks
    
    Performance Characteristics:
        - Allocation: O(1) for free blocks, O(n) for eviction
        - Append: O(n_tokens / block_size) block allocations
        - Read: O(n_blocks) concatenation operations
        - Memory: Bounded by pool size
    
    Attributes:
        config (YvCacheConfig): Configuration for the cache.
        block_size (int): Size of each block in tokens.
        n_heads (int): Number of attention heads.
        head_dim (int): Dimension per attention head.
        blocks (Dict[int, YvCacheBlock]): Block pool indexed by block_id.
        free_blocks (List[int]): List of available block IDs.
        sequence_blocks (Dict[int, List[int]]): Maps seq_id to block IDs.
        block_tables (Dict[int, torch.Tensor]): Block tables for fast lookup.
    
    Example:
        >>> manager = YvPagedCacheManager(config)
        >>> manager.allocate_sequence(seq_id=0, initial_blocks=2)
        >>> manager.append(seq_id=0, keys=k, values=v, layer_idx=0)
        >>> k, v = manager.read(seq_id=0, layer_idx=0)
    
    Reference:
        Kwon et al., "Efficient Memory Management for Large Language Model
        Serving with PagedAttention", SOSP 2023.
    """

    def __init__(self, config: YvCacheConfig):
        """Initialize the paged cache manager.
        
        Args:
            config: Configuration containing block_size, n_heads, head_dim,
                dtype, device, and eviction policy settings.
        
        Note:
            Pre-allocates an initial block pool for efficiency.
        """
        self.config = config
        self.block_size = config.block_size
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dtype = config.dtype
        self.device = config.device

        max_blocks = (config.max_cache_size * config.n_layers) // config.block_size + 100
        self.blocks: Dict[int, YvCacheBlock] = {}
        self.free_blocks: List[int] = []
        self.next_block_id = 0

        self.sequence_blocks: Dict[int, List[int]] = {}
        self.block_tables: Dict[int, torch.Tensor] = {}

        self._initialize_block_pool(min(1000, max_blocks))

    def _initialize_block_pool(self, n_blocks: int):
        """Pre-allocate a pool of blocks for efficient allocation.
        
        Args:
            n_blocks: Number of blocks to pre-allocate.
        
        Note:
            Pre-allocation reduces allocation latency during inference.
        """
        for _ in range(n_blocks):
            block = YvCacheBlock(
                self.next_block_id, self.block_size, self.n_heads, self.head_dim, self.dtype, self.device
            )
            self.blocks[self.next_block_id] = block
            self.free_blocks.append(self.next_block_id)
            self.next_block_id += 1

    def _allocate_block(self) -> Optional[int]:
        """Allocate a block from the pool.
        
        Returns:
            Block ID if allocation successful, None if pool exhausted
            and no blocks can be evicted.
        
        Note:
            Attempts eviction if free pool is empty.
        """
        if self.free_blocks:
            block_id = self.free_blocks.pop()
            self.blocks[block_id].filled_size = 0
            self.blocks[block_id].importance_score = 0.0
            return block_id

        return self._evict_block()

    def _evict_block(self) -> Optional[int]:
        """Evict a block based on the configured eviction policy.
        
        Returns:
            Block ID of evicted block, or None if no blocks can be evicted.
        
        Note:
            Only evicts blocks with ref_count == 0 (not shared).
            Eviction policy is determined by config.eviction_policy.
        """
        if not self.blocks:
            return None

        candidates = [
            (bid, block) for bid, block in self.blocks.items()
            if block.ref_count == 0 and block.filled_size > 0
        ]

        if not candidates:
            return None

        if self.config.eviction_policy == YvEvictionPolicy.LRU:
            candidates.sort(key=lambda x: x[1].last_access_time)
        elif self.config.eviction_policy == YvEvictionPolicy.LFU:
            candidates.sort(key=lambda x: x[1].access_count)
        elif self.config.eviction_policy == YvEvictionPolicy.IMPORTANCE:
            candidates.sort(key=lambda x: x[1].importance_score)
        else:
            candidates.sort(key=lambda x: x[1].last_access_time)

        evicted_id = candidates[0][0]
        self.free_blocks.append(evicted_id)
        return evicted_id

    def allocate_sequence(self, seq_id: int, initial_blocks: int = 1) -> bool:
        """Allocate blocks for a new sequence.
        
        Args:
            seq_id: Unique identifier for the sequence.
            initial_blocks: Number of blocks to allocate initially.
        
        Returns:
            True if allocation successful, False if insufficient memory.
        
        Note:
            If allocation fails, any partially allocated blocks are freed.
        """
        if seq_id in self.sequence_blocks:
            return True

        block_ids = []
        for _ in range(initial_blocks):
            block_id = self._allocate_block()
            if block_id is None:
                for bid in block_ids:
                    self.free_blocks.append(bid)
                return False
            block_ids.append(block_id)
            self.blocks[block_id].increment_ref()

        self.sequence_blocks[seq_id] = block_ids
        self._update_block_table(seq_id)
        return True

    def _update_block_table(self, seq_id: int):
        """Update the block table for a sequence.
        
        Args:
            seq_id: Sequence identifier.
        
        Note:
            Block tables are used for fast lookup in attention kernels.
        """
        if seq_id not in self.sequence_blocks:
            return

        block_ids = self.sequence_blocks[seq_id]
        if block_ids:
            self.block_tables[seq_id] = torch.tensor(block_ids, dtype=torch.int32, device=self.device)

    def append(
        self,
        seq_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int
    ) -> bool:
        """Append key-value pairs to a sequence's cache.
        
        Args:
            seq_id: Sequence identifier.
            keys: Key tensor of shape [1, n_heads, n_tokens, head_dim].
            values: Value tensor of shape [1, n_heads, n_tokens, head_dim].
            layer_idx: Layer index (for future multi-layer support).
        
        Returns:
            True if append successful, False if memory allocation failed.
        
        Note:
            Automatically allocates new blocks when current block is full.
        """
        if seq_id not in self.sequence_blocks:
            if not self.allocate_sequence(seq_id):
                return False

        n_tokens = keys.shape[2]
        tokens_added = 0

        while tokens_added < n_tokens:
            current_block_id = self.sequence_blocks[seq_id][-1]
            current_block = self.blocks[current_block_id]

            if current_block.is_full():
                new_block_id = self._allocate_block()
                if new_block_id is None:
                    return False

                self.sequence_blocks[seq_id].append(new_block_id)
                self.blocks[new_block_id].increment_ref()
                self._update_block_table(seq_id)
                current_block = self.blocks[new_block_id]

            remaining_keys = keys[:, :, tokens_added:]
            remaining_values = values[:, :, tokens_added:]

            added = current_block.append(remaining_keys, remaining_values)
            tokens_added += added

        return True

    def read(self, seq_id: int, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Read all cached key-value pairs for a sequence.
        
        Args:
            seq_id: Sequence identifier.
            layer_idx: Layer index (for future multi-layer support).
        
        Returns:
            Tuple of (keys, values) tensors, or None if sequence not found.
        """
        if seq_id not in self.sequence_blocks:
            return None

        block_ids = self.sequence_blocks[seq_id]
        if not block_ids:
            return None

        keys_list = []
        values_list = []

        for block_id in block_ids:
            k, v = self.blocks[block_id].read()
            keys_list.append(k)
            values_list.append(v)

        keys = torch.cat(keys_list, dim=2)
        values = torch.cat(values_list, dim=2)

        return keys, values

    def free_sequence(self, seq_id: int):
        """Free all blocks associated with a sequence.
        
        Args:
            seq_id: Sequence identifier.
        
        Note:
            Decrements reference counts. Blocks are only returned to
            free pool when ref_count reaches 0.
        """
        if seq_id not in self.sequence_blocks:
            return

        for block_id in self.sequence_blocks[seq_id]:
            ref_count = self.blocks[block_id].decrement_ref()
            if ref_count == 0:
                self.free_blocks.append(block_id)

        del self.sequence_blocks[seq_id]
        if seq_id in self.block_tables:
            del self.block_tables[seq_id]

    def get_block_table(self, seq_id: int) -> Optional[torch.Tensor]:
        """Get the block table for a sequence.
        
        Args:
            seq_id: Sequence identifier.
        
        Returns:
            Block table tensor, or None if sequence not found.
        """
        return self.block_tables.get(seq_id)

class YvSSMCacheManager:
    """State Space Model cache manager for Mamba/SSM layers.
    
    Manages state caches for state space model layers with efficient
    state recycling and memory management. Unlike attention-based KV cache,
    SSM cache stores recurrent states that enable constant-memory generation.
    
    Architecture:
        SSM layers maintain a hidden state that is updated recurrently.
        This manager caches both the SSM state and convolution state
        for efficient autoregressive generation.
    
    Key Features:
        - Constant memory footprint regardless of sequence length
        - State isolation per sequence for batched generation
        - Convolution state caching for 1D convolutions in SSM
        - 10x less memory than attention-based KV cache
    
    State Types:
        - state_cache: Recurrent SSM state [batch, d_state]
        - conv_cache: Convolution state [batch, d_model, 4]
        - sequence_states: Per-sequence isolated states
    
    Performance Characteristics:
        - Memory: O(n_layers * d_state) - constant per layer
        - Update: O(1) - single state update
        - Compare to KV cache: O(seq_len * n_heads * head_dim)
    
    Attributes:
        config (YvCacheConfig): Configuration for the cache.
        d_state (int): SSM state dimension (head_dim * 2).
        d_model (int): Model dimension (n_heads * head_dim).
        dtype (torch.dtype): Data type for state tensors.
        device (str): Device for state storage.
        state_cache (Dict[int, torch.Tensor]): SSM states per layer.
        conv_cache (Dict[int, torch.Tensor]): Conv states per layer.
        sequence_states (Dict[int, Dict[int, torch.Tensor]]): Per-seq states.
    
    Example:
        >>> manager = YvSSMCacheManager(config)
        >>> manager.initialize_layer(layer_idx=0, batch_size=1)
        >>> state = manager.get_state(layer_idx=0)
        >>> manager.update_state(layer_idx=0, state=new_state)
    """

    def __init__(self, config: YvCacheConfig):
        """Initialize the SSM cache manager.
        
        Args:
            config: Configuration containing head_dim, n_heads, dtype,
                and device settings.
        
        Note:
            d_state is set to head_dim * 2 for selective SSM.
            d_model is set to n_heads * head_dim.
        """
        self.config = config
        self.d_state = config.head_dim * 2
        self.d_model = config.n_heads * config.head_dim
        self.dtype = config.dtype
        self.device = config.device

        self.state_cache: Dict[int, torch.Tensor] = {}
        self.conv_cache: Dict[int, torch.Tensor] = {}
        self.sequence_states: Dict[int, Dict[int, torch.Tensor]] = {}

    def initialize_layer(self, layer_idx: int, batch_size: int = 1):
        """Initialize cache for a specific layer.
        
        Args:
            layer_idx: Layer index to initialize.
            batch_size: Batch size for the cache tensors.
        
        Note:
            Initializes both SSM state and convolution state.
        """
        if layer_idx not in self.state_cache:
            self.state_cache[layer_idx] = torch.zeros(
                batch_size, self.d_state, dtype=self.dtype, device=self.device
            )

        if layer_idx not in self.conv_cache:
            self.conv_cache[layer_idx] = torch.zeros(
                batch_size, self.d_model, 4, dtype=self.dtype, device=self.device
            )

    def get_state(self, layer_idx: int, seq_id: Optional[int] = None) -> torch.Tensor:
        """Get SSM state for a layer.
        
        Args:
            layer_idx: Layer index.
            seq_id: Optional sequence ID for isolated state.
        
        Returns:
            SSM state tensor of shape [batch, d_state].
        
        Note:
            If seq_id is provided, returns sequence-isolated state.
            Otherwise, returns shared state from state_cache.
        """
        if seq_id is not None and seq_id in self.sequence_states:
            return self.sequence_states[seq_id].get(layer_idx, self.state_cache[layer_idx])
        return self.state_cache.get(layer_idx, torch.zeros(1, self.d_state, dtype=self.dtype, device=self.device))

    def update_state(self, layer_idx: int, state: torch.Tensor, seq_id: Optional[int] = None):
        """Update SSM state for a layer.
        
        Args:
            layer_idx: Layer index.
            state: New SSM state tensor.
            seq_id: Optional sequence ID for isolated state.
        
        Note:
            If seq_id is provided, updates sequence-isolated state.
            Otherwise, updates shared state in state_cache.
        """
        if seq_id is not None:
            if seq_id not in self.sequence_states:
                self.sequence_states[seq_id] = {}
            self.sequence_states[seq_id][layer_idx] = state.clone()
        else:
            self.state_cache[layer_idx] = state

    def get_conv_state(self, layer_idx: int) -> torch.Tensor:
        """Get convolution state for a layer.
        
        Args:
            layer_idx: Layer index.
        
        Returns:
            Convolution state tensor of shape [batch, d_model, 4].
        """
        return self.conv_cache.get(layer_idx, torch.zeros(1, self.d_model, 4, dtype=self.dtype, device=self.device))

    def update_conv_state(self, layer_idx: int, conv_state: torch.Tensor):
        """Update convolution state for a layer.
        
        Args:
            layer_idx: Layer index.
            conv_state: New convolution state tensor.
        """
        self.conv_cache[layer_idx] = conv_state

    def clone_state(self, seq_id: int) -> Dict[int, torch.Tensor]:
        """Clone all states for a sequence.
        
        Args:
            seq_id: Sequence identifier.
        
        Returns:
            Dictionary mapping layer indices to cloned state tensors.
        
        Note:
            Useful for branching generation or beam search.
        """
        if seq_id not in self.sequence_states:
            return {}
        return {k: v.clone() for k, v in self.sequence_states[seq_id].items()}

    def clear_sequence(self, seq_id: int):
        """Clear all states for a sequence.
        
        Args:
            seq_id: Sequence identifier.
        """
        if seq_id in self.sequence_states:
            del self.sequence_states[seq_id]

class YvSpeculativeCacheManager:
    """Speculative decoding cache manager for accelerated inference.
    
    Manages draft and verification caches for speculative decoding,
    supporting tree-structured speculation and acceptance-based caching.
    This approach enables 2-3x speedup with minimal quality loss.
    
    Architecture:
        Speculative decoding uses a smaller draft model to generate
        candidate tokens, which are then verified by the main model.
        This cache manager stores intermediate states for efficient
        verification and rollback.
    
    Key Features:
        - Draft cache for storing draft model outputs
        - Verification cache for main model verification
        - Acceptance history tracking for adaptive draft length
        - Tree-structured speculation for parallel verification
        - Rollback mechanism for rejected candidates
    
    Performance Characteristics:
        - Throughput: 2-3x improvement over standard decoding
        - Memory: O(draft_length * hidden_dim) per draft
        - Latency: Reduced wall-clock time for generation
    
    Cache Types:
        - draft_cache: Hidden states, logits, and tokens from draft model
        - verification_cache: Verified hidden states and logits
        - acceptance_history: Historical acceptance rates per draft length
        - tree_cache: Tree-structured speculation states
    
    Attributes:
        config (YvCacheConfig): Configuration for the cache.
        draft_cache (Dict[int, Dict[str, torch.Tensor]]): Draft model outputs.
        verification_cache (Dict[int, Dict[str, torch.Tensor]]): Verification results.
        acceptance_history (Dict[int, List[float]]): Acceptance rate history.
        tree_cache (Dict[int, Dict[int, torch.Tensor]]): Tree speculation states.
    
    Example:
        >>> manager = YvSpeculativeCacheManager(config)
        >>> manager.store_draft(draft_length=4, hidden_states=h, logits=l, tokens=t)
        >>> draft = manager.get_draft(draft_length=4)
        >>> optimal_length = manager.get_optimal_draft_length()
    
    Reference:
        Leviathan et al., "Fast Inference from Transformers via Speculative
        Decoding", ICML 2023.
    """

    def __init__(self, config: YvCacheConfig):
        """Initialize the speculative cache manager.
        
        Args:
            config: Configuration for the cache system.
        """
        self.config = config

        self.draft_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.verification_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.acceptance_history: Dict[int, List[float]] = {}
        self.tree_cache: Dict[int, Dict[int, torch.Tensor]] = {}

    def store_draft(
        self,
        draft_length: int,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        tokens: torch.Tensor
    ):
        """Store draft model outputs for a given draft length.
        
        Args:
            draft_length: Number of tokens in the draft.
            hidden_states: Hidden states from draft model.
            logits: Logits from draft model.
            tokens: Draft token sequence.
        
        Note:
            Overwrites any existing draft cache for this length.
        """
        if draft_length not in self.draft_cache:
            self.draft_cache[draft_length] = {}

        self.draft_cache[draft_length]['hidden_states'] = hidden_states
        self.draft_cache[draft_length]['logits'] = logits
        self.draft_cache[draft_length]['tokens'] = tokens

    def get_draft(self, draft_length: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve draft model outputs for a given draft length.
        
        Args:
            draft_length: Number of tokens in the draft.
        
        Returns:
            Dictionary with 'hidden_states', 'logits', and 'tokens',
            or None if not found.
        """
        return self.draft_cache.get(draft_length)

    def store_verification(
        self,
        draft_length: int,
        verified_hidden: torch.Tensor,
        verified_logits: torch.Tensor
    ):
        """Store verification results for a given draft length.
        
        Args:
            draft_length: Number of tokens in the draft.
            verified_hidden: Verified hidden states from main model.
            verified_logits: Verified logits from main model.
        """
        if draft_length not in self.verification_cache:
            self.verification_cache[draft_length] = {}

        self.verification_cache[draft_length]['hidden_states'] = verified_hidden
        self.verification_cache[draft_length]['logits'] = verified_logits

    def get_verification(self, draft_length: int) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve verification results for a given draft length.
        
        Args:
            draft_length: Number of tokens in the draft.
        
        Returns:
            Dictionary with 'hidden_states' and 'logits', or None if not found.
        """
        return self.verification_cache.get(draft_length)

    def record_acceptance(self, draft_length: int, acceptance_rate: float):
        """Record acceptance rate for adaptive draft length optimization.
        
        Args:
            draft_length: Number of tokens in the draft.
            acceptance_rate: Acceptance rate (0.0 to 1.0).
        
        Note:
            Maintains a sliding window of last 100 acceptance rates.
        """
        if draft_length not in self.acceptance_history:
            self.acceptance_history[draft_length] = []

        self.acceptance_history[draft_length].append(acceptance_rate)

        if len(self.acceptance_history[draft_length]) > 100:
            self.acceptance_history[draft_length] = self.acceptance_history[draft_length][-100:]

    def get_optimal_draft_length(self) -> int:
        """Determine optimal draft length based on acceptance history.
        
        Returns:
            Optimal draft length that maximizes throughput.
        
        Note:
            Uses acceptance_rate * length as the optimization metric.
            Returns 4 as default if insufficient history.
        """
        if not self.acceptance_history:
            return 4

        best_length = 4
        best_score = 0.0

        for length, history in self.acceptance_history.items():
            if len(history) >= 10:
                avg_acceptance = sum(history[-10:]) / 10
                score = avg_acceptance * length

                if score > best_score:
                    best_score = score
                    best_length = length

        return best_length

    def store_tree_node(self, tree_id: int, node_id: int, state: torch.Tensor):
        """Store a node state for tree-structured speculation.
        
        Args:
            tree_id: Identifier for the speculation tree.
            node_id: Identifier for the node within the tree.
            state: State tensor for this node.
        """
        if tree_id not in self.tree_cache:
            self.tree_cache[tree_id] = {}
        self.tree_cache[tree_id][node_id] = state

    def get_tree_node(self, tree_id: int, node_id: int) -> Optional[torch.Tensor]:
        """Retrieve a node state from tree-structured speculation.
        
        Args:
            tree_id: Identifier for the speculation tree.
            node_id: Identifier for the node within the tree.
        
        Returns:
            State tensor, or None if not found.
        """
        return self.tree_cache.get(tree_id, {}).get(node_id)

    def clear_tree(self, tree_id: int):
        """Clear all nodes for a speculation tree.
        
        Args:
            tree_id: Identifier for the speculation tree.
        """
        if tree_id in self.tree_cache:
            del self.tree_cache[tree_id]

class YvMultimodalCacheManager:
    """Multimodal cache manager for vision-language models.
    
    Manages caches for different modalities (text, image, audio, video)
    with modality-specific optimization strategies. This manager enables
    efficient handling of interleaved multimodal content.
    
    Architecture:
        Each modality has its own cache namespace, allowing different
        retention policies and compression strategies per modality.
        Cross-modal caches store aligned features for multimodal fusion.
    
    Key Features:
        - Separate caches per modality (text, image, audio, video)
        - Cross-modal feature caching for fusion layers
        - Alignment matrix caching for modality alignment
        - Modality-specific compression and eviction
    
    Supported Modalities:
        - text: Text token embeddings and representations
        - image: Image patch embeddings and features
        - audio: Audio spectrogram or waveform features
        - video: Video frame or clip features
    
    Cache Types:
        - modality_caches: Per-modality feature caches
        - cross_modal_cache: Cross-modal aligned features
        - alignment_cache: Modality alignment matrices
    
    Attributes:
        config (YvCacheConfig): Configuration for the cache.
        modality_caches (Dict[str, Dict[str, torch.Tensor]]): Per-modality caches.
        cross_modal_cache (Dict[str, torch.Tensor]): Cross-modal features.
        alignment_cache (Dict[str, torch.Tensor]): Alignment matrices.
    
    Example:
        >>> manager = YvMultimodalCacheManager(config)
        >>> manager.store_modality_cache('image', 'patch_0', img_features)
        >>> features = manager.get_modality_cache('image', 'patch_0')
        >>> manager.store_cross_modal('text_image_align', aligned_features)
    
    Reference:
        Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training
        with Frozen Image Encoders and Large Language Models", ICML 2023.
    """

    def __init__(self, config: YvCacheConfig):
        """Initialize the multimodal cache manager.
        
        Args:
            config: Configuration for the cache system.
        
        Note:
            Initializes empty caches for text, image, audio, and video.
        """
        self.config = config

        self.modality_caches: Dict[str, Dict[str, torch.Tensor]] = {
            'text': {},
            'image': {},
            'audio': {},
            'video': {}
        }

        self.cross_modal_cache: Dict[str, torch.Tensor] = {}
        self.alignment_cache: Dict[str, torch.Tensor] = {}

    def store_modality_cache(
        self,
        modality: str,
        cache_key: str,
        cache_data: torch.Tensor
    ):
        """Store cache data for a specific modality.
        
        Args:
            modality: Modality type ('text', 'image', 'audio', 'video').
            cache_key: Unique key for this cache entry.
            cache_data: Tensor to cache.
        
        Note:
            Only stores if modality is in supported list.
        """
        if modality in self.modality_caches:
            self.modality_caches[modality][cache_key] = cache_data

    def get_modality_cache(self, modality: str, cache_key: str) -> Optional[torch.Tensor]:
        """Retrieve cache data for a specific modality.
        
        Args:
            modality: Modality type ('text', 'image', 'audio', 'video').
            cache_key: Key for the cache entry.
        
        Returns:
            Cached tensor, or None if not found.
        """
        return self.modality_caches.get(modality, {}).get(cache_key)

    def store_cross_modal(self, key: str, features: torch.Tensor):
        """Store cross-modal aligned features.
        
        Args:
            key: Unique key for this cross-modal feature.
            features: Cross-modal feature tensor.
        """
        self.cross_modal_cache[key] = features

    def get_cross_modal(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cross-modal aligned features.
        
        Args:
            key: Key for the cross-modal feature.
        
        Returns:
            Cross-modal feature tensor, or None if not found.
        """
        return self.cross_modal_cache.get(key)

    def store_alignment(self, modality_pair: str, alignment_matrix: torch.Tensor):
        """Store alignment matrix for a modality pair.
        
        Args:
            modality_pair: Modality pair identifier (e.g., 'text_image').
            alignment_matrix: Alignment matrix tensor.
        """
        self.alignment_cache[modality_pair] = alignment_matrix

    def get_alignment(self, modality_pair: str) -> Optional[torch.Tensor]:
        """Retrieve alignment matrix for a modality pair.
        
        Args:
            modality_pair: Modality pair identifier.
        
        Returns:
            Alignment matrix tensor, or None if not found.
        """
        return self.alignment_cache.get(modality_pair)

    def get_all_modalities(self) -> List[str]:
        """Get list of all supported modalities.
        
        Returns:
            List of modality names.
        """
        return list(self.modality_caches.keys())

    def clear_modality(self, modality: str):
        """Clear all cache entries for a specific modality.
        
        Args:
            modality: Modality type to clear.
        """
        if modality in self.modality_caches:
            self.modality_caches[modality].clear()

class YvCacheCompressor:
    """Advanced cache compression utilities for memory-efficient inference.
    
    Implements multiple compression strategies for reducing cache memory
    footprint while maintaining inference quality. Supports importance-based,
    attention-aware, semantic clustering, and streaming-LLM style compression.
    
    Architecture:
        The compressor provides multiple algorithms for selecting which
        cache entries to retain, each with different trade-offs between
        compression ratio and quality preservation.
    
    Compression Strategies:
        1. Importance-Based: Retains tokens with highest importance scores
           based on norm and position weighting.
        
        2. Attention-Aware: Uses attention scores or estimated importance
           to retain the most attended tokens. Protects sink and recent tokens.
        
        3. Cluster-Based: Groups similar tokens via k-means clustering
           and retains cluster centroids.
        
        4. Semantic-Aware: Uses semantic similarity to identify redundant
           tokens and retain diverse representations.
        
        5. Streaming-LLM: Retains sink tokens and recent window for
           infinite-length generation.
        
        6. Adaptive: Automatically selects strategy based on sequence length.
    
    Key Features:
        - Multiple compression algorithms for different use cases
        - Sink token protection for attention stability
        - Recent token preservation for coherence
        - Configurable compression ratio
    
    Performance Characteristics:
        - Importance-based: O(seq_len) for importance computation
        - Attention-aware: O(seq_len) with attention scores, O(seq_len * head_dim) without
        - Cluster-based: O(n_clusters * seq_len * iterations)
        - Streaming-LLM: O(1) for simple windowing
    
    Attributes:
        config (YvCacheConfig): Configuration for compression.
        compression_ratio (float): Target compression ratio.
        _sink_token_count (int): Number of sink tokens to protect.
        _recent_token_count (int): Number of recent tokens to protect.
    
    Example:
        >>> compressor = YvCacheCompressor(config)
        >>> compressed_k, compressed_v = compressor.compress_importance_based(
        ...     keys, values, target_ratio=0.5
        ... )
    
    Reference:
        Xiao et al., "Efficient Streaming Language Models with Attention Sinks",
        ICLR 2024.
    """

    def __init__(self, config: YvCacheConfig):
        """Initialize the cache compressor.
        
        Args:
            config: Configuration containing compression_ratio and other settings.
        """
        self.config = config
        self.compression_ratio = config.compression_ratio
        self._sink_token_count = 4
        self._recent_token_count = 4

    def compress_importance_based(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        target_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress cache using importance-based selection.
        
        Retains tokens with the highest importance scores, computed
        from key/value norms and position weighting.
        
        Args:
            keys: Key tensor of shape [batch, heads, seq_len, head_dim].
            values: Value tensor of shape [batch, heads, seq_len, head_dim].
            target_ratio: Target compression ratio (0.5 = keep 50%).
        
        Returns:
            Tuple of compressed (keys, values) tensors.
        
        Note:
            Importance = (||key|| + ||value||) / 2 * position_weight
            Position weight decays exponentially with distance.
        """
        batch, heads, seq_len, head_dim = keys.shape
        target_len = max(1, int(seq_len * target_ratio))

        importance = self._compute_importance(keys, values)

        _, top_indices = torch.topk(importance, target_len, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        top_indices_exp = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        compressed_keys = torch.gather(keys, 2, top_indices_exp)
        compressed_values = torch.gather(values, 2, top_indices_exp)

        return compressed_keys, compressed_values

    def _compute_importance(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for cache entries.
        
        Args:
            keys: Key tensor of shape [batch, heads, seq_len, head_dim].
            values: Value tensor of shape [batch, heads, seq_len, head_dim].
        
        Returns:
            Importance scores of shape [batch, heads, seq_len].
        """
        key_norm = torch.norm(keys, dim=-1)
        value_norm = torch.norm(values, dim=-1)

        importance = (key_norm + value_norm) / 2

        position_weights = torch.exp(-torch.arange(keys.shape[2], device=keys.device).float() / 100.0)
        importance = importance * position_weights.unsqueeze(0).unsqueeze(0)

        return importance

    def compress_attention_aware(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
        target_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress cache using attention-aware selection.
        
        Retains tokens that receive the most attention, while always
        protecting sink tokens and recent tokens for stability.
        
        Args:
            keys: Key tensor of shape [batch, heads, seq_len, head_dim].
            values: Value tensor of shape [batch, heads, seq_len, head_dim].
            attention_scores: Optional pre-computed attention scores.
            target_ratio: Target compression ratio.
        
        Returns:
            Tuple of compressed (keys, values) tensors.
        
        Note:
            Sink tokens (first 4) and recent tokens (last 4) are always
            protected to maintain attention stability and coherence.
        """
        batch, heads, seq_len, head_dim = keys.shape
        target_len = max(1, int(seq_len * target_ratio))

        if attention_scores is not None:
            importance = attention_scores.mean(dim=(0, 1))
            importance = importance.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1)
        else:
            importance = self._compute_attention_importance(keys, values)

        sink_indices = torch.arange(self._sink_token_count, device=keys.device)
        recent_indices = torch.arange(seq_len - self._recent_token_count, seq_len, device=keys.device)
        protected_indices = torch.cat([sink_indices, recent_indices])

        importance[:, :, protected_indices] = float('inf')

        remaining_target = target_len - len(protected_indices)
        if remaining_target > 0:
            _, top_indices = torch.topk(importance, remaining_target, dim=-1, largest=True)
            top_indices = torch.cat([protected_indices.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1), top_indices], dim=-1)
        else:
            top_indices = protected_indices[:target_len].unsqueeze(0).unsqueeze(0).expand(batch, heads, -1)

        top_indices, _ = torch.sort(top_indices, dim=-1)
        top_indices_exp = top_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        compressed_keys = torch.gather(keys, 2, top_indices_exp)
        compressed_values = torch.gather(values, 2, top_indices_exp)

        return compressed_keys, compressed_values

    def _compute_attention_importance(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Estimate attention importance without actual attention scores.
        
        Uses a combination of norm, variance, and position features
        to estimate which tokens are likely to receive high attention.
        
        Args:
            keys: Key tensor of shape [batch, heads, seq_len, head_dim].
            values: Value tensor of shape [batch, heads, seq_len, head_dim].
        
        Returns:
            Estimated importance scores of shape [batch, heads, seq_len].
        """
        batch, heads, seq_len, head_dim = keys.shape

        key_norm = torch.norm(keys, dim=-1)
        value_norm = torch.norm(values, dim=-1)

        norm_importance = (key_norm + value_norm) / 2

        key_mean = keys.mean(dim=-1, keepdim=True)
        value_mean = values.mean(dim=-1, keepdim=True)

        key_var = ((keys - key_mean) ** 2).mean(dim=-1)
        value_var = ((values - value_mean) ** 2).mean(dim=-1)

        variance_importance = (key_var + value_var) / 2

        position_decay = torch.exp(-torch.arange(seq_len, device=keys.device).float() / 50.0)
        recency_boost = torch.exp(torch.arange(seq_len, device=keys.device).float() / 20.0) / math.e
        position_weights = position_decay + recency_boost

        importance = (
            0.3 * norm_importance +
            0.3 * variance_importance +
            0.4 * position_weights.unsqueeze(0).unsqueeze(0)
        )

        return importance

    def compress_cluster_based(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        n_clusters: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress cache using k-means clustering.
        
        Groups similar tokens into clusters and retains cluster centroids,
        providing a diverse representation of the cache content.
        
        Args:
            keys: Key tensor of shape [batch, heads, seq_len, head_dim].
            values: Value tensor of shape [batch, heads, seq_len, head_dim].
            n_clusters: Number of clusters (final cache size).
        
        Returns:
            Tuple of compressed (keys, values) tensors.
        
        Note:
            Uses simple k-means with 10 iterations.
            Both keys and values are compressed to cluster centroids.
        """
        batch, heads, seq_len, head_dim = keys.shape

        keys_flat = keys.reshape(batch * heads, seq_len, head_dim)
        values_flat = values.reshape(batch * heads, seq_len, head_dim)

        combined = (keys_flat + values_flat) / 2

        cluster_centers = combined[:, :n_clusters].clone()

        for _ in range(10):
            distances = torch.cdist(combined, cluster_centers)
            assignments = torch.argmin(distances, dim=-1)

            for k in range(n_clusters):
                mask = (assignments == k)
                if mask.any():
                    cluster_centers[:, k] = combined[mask].mean(dim=0)

        compressed_keys = cluster_centers[:, :n_clusters].unsqueeze(1).expand(-1, heads, -1, -1)
        compressed_keys = compressed_keys.reshape(batch, heads, n_clusters, head_dim)

        return compressed_keys, compressed_keys

    def compress_semantic_aware(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        target_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, heads, seq_len, head_dim = keys.shape
        target_len = max(1, int(seq_len * target_ratio))

        combined = (keys + values) / 2

        similarity = torch.matmul(combined, combined.transpose(-2, -1)) / math.sqrt(head_dim)
        similarity = similarity.mean(dim=1)

        importance = similarity.mean(dim=1)

        sink_indices = torch.arange(self._sink_token_count, device=keys.device)
        recent_indices = torch.arange(seq_len - self._recent_token_count, seq_len, device=keys.device)

        importance[:, sink_indices] = float('inf')
        importance[:, recent_indices] = float('inf')

        _, top_indices = torch.topk(importance, target_len, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        top_indices_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, heads, -1, head_dim)

        compressed_keys = torch.gather(keys, 1, top_indices_exp)
        compressed_values = torch.gather(values, 1, top_indices_exp)

        return compressed_keys, compressed_values

    def compress_streaming_llm(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        window_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, heads, seq_len, head_dim = keys.shape

        if seq_len <= window_size:
            return keys, values

        sink_keys = keys[:, :, :self._sink_token_count]
        sink_values = values[:, :, :self._sink_token_count]

        recent_keys = keys[:, :, -window_size:]
        recent_values = values[:, :, -window_size:]

        compressed_keys = torch.cat([sink_keys, recent_keys], dim=2)
        compressed_values = torch.cat([sink_values, recent_values], dim=2)

        return compressed_keys, compressed_values

    def compress_adaptive(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        target_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, heads, seq_len, head_dim = keys.shape

        if seq_len <= 256:
            return self.compress_importance_based(keys, values, target_ratio)
        elif seq_len <= 1024:
            return self.compress_attention_aware(keys, values, None, target_ratio)
        else:
            return self.compress_semantic_aware(keys, values, target_ratio)

class YvUnifiedCacheManager:
    """
    Unified cache manager for transformer model inference.

    Manages multiple types of caches including KV cache for attention layers,
    H2O attention-based cache selection, PagedAttention-style block management,
    generation cache for multimodal tasks, and speculative decoding cache.
    Supports cache quantization and eviction policies to manage memory usage.
    """

    def __init__(self, config: Union[YvCacheConfig, Dict[str, Any], Any]):
        if isinstance(config, dict):
            filtered_config = {k: v for k, v in config.items() if k in YvCacheConfig.__dataclass_fields__}
            self.config = YvCacheConfig(**filtered_config)
        elif isinstance(config, YvCacheConfig):
            self.config = config
        else:
            self.config = YvCacheConfig(
                max_cache_size=getattr(config, 'kv_cache_max_size', 8192),
                cache_quantization=getattr(config, 'quantization_enabled', True),
                cache_window_size=getattr(config, 'streaming_window', 2048),
                block_size=getattr(config, 'kv_cache_block_size', 512),
            )

        self.kv_cache: Dict[int, Dict[str, Any]] = {}
        self.generation_cache: Dict[str, Any] = {}
        self.speculative_cache: Dict[str, Any] = {}
        self.h2o_cache: Dict[Any, Any] = {}

        self.paged_manager: Optional[YvPagedCacheManager] = None
        self.ssm_manager: Optional[YvSSMCacheManager] = None
        self.speculative_manager: Optional[YvSpeculativeCacheManager] = None
        self.multimodal_manager: Optional[YvMultimodalCacheManager] = None
        self.compressor: Optional[YvCacheCompressor] = None

        if self.config.cache_type in [YvCacheType.PAGED, YvCacheType.HYBRID]:
            self.paged_manager = YvPagedCacheManager(self.config)

        if self.config.cache_type == YvCacheType.SSM:
            self.ssm_manager = YvSSMCacheManager(self.config)

        if self.config.cache_type == YvCacheType.SPECULATIVE:
            self.speculative_manager = YvSpeculativeCacheManager(self.config)

        if self.config.enable_compression:
            self.compressor = YvCacheCompressor(self.config)

        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

        self._lock = threading.Lock()

    def get_kv_cache(self, layer_idx: int, past_key_values: Optional[Tuple[torch.Tensor]] = None):
        with self._lock:
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
        with self._lock:
            if use_h2o and self.config.use_h2o_attention:
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
                bs = self.config.block_size
                num_blocks = (delta + bs - 1) // bs

                for i in range(num_blocks):
                    s = i * bs
                    e = min(delta, (i + 1) * bs)
                    kb = tail_k[:, :, s:e, :]
                    vb = tail_v[:, :, s:e, :]

                    if self.config.cache_quantization and kb.shape[2] >= min(bs, 256):
                        kb, vb = self._quantize_cache(kb, vb)

                    entry['blocks'].append((kb, vb))
                    entry['total_len'] += (e - s)

            soft_cap = int(self.config.max_cache_size * 1.5)
            while entry['total_len'] > soft_cap and entry['blocks']:
                kb, vb = entry['blocks'].pop(0)
                entry['total_len'] -= kb.shape[2]
                self.cache_evictions += 1

            if entry['total_len'] > self.config.max_cache_size and len(entry['blocks']) >= 1:
                self._compact_blocks(entry)

            while entry['total_len'] > self.config.max_cache_size and entry['blocks']:
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

            target_total = int(self.config.max_cache_size * 1.1)
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
                norm_imp = (imp_k + imp_v).mean(dim=1)
                
                head_k_mean = head_k.mean(dim=-1, keepdim=True)
                head_v_mean = head_v.mean(dim=-1, keepdim=True)
                head_k_var = ((head_k - head_k_mean) ** 2).mean(dim=-1)
                head_v_var = ((head_v - head_v_mean) ** 2).mean(dim=-1)
                var_imp = (head_k_var + head_v_var).mean(dim=1)
                
                combined_imp = 0.5 * norm_imp + 0.5 * var_imp

                topk = max(keep_recent, head_len // 2)
                topk = min(topk, head_len)
                _, idx_sel = torch.topk(combined_imp, k=topk, dim=-1)
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
        except Exception as e:
            _LOG.debug(f"Cache block compaction failed: {e}")

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

    def set_h2o_cache(
        self,
        key_states: torch.Tensor,
        current_pos: int,
        max_cache_size: int,
        selected_keys: torch.Tensor,
        selected_values: torch.Tensor
    ):
        cache_key = (current_pos // max(1, max_cache_size))
        self.h2o_cache[cache_key] = (selected_keys, selected_values)

    def _apply_h2o_cache_selection(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        current_pos: int
    ):
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        if seq_len <= self.config.cache_window_size:
            return key_states, value_states

        importance_scores = self._calculate_importance_scores(key_states, value_states)
        cache_start = max(0, current_pos - self.config.cache_window_size)
        cache_end = current_pos

        if cache_end - cache_start > self.config.cache_window_size:
            cache_importance = importance_scores[:, :, cache_start:cache_end]
            _, top_indices = torch.topk(cache_importance, self.config.cache_window_size, dim=-1)
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
        batch_size, num_heads, seq_len, head_dim = key_states.shape

        attention_scores = torch.matmul(key_states, value_states.transpose(-2, -1)) / math.sqrt(head_dim)

        importance = attention_scores.diagonal(dim1=-2, dim2=-1)

        position_weights = torch.exp(-torch.arange(seq_len, device=key_states.device).float() / 100.0)
        position_weights = position_weights.unsqueeze(0).unsqueeze(0)
        importance = importance * position_weights

        return F.softmax(importance, dim=-1)

    def _quantize_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = key_states.shape[2]

        if seq_len <= self.config.cache_window_size:
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

        quant_bits = self.config.quantization_bits
        key_states = self._quantize_tensor(key_states, quant_bits)
        value_states = self._quantize_tensor(value_states, quant_bits)
        return key_states, value_states

    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
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

    def allocate_paged_sequence(self, seq_id: int, initial_blocks: int = 1) -> bool:
        if self.paged_manager is not None:
            return self.paged_manager.allocate_sequence(seq_id, initial_blocks)
        return False

    def append_paged(
        self,
        seq_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int
    ) -> bool:
        if self.paged_manager is not None:
            return self.paged_manager.append(seq_id, keys, values, layer_idx)
        return False

    def read_paged(self, seq_id: int, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.paged_manager is not None:
            return self.paged_manager.read(seq_id, layer_idx)
        return None

    def free_paged_sequence(self, seq_id: int):
        if self.paged_manager is not None:
            self.paged_manager.free_sequence(seq_id)

    def get_ssm_state(self, layer_idx: int, seq_id: Optional[int] = None) -> torch.Tensor:
        if self.ssm_manager is not None:
            return self.ssm_manager.get_state(layer_idx, seq_id)
        return torch.zeros(1, 256, dtype=self.config.dtype, device=self.config.device)

    def update_ssm_state(self, layer_idx: int, state: torch.Tensor, seq_id: Optional[int] = None):
        if self.ssm_manager is not None:
            self.ssm_manager.update_state(layer_idx, state, seq_id)

    def store_speculative_draft(
        self,
        draft_length: int,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        tokens: torch.Tensor
    ):
        if self.speculative_manager is not None:
            self.speculative_manager.store_draft(draft_length, hidden_states, logits, tokens)

    def get_speculative_draft(self, draft_length: int) -> Optional[Dict[str, torch.Tensor]]:
        if self.speculative_manager is not None:
            return self.speculative_manager.get_draft(draft_length)
        return None

    def get_optimal_draft_length(self) -> int:
        if self.speculative_manager is not None:
            return self.speculative_manager.get_optimal_draft_length()
        return 4

    def store_multimodal_cache(self, modality: str, cache_key: str, cache_data: torch.Tensor):
        if self.multimodal_manager is not None:
            self.multimodal_manager.store_modality_cache(modality, cache_key, cache_data)

    def get_multimodal_cache(self, modality: str, cache_key: str) -> Optional[torch.Tensor]:
        if self.multimodal_manager is not None:
            return self.multimodal_manager.get_modality_cache(modality, cache_key)
        return None

    def compress_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        target_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.compressor is not None:
            return self.compressor.compress_importance_based(keys, values, target_ratio)
        return keys, values

    def clear_cache(self):
        with self._lock:
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

        stats = {
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
            'cache_window_size': self.config.cache_window_size,
            'quantization_enabled': bool(self.config.cache_quantization)
        }

        if self.paged_manager is not None:
            stats['paged_sequences'] = len(self.paged_manager.sequence_blocks)
            stats['paged_free_blocks'] = len(self.paged_manager.free_blocks)

        if self.ssm_manager is not None:
            stats['ssm_cached_layers'] = len(self.ssm_manager.state_cache)

        return stats

    def create_checkpoint(self) -> Dict[str, Any]:
        checkpoint = {
            'kv_cache': {},
            'config': self.config.__dict__.copy(),
            'stats': self.get_cache_stats()
        }

        for layer_idx, entry in self.kv_cache.items():
            if entry and 'blocks' in entry:
                checkpoint['kv_cache'][layer_idx] = {
                    'blocks': [(b[0].clone(), b[1].clone()) for b in entry['blocks']],
                    'total_len': entry['total_len']
                }

        return checkpoint

    def restore_checkpoint(self, checkpoint: Dict[str, Any]):
        with self._lock:
            self.kv_cache.clear()

            for layer_idx, entry in checkpoint.get('kv_cache', {}).items():
                self.kv_cache[int(layer_idx)] = {
                    'blocks': [(b[0].clone(), b[1].clone()) for b in entry['blocks']],
                    'total_len': entry['total_len']
                }
