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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

from __future__ import annotations

"""
PiscesLx Data Cache Module.

This module provides high-performance caching utilities for data-intensive
machine learning workflows in the PiscesLx framework. It includes thread-safe
LRU caching and memory-mapped array storage for efficient handling of large
datasets.

Architecture Overview:
    The cache module provides two primary components:
    
    1. PiscesLxDataLRUCache: Thread-safe LRU cache for in-memory caching
       - O(1) average time complexity for get/put operations
       - Configurable size limits with automatic eviction
       - TTL (Time-To-Live) support for cache entries
       - Comprehensive statistics tracking
       - Serialization support for persistence
    
    2. PiscesLxDataMemoryMappedArray: Memory-mapped numpy array storage
       - On-demand data loading from disk
       - Support for datasets larger than available RAM
       - Random access with O(1) complexity
       - Efficient batch iteration
       - Thread-safe read operations

Key Features:
    - Thread-safe operations for concurrent access
    - Memory-efficient storage for large datasets
    - Transparent caching with automatic eviction
    - Integration with PiscesLx data pipeline
    - Comprehensive logging and statistics

Use Cases:
    - Caching preprocessed data samples
    - Storing large embedding matrices
    - Tokenized text sequence caching
    - Feature caches for multimodal data
    - Training data sharding

Performance Characteristics:
    LRU Cache:
        - Get: O(1) average
        - Put: O(1) average
        - Memory: O(n) where n is cache size
    
    Memory-Mapped Array:
        - Random access: O(1) with page fault overhead
        - Sequential access: O(n) with OS read-ahead
        - Memory: O(page_size) per accessed region

Example Usage:
    >>> from tools.data.cache import PiscesLxDataLRUCache, PiscesLxDataMemoryMappedArray
    >>> 
    >>> # LRU Cache example
    >>> cache = PiscesLxDataLRUCache(maxsize=1000, default_ttl=3600)
    >>> cache.put("sample_1", {"tokens": [1, 2, 3]})
    >>> data = cache.get("sample_1")
    >>> print(f"Hit rate: {cache.get_hit_rate():.2%}")
    >>> 
    >>> # Memory-mapped array example
    >>> arr = PiscesLxDataMemoryMappedArray.create(
    ...     "embeddings.mmap",
    ...     shape=(100000, 768),
    ...     dtype='float32'
    ... )
    >>> arr[0:100] = embeddings  # Write
    >>> batch = arr[0:32]  # Read on-demand
    >>> arr.close()

Module Components:
    - PiscesLxDataLRUCache: Thread-safe LRU cache implementation
    - PiscesLxDataMemoryMappedArray: Memory-mapped numpy array
    - PiscesLxDataCacheEntry: Internal cache entry structure
    - PiscesLxDataCacheStats: Cache statistics container
    - PiscesLxDataMMapHeader: Memory-mapped file header

Dependencies:
    - numpy: Array operations and memory mapping
    - threading: Thread safety
    - mmap: Operating system memory mapping
    - utils.dc: Logging utilities
    - utils.paths: Path utilities
"""

from tools.data.cache.lru_cache import (
    PiscesLxDataLRUCache,
    PiscesLxDataCacheEntry,
    PiscesLxDataCacheStats,
)

from tools.data.cache.mmap import (
    PiscesLxDataMemoryMappedArray,
    PiscesLxDataMMapHeader,
)

__all__ = [
    'PiscesLxDataLRUCache',
    'PiscesLxDataCacheEntry',
    'PiscesLxDataCacheStats',
    'PiscesLxDataMemoryMappedArray',
    'PiscesLxDataMMapHeader',
]

__version__ = '1.0.0'
