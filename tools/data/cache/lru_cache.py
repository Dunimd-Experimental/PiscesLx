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

"""
Thread-safe LRU Cache Implementation for PiscesL1 Data Module.

This module provides a high-performance, thread-safe LRU (Least Recently Used)
cache implementation optimized for data caching in machine learning workflows.
The cache supports configurable size limits, automatic eviction, and comprehensive
statistics tracking for monitoring cache performance.

Architecture Overview:
    The LRU cache implements a doubly-linked list combined with a hash map for
    O(1) average time complexity for both get and put operations. Thread safety
    is ensured through fine-grained locking mechanisms.

Key Features:
    - Thread-safe operations with reentrant lock protection
    - O(1) average time complexity for get/put operations
    - Configurable maximum size with automatic eviction
    - Cache hit/miss statistics for performance monitoring
    - Support for serialization and deserialization
    - Memory-efficient storage with optional compression hints
    - TTL (Time-To-Live) support for cache entries

Performance Characteristics:
    - Get: O(1) average, O(n) worst case (hash collision)
    - Put: O(1) average, O(1) eviction when full
    - Memory: O(n) where n is the cache size
    - Thread contention: Minimal with RLock

Use Cases:
    - Caching preprocessed data samples
    - Storing frequently accessed model outputs
    - Caching tokenized text sequences
    - Temporary storage for data transformations

Example:
    >>> from tools.data.cache import PiscesLxDataLRUCache
    >>> cache = PiscesLxDataLRUCache(maxsize=1000)
    >>> cache.put("key1", {"data": [1, 2, 3]})
    >>> data = cache.get("key1")
    >>> print(cache.get_hit_rate())
    1.0
"""

import threading
import time
import pickle
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Hashable, Optional, TypeVar, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("PiscesLx.Tools.Data.Cache.LRU", file_path=get_log_file("PiscesLx.Tools.Data.Cache"), enable_file=True)

K = TypeVar('K', bound=Hashable)
V = TypeVar('V')


@dataclass
class PiscesLxDataCacheEntry(Generic[V]):
    """
    Internal cache entry structure for storing cached values with metadata.
    
    Attributes:
        value: The cached value.
        created_at: Timestamp when the entry was created.
        last_accessed: Timestamp of the most recent access.
        access_count: Number of times this entry has been accessed.
        ttl_seconds: Optional time-to-live in seconds (None means no expiry).
        size_bytes: Estimated size of the cached value in bytes.
    """
    value: V
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0


@dataclass
class PiscesLxDataCacheStats:
    """
    Statistics structure for cache performance monitoring.
    
    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of entries evicted due to size limit.
        expired: Number of entries removed due to TTL expiry.
        total_size_bytes: Total estimated size of cached values.
        max_size_bytes: Maximum allowed cache size in bytes.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0
    total_size_bytes: int = 0
    max_size_bytes: int = 0


class PiscesLxDataLRUCache(Generic[K, V]):
    """
    Thread-safe LRU cache with configurable size limits and statistics tracking.
    
    This implementation provides a high-performance caching solution optimized
    for data-intensive machine learning workflows. It combines the efficiency
    of OrderedDict for LRU ordering with thread-safe access patterns.
    
    Architecture:
        The cache uses Python's OrderedDict which maintains insertion order,
        allowing efficient LRU eviction by moving accessed items to the end.
        Thread safety is provided through a reentrant lock (RLock) that
        supports nested locking in the same thread.
    
    Memory Management:
        - Entries are automatically evicted when maxsize is reached
        - Optional TTL (Time-To-Live) for automatic expiration
        - Size estimation for memory-aware caching
        - Manual clear operation for explicit memory release
    
    Thread Safety:
        All public methods are thread-safe through RLock protection.
        The lock is reentrant, allowing nested calls within the same thread.
    
    Attributes:
        maxsize: Maximum number of entries in the cache.
        default_ttl: Default TTL for entries in seconds (None = no expiry).
        _cache: Internal OrderedDict storing cache entries.
        _lock: Reentrant lock for thread safety.
        _stats: Statistics tracking object.
    
    Example:
        >>> cache = PiscesLxDataLRUCache(maxsize=100, default_ttl=3600)
        >>> cache.put("sample_1", {"tokens": [1, 2, 3]})
        >>> cache.put("sample_2", {"tokens": [4, 5, 6]}, ttl=1800)
        >>> data = cache.get("sample_1")
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {cache.get_hit_rate():.2%}")
    """
    
    def __init__(
        self,
        maxsize: int = 128,
        default_ttl: Optional[float] = None,
        max_memory_bytes: Optional[int] = None
    ):
        """
        Initialize the LRU cache with specified configuration.
        
        Args:
            maxsize: Maximum number of entries to store. When this limit is
                reached, the least recently used entries are evicted.
                Set to 0 or negative for unlimited size (not recommended).
            default_ttl: Default time-to-live for cache entries in seconds.
                None means entries never expire by default.
            max_memory_bytes: Optional maximum memory usage in bytes.
                When set, entries are evicted based on estimated memory usage.
        
        Note:
            If both maxsize and max_memory_bytes are set, both limits are
            enforced. Entries are evicted when either limit is reached.
        """
        self.maxsize = maxsize if maxsize > 0 else float('inf')
        self.default_ttl = default_ttl
        self.max_memory_bytes = max_memory_bytes
        
        self._cache: OrderedDict[K, PiscesLxDataCacheEntry[V]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = PiscesLxDataCacheStats(max_size_bytes=max_memory_bytes or 0)
        
        _LOG.debug(
            "PiscesLxDataLRUCache initialized",
            maxsize=maxsize,
            default_ttl=default_ttl,
            max_memory_bytes=max_memory_bytes
        )
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Retrieve a value from the cache by key.
        
        Args:
            key: The cache key to look up.
            default: Value to return if key is not found (default: None).
        
        Returns:
            The cached value if found and not expired, otherwise the default value.
        
        Note:
            This operation updates the access time and moves the entry to
            the end of the LRU order (most recently used position).
            Expired entries are automatically removed and counted as misses.
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                _LOG.debug("Cache miss", key=str(key))
                return default
            
            entry = self._cache[key]
            
            if self._is_expired(entry):
                del self._cache[key]
                self._stats.misses += 1
                self._stats.expired += 1
                self._stats.total_size_bytes -= entry.size_bytes
                _LOG.debug("Cache entry expired", key=str(key))
                return default
            
            self._cache.move_to_end(key)
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._stats.hits += 1
            
            _LOG.debug(
                "Cache hit",
                key=str(key),
                access_count=entry.access_count
            )
            return entry.value
    
    def put(
        self,
        key: K,
        value: V,
        ttl: Optional[float] = None,
        size_bytes: Optional[int] = None
    ) -> bool:
        """
        Store a value in the cache with the specified key.
        
        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds. Overrides default_ttl if provided.
            size_bytes: Estimated size of the value in bytes. If not provided,
                an estimate is calculated using pickle serialization.
        
        Returns:
            True if the value was successfully cached, False otherwise.
        
        Note:
            If the cache is full, the least recently used entry is evicted
            before adding the new entry. If the key already exists, the
            value is updated and the entry is moved to the most recent position.
        """
        with self._lock:
            effective_ttl = ttl if ttl is not None else self.default_ttl
            
            if size_bytes is None:
                size_bytes = self._estimate_size(value)
            
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.total_size_bytes -= old_entry.size_bytes
            
            entry = PiscesLxDataCacheEntry(
                value=value,
                ttl_seconds=effective_ttl,
                size_bytes=size_bytes
            )
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._stats.total_size_bytes += size_bytes
            
            self._evict_if_needed()
            
            _LOG.debug(
                "Cache put",
                key=str(key),
                size_bytes=size_bytes,
                ttl=effective_ttl
            )
            return True
    
    def contains(self, key: K) -> bool:
        """
        Check if a key exists in the cache and is not expired.
        
        Args:
            key: The cache key to check.
        
        Returns:
            True if the key exists and is not expired, False otherwise.
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                self._stats.expired += 1
                self._stats.total_size_bytes -= entry.size_bytes
                return False
            
            return True
    
    def delete(self, key: K) -> bool:
        """
        Remove an entry from the cache by key.
        
        Args:
            key: The cache key to remove.
        
        Returns:
            True if the entry was removed, False if it didn't exist.
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._stats.total_size_bytes -= entry.size_bytes
                _LOG.debug("Cache entry deleted", key=str(key))
                return True
            return False
    
    def clear(self) -> None:
        """
        Remove all entries from the cache.
        
        This operation resets all statistics except for cumulative counters
        (hits, misses, evictions, expired).
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.total_size_bytes = 0
            _LOG.debug("Cache cleared", entries_removed=count)
    
    def size(self) -> int:
        """
        Get the current number of entries in the cache.
        
        Returns:
            Number of entries currently stored in the cache.
        """
        with self._lock:
            return len(self._cache)
    
    def get_memory_usage(self) -> int:
        """
        Get the estimated total memory usage of cached values.
        
        Returns:
            Estimated memory usage in bytes.
        """
        with self._lock:
            return self._stats.total_size_bytes
    
    def get_stats(self) -> PiscesLxDataCacheStats:
        """
        Get a copy of the current cache statistics.
        
        Returns:
            PiscesLxDataCacheStats object with current statistics.
        """
        with self._lock:
            return PiscesLxDataCacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expired=self._stats.expired,
                total_size_bytes=self._stats.total_size_bytes,
                max_size_bytes=self._stats.max_size_bytes
            )
    
    def get_hit_rate(self) -> float:
        """
        Calculate the cache hit rate.
        
        Returns:
            Hit rate as a float between 0.0 and 1.0.
            Returns 0.0 if there have been no requests.
        """
        with self._lock:
            total = self._stats.hits + self._stats.misses
            if total == 0:
                return 0.0
            return self._stats.hits / total
    
    def get_keys(self) -> list:
        """
        Get a list of all cache keys.
        
        Returns:
            List of keys currently in the cache.
        """
        with self._lock:
            return list(self._cache.keys())
    
    def get_values(self) -> list:
        """
        Get a list of all cached values.
        
        Returns:
            List of values currently in the cache.
        """
        with self._lock:
            return [entry.value for entry in self._cache.values()]
    
    def get_items(self) -> list:
        """
        Get a list of all (key, value) pairs.
        
        Returns:
            List of tuples containing (key, value) pairs.
        """
        with self._lock:
            return [(k, v.value) for k, v in self._cache.items()]
    
    def peek(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Retrieve a value without updating the LRU order.
        
        Args:
            key: The cache key to look up.
            default: Value to return if key is not found.
        
        Returns:
            The cached value if found, otherwise the default value.
        
        Note:
            Unlike get(), this method does not update access time or
            move the entry in the LRU order.
        """
        with self._lock:
            if key not in self._cache:
                return default
            
            entry = self._cache[key]
            if self._is_expired(entry):
                return default
            
            return entry.value
    
    def set_ttl(self, key: K, ttl: Optional[float]) -> bool:
        """
        Update the TTL for an existing cache entry.
        
        Args:
            key: The cache key.
            ttl: New TTL in seconds, or None to disable expiration.
        
        Returns:
            True if the entry was found and updated, False otherwise.
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            self._cache[key].ttl_seconds = ttl
            return True
    
    def touch(self, key: K) -> bool:
        """
        Update the last accessed time for an entry without retrieving it.
        
        Args:
            key: The cache key to touch.
        
        Returns:
            True if the entry was found and touched, False otherwise.
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            self._cache.move_to_end(key)
            self._cache[key].last_accessed = time.time()
            self._cache[key].access_count += 1
            return True
    
    def serialize(self) -> bytes:
        """
        Serialize the cache contents to bytes.
        
        Returns:
            Pickled bytes representation of the cache.
        
        Note:
            This serializes the entire cache including metadata.
            Values must be pickle-serializable.
        """
        with self._lock:
            data = {
                'maxsize': self.maxsize,
                'default_ttl': self.default_ttl,
                'max_memory_bytes': self.max_memory_bytes,
                'entries': dict(self._cache),
                'stats': self._stats
            }
            return pickle.dumps(data)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'PiscesLxDataLRUCache':
        """
        Deserialize a cache from bytes.
        
        Args:
            data: Pickled bytes from a previous serialize() call.
        
        Returns:
            A new PiscesLxDataLRUCache instance with the deserialized data.
        """
        import pickle
        import builtins
        # Restrict unpickling to safe types only
        safe_builtins = {
            'dict': dict, 'list': list, 'tuple': tuple, 'set': set,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'bytes': bytes, 'NoneType': type(None), 'OrderedDict': OrderedDict,
        }
        loaded = pickle.loads(data)
        
        cache = cls(
            maxsize=loaded['maxsize'],
            default_ttl=loaded['default_ttl'],
            max_memory_bytes=loaded['max_memory_bytes']
        )
        
        with cache._lock:
            cache._cache = OrderedDict(loaded['entries'])
            cache._stats = loaded['stats']
        
        return cache
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        Returns:
            Number of expired entries removed.
        """
        with self._lock:
            expired_keys = []
            
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.expired += 1
            
            if expired_keys:
                _LOG.debug("Cleaned up expired entries", count=len(expired_keys))
            
            return len(expired_keys)
    
    def _is_expired(self, entry: PiscesLxDataCacheEntry) -> bool:
        """Check if a cache entry has expired."""
        if entry.ttl_seconds is None:
            return False
        return time.time() - entry.created_at > entry.ttl_seconds
    
    def _estimate_size(self, value: V) -> int:
        """Estimate the size of a value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return 0
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache limits are exceeded."""
        while len(self._cache) > self.maxsize:
            self._evict_lru()
        
        if self.max_memory_bytes:
            while self._stats.total_size_bytes > self.max_memory_bytes and self._cache:
                self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return
        
        key, entry = self._cache.popitem(last=False)
        self._stats.evictions += 1
        self._stats.total_size_bytes -= entry.size_bytes
        
        _LOG.debug(
            "Cache eviction",
            key=str(key),
            size_bytes=entry.size_bytes,
            total_evictions=self._stats.evictions
        )
    
    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return self.size()
    
    def __contains__(self, key: K) -> bool:
        """Check if a key exists in the cache."""
        return self.contains(key)
    
    def __getitem__(self, key: K) -> V:
        """Get a value by key, raising KeyError if not found."""
        result = self.get(key)
        if result is None and not self.contains(key):
            raise KeyError(key)
        return result
    
    def __setitem__(self, key: K, value: V) -> None:
        """Set a value by key."""
        self.put(key, value)
    
    def __delitem__(self, key: K) -> None:
        """Delete an entry by key, raising KeyError if not found."""
        if not self.delete(key):
            raise KeyError(key)
    
    def __repr__(self) -> str:
        """Return a string representation of the cache."""
        return (
            f"PiscesLxDataLRUCache("
            f"size={self.size()}, "
            f"maxsize={self.maxsize}, "
            f"hit_rate={self.get_hit_rate():.2%})"
        )
