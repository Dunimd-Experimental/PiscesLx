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
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import time
import threading
from utils.log.core import PiscesLxCoreLog
from utils.error import PiscesLxCoreCacheError
from typing import Any, Dict, Optional, Callable, Union
from utils.cache.core import PiscesLxCoreCache, get_default_cache

class PiscesLxCoreEnhancedCache:
    """
    Enhanced cache implementation with Redis integration, cache warming, and advanced monitoring features.
    """

    def __init__(self, cache_dir: Optional[str] = None, redis_url: Optional[str] = None,
                 max_mem_entries: int = 2048, enable_redis: bool = False) -> None:
        """
        Initialize an enhanced cache instance.

        Args:
            cache_dir (Optional[str]): Path to the local cache directory.
            redis_url (Optional[str]): Redis connection URL.
            max_mem_entries (int): Maximum number of entries in memory cache.
            enable_redis (bool): Whether to enable Redis support.
        """
        self.logger = PiscesLxCoreLog("pisceslx.cache.enhanced")
        self.local_cache = get_default_cache(cache_dir) if cache_dir else get_default_cache()
        # Avoid hard dependency on redis module in annotations
        self.redis_client: Optional[Any] = None
        self.enable_redis = enable_redis
        self.hit_stats: Dict[str, int] = {"local": 0, "redis": 0, "miss": 0}
        self._stats_lock = threading.RLock()

        # Lazy import to avoid top-level ImportError
        if enable_redis and redis_url:
            try:
                import redis as _redis  # lazy import
                self.redis_client = _redis.Redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()  # Test the connection
                self.logger.success("cache.redis.connected", {"url": redis_url})
            except Exception as e:
                self.logger.error("cache.redis.connect_failed", error=str(e))
                self.redis_client = None
                self.enable_redis = False

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get a value from the cache with priority: local memory > Redis > no cache.

        Args:
            namespace (str): Cache namespace.
            key (str): Cache key.

        Returns:
            Optional[Any]: Cached value or None if not found.
        """
        # First try to get from local memory
        value = self.local_cache.get(namespace, key)
        if value is not None:
            with self._stats_lock:
                self.hit_stats["local"] += 1
            self.logger.debug("cache.hit.local", {"namespace": namespace, "key": key})
            return value.get("value") if isinstance(value, dict) and "value" in value else value

        # If Redis is enabled, try to get from Redis
        if self.enable_redis and self.redis_client:
            try:
                redis_key = f"{namespace}:{key}"
                cached_data = self.redis_client.get(redis_key)
                if cached_data:
                    # Deserialize the data
                    data = json.loads(cached_data)
                    # Also put it into local cache
                    self.local_cache.set(namespace, key, data["value"], data["ttl"])
                    with self._stats_lock:
                        self.hit_stats["redis"] += 1
                    self.logger.debug("cache.hit.redis", {"namespace": namespace, "key": key})
                    return data["value"]
            except Exception as e:
                self.logger.error("cache.redis.get_error", error=str(e))

        # Cache miss
        with self._stats_lock:
            self.hit_stats["miss"] += 1
        self.logger.debug("cache.miss", {"namespace": namespace, "key": key})
        return None

    def set(self, namespace: str, key: str, value: Any, ttl: int) -> None:
        """
        Set a value in the cache.

        Args:
            namespace (str): Cache namespace.
            key (str): Cache key.
            value (Any): Value to cache.
            ttl (int): Time-to-live in seconds.
        """
        # Set local cache
        self.local_cache.set(namespace, key, value, ttl)

        # If Redis is enabled, set Redis cache as well
        if self.enable_redis and self.redis_client:
            try:
                redis_key = f"{namespace}:{key}"
                data = {
                    "value": value,
                    "ttl": ttl,
                    "created_at": int(time.time())
                }
                serialized_data = json.dumps(data)
                self.redis_client.setex(redis_key, ttl, serialized_data)
                self.logger.debug("cache.set.redis", {"namespace": namespace, "key": key})
            except Exception as e:
                self.logger.error("cache.redis.set_error", error=str(e))

    def get_or_set(self, namespace: str, key: str, ttl: int, producer: Callable[[], Any], *,
                   allow_stale_once: bool = True, async_refresh: bool = True) -> Any:
        """
        Get a cached value or generate and set it using a producer function.

        Args:
            namespace (str): Cache namespace.
            key (str): Cache key.
            ttl (int): Time-to-live in seconds.
            producer (Callable[[], Any]): Producer function to generate the value.
            allow_stale_once (bool): Whether to allow returning stale data once.
            async_refresh (bool): Whether to refresh expired data asynchronously.

        Returns:
            Any: Cached value.
        """
        value = self.get(namespace, key)
        if value is not None:
            return value

        # Generate a new value and cache it
        value = producer()
        self.set(namespace, key, value, ttl)
        return value

    def warm_up(self, warm_up_data: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """
        Warm up the cache with pre-defined data.

        Args:
            warm_up_data (Dict[str, Dict[str, Dict[str, Any]]]): Pre-warm data.
                Format: {namespace: {key: {"value": value, "ttl": ttl}}}
        """
        self.logger.info("cache.warm_up.start", {"namespaces": list(warm_up_data.keys())})
        warmed_count = 0

        for namespace, keys_data in warm_up_data.items():
            for key, data in keys_data.items():
                try:
                    value = data.get("value")
                    ttl = data.get("ttl", 3600)  # Default 1 hour
                    self.set(namespace, key, value, ttl)
                    warmed_count += 1
                except Exception as e:
                    self.logger.error("cache.warm_up.error", 
                        namespace=namespace, 
                        key=key, 
                        error=str(e)
                    )

        self.logger.success("cache.warm_up.completed", {"warmed_count": warmed_count})

    def get_hit_rate(self) -> Dict[str, float]:
        """
        Get cache hit rate statistics.

        Returns:
            Dict[str, float]: Hit rates for each cache layer.
        """
        with self._stats_lock:
            total = sum(self.hit_stats.values())
            if total == 0:
                return {"local": 0.0, "redis": 0.0, "overall": 0.0}

            return {
                "local": self.hit_stats["local"] / total,
                "redis": self.hit_stats["redis"] / total,
                "miss": self.hit_stats["miss"] / total,
                "overall": (self.hit_stats["local"] + self.hit_stats["redis"]) / total
            }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics information.
        """
        local_stats = self.local_cache.get_stats()
        hit_rates = self.get_hit_rate()

        return {
            "local_cache": local_stats,
            "hit_rates": hit_rates,
            "hit_stats": dict(self.hit_stats),
            "redis_enabled": self.enable_redis,
            "redis_connected": self.redis_client is not None if self.enable_redis else False
        }

    def invalidate(self, namespace: str, key: Optional[str] = None) -> None:
        """
        Invalidate cache entries.

        Args:
            namespace (str): Cache namespace.
            key (Optional[str]): Cache key. If None, invalidate the entire namespace.
        """
        # Invalidate local cache
        self.local_cache.invalidate(namespace, key)

        # If Redis is enabled, invalidate Redis cache
        if self.enable_redis and self.redis_client:
            try:
                if key is None:
                    # Invalidate the entire namespace
                    pattern = f"{namespace}:*"
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                else:
                    # Invalidate a specific key
                    redis_key = f"{namespace}:{key}"
                    self.redis_client.delete(redis_key)
                self.logger.debug("cache.invalidate.redis", {"namespace": namespace, "key": key})
            except Exception as e:
                self.logger.error("cache.redis.invalidate_error", error=str(e))

    def preload(self, namespace: str, keys: list, loader: Callable[[str], Any], ttl: int = 3600) -> None:
        """
        Preload cache data.

        Args:
            namespace (str): Cache namespace.
            keys (list): List of keys to preload.
            loader (Callable[[str], Any]): Loader function to fetch values.
            ttl (int): Time-to-live in seconds.
        """
        self.logger.info("cache.preload.start", {"namespace": namespace, "key_count": len(keys)})
        loaded_count = 0

        for key in keys:
            try:
                value = loader(key)
                self.set(namespace, key, value, ttl)
                loaded_count += 1
            except Exception as e:
                self.logger.error("cache.preload.error", {
                    "namespace": namespace, 
                    "key": key, 
                    "error": str(e)
                })

        self.logger.success("cache.preload.completed", {"loaded_count": loaded_count})


class PiscesLxCoreEnhancedCacheManager:
    """
    Enhanced cache manager that provides global cache instance management.
    """
    
    _instance: Optional['PiscesLxCoreEnhancedCacheManager'] = None
    _default_cache: Optional[PiscesLxCoreEnhancedCache] = None
    _lock = threading.RLock()
    
    def __init__(self) -> None:
        """Initialize the cache manager."""
        self.logger = PiscesLxCoreLog("pisceslx.cache.enhanced.manager")
    
    @classmethod
    def get_instance(cls) -> 'PiscesLxCoreEnhancedCacheManager':
        """
        Get the singleton instance of the cache manager.

        Returns:
            PiscesLxCoreEnhancedCacheManager: The cache manager instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_default_cache(self, cache_dir: Optional[str] = None, redis_url: Optional[str] = None,
                         enable_redis: bool = False) -> PiscesLxCoreEnhancedCache:
        """
        Get or create the default enhanced cache instance.

        Args:
            cache_dir (Optional[str]): Local cache directory.
            redis_url (Optional[str]): Redis connection URL.
            enable_redis (bool): Whether to enable Redis support.

        Returns:
            PiscesLxCoreEnhancedCache: The enhanced cache instance.
        """
        if self._default_cache is not None:
            return self._default_cache
            
        with self._lock:
            if self._default_cache is None:
                self._default_cache = PiscesLxCoreEnhancedCache(
                    cache_dir=cache_dir,
                    redis_url=redis_url,
                    enable_redis=enable_redis
                )
        return self._default_cache
    
    def reset_default_cache(self) -> None:
        """Reset the default cache instance."""
        with self._lock:
            self._default_cache = None