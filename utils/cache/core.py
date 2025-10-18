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

import os
import json
import time
import random
import hashlib
import tempfile
import threading
from pathlib import Path
from utils.log.core import PiscesLxCoreLog
from typing import Any, Dict, Optional, Callable, Tuple, List

_DEFAULT_CACHE: Optional["PiscesLxCoreCache"] = None
logger = PiscesLxCoreLog("PiscesLx.Utils.Cache")

class PiscesLxCoreCache:
    """
    A core cache implementation for PiscesLx that supports both in-memory and disk storage.
    It provides features like LRU eviction, TTL expiration, and soft expiration.
    """

    def __init__(self, cache_dir: str, max_mem_entries: int = 2048, max_size: int | None = None,
                 ttl_jitter: float = 0.1, enable_cleanup: bool = True, cleanup_interval_sec: int = 300,
                 max_disk_bytes: Optional[int] = None) -> None:
        """
        Initialize the cache instance.

        Args:
            cache_dir (str): Directory path to store cached data on disk.
            max_mem_entries (int, optional): Maximum number of entries to store in memory. Defaults to 2048.
            max_size (int | None, optional): Explicit maximum memory size. If provided, it overrides max_mem_entries. Defaults to None.
            ttl_jitter (float, optional): TTL jitter fraction (e.g., 0.1 => ±10%) to mitigate cache stampede. Defaults to 0.1.
            enable_cleanup (bool, optional): Whether to enable the cleanup thread. Defaults to True.
            cleanup_interval_sec (int, optional): Cleanup interval in seconds. Defaults to 300.
            max_disk_bytes (Optional[int], optional): Maximum disk space in bytes. Defaults to None.
        """
        self.cache_dir = self._resolve_cache_dir(cache_dir)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception as e:
            logger.warning("failed to create cache directory", event="cache.init.mkdir_error", cache_dir=self.cache_dir, error=str(e))

        self._mem: Dict[str, Dict[str, Any]] = {}  # In-memory index to store cached data
        self._mem_access: Dict[str, Dict[str, float]] = {}  # In-memory access records to track last access time
        self._max_mem_entries = int(max_size) if (max_size is not None) else max_mem_entries  # Prefer explicit max_size if provided, otherwise use max_mem_entries
        self._lock = threading.RLock()  # Reentrant lock for thread-safe operations
        self._hits = 0  # Cache hit count
        self._misses = 0  # Cache miss count
        self._ttl_jitter = max(0.0, min(0.5, float(ttl_jitter)))  # TTL jitter fraction to mitigate cache stampede
        self._inflight: Dict[Tuple[str, str], Tuple[threading.Event, Optional[BaseException], Optional[Any]]] = {}  # Single-flight in-flight map: (ns,key) -> (Event, result/exception)
        self._keylocks: Dict[Tuple[str, str], threading.Lock] = {}  # Per-key locks to serialize writers minimally
        self.metrics: Dict[str, int] = {  # Basic metrics
            "disk_reads": 0,
            "disk_writes": 0,
            "evictions": 0,
            "corrupt_files": 0,
            "soft_expired_served": 0,
            "inflight_waiters": 0,
        }
        self._enable_cleanup = bool(enable_cleanup)  # Whether to enable the cleanup thread
        self._cleanup_interval = int(cleanup_interval_sec)
        self._stop_cleanup = threading.Event()
        self._max_disk_bytes = int(max_disk_bytes) if max_disk_bytes is not None else None
        self._disk_bytes_est: int = 0  # Estimated disk usage
        self._last_cap_check: float = 0.0  # Last time disk capacity was checked
        self._cap_min_interval: float = 30.0  # Minimum interval for capacity check
        self._path_index: Dict[str, Tuple[str, str]] = {}  # Map file path to (namespace, key) for lock cleanup
        if self._enable_cleanup:
            t = threading.Thread(target=self._cleanup_loop, name="PiscesCacheCleanup", daemon=True)
            t.start()

    def _get_key_lock(self, namespace: str, key: str) -> threading.Lock:
        """
        Get or create a lock for the given namespace and key.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.

        Returns:
            threading.Lock: A lock object for the given namespace and key.
        """
        ns_key = (namespace, key)
        with self._lock:
            if ns_key not in self._keylocks:
                self._keylocks[ns_key] = threading.Lock()
            return self._keylocks[ns_key]

    def get(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached item from memory or disk.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.

        Returns:
            Optional[Dict[str, Any]]: The cached item if found, otherwise None.
        """
        node = self._mem_get(namespace, key)
        if node is not None:
            try:
                self._hits += 1
            except Exception as e:
                logger.debug("failed to increment hit counter", event="cache.hit_counter.error", error=str(e))
            self._emit("cache.get.hit", namespace=namespace, key=key)
            return node

        node = self._disk_get(namespace, key)
        if node is not None:
            # Populate the item into memory
            self._mem_put(namespace, key, node)
            try:
                self._hits += 1
            except Exception as e:
                logger.debug("failed to increment hit counter from disk", event="cache.disk_hit.error", error=str(e))
            self._emit("cache.get.hit", namespace=namespace, key=key)
        else:
            try:
                self._misses += 1
            except Exception as e:
                logger.debug("failed to increment miss counter", event="cache.miss.error", error=str(e))
            self._emit("cache.get.miss", namespace=namespace, key=key)
        return node

    def set(self, namespace: str, key: str, value: Any, ttl: int) -> None:
        """
        Set a cached item in both memory and disk.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.
            value (Any): Value to be cached.
            ttl (int): Time-to-live in seconds.
        """
        node = {
            "value": value,
            "created_at": int(time.time()),
            "ttl": self._apply_jitter(ttl),
            "soft_expired": False,
        }
        # Serialize writes for this key to avoid interleaving
        lock = self._get_key_lock(namespace, key)
        with lock:
            self._mem_put(namespace, key, node)
            self._disk_put(namespace, key, node)
        self._emit("cache.set", namespace=namespace, key=key)

    def get_or_set(self, namespace: str, key: str, ttl: int, producer: Callable[[], Any], *,
                   allow_stale_once: bool = True, async_refresh: bool = True) -> Any:
        """
        Get a value or compute-and-set it with single-flight protection.

        - If fresh present: return it.
        - If expired and allow_stale_once: return stale immediately, optionally refresh in background.
        - Otherwise compute once (single-flight), other callers wait.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.
            ttl (int): Time-to-live in seconds.
            producer (Callable[[], Any]): Function to produce the value if not cached.
            allow_stale_once (bool, optional): Whether to allow serving stale value once. Defaults to True.
            async_refresh (bool, optional): Whether to refresh the value asynchronously. Defaults to True.

        Returns:
            Any: The cached or newly computed value.
        """
        # Fast path: fresh
        node = self.get(namespace, key)
        if node is not None and not self._is_expired(node):
            return node.get("value")

        # Serve stale once and refresh in background
        if node is not None and self._is_expired(node) and allow_stale_once:
            try:
                self.metrics["soft_expired_served"] += 1
            except Exception as e:
                logger.debug("failed to increment soft_expired_served counter", event="cache.soft_expired.error", error=str(e))
            self._emit("cache.soft_expired.served", namespace=namespace, key=key)
            if async_refresh:
                self._emit("cache.refresh.async.start", namespace=namespace, key=key)
                threading.Thread(target=self._refresh_background, args=(namespace, key, ttl, producer), daemon=True).start()
            return node.get("value")

        # Single-flight compute
        return self._single_flight_compute(namespace, key, ttl, producer)

    def get_fresh(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get a non-expired cached item.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.

        Returns:
            Optional[Any]: The value of the non-expired cached item if found, otherwise None.
        """
        node = self.get(namespace, key)
        if node is None:
            return None
        if self._is_expired(node):
            return None
        return node.get("value")

    def get_with_soft_expiry(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get a cached item with soft expiration support.
        If the item is expired, mark it as soft expired and return the stale value once.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.

        Returns:
            Optional[Any]: The value of the cached item if found, otherwise None.
        """
        node = self.get(namespace, key)
        if node is None:
            return None
        if self._is_expired(node):
            # Mark the item as soft expired and return the stale value once
            node["soft_expired"] = True
            self._mem_put(namespace, key, node)
            self._disk_put(namespace, key, node)
            try:
                self.metrics["soft_expired_served"] += 1
            except Exception as e:
                logger.debug("failed to increment soft_expired_served counter in get_with_soft_expiry", event="cache.soft_expiry.error", error=str(e))
            return node.get("value")
        return node.get("value")

    def hit_rate(self) -> float:
        """
        Return cache hit rate in the range [0,1].

        Returns:
            float: Cache hit rate. If total accesses are 0, return 0.0.
        """
        try:
            total = self._hits + self._misses
            if total <= 0:
                return 0.0
            return float(self._hits) / float(total)
        except Exception as e:
            logger.debug("failed to calculate hit rate", event="cache.hit_rate.error", error=str(e))
            return 0.0

    def invalidate(self, namespace: str, key: Optional[str] = None) -> None:
        """
        Invalidate cached items.

        Args:
            namespace (str): Namespace of the cached item(s).
            key (Optional[str], optional): Key of the cached item. If None, invalidate all items in the namespace. Defaults to None.
        """
        if key is None:
            # Remove all items in the namespace
            with self._lock:
                self._mem.pop(namespace, None)
                self._mem_access.pop(namespace, None)
                # Clean up per-key locks for the namespace
                to_del = [fk for fk in self._keylocks if fk[0] == namespace]
                for fk in to_del:
                    self._keylocks.pop(fk, None)
            ns_dir = os.path.join(self.cache_dir, namespace)
            try:
                for fn in os.listdir(ns_dir):
                    try:
                        os.remove(os.path.join(ns_dir, fn))
                    except Exception as e:
                        logger.debug("failed to remove cache file during namespace invalidation", event="cache.invalidate.file_error", file=os.path.join(ns_dir, fn), error=str(e))
            except Exception as e:
                logger.debug("failed to list namespace directory during invalidation", event="cache.invalidate.listdir_error", ns_dir=ns_dir, error=str(e))
                return

        # Remove a specific item
        with self._lock:
            if namespace in self._mem and key in self._mem[namespace]:
                self._mem[namespace].pop(key, None)
            if namespace in self._mem_access and key in self._mem_access[namespace]:
                self._mem_access[namespace].pop(key, None)
            self._keylocks.pop((namespace, key), None)
        path = self._path(namespace, key)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.debug("failed to remove cache file during key invalidation", event="cache.invalidate.key_error", path=path, error=str(e))

    def _emit(self, event: str, **kwargs) -> None:
        """
        Emit a cache event with structured logging.

        Args:
            event (str): Event name/type.
            **kwargs: Additional event data (namespace, key, etc.).
        """
        try:
            logger.info(f"Cache event: {event}", event=event, **kwargs)
        except Exception as e:
            # Best effort: ignore logging errors
            logger.debug("failed to emit cache event", event="cache.emit.error", original_event=event, error=str(e))

    def _is_expired(self, node: Dict[str, Any]) -> bool:
        """
        Check if a cached item is expired.

        Args:
            node (Dict[str, Any]): Cached item node.

        Returns:
            bool: True if the item is expired, otherwise False.
        """
        ttl = int(node.get("ttl", 0))
        if ttl <= 0:
            return False
        created = int(node.get("created_at", 0))
        return time.time() > created + ttl

    def _mem_get(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached item from memory.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.

        Returns:
            Optional[Dict[str, Any]]: The cached item if found in memory, otherwise None.
        """
        with self._lock:
            ns = self._mem.get(namespace)
            if not ns:
                return None
            node = ns.get(key)
            if node is not None:
                # Update access time while holding the lock
                self._mem_access.setdefault(namespace, {})[key] = time.time()
            return node

    def _mem_put(self, namespace: str, key: str, node: Dict[str, Any]) -> None:
        """
        Put a cached item into memory.
        If the number of memory entries exceeds the limit, evict the least recently used item.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.
            node (Dict[str, Any]): Cached item node.
        """
        with self._lock:
            self._mem.setdefault(namespace, {})[key] = node
            self._touch(namespace, key)
            # Simple LRU cap per namespace
            if sum(len(d) for d in self._mem.values()) > self._max_mem_entries:
                self._evict_lru()

    def _disk_get(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached item from disk.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.

        Returns:
            Optional[Dict[str, Any]]: The cached item if found on disk and valid, otherwise None.
        """
        path = self._path(namespace, key)
        try:
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                try:
                    self.metrics["disk_reads"] += 1
                except Exception as e:
                    logger.debug("failed to increment disk_reads counter", event="cache.disk_reads.error", error=str(e))
                self._emit("cache.disk.read", namespace=namespace, key=key)
                return data
        except json.JSONDecodeError:
            # Remove corrupted file
            try:
                os.remove(path)
            except Exception as e:
                logger.debug("failed to remove corrupted cache file", event="cache.corrupt.remove_error", path=path, error=str(e))
            try:
                self.metrics["corrupt_files"] += 1
            except Exception as e:
                logger.debug("failed to increment corrupt_files counter", event="cache.corrupt.counter_error", error=str(e))
            try:
                logger.warning(
                    "corrupted cache file removed",
                    event="cache.disk.corrupt",
                    path=path,
                )
            except Exception as e:
                logger.debug("failed to log corrupted cache file removal", event="cache.corrupt.log_error", path=path, error=str(e))
            self._emit("cache.disk.corrupt", namespace=namespace, key=key)
            return None
        except Exception:
            try:
                logger.warning(
                    "cache disk read error",
                    event="cache.disk.error",
                    path=path,
                )
            except Exception as e:
                logger.debug("failed to log cache disk read error", event="cache.disk.error_log_error", path=path, error=str(e))
            self._emit("cache.disk.error", namespace=namespace, key=key)
            return None

    def _disk_put(self, namespace: str, key: str, node: Dict[str, Any]) -> None:
        """
        Put a cached item into disk with atomic write.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.
            node (Dict[str, Any]): Cached item node.
        """
        path = self._path(namespace, key)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Atomic write: use temp file and then replace
            dirp = os.path.dirname(path)
            fd, tmp = tempfile.mkstemp(prefix=".tmp", dir=dirp)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(node, f, ensure_ascii=False)
                os.replace(tmp, path)
                try:
                    self.metrics["disk_writes"] += 1
                except Exception as e:
                    logger.debug("failed to increment disk_writes counter", event="cache.disk_writes.error", error=str(e))
                self._emit("cache.disk.write", namespace=namespace, key=key)
                # Track file in index and update size estimate
                try:
                    self._path_index[path] = (namespace, key)
                    sz = os.path.getsize(path)
                    self._disk_bytes_est = max(0, self._disk_bytes_est) + int(sz)
                except Exception as e:
                    logger.debug("failed to update path index and size estimate", event="cache.disk.index_error", path=path, error=str(e))
                # Throttled disk capacity enforcement (auto-tuned interval)
                if self._max_disk_bytes is not None and self._disk_bytes_est > self._max_disk_bytes:
                    now = time.time()
                    if now - self._last_cap_check > PiscesLxCoreCache._cap_interval(self):
                        self._enforce_disk_cap()
            except Exception:
                try:
                    os.remove(tmp)
                except Exception as e:
                    logger.debug("failed to remove temporary file after disk write error", event="cache.disk.cleanup_error", tmp_file=tmp, error=str(e))
        except Exception as e:
            try:
                logger.error(
                    "cache disk write error",
                    event="cache.disk.write.error",
                    path=path,
                    error=str(e),
                    error_class=type(e).__name__,
                )
            except Exception as e2:
                logger.debug("failed to log cache disk write error", event="cache.disk.write.log_error", path=path, original_error=str(e), log_error=str(e2))

    def _path(self, namespace: str, key: str) -> str:
        """
        Generate the disk path for a cached item.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.

        Returns:
            str: The disk path of the cached item.
        """
        ns_dir = os.path.join(self.cache_dir, namespace)
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        shard = h[:2]
        return os.path.join(ns_dir, shard, f"{h}.json")

    def _touch(self, namespace: str, key: str) -> None:
        """
        Update the last access time of a cached item in memory.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.
        """
        with self._lock:
            self._mem_access.setdefault(namespace, {})[key] = time.time()

    def _evict_lru(self) -> None:
        """
        Evict the least recently used cached item across all namespaces.
        """
        with self._lock:
            oldest_ns = None
            oldest_key = None
            oldest_ts = float("inf")
            for ns, d in self._mem_access.items():
                for k, ts in d.items():
                    if ts < oldest_ts:
                        oldest_ns, oldest_key, oldest_ts = ns, k, ts
            if oldest_ns is not None and oldest_key is not None:
                try:
                    self._mem[oldest_ns].pop(oldest_key, None)
                    self._mem_access[oldest_ns].pop(oldest_key, None)
                    try:
                        self.metrics["evictions"] += 1
                    except Exception as e:
                        logger.debug("failed to increment evictions counter", event="cache.evictions.error", error=str(e))
                except Exception as e:
                    logger.debug("failed to evict LRU item", event="cache.eviction.error", oldest_ns=oldest_ns, oldest_key=oldest_key, error=str(e))

    # -------------------- helpers & advanced features --------------------
    def _apply_jitter(self, ttl: int) -> int:
        """
        Apply jitter to the TTL value to mitigate cache stampede.

        Args:
            ttl (int): Original time-to-live in seconds.

        Returns:
            int: TTL value with jitter applied.
        """
        try:
            ttl = int(ttl)
            if ttl <= 0 or self._ttl_jitter <= 0:
                return ttl
            jitter_span = int(ttl * self._ttl_jitter)
            return max(1, ttl + random.randint(-jitter_span, jitter_span))
        except Exception as e:
            logger.debug("failed to apply TTL jitter", event="cache.jitter.error", original_ttl=ttl, error=str(e))
            return int(ttl)

    def _single_flight_compute(self, namespace: str, key: str, ttl: int, producer: Callable[[], Any]) -> Any:
        """
        Compute value with single-flight protection to prevent duplicate computation.

        Args:
            namespace (str): Namespace for the cache key.
            key (str): Cache key.
            ttl (int): Time-to-live in seconds.
            producer (Callable[[], Any]): Function to produce the value if not cached.

        Returns:
            Any: The computed or cached value.

        Raises:
            BaseException: If the producer function raises an exception.
        """
        flight_key = (namespace, key)

        # Decide producer vs waiter
        # Treat an inflight entry as active only if ev is not set and result/exception are None
        with self._lock:
            entry = self._inflight.get(flight_key)
            if entry is None:
                ev = threading.Event()
                # Store (event, exception, value, timestamp)
                self._inflight[flight_key] = (ev, None, None, time.time())
                is_producer = True
            else:
                ev, exc0, val0 = entry
                if (not ev.is_set()) and exc0 is None and val0 is None:
                    try:
                        self.metrics["inflight_waiters"] += 1
                    except Exception as e:
                        logger.debug("failed to increment inflight_waiters counter", event="cache.inflight_waiters.error", error=str(e))
                    is_producer = False
                else:
                    # Previous round finished; start a new production
                    ev = threading.Event()
                    self._inflight[flight_key] = (ev, None, None, time.time())
                    is_producer = True

        if not is_producer:
            # Wait for producer to finish and return its result
            ev.wait()
            with self._lock:
                _, exc, val = self._inflight.get(flight_key, (None, None, None))
            if exc:
                raise exc
            return val

        # We are the producer
        try:
            self._emit("cache.singleflight.start", namespace=namespace, key=key)
            val = producer()
            # Serialize writes for this key
            lock = self._get_key_lock(namespace, key)
            with lock:
                self.set(namespace, key, val, ttl)
            with self._lock:
                entry = self._inflight.get(flight_key, (None, None, None, None))
                ev = entry[0] if entry else None
                if ev:
                    self._inflight[flight_key] = (ev, None, val, time.time())
                    ev.set()
            self._emit("cache.singleflight.end", namespace=namespace, key=key)
            return val
        except BaseException as exc:
            with self._lock:
                entry = self._inflight.get(flight_key, (None, None, None, None))
                ev = entry[0] if entry else None
                if ev:
                    self._inflight[flight_key] = (ev, exc, None, time.time())
                    ev.set()
            self._emit("cache.singleflight.error", namespace=namespace, key=key, error=str(exc))
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Return cache statistics for observability.

        Returns:
            Dict[str, Any]: A dictionary containing cache statistics.
        """
        try:
            with self._lock:
                total = self._hits + self._misses
                hit_rate = (float(self._hits) / float(total)) if total > 0 else 0.0
                return {
                    "hit_rate": hit_rate,
                    "hits": self._hits,
                    "misses": self._misses,
                    "metrics": dict(self.metrics),
                    "inflight_size": len(self._inflight),
                    "keylocks_size": len(self._keylocks),
                    "disk_bytes_est": self._disk_bytes_est,
                    "max_disk_bytes": self._max_disk_bytes,
                }
        except Exception as e:
            logger.debug("failed to get cache stats", event="cache.stats.error", error=str(e))
            return {
                "hit_rate": 0.0,
                "hits": self._hits,
                "misses": self._misses,
                "metrics": dict(self.metrics),
            }

    def _refresh_background(self, namespace: str, key: str, ttl: int, producer: Callable[[], Any]) -> None:
        """
        Refresh the cached item in the background.

        Args:
            namespace (str): Namespace of the cached item.
            key (str): Key of the cached item.
            ttl (int): Time-to-live in seconds.
            producer (Callable[[], Any]): Function to produce the new value.
        """
        try:
            val = producer()
            self.set(namespace, key, val, ttl)
            self._emit("cache.refresh.async.end", namespace=namespace, key=key)
        except Exception as e:
            # Best effort: emit error and exit
            self._emit("cache.refresh.async.error", namespace=namespace, key=key)
            logger.debug("failed to refresh cache in background", event="cache.refresh.error", namespace=namespace, key=key, error=str(e))
            return

    def _cleanup_loop(self) -> None:
        """
        Periodically clean up expired cached items and stale inflight entries.
        """
        while not self._stop_cleanup.wait(self._cleanup_interval):
            try:
                # Disk cleanup (bounded work per cycle; auto-tuned)
                max_check = PiscesLxCoreCache._cleanup_max_check(self)
                checked = 0
                for ns in os.listdir(self.cache_dir):
                    ns_dir = os.path.join(self.cache_dir, ns)
                    if not os.path.isdir(ns_dir):
                        continue
                    # Traverse shard subdirs
                    for shard in os.listdir(ns_dir):
                        shard_dir = os.path.join(ns_dir, shard)
                        if not os.path.isdir(shard_dir):
                            continue
                        for fn in os.listdir(shard_dir):
                            if not fn.endswith('.json'):
                                continue
                            path = os.path.join(shard_dir, fn)
                            try:
                                with open(path, 'r', encoding='utf-8') as f:
                                    node = json.load(f)
                                if self._is_expired(node):
                                    os.remove(path)
                                    # Clean up indexes and key locks if tracked
                                    try:
                                        ns_key = self._path_index.pop(path, None)
                                        if ns_key:
                                            with self._lock:
                                                self._keylocks.pop(ns_key, None)
                                    except Exception as e:
                                        logger.debug("failed to cleanup path index and key locks", event="cache.cleanup.index_error", path=path, error=str(e))
                            except Exception:
                                # Ignore file errors but trace in debug
                                try:
                                    logger.debug(
                                        "cleanup: failed to inspect or remove file",
                                        event="cache.cleanup.file_error",
                                        path=path,
                                    )
                                except Exception as e:
                                    logger.debug("failed to log cleanup file error", event="cache.cleanup.log_error", path=path, error=str(e))
                            checked += 1
                            if checked >= max_check:
                                break
                        if checked >= max_check:
                            break
                    if checked >= max_check:
                        break

                # Inflight stale cleanup (ev set and older than 10 minutes)
                try:
                    now = time.time()
                    stale_keys: List[Tuple[str, str]] = []
                    with self._lock:
                        for k, entry in list(self._inflight.items()):
                            # Entry may be 3-tuple (legacy) or 4-tuple (with ts)
                            if len(entry) >= 3:
                                ev = entry[0]
                                ts = entry[3] if len(entry) >= 4 else None
                                if getattr(ev, 'is_set', lambda: False)() and (ts is None or now - ts > 600):
                                    stale_keys.append(k)
                    for k in stale_keys:
                        with self._lock:
                            self._inflight.pop(k, None)
                except Exception as e:
                    logger.debug("failed to cleanup inflight entries", event="cache.inflight.cleanup_error", error=str(e))
            except Exception as e:
                logger.debug("failed to run cleanup loop", event="cache.cleanup.loop_error", error=str(e))

    def _cap_interval(self) -> float:
        """
        Calculate the interval for checking disk capacity.
        Direct instance method (no alias binding).
        """
        try:
            return _auto_cap_interval(self)
        except Exception as e:
            logger.debug("failed to calculate auto cap interval", event="cache.cap_interval.error", error=str(e))
            return 30.0

    def _cleanup_max_check(self) -> int:
        """
        Calculate the maximum number of files to check during cleanup.
        Direct instance method (no alias binding).
        """
        try:
            # Prefer estimate; if unknown, do a quick partial count
            if self._disk_bytes_est <= 0:
                approx = _list_json_count(self.cache_dir)
                return _clamp(approx // 2 + 200, 200, 2000)
            return _auto_cleanup_max_check(self)
        except Exception as e:
            logger.debug("failed to calculate cleanup max check", event="cache.cleanup_max.error", error=str(e))
            return 500

    @classmethod
    def _resolve_cache_dir(cls, cache_dir: Optional[str]) -> str:
        """
        Resolve cache directory under project root.

        Args:
            cache_dir (Optional[str]): Cache directory path. If None, use default.

        Returns:
            str: Resolved cache directory path.
        """
        from utils.config.loader import PiscesLxCoreConfigLoader
        loader = PiscesLxCoreConfigLoader()
        base = str(loader._project_root)
        sub = (cache_dir or "cache").strip().strip("/\\")
        return os.path.join(base, sub) if sub else base


def get_default_cache(cache_dir: Optional[str] = None) -> PiscesLxCoreCache:
    """
    Get or create a module-level default cache.
    If cache_dir is None, use <project_root>/.pisceslx/cache.

    Args:
        cache_dir (Optional[str], optional): Cache directory path. Defaults to None.

    Returns:
        PiscesLxCoreCache: The default cache instance.
    """
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is not None and cache_dir is None:
        return _DEFAULT_CACHE
    # Always resolve under project_root/.pisceslx
    if cache_dir is None:
        cache_dir = "cache"
    _DEFAULT_CACHE = PiscesLxCoreCache(cache_dir=cache_dir)
    return _DEFAULT_CACHE

def _clamp(v: int, lo: int, hi: int) -> int:
    """
    Clamp a value between a lower and upper bound.

    Args:
        v (int): The value to clamp.
        lo (int): Lower bound.
        hi (int): Upper bound.

    Returns:
        int: The clamped value.
    """
    return max(lo, min(hi, v))

def _mb(nbytes: int) -> float:
    """
    Convert bytes to megabytes.

    Args:
        nbytes (int): Number of bytes.

    Returns:
        float: Number of megabytes.
    """
    try:
        return float(nbytes) / (1024.0 * 1024.0)
    except Exception as e:
        logger.debug("failed to convert bytes to MB", event="cache.util.mb_error", bytes=nbytes, error=str(e))
        return 0.0

def _gb(nbytes: int) -> float:
    """
    Convert bytes to gigabytes.

    Args:
        nbytes (int): Number of bytes.

    Returns:
        float: Number of gigabytes.
    """
    try:
        return float(nbytes) / (1024.0 * 1024.0 * 1024.0)
    except Exception as e:
        logger.debug("failed to convert bytes to GB", event="cache.util.gb_error", bytes=nbytes, error=str(e))
        return 0.0

def _safe_ratio(a: Optional[int], b: Optional[int]) -> float:
    """
    Calculate a safe ratio to avoid division by zero.

    Args:
        a (Optional[int]): Numerator.
        b (Optional[int]): Denominator.

    Returns:
        float: The calculated ratio or 0.0 if division is not possible.
    """
    try:
        if not a or not b or b <= 0:
            return 0.0
        return float(a) / float(b)
    except Exception as e:
        logger.debug("failed to calculate safe ratio", event="cache.util.ratio_error", a=a, b=b, error=str(e))
        return 0.0

def _list_json_count(root: str, max_dirs: int = 50) -> int:
    """
    Perform a best-effort fast count of JSON files by scanning a limited number of directories.

    Args:
        root (str): Root directory to start scanning from.
        max_dirs (int, optional): Maximum number of directories to scan at each level. Defaults to 50.

    Returns:
        int: Estimated count of JSON files.
    """
    count = 0
    try:
        for ns in os.listdir(root)[:max_dirs]:
            ns_dir = os.path.join(root, ns)
            if not os.path.isdir(ns_dir):
                continue
            for shard in os.listdir(ns_dir)[:max_dirs]:
                shard_dir = os.path.join(ns_dir, shard)
                if not os.path.isdir(shard_dir):
                    continue
                try:
                    count += sum(1 for fn in os.listdir(shard_dir) if fn.endswith('.json'))
                except Exception as e:
                    logger.debug("failed to list shard directory during json count", event="cache.util.shard_list_error", shard_dir=shard_dir, error=str(e))
                    continue
    except Exception as e:
        logger.debug("failed to list json files", event="cache.util.list_error", root=root, error=str(e))
        pass
    return count

def _auto_cleanup_max_check(cache: "PiscesLxCoreCache") -> int:
    """
    Automatically determine the maximum number of files to check during cleanup.

    Args:
        cache (PiscesLxCoreCache): Cache instance.

    Returns:
        int: Maximum number of files to check.
    """
    # Scale with estimated size: 200..2000
    est_gb = _gb(cache._disk_bytes_est)
    if est_gb <= 0.1:
        return 200
    if est_gb <= 1.0:
        return 500
    if est_gb <= 5.0:
        return 1000
    return 2000

def _auto_cap_interval(cache: "PiscesLxCoreCache") -> float:
    """
    Automatically determine the interval for checking disk capacity.

    Args:
        cache (PiscesLxCoreCache): Cache instance.

    Returns:
        float: Interval in seconds.
    """
    # Dynamic throttle based on fill ratio: 5..60 seconds
    ratio = _safe_ratio(cache._disk_bytes_est, cache._max_disk_bytes)
    if ratio >= 0.95:
        return 5.0
    if ratio >= 0.90:
        return 10.0
    if ratio >= 0.80:
        return 20.0
    return 60.0

# Bind helpers as methods without changing public API
def _cap_interval(self: "PiscesLxCoreCache") -> float:
    """
    Calculate the interval for checking disk capacity.

    Args:
        self (PiscesLxCoreCache): Cache instance.

    Returns:
        float: Interval in seconds.
    """
    try:
        return _auto_cap_interval(self)
    except Exception as e:
        logger.debug("failed to calculate auto cap interval", event="cache.cap_interval.error", error=str(e))
        return 30.0

def _cleanup_max_check(self: "PiscesLxCoreCache") -> int:
    """
    Calculate the maximum number of files to check during cleanup.

    Args:
        self (PiscesLxCoreCache): Cache instance.

    Returns:
        int: Maximum number of files to check.
    """
    try:
        # Prefer estimate; if unknown, do a quick partial count
        if self._disk_bytes_est <= 0:
            approx = _list_json_count(self.cache_dir)
            return _clamp(approx // 2 + 200, 200, 2000)
        return _auto_cleanup_max_check(self)
    except Exception as e:
        logger.debug("failed to calculate cleanup max check", event="cache.cleanup_max.error", error=str(e))
        return 500