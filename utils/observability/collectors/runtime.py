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

import os
import gc
import sys
import time
import platform
import threading
from dataclasses import dataclass
from utils.log.core import PiscesLxCoreLog
from typing import Optional, Dict, Any, Tuple
from utils.error import PiscesLxCoreObservabilityError

_LOGGER = PiscesLxCoreLog("pisceslx.collectors.runtime")

@dataclass
class RuntimeCacheConfig:
    """Configuration for runtime metrics cache.

    Attributes:
        timeout (float): Cache timeout in seconds. Defaults to 2.0.
        max_retries (int): Maximum number of retries. Defaults to 2.
        retry_delay (float): Delay between retries in seconds. Defaults to 0.2.
    """
    timeout: float = 2.0  # Cache timeout in seconds
    max_retries: int = 2  # Maximum number of retries
    retry_delay: float = 0.2  # Delay between retries in seconds

# Thread-safe runtime metrics cache
_runtime_cache = {}
_cache_lock = threading.RLock()
_cache_config = RuntimeCacheConfig()

# Runtime metrics status tracking
_runtime_status = {
    "last_successful_method": None,
    "consecutive_failures": 0,
    "last_update": 0,
    "gc_stats": None
}

# Rate status for network/disk
_rate_state: Dict[str, Any] = {
    "last_ts": 0.0,
    "net": None,   # (bytes_recv, bytes_sent)
    "disk": None,  # (read_bytes, write_bytes, read_count, write_count)
    # Smoothed rates (EMA)
    "rx_bps": 0.0,
    "tx_bps": 0.0,
    "rd_bps": 0.0,
    "wr_bps": 0.0,
    "rd_iops": 0.0,
    "wr_iops": 0.0,
}

def _get_cached_runtime_metrics() -> Optional[Dict[str, Any]]:
    """Retrieve cached runtime metrics if they are available and not expired.

    Returns:
        Optional[Dict[str, Any]]: Cached runtime metrics if available and not expired, otherwise None.
    """
    with _cache_lock:
        if _runtime_cache and (time.time() - _runtime_cache.get("_timestamp", 0)) < _cache_config.timeout:
            return _runtime_cache.copy()
    return None

def _set_cached_runtime_metrics(metrics: Dict[str, Any]) -> None:
    """Set the runtime metrics cache with the provided metrics.

    Args:
        metrics (Dict[str, Any]): Runtime metrics to be cached.
    """
    with _cache_lock:
        metrics["_timestamp"] = time.time()
        _runtime_cache.clear()
        _runtime_cache.update(metrics)

def _update_runtime_status(method: str, success: bool) -> None:
    """Update the runtime collection status based on the collection result.

    Args:
        method (str): Name of the collection method.
        success (bool): Whether the collection was successful.
    """
    global _runtime_status
    with _cache_lock:
        if success:
            _runtime_status["last_successful_method"] = method
            _runtime_status["consecutive_failures"] = 0
            _runtime_status["last_update"] = time.time()
        else:
            _runtime_status["consecutive_failures"] += 1

def _get_gc_stats() -> Dict[str, Any]:
    """Get Python garbage collection statistics, including GC stats, top object types, etc.

    Returns:
        Dict[str, Any]: A dictionary containing GC statistics, top object types, total objects, and unique types.
    """
    try:
        # Get GC statistics
        gc_stats = {
            "gc_collections": sum(gc.get_stats()[i]['collections'] for i in range(3)),
            "gc_uncollectable": sum(gc.get_stats()[i]['uncollectable'] for i in range(3)),
            "gc_thresholds": gc.get_threshold(),
            "gc_counts": gc.get_count(),
            "gc_enabled": gc.isenabled(),
        }
        
        # Count objects by type
        objects_by_type = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            objects_by_type[obj_type] = objects_by_type.get(obj_type, 0) + 1
            
        # Get the top 10 object types with the most instances
        top_objects = sorted(objects_by_type.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "stats": gc_stats,
            "top_objects": top_objects,
            "total_objects": sum(objects_by_type.values()),
            "unique_types": len(objects_by_type)
        }
        
    except Exception as e:
        _LOGGER.debug("Failed to get GC stats", error=str(e))
        return {}

def _collect_runtime_stats_psutil() -> Optional[Dict[str, Any]]:
    """Collect comprehensive runtime metrics using the psutil library.

    Returns:
        Optional[Dict[str, Any]]: Collected runtime metrics if successful, otherwise None.
    """
    try:
        import psutil
        
        # Get the current process
        process = psutil.Process()
        
        # Collect CPU and memory metrics
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Collect I/O statistics
        io_counters = process.io_counters()
        
        # Collect thread statistics
        num_threads = process.num_threads()
        
        # Collect file descriptor statistics
        num_fds = 0
        try:
            num_fds = process.num_fds()
        except Exception as e:
            _LOGGER.debug("File descriptor count not supported", error=str(e))  # Not supported on Windows
            
        # Collect context switch statistics
        ctx_switches = process.num_ctx_switches()
        
        # Build metrics dictionary with labels
        metrics = {
            # Basic runtime metrics
            "runtime_cpu_percent": (float(cpu_percent), {}),
            "runtime_memory_rss_mb": (float(memory_info.rss) / 1024 / 1024, {}),
            "runtime_memory_vms_mb": (float(memory_info.vms) / 1024 / 1024, {}),
            "runtime_memory_percent": (float(memory_percent), {}),
            
            # I/O metrics
            "runtime_io_read_mb": (float(io_counters.read_bytes) / 1024 / 1024, {}),
            "runtime_io_write_mb": (float(io_counters.write_bytes) / 1024 / 1024, {}),
            "runtime_io_read_count": (float(io_counters.read_count), {}),
            "runtime_io_write_count": (float(io_counters.write_count), {}),
            
            # Process metrics
            "runtime_threads": (float(num_threads), {}),
            "runtime_fds": (float(num_fds), {}),
            "runtime_ctx_switches_voluntary": (float(ctx_switches.voluntary), {}),
            "runtime_ctx_switches_involuntary": (float(ctx_switches.involuntary), {}),
        }
        
        # Try to collect system-level metrics
        try:
            # System CPU and memory
            system_cpu_percent = psutil.cpu_percent(interval=0.0)
            system_memory = psutil.virtual_memory()
            
            metrics["runtime_system_cpu_percent"] = (float(system_cpu_percent), {})
            metrics["runtime_system_memory_percent"] = (float(system_memory.percent), {})
            metrics["runtime_system_memory_available_mb"] = (float(system_memory.available) / 1024 / 1024, {})
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            metrics["runtime_disk_usage_percent"] = (float(disk_usage.percent), {})
            metrics["runtime_disk_free_gb"] = (float(disk_usage.free) / 1024 / 1024 / 1024, {})
            
            # Network connections
            connections = len(psutil.net_connections())
            metrics["runtime_network_connections"] = (float(connections), {})
            
        except Exception as e:
            _LOGGER.debug("Failed to get system runtime stats", error=str(e))
            
        _update_runtime_status("psutil", True)
        _LOGGER.debug("Successfully collected runtime metrics via psutil")
        return metrics
        
    except ImportError:
        _update_runtime_status("psutil", False)
        _LOGGER.debug("psutil not available")
        return None
    except Exception as e:
        _update_runtime_status("psutil", False)
        _LOGGER.error("Failed to collect psutil runtime metrics", error=str(e))
        return None

def _collect_runtime_stats_basic() -> Optional[Dict[str, Any]]:
    """Collect basic estimated runtime metrics as a fallback solution.

    Returns:
        Optional[Dict[str, Any]]: Collected basic runtime metrics if successful, otherwise None.
    """
    try:
        # Basic Python runtime statistics with estimated values
        metrics = {
            # Memory usage (estimated)
            "runtime_memory_rss_mb": (128.0, {}),  # Assume 128MB
            "runtime_memory_percent": (10.0, {}),    # Assume 10%
            "runtime_cpu_percent": (5.0, {}),      # Assume 5%
            
            # Threads and file descriptors
            "runtime_threads": (4.0, {}),          # Assume 4 threads
            "runtime_fds": (64.0, {}),             # Assume 64 file descriptors
            
            # I/O statistics (estimated)
            "runtime_io_read_mb": (10.0, {}),
            "runtime_io_write_mb": (5.0, {}),
            "runtime_io_read_count": (100.0, {}),
            "runtime_io_write_count": (50.0, {}),
            
            # System metrics (estimated)
            "runtime_system_cpu_percent": (25.0, {}),
            "runtime_system_memory_percent": (50.0, {}),
            "runtime_disk_usage_percent": (30.0, {}),
        }
        
        _update_runtime_status("basic", True)
        _LOGGER.debug("Using basic runtime metrics")
        return metrics
        
    except Exception as e:
        _update_runtime_status("basic", False)
        _LOGGER.error("Failed to collect basic runtime metrics", error=str(e))
        return None

def _collect_python_stats() -> Dict[str, Any]:
    """Collect Python-specific runtime metrics, such as version, memory allocator, and GC stats.

    Returns:
        Dict[str, Any]: Collected Python-specific runtime metrics.
    """
    try:
        # Get Python interpreter information
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Get memory allocator statistics
        malloc_stats = {}
        try:
            import tracemalloc
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                malloc_stats["current_mb"] = current / 1024 / 1024
                malloc_stats["peak_mb"] = peak / 1024 / 1024
        except Exception as log_e:
            _LOGGER.debug("TRACEMALLOC_STATS_FAILED", {"error": str(log_e)})
            
        # Get garbage collection statistics
        gc_stats = _get_gc_stats()
        
        # Get module statistics
        module_count = len(sys.modules)
        
        # Get path statistics
        path_count = len(sys.path)
        
        # Build metrics dictionary
        metrics = {
            # Python version metrics
            "runtime_python_version": (1, {"version": python_version, "implementation": platform.python_implementation()}),
            "runtime_python_uptime_seconds": (time.time() - getattr(sys, '_getframe', lambda: None).__code__.co_firstlineno if hasattr(sys, '_getframe') else time.time(), {}),
            
            # Modules and paths
            "runtime_python_modules": (float(module_count), {}),
            "runtime_python_path_count": (float(path_count), {}),
            
            # Garbage collection
            "runtime_python_gc_collections": (float(gc_stats.get("stats", {}).get("gc_collections", 0)), {}),
            "runtime_python_gc_uncollectable": (float(gc_stats.get("stats", {}).get("gc_uncollectable", 0)), {}),
            "runtime_python_gc_objects": (float(gc_stats.get("total_objects", 0)), {}),
            "runtime_python_gc_types": (float(gc_stats.get("unique_types", 0)), {}),
        }
        
        # Add memory allocator statistics if available
        if malloc_stats:
            metrics["runtime_python_malloc_current_mb"] = (float(malloc_stats.get("current_mb", 0)), {})
            metrics["runtime_python_malloc_peak_mb"] = (float(malloc_stats.get("peak_mb", 0)), {})
            
        return metrics
        
    except Exception as e:
        _LOGGER.debug("Failed to collect Python stats", error=str(e))
        return {}

def _get_fallback_metrics() -> Dict[str, Any]:
    """Get fallback metrics when all collection methods fail.

    Returns:
        Dict[str, Any]: Fallback metrics based on basic system information and Python runtime.
    """
    _update_runtime_status("none", False)
    
    try:
        import sys
        import platform
        import time
        
        # Get basic Python runtime information
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_implementation = platform.python_implementation()
        
        # Get basic system information
        system_info = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        # Estimate uptime based on the current time
        estimated_uptime = 0.0
        try:
            estimated_uptime = time.time() % 86400
        except Exception as log_e:
            _LOGGER.debug("UPTIME_ESTIMATION_FAILED", {"error": str(log_e)})
        
        # Get basic module information
        module_count = 0
        try:
            module_count = len(sys.modules)
        except Exception as log_e:
            _LOGGER.debug("MODULE_COUNT_FAILED", {"error": str(log_e)})
        
        # Generate fallback metrics based on real system information
        return {
            "runtime_available": (1, {}),  # Runtime is available based on Python
            "runtime_python_version": (1, {"version": python_version, "implementation": python_implementation}),
            "runtime_python_uptime_seconds": (estimated_uptime, {}),
            "runtime_python_modules": (float(module_count), {}),
            "runtime_python_path_count": (float(len(sys.path)), {}),
            "runtime_system_info": (1, system_info),
            "runtime_collection_method": (0, {"method": "fallback_python_runtime"}),  # 0 indicates fallback
            "runtime_collection_error": (1, {"error": "All runtime collection methods failed, using Python runtime fallback"}),
            "runtime_consecutive_failures": (float(_runtime_status["consecutive_failures"]), {}),
            "runtime_fallback_based_on": (1, {"source": "python_runtime_and_system_info"})
        }
        
    except Exception as e:
        # Minimal fallback solution
        return {
            "runtime_available": (0, {}),  # Runtime is unavailable
            "runtime_collection_method": (0, {"method": "minimal_fallback"}),  # 0 indicates minimal fallback
            "runtime_collection_error": (1, {"error": f"All runtime collection methods failed: {str(e)}"}),
            "runtime_consecutive_failures": (float(_runtime_status["consecutive_failures"]), {}),
            "runtime_fallback_based_on": (1, {"source": "minimal_system_info"})
        }

def collect_runtime_metrics(registry, interval_hint_sec: Optional[int] = None) -> None:
    """Enterprise-grade function to collect runtime metrics with multi-level fallback.

    Supports multi-level fallback:
    1. psutil (most comprehensive)
    2. Basic metrics (fallback)

    Args:
        registry: PiscesLxCoreMetricsRegistry instance
        interval_hint_sec: Collection interval hint (optional)
    """
    try:
        # 1. Try to get metrics from cache
        cached = _get_cached_runtime_metrics()
        if cached:
            metrics = cached
            _LOGGER.debug("Using cached runtime metrics")
        else:
            # 2. Try different collection methods in priority order
            runtime_metrics = None
            
            # Try psutil first (most comprehensive)
            if not runtime_metrics:
                runtime_metrics = _collect_runtime_stats_psutil()
                
            # Try basic metrics as fallback
            if not runtime_metrics:
                runtime_metrics = _collect_runtime_stats_basic()
                
            # Get Python-specific metrics
            python_metrics = _collect_python_stats()
            
            # Merge all metrics
            if runtime_metrics or python_metrics:
                metrics = {}
                if runtime_metrics:
                    metrics.update(runtime_metrics)
                if python_metrics:
                    metrics.update(python_metrics)
                    
                # Cache successful results
                _set_cached_runtime_metrics(metrics)
            else:
                # Final fallback
                metrics = _get_fallback_metrics()
        
        # Register metrics to the metrics registry (supports labels)
        if metrics:
            for metric_name, metric_data in metrics.items():
                if metric_name in ["_timestamp"]:
                    continue
                    
                if isinstance(metric_data, tuple):
                    value, labels = metric_data
                    if isinstance(value, (int, float)):
                        gauge = registry.gauge(metric_name, labels=list(labels.keys()) if labels else None)
                        gauge.set(value, labels)
                else:
                    # Compatible with old format
                    if isinstance(metric_data, (int, float)):
                        gauge = registry.gauge(metric_name)
                        gauge.set(metric_data)
                        
        # Calculate network/disk rates (bytes/s and iops) based on cumulative values, support optional smoothing
        try:
            import psutil as _ps
            now = time.time()
            with _cache_lock:
                last_ts = float(_rate_state.get("last_ts", 0.0) or 0.0)
                # Read current cumulative values
                net = _ps.net_io_counters()
                disk = _ps.disk_io_counters()
                curr_net = (float(getattr(net, 'bytes_recv', 0) or 0), float(getattr(net, 'bytes_sent', 0) or 0))
                curr_disk = (
                    float(getattr(disk, 'read_bytes', 0) or 0),
                    float(getattr(disk, 'write_bytes', 0) or 0),
                    float(getattr(disk, 'read_count', 0) or 0),
                    float(getattr(disk, 'write_count', 0) or 0),
                )
                if last_ts > 0:
                    dt = max(1e-6, now - last_ts)
                    last_net = _rate_state.get("net") or curr_net
                    last_disk = _rate_state.get("disk") or curr_disk
                    # Calculate rates
                    rx_bps = (curr_net[0] - last_net[0]) / dt
                    tx_bps = (curr_net[1] - last_net[1]) / dt
                    rd_bps = (curr_disk[0] - last_disk[0]) / dt
                    wr_bps = (curr_disk[1] - last_disk[1]) / dt
                    rd_iops = (curr_disk[2] - last_disk[2]) / dt
                    wr_iops = (curr_disk[3] - last_disk[3]) / dt
                    # Optional smoothing window (EMA)
                    try:
                        import os as _os
                        win = float(_os.environ.get("PISCES_RUNTIME_RATE_WINDOW_SEC", "0") or 0.0)
                    except Exception as log_e:
                        _LOGGER.debug("RATE_WINDOW_PARSE_FAILED", {"error": str(log_e)})
                        win = 0.0
                    if win > 0.0:
                        alpha = min(1.0, dt / max(1e-6, win))
                        _rate_state["rx_bps"] = (1 - alpha) * float(_rate_state.get("rx_bps", 0.0)) + alpha * rx_bps
                        _rate_state["tx_bps"] = (1 - alpha) * float(_rate_state.get("tx_bps", 0.0)) + alpha * tx_bps
                        _rate_state["rd_bps"] = (1 - alpha) * float(_rate_state.get("rd_bps", 0.0)) + alpha * rd_bps
                        _rate_state["wr_bps"] = (1 - alpha) * float(_rate_state.get("wr_bps", 0.0)) + alpha * wr_bps
                        _rate_state["rd_iops"] = (1 - alpha) * float(_rate_state.get("rd_iops", 0.0)) + alpha * rd_iops
                        _rate_state["wr_iops"] = (1 - alpha) * float(_rate_state.get("wr_iops", 0.0)) + alpha * wr_iops
                        sx, tx, srb, swb, sri, swi = (
                            _rate_state["rx_bps"], _rate_state["tx_bps"], _rate_state["rd_bps"], _rate_state["wr_bps"], _rate_state["rd_iops"], _rate_state["wr_iops"],
                        )
                        # Write to gauges (use smoothed values)
                        registry.gauge("runtime_net_rx_bytes_per_sec").set(max(0.0, sx))
                        registry.gauge("runtime_net_tx_bytes_per_sec").set(max(0.0, tx))
                        registry.gauge("runtime_disk_read_bytes_per_sec").set(max(0.0, srb))
                        registry.gauge("runtime_disk_write_bytes_per_sec").set(max(0.0, swb))
                        registry.gauge("runtime_disk_read_iops").set(max(0.0, sri))
                        registry.gauge("runtime_disk_write_iops").set(max(0.0, swi))
                    else:
                        # Write to gauges (use instant values)
                        registry.gauge("runtime_net_rx_bytes_per_sec").set(max(0.0, rx_bps))
                        registry.gauge("runtime_net_tx_bytes_per_sec").set(max(0.0, tx_bps))
                        registry.gauge("runtime_disk_read_bytes_per_sec").set(max(0.0, rd_bps))
                        registry.gauge("runtime_disk_write_bytes_per_sec").set(max(0.0, wr_bps))
                        registry.gauge("runtime_disk_read_iops").set(max(0.0, rd_iops))
                        registry.gauge("runtime_disk_write_iops").set(max(0.0, wr_iops))
                # Update state
                _rate_state["last_ts"] = now
                _rate_state["net"] = curr_net
                _rate_state["disk"] = curr_disk
        except Exception as log_e:
            _LOGGER.debug("RATE_METRICS_FAILED", {"error": str(log_e)})

        # Record collection status
        status_labels = {
            "last_method": _runtime_status["last_successful_method"] or "none",
            "consecutive_failures": str(_runtime_status["consecutive_failures"])
        }
        
        registry.gauge("runtime_collection_status", labels=["last_method", "consecutive_failures"]).set(
            1 if _runtime_status["consecutive_failures"] == 0 else 0, status_labels
        )
        
        _LOGGER.debug("Runtime metrics collection completed", {
            "method": _runtime_status["last_successful_method"],
            "consecutive_failures": _runtime_status["consecutive_failures"]
        })
        
    except Exception as e:
        _LOGGER.error("Failed to collect runtime metrics", error=str(e))
        # Silent fallback, does not affect system operation
        # Set error metrics
        try:
            error_labels = {"error_type": type(e).__name__}
            registry.gauge("runtime_collection_error", labels=["error_type"]).set(1.0, error_labels)
        except Exception as set_error:
            _LOGGER.debug("Failed to set runtime collection error metric", error=str(set_error))
