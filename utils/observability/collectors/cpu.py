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

import time
import threading
import platform
import subprocess
from dataclasses import dataclass
from utils.log.core import PiscesLxCoreLog
from typing import Dict, Optional, List, Tuple, Any
from utils.error import PiscesLxCoreObservabilityError

_LOGGER = PiscesLxCoreLog("pisceslx.collectors.cpu")

@dataclass
class CPUCacheConfig:
    """Configuration class for caching CPU metrics.

    Attributes:
        timeout (float): Cache expiration time in seconds. Default is 3.0 seconds.
        max_retries (int): Maximum number of retries. Default is 2.
        retry_delay (float): Delay between retries in seconds. Default is 0.5 seconds.
    """
    timeout: float = 3.0
    max_retries: int = 2
    retry_delay: float = 0.5

# Thread-safe cache for CPU metrics
_cpu_cache = {}
_cache_lock = threading.RLock()
_cache_config = CPUCacheConfig()

# Tracking status of CPU metrics collection
_cpu_status = {
    "last_successful_method": None,
    "consecutive_failures": 0,
    "last_update": 0,
    "cpu_count": None
}

def _get_cached_cpu_metrics() -> Optional[Dict[str, Any]]:
    """Retrieve cached CPU metrics if they are still valid.

    Returns:
        Optional[Dict[str, Any]]: A copy of the cached CPU metrics if valid; otherwise, None.
    """
    with _cache_lock:
        if _cpu_cache and (time.time() - _cpu_cache.get("_timestamp", 0)) < _cache_config.timeout:
            return _cpu_cache.copy()
    return None

def _set_cached_cpu_metrics(metrics: Dict[str, Any]) -> None:
    """Cache the provided CPU metrics.

    Args:
        metrics (Dict[str, Any]): CPU metrics to be cached.
    """
    with _cache_lock:
        metrics["_timestamp"] = time.time()
        _cpu_cache.clear()
        _cpu_cache.update(metrics)

def _update_cpu_status(method: str, success: bool) -> None:
    """Update the status of CPU metrics collection.

    Args:
        method (str): The collection method used.
        success (bool): Whether the collection was successful.
    """
    global _cpu_status
    with _cache_lock:
        if success:
            _cpu_status["last_successful_method"] = method
            _cpu_status["consecutive_failures"] = 0
            _cpu_status["last_update"] = time.time()
        else:
            _cpu_status["consecutive_failures"] += 1

def _execute_with_retry(cmd: List[str], timeout: int = 5) -> Optional[subprocess.CompletedProcess]:
    """Execute a command with a retry mechanism and exponential backoff.

    Args:
        cmd (List[str]): Command to execute, represented as a list of strings.
        timeout (int, optional): Timeout for each command execution in seconds. Defaults to 5.

    Returns:
        Optional[subprocess.CompletedProcess]: The result of the command if successful; otherwise, None.
    """
    for attempt in range(_cache_config.max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return result
                
            _LOGGER.warning(f"Command attempt {attempt + 1} failed", {
                "cmd": " ".join(cmd),
                "returncode": result.returncode,
                "stderr": result.stderr[:100]
            })
            
        except subprocess.TimeoutExpired:
            _LOGGER.warning(f"Command attempt {attempt + 1} timed out", {
                "cmd": " ".join(cmd),
                "timeout": timeout
            })
        except Exception as e:
            _LOGGER.warning(f"Command attempt {attempt + 1} encountered an error", {
                "cmd": " ".join(cmd),
                "error": str(e)
            })
            
        if attempt < _cache_config.max_retries - 1:
            time.sleep(_cache_config.retry_delay * (2 ** attempt))  # Exponential backoff
            
    return None

def _get_cpu_count() -> int:
    """Retrieve the number of CPU cores using multiple methods.

    Returns:
        int: The number of CPU cores. If all methods fail, defaults to 1.
    """
    if _cpu_status["cpu_count"] is not None:
        return _cpu_status["cpu_count"]
        
    try:
        # Method 1: Use multiprocessing module
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # Method 2: Use os module for verification
        import os
        os_count = os.cpu_count()
        if os_count and os_count != cpu_count:
            _LOGGER.debug("CPU count mismatch", {
                "multiprocessing": cpu_count,
                "os": os_count
            })
            cpu_count = max(cpu_count, os_count)  # Take the larger value
            
        # Method 3: Platform-specific detection
        if platform.system() == "Linux":
            try:
                result = subprocess.run(["nproc"], capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    nproc_count = int(result.stdout.strip())
                    if nproc_count != cpu_count:
                        _LOGGER.debug("CPU count from nproc differs", {
                            "nproc": nproc_count,
                            "current": cpu_count
                        })
                        cpu_count = nproc_count
            except:
                pass
                
        _cpu_status["cpu_count"] = cpu_count
        return cpu_count
        
    except Exception as e:
        _LOGGER.error("Failed to get CPU core count", {"error": str(e)})
        return 1  # Fallback to 1 core

def _collect_cpu_stats_psutil() -> Optional[Dict[str, Any]]:
    """Collect CPU metrics using the psutil library.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing CPU metrics if collection is successful; otherwise, None.
    """
    try:
        import psutil
        
        # Get CPU statistics
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        
        # Get memory statistics
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Get load statistics (Unix systems only)
        load_avg = None
        if hasattr(psutil, "getloadavg"):
            try:
                load_avg = psutil.getloadavg()
            except:
                pass
        
        # Get process statistics
        process_count = len(psutil.pids())
        
        # Build metrics dictionary with labels
        metrics = {
            # System-level metrics
            "cpu_available": (1, {}),
            "cpu_count": (float(cpu_count), {}),
            "cpu_utilization_percent": (float(cpu_percent), {}),
            
            # Memory metrics
            "cpu_memory_total_mb": (float(mem.total) / 1024 / 1024, {}),
            "cpu_memory_used_mb": (float(mem.used) / 1024 / 1024, {}),
            "cpu_memory_available_mb": (float(mem.available) / 1024 / 1024, {}),
            "cpu_memory_percent": (float(mem.percent), {}),
            "cpu_memory_cached_mb": (float(getattr(mem, 'cached', 0)) / 1024 / 1024, {}),
            "cpu_memory_buffers_mb": (float(getattr(mem, 'buffers', 0)) / 1024 / 1024, {}),
            
            # Swap metrics
            "cpu_swap_total_mb": (float(swap.total) / 1024 / 1024, {}),
            "cpu_swap_used_mb": (float(swap.used) / 1024 / 1024, {}),
            "cpu_swap_percent": (float(swap.percent), {}),
            
            # Process metrics
            "cpu_process_count": (float(process_count), {}),
            
            # Frequency metrics
            "cpu_frequency_mhz": (float(cpu_freq.current) if cpu_freq else 0.0, {}),
            "cpu_frequency_max_mhz": (float(cpu_freq.max) if cpu_freq else 0.0, {}),
            "cpu_frequency_min_mhz": (float(cpu_freq.min) if cpu_freq else 0.0, {}),
        }
        
        # Add load metrics if available
        if load_avg:
            metrics["cpu_load_1min"] = (float(load_avg[0]), {})
            metrics["cpu_load_5min"] = (float(load_avg[1]), {})
            metrics["cpu_load_15min"] = (float(load_avg[2]), {})
            
        # Add CPU time statistics
        cpu_times = psutil.cpu_times()
        total_time = sum(cpu_times)
        if total_time > 0:
            metrics["cpu_time_user_percent"] = (float(cpu_times.user / total_time * 100), {})
            metrics["cpu_time_system_percent"] = (float(cpu_times.system / total_time * 100), {})
            metrics["cpu_time_idle_percent"] = (float(cpu_times.idle / total_time * 100), {})
            
            # Add platform-specific CPU time metrics
            if hasattr(cpu_times, 'iowait'):
                metrics["cpu_time_iowait_percent"] = (float(cpu_times.iowait / total_time * 100), {})
            if hasattr(cpu_times, 'irq'):
                metrics["cpu_time_irq_percent"] = (float(cpu_times.irq / total_time * 100), {})
            if hasattr(cpu_times, 'nice'):
                metrics["cpu_time_nice_percent"] = (float(cpu_times.nice / total_time * 100), {})
                
        _update_cpu_status("psutil", True)
        _LOGGER.debug("Successfully collected CPU metrics using psutil")
        return metrics
        
    except ImportError:
        _update_cpu_status("psutil", False)
        _LOGGER.debug("psutil library is not available")
        return None
    except Exception as e:
        _update_cpu_status("psutil", False)
        _LOGGER.error("Failed to collect CPU metrics using psutil", {"error": str(e)})
        return None

def _collect_cpu_stats_proc() -> Optional[Dict[str, Any]]:
    """Collect CPU metrics from the /proc filesystem (Linux-specific).

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing CPU metrics if collection is successful; otherwise, None.
    """
    try:
        if platform.system() != "Linux":
            return None
            
        # Read CPU statistics from /proc/stat
        with open("/proc/stat", "r") as f:
            stat_lines = f.readlines()
            
        cpu_stats = {}
        for line in stat_lines:
            if line.startswith("cpu"):
                parts = line.strip().split()
                cpu_name = parts[0]
                if cpu_name == "cpu":  # Total CPU statistics
                    # user, nice, system, idle, iowait, irq, softirq, steal, guest, guest_nice
                    values = list(map(float, parts[1:11]))
                    total = sum(values)
                    if total > 0:
                        cpu_stats["total"] = {
                            "user": values[0] / total * 100,
                            "nice": values[1] / total * 100,
                            "system": values[2] / total * 100,
                            "idle": values[3] / total * 100,
                            "iowait": values[4] / total * 100,
                            "irq": values[5] / total * 100,
                            "softirq": values[6] / total * 100,
                            "steal": values[7] / total * 100,
                            "guest": values[8] / total * 100,
                            "guest_nice": values[9] / total * 100,
                        }
                        
        # Read memory statistics from /proc/meminfo
        with open("/proc/meminfo", "r") as f:
            meminfo_lines = f.readlines()
            
        mem_stats = {}
        for line in meminfo_lines:
            if ":" in line:
                key, value = line.split(":")
                value = value.strip().replace(" kB", "")
                try:
                    mem_stats[key.strip()] = float(value) / 1024  # Convert to MB
                except:
                    pass
                    
        # Read load average from /proc/loadavg
        try:
            with open("/proc/loadavg", "r") as f:
                loadavg = f.read().strip().split()
                load_1min = float(loadavg[0])
                load_5min = float(loadavg[1])
                load_15min = float(loadavg[2])
        except:
            load_1min = load_5min = load_15min = 0.0
            
        cpu_count = _get_cpu_count()
        
        # Build metrics dictionary
        metrics = {
            "cpu_available": (1, {}),
            "cpu_count": (float(cpu_count), {}),
            "cpu_load_1min": (load_1min, {}),
            "cpu_load_5min": (load_5min, {}),
            "cpu_load_15min": (load_15min, {}),
        }
        
        # Add CPU time distribution metrics
        if "total" in cpu_stats:
            total = cpu_stats["total"]
            metrics["cpu_time_user_percent"] = (total["user"], {})
            metrics["cpu_time_system_percent"] = (total["system"], {})
            metrics["cpu_time_idle_percent"] = (total["idle"], {})
            metrics["cpu_time_iowait_percent"] = (total["iowait"], {})
            metrics["cpu_utilization_percent"] = (100.0 - total["idle"], {})
            
        # Add memory statistics metrics
        if "MemTotal" in mem_stats:
            total_mb = mem_stats["MemTotal"]
            available_mb = mem_stats.get("MemAvailable", total_mb * 0.8)  # Estimation
            used_mb = total_mb - available_mb
            
            metrics["cpu_memory_total_mb"] = (total_mb, {})
            metrics["cpu_memory_used_mb"] = (used_mb, {})
            metrics["cpu_memory_available_mb"] = (available_mb, {})
            metrics["cpu_memory_percent"] = (used_mb / total_mb * 100, {})
            
            if "Cached" in mem_stats:
                metrics["cpu_memory_cached_mb"] = (mem_stats["Cached"], {})
            if "Buffers" in mem_stats:
                metrics["cpu_memory_buffers_mb"] = (mem_stats["Buffers"], {})
                
        # Add swap statistics metrics
        if "SwapTotal" in mem_stats:
            swap_total_mb = mem_stats["SwapTotal"]
            swap_free_mb = mem_stats.get("SwapFree", swap_total_mb)
            swap_used_mb = swap_total_mb - swap_free_mb
            
            metrics["cpu_swap_total_mb"] = (swap_total_mb, {})
            metrics["cpu_swap_used_mb"] = (swap_used_mb, {})
            metrics["cpu_swap_percent"] = (swap_used_mb / swap_total_mb * 100 if swap_total_mb > 0 else 0, {})
            
        _update_cpu_status("proc", True)
        _LOGGER.debug("Successfully collected CPU metrics from /proc filesystem")
        return metrics
        
    except Exception as e:
        _update_cpu_status("proc", False)
        _LOGGER.error("Failed to collect CPU metrics from /proc filesystem", {"error": str(e)})
        return None

def _collect_cpu_stats_basic() -> Optional[Dict[str, Any]]:
    """Collect basic CPU metrics using platform information as a fallback method.

    Returns:
        Optional[Dict[str, Any]]: Basic CPU metrics if collection is successful; otherwise, None.
    """
    try:
        import os
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        
        # Get actual load average from system
        try:
            if hasattr(os, 'getloadavg'):
                load_1min, load_5min, load_15min = os.getloadavg()
            else:
                # For Windows or systems without getloadavg, try alternative methods
                load_1min = load_5min = load_15min = _get_windows_load_average() if os.name == 'nt' else 0.0
        except:
            load_1min = load_5min = load_15min = 0.0
        
        metrics = {
            "cpu_available": (1, {}),
            "cpu_count": (float(cpu_count), {}),
            "cpu_load_1min": (load_1min, {}),
            "cpu_load_5min": (load_5min, {}),
            "cpu_load_15min": (load_15min, {}),
            "cpu_utilization_percent": (min(100.0, load_1min * 100.0 / max(1, cpu_count)), {}),
            "cpu_collection_method": (4, {"method": "basic"}),  # 4 indicates basic method
        }
        
        _update_cpu_status("basic", True)
        _LOGGER.debug("Successfully collected basic CPU metrics")
        return metrics
        
    except Exception as e:
        _update_cpu_status("basic", False)
        _LOGGER.error("Failed to collect basic CPU metrics", {"error": str(e)})
        return None

def _get_windows_load_average() -> tuple:
    """Get load average on Windows using performance counters.
    
    Returns:
        tuple: (load_1min, load_5min, load_15min) load averages
    """
    try:
        import subprocess
        import re
        
        # Try to get CPU queue length as a proxy for load average
        result = subprocess.run([
            'typeperf', '\\Processor(_Total)\\Processor Queue Length',
            '-sc', '1', '-y'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            # Parse the output to get the queue length
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if re.search(r'\d+\.?\d*', line):
                    queue_length = float(re.search(r'\d+\.?\d*', line).group())
                    # Convert queue length to approximate load average
                    # This is a rough approximation based on system behavior
                    load_1min = max(0.0, queue_length / 4.0)  # Approximate conversion
                    load_5min = load_1min * 0.8  # Smooth over time
                    load_15min = load_1min * 0.6
                    return (load_1min, load_5min, load_15min)
    except Exception:
        pass
    
    return (0.0, 0.0, 0.0)

def _get_fallback_metrics() -> Dict[str, Any]:
    """Get fallback metrics when all collection methods fail.

    Returns:
        Dict[str, Any]: A dictionary containing fallback CPU metrics based on system information.
    """
    _update_cpu_status("none", False)
    
    cpu_count = _get_cpu_count()
    
    try:
        import os
        import multiprocessing
        
        # Get basic system information
        cpu_count = multiprocessing.cpu_count()
        
        # Try to get load average
        try:
            if hasattr(os, 'getloadavg'):
                load_1min, load_5min, load_15min = os.getloadavg()
            else:
                load_1min = load_5min = load_15min = _get_windows_load_average() if os.name == 'nt' else 0.0
        except:
            load_1min = load_5min = load_15min = 0.0
        
        # Calculate basic metrics based on system information
        estimated_utilization = min(100.0, load_1min * 100.0 / max(1, cpu_count))
        
        return {
            "cpu_available": (1, {}),  # CPU is available based on system information
            "cpu_count": (float(cpu_count), {}),
            "cpu_load_1min": (load_1min, {}),
            "cpu_load_5min": (load_5min, {}),
            "cpu_load_15min": (load_15min, {}),
            "cpu_utilization_percent": (estimated_utilization, {}),
            "cpu_collection_method": (0, {"method": "fallback_system_info"}),  # 0 indicates fallback
            "cpu_collection_error": (1, {"error": "All CPU collection methods failed, using system info fallback"}),
            "cpu_consecutive_failures": (float(_cpu_status["consecutive_failures"]), {}),
            "cpu_fallback_based_on": (1, {"source": "system_load_and_core_count"})
        }
        
    except Exception as e:
        # Minimal fallback metrics
        return {
            "cpu_available": (0, {}),  # CPU is unavailable
            "cpu_count": (float(cpu_count), {}),
            "cpu_utilization_percent": (0.0, {}),
            "cpu_collection_method": (0, {"method": "minimal_fallback"}),  # 0 indicates minimal fallback
            "cpu_collection_error": (1, {"error": f"All CPU collection methods failed: {str(e)}"}),
            "cpu_consecutive_failures": (float(_cpu_status["consecutive_failures"]), {}),
            "cpu_fallback_based_on": (1, {"source": "minimal_system_info"})
        }

def collect_cpu_metrics(registry) -> None:
    """Collect enterprise-level CPU metrics with multi-level fallback strategies.

    The fallback strategies are as follows:
    1. psutil (most comprehensive)
    2. /proc filesystem (Linux)
    3. Basic metrics (fallback)

    Args:
        registry: An instance of PiscesLxCoreMetricsRegistry.
    """
    try:
        # 1. Try to get metrics from cache
        cached = _get_cached_cpu_metrics()
        if cached:
            metrics = cached
            _LOGGER.debug("Using cached CPU metrics")
        else:
            # 2. Try different collection methods in priority order
            metrics = None
            
            # Try psutil first (most comprehensive)
            if not metrics:
                metrics = _collect_cpu_stats_psutil()
                
            # Try /proc filesystem (Linux only)
            if not metrics:
                metrics = _collect_cpu_stats_proc()
                
            # Try basic metrics as fallback
            if not metrics:
                metrics = _collect_cpu_stats_basic()
                
            # Final fallback
            if not metrics:
                metrics = _get_fallback_metrics()
            else:
                # Cache successful results
                _set_cached_cpu_metrics(metrics)
        
        # Register metrics to the registry (supports labels)
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
                        
        # Record collection status
        status_labels = {
            "last_method": _cpu_status["last_successful_method"] or "none",
            "consecutive_failures": str(_cpu_status["consecutive_failures"])
        }
        
        registry.gauge("cpu_collection_status", labels=["last_method", "consecutive_failures"]).set(
            1 if _cpu_status["consecutive_failures"] == 0 else 0, status_labels
        )
        
        cpu_count_value = metrics.get("cpu_count", (0,))[0] if isinstance(metrics.get("cpu_count"), tuple) else 0
        _LOGGER.debug("CPU metrics collection completed", {
            "method": _cpu_status["last_successful_method"],
            "cpu_count": cpu_count_value,
            "consecutive_failures": _cpu_status["consecutive_failures"]
        })
        
    except Exception as e:
        _LOGGER.error("Failed to collect CPU metrics", {"error": str(e)})
        # Silent fallback, does not affect system operation
        # Set error metrics
        try:
            error_labels = {"error_type": type(e).__name__}
            registry.gauge("cpu_collection_error", labels=["error_type"]).set(1.0, error_labels)
        except:
            pass