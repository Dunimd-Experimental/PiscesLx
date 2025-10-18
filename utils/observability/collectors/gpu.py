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

import re
import time
import json
import threading
import subprocess
from dataclasses import dataclass
from utils import PiscesLxCoreLog
from typing import Dict, Any, Optional, List, Tuple
from utils.error import PiscesLxCoreObservabilityError

_LOGGER = PiscesLxCoreLog("pisceslx.collectors.gpu")

@dataclass
class GPUCacheConfig:
    """Configuration class for GPU metric cache.

    Attributes:
        timeout (float): Cache timeout in seconds. Default is 5.0 seconds.
        max_retries (int): Maximum number of retries for command execution. Default is 3.
        retry_delay (float): Delay between retries in seconds. Default is 1.0 second.
    """
    timeout: float = 5.0
    max_retries: int = 3
    retry_delay: float = 1.0

# Thread-safe GPU metrics cache
_gpu_cache = {}
_cache_lock = threading.RLock()
_cache_config = GPUCacheConfig()

# GPU metrics status tracking
_gpu_status = {
    "nvidia_smi_available": None,
    "nvml_available": None,
    "torch_cuda_available": None,
    "last_successful_method": None,
    "consecutive_failures": 0,
    "last_update": 0
}

# Prometheus format label keys
GPU_LABELS = ["gpu_index", "gpu_name", "gpu_uuid"]

def _get_cached_gpu_metrics() -> Optional[Dict[str, Any]]:
    """Retrieve cached GPU metrics if they are still valid and not expired.

    Returns:
        Optional[Dict[str, Any]]: Cached GPU metrics if valid and not expired, None otherwise.
    """
    try:
        with _cache_lock:
            if _gpu_cache and (time.time() - _gpu_cache.get("_timestamp", 0)) < _cache_config.timeout:
                return _gpu_cache.copy()
    except Exception as e:
        _LOGGER.debug("Failed to get cached GPU metrics", error=str(e))
    return None

def _set_cached_gpu_metrics(metrics: Dict[str, Any]) -> None:
    """Set the GPU metrics cache with the provided metrics.

    Args:
        metrics (Dict[str, Any]): GPU metrics to be cached.
    """
    try:
        with _cache_lock:
            metrics["_timestamp"] = time.time()
            _gpu_cache.clear()
            _gpu_cache.update(metrics)
    except Exception as e:
        _LOGGER.debug("Failed to set GPU metrics cache", error=str(e))

def _update_gpu_status(method: str, success: bool) -> None:
    """Update the GPU collection status based on the collection result.

    Args:
        method (str): Name of the collection method, e.g., "nvidia_smi", "nvml", "torch_cuda".
        success (bool): Whether the collection was successful.
    """
    global _gpu_status
    try:
        with _cache_lock:
            if success:
                _gpu_status["last_successful_method"] = method
                _gpu_status["consecutive_failures"] = 0
                _gpu_status["last_update"] = time.time()
            else:
                _gpu_status["consecutive_failures"] += 1
            
            if method == "nvidia_smi":
                _gpu_status["nvidia_smi_available"] = success
            elif method == "nvml":
                _gpu_status["nvml_available"] = success
            elif method == "torch_cuda":
                _gpu_status["torch_cuda_available"] = success
    except Exception as e:
        _LOGGER.debug("Failed to update GPU status", error=str(e))

def _execute_with_retry(cmd: List[str], timeout: int = 10) -> Optional[subprocess.CompletedProcess]:
    """Execute a command with a retry mechanism.

    Args:
        cmd (List[str]): Command to execute, represented as a list of strings.
        timeout (int, optional): Timeout for each command execution in seconds. Defaults to 10.

    Returns:
        Optional[subprocess.CompletedProcess]: Completed process object if the command succeeds, None otherwise.
    """
    for attempt in range(_cache_config.max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return result
                
            _LOGGER.warning(f"Command attempt {attempt + 1} failed", {
                "cmd": " ".join(cmd),
                "returncode": result.returncode,
                "stderr": result.stderr[:200]
            })
            
        except subprocess.TimeoutExpired:
            _LOGGER.warning(f"Command attempt {attempt + 1} timed out", {
                "cmd": " ".join(cmd),
                "timeout": timeout
            })
        except Exception as e:
            _LOGGER.warning(f"Command attempt {attempt + 1} error", {
                "cmd": " ".join(cmd),
                "error": str(e)
            })
            
        if attempt < _cache_config.max_retries - 1:
            time.sleep(_cache_config.retry_delay * (2 ** attempt))  # Exponential backoff
            
    return None

def _query_nvidia_smi() -> Optional[Dict[str, Any]]:
    """Retrieve GPU metrics using the nvidia-smi command.

    Returns:
        Optional[Dict[str, Any]]: Collected GPU metrics if the operation is successful, None otherwise.
    """
    try:
        # Check nvidia-smi availability
        check_result = subprocess.run(["nvidia-smi", "--version"], capture_output=True, timeout=5)
        if check_result.returncode != 0:
            _update_gpu_status("nvidia_smi", False)
            return None
            
        # Complete GPU query command
        cmd = [
            "nvidia-smi", 
            "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics,clocks.max.graphics,fan.speed",
            "--format=csv,noheader,nounits"
        ]
        
        result = _execute_with_retry(cmd, timeout=15)
        if not result:
            _update_gpu_status("nvidia_smi", False)
            return None
            
        metrics = {}
        lines = result.stdout.strip().split('\n')
        gpu_count = 0
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 10:
                gpu_idx = int(parts[0])
                gpu_name = parts[1]
                gpu_uuid = parts[2]
                
                # Create labels
                labels = {
                    "gpu_index": str(gpu_idx),
                    "gpu_name": gpu_name,
                    "gpu_uuid": gpu_uuid,
                    "vendor": "nvidia",
                }
                
                # Core metrics
                metrics[f"gpu_utilization_percent"] = (float(parts[3]) if parts[3] != '[N/A]' else 0.0, labels)
                metrics[f"gpu_memory_used_mb"] = (float(parts[4]) if parts[4] != '[N/A]' else 0.0, labels)
                metrics[f"gpu_memory_total_mb"] = (float(parts[5]) if parts[5] != '[N/A]' else 0.0, labels)
                metrics[f"gpu_temperature_celsius"] = (float(parts[6]) if parts[6] != '[N/A]' else 0.0, labels)
                metrics[f"gpu_power_draw_watts"] = (float(parts[7]) if parts[7] != '[N/A]' else 0.0, labels)
                metrics[f"gpu_clock_current_mhz"] = (float(parts[8]) if parts[8] != '[N/A]' else 0.0, labels)
                metrics[f"gpu_clock_max_mhz"] = (float(parts[9]) if parts[9] != '[N/A]' else 0.0, labels)
                metrics[f"gpu_fan_speed_percent"] = (float(parts[9]) if parts[9] != '[N/A]' else 0.0, labels)
                
                # Calculate derived metrics
                memory_util = (float(parts[4]) / float(parts[5]) * 100) if parts[4] != '[N/A]' and parts[5] != '[N/A]' else 0.0
                metrics[f"gpu_memory_utilization_percent"] = (memory_util, labels)
                
                gpu_count += 1
                
        # System-level metrics
        metrics["gpu_available"] = (1, {})
        metrics["gpu_count"] = (gpu_count, {})
        metrics["gpu_collection_method"] = (1, {"method": "nvidia_smi"})  # 1 indicates nvidia-smi
        
        _update_gpu_status("nvidia_smi", True)
        _LOGGER.info(f"Successfully collected GPU metrics via nvidia-smi", gpu_count=gpu_count)
        return metrics
        
    except Exception as e:
        _update_gpu_status("nvidia_smi", False)
        _LOGGER.error("Failed to query nvidia-smi", error=str(e))
        return None

def _query_nvml() -> Optional[Dict[str, Any]]:
    """Query NVIDIA GPUs using NVML (pynvml) to obtain accurate utilization and telemetry data.

    Returns:
        Optional[Dict[str, Any]]: Collected GPU metrics if the operation is successful, None otherwise.
    """
    try:
        import pynvml  # type: ignore
    except Exception as log_e:
        _update_gpu_status("nvml", False)
        _LOGGER.debug("NVML_IMPORT_FAILED", error=str(log_e))
        return None
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        metrics: Dict[str, Any] = {}
        gpu_count = int(count)
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode("utf-8", errors="ignore") if hasattr(pynvml.nvmlDeviceGetName(h), 'decode') else str(pynvml.nvmlDeviceGetName(h))
            try:
                uuid = pynvml.nvmlDeviceGetUUID(h).decode("utf-8")
            except Exception as log_e:
                _LOGGER.debug("NVML_UUID_FAILED", error=str(log_e))
                uuid = "unknown"
            labels = {"gpu_index": str(i), "gpu_name": name, "gpu_uuid": uuid, "vendor": "nvidia"}
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                metrics["gpu_utilization_percent"] = (float(util.gpu), labels)
            except Exception as log_e:
                _LOGGER.debug("NVML_UTILIZATION_FAILED", error=str(log_e))
                metrics["gpu_utilization_percent"] = (0.0, labels)
            # Memory
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                metrics["gpu_memory_used_mb"] = (float(mem.used) / 1024 / 1024, labels)
                metrics["gpu_memory_total_mb"] = (float(mem.total) / 1024 / 1024, labels)
                mu = (float(mem.used) / float(mem.total) * 100.0) if mem.total else 0.0
                metrics["gpu_memory_utilization_percent"] = (mu, labels)
            except Exception as log_e:
                _LOGGER.debug("NVML_MEMORY_INFO_FAILED", error=str(log_e))
            # Temperature
            try:
                t = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                metrics["gpu_temperature_celsius"] = (float(t), labels)
            except Exception as log_e:
                _LOGGER.debug("NVML_TEMPERATURE_FAILED", error=str(log_e))
            # Power draw
            try:
                p = pynvml.nvmlDeviceGetPowerUsage(h)  # mW
                metrics["gpu_power_draw_watts"] = (float(p) / 1000.0, labels)
            except Exception as log_e:
                _LOGGER.debug("NVML_POWER_DRAW_FAILED", error=str(log_e))
            # Clocks
            try:
                cc = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
                metrics["gpu_clock_current_mhz"] = (float(cc), labels)
            except Exception as log_e:
                _LOGGER.debug("NVML_CLOCK_INFO_FAILED", error=str(log_e))
            try:
                mc = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)
                metrics["gpu_clock_max_mhz"] = (float(mc), labels)
            except Exception as log_e:
                _LOGGER.debug("NVML_MAX_CLOCK_INFO_FAILED", error=str(log_e))
        metrics["gpu_available"] = (1, {})
        metrics["gpu_count"] = (gpu_count, {})
        metrics["gpu_collection_method"] = (4, {"method": "nvml"})  # 4 for nvml
        _update_gpu_status("nvml", True)
        try:
            pynvml.nvmlShutdown()
        except Exception as log_e:
            _LOGGER.debug("NVML_SHUTDOWN_FAILED", error=str(log_e))
        _LOGGER.info("Successfully collected GPU metrics via NVML", gpu_count=gpu_count)
        return metrics
    except Exception as e:
        _update_gpu_status("nvml", False)
        _LOGGER.debug("Failed to query NVML", error=str(e))
        try:
            pynvml.nvmlShutdown()
        except Exception as log_e:
            _LOGGER.debug("NVML_SHUTDOWN_FAILED", error=str(log_e))
        return None

def _query_torch_cuda() -> Optional[Dict[str, Any]]:
    """Retrieve GPU metrics using torch.cuda.

    Returns:
        Optional[Dict[str, Any]]: Collected GPU metrics if the operation is successful, None otherwise.
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            _update_gpu_status("torch_cuda", False)
            return None
            
        metrics = {}
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            
            # Create labels
            labels = {
                "gpu_index": str(i),
                "gpu_name": props.name,
                "gpu_uuid": "unknown"  # torch cannot get UUID
            }
            
            # Basic memory metrics
            total_memory_mb = props.total_memory / 1024 / 1024
            allocated_memory_mb = allocated / 1024 / 1024
            reserved_memory_mb = reserved / 1024 / 1024
            
            metrics[f"gpu_memory_total_mb"] = (total_memory_mb, labels)
            metrics[f"gpu_memory_used_mb"] = (allocated_memory_mb, labels)
            metrics[f"gpu_memory_reserved_mb"] = (reserved_memory_mb, labels)
            
            # Memory utilization
            memory_util = (allocated_memory_mb / total_memory_mb * 100) if total_memory_mb > 0 else 0.0
            metrics[f"gpu_memory_utilization_percent"] = (memory_util, labels)
            
            # Improved utilization estimation (based on memory usage pattern)
            # Use memory usage and reservation rates to estimate compute utilization
            reserved_util = (reserved_memory_mb / total_memory_mb * 100) if total_memory_mb > 0 else 0.0
            estimated_compute_util = min(95.0, (memory_util + reserved_util) * 0.6)
            metrics[f"gpu_utilization_percent"] = (estimated_compute_util, labels)
            
            # Mark unavailable metrics
            metrics[f"gpu_temperature_celsius"] = (-1.0, labels)  # Mark as unavailable
            metrics[f"gpu_power_draw_watts"] = (-1.0, labels)  # Mark as unavailable
            metrics[f"gpu_clock_current_mhz"] = (-1.0, labels)  # Mark as unavailable
            metrics[f"gpu_clock_max_mhz"] = (-1.0, labels)  # Mark as unavailable
            metrics[f"gpu_fan_speed_percent"] = (-1.0, labels)  # Mark as unavailable
            
        # System-level metrics
        metrics["gpu_available"] = (1, {})
        metrics["gpu_count"] = (device_count, {})
        metrics["gpu_collection_method"] = (2, {"method": "torch_cuda"})  # 2 indicates torch.cuda
        
        _update_gpu_status("torch_cuda", True)
        _LOGGER.info(f"Successfully collected GPU metrics via torch.cuda", gpu_count=device_count)
        return metrics
        
    except ImportError:
        _update_gpu_status("torch_cuda", False)
        _LOGGER.debug("PyTorch not available")
        return None
    except Exception as e:
        _update_gpu_status("torch_cuda", False)
        _LOGGER.error("Failed to query torch.cuda", error=str(e))
        return None

def _query_rocm_smi() -> Optional[Dict[str, Any]]:
    """Retrieve AMD GPU metrics using the rocm-smi command.

    Returns:
        Optional[Dict[str, Any]]: Collected GPU metrics if the operation is successful, None otherwise.
    """
    try:
        # Check rocm-smi availability
        check_result = subprocess.run(["rocm-smi", "--version"], capture_output=True, timeout=5)
        if check_result.returncode != 0:
            return None
            
        cmd = ["rocm-smi", "--showtemp", "--showuse", "--showmeminfo", "vram", "--json"]
        result = _execute_with_retry(cmd, timeout=15)
        
        if not result:
            return None
            
        rocm_data = json.loads(result.stdout)
        metrics = {}
        gpu_count = 0
        
        for gpu_id, gpu_data in rocm_data.items():
            # Create labels
            labels = {
                "gpu_index": str(gpu_id),
                "gpu_name": gpu_data.get("Card name", "AMD GPU"),
                "gpu_uuid": gpu_data.get("GPU ID", "unknown"),
                "vendor": "amd",
            }
            
            # Temperature
            temp = gpu_data.get("Temperature (Sensor memory) (C)", "0")
            metrics[f"gpu_temperature_celsius"] = (float(temp) if temp != "N/A" else 0.0, labels)
            
            # Utilization
            util = gpu_data.get("GPU use (%)", "0")
            metrics[f"gpu_utilization_percent"] = (float(util) if util != "N/A" else 0.0, labels)
            
            # Memory information
            mem_info = gpu_data.get("VRAM Memory Used", "0 B")
            if "MB" in mem_info:
                used_mb = float(mem_info.replace(" MB", ""))
                metrics[f"gpu_memory_used_mb"] = (used_mb, labels)
                
            gpu_count += 1
            
        # System-level metrics
        metrics["gpu_available"] = (1, {})
        metrics["gpu_count"] = (gpu_count, {})
        metrics["gpu_collection_method"] = (3, {"method": "rocm_smi"})  # 3 indicates rocm-smi
        
        _LOGGER.info(f"Successfully collected GPU metrics via rocm-smi", gpu_count=gpu_count)
        return metrics
        
    except Exception as e:
        _LOGGER.error("Failed to query rocm-smi", error=str(e))
        return None

def _get_fallback_metrics() -> Dict[str, Any]:
    """Obtain fallback metrics for environments without GPUs or when GPU metrics collection fails.

    Returns:
        Dict[str, Any]: Fallback GPU metrics based on system detection and configuration.
    """
    _update_gpu_status("none", False)
    
    try:
        import subprocess
        import os
        
        # Detect if the system supports GPUs based on actual detection results
        gpu_detected = False
        detection_methods = []
        
        # Try to detect NVIDIA GPUs
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                                  capture_output=True, timeout=5, text=True)
            if result.returncode == 0 and result.stdout.strip():
                gpu_detected = True
                detection_methods.append("nvidia-smi")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Try to detect AMD GPUs
        try:
            result = subprocess.run(["rocm-smi", "--showproductname"], 
                                  capture_output=True, timeout=5, text=True)
            if result.returncode == 0:
                gpu_detected = True
                detection_methods.append("rocm-smi")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
            
        # Detect PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                gpu_detected = True
                detection_methods.append("torch-cuda")
        except ImportError:
            pass
            
        # Build basic system metrics
        metrics = {}
        
        if gpu_detected:
            # GPUs are available but detailed metrics cannot be obtained
            metrics["gpu_available"] = (1, {})
            metrics["gpu_count"] = (0, {})  # Unknown count
            metrics["gpu_detection_methods"] = (1, {"methods": ",".join(detection_methods)})
            metrics["gpu_collection_method"] = (0, {"method": "fallback_detected"})
            
            # Provide fallback warning metrics
            metrics["gpu_fallback_warning"] = (1, {"reason": "detection_available_but_collection_failed"})
        else:
            # No GPU environment
            metrics["gpu_available"] = (0, {})
            metrics["gpu_count"] = (0, {})
            metrics["gpu_collection_method"] = (0, {"method": "none"})
            metrics["gpu_fallback_warning"] = (0, {"reason": "no_gpu_detected"})
        
        # Add system-level fallback metrics
        metrics["gpu_collection_status"] = (1 if gpu_detected else 0, {})
        metrics["gpu_detection_timestamp"] = (time.time(), {})
        
        _LOGGER.info("Using fallback GPU metrics", 
                    gpu_detected=gpu_detected,
                    detection_methods=detection_methods,
                    metrics_count=len(metrics))
        
        return metrics
        
    except Exception as e:
        _LOGGER.error("Failed to generate fallback GPU metrics", error=str(e))
        # Basic fallback metrics
        return {
            "gpu_available": (0, {}),
            "gpu_count": (0, {}),
            "gpu_collection_method": (0, {"method": "error_fallback"}),
            "gpu_fallback_error": (1, {"error": str(e)})
        }

def collect_gpu_metrics(registry) -> None:
    """Enterprise-level GPU metrics collection function.

    Supports multi-level fallback mechanisms:
    1. AMD GPU (rocm-smi)
    2. NVIDIA GPU (NVML)
    3. NVIDIA GPU (nvidia-smi)
    4. PyTorch CUDA
    5. Final fallback (no GPU)

    Args:
        registry: PiscesLxCoreMetricsRegistry instance
    """
    try:
        # 1. Try to get from cache
        cached = _get_cached_gpu_metrics()
        if cached:
            metrics = cached
            _LOGGER.debug("Using cached GPU metrics")
        else:
            # 2. Try different collection methods in priority order
            metrics = None
            
            # Try AMD GPU (rocm-smi)
            if not metrics:
                metrics = _query_rocm_smi()
                
            # Try NVIDIA GPU (NVML)
            if not metrics:
                metrics = _query_nvml()

            # Try NVIDIA GPU (nvidia-smi)
            if not metrics:
                metrics = _query_nvidia_smi()
                
            # Try PyTorch CUDA
            if not metrics:
                metrics = _query_torch_cuda()
                
            # Final fallback
            if not metrics:
                metrics = _get_fallback_metrics()
            else:
                # Cache successful results
                _set_cached_gpu_metrics(metrics)
        
        # Register to the metrics registry (support labels)
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
            "last_method": _gpu_status["last_successful_method"] or "none",
            "consecutive_failures": str(_gpu_status["consecutive_failures"])
        }
        
        registry.gauge("gpu_collection_status", labels=["last_method", "consecutive_failures"]).set(
            1 if _gpu_status["consecutive_failures"] == 0 else 0, status_labels
        )
        
        _LOGGER.debug("GPU metrics collection completed", 
                     method=_gpu_status["last_successful_method"],
                     gpu_count=metrics.get("gpu_count", [0])[0] if isinstance(metrics.get("gpu_count"), tuple) else 0,
                     consecutive_failures=_gpu_status["consecutive_failures"])
        
    except Exception as e:
        _LOGGER.error("Failed to collect GPU metrics", error=str(e))
        # Silent fallback, does not affect system operation
        # Set error metrics
        try:
            error_labels = {"error_type": type(e).__name__}
            registry.gauge("gpu_collection_error", labels=["error_type"]).set(1.0, error_labels)
        except Exception as log_e:
            _LOGGER.debug("GPU_COLLECTION_ERROR_METRICS_FAILED", error=str(log_e))
