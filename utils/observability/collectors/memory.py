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

import psutil
from datetime import datetime
from .base import BaseCollector
from dataclasses import dataclass
from typing import Dict, Any, Optional
from ..registry import PiscesLxCoreMetricsRegistry
from utils.log.core import PiscesLxCoreLog

@dataclass
class MemoryMetrics:
    """Data structure for memory metrics."""
    # System memory
    total_memory_gb: float  # Total system memory in gigabytes
    available_memory_gb: float  # Available system memory in gigabytes
    used_memory_gb: float  # Used system memory in gigabytes
    memory_usage_percent: float  # System memory usage percentage
    
    # Swap memory
    total_swap_gb: float  # Total swap memory in gigabytes
    used_swap_gb: float  # Used swap memory in gigabytes
    free_swap_gb: float  # Free swap memory in gigabytes
    swap_usage_percent: float  # Swap memory usage percentage
    
    # Process memory
    process_memory_mb: float  # Resident Set Size (RSS) of the process in megabytes
    process_memory_percent: float  # Memory percentage used by the process
    process_virtual_memory_mb: float  # Virtual memory size of the process in megabytes
    
    # Detailed memory information
    cached_memory_gb: float  # Cached memory in gigabytes
    buffered_memory_gb: float  # Buffered memory in gigabytes
    shared_memory_gb: float  # Shared memory in gigabytes
    
    timestamp: datetime  # Timestamp when the metrics were collected

class PiscesLxCoreMemoryCollector(BaseCollector):
    """Collector for memory metrics."""
    
    def __init__(self, registry: Optional[PiscesLxCoreMetricsRegistry] = None):
        """
        Initialize the memory metrics collector.

        Args:
            registry (Optional[PiscesLxCoreMetricsRegistry]): Metrics registry to register collected metrics. Defaults to None.
        """
        super().__init__("memory", registry)
        self.logger = PiscesLxCoreLog("pisceslx.observability.memory")
        self._process = psutil.Process()
        
    def collect(self) -> Dict[str, Any]:
        """
        Collect memory metrics.

        Returns:
            Dict[str, Any]: A dictionary containing memory metrics. If collection fails, returns fallback metrics.
        """
        try:
            metrics = self._collect_memory_metrics()
            self._register_metrics(metrics)
            self._record_success()
            return metrics.__dict__
        except Exception as e:
            self.logger.error(f"Failed to collect memory metrics: {e}")
            self._record_error()
            return self._get_fallback_metrics()
    
    def _collect_memory_metrics(self) -> MemoryMetrics:
        """
        Collect detailed memory metrics.

        Returns:
            MemoryMetrics: An instance of MemoryMetrics containing detailed memory metrics.
        """
        # Get system memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Get process memory information
        process_memory = self._process.memory_info()
        process_memory_percent = self._process.memory_percent()
        
        # Get detailed memory information if available
        try:
            cached_gb = getattr(memory, 'cached', 0) / (1024**3)
            buffered_gb = getattr(memory, 'buffers', 0) / (1024**3)
            shared_gb = getattr(memory, 'shared', 0) / (1024**3)
        except AttributeError:
            # These fields may not be available on Windows systems
            cached_gb = 0.0
            buffered_gb = 0.0
            shared_gb = 0.0
        
        return MemoryMetrics(
            # System memory
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            used_memory_gb=(memory.total - memory.available) / (1024**3),
            memory_usage_percent=memory.percent,
            
            # Swap memory
            total_swap_gb=swap.total / (1024**3),
            used_swap_gb=swap.used / (1024**3),
            free_swap_gb=swap.free / (1024**3),
            swap_usage_percent=(swap.percent if swap.total > 0 else 0.0),
            
            # Process memory
            process_memory_mb=process_memory.rss / (1024**2),
            process_memory_percent=process_memory_percent,
            process_virtual_memory_mb=process_memory.vms / (1024**2),
            
            # Detailed memory information
            cached_memory_gb=cached_gb,
            buffered_memory_gb=buffered_gb,
            shared_memory_gb=shared_gb,
            
            timestamp=datetime.now()
        )
    
    def _register_metrics(self, metrics: MemoryMetrics) -> None:
        """
        Register memory metrics to the metrics registry.

        Args:
            metrics (MemoryMetrics): An instance of MemoryMetrics containing memory metrics to be registered.
        """
        if not self.registry:
            return
            
        # System memory metrics
        self.registry.gauge("system_memory_total_gb", "Total system memory in GB").set(metrics.total_memory_gb)
        self.registry.gauge("system_memory_used_gb", "Used system memory in GB").set(metrics.used_memory_gb)
        self.registry.gauge("system_memory_available_gb", "Available system memory in GB").set(metrics.available_memory_gb)
        self.registry.gauge("system_memory_usage_percent", "System memory usage percentage").set(metrics.memory_usage_percent)
        
        # Swap memory metrics
        self.registry.gauge("swap_memory_total_gb", "Total swap memory in GB").set(metrics.total_swap_gb)
        self.registry.gauge("swap_memory_used_gb", "Used swap memory in GB").set(metrics.used_swap_gb)
        self.registry.gauge("swap_memory_free_gb", "Free swap memory in GB").set(metrics.free_swap_gb)
        self.registry.gauge("swap_memory_usage_percent", "Swap memory usage percentage").set(metrics.swap_usage_percent)
        
        # Process memory metrics
        self.registry.gauge("process_memory_rss_mb", "Process RSS memory in MB").set(metrics.process_memory_mb)
        self.registry.gauge("process_memory_percent", "Process memory percentage").set(metrics.process_memory_percent)
        self.registry.gauge("process_memory_virtual_mb", "Process virtual memory in MB").set(metrics.process_virtual_memory_mb)
        
        # Detailed memory information
        self.registry.gauge("memory_cached_gb", "Cached memory in GB").set(metrics.cached_memory_gb)
        self.registry.gauge("memory_buffered_gb", "Buffered memory in GB").set(metrics.buffered_memory_gb)
        self.registry.gauge("memory_shared_gb", "Shared memory in GB").set(metrics.shared_memory_gb)
        
        # Memory usage alert metrics
        if metrics.memory_usage_percent > 90:
            self.registry.counter("memory_high_usage_alerts", "High memory usage alerts").inc()
        
        if metrics.swap_usage_percent > 80:
            self.registry.counter("swap_high_usage_alerts", "High swap usage alerts").inc()
    
    def _get_fallback_metrics(self) -> Dict[str, Any]:
        """
        Get fallback metrics when memory metrics collection fails.

        Returns:
            Dict[str, Any]: A dictionary containing fallback memory metrics.
        """
        current_time = datetime.now()
        
        # Generate fallback metrics based on basic system information
        try:
            # Try to get basic memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "system_memory_total_gb": memory.total / (1024**3),
                "system_memory_used_gb": (memory.total - memory.available) / (1024**3),
                "system_memory_available_gb": memory.available / (1024**3),
                "system_memory_usage_percent": memory.percent,
                "swap_memory_total_gb": swap.total / (1024**3),
                "swap_memory_used_gb": swap.used / (1024**3),
                "swap_memory_free_gb": swap.free / (1024**3),
                "swap_memory_usage_percent": swap.percent if swap.total > 0 else 0.0,
                "process_memory_rss_mb": 0.0,  # Process information is unavailable
                "process_memory_percent": 0.0,
                "process_memory_virtual_mb": 0.0,
                "memory_cached_gb": 0.0,
                "memory_buffered_gb": 0.0,
                "memory_shared_gb": 0.0,
                "timestamp": current_time.isoformat(),
                "status": "degraded",
                "error": "Failed to collect detailed metrics"
            }
        except Exception as e:
            # Fall back to basic estimates
            self.logger.error(f"Failed to get fallback memory metrics: {e}")
            return {
                "system_memory_total_gb": 16.0,  # Assume 16GB
                "system_memory_used_gb": 8.0,
                "system_memory_available_gb": 8.0,
                "system_memory_usage_percent": 50.0,
                "swap_memory_total_gb": 4.0,
                "swap_memory_used_gb": 1.0,
                "swap_memory_free_gb": 3.0,
                "swap_memory_usage_percent": 25.0,
                "process_memory_rss_mb": 512.0,
                "process_memory_percent": 3.2,
                "process_memory_virtual_mb": 1024.0,
                "memory_cached_gb": 2.0,
                "memory_buffered_gb": 1.0,
                "memory_shared_gb": 0.5,
                "timestamp": current_time.isoformat(),
                "status": "fallback",
                "error": "Using fallback estimates"
            }
    
    def get_health_status(self) -> Dict[str, str]:
        """
        Get the health status of the collector.

        Returns:
            Dict[str, str]: A dictionary containing the collector's name, status, success count, error count, and last collection time.
        """
        return {
            "name": self.name,
            "status": "healthy" if self.success_count > self.error_count else "degraded",
            "success_count": str(self.success_count),
            "error_count": str(self.error_count),
            "last_collection": self.last_collection_time.isoformat() if self.last_collection_time else "never"
        }

def collect_memory_metrics(registry: Optional[PiscesLxCoreMetricsRegistry] = None) -> Dict[str, Any]:
    """
    A convenient function to collect memory metrics.

    Args:
        registry (Optional[PiscesLxCoreMetricsRegistry]): Metrics registry to register collected metrics. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing memory metrics.
    """
    collector = PiscesLxCoreMemoryCollector(registry)
    return collector.collect()