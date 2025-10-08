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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import List, Dict, Any, Optional
from utils.concurrency import PiscesLxCoreTimeout
from utils.cache.enhanced import PiscesLxCoreEnhancedCacheManager

class PiscesLxMonitorAlertManager:
    """Alert manager for system monitoring with caching and timeout protection."""
    
    def __init__(self, cache_manager: PiscesLxCoreEnhancedCacheManager, logger):
        """Initialize the alert manager."""
        self.cache_manager = cache_manager
        self.logger = logger
        self.alert_thresholds = {
            'cpu_percent': 90,
            'memory_percent': 90,
            'gpu_util': 95,
            'gpu_mem_percent': 95,
            'disk_percent': 90,
            'error_rate': 0.01  # 1%
        }
    
    @PiscesLxCoreTimeout(seconds=10)
    def check_alerts(self, stats: Dict[str, Any], observability_metrics: Optional[Dict[str, Any]] = None) -> List[str]:
        """Check for system alerts and log them."""
        try:
            alerts = []
            
            # Check CPU usage
            if stats and 'cpu_percent_total' in stats:
                cpu_percent = stats['cpu_percent_total']
                if cpu_percent > self.alert_thresholds['cpu_percent']:
                    alerts.append(f"High CPU usage: {cpu_percent:.1f}%")
                    self.logger.warning(f"ALERT: High CPU usage detected: {cpu_percent:.1f}%")
            
            # Check memory usage
            if stats and 'memory' in stats:
                mem_percent = stats['memory'].get('percent', 0)
                if mem_percent > self.alert_thresholds['memory_percent']:
                    alerts.append(f"High memory usage: {mem_percent:.1f}%")
                    self.logger.warning(f"ALERT: High memory usage detected: {mem_percent:.1f}%")
            
            # Check GPU usage
            if stats and 'gpu' in stats:
                for i, gpu in enumerate(stats['gpu']):
                    gpu_util = gpu.get('util', 0)
                    if gpu_util > self.alert_thresholds['gpu_util']:
                        alerts.append(f"GPU {i} high utilization: {gpu_util}%")
                        self.logger.warning(f"ALERT: GPU {i} high utilization: {gpu_util}%")
                    
                    gpu_mem_percent = gpu.get('mem_percent', 0)
                    if gpu_mem_percent > self.alert_thresholds['gpu_mem_percent']:
                        alerts.append(f"GPU {i} high memory usage: {gpu_mem_percent:.1f}%")
                        self.logger.warning(f"ALERT: GPU {i} high memory usage: {gpu_mem_percent:.1f}%")
            
            # Check disk usage
            if stats and 'disk' in stats:
                disk_percent = stats['disk'].get('percent', 0)
                if disk_percent > self.alert_thresholds['disk_percent']:
                    alerts.append(f"High disk usage: {disk_percent:.1f}%")
                    self.logger.warning(f"ALERT: High disk usage detected: {disk_percent:.1f}%")
            
            # Check observability metrics
            if observability_metrics:
                error_rate = observability_metrics.get('error_rate', 0)
                if error_rate > self.alert_thresholds['error_rate']:
                    alerts.append(f"High error rate: {error_rate:.4f}")
                    self.logger.warning(f"ALERT: High error rate detected: {error_rate:.4f}")
                
                # Log metrics to file only
                throughput = observability_metrics.get('throughput', 0)
                p95_latency = observability_metrics.get('p95_latency', 0)
                self.logger.info(f"Observability metrics: throughput={throughput:.2f}, "
                               f"p95_latency={p95_latency:.2f}, error_rate={error_rate:.4f}")
            
            # Cache alerts for history tracking
            if alerts:
                self.cache_manager.set("last_alerts", {
                    'timestamp': time.time(),
                    'alerts': alerts
                }, ttl=300.0)  # Keep for 5 minutes
            
            return alerts
        except Exception as e:
            self.logger.error("Error checking alerts", error=str(e))
            return []
    
    def update_threshold(self, metric: str, threshold: float) -> None:
        """Update alert threshold for a specific metric."""
        if metric in self.alert_thresholds:
            old_threshold = self.alert_thresholds[metric]
            self.alert_thresholds[metric] = threshold
            self.logger.info(f"Updated alert threshold for {metric}: {old_threshold} -> {threshold}")
        else:
            self.logger.warning(f"Unknown metric for threshold update: {metric}")
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current alert thresholds."""
        return self.alert_thresholds.copy()