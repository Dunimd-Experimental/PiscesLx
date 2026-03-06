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

import time
from typing import List, Dict, Any, Optional


class PiscesLxCoreEnhancedCacheManager:
    """Simple cache manager fallback."""
    def __init__(self):
        self._cache: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any, ttl: float = 300.0) -> None:
        self._cache[key] = {"value": value, "expires": time.time() + ttl}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() < entry["expires"]:
                return entry["value"]
            del self._cache[key]
        return None
        

class PiscesLxMonitorAlertManager:
    """Alert manager for system monitoring with caching and threshold checking."""
    
    def __init__(self, config, cache):
        """Initialize the alert manager."""
        self.config = config
        self.cache = cache
        self.alert_thresholds = config.thresholds if hasattr(config, 'thresholds') else {
            'cpu_percent': 90,
            'memory_percent': 90,
            'gpu_util': 95,
            'gpu_mem_percent': 95,
            'disk_percent': 90,
            'error_rate': 0.01
        }
    
    def check(self, stats: Dict[str, Any]) -> List[str]:
        """Check for system alerts based on thresholds."""
        alerts = []
        
        if stats.get('cpu_percent_total', 0) > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU: {stats['cpu_percent_total']:.1f}%")
        
        mem = stats.get('memory', {}) or {}
        if mem.get('percent', 0) > self.alert_thresholds['memory_percent']:
            alerts.append(f"High Memory: {mem['percent']:.1f}%")
        
        for i, gpu in enumerate(stats.get('gpu', []) or []):
            if gpu.get('util', 0) > self.alert_thresholds['gpu_util']:
                alerts.append(f"GPU {i} High Util: {gpu['util']}%")
            if gpu.get('mem_percent', 0) > self.alert_thresholds['gpu_mem_percent']:
                alerts.append(f"GPU {i} High Mem: {gpu['mem_percent']:.1f}%")
        
        for disk in stats.get('disk_usage', []) or []:
            if disk.get('percent', 0) > self.alert_thresholds['disk_percent']:
                alerts.append(f"Disk {disk['mountpoint']}: {disk['percent']}%")
        
        if alerts:
            self.cache.set('last_alerts', {'timestamp': time.time(), 'alerts': alerts})
        
        return alerts
    
    def check_alerts(self, stats: Dict[str, Any], observability_metrics: Optional[Dict[str, Any]] = None) -> List[str]:
        """Check for system alerts and log them."""
        return self.check(stats)
    
    def update_threshold(self, metric: str, threshold: float) -> None:
        """Update alert threshold for a specific metric."""
        if metric in self.alert_thresholds:
            old_threshold = self.alert_thresholds[metric]
            self.alert_thresholds[metric] = threshold
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current alert thresholds."""
        return self.alert_thresholds.copy()
