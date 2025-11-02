#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from utils import PiscesLxCoreEnhancedCacheManager


class PiscesLxMonitorDataManager:
    """Data manager for monitoring data with buffering and persistence."""
    
    def __init__(self, cache_manager: PiscesLxCoreEnhancedCacheManager, logger, 
                 log_interval: int = 60, buffer_size: int = 60):
        """Initialize the data manager."""
        self.cache_manager = cache_manager
        self.logger = logger
        self.log_interval = log_interval
        self.buffer_size = buffer_size
        
        # Data buffer for storing samples
        self.data_buffer = []
        self.last_recorded_averages = {}
        self.last_log_time = time.time()
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            'cpu_percent_total': 20,
            'memory_percent': 10,
            'gpu_util': 15,
            'disk_percent': 5
        }
    
    def add_sample(self, stats: Dict[str, Any], net_io: Optional[Any], 
                   disk_io: Optional[Any], force_save: bool = False) -> bool:
        """Add a monitoring sample to the buffer and save if needed."""
        try:
            # Add sample to buffer
            sample = {
                'timestamp': time.time(),
                'stats': stats,
                'net_io': {
                    'bytes_sent': net_io.bytes_sent if net_io else 0,
                    'bytes_recv': net_io.bytes_recv if net_io else 0,
                    'packets_sent': net_io.packets_sent if net_io else 0,
                    'packets_recv': net_io.packets_recv if net_io else 0
                } if net_io else None,
                'disk_io': {
                    'read_count': disk_io.read_count if disk_io else 0,
                    'write_count': disk_io.write_count if disk_io else 0,
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                } if disk_io else None
            }
            
            self.data_buffer.append(sample)
            
            # Check if we should save data
            should_save = force_save or self._should_save_data()
            
            if should_save and self.data_buffer:
                self._save_monitor_data()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error("Error adding monitoring sample", error=str(e))
            return False
    
    def _should_save_data(self) -> bool:
        """Determine if data should be saved based on time interval and anomalies."""
        current_time = time.time()
        
        # Check time interval
        if current_time - self.last_log_time >= self.log_interval:
            return True
        
        # Check for anomalies if we have previous data
        if self.last_recorded_averages and len(self.data_buffer) >= 5:
            current_averages = self._calculate_averages()
            
            # Check CPU anomaly
            if 'cpu_percent_total' in current_averages and 'cpu_percent_total' in self.last_recorded_averages:
                cpu_diff = abs(current_averages['cpu_percent_total'] - 
                              self.last_recorded_averages['cpu_percent_total'])
                if cpu_diff > self.anomaly_thresholds['cpu_percent_total']:
                    self.logger.info(f"CPU anomaly detected: {cpu_diff:.1f}% change")
                    return True
            
            # Check memory anomaly
            if 'memory_percent' in current_averages and 'memory_percent' in self.last_recorded_averages:
                mem_diff = abs(current_averages['memory_percent'] - 
                              self.last_recorded_averages['memory_percent'])
                if mem_diff > self.anomaly_thresholds['memory_percent']:
                    self.logger.info(f"Memory anomaly detected: {mem_diff:.1f}% change")
                    return True
            
            # Check GPU anomaly
            if 'gpu_util_avg' in current_averages and 'gpu_util_avg' in self.last_recorded_averages:
                gpu_diff = abs(current_averages['gpu_util_avg'] - 
                              self.last_recorded_averages['gpu_util_avg'])
                if gpu_diff > self.anomaly_thresholds['gpu_util']:
                    self.logger.info(f"GPU anomaly detected: {gpu_diff:.1f}% change")
                    return True
        
        return False
    
    def _calculate_averages(self) -> Dict[str, float]:
        """Calculate average values from the data buffer."""
        if not self.data_buffer:
            return {}
        
        # Calculate averages for key metrics
        cpu_total_sum = 0
        memory_percent_sum = 0
        gpu_util_sum = 0
        gpu_count = 0
        disk_percent_sum = 0
        
        for sample in self.data_buffer:
            stats = sample.get('stats', {})
            
            # CPU average
            if 'cpu_percent_total' in stats:
                cpu_total_sum += stats['cpu_percent_total']
            
            # Memory average
            if 'memory' in stats and 'percent' in stats['memory']:
                memory_percent_sum += stats['memory']['percent']
            
            # GPU average
            if 'gpu' in stats and stats['gpu']:
                for gpu in stats['gpu']:
                    if 'util' in gpu:
                        gpu_util_sum += gpu['util']
                        gpu_count += 1
            
            # Disk average
            if 'disk' in stats and 'percent' in stats['disk']:
                disk_percent_sum += stats['disk']['percent']
        
        sample_count = len(self.data_buffer)
        averages = {}
        
        if sample_count > 0:
            averages['cpu_percent_total'] = cpu_total_sum / sample_count
            averages['memory_percent'] = memory_percent_sum / sample_count
            averages['disk_percent'] = disk_percent_sum / sample_count
            if gpu_count > 0:
                averages['gpu_util_avg'] = gpu_util_sum / gpu_count
        
        return averages
    
    def _save_monitor_data(self) -> None:
        """Save monitoring data and update state."""
        try:
            if not self.data_buffer:
                return
            
            # Calculate averages
            averages = self._calculate_averages()
            
            # Log summary
            self.logger.info(f"Monitor data saved: {len(self.data_buffer)} samples, "
                           f"CPU avg: {averages.get('cpu_percent_total', 0):.1f}%, "
                           f"Memory avg: {averages.get('memory_percent', 0):.1f}%, "
                           f"GPU avg: {averages.get('gpu_util_avg', 0):.1f}%")
            
            # Cache the averages for anomaly detection
            self.last_recorded_averages = averages
            self.last_log_time = time.time()
            
            # Clear buffer
            self.data_buffer.clear()
            
        except Exception as e:
            self.logger.error("Error saving monitor data", error=str(e))
    
    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self.data_buffer)
    
    def clear_buffer(self) -> None:
        """Clear the data buffer."""
        self.data_buffer.clear()
    
    def get_cached_averages(self) -> Dict[str, float]:
        """Get last recorded averages."""
        return self.last_recorded_averages.copy()