#!/usr/bin/env/python3

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

import psutil
from typing import Dict, Any, Optional, List
from utils.concurrency import PiscesLxCoreRetry
from utils import PiscesLxCoreEnhancedCacheManager
from utils import PiscesLxCoreDeviceFacade

class PiscesLxMonitorStatsCollector:
    """System statistics collector with caching and retry mechanisms."""
    
    def __init__(self, cache_manager: PiscesLxCoreEnhancedCacheManager, 
                 device_manager: PiscesLxCoreDeviceFacade):
        """Initialize the stats collector with cache and device managers."""
        self.cache_manager = cache_manager
        self.device_manager = device_manager
        self.gpu_enabled = False
        self.gpu_count = 0
        self._init_gpu_detection()
    
    def _init_gpu_detection(self):
        """Initialize GPU detection using device manager with fallback."""
        try:
            gpu_devices = self.device_manager.get_gpu_devices()
            self.gpu_count = len(gpu_devices)
            if self.gpu_count > 0:
                self.gpu_enabled = True
        except Exception:
            # Fallback to pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                if self.gpu_count > 0:
                    self.gpu_enabled = True
            except Exception:
                self.gpu_enabled = False
    
    @PiscesLxCoreRetry(max_attempts=2, delay=0.5)
    def get_cpu_info(self) -> Optional[Dict[str, Any]]:
        """Get CPU information with caching."""
        try:
            # Check cache first
            cache_key = "cpu_info"
            cached_info = self.cache_manager.get(cache_key)
            if cached_info and time.time() - cached_info.get('timestamp', 0) < 2.0:
                return cached_info.get('data')
            
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            cpu_stats = psutil.cpu_stats()
            
            cpu_info = {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "cpu_freq_current": cpu_freq.current if cpu_freq else 0,
                "cpu_freq_max": cpu_freq.max if cpu_freq else 0,
                "cpu_ctx_switches": cpu_stats.ctx_switches,
                "cpu_interrupts": cpu_stats.interrupts
            }
            
            # Cache the result
            self.cache_manager.set(cache_key, {
                'timestamp': time.time(),
                'data': cpu_info
            }, ttl=2.0)
            
            return cpu_info
        except Exception:
            return None
    
    @PiscesLxCoreRetry(max_attempts=2, delay=0.5)
    def get_gpu_stats(self) -> Optional[List[Dict[str, Any]]]:
        """Get GPU statistics with device manager integration."""
        if not self.gpu_enabled:
            return None
        
        try:
            # Check cache first
            cache_key = "gpu_stats"
            cached_stats = self.cache_manager.get(cache_key)
            if cached_stats and time.time() - cached_stats.get('timestamp', 0) < 2.0:
                return cached_stats.get('data')
            
            gpu_stats = []
            
            # Try device manager first
            try:
                gpu_devices = self.device_manager.get_gpu_devices()
                for device in gpu_devices:
                    gpu_info = self.device_manager.get_gpu_info(device)
                    if gpu_info:
                        gpu_stats.append({
                            'name': gpu_info.get("name", "Unknown"),
                            'util': gpu_info.get("gpu_utilization", 0),
                            'mem_total': gpu_info.get("memory_total", 0),
                            'mem_used': gpu_info.get("memory_used", 0),
                            'mem_percent': gpu_info.get("memory_percent", 0)
                        })
            except Exception:
                # Fallback to pynvml
                if self.gpu_count > 0:
                    import pynvml
                    for i in range(self.gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        name = pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                        gpu_stats.append({
                            'name': name,
                            'util': util.gpu,
                            'mem_total': mem_info.total,
                            'mem_used': mem_info.used,
                            'mem_percent': (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
                        })
            
            # Cache the results
            self.cache_manager.set(cache_key, {
                'timestamp': time.time(),
                'data': gpu_stats
            }, ttl=2.0)
            
            return gpu_stats
        except Exception:
            return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Gather all system statistics."""
        stats = {}
        
        # CPU usage statistics
        stats['cpu_percent_total'] = psutil.cpu_percent(percpu=False)
        stats['cpu_percent_per_core'] = psutil.cpu_percent(percpu=True)
        try:
            stats['cpu_freq'] = psutil.cpu_freq(percpu=True)
        except Exception:
            stats['cpu_freq'] = None
        
        # Memory usage statistics
        mem = psutil.virtual_memory()
        stats['memory'] = {
            'total': mem.total,
            'available': mem.available,
            'percent': mem.percent,
            'used': mem.used,
            'free': mem.free
        }
        
        # Swap usage statistics
        swap = psutil.swap_memory()
        stats['swap'] = {
            'total': swap.total,
            'used': swap.used,
            'free': swap.free,
            'percent': swap.percent
        }
        
        # Disk usage statistics
        disk = psutil.disk_usage('/')
        stats['disk'] = {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': (disk.used / disk.total) * 100 if disk.total > 0 else 0
        }
        
        # Network statistics
        net_io = psutil.net_io_counters()
        stats['network'] = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errin': net_io.errin,
            'errout': net_io.errout,
            'dropin': net_io.dropin,
            'dropout': net_io.dropout
        }
        
        # Disk I/O statistics
        disk_io = psutil.disk_io_counters()
        if disk_io:
            stats['disk_io'] = {
                'read_count': disk_io.read_count,
                'write_count': disk_io.write_count,
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes,
                'read_time': disk_io.read_time,
                'write_time': disk_io.write_time
            }
        else:
            stats['disk_io'] = None
        
        # Process statistics
        stats['process_count'] = len(psutil.pids())
        
        # Boot time
        stats['boot_time'] = psutil.boot_time()
        
        return stats