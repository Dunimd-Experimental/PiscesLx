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

import psutil
from typing import Dict, Any, Optional, List


class PiscesLxMonitorStatsCollector:
    """System statistics collector."""

    def __init__(self, cache_manager=None, device_manager=None):
        self.cache_manager = cache_manager
        self.device_manager = device_manager
        self.gpu_enabled = False
        self.gpu_count = 0
        self.pynvml = None
        self._init_gpu()

    def _init_gpu(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            if self.gpu_count > 0:
                self.gpu_enabled = True
                self.pynvml = pynvml
        except Exception:
            pass

    def collect(self) -> Dict[str, Any]:
        stats = {}
        
        stats['cpu_percent_total'] = psutil.cpu_percent(percpu=False)
        stats['cpu_percent_per_core'] = psutil.cpu_percent(percpu=True)
        try:
            stats['cpu_freq'] = psutil.cpu_freq(percpu=True)
        except Exception:
            stats['cpu_freq'] = []
        
        mem = psutil.virtual_memory()
        stats['memory'] = {
            'total': mem.total,
            'used': mem.used,
            'free': mem.free,
            'percent': mem.percent
        }
        
        swap = psutil.swap_memory()
        stats['swap'] = {
            'total': swap.total,
            'used': swap.used,
            'percent': swap.percent
        }
        
        if self.gpu_enabled:
            stats['gpu'] = self._collect_gpu()
        
        stats['disk_usage'] = self._collect_disk()
        stats['network'] = self._collect_network()
        
        return stats

    def _collect_gpu(self) -> List[Dict[str, Any]]:
        gpu_stats = []
        for i in range(self.gpu_count):
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
            util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = self.pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            gpu_stats.append({
                'name': name,
                'util': util.gpu,
                'mem_total': mem_info.total,
                'mem_used': mem_info.used,
                'mem_percent': (mem_info.used / mem_info.total) * 100 if mem_info.total > 0 else 0
            })
        return gpu_stats

    def _collect_disk(self) -> List[Dict[str, Any]]:
        disk_stats = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_stats.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'total': usage.total,
                    'used': usage.used,
                    'percent': usage.percent
                })
            except (PermissionError, FileNotFoundError):
                continue
        return disk_stats

    def _collect_network(self) -> Dict[str, Any]:
        net = psutil.net_io_counters()
        return {
            'bytes_sent': net.bytes_sent,
            'bytes_recv': net.bytes_recv,
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv
        }

    def get_cpu_usage(self) -> float:
        return psutil.cpu_percent(interval=0.1)

    def get_memory_usage(self) -> Dict[str, Any]:
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "percent": mem.percent,
            "used": mem.used,
            "free": mem.free
        }

    def get_disk_usage(self) -> Dict[str, Any]:
        disk = psutil.disk_usage('/')
        return {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }

    def get_network_io(self) -> Dict[str, Any]:
        net = psutil.net_io_counters()
        return {
            "bytes_sent": net.bytes_sent,
            "bytes_recv": net.bytes_recv,
            "packets_sent": net.packets_sent,
            "packets_recv": net.packets_recv
        }

    def get_gpu_usage(self) -> List[Dict[str, Any]]:
        if self.gpu_enabled:
            return self._collect_gpu()
        return []

    def get_all_stats(self) -> Dict[str, Any]:
        return self.collect()

    def get_system_stats(self) -> Dict[str, Any]:
        return self.collect()

    def shutdown(self):
        if self.gpu_enabled and self.pynvml:
            self.pynvml.nvmlShutdown()
