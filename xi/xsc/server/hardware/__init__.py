#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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

"""
Hardware monitoring for Xi Server.

Provides GPU monitoring for NVIDIA, AMD, and Intel GPUs.
"""

import sys
from typing import List, Dict, Any
from datetime import datetime

from ...core.types import XiSystemStats


def get_nvidia_gpus() -> List[Dict[str, Any]]:
    """Get NVIDIA GPU information using pynvml."""
    gpus = []
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            
            temp = 0.0
            power = 0.0
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                pass
            
            gpus.append({
                "vendor": "nvidia",
                "name": name if isinstance(name, str) else name.decode(),
                "utilization": util.gpu,
                "memory_used_gb": mem.used / (1024**3),
                "memory_total_gb": mem.total / (1024**3),
                "temperature": temp,
                "power_draw": power,
            })
        
        pynvml.nvmlShutdown()
    except Exception:
        pass
    
    return gpus


def get_amd_gpus() -> List[Dict[str, Any]]:
    """Get AMD GPU information using pyamdgpuinfo or rocm-smi."""
    gpus = []
    
    try:
        import pyamdgpuinfo
        gpu_count = pyamdgpuinfo.detect_gpus()
        
        for i in range(gpu_count):
            gpu = pyamdgpuinfo.get_gpu(i)
            gpus.append({
                "vendor": "amd",
                "name": gpu.name if hasattr(gpu, 'name') else f"AMD GPU {i}",
                "utilization": gpu.load * 100 if hasattr(gpu, 'load') else 0.0,
                "memory_used_gb": gpu.memory_used / (1024**3) if hasattr(gpu, 'memory_used') else 0.0,
                "memory_total_gb": gpu.memory_total / (1024**3) if hasattr(gpu, 'memory_total') else 0.0,
                "temperature": gpu.temperature if hasattr(gpu, 'temperature') else 0.0,
                "power_draw": gpu.power if hasattr(gpu, 'power') else 0.0,
            })
    except ImportError:
        pass
    
    if not gpus:
        try:
            import subprocess
            import json
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "--showtemp", "--showpower", "--json"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if "card" in data:
                    for card_id, card_data in data["card"].items():
                        gpus.append({
                            "vendor": "amd",
                            "name": card_data.get("Card series", f"AMD GPU {card_id}"),
                            "utilization": float(card_data.get("GPU use (%)", 0)),
                            "memory_used_gb": float(card_data.get("GPU memory used (MB)", 0)) / 1024,
                            "memory_total_gb": float(card_data.get("GPU memory total (MB)", 0)) / 1024,
                            "temperature": float(card_data.get("Temperature (Sensor edge) (C)", 0)),
                            "power_draw": float(card_data.get("Average Graphics Package Power (W)", 0)),
                        })
        except Exception:
            pass
    
    if not gpus:
        try:
            import os
            import glob
            drm_path = "/sys/class/drm"
            if os.path.exists(drm_path):
                card_pattern = os.path.join(drm_path, "card*")
                for card_path in glob.glob(card_pattern):
                    device_path = os.path.join(card_path, "device")
                    if os.path.exists(device_path):
                        vendor_path = os.path.join(device_path, "vendor")
                        if os.path.exists(vendor_path):
                            with open(vendor_path, 'r') as f:
                                vendor = f.read().strip().lower()
                                if '0x1002' in vendor or 'ati' in vendor or 'amd' in vendor:
                                    name = "AMD Integrated GPU"
                                    uevent_path = os.path.join(device_path, "uevent")
                                    if os.path.exists(uevent_path):
                                        with open(uevent_path, 'r') as f:
                                            uevent = f.read()
                                            for line in uevent.split('\n'):
                                                if 'PCI_ID=' in line:
                                                    pci_id = line.split('=')[1]
                                                    name = f"AMD GPU ({pci_id})"
                                    
                                    util = 0.0
                                    mem_used = 0.0
                                    mem_total = 0.0
                                    temp = 0.0
                                    power = 0.0
                                    
                                    try:
                                        util_path = os.path.join(card_path, "device", "gpu_busy_percent")
                                        if os.path.exists(util_path):
                                            with open(util_path, 'r') as f:
                                                util = float(f.read().strip())
                                    except Exception:
                                        pass
                                    
                                    try:
                                        mem_info_path = os.path.join(card_path, "device", "mem_info_vram_total")
                                        if os.path.exists(mem_info_path):
                                            with open(mem_info_path, 'r') as f:
                                                mem_total = float(f.read().strip()) / (1024**3)
                                    except Exception:
                                        pass
                                    
                                    try:
                                        temp_path = os.path.join(card_path, "hwmon", "temp1_input")
                                        if os.path.exists(temp_path):
                                            with open(temp_path, 'r') as f:
                                                temp = float(f.read().strip()) / 1000.0
                                    except Exception:
                                        pass
                                    
                                    gpus.append({
                                        "vendor": "amd",
                                        "name": name,
                                        "utilization": util,
                                        "memory_used_gb": mem_used,
                                        "memory_total_gb": mem_total,
                                        "temperature": temp,
                                        "power_draw": power,
                                    })
        except Exception:
            pass
    
    return gpus


def get_intel_gpus() -> List[Dict[str, Any]]:
    """Get Intel GPU information using xpu-smi or sysfs."""
    gpus = []
    
    try:
        import subprocess
        import re
        result = subprocess.run(
            ["xpu-smi", "discovery", "-l"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if 'Device' in line or 'GPU' in line:
                    device_id = i
                    stats_result = subprocess.run(
                        ["xpu-smi", "stats", "-d", str(device_id), "-m", "0,1,2,3"],
                        capture_output=True, text=True, timeout=5
                    )
                    if stats_result.returncode == 0:
                        util = 0.0
                        mem_used = 0.0
                        mem_total = 0.0
                        temp = 0.0
                        power = 0.0
                        
                        for stat_line in stats_result.stdout.split('\n'):
                            if 'GPU Utilization' in stat_line:
                                match = re.search(r'(\d+\.?\d*)', stat_line)
                                if match:
                                    util = float(match.group(1))
                            elif 'Memory Used' in stat_line:
                                match = re.search(r'(\d+\.?\d*)', stat_line)
                                if match:
                                    mem_used = float(match.group(1)) / 1024
                            elif 'Memory Total' in stat_line:
                                match = re.search(r'(\d+\.?\d*)', stat_line)
                                if match:
                                    mem_total = float(match.group(1)) / 1024
                            elif 'Temperature' in stat_line:
                                match = re.search(r'(\d+\.?\d*)', stat_line)
                                if match:
                                    temp = float(match.group(1))
                            elif 'Power' in stat_line:
                                match = re.search(r'(\d+\.?\d*)', stat_line)
                                if match:
                                    power = float(match.group(1))
                        
                        gpus.append({
                            "vendor": "intel",
                            "name": f"Intel GPU {device_id}",
                            "utilization": util,
                            "memory_used_gb": mem_used,
                            "memory_total_gb": mem_total,
                            "temperature": temp,
                            "power_draw": power,
                        })
    except Exception:
        pass
    
    if not gpus:
        try:
            import os
            import glob
            drm_path = "/sys/class/drm"
            if os.path.exists(drm_path):
                card_pattern = os.path.join(drm_path, "card*")
                for card_path in glob.glob(card_pattern):
                    device_path = os.path.join(card_path, "device")
                    if os.path.exists(device_path):
                        vendor_path = os.path.join(device_path, "vendor")
                        if os.path.exists(vendor_path):
                            with open(vendor_path, 'r') as f:
                                vendor = f.read().strip().lower()
                                if '0x8086' in vendor or 'intel' in vendor:
                                    name = "Intel Integrated GPU"
                                    uevent_path = os.path.join(device_path, "uevent")
                                    if os.path.exists(uevent_path):
                                        with open(uevent_path, 'r') as f:
                                            uevent = f.read()
                                            for line in uevent.split('\n'):
                                                if 'PCI_ID=' in line:
                                                    pci_id = line.split('=')[1]
                                                    name = f"Intel GPU ({pci_id})"
                                    
                                    util = 0.0
                                    mem_used = 0.0
                                    mem_total = 0.0
                                    temp = 0.0
                                    power = 0.0
                                    
                                    try:
                                        util_path = os.path.join(card_path, "device", "gt_cur_freq_mhz")
                                        if os.path.exists(util_path):
                                            with open(util_path, 'r') as f:
                                                util = float(f.read().strip()) / 100.0
                                    except Exception:
                                        pass
                                    
                                    try:
                                        mem_info_path = os.path.join(card_path, "device", "memory_info")
                                        if os.path.exists(mem_info_path):
                                            with open(mem_info_path, 'r') as f:
                                                for line in f:
                                                    if 'size:' in line.lower():
                                                        mem_total = float(line.split(':')[1].strip().split()[0]) / (1024**3)
                                    except Exception:
                                        pass
                                    
                                    try:
                                        temp_path = os.path.join(card_path, "hwmon", "temp1_input")
                                        if os.path.exists(temp_path):
                                            with open(temp_path, 'r') as f:
                                                temp = float(f.read().strip()) / 1000.0
                                    except Exception:
                                        pass
                                    
                                    gpus.append({
                                        "vendor": "intel",
                                        "name": name,
                                        "utilization": util,
                                        "memory_used_gb": mem_used,
                                        "memory_total_gb": mem_total,
                                        "temperature": temp,
                                        "power_draw": power,
                                    })
        except Exception:
            pass
    
    return gpus


async def collect_system_stats(
    start_time: datetime = None,
    request_count: int = 0
) -> XiSystemStats:
    """
    Collect system statistics including CPU, memory, and GPU.
    
    Args:
        start_time: Server start time for uptime calculation
        request_count: Total request count for QPS calculation
        
    Returns:
        XiSystemStats with current system statistics
    """
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    all_gpus = []
    all_gpus.extend(get_nvidia_gpus())
    all_gpus.extend(get_amd_gpus())
    all_gpus.extend(get_intel_gpus())
    
    if sys.platform == "win32" and not all_gpus:
        all_gpus.extend(get_windows_gpus())
    
    gpu_utilization = [g["utilization"] for g in all_gpus]
    gpu_memory_used = [g["memory_used_gb"] for g in all_gpus]
    gpu_memory_total = [g["memory_total_gb"] for g in all_gpus]
    gpu_vendors = [g["vendor"] for g in all_gpus]
    gpu_names = [g["name"] for g in all_gpus]
    gpu_temperatures = [g["temperature"] for g in all_gpus]
    gpu_power_draw = [g["power_draw"] for g in all_gpus]
    
    uptime = 0.0
    if start_time:
        uptime = (datetime.now() - start_time).total_seconds()
    
    return XiSystemStats(
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_used_gb=memory.used / (1024**3),
        memory_total_gb=memory.total / (1024**3),
        gpu_count=len(all_gpus),
        gpu_utilization=gpu_utilization,
        gpu_memory_used=gpu_memory_used,
        gpu_memory_total=gpu_memory_total,
        gpu_vendors=gpu_vendors,
        gpu_names=gpu_names,
        gpu_temperatures=gpu_temperatures,
        gpu_power_draw=gpu_power_draw,
        uptime_seconds=uptime,
        request_count=request_count,
        qps=request_count / max(uptime, 1.0)
    )


def get_windows_gpus() -> List[Dict[str, Any]]:
    """Get GPU information on Windows using WMI or subprocess."""
    gpus = []
    
    try:
        import wmi
        c = wmi.WMI()
        
        for gpu in c.Win32_VideoController():
            name = gpu.Name or "Unknown GPU"
            driver_version = gpu.DriverVersion or ""
            
            vendor = "unknown"
            name_lower = name.lower()
            
            if "nvidia" in name_lower or "geforce" in name_lower or "quadro" in name_lower or "tesla" in name_lower:
                vendor = "nvidia"
            elif "amd" in name_lower or "radeon" in name_lower or "ati" in name_lower:
                vendor = "amd"
            elif "intel" in name_lower or "iris" in name_lower or "uhd" in name_lower or "hd graphics" in name_lower:
                vendor = "intel"
            
            adapter_ram = gpu.AdapterRAM or 0
            mem_total_gb = adapter_ram / (1024**3) if adapter_ram else 0.0
            
            gpus.append({
                "vendor": vendor,
                "name": name,
                "utilization": 0.0,
                "memory_used_gb": 0.0,
                "memory_total_gb": mem_total_gb,
                "temperature": 0.0,
                "power_draw": 0.0,
            })
    except ImportError as e:
        pass
    except Exception as e:
        pass
    
    if not gpus:
        try:
            import subprocess
            import re
            
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM", "/format:list"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                
                for line in lines:
                    line = line.strip()
                    if not line or "AdapterRAM" in line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            adapter_ram = int(parts[0])
                            name = ' '.join(parts[1:])
                            name_lower = name.lower()
                            
                            vendor = "unknown"
                            if "nvidia" in name_lower or "geforce" in name_lower or "quadro" in name_lower or "tesla" in name_lower:
                                vendor = "nvidia"
                            elif "amd" in name_lower or "radeon" in name_lower or "ati" in name_lower:
                                vendor = "amd"
                            elif "intel" in name_lower or "iris" in name_lower or "uhd" in name_lower or "hd graphics" in name_lower:
                                vendor = "intel"
                            
                            mem_total_gb = adapter_ram / (1024**3)
                            
                            gpus.append({
                                "vendor": vendor,
                                "name": name,
                                "utilization": 0.0,
                                "memory_used_gb": 0.0,
                                "memory_total_gb": mem_total_gb,
                                "temperature": 0.0,
                                "power_draw": 0.0,
                            })
                        except (ValueError, TypeError, IndexError):
                            pass
        except Exception as e:
            pass
    
    return gpus
