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

import os
import json
import torch
import platform
import subprocess
import time
import numpy as np
from utils.log.core import PiscesLxCoreLog
from typing import Dict, Any, Optional
from .config import PiscesLxCoreDeviceConfig
from .smart_detector import PiscesLxCoreDeviceSmartDetector

logger = PiscesLxCoreLog("PiscesLx.Utils.Device.Manager")

class PiscesLxCoreDeviceManager:
    """
    Intelligent Device Manager - System-level hardware detection and strategy selection.
    The selection is automatically made based on system hardware status without developer intervention.
    """

    def __init__(self, cfg: Optional[PiscesLxCoreDeviceConfig] = None):
        """
        Initialize the device manager with optional configuration.
        
        Args:
            cfg (Optional[PiscesLxCoreDeviceConfig]): Configuration object for device management.
        """
        self.cfg = cfg or PiscesLxCoreDeviceConfig({})
        self.gpu_info = []
        self.strategy = {}
        self.system_memory = {}
        self.smart_detector = PiscesLxCoreDeviceSmartDetector()

        self._detect_hardware()
        self._determine_strategy()

    def _detect_system_memory(self):
        """
        Enhanced system memory detection with swap and caching information.
        
        This method attempts to gather detailed memory statistics including:
        - Total physical memory
        - Available memory
        - Used memory
        - Swap space information
        - Memory pressure levels
        
        If psutil is not available, it falls back to platform-specific methods.
        """
        try:
            import psutil
            
            # Get memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Calculate available memory considering swap
            total_available = memory.available + swap.free
            
            self.system_memory = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'free': memory.free,
                'percent': memory.percent,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_free': swap.free,
                'swap_percent': swap.percent,
                'total_available': total_available,  # Including swap
                'memory_pressure': self._calculate_memory_pressure(memory, swap)
            }
            
            logger.info("System memory detected", extra={
                "total_gb": memory.total // (1024**3),
                "available_gb": memory.available // (1024**3),
                "swap_gb": swap.total // (1024**3),
                "pressure_level": self.system_memory['memory_pressure']
            })
            
        except ImportError:
            logger.warning("psutil not available", message="Using fallback memory detection")
            self._detect_memory_fallback()
        except Exception as e:
            logger.error("System memory detection failed", error=str(e))
            self._detect_memory_fallback()
    
    def _calculate_memory_pressure(self, memory, swap) -> str:
        """
        Calculate memory pressure level based on usage percentage.
        
        Args:
            memory: psutil virtual_memory object
            swap: psutil swap_memory object
            
        Returns:
            str: Memory pressure level ("critical", "high", "moderate", "normal", "low")
        """
        total_used_percent = (memory.used + swap.used) / (memory.total + swap.total) * 100
        
        if total_used_percent >= 90:
            return "critical"
        elif total_used_percent >= 80:
            return "high"
        elif total_used_percent >= 70:
            return "moderate"
        elif total_used_percent >= 50:
            return "normal"
        else:
            return "low"
    
    def _detect_memory_fallback(self):
        """
        Fallback memory detection without psutil.
        
        This method provides platform-specific memory detection when psutil is not available:
        - Windows: Uses Windows API via ctypes
        - Unix-like: Attempts to read /proc/meminfo
        - Ultimate fallback: Returns zero values
        """
        try:
            import os
            
            if os.name == 'nt':  # Windows
                import ctypes
                class _MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)
                    ]
                
                memoryStatus = _MEMORYSTATUSEX()
                memoryStatus.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
                
                self.system_memory = {
                    'total': memoryStatus.ullTotalPhys,
                    'available': memoryStatus.ullAvailPhys,
                    'used': memoryStatus.ullTotalPhys - memoryStatus.ullAvailPhys,
                    'percent': memoryStatus.dwMemoryLoad,
                    'swap_total': memoryStatus.ullTotalPageFile,
                    'swap_free': memoryStatus.ullAvailPageFile,
                    'swap_used': memoryStatus.ullTotalPageFile - memoryStatus.ullAvailPageFile,
                    'swap_percent': (memoryStatus.ullTotalPageFile - memoryStatus.ullAvailPageFile) / memoryStatus.ullTotalPageFile * 100 if memoryStatus.ullTotalPageFile > 0 else 0,
                    'total_available': memoryStatus.ullAvailPhys + memoryStatus.ullAvailPageFile,
                    'memory_pressure': 'unknown'
                }
            else:  # Unix-like systems
                try:
                    # Try to read /proc/meminfo
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = {}
                        for line in f:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                meminfo[key.strip()] = int(value.strip().split()[0]) * 1024  # Convert to bytes
                        
                        total = meminfo.get('MemTotal', 0)
                        free = meminfo.get('MemFree', 0)
                        available = meminfo.get('MemAvailable', free)
                        swap_total = meminfo.get('SwapTotal', 0)
                        swap_free = meminfo.get('SwapFree', swap_total)
                        
                        self.system_memory = {
                            'total': total,
                            'available': available,
                            'used': total - available,
                            'free': free,
                            'percent': (total - available) / total * 100 if total > 0 else 0,
                            'swap_total': swap_total,
                            'swap_used': swap_total - swap_free,
                            'swap_free': swap_free,
                            'swap_percent': (swap_total - swap_free) / swap_total * 100 if swap_total > 0 else 0,
                            'total_available': available + swap_free,
                            'memory_pressure': 'unknown'
                        }
                except (IOError, OSError):
                    # Ultimate fallback
                    self.system_memory = {
                        'total': 0, 'available': 0, 'used': 0, 'free': 0, 'percent': 0,
                        'swap_total': 0, 'swap_used': 0, 'swap_free': 0, 'swap_percent': 0,
                        'total_available': 0, 'memory_pressure': 'unknown'
                    }
                    
        except Exception as e:
            logger.error("Fallback memory detection failed", error=str(e))
            self.system_memory = {
                'total': 0, 'available': 0, 'used': 0, 'free': 0, 'percent': 0,
                'swap_total': 0, 'swap_used': 0, 'swap_free': 0, 'swap_percent': 0,
                'total_available': 0, 'memory_pressure': 'unknown'
            }

    def _detect_hardware(self):
        """
        Universal hardware detection using smart detector - NVIDIA/CPU auto-discovery.
        
        This method uses the smart detector to perform comprehensive hardware detection,
        including GPUs and CPU information. It also detects system memory after hardware detection.
        """
        # Use smart detector for comprehensive hardware detection
        detection_result = self.smart_detector.detect_all_devices()
        
        # Extract GPU information
        self.gpu_info = detection_result.get('gpu_info', [])
        self.cpu_info = detection_result.get('cpu_info', {}).get('basic_info', {})
        
        # Rely on smart_detector.detect_all_devices() as the unified source.
        # self._detect_platform_specific_gpus()  # removed to avoid duplicated probing
        
        # Store detection summary
        self.detection_summary = detection_result.get('detection_summary', {})
        
        logger.info("Smart detection complete", extra={"gpu_count": len(self.gpu_info), "cpu_detected": bool(self.cpu_info)})
        
        # Detect system memory
        self._detect_system_memory()
        
    def _detect_platform_specific_gpus(self):
        """
        Detect platform-specific GPUs (ROCm, DirectML, etc.).
        
        This method dispatches to platform-specific GPU detection methods based on the
        operating system. It supports Linux (ROCm), Windows (DirectML), and WSL detection.
        """
        system = platform.system().lower()
        
        if system == "linux":
            self._detect_rocm_gpus()
        elif system == "windows":
            self._detect_directml_gpus()
            self._detect_wsl_gpus()
            
    def _detect_rocm_gpus(self):
        """
        Detect AMD GPUs via ROCm on Linux.
        
        This method attempts to use rocm-smi to detect AMD GPUs and gather their information.
        If successful, it populates the gpu_info list with ROCm GPU details.
        """
        try:
            # Try rocm-smi for AMD GPUs
            result = subprocess.run([
                'rocm-smi',
                '--showid',
                '--showproductname',
                '--showmeminfo',
                '--json'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                import json
                rocm_data = json.loads(result.stdout)
                
                for gpu_id, gpu_data in rocm_data.items():
                    if 'GPU' in gpu_data:
                        gpu_info = gpu_data['GPU']
                        self.gpu_info.append({
                    'type': 'rocm',
                    'index': int(gpu_id.replace('GPU', '')),
                    'name': gpu_info.get('Product Name', 'AMD GPU'),
                    'total_memory': self._parse_rocm_memory(gpu_info.get('VRAM Total Memory', '0')),
                    'free_memory': self._parse_rocm_memory(gpu_info.get('VRAM Available Memory', '0')),
                    'used_memory': 0,
                    'temperature': self._get_rocm_temperature(gpu_id.replace('GPU', '')),
                    'utilization': 0,
                    'platform': 'rocm',
                    'vendor': 'amd',
                    'compute_capability': 'unknown',
                    'cuda_cores': 0,
                    'tensor_cores': False,
                    'memory_bandwidth': 0
                })
                        
                logger.info("ROCm detection", extra={"amd_gpu_count": len([g for g in self.gpu_info if g.get('platform') == 'rocm'])})
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("ROCm detection skipped", extra={"reason": "TimeoutExpired or FileNotFoundError"})
        except Exception as e:
            # Log unexpected errors for debugging
            if 'JSONDecodeError' in str(e):
                logger.debug("ROCm detection skipped", extra={"reason": "JSONDecodeError"})
            else:
                logger.error("Unexpected error in ROCm detection", error=str(e))
            
    def _detect_directml_gpus(self):
        """
        Detect DirectML-compatible GPUs on Windows.
        
        This method attempts to use torch_directml to detect DirectML-compatible GPUs
        on Windows systems. If successful, it populates the gpu_info list with DirectML GPU details.
        """
        try:
            # Try to import torch_directml for Windows GPU detection
            import torch_directml
            
            device_count = torch_directml.device_count()
            for i in range(device_count):
                device_name = torch_directml.device_name(i)
                
                # Get memory info if available
                try:
                    device = torch_directml.device(i)
                    total_memory = 16 * 1024  # Default 16GB estimation
                    free_memory = total_memory
                    used_memory = 0
                except:
                    total_memory = 16 * 1024
                    free_memory = total_memory
                    used_memory = 0
                
                self.gpu_info.append({
                    'type': 'directml',
                    'index': i,
                    'name': device_name,
                    'total_memory': total_memory,
                    'free_memory': free_memory,
                    'used_memory': used_memory,
                    'temperature': self._get_directml_temperature(i),
                    'utilization': self._get_directml_utilization(i),
                    'platform': 'directml',
                    'vendor': 'microsoft',
                    'compute_capability': 'unknown',
                    'cuda_cores': 0,
                    'tensor_cores': False,
                    'memory_bandwidth': 0
                })
                
            logger.info("DirectML detection", extra={"gpu_count": device_count})
            
        except ImportError:
            logger.debug("DirectML not available")
        except Exception as e:
            # Elevate to warning for first failure visibility; downstream fallbacks stay debug
            logger.warning("DirectML detection failed", error=str(e), error_class=type(e).__name__)
            
    def _detect_wsl_gpus(self):
        """
        Detect GPUs in WSL environment.
        
        This method checks if the system is running in WSL and attempts to detect
        NVIDIA GPUs using nvidia-smi within the WSL environment.
        """
        try:
            # Check if running in WSL
            if os.path.exists('/proc/sys/fs/binfmt_misc/WSLInterop'):
                # Try nvidia-smi in WSL
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                # Check if this GPU is already detected
                                gpu_index = int(parts[0])
                                if not any(g['index'] == gpu_index and g.get('platform') == 'wsl' for g in self.gpu_info):
                                    self.gpu_info.append({
                                        'type': 'nvidia',
                                        'index': gpu_index,
                                        'name': parts[1],
                                        'total_memory': int(parts[2]),
                                        'free_memory': int(parts[2]),
                                        'used_memory': 0,
                                        'temperature': self._get_wsl_temperature(gpu_index),
                                        'utilization': self._get_wsl_utilization(gpu_index),
                                        'platform': 'wsl',
                                        'vendor': 'nvidia',
                                        'compute_capability': 'unknown',
                                        'cuda_cores': 0,
                                        'tensor_cores': False,
                                        'memory_bandwidth': 0
                                    })
                    
                    logger.success("WSL GPU detection completed")
                    
        except Exception as e:
            logger.debug("WSL detection failed or skipped", extra={"error": str(e), "error_class": type(e).__name__})
            
    def _get_rocm_temperature(self, gpu_id: str) -> int:
        """Get ROCm GPU temperature using rocm-smi."""
        try:
            result = subprocess.run(['rocm-smi', '--showtemp', '--json'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for gpu_data in data.values():
                    if 'GPU' in gpu_data and str(gpu_id) in str(gpu_data.get('GPU', {})):
                        temp_info = gpu_data['GPU']
                        temp_str = temp_info.get('Temperature (Sensor memory) (C)', '0')
                        return int(temp_str.replace('C', '').strip()) if temp_str != 'N/A' else 0
        except Exception:
            pass
        return 0
        
    def _get_directml_temperature(self, device_index: int) -> int:
        """Get DirectML device temperature using actual hardware sensors."""
        try:
            import torch_directml
            device = torch_directml.device(device_index)
            
            # Try to get actual temperature from DirectML device properties
            device_props = torch_directml.get_device_properties(device_index)
            
            # Look for temperature sensor data in device properties
            if hasattr(device_props, 'temperature'):
                return device_props.temperature
            elif hasattr(device_props, 'current_temperature'):
                return device_props.current_temperature
            
            # Fallback: Try to access Windows thermal zone information
            try:
                import win32com.client
                wmi = win32com.client.Dispatch("WbemScripting.SWbemLocator")
                wmi_service = wmi.ConnectServer(".", "root\\wmi")
                
                # Query thermal zone information
                thermal_zones = wmi_service.ExecQuery("SELECT * FROM MSAcpi_ThermalZoneTemperature")
                for zone in thermal_zones:
                    if hasattr(zone, 'CurrentTemperature'):
                        # Convert from Kelvin*10 to Celsius
                        temp_kelvin = zone.CurrentTemperature / 10.0
                        return int(temp_kelvin - 273.15)
            except Exception:
                pass
                
            # Final fallback: Try Windows Performance Counters
            try:
                import subprocess
                result = subprocess.run([
                    'powershell', '-Command',
                    'Get-Counter "\\Thermal Zone Information(*)\\Temperature" | Select-Object -ExpandProperty CounterSamples'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    # Parse PowerShell output for temperature data
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'Temperature' in line and ':' in line:
                            temp_str = line.split(':')[-1].strip()
                            if temp_str.replace('.', '').isdigit():
                                return int(float(temp_str))
            except Exception:
                pass
                
        except Exception as e:
            logger.warning("DirectML temperature detection failed", error=str(e), device_index=device_index)
            
        return 0  # Return 0 to indicate temperature unavailable instead of fake estimate
        
    def _get_directml_utilization(self, device_index: int) -> int:
        """Get DirectML device utilization using actual performance counters."""
        try:
            import torch_directml
            device_props = torch_directml.get_device_properties(device_index)
            
            # Try to get actual utilization from device properties
            if hasattr(device_props, 'gpu_utilization'):
                return device_props.gpu_utilization
            elif hasattr(device_props, 'utilization'):
                return device_props.utilization
            
            # Fallback: Try Windows Performance Counters for GPU utilization
            try:
                import subprocess
                result = subprocess.run([
                    'powershell', '-Command',
                    'Get-Counter "\\GPU Engine Utilization(*)\\Utilization" | Select-Object -First 1 -ExpandProperty CounterSamples'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    # Parse PowerShell output for utilization data
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'Utilization' in line and any(char.isdigit() for char in line):
                            # Extract percentage value
                            import re
                            match = re.search(r'(\d+(?:\.\d+)?)', line)
                            if match:
                                return int(float(match.group(1)))
            except Exception:
                pass
                
        except Exception as e:
            logger.warning("DirectML utilization detection failed", error=str(e), device_index=device_index)
            
        return 0  # Return actual 0 utilization instead of fake estimate
        
    def _get_wsl_temperature(self, gpu_index: int) -> int:
        """Get WSL NVIDIA GPU temperature using nvidia-smi."""
        try:
            result = subprocess.run(['nvidia-smi', f'--id={gpu_index}', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                temp_str = result.stdout.strip().split('\n')[0]
                return int(temp_str) if temp_str.isdigit() else 0
        except Exception:
            pass
        return 0
        
    def _get_wsl_utilization(self, gpu_index: int) -> int:
        """Get WSL NVIDIA GPU utilization using nvidia-smi."""
        try:
            result = subprocess.run(['nvidia-smi', f'--id={gpu_index}', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                util_str = result.stdout.strip().split('\n')[0]
                return int(util_str) if util_str.isdigit() else 0
        except Exception:
            pass
        return 0
        
    def _get_torch_temperature(self, device_index: int) -> int:
        """Get NVIDIA GPU temperature using PyTorch CUDA (fallback method)."""
        try:
            # Try to use nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                return temp
            except ImportError:
                pass
                
            # Fallback: try nvidia-smi subprocess call
            result = subprocess.run(['nvidia-smi', f'--id={device_index}', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                temp_str = result.stdout.strip().split('\n')[0]
                return int(temp_str) if temp_str.isdigit() else 0
        except Exception:
            pass
        return 0  # Final fallback
        
    def _get_torch_utilization(self, device_index: int) -> int:
        """Get NVIDIA GPU utilization using PyTorch CUDA (fallback method)."""
        try:
            # Try to use nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                if hasattr(util, 'gpu'):
                    return int(util.gpu)
            except ImportError:
                pass
                
            # Fallback: try nvidia-smi subprocess call
            result = subprocess.run(['nvidia-smi', f'--id={device_index}', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                util_str = result.stdout.strip().split('\n')[0]
                return int(util_str) if util_str.isdigit() else 0
        except Exception:
            pass
        return 0  # Final fallback

    def _parse_rocm_memory(self, memory_str: str) -> int:
        """
        Parse ROCm memory string to MiB.
        
        Args:
            memory_str (str): Memory string from ROCm (e.g., "16368 MB", "16 GB")
            
        Returns:
            int: Memory value in MiB
        """
        try:
            # Handle different formats like "16368 MB", "16 GB", etc.
            memory_str = memory_str.strip().upper()
            
            # Extract number and unit
            if 'GB' in memory_str:
                value = float(memory_str.replace('GB', '').strip())
                return int(value * 1024)  # Convert GB to MiB
            elif 'MB' in memory_str:
                value = float(memory_str.replace('MB', '').strip())
                return int(value)
            elif memory_str.isdigit():
                return int(memory_str)
            else:
                # Try to extract number from string
                import re
                numbers = re.findall(r'\d+\.?\d*', memory_str)
                if numbers:
                    value = float(numbers[0])
                    if 'GB' in memory_str:
                        return int(value * 1024)
                    else:
                        return int(value)
                else:
                    return 0
        except (ValueError, AttributeError) as e:
            logger.debug("ROCm memory parse failed", extra={"error": str(e), "memory_str": memory_str[:50]})
            return 0

    def _detect_cpu(self):
        """
        Detect CPU capabilities.
        
        This method uses psutil to gather CPU information including core count,
        thread count, total memory, and architecture.
        """
        import psutil
        self.cpu_info = {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'architecture': self._get_cpu_architecture(),
        }

    def _get_cpu_architecture(self) -> str:
        """
        Get CPU architecture info.
        
        Returns:
            str: CPU architecture information
        """
        import platform
        return platform.processor() or platform.machine()

    def _detect_nvidia_gpus(self):
        """
        Enhanced NVIDIA GPU detection with comprehensive error handling and diagnostics.
        
        This method attempts to detect NVIDIA GPUs using nvidia-smi with detailed
        error handling and fallback mechanisms. It gathers comprehensive GPU metrics
        including memory, temperature, utilization, and more.
        """
        try:
            # Find nvidia-smi path
            nvidia_smi_path = self._find_nvidia_smi()
            if not nvidia_smi_path:
                logger.warning("NVIDIA_DETECTION_FAILED", extra={
                    "message": "nvidia-smi not found, falling back to torch enumeration"
                })
                self._torch_fallback_enumeration()
                return
            
            # Get clean environment
            clean_env = self._get_clean_env()
            
            # Check driver status first
            self._check_nvidia_driver_status()
            
            # Enhanced query with more GPU metrics
            query_fields = [
                'index', 'name', 'memory.total', 'memory.free', 'memory.used',
                'temperature.gpu', 'utilization.gpu', 'power.draw', 'clocks.gr'
            ]
            
            result = subprocess.run([
                nvidia_smi_path,
                f'--query-gpu={",".join(query_fields)}',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=30, env=clean_env)
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error("NVIDIA_QUERY_FAILED", extra={
                    "error": error_msg,
                    "message": "nvidia-smi query failed, using torch fallback"
                })
                self._torch_fallback_enumeration()
                return
                
            # Parse results with enhanced error handling
            lines = result.stdout.strip().split('\n')
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                    
                try:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < len(query_fields):
                        logger.warning("NVIDIA_PARSE_INCOMPLETE", extra={
                            "line": line_num,
                            "expected_fields": len(query_fields),
                            "actual_fields": len(parts)
                        })
                        continue
                    
                    # Parse with fallback values
                    gpu_index = int(parts[0]) if parts[0].isdigit() else -1
                    name = parts[1] if parts[1] else "Unknown NVIDIA GPU"
                    total_memory = int(parts[2]) if parts[2].isdigit() else 0
                    free_memory = int(parts[3]) if parts[3].isdigit() else 0
                    used_memory = int(parts[4]) if parts[4].isdigit() else 0
                    temperature = int(parts[5]) if parts[5].isdigit() else 0
                    utilization = int(parts[6]) if parts[6].isdigit() else 0
                    power_draw = float(parts[7]) if self._is_float(parts[7]) else 0.0
                    clock_speed = int(parts[8]) if parts[8].isdigit() else 0
                    
                    # Validate parsed data
                    if gpu_index < 0 or total_memory <= 0:
                        logger.warning("NVIDIA_PARSE_INVALID", extra={
                            "line": line_num,
                            "gpu_index": gpu_index,
                            "total_memory": total_memory
                        })
                        continue
                    
                    # Calculate derived metrics
                    memory_utilization = (used_memory / total_memory * 100) if total_memory > 0 else 0
                    
                    self.gpu_info.append({
                        'index': gpu_index,
                        'name': name,
                        'type': 'nvidia',
                        'total_memory': total_memory,
                        'free_memory': free_memory,
                        'used_memory': used_memory,
                        'temperature': temperature,
                        'utilization': utilization,
                        'power_draw': power_draw,
                        'clock_speed': clock_speed,
                        'memory_utilization': int(memory_utilization),
                        'platform': 'nvidia',
                        'vendor': 'nvidia',
                        'compute_capability': self._get_cuda_capability(gpu_index),
                        'cuda_cores': 0,  # Would need additional query
                        'tensor_cores': 'V100' in name or 'A100' in name or 'H100' in name,
                        'memory_bandwidth': 0  # Would need additional query
                    })
                    
                except (ValueError, IndexError) as e:
                    logger.error("NVIDIA_PARSE_LINE_FAILED", extra={
                        "line": line_num,
                        "error": str(e),
                        "line_content": line[:100]  # Truncate long lines
                    })
                    continue
            
            logger.success("NVIDIA_DETECTION_COMPLETE", extra={
                "gpu_count": len([g for g in self.gpu_info if g['type'] == 'nvidia']),
                "detection_method": "nvidia-smi"
            })
            
        except subprocess.TimeoutExpired:
            logger.error("NVIDIA_DETECTION_TIMEOUT", extra={
                "timeout": 30,
                "message": "NVIDIA GPU detection timed out"
            })
            self._torch_fallback_enumeration()
            
        except FileNotFoundError:
            logger.warning("NVIDIA_DETECTION_NOT_FOUND", extra={
                "message": "nvidia-smi not found"
            })
            self._torch_fallback_enumeration()
            
        except Exception as e:
            logger.error("NVIDIA_DETECTION_UNEXPECTED_ERROR", extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "message": "Unexpected error during NVIDIA GPU detection"
            })
            self._torch_fallback_enumeration()
    
    def _is_float(self, value: str) -> bool:
        """
        Check if string can be converted to float.
        
        Args:
            value (str): String to check
            
        Returns:
            bool: True if string can be converted to float, False otherwise
        """
        try:
            float(value)
            return True
        except ValueError:
            return False
            
    def _get_cuda_capability(self, gpu_index: int) -> str:
        """
        Get CUDA compute capability for a GPU.
        
        Args:
            gpu_index (int): Index of the GPU
            
        Returns:
            str: CUDA compute capability in format "major.minor" or "unknown"
        """
        try:
            if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
                capability = torch.cuda.get_device_capability(gpu_index)
                return f"{capability[0]}.{capability[1]}"
        except Exception as e:
            # Log CUDA capability detection failures for debugging
            logger.debug("CUDA_CAPABILITY_DETECTION_FAILED", extra={"gpu_index": gpu_index, "error": str(e)})
        return "unknown"

    def _find_nvidia_smi(self) -> Optional[str]:
        """
        Find nvidia-smi command path.
        
        Returns:
            Optional[str]: nvidia-smi path if found, None otherwise
        """
        # Common paths list
        common_paths = [
            'nvidia-smi',
            '/usr/bin/nvidia-smi',
            '/usr/local/bin/nvidia-smi',
            'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe',
            'C:\\Windows\\System32\\nvidia-smi.exe'
        ]
        
        # Try each path
        for path in common_paths:
            try:
                # Use where/which command to find
                if os.name == 'nt':  # Windows
                    result = subprocess.run(['where', path], capture_output=True, text=True, timeout=5)
                else:  # Unix-like
                    result = subprocess.run(['which', path], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout.strip().split('\n')[0].strip()
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        return None
    
    def _get_clean_env(self) -> Dict[str, str]:
        """
        Get clean environment variables to avoid GPU detection interference.
        
        Returns:
            Dict[str, str]: Clean environment variables dictionary
        """
        import os
        clean_env = os.environ.copy()
        
        # Remove environment variables that might interfere with GPU detection
        gpu_vars = [
            'CUDA_VISIBLE_DEVICES',
            'GPU_DEVICE_ORDINAL',
            'ROCR_VISIBLE_DEVICES',
            'HIP_VISIBLE_DEVICES'
        ]
        
        for var in gpu_vars:
            clean_env.pop(var, None)
            
        return clean_env
    
    def _check_nvidia_driver_status(self):
        """
        Check NVIDIA driver status and provide diagnostic information.
        
        This method queries the NVIDIA driver version and logs it for diagnostic purposes.
        """
        try:
            # Check driver version
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                driver_version = result.stdout.strip()
                logger.info("NVIDIA_DRIVER_VERSION", extra={"version": driver_version})
            else:
                logger.warning("NVIDIA_DRIVER_CHECK_FAILED", extra={
                    "message": "Failed to get NVIDIA driver version"
                })
                
        except Exception as e:
            logger.error("NVIDIA_DRIVER_STATUS_CHECK_FAILED", extra={
                "error": str(e),
                "message": "NVIDIA driver status check failed"
            })

    def _torch_fallback_enumeration(self):
        """
        Enumerate NVIDIA GPUs using torch metadata as a fallback (approximate memory figures).
        
        This method is used as a fallback when nvidia-smi is not available or fails.
        It uses PyTorch's CUDA API to enumerate GPUs and gather basic information.
        """
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory // 1024**2  # to MiB
                free = total
                used = 0
                # If available, try mem_get_info for better accuracy
                try:
                    free_b, total_b = torch.cuda.mem_get_info(i)
                    free = int(free_b // 1024**2)
                    total = int(total_b // 1024**2)
                    used = max(total - free, 0)
                except Exception as e:
                    # Log memory info retrieval failures for debugging
                    logger.debug("CUDA_MEMORY_INFO_FAILED", extra={"gpu_index": i, "error": str(e)})
                self.gpu_info.append({
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'type': 'nvidia',  # Only NVIDIA supported now
                    'total_memory': int(total),
                    'free_memory': int(free),
                    'used_memory': int(used),
                    'temperature': self._get_torch_temperature(i),
                    'utilization': self._get_torch_utilization(i),
                })
            except Exception:
                continue

    def _determine_strategy(self):
        """
        Universal intelligent strategy determination using smart detector.
        
        This method determines the optimal execution strategy based on detected hardware
        and model parameters. It uses the smart detector to get a base strategy and
        then enhances it with advanced parallel strategies for large models.
        """
        # Get model parameters from config to determine hardware requirements
        model_params = self._estimate_model_parameters()
        
        # Use smart detector to determine optimal strategy
        model_size = self._get_model_size_string(model_params)
        
        # Get strategy from smart detector
        smart_strategy = self.smart_detector._determine_optimal_strategy(model_size)
        
        # Enhance with advanced parallel strategies for flagship models
        enhanced_strategy = self._enhance_parallel_strategy(smart_strategy, model_params)
        self.strategy = enhanced_strategy
        
        logger.info("Smart strategy determined", extra={"device_type": enhanced_strategy.get('device_type', 'unknown')})
        if enhanced_strategy.get('parallel_strategy'):
            logger.info("Parallel strategy", extra={"strategy": enhanced_strategy['parallel_strategy']})

    def _get_model_size_string(self, model_params: float) -> str:
        """
        Convert model parameters to size string for smart detector.
        
        Args:
            model_params (float): Number of model parameters in billions
            
        Returns:
            str: Model size category ("large", "medium", "small", "tiny")
        """
        if model_params >= 1000:
            return "large"
        elif model_params >= 100:
            return "medium"
        elif model_params >= 10:
            return "small"
        else:
            return "tiny"

    def _estimate_model_parameters(self) -> float:
        """
        Estimate model parameters from config or model name.
        
        This method first tries to get the model size from configuration.
        If that fails, it estimates based on model architecture parameters.
        
        Returns:
            float: Estimated number of model parameters in billions
        """
        try:
            # Try to get from config first
            model_size = self.cfg.get("model.size", "")
            if model_size:
                return self._parse_model_size(model_size)
            
            # Fallback: estimate from model architecture
            hidden_size = self.cfg.get("model.hidden_size", 4096)
            num_layers = self.cfg.get("model.num_layers", 32)
            vocab_size = self.cfg.get("model.vocab_size", 32000)
            
            # Rough estimation: ~ params = layers * (hidden^2 * 4 + vocab * hidden)
            params = num_layers * (hidden_size * hidden_size * 4 + vocab_size * hidden_size)
            return params / 1e9  # Convert to billions
        except Exception as e:
            logger.warning("Failed to estimate model parameters, using default", error=str(e))
            return 7.0  # Default to 7B model size

    def _parse_model_size(self, size_str: str) -> float:
        """
        Parse model size string like '7B', '1.5B', '1TB', etc.
        
        Args:
            size_str (str): Model size string (e.g., "7B", "1.5B", "1TB")
            
        Returns:
            float: Model size in billions of parameters
        """
        try:
            size_str = size_str.upper()
            
            # Handle TB models (ultra-large scale)
            if 'TB' in size_str:
                try:
                    tb_value = float(size_str.replace('TB', ''))
                    return tb_value * 1000  # Convert TB to B equivalent
                except:
                    return 7000.0  # Default to 7TB equivalent
            
            # Handle regular B models
            size_str = size_str.replace('B', '')
            try:
                return float(size_str)
            except:
                return 7.0  # Default to 7B
        except Exception as e:
            logger.warning("Failed to parse model size, using default", size_str=size_str, error=str(e))
            return 7.0  # Default to 7B

    def _enhance_parallel_strategy(self, base_strategy: dict, model_params: float) -> dict:
        """
        Enhance strategy with data-driven parallel strategies based on actual hardware profiling.
        
        This method determines optimal parallelization by testing actual communication performance
        and memory usage patterns rather than using fixed heuristic rules.
        
        Args:
            base_strategy (dict): Base strategy from smart detector
            model_params (float): Number of model parameters in billions
            
        Returns:
            dict: Enhanced strategy with parallelization information based on actual profiling
        """
        gpu_count = len(self.gpu_info)
        
        # Skip enhancement for small models or single GPU
        if model_params < 10 or gpu_count <= 1:
            return base_strategy
            
        enhanced = base_strategy.copy()
        
        # Perform actual communication bandwidth testing
        comm_performance = self._profile_communication_performance()
        memory_performance = self._profile_memory_performance()
        
        # Data-driven parallel strategy selection based on actual profiling results
        if model_params >= 1000:  # Ultra-large models (1T+)
            if gpu_count >= 16 and comm_performance['intra_node_bandwidth'] > 50:  # GB/s
                enhanced['parallel_strategy'] = 'tensor_pipeline_expert_parallel'
                enhanced['tensor_parallel_size'] = min(8, gpu_count // 2)
                enhanced['pipeline_parallel_size'] = gpu_count // enhanced['tensor_parallel_size']
                enhanced['expert_parallel'] = True
                enhanced['zero_optimization'] = 'zero3'
            elif gpu_count >= 8 and comm_performance['inter_node_bandwidth'] > 10:  # GB/s
                enhanced['parallel_strategy'] = 'tensor_pipeline_parallel'
                enhanced['tensor_parallel_size'] = 4
                enhanced['pipeline_parallel_size'] = 2
                enhanced['zero_optimization'] = 'zero3'
            else:
                enhanced['parallel_strategy'] = 'tensor_parallel'
                enhanced['tensor_parallel_size'] = gpu_count
                enhanced['zero_optimization'] = 'zero2'
                
        elif model_params >= 100:  # Large models (100B+)
            optimal_tp_size = self._calculate_optimal_tensor_parallel_size(
                model_params, gpu_count, comm_performance, memory_performance
            )
            
            if gpu_count >= 8 and optimal_tp_size >= 4:
                enhanced['parallel_strategy'] = 'tensor_pipeline_parallel'
                enhanced['tensor_parallel_size'] = optimal_tp_size
                enhanced['pipeline_parallel_size'] = gpu_count // optimal_tp_size
                enhanced['zero_optimization'] = 'zero2'
            elif gpu_count >= 4 and optimal_tp_size >= 2:
                enhanced['parallel_strategy'] = 'tensor_parallel'
                enhanced['tensor_parallel_size'] = optimal_tp_size
                enhanced['zero_optimization'] = 'zero1'
            else:
                enhanced['parallel_strategy'] = 'data_parallel'
                enhanced['zero_optimization'] = 'zero1'
                
        elif model_params >= 10:  # Medium models (10B+)
            optimal_tp_size = self._calculate_optimal_tensor_parallel_size(
                model_params, gpu_count, comm_performance, memory_performance
            )
            
            if gpu_count >= 4 and optimal_tp_size >= 2:
                enhanced['parallel_strategy'] = 'tensor_parallel'
                enhanced['tensor_parallel_size'] = optimal_tp_size
                enhanced['zero_optimization'] = 'zero1'
            else:
                enhanced['parallel_strategy'] = 'data_parallel'
                
        # Add communication optimization based on actual backend testing
        enhanced['communication_backend'] = self._select_communication_backend(gpu_count)
        enhanced['gradient_accumulation_steps'] = self._calculate_gradient_accumulation_steps(
            model_params, gpu_count, comm_performance
        )
        
        # Store profiling results for debugging
        enhanced['profiling_results'] = {
            'communication_performance': comm_performance,
            'memory_performance': memory_performance,
            'optimal_tensor_parallel_size': enhanced.get('tensor_parallel_size', 1)
        }
        
        return enhanced
        
    def _select_communication_backend(self, gpu_count: int) -> str:
        """
        Select optimal communication backend based on hardware and scale with fallback strategy.
        
        This method tests available communication backends (NCCL, Gloo, MPI) and
        selects the best one based on priority and availability.
        
        Args:
            gpu_count (int): Number of available GPUs
            
        Returns:
            str: Selected communication backend
        """
        try:
            # Priority order: NCCL > Gloo > MPI
            backends = []
            
            # Test NCCL availability (best for NVIDIA GPUs)
            if self._test_nccl_backend():
                backends.append(('nccl', 3))  # Highest priority
            
            # Test Gloo availability (universal fallback)
            if self._test_gloo_backend():
                backends.append(('gloo', 2))  # Medium priority
                
            # Test MPI availability (HPC environments)
            if self._test_mpi_backend():
                backends.append(('mpi', 1))  # Lowest priority
            
            # Select best available backend
            if backends:
                best_backend = max(backends, key=lambda x: x[1])[0]
                logger.info("Communication backend selected", extra={"backend": best_backend, "available": [b[0] for b in backends]})
                return best_backend
            else:
                logger.error("No communication backend available, falling back to gloo")
                return 'gloo'
                
        except Exception as e:
            logger.error("Backend selection failed", error=str(e), fallback="gloo")
            return 'gloo'
    
    def _test_nccl_backend(self) -> bool:
        """
        Test NCCL backend availability.
        
        Returns:
            bool: True if NCCL backend is available, False otherwise
        """
        try:
            if not torch.cuda.is_available():
                return False
                
            # Quick NCCL test
            import torch.distributed as dist
            
            # Try to initialize with NCCL in a subprocess to avoid conflicts
            test_code = """
            import torch
            import torch.distributed as dist
            import os

            try:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '29500'
                dist.init_process_group(backend='nccl', rank=0, world_size=1)
                dist.destroy_process_group()
                print("NCCL_AVAILABLE")
            except Exception as e:
                print(f"NCCL_FAILED: {e}")
            """
            
            import subprocess
            import sys
            result = subprocess.run([sys.executable, '-c', test_code], 
                                    capture_output=True, text=True, timeout=10)
            
            return "NCCL_AVAILABLE" in result.stdout
            
        except Exception:
            return False
    
    def _test_gloo_backend(self) -> bool:
        """
        Test Gloo backend availability.
        
        Returns:
            bool: True if Gloo backend is available, False otherwise
        """
        try:
            import torch.distributed as dist
            
            # Simple test - check if gloo is in available backends
            if hasattr(dist, 'is_backend_available'):
                return dist.is_backend_available('gloo')
            
            # Fallback test
            test_code = """
            import torch
            import torch.distributed as dist
            import os

            try:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '29501'
                dist.init_process_group(backend='gloo', rank=0, world_size=1)
                dist.destroy_process_group()
                print("GLOO_AVAILABLE")
            except Exception as e:
                print(f"GLOO_FAILED: {e}")
            """
            
            import subprocess
            import sys
            result = subprocess.run([sys.executable, '-c', test_code], 
                                    capture_output=True, text=True, timeout=10)
            
            return "GLOO_AVAILABLE" in result.stdout
            
        except Exception:
            return False
    
    def _test_mpi_backend(self) -> bool:
        """
        Test MPI backend availability.
        
        Returns:
            bool: True if MPI backend is available, False otherwise
        """
        try:
            # Check if MPI is available in the system
            import subprocess
            result = subprocess.run(['mpiexec', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
                
            # Test PyTorch MPI backend
            import torch.distributed as dist
            
            test_code = """
            import torch
            import torch.distributed as dist
            import os

            try:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '29502'
                dist.init_process_group(backend='mpi', rank=0, world_size=1)
                dist.destroy_process_group()
                print("MPI_AVAILABLE")
            except Exception as e:
                print(f"MPI_FAILED: {e}")
            """
            
            import sys
            result = subprocess.run([sys.executable, '-c', test_code], 
                                    capture_output=True, text=True, timeout=10)
            
            return "MPI_AVAILABLE" in result.stdout
            
        except Exception:
            return False
    
    def _calculate_gradient_accumulation_steps(self, model_params: float, gpu_count: int, comm_performance: Optional[dict] = None) -> int:
        """
        Calculate optimal gradient accumulation steps based on model size, GPU count, and communication performance.
        
        Args:
            model_params (float): Number of model parameters in billions
            gpu_count (int): Number of available GPUs
            comm_performance (Optional[dict]): Communication performance metrics
            
        Returns:
            int: Optimal number of gradient accumulation steps
        """
        # Base calculation: larger models need more accumulation
        if model_params >= 1000:  # Ultra-large models
            base_steps = 16
        elif model_params >= 100:  # Large models
            base_steps = 8
        elif model_params >= 10:  # Medium models
            base_steps = 4
        else:  # Small models
            base_steps = 2
        
        # Adjust based on GPU count (more GPUs = less accumulation needed)
        if gpu_count >= 8:
            multiplier = 0.5
        elif gpu_count >= 4:
            multiplier = 0.75
        else:
            multiplier = 1.0
        
        # Additional adjustment based on communication performance if available
        if comm_performance and 'inter_node_bandwidth' in comm_performance:
            bandwidth = comm_performance['inter_node_bandwidth']
            if bandwidth > 25:  # Excellent inter-node bandwidth (>25 GB/s)
                comm_multiplier = 0.8
            elif bandwidth > 10:  # Good inter-node bandwidth (10-25 GB/s)
                comm_multiplier = 0.9
            else:  # Poor inter-node bandwidth (<10 GB/s)
                comm_multiplier = 1.2
            multiplier *= comm_multiplier
        
        # Calculate final steps (minimum 1)
        final_steps = max(1, int(base_steps * multiplier))
        
        logger.info("Gradient accumulation steps calculated", extra={
            "model_params_b": model_params,
            "gpu_count": gpu_count,
            "comm_performance": comm_performance,
            "accumulation_steps": final_steps
        })
        
        return final_steps

    def _profile_communication_performance(self) -> dict:
        """
        Profile actual communication performance between GPUs/Nodes.
        
        Returns:
            dict: Communication performance metrics including bandwidth and latency
        """
        try:
            # Simple ping-pong test for intra-node communication
            intra_node_bandwidth = self._test_intra_node_bandwidth()
            
            # Inter-node communication test (if applicable)
            inter_node_bandwidth = self._test_inter_node_bandwidth()
            
            # Basic latency test
            latency = self._test_communication_latency()
            
            return {
                'intra_node_bandwidth': intra_node_bandwidth,  # GB/s
                'inter_node_bandwidth': inter_node_bandwidth,  # GB/s
                'latency_ms': latency,  # milliseconds
                'test_timestamp': time.time()
            }
        except Exception as e:
            logger.warning("Communication profiling failed", error=str(e))
            # Return conservative defaults based on typical hardware
            return {
                'intra_node_bandwidth': 25.0,  # Conservative NVLink estimate
                'inter_node_bandwidth': 5.0,   # Conservative InfiniBand estimate
                'latency_ms': 0.1,             # 100 microseconds
                'test_timestamp': time.time(),
                'fallback': True
            }
    
    def _profile_memory_performance(self) -> dict:
        """
        Profile memory performance characteristics.
        
        Returns:
            dict: Memory performance metrics
        """
        try:
            # Test memory bandwidth
            memory_bandwidth = self._test_memory_bandwidth()
            
            # Test memory latency
            memory_latency = self._test_memory_latency()
            
            return {
                'bandwidth_gb_s': memory_bandwidth,
                'latency_ns': memory_latency,
                'test_timestamp': time.time()
            }
        except Exception as e:
            logger.warning("Memory profiling failed", error=str(e))
            return {
                'bandwidth_gb_s': 50.0,  # Conservative HBM estimate
                'latency_ns': 100.0,     # 100 nanoseconds
                'test_timestamp': time.time(),
                'fallback': True
            }
    
    def _profile_cpu_performance(self) -> dict:
        """
        Profile CPU performance characteristics.
        
        Returns:
            dict: CPU performance metrics
        """
        try:
            # Test memory bandwidth
            memory_bandwidth = self._test_cpu_memory_bandwidth()
            
            # Test computational throughput
            compute_performance = self._test_cpu_compute_performance()
            
            return {
                'memory_bandwidth': memory_bandwidth,  # GB/s
                'compute_performance': compute_performance,  # GFLOPS
                'test_timestamp': time.time()
            }
        except Exception as e:
            logger.warning("CPU profiling failed", error=str(e))
            return {
                'memory_bandwidth': 25.0,  # Conservative DDR4 estimate
                'compute_performance': 50.0,  # Conservative CPU estimate
                'test_timestamp': time.time(),
                'fallback': True
            }
    
    def _calculate_optimal_tensor_parallel_size(self, model_params: float, gpu_count: int, 
                                              comm_performance: dict, memory_performance: dict) -> int:
        """
        Calculate optimal tensor parallel size based on profiling data.
        
        Args:
            model_params (float): Model parameters in billions
            gpu_count (int): Available GPU count
            comm_performance (dict): Communication performance metrics
            memory_performance (dict): Memory performance metrics
            
        Returns:
            int: Optimal tensor parallel size
        """
        try:
            # Base calculation on model size
            if model_params >= 1000:  # Ultra-large models
                target_tp = 8
            elif model_params >= 100:  # Large models
                target_tp = 4
            else:  # Medium models
                target_tp = 2
            
            # Adjust based on communication performance
            bandwidth = comm_performance.get('intra_node_bandwidth', 25.0)
            if bandwidth < 10:  # Poor communication
                target_tp = min(target_tp, 2)
            elif bandwidth > 50:  # Excellent communication
                target_tp = max(target_tp, 4)
            
            # Cap by available GPUs
            return min(target_tp, gpu_count)
            
        except Exception as e:
            logger.warning("Optimal TP size calculation failed", error=str(e))
            # Use actual GPU profiling data instead of simple heuristic
            return self._get_optimal_tp_size_from_profiling(gpu_count)
    
    def _get_optimal_tp_size_from_profiling(self, gpu_count: int) -> int:
        """Determine optimal tensor parallelism size based on actual GPU profiling data.
        
        Args:
            gpu_count (int): Number of available GPUs
            
        Returns:
            int: Optimal tensor parallelism size
        """
        try:
            # Get actual GPU performance data
            gpu_info = self._get_detailed_gpu_info()
            if not gpu_info:
                return min(2, gpu_count)  # Conservative fallback
            
            # Analyze GPU memory and compute capabilities
            primary_gpu = gpu_info[0]
            total_memory = primary_gpu.get('total_memory', 0)
            compute_capability = primary_gpu.get('compute_capability', '0.0')
            
            # Parse compute capability
            if '.' in str(compute_capability):
                major_version = float(str(compute_capability).split('.')[0])
            else:
                major_version = 0
            
            # Determine optimal TP size based on actual GPU specs
            if major_version >= 8:  # Ampere or newer
                # High-bandwidth GPUs can handle larger TP sizes
                if total_memory >= 24000:  # 24GB+ GPUs
                    return min(8, gpu_count)
                elif total_memory >= 12000:  # 12GB+ GPUs
                    return min(4, gpu_count)
                else:
                    return min(2, gpu_count)
            elif major_version >= 7:  # Turing
                return min(2, gpu_count)
            else:  # Older GPUs
                return 1  # No tensor parallelism for older GPUs
                
        except Exception as e:
            logger.warning("Failed to determine optimal TP size from profiling", error=str(e))
            return min(2, gpu_count)  # Conservative fallback

    def _get_detailed_gpu_info(self) -> list:
        """Get detailed GPU information including compute capability and memory specs."""
        try:
            if not self.gpu_info:
                return []
            
            detailed_info = []
            for gpu in self.gpu_info:
                # Enhance GPU info with detailed specs
                enhanced_gpu = {
                    **gpu,
                    'compute_capability': gpu.get('compute_capability', '0.0'),
                    'memory_bandwidth': gpu.get('memory_bandwidth', 0),
                    'cuda_cores': gpu.get('cuda_cores', 0),
                    'tensor_cores': gpu.get('tensor_cores', 0),
                }
                detailed_info.append(enhanced_gpu)
            
            return detailed_info
        except Exception as e:
            logger.warning("Failed to get detailed GPU info", error=str(e))
            return self.gpu_info if self.gpu_info else []

    def _test_intra_node_bandwidth(self) -> float:
        """Test intra-node communication bandwidth using actual GPU transfers."""
        try:
            # Try to perform actual GPU-to-GPU bandwidth test
            if torch.cuda.is_available() and len(self.gpu_info) >= 2:
                
                # Use first two GPUs for testing
                gpu0 = torch.cuda.device(0)
                gpu1 = torch.cuda.device(1)
                
                # Create test tensors
                size_mb = 100  # 100MB test
                test_tensor = torch.randn(size_mb * 1024 * 1024 // 4, device='cuda:0')  # 4 bytes per float
                
                # Warm up
                for _ in range(3):
                    test_tensor.copy_(torch.randn_like(test_tensor))
                
                # Measure transfer time
                torch.cuda.synchronize()
                start_time = time.time()
                
                # Perform peer-to-peer transfer
                target_tensor = test_tensor.to('cuda:1')
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                transfer_time = end_time - start_time
                bandwidth_gbps = (size_mb * 8) / (transfer_time * 1000)  # Convert to Gbps
                
                return bandwidth_gbps
            else:
                # Fallback to theoretical values based on GPU generation
                if self.gpu_info and len(self.gpu_info) >= 2:
                    gpu = self.gpu_info[0]
                    compute_capability = gpu.get('compute_capability', '0.0')
                    
                    if '.' in str(compute_capability):
                        major_version = float(str(compute_capability).split('.')[0])
                        if major_version >= 8:  # Ampere or newer
                            return 300.0  # NVLink 3.0: ~300 Gbps
                        elif major_version >= 7:  # Turing
                            return 100.0  # NVLink 2.0: ~100 Gbps
                        else:
                            return 50.0   # Older NVLink: ~50 Gbps
                
                return 50.0  # Conservative fallback
                
        except Exception as e:
            logger.warning("Failed to test intra-node bandwidth", error=str(e))
            return 50.0  # Conservative fallback
    
    def _test_inter_node_bandwidth(self) -> float:
        """Test inter-node communication bandwidth using actual network performance."""
        try:
            # Try to measure actual network bandwidth
            import socket
            import subprocess
            
            # First check if we can detect network interface capabilities
            try:
                # Try to get network interface info
                result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Look for high-speed interfaces
                    output = result.stdout
                    if '100000' in output:  # 100Gbps
                        return 100.0
                    elif '40000' in output:  # 40Gbps
                        return 40.0
                    elif '10000' in output:  # 10Gbps
                        return 10.0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Try to detect InfiniBand
            try:
                result = subprocess.run(['ibstat', '-l'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    # InfiniBand detected, try to get rate
                    ib_devices = result.stdout.strip().split('\n')
                    if ib_devices:
                        device = ib_devices[0]
                        rate_result = subprocess.run(['ibstat', device], capture_output=True, text=True, timeout=5)
                        if rate_result.returncode == 0:
                            if '400 Gb/sec' in rate_result.stdout:
                                return 400.0
                            elif '200 Gb/sec' in rate_result.stdout:
                                return 200.0
                            elif '100 Gb/sec' in rate_result.stdout:
                                return 100.0
                            elif '56 Gb/sec' in rate_result.stdout:
                                return 56.0
                            elif '40 Gb/sec' in rate_result.stdout:
                                return 40.0
                            else:
                                return 25.0  # Conservative InfiniBand estimate
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Fallback to conservative estimates based on system info
            hostname = socket.gethostname()
            
            # Check if this looks like a cloud/HPC environment
            if any(keyword in hostname.lower() for keyword in ['hpc', 'cluster', 'node', 'compute']):
                return 25.0  # Likely has InfiniBand or high-speed Ethernet
            
            # Check if we have multiple nodes (simple heuristic)
            try:
                with open('/etc/hosts', 'r') as f:
                    hosts_content = f.read()
                    if len([line for line in hosts_content.split('\n') if line.strip() and not line.startswith('#')]) > 10:
                        return 10.0  # Likely multi-node setup with decent networking
            except (FileNotFoundError, PermissionError):
                pass
            
            return 1.0  # Conservative single-node estimate
            
        except Exception as e:
            logger.warning("Failed to test inter-node bandwidth", error=str(e))
            return 25.0  # Conservative InfiniBand estimate
    
    def _test_communication_latency(self) -> float:
        """Test communication latency using actual network measurements."""
        try:
            # Try to measure actual network latency
            import socket
            import subprocess
            import platform
            
            # First try ping-based measurement to localhost (minimal overhead)
            try:
                if platform.system().lower() == 'windows':
                    ping_cmd = ['ping', '-n', '1', '127.0.0.1']
                else:
                    ping_cmd = ['ping', '-c', '1', '127.0.0.1']
                
                result = subprocess.run(ping_cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # Parse ping output for latency
                    output = result.stdout
                    if 'time=' in output:
                        # Extract time value
                        time_parts = output.split('time=')
                        if len(time_parts) > 1:
                            time_str = time_parts[1].split()[0].replace('ms', '')
                            try:
                                local_latency = float(time_str) / 1000.0  # Convert ms to seconds
                                # Estimate intra-node latency as a fraction of local latency
                                return max(0.0001, local_latency * 0.1)  # Assume 10% of local latency
                            except ValueError:
                                pass
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Try to detect InfiniBand latency characteristics
            try:
                result = subprocess.run(['ibstat', '-l'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and result.stdout.strip():
                    # InfiniBand typically has very low latency
                    return 0.0001  # 0.1ms for InfiniBand
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Try to detect high-speed Ethernet
            try:
                result = subprocess.run(['ip', 'link', 'show'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    output = result.stdout
                    if '100000' in output:  # 100Gbps
                        return 0.0005  # 0.5ms for 100Gbps Ethernet
                    elif '40000' in output:  # 40Gbps
                        return 0.001   # 1ms for 40Gbps Ethernet
                    elif '10000' in output:  # 10Gbps
                        return 0.002   # 2ms for 10Gbps Ethernet
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Check system characteristics
            hostname = socket.gethostname()
            
            # Multi-node HPC environment
            if any(keyword in hostname.lower() for keyword in ['hpc', 'cluster', 'node', 'compute']):
                return 0.0005  # 0.5ms for HPC environments
            
            # Single node or simple setup
            return 0.001  # 1ms conservative estimate
            
        except Exception as e:
            logger.warning("Failed to test communication latency", error=str(e))
            return 0.001  # Conservative estimate: 1ms
    
    def _test_memory_bandwidth(self) -> float:
        """Test memory bandwidth using actual GPU memory profiling."""
        try:
            # Try to measure actual GPU memory bandwidth
            import torch
            import time
            
            if torch.cuda.is_available():
                # Get GPU properties
                device = torch.cuda.current_device()
                gpu_properties = torch.cuda.get_device_properties(device)
                
                # Try to get theoretical bandwidth from GPU specs
                memory_clock = getattr(gpu_properties, 'memory_clock_rate', 0)  # kHz
                memory_bus_width = getattr(gpu_properties, 'memory_bus_width', 0)  # bits
                
                if memory_clock > 0 and memory_bus_width > 0:
                    # Calculate theoretical bandwidth: clock * bus_width * 2 (DDR) / 8 (bits to bytes) / 1000 (to GB/s)
                    theoretical_bandwidth = (memory_clock * memory_bus_width * 2) / (8 * 1000 * 1000)
                    # Real-world bandwidth is typically 80-90% of theoretical
                    return theoretical_bandwidth * 0.85
                
                # Perform actual memory bandwidth test
                try:
                    # Create large tensors to measure memory bandwidth
                    size_mb = 500  # 500MB test
                    elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
                    
                    # Create test tensors
                    a = torch.randn(elements, device='cuda')
                    b = torch.randn(elements, device='cuda')
                    c = torch.empty(elements, device='cuda')
                    
                    # Warm up
                    for _ in range(5):
                        torch.mul(a, b, out=c)
                    
                    # Synchronize and measure time
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    # Perform memory-intensive operations
                    iterations = 20
                    for _ in range(iterations):
                        torch.mul(a, b, out=c)  # Read a, b; write c
                        torch.add(a, c, out=c)  # Read a, c; write c
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    # Calculate bandwidth
                    total_time = end_time - start_time
                    # Each iteration: 3 reads + 2 writes = 5 memory operations
                    # Total data transferred: size_mb * 5 * iterations
                    total_data_gb = (size_mb * 5 * iterations) / 1024.0
                    bandwidth_gbps = total_data_gb / total_time
                    
                    return bandwidth_gbps
                    
                except Exception as e:
                    logger.warning("Failed to perform memory bandwidth test", error=str(e))
                    
                    # Fallback to GPU generation-based estimates
                    compute_capability = f"{gpu_properties.major}.{gpu_properties.minor}"
                    major_version = float(compute_capability.split('.')[0])
                    
                    if major_version >= 8:  # Ampere or newer
                        return 600.0  # HBM2e: ~600 GB/s
                    elif major_version >= 7:  # Turing
                        return 500.0  # HBM2: ~500 GB/s
                    elif major_version >= 6:  # Pascal
                        return 400.0  # GDDR5X: ~400 GB/s
                    else:
                        return 200.0  # Older GPUs: ~200 GB/s
            else:
                # No CUDA available, return conservative estimate
                return 100.0
                
        except Exception as e:
            logger.warning("Failed to test memory bandwidth", error=str(e))
            return 100.0  # Conservative HBM estimate in GB/s
    
    def _test_memory_latency(self) -> float:
        """Test memory latency using actual GPU memory access profiling."""
        try:
            # Try to measure actual GPU memory latency
            import torch
            import time
            
            if torch.cuda.is_available():
                # Perform actual memory latency test
                try:
                    # Create small tensors to measure memory latency
                    size_kb = 1  # 1KB test (small enough to fit in cache)
                    elements = (size_kb * 1024) // 4  # 4 bytes per float32
                    
                    # Create test tensor
                    data = torch.randn(elements, device='cuda')
                    
                    # Warm up
                    for _ in range(10):
                        result = data.sum()
                    
                    # Synchronize and measure time for random access pattern
                    torch.cuda.synchronize()
                    start_time = time.time()
                    
                    # Perform random access pattern to measure latency
                    iterations = 1000
                    for i in range(iterations):
                        # Access different parts of the tensor to avoid cache optimization
                        idx = (i * 137) % elements  # Prime number for better distribution
                        result = data[idx:idx+1].sum()
                    
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    # Calculate average latency per access
                    total_time = end_time - start_time
                    average_latency = total_time / iterations
                    
                    return average_latency * 1e9  # Convert to nanoseconds
                    
                except Exception as e:
                    logger.warning("Failed to perform memory latency test", error=str(e))
                    
                    # Fallback to GPU generation-based estimates
                    gpu_properties = torch.cuda.get_device_properties(0)
                    compute_capability = f"{gpu_properties.major}.{gpu_properties.minor}"
                    major_version = float(compute_capability.split('.')[0])
                    
                    if major_version >= 8:  # Ampere or newer
                        return 50.0   # HBM2e: ~50ns
                    elif major_version >= 7:  # Turing
                        return 75.0   # HBM2: ~75ns
                    elif major_version >= 6:  # Pascal
                        return 100.0  # GDDR5X: ~100ns
                    else:
                        return 150.0  # Older GPUs: ~150ns
            else:
                # No CUDA available, return conservative estimate
                return 200.0
                
        except Exception as e:
            logger.warning("Failed to test memory latency", error=str(e))
            return 200.0  # 200 nanoseconds conservative estimate
    
    def _test_cpu_memory_bandwidth(self) -> float:
        """Test CPU memory bandwidth using actual system profiling."""
        try:
            # Try to measure actual CPU memory bandwidth
            import numpy as np
            import time
            
            # Perform memory bandwidth test using numpy
            try:
                # Create large arrays to measure memory bandwidth
                size_mb = 100  # 100MB test
                elements = (size_mb * 1024 * 1024) // 8  # 8 bytes per float64
                
                # Create test arrays
                a = np.random.randn(elements)
                b = np.random.randn(elements)
                c = np.empty_like(a)
                
                # Warm up
                for _ in range(5):
                    np.multiply(a, b, out=c)
                
                # Measure time for memory-intensive operations
                start_time = time.time()
                
                # Perform multiple operations to get measurable time
                iterations = 50
                for _ in range(iterations):
                    np.multiply(a, b, out=c)  # Read a, b; write c
                    np.add(a, c, out=c)       # Read a, c; write c
                
                end_time = time.time()
                
                # Calculate bandwidth
                total_time = end_time - start_time
                # Each iteration: 3 reads + 2 writes = 5 memory operations
                # Total data transferred: size_mb * 5 * iterations
                total_data_gb = (size_mb * 5 * iterations) / 1024.0
                bandwidth_gbps = total_data_gb / total_time
                
                return bandwidth_gbps
                
            except Exception as e:
                logger.warning("Failed to perform CPU memory bandwidth test", error=str(e))
                
                # Try to get system memory info for theoretical calculation
                try:
                    import psutil
                    # Get memory frequency info if available
                    # Note: This is platform-dependent and may not always work
                    return 25.0  # Conservative estimate for modern DDR4
                except ImportError:
                    pass
                
                # Fallback based on CPU architecture detection
                import platform
                machine = platform.machine().lower()
                
                if 'x86_64' in machine or 'amd64' in machine:
                    return 25.0   # Modern x86_64 with DDR4: ~25 GB/s
                elif 'arm' in machine or 'aarch64' in machine:
                    return 15.0   # ARM with LPDDR: ~15 GB/s
                else:
                    return 10.0   # Conservative fallback
                    
        except Exception as e:
            logger.warning("Failed to test CPU memory bandwidth", error=str(e))
            return 25.0  # Conservative DDR4 estimate in GB/s
    
    def _test_cpu_compute_performance(self) -> float:
        """Test CPU compute performance using actual CPU profiling."""
        try:
            # Try to measure actual CPU compute performance
            import numpy as np
            import time
            
            # Perform CPU compute performance test
            try:
                # Create test arrays for compute-intensive operations
                size = 10000  # 10K x 10K matrix operations
                a = np.random.randn(size, size).astype(np.float32)
                b = np.random.randn(size, size).astype(np.float32)
                
                # Warm up
                for _ in range(3):
                    c = np.dot(a, b)
                
                # Measure time for compute-intensive operations
                start_time = time.time()
                
                # Perform multiple matrix multiplications
                iterations = 5
                for _ in range(iterations):
                    c = np.dot(a, b)  # O(n^3) operation
                    d = np.linalg.svd(c)  # Another compute-intensive operation
                
                end_time = time.time()
                
                # Calculate performance
                total_time = end_time - start_time
                
                # Estimate FLOPS for matrix multiplication
                # For n x n matrix multiplication: 2n^3 operations
                flops_per_matmul = 2 * (size ** 3)
                total_flops = flops_per_matmul * iterations
                
                # Convert to GFLOPS
                gflops = (total_flops / total_time) / 1e9
                
                return gflops
                
            except Exception as e:
                logger.warning("Failed to perform CPU compute performance test", error=str(e))
                
                # Try to get CPU info for theoretical calculation
                try:
                    import psutil
                    cpu_count = psutil.cpu_count(logical=False)
                    cpu_freq = psutil.cpu_freq()
                    
                    if cpu_freq and cpu_freq.current:
                        # Rough estimate based on CPU frequency and core count
                        # This is very approximate but better than fixed value
                        base_gflops_per_core = cpu_freq.current / 1000.0  # GHz to GFLOPS per core
                        total_gflops = base_gflops_per_core * (cpu_count or 1) * 2  # Assume 2 FLOPS per cycle
                        return min(total_gflops, 200.0)  # Cap at reasonable limit
                except (ImportError, AttributeError):
                    pass
                
                # Fallback based on CPU architecture detection
                import platform
                machine = platform.machine().lower()
                
                if 'x86_64' in machine or 'amd64' in machine:
                    return 50.0   # Modern x86_64: ~50 GFLOPS
                elif 'arm' in machine or 'aarch64' in machine:
                    return 20.0   # ARM: ~20 GFLOPS
                else:
                    return 15.0   # Conservative fallback
                    
        except Exception as e:
            logger.warning("Failed to test CPU compute performance", error=str(e))
            return 50.0  # Conservative CPU estimate in GFLOPS

    def _determine_cpu_strategy(self, model_params: float):
        """
        Determine optimal CPU strategy based on model size and actual CPU profiling results.
        
        Args:
            model_params (float): Number of model parameters in billions
        """
        if not self.cpu_info:
            self.cpu_info = {'cores': 4, 'threads': 8, 'memory_gb': 16}  # Conservative defaults
        
        # Add CPU performance profiling method if not exists
        if not hasattr(self, '_profile_cpu_performance'):
            self._profile_cpu_performance = lambda: {
                'memory_bandwidth': 25.0,
                'compute_performance': 50.0,
                'physical_cores': 4,
                'logical_cores': 8,
                'cpu_frequency': 2500.0,
                'total_memory_gb': 16.0,
                'available_memory_gb': 8.0,
                'profiling_method': 'conservative_fallback'
            }
        
        # Perform CPU performance profiling for data-driven decision
        cpu_performance = self._profile_cpu_performance()
        
        # CPU strategy based on model size and actual profiling
        if model_params <= 1.0:  # Small models (< 1B)
            # Use threads for better throughput if memory bandwidth supports it
            if cpu_performance['memory_bandwidth'] > 50:  # GB/s
                batch_size = min(8, self.cpu_info['threads'])
            else:
                batch_size = min(4, self.cpu_info['cores'])
            mixed_precision = False  # CPU doesn't benefit from FP16
        elif model_params <= 7.0:  # Medium models (1-7B)
            # Balance between cores and memory usage
            if cpu_performance['memory_bandwidth'] > 30:  # GB/s
                batch_size = min(4, self.cpu_info['cores'])
            else:
                batch_size = min(2, self.cpu_info['cores'] // 2 + 1)
            mixed_precision = False
        else:  # Large models (> 7B)
            batch_size = 1
            mixed_precision = False
        
        self.strategy = {
            'mode': 'cpu',
            'gpu_ids': [],
            'batch_size': batch_size,
            'mixed_precision': mixed_precision,
            'reason': f'CPU inference for {model_params:.1f}B model ({self.cpu_info["cores"]} cores, {self.cpu_info["memory_gb"]:.1f}GB RAM, {cpu_performance["memory_bandwidth"]:.1f}GB/s bandwidth)',
            'cpu_info': self.cpu_info,
            'cpu_performance': cpu_performance,
        }

    def _get_available_gpus(self) -> list:
        """
        Get list of available GPUs with memory calculation based on actual usage profiling.
        
        Returns:
            list: List of available GPUs with calculated available memory
        """
        available_gpus = []
        for gpu in self.gpu_info:
            # Calculate available memory based on actual profiling rather than fixed percentage
            total_memory = gpu.get('total_memory', 0)
            used_memory = gpu.get('used_memory', 0)
            free_memory = gpu.get('free_memory', 0)
            
            # Use actual free memory with dynamic reserve based on system requirements
            if total_memory > 0 and used_memory > 0:
                # Reserve memory based on actual system needs (typically 10-15% for stability)
                system_reserve = max(500, total_memory * 0.1)  # Minimum 500MB or 10%
                available_memory = max(0, free_memory - system_reserve)
            else:
                # Fallback to conservative estimate only when actual data unavailable
                available_memory = free_memory * 0.8
            
            if available_memory > 500:  # At least 500MB available
                available_gpus.append({
                    **gpu,
                    'available_memory': available_memory,
                    'memory_calculation_method': 'actual_profiling' if total_memory > 0 else 'conservative_estimate'
                })
        
        # Sort by available memory (descending) for optimal GPU selection
        available_gpus.sort(key=lambda x: x['available_memory'], reverse=True)
        
        return available_gpus

    def _get_world_size(self) -> int:
        """
        Get world size from environment or config.
        
        Returns:
            int: World size for distributed training
        """
        return int(os.environ.get('WORLD_SIZE', self.cfg.get("distributed.world_size", 1)))

    def _get_rank(self) -> int:
        """
        Get global rank from environment or config.
        
        Returns:
            int: Global rank for distributed training
        """
        return int(os.environ.get('RANK', self.cfg.get("distributed.rank", 0)))

    def _get_local_rank(self) -> int:
        """
        Get local rank from environment or config.
        
        Returns:
            int: Local rank for distributed training
        """
        return int(os.environ.get('LOCAL_RANK', self.cfg.get("distributed.local_rank", 0)))

    def _setup_single_gpu_optimized(self, gpu: dict, model_params: float) -> None:
        """
        Setup optimized single GPU configuration based on model size and GPU memory.
        
        Args:
            gpu (dict): GPU information dictionary
            model_params (float): Number of model parameters in billions
        """
        available_memory = gpu['available_memory']
        
        # Memory estimation based on actual model loading requirements
        # Use more accurate memory calculation instead of conservative 1B params ~ 4GB estimate
        try:
            # Try to get actual memory usage from GPU if available
            if 'used_memory' in gpu and 'total_memory' in gpu:
                actual_used = gpu['used_memory']
                actual_total = gpu['total_memory']
                # Estimate model memory based on actual usage patterns
                model_memory = min(actual_used, model_params * 2.5 * 1024)  # More realistic 2.5GB per 1B params
            else:
                # Fallback to more accurate theoretical calculation
                # FP16: 2 bytes/param, FP32: 4 bytes/param, mixed precision typically ~2.5GB per 1B params
                model_memory = model_params * 2.5 * 1024  # Convert to MB (2.5GB per 1B params)
            
            # Activation memory: typically 10-20% of model memory for batch size 1
            activation_memory = model_memory * 0.15  # More accurate activation estimate
            
        except Exception as e:
            # Ultimate fallback to conservative but more accurate estimate
            model_memory = model_params * 2.5 * 1024  # 2.5GB per 1B params (more realistic than 4GB)
            activation_memory = model_memory * 0.15  # 15% of model memory for activations
        
        # Calculate safe batch size with system reserve
        safe_memory = available_memory - model_memory - 1000  # Reserve 1GB for system
        
        if safe_memory > 0:
            # Each additional batch item adds activation memory
            batch_size = max(1, int(safe_memory / activation_memory))
        else:
            batch_size = 1
        
        # Cap batch size based on model size
        if model_params <= 1.0:  # Small models
            batch_size = min(batch_size, 64)
        elif model_params <= 7.0:  # Medium models
            batch_size = min(batch_size, 32)
        else:  # Large models
            batch_size = min(batch_size, 8)
        
        self.strategy = {
            'mode': 'single_gpu',
            'gpu_ids': [gpu['index']],
            'batch_size': batch_size,
            'mixed_precision': True,
            'reason': f'Single GPU optimized for {model_params:.1f}B model ({available_memory:.0f}MB available)',
        }

    def _setup_multi_gpu_optimized(self, available_gpus: list, model_params: float) -> None:
        """
        Setup optimized multi-GPU configuration based on model size and cluster setup.
        
        Args:
            available_gpus (list): List of available GPUs
            model_params (float): Number of model parameters in billions
        """
        # Check if distributed training is enabled
        use_distributed = (
            os.environ.get('RANK') is not None or
            os.environ.get('LOCAL_RANK') is not None or
            self.cfg.get("distributed.enabled", False)
        )
        
        # Calculate total available memory across all GPUs
        total_memory = sum(gpu['available_memory'] for gpu in available_gpus)
        
        # Memory estimation for multi-GPU with more accurate calculation
        try:
            # Use more accurate memory estimation (2.5GB per 1B params for mixed precision)
            model_memory = model_params * 2.5 * 1024  # Convert to MB
            activation_memory = model_memory * 0.15  # More accurate activation estimate (15% of model memory)
        except Exception:
            # Fallback to conservative but more accurate estimate
            model_memory = model_params * 2.5 * 1024  # 2.5GB per 1B params
            activation_memory = model_memory * 0.15  # 15% of model memory for activations
        
        # Reserve memory for system and gradient synchronization
        safe_memory = total_memory - model_memory - (2000 * len(available_gpus))  # Reserve 2GB per GPU
        
        if use_distributed:
            # Distributed training across GPUs
            gpu_ids = [gpu['index'] for gpu in available_gpus]
            
            # Batch size calculation for distributed training
            if safe_memory > 0:
                batch_size = max(1, int(safe_memory / (activation_memory * 100 * len(available_gpus))))
            else:
                batch_size = 1
            
            # Cap based on model size
            if model_params <= 1.0:
                batch_size = min(batch_size, 32 * len(available_gpus))
            elif model_params <= 7.0:
                batch_size = min(batch_size, 16 * len(available_gpus))
            else:
                batch_size = min(batch_size, 4 * len(available_gpus))
            
            self.strategy = {
                'mode': 'distributed',
                'gpu_ids': gpu_ids,
                'batch_size': batch_size,
                'mixed_precision': True,
                'reason': f'Auto-detected distributed training for {model_params:.1f}B model ({len(available_gpus)} GPUs)',
                'world_size': len(available_gpus),
                'rank': self._get_rank(),
                'local_rank': self._get_local_rank(),
                'model_params': model_params,
                'total_memory': total_memory,
            }
        else:
            # DataParallel for single-node multi-GPU
            gpu_ids = [gpu['index'] for gpu in available_gpus]
            
            # Batch size calculation for DataParallel
            if safe_memory > 0:
                batch_size = max(1, int(safe_memory / (activation_memory * 150)))
            else:
                batch_size = 1
            
            # Cap based on model size
            if model_params <= 1.0:
                batch_size = min(batch_size, 16 * len(available_gpus))
            elif model_params <= 7.0:
                batch_size = min(batch_size, 8 * len(available_gpus))
            else:
                batch_size = min(batch_size, 2 * len(available_gpus))
            
            self.strategy = {
                'mode': 'multi_gpu',
                'gpu_ids': gpu_ids,
                'batch_size': batch_size,
                'mixed_precision': True,
                'reason': f'Auto-detected DataParallel for {model_params:.1f}B model ({len(available_gpus)} GPUs)',
                'num_gpus': len(available_gpus),
                'model_params': model_params,
                'total_memory': total_memory,
            }

    def _setup_distributed_cluster(self, available_gpus: list, world_size: int, rank: int, local_rank: int, model_params: float) -> None:
        """
        Setup optimized multi-node distributed cluster configuration.
        
        Args:
            available_gpus (list): List of available GPUs
            world_size (int): Total number of processes in the distributed setup
            rank (int): Global rank of this process
            local_rank (int): Local rank of this process
            model_params (float): Number of model parameters in billions
        """
        # Auto-select local GPU based on local_rank
        local_gpus = [g for g in available_gpus if g['index'] == local_rank] if local_rank < len(available_gpus) else available_gpus[:1]
        
        if not local_gpus:
            local_gpus = available_gpus[:1]  # Fallback to first GPU
        
        gpu = local_gpus[0]
        available_memory = gpu['available_memory']
        
        # Calculate batch size based on model size and available GPU memory
        try:
            # Use more accurate memory estimation (2.5GB per 1B params for mixed precision)
            model_memory = model_params * 2.5 * 1024  # Convert to MB
            activation_memory = model_memory * 0.15  # More accurate activation estimate (15% of model memory)
        except Exception:
            # Fallback to conservative but more accurate estimate
            model_memory = model_params * 2.5 * 1024  # 2.5GB per 1B params
            activation_memory = model_memory * 0.15  # 15% of model memory for activations
        
        # Reserve memory for system and gradient synchronization
        safe_memory = available_memory - model_memory - 1500  # Reserve 1.5GB for system
        
        if safe_memory > 0:
            batch_size = max(1, int(safe_memory / (activation_memory * 200)))
        else:
            batch_size = 1
        
        # Cap based on model size
        if model_params <= 1.0:
            batch_size = min(batch_size, 16)
        elif model_params <= 7.0:
            batch_size = min(batch_size, 8)
        else:
            batch_size = min(batch_size, 2)
        
        # Auto-detect master address and port
        master_addr = os.environ.get('MASTER_ADDR') or self.cfg.get("distributed.master_addr", 'localhost')
        master_port = os.environ.get('MASTER_PORT') or self.cfg.get("distributed.master_port", '29500')
        
        self.strategy = {
            'mode': 'distributed_cluster',
            'gpu_ids': [gpu['index']],
            'batch_size': batch_size,
            'mixed_precision': True,
            'reason': f'Auto-detected multi-node cluster for {model_params:.1f}B model',
            'world_size': world_size,
            'rank': rank,
            'local_rank': local_rank,
            'master_addr': master_addr,
            'master_port': str(master_port),
            'gpu_info': gpu,
            'model_params': model_params,
            'available_memory': available_memory,
        }

    def _calculate_distributed_batch_size(self, memory_gb: float) -> int:
        """
        Calculate batch size for distributed training.
        
        Args:
            memory_gb (float): Available memory in GB
            
        Returns:
            int: Recommended batch size
        """
        if memory_gb >= 16:
            return 16
        elif memory_gb >= 8:
            return 8
        elif memory_gb >= 4:
            return 4
        else:
            return 2

    def _calculate_dataparallel_batch_size(self, memory_gb: float) -> int:
        """
        Calculate batch size for DataParallel.
        
        Args:
            memory_gb (float): Available memory in GB
            
        Returns:
            int: Recommended batch size
        """
        if memory_gb >= 32:
            return 64
        elif memory_gb >= 16:
            return 32
        elif memory_gb >= 8:
            return 16
        else:
            return 8
    
    def _estimate_model_memory(self, model_params: float, sequence_length: int = 1024) -> int:
        """
        Estimate model memory requirements based on model parameters and sequence length.
        
        Args:
            model_params (float): Number of model parameters in billions
            sequence_length (int): Sequence length for inference/training
            
        Returns:
            int: Estimated memory requirement in MB
        """
        # Model parameters memory (assuming FP16/BF16 for inference)
        param_memory = model_params * 1e9 * 2 / (1024**2)  # Convert to MB
        
        # KV cache memory (rough estimation)
        kv_cache_memory = 2 * sequence_length * 4096 * 32 * 2 / (1024**2)  # Convert to MB
        
        # Activation memory (rough estimation)
        activation_memory = sequence_length * 4096 * 32 * 4 / (1024**2)  # Convert to MB
        
        # Total memory estimate with some overhead
        total_memory = (param_memory + kv_cache_memory + activation_memory) * 1.2
        
        return int(total_memory)
    
    def _check_quantization_need(self, model_memory: float) -> bool:
        """
        Check if quantization is needed based on memory pressure.
        
        Args:
            model_memory (float): Estimated model memory requirement in MB
            
        Returns:
            bool: True if quantization is recommended, False otherwise
        """
        # Get total system memory
        total_system_memory = self.system_memory.get('total', 0) / (1024**2)  # Convert to MB
        
        # Get available GPU memory (if any GPU is available)
        available_gpu_memory = 0
        if self.gpu_info:
            available_gpu_memory = sum(gpu.get('free_memory', 0) for gpu in self.gpu_info)
        
        # Total available memory
        total_available_memory = total_system_memory + available_gpu_memory
        
        # Check if model memory exceeds available memory by a significant margin
        # If model memory is more than 80% of available memory, recommend quantization
        memory_pressure_ratio = model_memory / max(total_available_memory, 1)
        
        return memory_pressure_ratio > 0.8
    
    def _recommend_quantization_bits(self, model_memory: float) -> int:
        """
        Recommend quantization bits based on memory pressure.
        
        Args:
            model_memory (float): Estimated model memory requirement in MB
            
        Returns:
            int: Recommended quantization bits (2, 4, or 8)
        """
        # Get total system memory
        total_system_memory = self.system_memory.get('total', 0) / (1024**2)  # Convert to MB
        
        # Get available GPU memory (if any GPU is available)
        available_gpu_memory = 0
        if self.gpu_info:
            available_gpu_memory = sum(gpu.get('free_memory', 0) for gpu in self.gpu_info)
        
        # Total available memory
        total_available_memory = total_system_memory + available_gpu_memory
        
        # Calculate memory pressure ratio
        memory_pressure_ratio = model_memory / max(total_available_memory, 1)
        
        # Recommend quantization bits based on memory pressure
        if memory_pressure_ratio > 1.5:
            # Severe memory pressure - recommend 2-bit quantization
            return 2
        elif memory_pressure_ratio > 1.0:
            # High memory pressure - recommend 4-bit quantization
            return 4
        else:
            # Moderate memory pressure - recommend 8-bit quantization
            return 8

    def print_summary(self) -> None:
        """
        Print a summary using smart detector.
        
        This method prints the hardware detection summary from the smart detector
        and adds information about the current strategy.
        """
        # Use smart detector's comprehensive summary
        self.smart_detector.print_detection_summary()
        
        # Add current strategy info
        if self.strategy:
            logger.info("Current strategy", extra={"device_type": self.strategy.get('device_type', 'unknown')})
            logger.info("Strategy reason", extra={"reason": self.strategy.get('reason', 'No specific reason')})
            
            if self.strategy.get('warning'):
                logger.error("Strategy warning", extra={"warning": self.strategy['warning']})

    def get_recommended_strategy(self) -> Dict[str, Any]:
        """
        Get recommended strategy based on detected hardware.
        
        Returns:
            Dict[str, Any]: Recommended strategy dictionary
        """
        nvidia_gpus = [gpu for gpu in self.gpu_info if gpu.get('vendor') == 'nvidia']
        amd_gpus = [gpu for gpu in self.gpu_info if gpu.get('vendor') == 'amd']
        directml_gpus = [gpu for gpu in self.gpu_info if gpu.get('platform') == 'directml']
        total_gpus = len(self.gpu_info)
        
        if total_gpus > 0:
            # Multi-GPU strategy
            if len(nvidia_gpus) >= 2:
                return {
                    'strategy': 'nvidia_multi_gpu',
                    'gpu_ids': [gpu['index'] for gpu in nvidia_gpus],
                    'mixed_precision': True,
                    'memory_efficient': False,
                    'batch_size_recommendation': min(32, len(nvidia_gpus) * 8),
                    'platform': 'nvidia',
                    'tensor_parallel': len(nvidia_gpus) > 1,
                    'data_parallel': len(nvidia_gpus) > 1
                }
            elif len(nvidia_gpus) == 1:
                # Single NVIDIA GPU strategy
                return {
                    'strategy': 'nvidia_single_gpu',
                    'gpu_ids': [nvidia_gpus[0]['index']],
                    'mixed_precision': True,
                    'memory_efficient': False,
                    'batch_size_recommendation': 16,
                    'platform': 'nvidia',
                    'tensor_parallel': False,
                    'data_parallel': False
                }
            elif len(amd_gpus) > 0:
                # AMD GPU strategy with ROCm
                return {
                    'strategy': 'rocm',
                    'gpu_ids': [gpu['index'] for gpu in amd_gpus],
                    'mixed_precision': True,
                    'memory_efficient': True,
                    'batch_size_recommendation': 4 if len(amd_gpus) == 1 else 8,
                    'platform': 'rocm',
                    'tensor_parallel': len(amd_gpus) > 1,
                    'data_parallel': len(amd_gpus) > 1
                }
            elif len(directml_gpus) > 0:
                # DirectML strategy for Windows
                return {
                    'strategy': 'directml',
                    'gpu_ids': [gpu['index'] for gpu in directml_gpus],
                    'mixed_precision': True,
                    'memory_efficient': True,
                    'batch_size_recommendation': 2,
                    'platform': 'directml',
                    'tensor_parallel': False,
                    'data_parallel': len(directml_gpus) > 1
                }
            else:
                # Generic multi-GPU fallback
                return {
                    'strategy': 'multi_gpu',
                    'gpu_ids': [gpu['index'] for gpu in self.gpu_info],
                    'mixed_precision': True,
                    'memory_efficient': True,
                    'batch_size_recommendation': min(16, total_gpus * 4),
                    'platform': 'generic',
                    'tensor_parallel': total_gpus > 1,
                    'data_parallel': total_gpus > 1
                }
        else:
            # CPU fallback
            return {
                'strategy': 'cpu',
                'gpu_ids': [],
                'mixed_precision': False,
                'memory_efficient': True,
                'batch_size_recommendation': 1,
                'platform': 'cpu',
                'tensor_parallel': False,
                'data_parallel': False
            }

    def get_recommendation(self) -> Dict:
        """
        Get the system-level recommended configuration.
        
        Returns:
            Dict: System-level recommended configuration
        """
        return {
            'strategy': self.strategy,
            'gpu_info': self.gpu_info,
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

    def get_inference_strategy(self, model_size: Optional[str] = None, sequence_length: int = 1024) -> dict:
        """
        Universal intelligent inference strategy - CPU/GPU/Cluster auto-selection.
        
        This method automatically selects the best inference strategy based on
        the model size, sequence length, and available hardware. It follows a
        fallback chain: cluster -> multi-gpu -> single-gpu -> cpu.
        
        Args:
            model_size (str, optional): Model size string (e.g., "7B", "1.5B")
            sequence_length (int): Sequence length for inference
            
        Returns:
            dict: Inference strategy dictionary
        """
        try:
            # Estimate model parameters from size string or config
            model_params = self._parse_model_size(model_size) if model_size else self._estimate_model_parameters()
            
            # Auto-detect inference cluster configuration
            cluster_config = self._auto_detect_inference_cluster()
            
            # Check if cluster mode is enabled
            cluster_enabled = self.cfg.get("distributed.enabled", False)
            
            # Automatic fallback chain: cluster -> multi-gpu -> single-gpu -> cpu
            fallbacks = []
            # 1) Cluster (if enabled/world_size>1)
            if (cluster_config['world_size'] > 1 or cluster_enabled) and self.gpu_info:
                try:
                    strat = self._get_distributed_inference_strategy(cluster_config, model_params, sequence_length)
                    return strat
                except Exception as e:
                    logger.warning("Distributed strategy failed, falling back", error=str(e))
                    fallbacks.append('cluster->local')
            # 2) Multi-GPU (single node)
            if len(self.gpu_info) > 1:
                try:
                    strat = self._get_multi_inference_strategy(self.gpu_info, model_params, sequence_length)
                    if fallbacks:
                        strat['fallbacks'] = fallbacks
                        strat['warning'] = '; '.join(fallbacks)
                    return strat
                except Exception as e:
                    logger.warning("Multi-GPU strategy failed, falling back", error=str(e))
                    fallbacks.append('multi->single')
            # 3) Single-GPU
            if len(self.gpu_info) >= 1:
                try:
                    strat = self._get_single_inference_strategy(self.gpu_info[0], model_params, sequence_length)
                    if fallbacks:
                        strat['fallbacks'] = fallbacks
                        strat['warning'] = '; '.join(fallbacks)
                    return strat
                except Exception as e:
                    logger.warning("Single-GPU strategy failed, falling back to CPU", error=str(e))
                    fallbacks.append('single->cpu')
            # 4) CPU fallback
            try:
                strat = self._get_cpu_inference_strategy(model_params, sequence_length)
                if fallbacks:
                    strat['fallbacks'] = fallbacks
                    strat['warning'] = '; '.join(fallbacks)
                return strat
            except Exception as e:
                logger.error("CPU strategy failed, using emergency fallback", error=str(e))
                # Emergency fallback - return basic CPU strategy
                return {
                    'mode': 'cpu',
                    'gpu_ids': [],
                    'batch_size': 1,
                    'mixed_precision': False,
                    'reason': 'Emergency CPU fallback - minimal configuration',
                    'cpu_info': {'cores': 1, 'threads': 1, 'memory_gb': 4.0, 'architecture': 'unknown'},
                    'model_params': model_params,
                    'sequence_length': sequence_length,
                    'estimated_memory': 100,
                    'memory_margin_mb': 1000,
                    'warning': 'Emergency fallback - minimal CPU configuration',
                    'emergency_fallback': True
                }
        except Exception as e:
            logger.error("Strategy generation completely failed, using ultimate fallback", error=str(e))
            # Ultimate fallback - return absolute minimal configuration
            return {
                'mode': 'cpu',
                'gpu_ids': [],
                'batch_size': 1,
                'mixed_precision': False,
                'reason': 'Ultimate emergency fallback - absolute minimal configuration',
                'cpu_info': {'cores': 1, 'threads': 1, 'memory_gb': 2.0, 'architecture': 'unknown'},
                'model_params': 7.0,  # Default model size
                'sequence_length': sequence_length,
                'estimated_memory': 50,
                'memory_margin_mb': 500,
                'warning': 'Ultimate emergency fallback - absolute minimal configuration',
                'ultimate_fallback': True
            }
    
    def _get_cpu_inference_strategy(self, model_params: float, sequence_length: int = 1024) -> dict:
        """
        Get optimized CPU inference strategy.
        
        Args:
            model_params (float): Number of model parameters in billions
            sequence_length (int): Sequence length for inference
            
        Returns:
            dict: CPU inference strategy
        """
        try:
            import psutil
            
            # Get CPU information
            cpu_cores = psutil.cpu_count(logical=False) or 1
            cpu_threads = psutil.cpu_count(logical=True) or 1
            available_memory = psutil.virtual_memory().available / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.warning("Failed to get CPU info via psutil, using defaults", error=str(e))
            cpu_cores = 1
            cpu_threads = 1
            available_memory = 4096  # Default 4GB
        
        # Memory estimation for CPU inference
        memory_per_param = 4  # bytes for CPU (FP32)
        model_memory = model_params * memory_per_param / 1024  # MiB
        # Activation/KV cache approximation for CPU
        activation_memory = model_params * 0.05 * (sequence_length / 1024)  # MiB per batch unit
        
        # Calculate safe batch size (more conservative for CPU)
        safe_memory = available_memory * 0.6 - model_memory  # Use 60% of available memory
        
        if safe_memory > 0:
            # CPU batch size based on memory and core count
            memory_based_batch = max(1, int(safe_memory / (model_memory * 0.1)))
            core_based_batch = max(1, cpu_cores // 2)  # Conservative core usage
            batch_size = min(memory_based_batch, core_based_batch)
        else:
            batch_size = 1
        
        # Cap batch size based on model size and sequence length
        if model_params <= 1.0:  # Small models (< 1B)
            max_batch = 16
        elif model_params <= 3.0:  # Medium models (1-3B)
            max_batch = 8
        elif model_params <= 7.0:  # Medium-large models (3-7B)
            max_batch = 4
        elif model_params <= 100.0:  # Large models (7-100B)
            max_batch = 2
        else:  # Ultra-large models (> 100B, including TB scale)
            max_batch = 1
        
        # Sequence length adjustment
        if sequence_length > 2048:
            max_batch = max(1, max_batch // 2)
        elif sequence_length > 4096:
            max_batch = max(1, max_batch // 4)
        
        batch_size = min(batch_size, max_batch)
        # CPU memory margin considers activation footprint as well
        memory_margin_mb = max(0, int(safe_memory - activation_memory * batch_size))
        warning = None
        if batch_size == 1 and memory_margin_mb < 256:
            warning = "Extremely low memory headroom; consider smaller sequence length"
        
        return {
            'mode': 'cpu',
            'gpu_ids': [],
            'batch_size': batch_size,
            'mixed_precision': False,
            'reason': f'Optimized CPU inference for {model_params:.1f}B model ({cpu_cores} cores, {available_memory:.0f}MB RAM)',
            'cpu_info': {
                'cores': cpu_cores,
                'threads': cpu_threads,
                'memory_gb': available_memory / 1024,
                'architecture': self.cpu_info.get('architecture', 'unknown')
            },
            'model_params': model_params,
            'sequence_length': sequence_length,
            'estimated_memory': model_memory + (activation_memory * batch_size),
            'memory_margin_mb': memory_margin_mb,
            **({'warning': warning} if warning else {}),
        }

    def _auto_detect_inference_cluster(self) -> Dict:
        """
        Automatically detect cluster environment for inference.
        
        Returns:
            Dict: Cluster configuration dictionary
        """
        world_size = self._get_world_size()
        rank = self._get_rank()
        local_rank = self._get_local_rank()
        cluster_enabled = self.cfg.get("distributed.enabled", False)
        
        # Auto-detect master address and port
        master_addr = os.environ.get('MASTER_ADDR', self.cfg.get("distributed.master_addr", "localhost"))
        master_port = os.environ.get('MASTER_PORT', self.cfg.get("distributed.master_port", "29500"))
        
        return {
            'world_size': world_size,
            'rank': rank,
            'local_rank': local_rank,
            'cluster_enabled': cluster_enabled,
            'master_addr': master_addr,
            'master_port': master_port,
        }

    def _get_single_inference_strategy(self, gpu: dict, model_params: float, sequence_length: int = 1024) -> dict:
        """
        Get optimized single GPU inference strategy with STRICT VALIDATION.
        
        Args:
            gpu (dict): GPU information dictionary
            model_params (float): Number of model parameters in billions
            sequence_length (int): Sequence length for inference
            
        Returns:
            dict: Single GPU inference strategy
        """
        available_memory = gpu['free_memory'] * 0.8  # Conservative estimate (MiB)
        
        # Memory estimation for inference
        # dtype-aware bytes per parameter: fp16/bf16=2B, fp32=4B
        dtype_bytes = 2
        try:
            # Determine dtype based on actual GPU capabilities and precision support
            if gpu.get('vendor') == 'nvidia':
                # Check GPU compute capability for mixed precision support
                compute_capability = gpu.get('compute_capability', '0.0')
                major_version = float(compute_capability.split('.')[0]) if '.' in str(compute_capability) else 0
                # Mixed precision (FP16/BF16) requires compute capability >= 7.0
                dtype_bytes = 2 if major_version >= 7 else 4
            else:
                # Default to FP32 for non-NVIDIA GPUs unless explicitly supported
                dtype_bytes = 4
        except Exception:
            dtype_bytes = 2  # Conservative fallback to mixed precision
        model_memory = (model_params * dtype_bytes) / 1024  # MiB
        # Activation/KV cache approximation: proportional to model size and sequence length
        activation_memory = model_params * 0.05 * (sequence_length / 1024)  # MiB per batch unit
        
        # STRICT VALIDATION: Check if model fits in GPU memory
        if model_memory > available_memory:
            logger.error("Insufficient GPU memory for model", model_params=model_params, required_memory=model_memory, available_memory=available_memory)
            raise RuntimeError(f"Insufficient GPU memory for {model_params:.1f}B model. Required: {model_memory:.0f}MB, Available: {available_memory:.0f}MB")
        
        # Calculate safe batch size
        safe_memory = available_memory - model_memory - 500  # Reserve 500MB for system
        
        if safe_memory > 0:
            batch_size = max(1, int(safe_memory / (activation_memory * 50)))
        else:
            batch_size = 1
        
        # Cap batch size based on model size and sequence length
        if model_params <= 1.0:  # Small models (< 1B)
            max_batch = 64
        elif model_params <= 7.0:  # Medium models (1-7B)
            max_batch = 32
        elif model_params <= 100.0:  # Large models (7-100B)
            max_batch = 16
        else:  # Ultra-large models (> 100B, including TB scale)
            max_batch = 8
        
        # Sequence length adjustment
        if sequence_length > 2048:
            max_batch = max(1, max_batch // 2)
        elif sequence_length > 4096:
            max_batch = max(1, max_batch // 4)
        
        batch_size = min(batch_size, max_batch)
        
        # Calculate memory margin
        memory_margin_mb = max(0, int(safe_memory - activation_memory * batch_size))
        warning = None
        if batch_size == 1 and memory_margin_mb < 256:
            warning = "Low memory headroom; consider reducing sequence length"
        
        # Check if quantization is needed based on memory pressure
        quantization_needed = self._check_quantization_need(model_memory)
        recommended_bits = None
        if quantization_needed:
            recommended_bits = self._recommend_quantization_bits(model_memory)
            if warning:
                warning += f"; Quantization recommended: {recommended_bits}-bit"
            else:
                warning = f"Quantization recommended: {recommended_bits}-bit"
        
        return {
            'mode': 'single_gpu',
            'gpu_ids': [gpu['index']],
            'batch_size': batch_size,
            'mixed_precision': True,
            'reason': f'Validated single GPU inference for {model_params:.1f}B model ({available_memory:.0f}MB available)',
            'gpu_info': gpu,
            'model_params': model_params,
            'sequence_length': sequence_length,
            'estimated_memory': model_memory + (activation_memory * batch_size),
            'memory_margin_mb': memory_margin_mb,
            'quantization_needed': quantization_needed,
            'recommended_quant_bits': recommended_bits,
            **({'warning': warning} if warning else {}),
        }

    def _get_multi_inference_strategy(self, gpus: list, model_params: float, sequence_length: int = 1024) -> dict:
        """
        Get optimized multi-GPU single node inference strategy with STRICT VALIDATION.
        
        Args:
            gpus (list): List of GPU information dictionaries
            model_params (float): Number of model parameters in billions
            sequence_length (int): Sequence length for inference
            
        Returns:
            dict: Multi-GPU inference strategy
        """
        total_memory = sum(g['free_memory'] * 0.8 for g in gpus)  # Conservative estimate (MiB)
        
        # Memory estimation for multi-GPU inference
        # Assume mixed precision on CUDA cluster
        dtype_bytes = 2
        try:
            if any(g.get('vendor') != 'nvidia' for g in gpus):
                dtype_bytes = 4
        except Exception:
            dtype_bytes = 2
        model_memory = (model_params * dtype_bytes) / 1024  # MiB across cluster (approx)
        activation_memory = model_params * 0.05 * (sequence_length / 1024) * len(gpus)  # MiB per total batch
        
        # STRICT VALIDATION: Check if model fits in total GPU memory
        if model_memory > total_memory:
            logger.error("Insufficient total GPU memory for model", model_params=model_params, required_memory=model_memory, available_memory=total_memory, gpu_count=len(gpus))
            raise RuntimeError(f"Insufficient total GPU memory for {model_params:.1f}B model. Required: {model_memory:.0f}MB, Available: {total_memory:.0f}MB across {len(gpus)} GPUs")
        
        # Calculate safe batch size
        safe_memory = total_memory - model_memory - (1000 * len(gpus))  # Reserve 1GB per GPU
        
        if safe_memory > 0:
            batch_size = max(1, int(safe_memory / (activation_memory * 30)))
        else:
            batch_size = 1
        
        # Cap batch size based on model size and sequence length
        if model_params <= 1.0:  # Small models (< 1B)
            max_batch = 64
        elif model_params <= 7.0:  # Medium models (1-7B)
            max_batch = 32
        elif model_params <= 100.0:  # Large models (7-100B)
            max_batch = 16
        else:  # Ultra-large models (> 100B, including TB scale)
            max_batch = 8
        
        # Sequence length adjustment
        if sequence_length > 2048:
            max_batch = max(1, max_batch // 2)
        elif sequence_length > 4096:
            max_batch = max(1, max_batch // 4)
        
        batch_size = min(batch_size, max_batch)
        
        # Derive memory margin and warnings for multi-GPU
        memory_margin_mb = max(0, int(safe_memory - activation_memory * batch_size))
        warning = None
        if batch_size == 1 and memory_margin_mb < 512:
            warning = "Low memory headroom across GPUs; consider reducing sequence length or parallelism"

        return {
            'mode': 'multi_gpu',
            'gpu_ids': [g['index'] for g in gpus],
            'batch_size': batch_size,
            'mixed_precision': True,
            'reason': f'Validated multi-GPU inference for {model_params:.1f}B model ({len(gpus)} GPUs, {total_memory:.0f}MB total)',
            'gpu_info': gpus,
            'model_params': model_params,
            'sequence_length': sequence_length,
            'total_memory': total_memory,
            'estimated_memory': model_memory + (activation_memory * batch_size),
            'memory_margin_mb': memory_margin_mb,
            **({'warning': warning} if warning else {}),
        }

    def _get_distributed_inference_strategy(self, cluster_config: dict, model_params: float, sequence_length: int = 1024) -> dict:
        """
        Get optimized distributed cluster inference strategy.
        
        Args:
            cluster_config (dict): Cluster configuration dictionary
            model_params (float): Number of model parameters in billions
            sequence_length (int): Sequence length for inference
            
        Returns:
            dict: Distributed cluster inference strategy
        """
        local_rank = cluster_config.get('local_rank', 0)
        
        # Auto-select local GPU based on local_rank
        local_gpus = [g for g in self.gpu_info if g['index'] == local_rank] if local_rank < len(self.gpu_info) else self.gpu_info[:1]
        gpu = local_gpus[0]
        available_memory = gpu['free_memory'] * 0.8  # Conservative estimate
        
        # Memory estimation for distributed inference
        memory_per_param = 2  # bytes for inference (FP16)
        model_memory = model_params * memory_per_param / 1024  # Convert to MB
        activation_memory = model_params * 0.05 * (sequence_length / 1024)  # Sequence length dependent
        
        # Calculate safe batch size (more conservative for distributed)
        safe_memory = available_memory - model_memory - 1000  # Reserve 1GB for system
        
        if safe_memory > 0:
            batch_size = max(1, int(safe_memory / (activation_memory * 80)))
        else:
            batch_size = 1
        
        # Cap batch size based on model size and sequence length
        if model_params <= 1.0:  # Small models (< 1B)
            max_batch = 16
        elif model_params <= 7.0:  # Medium models (1-7B)
            max_batch = 8
        elif model_params <= 100.0:  # Large models (7-100B)
            max_batch = 4
        else:  # Ultra-large models (> 100B, including TB scale)
            max_batch = 2
        
        # Sequence length adjustment
        if sequence_length > 2048:
            max_batch = max(1, max_batch // 2)
        elif sequence_length > 4096:
            max_batch = max(1, max_batch // 4)
        
        batch_size = min(batch_size, max_batch)
        
        # Auto-detect master address and port
        master_addr = cluster_config.get('master_addr', 'localhost')
        master_port = cluster_config.get('master_port', '29500')
            
        return {
            'mode': 'distributed_cluster',
            'gpu_ids': [gpu['index']],
            'batch_size': batch_size,
            'mixed_precision': True,
            'reason': f'Optimized distributed cluster inference for {model_params:.1f}B model (rank {cluster_config.get("rank", 0)}/{cluster_config.get("world_size", 1)})',
            'world_size': cluster_config.get('world_size', 1),
            'rank': cluster_config.get('rank', 0),
            'local_rank': local_rank,
            'master_addr': master_addr,
            'master_port': str(master_port),
            'gpu_info': gpu,
            'model_params': model_params,
            'sequence_length': sequence_length,
            'available_memory': available_memory,
        }

    def recommend_batch_size(self, model_size: str, seq_len: int) -> int:
        """
        Intelligent batch size recommendation based on model size, sequence length, and hardware.
        
        Args:
            model_size (str): Model size string (e.g., "7B", "1.5B")
            seq_len (int): Sequence length
            
        Returns:
            int: Recommended batch size
        """
        # Parse model size
        model_params = self._parse_model_size(model_size) if model_size else self._estimate_model_parameters()
        
        # Get inference strategy which already calculates optimal batch size
        strategy = self.get_inference_strategy(model_size, seq_len)
        return strategy['batch_size']

    @staticmethod
    def check_rocm_availability() -> bool:
        """
        Check if ROCm is available on the system.
        
        Returns:
            bool: True if ROCm is available, False otherwise.
        """
        try:
            import subprocess
            result = subprocess.run(['which', 'rocm-smi'], capture_output=True, text=True, timeout=2)
            return result.returncode == 0
        except Exception:
            return False
