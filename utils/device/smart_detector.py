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

import time
import torch
import platform
from utils.log.core import PiscesLxCoreLog
from typing import Dict, Any, List, Optional
from utils.hooks.bus import get_global_hook_bus
from .cpu_detector import PiscesLxCoreDeviceCpuDetector
from .nvidia_detector import PiscesLxCoreDeviceNvidiaDetector
from utils.observability.service import PiscesLxCoreObservabilityService
from utils.observability.decorators import PiscesLxCoreDecorators as ObsDec

logger = PiscesLxCoreLog("PiscesLx.Utils.Device.SmartDetector")

class PiscesLxCoreDeviceSmartDetector:
    """
    An intelligent device detector that orchestrates the detection of NVIDIA, AMD GPUs, and CPUs.
    It also determines the optimal device strategy based on the detected hardware.
    """
    
    def __init__(self):
        """
        Initialize the device smart detector.
        Create instances of NVIDIA and CPU detectors, and initialize device information storage.
        """
        self.nvidia_detector = PiscesLxCoreDeviceNvidiaDetector()
        self.cpu_detector = PiscesLxCoreDeviceCpuDetector()
        
        self.all_gpu_info = []
        self.cpu_info = {}
        self.detection_summary = {}

        
    @ObsDec.auto_cached_logged(namespace="device.detect", ttl=60)
    def detect_all_devices(self, model_size: str = None) -> Dict[str, Any]:
        """
        Detect all available devices, including GPUs and CPU, and determine the optimal device strategy.

        Args:
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing GPU information, CPU information, 
                           detection summary, recommended strategy, and device priority.
        """
        # Log start of device detection process
        logger.info("Starting comprehensive device detection", event="DETECT", message="Starting comprehensive device detection")
        
        # Get global hook bus instance for emitting events
        bus = get_global_hook_bus()
        t0 = time.time()
        
        # Emit device detection start event
        try:
            bus.emit("device.detect.start", model_size=model_size)
        except Exception as e:
            logger.debug("emit device.detect.start failed", event="DEVICE", message="emit device.detect.start failed", error=str(e), error_class=type(e).__name__)
        
        # Detect NVIDIA GPUs using dedicated detector
        nvidia_gpus = self.nvidia_detector.detect()
        self.all_gpu_info.extend(nvidia_gpus)
        
        # Detect platform-specific devices (ROCm, DirectML, WSL)
        self._detect_platform_specific_devices()
        
        # Detect CPU using dedicated detector
        cpu_info = self.cpu_detector.detect()
        self.cpu_info = cpu_info
        
        # Calculate totals for different GPU types
        nvidia_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'nvidia'])
        amd_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'amd'])
        
        # Generate detection summary with key metrics
        self.detection_summary = {
            'nvidia_gpus': nvidia_count,
            'amd_gpus': amd_count,
            'total_gpus': len(self.all_gpu_info),
            'cpu_detected': bool(cpu_info),
            'cuda_available': torch.cuda.is_available(),
            'rocm_available': self._check_rocm_availability(),
            'directml_available': self._check_directml_availability(),
            'device_capabilities': self._get_device_capabilities(self.all_gpu_info, cpu_info)
        }
        
        # Log completion of device detection
        logger.info("Device detection completed", event="DETECT", message="Device detection completed", nvidia_gpus=nvidia_count, amd_gpus=amd_count, cpu_detected=True)
        
        # Determine optimal strategy based on detected hardware
        strategy = self._determine_optimal_strategy(model_size)
        
        # Prepare final result dictionary
        result = {
            'gpu_info': self.all_gpu_info,
            'cpu_info': self.cpu_info,
            'detection_summary': self.detection_summary,
            'recommended_strategy': strategy,
            'device_priority': self._get_device_priority()
        }

        # auto-generate device report (best-effort)
        try:
            obs = PiscesLxCoreObservabilityService.instance()
            obs.write_device_report(result)
        except Exception as e:
            logger.debug("write_device_report failed", error=str(e), error_class=type(e).__name__)
        # emit end event
        try:
            duration = int((time.time() - t0) * 1000)
            bus.emit(
                "device.detect.end",
                model_size=model_size,
                duration_ms=duration,
                summary=self.detection_summary,
                gpu_info=self.all_gpu_info,
            )
        except Exception as e:
            logger.debug("emit device.detect.end failed", event="DEVICE", message="emit device.detect.end failed", error=str(e), error_class=type(e).__name__)
        return result
    
    def _determine_optimal_strategy(self, model_size: str = None) -> Dict[str, Any]:
        """
        Determine the optimal device strategy based on the detected hardware.

        Args:
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary representing the optimal device strategy.
        """
        # Get total number of GPUs detected
        total_gpus = len(self.all_gpu_info)
        
        # Case 1: No GPUs available - CPU only
        if total_gpus == 0:
            return self._get_cpu_strategy(model_size)
        
        # Case 2: Only NVIDIA GPUs
        nvidia_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'nvidia'])
        
        if nvidia_count > 0:
            return self._get_nvidia_strategy(model_size)
        
        # Case 3: No GPUs - CPU only
        return self._get_cpu_strategy(model_size)
    
    def _get_cpu_strategy(self, model_size: str = None) -> Dict[str, Any]:
        """
        Get the CPU-only strategy.

        Args:
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary representing the CPU-only strategy.
        """
        # Get CPU strategy from CPU detector
        cpu_strategy = self.cpu_detector.get_recommended_strategy(model_size)
        
        # Return CPU-only strategy configuration
        return {
            'device_type': 'cpu',
            'strategy': cpu_strategy,
            'reason': 'No GPUs detected, falling back to CPU',
            'fallback': True,
            'performance_warning': cpu_strategy['mode'] == 'cpu_fallback'
        }
    
    def get_recommended_strategy(self, model_size: str = None) -> Dict[str, Any]:
        """
        Get the recommended device strategy based on the detected devices.

        Args:
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary representing the recommended device strategy.
        """
        # Perform device detection if not already done
        if not self.detection_summary:
            self.detect_all_devices(model_size)
        
        # Get counts of different GPU types
        nvidia_count = self.detection_summary['nvidia_gpus']
        amd_count = self.detection_summary['amd_gpus']
        
        # Check platform availability
        rocm_available = self._check_rocm_availability()
        directml_available = self._check_directml_availability()
        
        # Determine strategy based on available hardware
        if nvidia_count > 0:
            # Use NVIDIA GPUs with optimal settings
            return self._get_nvidia_strategy(model_size)
        elif amd_count > 0 and rocm_available:
            # Use AMD GPUs with ROCm
            return {
                'strategy': 'rocm',
                'gpu_ids': list(range(amd_count)),
                'mixed_precision': True,
                'memory_efficient': True,
                'batch_size_recommendation': 4 if amd_count == 1 else 8,
                'flash_attention': False,  # ROCm doesn't support Flash Attention yet
                'tensor_parallel': amd_count > 1,
                'pipeline_parallel': False,
                'data_parallel': amd_count > 1,
                'platform': 'rocm',
                'rocm_available': rocm_available
            }
        elif directml_available and len([gpu for gpu in self.all_gpu_info if gpu.get('platform') == 'directml']) > 0:
            # Use DirectML on Windows
            directml_gpus = [gpu for gpu in self.all_gpu_info if gpu.get('platform') == 'directml']
            return {
                'strategy': 'directml',
                'gpu_ids': [gpu['index'] for gpu in directml_gpus],
                'mixed_precision': True,
                'memory_efficient': True,
                'batch_size_recommendation': 2,
                'flash_attention': False,
                'tensor_parallel': False,
                'pipeline_parallel': False,
                'data_parallel': len(directml_gpus) > 1,
                'platform': 'directml',
                'directml_available': directml_available
            }
        else:
            # CPU-only fallback
            return {
                'strategy': 'cpu',
                'gpu_ids': [],
                'mixed_precision': False,
                'memory_efficient': True,
                'batch_size_recommendation': 1,
                'flash_attention': False,
                'tensor_parallel': False,
                'pipeline_parallel': False,
                'data_parallel': False,
                'platform': 'cpu'
            }

    def _get_device_capabilities(self, gpu_info: List[Dict], cpu_info: Dict) -> Dict[str, Any]:
        """
        Get a comprehensive assessment of device capabilities.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.
            cpu_info (Dict): A dictionary containing CPU information.

        Returns:
            Dict[str, Any]: A dictionary containing compute capability, memory capacity, 
                           parallel efficiency, and platform optimization.
        """
        # Evaluate different aspects of device capabilities
        capabilities = {
            'compute_capability': self._assess_compute_capability(gpu_info, cpu_info),
            'memory_capacity': self._assess_memory_capacity(gpu_info, cpu_info),
            'parallel_efficiency': self._assess_parallel_efficiency(gpu_info),
            'platform_optimization': self._assess_platform_optimization(gpu_info)
        }
        return capabilities
    
    def _assess_compute_capability(self, gpu_info: List[Dict], cpu_info: Dict) -> Dict[str, Any]:
        """
        Assess the compute capability of the detected devices.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.
            cpu_info (Dict): A dictionary containing CPU information.

        Returns:
            Dict[str, Any]: A dictionary containing the compute capability score, level, and details.
        """
        # Separate GPUs by vendor
        nvidia_gpus = [gpu for gpu in gpu_info if gpu.get('vendor') == 'nvidia']
        amd_gpus = [gpu for gpu in gpu_info if gpu.get('vendor') == 'amd']
        
        # Initialize capability score and details list
        capability_score = 0
        details = []
        
        # NVIDIA GPUs with Tensor Cores
        for gpu in nvidia_gpus:
            if gpu.get('tensor_cores', False):
                capability_score += 10
                details.append(f"{gpu['name']}: Tensor Cores")
            else:
                capability_score += 6
                details.append(f"{gpu['name']}: Standard CUDA")
        
        # AMD GPUs with ROCm
        for gpu in amd_gpus:
            capability_score += 4
            details.append(f"{gpu['name']}: ROCm support")
        
        # CPU capabilities
        if cpu_info:
            if cpu_info.get('has_avx512', False):
                capability_score += 2
                details.append("CPU: AVX-512 support")
            elif cpu_info.get('has_avx2', False):
                capability_score += 1
                details.append("CPU: AVX2 support")
        
        # Return capability assessment
        return {
            'score': capability_score,
            'level': 'high' if capability_score >= 10 else 'medium' if capability_score >= 5 else 'low',
            'details': details
        }
    
    def _assess_memory_capacity(self, gpu_info: List[Dict], cpu_info: Dict) -> Dict[str, Any]:
        """
        Assess the memory capacity of the detected devices.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.
            cpu_info (Dict): A dictionary containing CPU information.

        Returns:
            Dict[str, Any]: A dictionary containing the memory capacity score, level, 
                           total GPU memory, and details.
        """
        # Calculate total GPU memory across all GPUs
        total_gpu_memory = sum(gpu.get('total_memory', 0) for gpu in gpu_info)
        
        # Initialize memory score and details list
        memory_score = 0
        details = []
        
        # Assess memory capacity based on thresholds
        if total_gpu_memory >= 48000:  # 48GB+
            memory_score = 10
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Excellent")
        elif total_gpu_memory >= 24000:  # 24GB+
            memory_score = 8
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Very Good")
        elif total_gpu_memory >= 16000:  # 16GB+
            memory_score = 6
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Good")
        elif total_gpu_memory >= 8000:  # 8GB+
            memory_score = 4
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Adequate")
        elif total_gpu_memory > 0:
            memory_score = 2
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Limited")
        else:
            details.append("No GPU memory available")
        
        # Add per-GPU details
        for gpu in gpu_info:
            details.append(f"  {gpu['name']}: {gpu.get('total_memory', 0)//1024}GB")
        
        # Return memory capacity assessment
        return {
            'score': memory_score,
            'level': 'high' if memory_score >= 8 else 'medium' if memory_score >= 4 else 'low',
            'total_gpu_memory': total_gpu_memory,
            'details': details
        }
    
    def _assess_parallel_efficiency(self, gpu_info: List[Dict]) -> Dict[str, Any]:
        """
        Assess the parallel efficiency potential of the detected GPUs.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.

        Returns:
            Dict[str, Any]: A dictionary containing the parallel efficiency score, level, 
                           GPU count, and details.
        """
        # Get counts of different GPU types
        gpu_count = len(gpu_info)
        nvidia_count = len([gpu for gpu in gpu_info if gpu.get('vendor') == 'nvidia'])
        amd_count = len([gpu for gpu in gpu_info if gpu.get('vendor') == 'amd'])
        
        # Initialize efficiency score and details list
        efficiency_score = 0
        details = []
        
        # Multi-GPU setups
        if gpu_count >= 4:
            efficiency_score = 10
            details.append(f"{gpu_count} GPUs - Excellent scaling potential")
        elif gpu_count >= 2:
            efficiency_score = 8
            details.append(f"{gpu_count} GPUs - Good scaling potential")
        elif gpu_count == 1:
            efficiency_score = 4
            details.append("Single GPU - Standard performance")
        else:
            details.append("No GPU - CPU only")
        
        # Mixed vendor considerations
        if nvidia_count > 0 and amd_count > 0:
            efficiency_score -= 2
            details.append("Mixed vendor setup - Complexity penalty")
        
        # Return parallel efficiency assessment
        return {
            'score': efficiency_score,
            'level': 'high' if efficiency_score >= 8 else 'medium' if efficiency_score >= 4 else 'low',
            'gpu_count': gpu_count,
            'details': details
        }
    
    def _assess_platform_optimization(self, gpu_info: List[Dict]) -> Dict[str, Any]:
        """
        Assess the platform-specific optimizations of the detected devices.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.

        Returns:
            Dict[str, Any]: A dictionary containing the optimization score, level, 
                           available platforms, and details.
        """
        # Get unique platforms from GPU info
        platforms = set(gpu.get('platform', 'unknown') for gpu in gpu_info)
        
        # Initialize optimization score and details list
        optimization_score = 0
        details = []
        
        # Assess platform-specific optimizations
        if 'nvidia' in platforms:
            optimization_score += 4
            details.append("NVIDIA: CUDA, cuDNN, TensorRT optimizations")
        
        if 'rocm' in platforms:
            optimization_score += 3
            details.append("AMD: ROCm optimizations")
        
        if 'directml' in platforms:
            optimization_score += 2
            details.append("Microsoft: DirectML optimizations")
        
        if 'wsl' in platforms:
            optimization_score += 1
            details.append("WSL: Windows Subsystem for Linux support")
        
        # Return platform optimization assessment
        return {
            'score': optimization_score,
            'level': 'high' if optimization_score >= 6 else 'medium' if optimization_score >= 3 else 'low',
            'platforms': list(platforms),
            'details': details
        }
        
        # Get NVIDIA-specific strategy
        nvidia_strategy = self.nvidia_detector.get_recommended_strategy(model_size)
        
        # Enhance with PiscesL1 specific optimizations
        enhanced_strategy = {
            'device_type': 'cuda',
            'vendor': 'nvidia',
            'strategy': nvidia_strategy,
            'gpu_info': nvidia_gpus,
            'reason': f"NVIDIA GPU{'s' if len(nvidia_gpus) > 1 else ''} detected: {nvidia_strategy.get('reason', '')}",
            'cuda_optimizations': self._get_nvidia_optimizations(nvidia_gpus),
            'memory_efficient': self._should_use_memory_efficient(nvidia_gpus, model_size)
        }
        
        return enhanced_strategy
    

    
    def _get_nvidia_optimizations(self, nvidia_gpus: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get the NVIDIA-specific optimizations based on the detected NVIDIA GPUs.

        Args:
            nvidia_gpus (List[Dict[str, Any]]): A list of dictionaries containing NVIDIA GPU information.

        Returns:
            Dict[str, Any]: A dictionary containing NVIDIA-specific optimizations.
        """
        # Initialize default optimizations
        optimizations = {
            'tensor_cores': False,
            'mixed_precision': False,
            'flash_attention': False,
            'cudnn_benchmark': True
        }
        
        # Check for modern NVIDIA GPUs with Tensor Cores
        for gpu in nvidia_gpus:
            name = gpu.get('name', '').lower()
            compute_capability = gpu.get('compute_capability', '0.0')
            
            # Check for Tensor Core support (Volta and newer)
            try:
                major, minor = map(int, compute_capability.split('.'))
                if major >= 7:  # Volta (7.x) and newer have Tensor Cores
                    optimizations['tensor_cores'] = True
                    optimizations['mixed_precision'] = True
                    optimizations['flash_attention'] = True
            except Exception as e:
                logger.debug("tensor core detection failed", event="DEVICE", message="tensor core detection failed", error=str(e), error_class=type(e).__name__)
                
        return optimizations
    

    
    def _should_use_memory_efficient(self, gpus: List[Dict[str, Any]], model_size: str = None) -> bool:
        """
        Determine whether to use memory-efficient mode based on the model size and GPU memory.

        Args:
            gpus (List[Dict[str, Any]]): A list of dictionaries containing GPU information.
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            bool: True if memory-efficient mode should be used, False otherwise.
        """
        # Return False if no model size specified
        if not model_size:
            return False
            
        try:
            # Parse model size
            size_str = model_size.upper().replace('B', '')
            params_b = float(size_str)
            
            # Large models (>7B) should use memory efficient mode
            if params_b >= 7:
                return True
                
            # Check total GPU memory
            total_memory = sum(gpu.get('total_memory', 0) for gpu in gpus)
            total_memory_gb = total_memory / 1024  # Convert MiB to GiB
            
            # If total GPU memory is less than 24GB and model is >3B
            if total_memory_gb < 24 and params_b > 3:
                return True
                
        except Exception as e:
            logger.debug("WSL detection skipped", event="DEVICE", message="WSL detection skipped", error=type(e).__name__)
            
        return False
    
    def _detect_platform_specific_devices(self) -> None:
        """
        Detect platform-specific devices such as ROCm, DirectML, and WSL GPUs.
        """
        # Get current system platform
        system = platform.system().lower()
        
        # Detect platform-specific devices based on OS
        if system == "linux":
            self._detect_rocm_devices()
        elif system == "windows":
            self._detect_directml_devices()
            self._detect_wsl_devices()
    
    def _detect_rocm_devices(self) -> None:
        """
        Detect AMD GPUs via ROCm and add them to the GPU information list.
        """
        try:
            import subprocess
            import json
            
            # Try rocm-smi for AMD GPUs
            result = subprocess.run([
                'rocm-smi',
                '--showid',
                '--showproductname',
                '--showmeminfo',
                '--json'
            ], capture_output=True, text=True, timeout=5)
            
            # Process ROCm detection results
            if result.returncode == 0 and result.stdout.strip():
                rocm_data = json.loads(result.stdout)
                
                for gpu_id, gpu_data in rocm_data.items():
                    if 'GPU' in gpu_data:
                        gpu_info = gpu_data['GPU']
                        self.all_gpu_info.append({
                            'type': 'rocm',
                            'vendor': 'amd',
                            'index': int(gpu_id.replace('GPU', '')),
                            'name': gpu_info.get('Product Name', 'AMD GPU'),
                            'total_memory': self._parse_rocm_memory(gpu_info.get('VRAM Total Memory', '0')),
                            'free_memory': self._parse_rocm_memory(gpu_info.get('VRAM Available Memory', '0')),
                            'used_memory': 0,
                            'temperature': self._get_rocm_temperature(int(gpu_id.replace('GPU', ''))),
                            'utilization': self._get_rocm_utilization(int(gpu_id.replace('GPU', ''))),
                            'platform': 'rocm',
                            'driver_version': self._get_rocm_driver_version()
                        })
                        
                logger.info("ROCm detection", event="DEVICE", message="ROCm detection", amd_gpus=len([g for g in self.all_gpu_info if g.get('platform') == 'rocm']))

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, ImportError) as e:
            logger.debug("ROCm detection skipped", event="DEVICE", message="ROCm detection skipped", reason=type(e).__name__)
    
    def _detect_directml_devices(self) -> None:
        """
        Detect DirectML-compatible GPUs on Windows and add them to the GPU information list.
        """
        try:
            import torch_directml
            
            # Get number of DirectML devices
            device_count = torch_directml.device_count()
            for i in range(device_count):
                device_name = torch_directml.device_name(i)
                
                # Add DirectML device to GPU info list
                self.all_gpu_info.append({
                    'type': 'directml',
                    'vendor': 'microsoft',  # DirectML is Microsoft's API
                    'index': i,
                    'name': device_name,
                    'total_memory': 16 * 1024,  # Default 16GB estimation
                    'free_memory': 16 * 1024,
                    'used_memory': 0,
                    'temperature': self._get_directml_temperature(i),
                    'utilization': self._get_directml_utilization(i),
                    'platform': 'directml',
                    'driver_version': 'unknown'
                })
                
            logger.info("DirectML detection", event="DEVICE", message="DirectML detection", gpu_count=device_count)
            
        except ImportError:
            logger.debug("DirectML not available", event="DEVICE", message="DirectML not available")
        except Exception as e:
            logger.debug("DirectML detection failed", event="DEVICE", message="DirectML detection failed", error=str(e))
    
    def _detect_wsl_devices(self) -> None:
        """
        Detect GPUs in the WSL environment and add them to the GPU information list.
        """
        try:
            import os
            import subprocess
            
            # Check if running in WSL
            if os.path.exists('/proc/sys/fs/binfmt_misc/WSLInterop'):
                # Try nvidia-smi in WSL
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=5)
                
                # Process WSL GPU detection results
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                gpu_index = int(parts[0])
                                if not any(g['index'] == gpu_index and g.get('platform') == 'wsl' for g in self.all_gpu_info):
                                    self.all_gpu_info.append({
                                        'type': 'nvidia',
                                        'vendor': 'nvidia',
                                        'index': gpu_index,
                                        'name': parts[1],
                                        'total_memory': int(parts[2]),
                                        'free_memory': int(parts[2]),
                                        'used_memory': 0,
                                        'temperature': self._get_wsl_temperature(gpu_index),
                                        'utilization': self._get_wsl_utilization(gpu_index),
                                        'platform': 'wsl',
                                        'driver_version': 'unknown'
                                    })
                    
                    logger.info("WSL GPU detection completed", event="DEVICE", message="WSL GPU detection completed")
                    
        except Exception as e:
            logger.debug("WSL detection failed or skipped", event="DEVICE", message="WSL detection failed or skipped", error=str(e), error_class=type(e).__name__)
    
    def _parse_rocm_memory(self, memory_str: str) -> int:
        """
        Parse the ROCm memory string to an integer value in MiB.

        Args:
            memory_str (str): The memory string from ROCm, e.g., "16GB", "8192MB".

        Returns:
            int: The memory value in MiB. If parsing fails, return 16368 (default 16GB).
        """
        try:
            # Normalize memory string format
            memory_str = memory_str.strip().upper()
            if 'GB' in memory_str:
                return int(float(memory_str.replace('GB', '').strip()) * 1024)
            elif 'MB' in memory_str:
                return int(float(memory_str.replace('MB', '').strip()))
            else:
                return int(memory_str) // (1024*1024)
        except Exception as e:
            logger.debug("parse ROCm memory failed", event="DEVICE", message="parse ROCm memory failed", error=str(e), error_class=type(e).__name__)
            return 16368  # Default 16GB
    
    def _get_rocm_driver_version(self) -> str:
        """
        Get the ROCm driver version.

        Returns:
            str: The ROCm driver version. If retrieval fails, return 'unknown'.
        """
        try:
            import subprocess
            # Run rocm-smi to get driver version
            result = subprocess.run(['rocm-smi', '--showdriverversion'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except FileNotFoundError:
            logger.debug("get ROCM driver version failed", event="DEVICE", message="get ROCM driver version failed", error="rocm-smi not found")
        except subprocess.TimeoutExpired:
            logger.debug("get ROCM driver version failed", event="DEVICE", message="get ROCM driver version failed", error="timeout")
        except Exception as e:
            logger.debug("get ROCM driver version failed", event="DEVICE", message="get ROCM driver version failed", error=str(e), error_class=type(e).__name__)
        return 'unknown'

    def _check_rocm_availability(self) -> bool:
        """
        Check if ROCm is available on the system.

        Returns:
            bool: True if ROCm is available, False otherwise.
        """
        # Import locally to avoid circular import
        from .manager import PiscesLxCoreDeviceManager
        return PiscesLxCoreDeviceManager.check_rocm_availability()
    
    def _get_rocm_temperature(self, gpu_index: int) -> int:
        """Get ROCm GPU temperature using rocm-smi."""
        try:
            result = subprocess.run(['rocm-smi', '--showtemp', '--json'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for gpu_data in data.values():
                    if 'GPU' in gpu_data and str(gpu_index) in str(gpu_data.get('GPU', {})):
                        temp_info = gpu_data['GPU']
                        temp_str = temp_info.get('Temperature (Sensor memory) (C)', '0')
                        return int(temp_str.replace('C', '').strip()) if temp_str != 'N/A' else 0
        except Exception:
            pass
        return 0
        
    def _get_rocm_utilization(self, gpu_index: int) -> int:
        """Get ROCm GPU utilization using rocm-smi."""
        try:
            result = subprocess.run(['rocm-smi', '--showuse', '--json'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for gpu_data in data.values():
                    if 'GPU' in gpu_data and str(gpu_index) in str(gpu_data.get('GPU', {})):
                        util_info = gpu_data['GPU']
                        util_str = util_info.get('GPU use (%)', '0')
                        return int(util_str.replace('%', '').strip()) if util_str != 'N/A' else 0
        except Exception:
            pass
        return 0
        
    def _get_directml_temperature(self, device_index: int) -> int:
        """Get DirectML device temperature. DirectML does not provide temperature information."""
        # DirectML does not expose temperature sensors
        # Return -1 to indicate temperature is unavailable
        return -1
        
    def _get_directml_utilization(self, device_index: int) -> int:
        """Get DirectML device utilization. DirectML does not provide utilization information."""
        # DirectML does not expose utilization metrics
        # Return -1 to indicate utilization is unavailable
        return -1
        
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

    def _check_directml_availability(self) -> bool:
        """
        Check if DirectML is available on the system.

        Returns:
            bool: True if DirectML is available, False otherwise.
        """
        try:
            import torch_directml
            return torch_directml.is_available()
        except ImportError:
            return False
    
    def _get_device_priority(self) -> List[str]:
        """
        Get the device priority list for fallback scenarios.

        Returns:
            List[str]: A list of device vendors in priority order.
        """
        # Get counts of different GPU types
        nvidia_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'nvidia'])
        amd_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'amd'])
        
        # Initialize priority list
        priority = []
        
        # NVIDIA GPUs first (best performance)
        if nvidia_count > 0:
            priority.append('nvidia')
        
        # AMD GPUs second (good performance with ROCm)
        if amd_count > 0:
            priority.append('amd')
            
        priority.append('cpu')
        
        return priority
    
    def print_detection_summary(self) -> None:
        """
        Print a comprehensive detection summary of the detected devices and the recommended strategy.
        """
        # Print header section
        logger.info("=" * 60, event="DEVICE", message="=" * 60)
        logger.info("PiscesL1 Intelligent Device Detection Report", event="DEVICE", message="PiscesL1 Intelligent Device Detection Report")
        logger.info("=" * 60, event="DEVICE", message="=" * 60)
        
        # GPU Summary
        if self.all_gpu_info:
            logger.info("Detected GPUs", event="DEVICE", message="Detected GPUs", count=len(self.all_gpu_info))
            for gpu in self.all_gpu_info:
                vendor = gpu.get('vendor', 'unknown').upper()
                name = gpu.get('name', 'Unknown')
                memory_gb = gpu.get('total_memory', 0) / 1024
                free_gb = gpu.get('free_memory', 0) / 1024
                temp = gpu.get('temperature', 0)
                
                # Log GPU information
                logger.info("GPU Info", event="DEVICE", message="GPU Info", vendor=vendor, index=gpu.get('index', '?'), name=name)
                logger.info("GPU Memory", event="DEVICE", message="GPU Memory", total_gb=memory_gb, free_gb=free_gb)
                logger.info("GPU Status", event="DEVICE", message="GPU Status", temperature=temp, utilization=gpu.get('utilization', 0))
                
                # Log additional NVIDIA-specific info
                if vendor == 'nvidia':
                    compute_cap = gpu.get('compute_capability', 'unknown')
                    logger.info("Compute Capability", event="DEVICE", message="Compute Capability", capability=compute_cap)
        else:
            logger.error("No GPUs detected", event="DEVICE", message="No GPUs detected")
        
        # CPU Summary
        if self.cpu_info:
            basic_info = self.cpu_info.get('basic_info', {})
            arch_info = self.cpu_info.get('architecture', {})
            capability = self.cpu_info.get('is_capable', {})
            
            # Extract CPU information
            physical_cores = basic_info.get('physical_cores', 0)
            logical_cores = basic_info.get('logical_cores', 0)
            memory_gb = basic_info.get('memory', {}).get('total_gb', 0)
            brand = arch_info.get('brand_raw', 'Unknown')
            simd_level = arch_info.get('simd_level', 'basic')
            
            # Log CPU information
            logger.info("CPU Info", event="DEVICE", message="CPU Info", brand=brand)
            logger.info("CPU Cores", event="DEVICE", message="CPU Cores", physical_cores=physical_cores, logical_cores=logical_cores)
            logger.info("CPU Memory", event="DEVICE", message="CPU Memory", memory_gb=memory_gb)
            logger.info("CPU SIMD", event="DEVICE", message="CPU SIMD", simd_level=simd_level.upper())
            logger.info("CPU AI Capability", event="DEVICE", message="CPU AI Capability", level=capability.get('level', 'unknown'), score=capability.get('score', 0))
        
        # Strategy Recommendation
        if hasattr(self, 'current_strategy'):
            strategy = self.current_strategy
            logger.info("Recommended Strategy", event="DEVICE", message="Recommended Strategy", device_type=strategy.get('device_type', 'unknown'))
            logger.info("Strategy Reason", event="DEVICE", message="Strategy Reason", reason=strategy.get('reason', 'No recommendation'))
            
            # Log strategy warning if present
            if strategy.get('warning'):
                logger.warning("Strategy Warning", event="DEVICE", message="Strategy Warning", warning=strategy['warning'])
        
        # Print footer section
        logger.info("=" * 60, event="DEVICE", message="=" * 60)

    """
    An intelligent device detector that orchestrates the detection of NVIDIA, AMD GPUs, and CPUs.
    It also determines the optimal device strategy based on the detected hardware.
    """
    
    def __init__(self):
        """
        Initialize the device smart detector.
        Create instances of NVIDIA and CPU detectors, and initialize device information storage.
        """
        self.nvidia_detector = PiscesLxCoreDeviceNvidiaDetector()
        self.cpu_detector = PiscesLxCoreDeviceCpuDetector()
        
        self.all_gpu_info = []
        self.cpu_info = {}
        self.detection_summary = {}

        
    @ObsDec.auto_cached_logged(namespace="device.detect", ttl=60)
    def detect_all_devices(self, model_size: str = None) -> Dict[str, Any]:
        """
        Detect all available devices, including GPUs and CPU, and determine the optimal device strategy.

        Args:
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing GPU information, CPU information, 
                           detection summary, recommended strategy, and device priority.
        """
        logger.info("Starting comprehensive device detection", event="DETECT", message="Starting comprehensive device detection")
        bus = get_global_hook_bus()
        t0 = time.time()
        try:
            bus.emit("device.detect.start", model_size=model_size)
        except Exception as e:
            logger.debug("emit device.detect.start failed", event="DEVICE", message="emit device.detect.start failed", error=str(e), error_class=type(e).__name__)
        
        # Detect NVIDIA GPUs
        nvidia_gpus = self.nvidia_detector.detect()
        self.all_gpu_info.extend(nvidia_gpus)
        
        # Detect platform-specific devices (ROCm, DirectML, WSL)
        self._detect_platform_specific_devices()
        
        # Detect CPU
        cpu_info = self.cpu_detector.detect()
        self.cpu_info = cpu_info
        
        # Calculate totals
        nvidia_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'nvidia'])
        amd_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'amd'])
        
        # Generate detection summary
        self.detection_summary = {
            'nvidia_gpus': nvidia_count,
            'amd_gpus': amd_count,
            'total_gpus': len(self.all_gpu_info),
            'cpu_detected': bool(cpu_info),
            'cuda_available': torch.cuda.is_available(),
            'rocm_available': self._check_rocm_availability(),
            'directml_available': self._check_directml_availability(),
            'device_capabilities': self._get_device_capabilities(self.all_gpu_info, cpu_info)
        }
        
        logger.info("Device detection completed", event="DETECT", message="Device detection completed", nvidia_gpus=nvidia_count, amd_gpus=amd_count, cpu_detected=True)
        
        # Determine optimal strategy
        strategy = self._determine_optimal_strategy(model_size)
        
        result = {
            'gpu_info': self.all_gpu_info,
            'cpu_info': self.cpu_info,
            'detection_summary': self.detection_summary,
            'recommended_strategy': strategy,
            'device_priority': self._get_device_priority()
        }

        # auto-generate device report (best-effort)
        try:
            obs = PiscesLxCoreObservabilityService.instance()
            obs.write_device_report(result)
        except Exception as e:
            logger.debug("write_device_report failed", error=str(e), error_class=type(e).__name__)
        # emit end event
        try:
            duration = int((time.time() - t0) * 1000)
            bus.emit(
                "device.detect.end",
                model_size=model_size,
                duration_ms=duration,
                summary=self.detection_summary,
                gpu_info=self.all_gpu_info,
            )
        except Exception as e:
            logger.debug("emit device.detect.end failed", event="DEVICE", message="emit device.detect.end failed", error=str(e), error_class=type(e).__name__)
        return result
    
    def _determine_optimal_strategy(self, model_size: str = None) -> Dict[str, Any]:
        """
        Determine the optimal device strategy based on the detected hardware.

        Args:
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary representing the optimal device strategy.
        """
        total_gpus = len(self.all_gpu_info)
        
        # Case 1: No GPUs available - CPU only
        if total_gpus == 0:
            return self._get_cpu_strategy(model_size)
        
        # Case 2: Only NVIDIA GPUs
        nvidia_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'nvidia'])
        
        if nvidia_count > 0:
            return self._get_nvidia_strategy(model_size)
        
        # Case 3: No GPUs - CPU only
        return self._get_cpu_strategy(model_size)
    
    def _get_cpu_strategy(self, model_size: str = None) -> Dict[str, Any]:
        """
        Get the CPU-only strategy.

        Args:
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary representing the CPU-only strategy.
        """
        cpu_strategy = self.cpu_detector.get_recommended_strategy(model_size)
        
        return {
            'device_type': 'cpu',
            'strategy': cpu_strategy,
            'reason': 'No GPUs detected, falling back to CPU',
            'fallback': True,
            'performance_warning': cpu_strategy['mode'] == 'cpu_fallback'
        }
    
    def get_recommended_strategy(self, model_size: str = None) -> Dict[str, Any]:
        """
        Get the recommended device strategy based on the detected devices.

        Args:
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary representing the recommended device strategy.
        """
        if not self.detection_summary:
            self.detect_all_devices(model_size)
        
        nvidia_count = self.detection_summary['nvidia_gpus']
        amd_count = self.detection_summary['amd_gpus']
        
        # Check platform availability
        rocm_available = self._check_rocm_availability()
        directml_available = self._check_directml_availability()
        
        # Determine strategy based on available hardware
        if nvidia_count > 0:
            # Use NVIDIA GPUs with optimal settings
            return self._get_nvidia_strategy(model_size)
        elif amd_count > 0 and rocm_available:
            # Use AMD GPUs with ROCm
            return {
                'strategy': 'rocm',
                'gpu_ids': list(range(amd_count)),
                'mixed_precision': True,
                'memory_efficient': True,
                'batch_size_recommendation': 4 if amd_count == 1 else 8,
                'flash_attention': False,  # ROCm doesn't support Flash Attention yet
                'tensor_parallel': amd_count > 1,
                'pipeline_parallel': False,
                'data_parallel': amd_count > 1,
                'platform': 'rocm',
                'rocm_available': rocm_available
            }
        elif directml_available and len([gpu for gpu in self.all_gpu_info if gpu.get('platform') == 'directml']) > 0:
            # Use DirectML on Windows
            directml_gpus = [gpu for gpu in self.all_gpu_info if gpu.get('platform') == 'directml']
            return {
                'strategy': 'directml',
                'gpu_ids': [gpu['index'] for gpu in directml_gpus],
                'mixed_precision': True,
                'memory_efficient': True,
                'batch_size_recommendation': 2,
                'flash_attention': False,
                'tensor_parallel': False,
                'pipeline_parallel': False,
                'data_parallel': len(directml_gpus) > 1,
                'platform': 'directml',
                'directml_available': directml_available
            }
        else:
            # CPU-only fallback
            return {
                'strategy': 'cpu',
                'gpu_ids': [],
                'mixed_precision': False,
                'memory_efficient': True,
                'batch_size_recommendation': 1,
                'flash_attention': False,
                'tensor_parallel': False,
                'pipeline_parallel': False,
                'data_parallel': False,
                'platform': 'cpu'
            }

    def _get_device_capabilities(self, gpu_info: List[Dict], cpu_info: Dict) -> Dict[str, Any]:
        """
        Get a comprehensive assessment of device capabilities.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.
            cpu_info (Dict): A dictionary containing CPU information.

        Returns:
            Dict[str, Any]: A dictionary containing compute capability, memory capacity, 
                           parallel efficiency, and platform optimization.
        """
        capabilities = {
            'compute_capability': self._assess_compute_capability(gpu_info, cpu_info),
            'memory_capacity': self._assess_memory_capacity(gpu_info, cpu_info),
            'parallel_efficiency': self._assess_parallel_efficiency(gpu_info),
            'platform_optimization': self._assess_platform_optimization(gpu_info)
        }
        return capabilities
    
    def _assess_compute_capability(self, gpu_info: List[Dict], cpu_info: Dict) -> Dict[str, Any]:
        """
        Assess the compute capability of the detected devices.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.
            cpu_info (Dict): A dictionary containing CPU information.

        Returns:
            Dict[str, Any]: A dictionary containing the compute capability score, level, and details.
        """
        nvidia_gpus = [gpu for gpu in gpu_info if gpu.get('vendor') == 'nvidia']
        amd_gpus = [gpu for gpu in gpu_info if gpu.get('vendor') == 'amd']
        
        capability_score = 0
        details = []
        
        # NVIDIA GPUs with Tensor Cores
        for gpu in nvidia_gpus:
            if gpu.get('tensor_cores', False):
                capability_score += 10
                details.append(f"{gpu['name']}: Tensor Cores")
            else:
                capability_score += 6
                details.append(f"{gpu['name']}: Standard CUDA")
        
        # AMD GPUs with ROCm
        for gpu in amd_gpus:
            capability_score += 4
            details.append(f"{gpu['name']}: ROCm support")
        
        # CPU capabilities
        if cpu_info:
            if cpu_info.get('has_avx512', False):
                capability_score += 2
                details.append("CPU: AVX-512 support")
            elif cpu_info.get('has_avx2', False):
                capability_score += 1
                details.append("CPU: AVX2 support")
        
        return {
            'score': capability_score,
            'level': 'high' if capability_score >= 10 else 'medium' if capability_score >= 5 else 'low',
            'details': details
        }
    
    def _assess_memory_capacity(self, gpu_info: List[Dict], cpu_info: Dict) -> Dict[str, Any]:
        """
        Assess the memory capacity of the detected devices.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.
            cpu_info (Dict): A dictionary containing CPU information.

        Returns:
            Dict[str, Any]: A dictionary containing the memory capacity score, level, 
                           total GPU memory, and details.
        """
        total_gpu_memory = sum(gpu.get('total_memory', 0) for gpu in gpu_info)
        
        memory_score = 0
        details = []
        
        if total_gpu_memory >= 48000:  # 48GB+
            memory_score = 10
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Excellent")
        elif total_gpu_memory >= 24000:  # 24GB+
            memory_score = 8
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Very Good")
        elif total_gpu_memory >= 16000:  # 16GB+
            memory_score = 6
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Good")
        elif total_gpu_memory >= 8000:  # 8GB+
            memory_score = 4
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Adequate")
        elif total_gpu_memory > 0:
            memory_score = 2
            details.append(f"Total GPU memory: {total_gpu_memory//1024}GB - Limited")
        else:
            details.append("No GPU memory available")
        
        # Add per-GPU details
        for gpu in gpu_info:
            details.append(f"  {gpu['name']}: {gpu.get('total_memory', 0)//1024}GB")
        
        return {
            'score': memory_score,
            'level': 'high' if memory_score >= 8 else 'medium' if memory_score >= 4 else 'low',
            'total_gpu_memory': total_gpu_memory,
            'details': details
        }
    
    def _assess_parallel_efficiency(self, gpu_info: List[Dict]) -> Dict[str, Any]:
        """
        Assess the parallel efficiency potential of the detected GPUs.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.

        Returns:
            Dict[str, Any]: A dictionary containing the parallel efficiency score, level, 
                           GPU count, and details.
        """
        gpu_count = len(gpu_info)
        nvidia_count = len([gpu for gpu in gpu_info if gpu.get('vendor') == 'nvidia'])
        amd_count = len([gpu for gpu in gpu_info if gpu.get('vendor') == 'amd'])
        
        efficiency_score = 0
        details = []
        
        # Multi-GPU setups
        if gpu_count >= 4:
            efficiency_score = 10
            details.append(f"{gpu_count} GPUs - Excellent scaling potential")
        elif gpu_count >= 2:
            efficiency_score = 8
            details.append(f"{gpu_count} GPUs - Good scaling potential")
        elif gpu_count == 1:
            efficiency_score = 4
            details.append("Single GPU - Standard performance")
        else:
            details.append("No GPU - CPU only")
        
        # Mixed vendor considerations
        if nvidia_count > 0 and amd_count > 0:
            efficiency_score -= 2
            details.append("Mixed vendor setup - Complexity penalty")
        
        return {
            'score': efficiency_score,
            'level': 'high' if efficiency_score >= 8 else 'medium' if efficiency_score >= 4 else 'low',
            'gpu_count': gpu_count,
            'details': details
        }
    
    def _assess_platform_optimization(self, gpu_info: List[Dict]) -> Dict[str, Any]:
        """
        Assess the platform-specific optimizations of the detected devices.

        Args:
            gpu_info (List[Dict]): A list of dictionaries containing GPU information.

        Returns:
            Dict[str, Any]: A dictionary containing the optimization score, level, 
                           available platforms, and details.
        """
        platforms = set(gpu.get('platform', 'unknown') for gpu in gpu_info)
        
        optimization_score = 0
        details = []
        
        if 'nvidia' in platforms:
            optimization_score += 4
            details.append("NVIDIA: CUDA, cuDNN, TensorRT optimizations")
        
        if 'rocm' in platforms:
            optimization_score += 3
            details.append("AMD: ROCm optimizations")
        
        if 'directml' in platforms:
            optimization_score += 2
            details.append("Microsoft: DirectML optimizations")
        
        if 'wsl' in platforms:
            optimization_score += 1
            details.append("WSL: Windows Subsystem for Linux support")
        
        return {
            'score': optimization_score,
            'level': 'high' if optimization_score >= 6 else 'medium' if optimization_score >= 3 else 'low',
            'platforms': list(platforms),
            'details': details
        }
        
        # Get NVIDIA-specific strategy
        nvidia_strategy = self.nvidia_detector.get_recommended_strategy(model_size)
        
        # Enhance with PiscesL1 specific optimizations
        enhanced_strategy = {
            'device_type': 'cuda',
            'vendor': 'nvidia',
            'strategy': nvidia_strategy,
            'gpu_info': nvidia_gpus,
            'reason': f"NVIDIA GPU{'s' if len(nvidia_gpus) > 1 else ''} detected: {nvidia_strategy.get('reason', '')}",
            'cuda_optimizations': self._get_nvidia_optimizations(nvidia_gpus),
            'memory_efficient': self._should_use_memory_efficient(nvidia_gpus, model_size)
        }
        
        return enhanced_strategy
    

    
    def _get_nvidia_optimizations(self, nvidia_gpus: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get the NVIDIA-specific optimizations based on the detected NVIDIA GPUs.

        Args:
            nvidia_gpus (List[Dict[str, Any]]): A list of dictionaries containing NVIDIA GPU information.

        Returns:
            Dict[str, Any]: A dictionary containing NVIDIA-specific optimizations.
        """
        optimizations = {
            'tensor_cores': False,
            'mixed_precision': False,
            'flash_attention': False,
            'cudnn_benchmark': True
        }
        
        # Check for modern NVIDIA GPUs with Tensor Cores
        for gpu in nvidia_gpus:
            name = gpu.get('name', '').lower()
            compute_capability = gpu.get('compute_capability', '0.0')
            
            # Check for Tensor Core support (Volta and newer)
            try:
                major, minor = map(int, compute_capability.split('.'))
                if major >= 7:  # Volta (7.x) and newer have Tensor Cores
                    optimizations['tensor_cores'] = True
                    optimizations['mixed_precision'] = True
                    optimizations['flash_attention'] = True
            except Exception as e:
                logger.debug("tensor core detection failed", event="DEVICE", message="tensor core detection failed", error=str(e), error_class=type(e).__name__)
                
        return optimizations
    

    
    def _should_use_memory_efficient(self, gpus: List[Dict[str, Any]], model_size: str = None) -> bool:
        """
        Determine whether to use memory-efficient mode based on the model size and GPU memory.

        Args:
            gpus (List[Dict[str, Any]]): A list of dictionaries containing GPU information.
            model_size (str, optional): The size of the model, e.g., "7B", "13B". Defaults to None.

        Returns:
            bool: True if memory-efficient mode should be used, False otherwise.
        """
        if not model_size:
            return False
            
        try:
            # Parse model size
            size_str = model_size.upper().replace('B', '')
            params_b = float(size_str)
            
            # Large models (>7B) should use memory efficient mode
            if params_b >= 7:
                return True
                
            # Check total GPU memory
            total_memory = sum(gpu.get('total_memory', 0) for gpu in gpus)
            total_memory_gb = total_memory / 1024  # Convert MiB to GiB
            
            # If total GPU memory is less than 24GB and model is >3B
            if total_memory_gb < 24 and params_b > 3:
                return True
                
        except Exception as e:
            logger.debug("WSL detection skipped", event="DEVICE", message="WSL detection skipped", error=type(e).__name__)
            
        return False
    
    def _detect_platform_specific_devices(self) -> None:
        """
        Detect platform-specific devices such as ROCm, DirectML, and WSL GPUs.
        """
        system = platform.system().lower()
        
        if system == "linux":
            self._detect_rocm_devices()
        elif system == "windows":
            self._detect_directml_devices()
            self._detect_wsl_devices()
    
    def _detect_rocm_devices(self) -> None:
        """
        Detect AMD GPUs via ROCm and add them to the GPU information list.
        """
        try:
            import subprocess
            import json
            
            # Try rocm-smi for AMD GPUs
            result = subprocess.run([
                'rocm-smi',
                '--showid',
                '--showproductname',
                '--showmeminfo',
                '--json'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                rocm_data = json.loads(result.stdout)
                
                for gpu_id, gpu_data in rocm_data.items():
                    if 'GPU' in gpu_data:
                        gpu_info = gpu_data['GPU']
                        self.all_gpu_info.append({
                            'type': 'rocm',
                            'vendor': 'amd',
                            'index': int(gpu_id.replace('GPU', '')),
                            'name': gpu_info.get('Product Name', 'AMD GPU'),
                            'total_memory': self._parse_rocm_memory(gpu_info.get('VRAM Total Memory', '0')),
                            'free_memory': self._parse_rocm_memory(gpu_info.get('VRAM Available Memory', '0')),
                            'used_memory': 0,
                            'temperature': 0,
                            'utilization': 0,
                            'platform': 'rocm',
                            'driver_version': self._get_rocm_driver_version()
                        })
                        
                logger.info("ROCm detection", event="DEVICE", message="ROCm detection", amd_gpus=len([g for g in self.all_gpu_info if g.get('platform') == 'rocm']))

        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError, ImportError) as e:
            logger.debug("ROCm detection skipped", event="DEVICE", message="ROCm detection skipped", reason=type(e).__name__)
    
    def _detect_directml_devices(self) -> None:
        """
        Detect DirectML-compatible GPUs on Windows and add them to the GPU information list.
        """
        try:
            import torch_directml
            
            device_count = torch_directml.device_count()
            for i in range(device_count):
                device_name = torch_directml.device_name(i)
                
                self.all_gpu_info.append({
                    'type': 'directml',
                    'vendor': 'microsoft',  # DirectML is Microsoft's API
                    'index': i,
                    'name': device_name,
                    'total_memory': 16 * 1024,  # Default 16GB estimation
                    'free_memory': 16 * 1024,
                    'used_memory': 0,
                    'temperature': 0,
                    'utilization': 0,
                    'platform': 'directml',
                    'driver_version': 'unknown'
                })
                
            logger.info("DirectML detection", event="DEVICE", message="DirectML detection", gpu_count=device_count)
            
        except ImportError:
            logger.debug("DirectML not available", event="DEVICE", message="DirectML not available")
        except Exception as e:
            logger.debug("DirectML detection failed", event="DEVICE", message="DirectML detection failed", error=str(e))
    
    def _detect_wsl_devices(self) -> None:
        """
        Detect GPUs in the WSL environment and add them to the GPU information list.
        """
        try:
            import os
            import subprocess
            
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
                                gpu_index = int(parts[0])
                                if not any(g['index'] == gpu_index and g.get('platform') == 'wsl' for g in self.all_gpu_info):
                                    self.all_gpu_info.append({
                                        'type': 'nvidia',
                                        'vendor': 'nvidia',
                                        'index': gpu_index,
                                        'name': parts[1],
                                        'total_memory': int(parts[2]),
                                        'free_memory': int(parts[2]),
                                        'used_memory': 0,
                                        'temperature': 0,
                                        'utilization': 0,
                                        'platform': 'wsl',
                                        'driver_version': 'unknown'
                                    })
                    
                    logger.info("WSL GPU detection completed", event="DEVICE", message="WSL GPU detection completed")
                    
        except Exception as e:
            logger.debug("WSL detection failed or skipped", event="DEVICE", message="WSL detection failed or skipped", error=str(e), error_class=type(e).__name__)
    
    def _parse_rocm_memory(self, memory_str: str) -> int:
        """
        Parse the ROCm memory string to an integer value in MiB.

        Args:
            memory_str (str): The memory string from ROCm, e.g., "16GB", "8192MB".

        Returns:
            int: The memory value in MiB. If parsing fails, return 16368 (default 16GB).
        """
        try:
            memory_str = memory_str.strip().upper()
            if 'GB' in memory_str:
                return int(float(memory_str.replace('GB', '').strip()) * 1024)
            elif 'MB' in memory_str:
                return int(float(memory_str.replace('MB', '').strip()))
            else:
                return int(memory_str) // (1024*1024)
        except Exception as e:
            logger.debug("parse ROCm memory failed", event="DEVICE", message="parse ROCm memory failed", error=str(e), error_class=type(e).__name__)
            return 16368  # Default 16GB
    
    def _get_rocm_driver_version(self) -> str:
        """
        Get the ROCm driver version.

        Returns:
            str: The ROCm driver version. If retrieval fails, return 'unknown'.
        """
        try:
            import subprocess
            result = subprocess.run(['rocm-smi', '--showdriverversion'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except FileNotFoundError:
            logger.debug("get ROCM driver version failed", event="DEVICE", message="get ROCM driver version failed", error="rocm-smi not found")
        except subprocess.TimeoutExpired:
            logger.debug("get ROCM driver version failed", event="DEVICE", message="get ROCM driver version failed", error="timeout")
        except Exception as e:
            logger.debug("get ROCM driver version failed", event="DEVICE", message="get ROCM driver version failed", error=str(e), error_class=type(e).__name__)
        return 'unknown'

    # Removed duplicate _check_rocm_availability method - use unified implementation
    
    def _check_directml_availability(self) -> bool:
        """
        Check if DirectML is available on the system.

        Returns:
            bool: True if DirectML is available, False otherwise.
        """
        try:
            import torch_directml
            return torch_directml.is_available()
        except ImportError:
            return False
    
    def _get_device_priority(self) -> List[str]:
        """
        Get the device priority list for fallback scenarios.

        Returns:
            List[str]: A list of device vendors in priority order.
        """
        nvidia_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'nvidia'])
        amd_count = len([gpu for gpu in self.all_gpu_info if gpu.get('vendor') == 'amd'])
        
        priority = []
        
        # NVIDIA GPUs first (best performance)
        if nvidia_count > 0:
            priority.append('nvidia')
        
        # AMD GPUs second (good performance with ROCm)
        if amd_count > 0:
            priority.append('amd')
            
        priority.append('cpu')
        
        return priority
    
    def print_detection_summary(self) -> None:
        """
        Print a comprehensive detection summary of the detected devices and the recommended strategy.
        """
        logger.info("=" * 60, event="DEVICE", message="=" * 60)
        logger.info("PiscesL1 Intelligent Device Detection Report", event="DEVICE", message="PiscesL1 Intelligent Device Detection Report")
        logger.info("=" * 60, event="DEVICE", message="=" * 60)
        
        # GPU Summary
        if self.all_gpu_info:
            logger.info("Detected GPUs", event="DEVICE", message="Detected GPUs", count=len(self.all_gpu_info))
            for gpu in self.all_gpu_info:
                vendor = gpu.get('vendor', 'unknown').upper()
                name = gpu.get('name', 'Unknown')
                memory_gb = gpu.get('total_memory', 0) / 1024
                free_gb = gpu.get('free_memory', 0) / 1024
                temp = gpu.get('temperature', 0)
                
                logger.info("GPU Info", event="DEVICE", message="GPU Info", vendor=vendor, index=gpu.get('index', '?'), name=name)
                logger.info("GPU Memory", event="DEVICE", message="GPU Memory", total_gb=memory_gb, free_gb=free_gb)
                logger.info("GPU Status", event="DEVICE", message="GPU Status", temperature=temp, utilization=gpu.get('utilization', 0))
                
                if vendor == 'nvidia':
                    compute_cap = gpu.get('compute_capability', 'unknown')
                    logger.info("Compute Capability", event="DEVICE", message="Compute Capability", capability=compute_cap)
        else:
            logger.error("No GPUs detected", event="DEVICE", message="No GPUs detected")
        
        # CPU Summary
        if self.cpu_info:
            basic_info = self.cpu_info.get('basic_info', {})
            arch_info = self.cpu_info.get('architecture', {})
            capability = self.cpu_info.get('is_capable', {})
            
            physical_cores = basic_info.get('physical_cores', 0)
            logical_cores = basic_info.get('logical_cores', 0)
            memory_gb = basic_info.get('memory', {}).get('total_gb', 0)
            brand = arch_info.get('brand_raw', 'Unknown')
            simd_level = arch_info.get('simd_level', 'basic')
            
            logger.info("CPU Info", event="DEVICE", message="CPU Info", brand=brand)
            logger.info("CPU Cores", event="DEVICE", message="CPU Cores", physical_cores=physical_cores, logical_cores=logical_cores)
            logger.info("CPU Memory", event="DEVICE", message="CPU Memory", memory_gb=memory_gb)
            logger.info("CPU SIMD", event="DEVICE", message="CPU SIMD", simd_level=simd_level.upper())
            logger.info("CPU AI Capability", event="DEVICE", message="CPU AI Capability", level=capability.get('level', 'unknown'), score=capability.get('score', 0))
        
        # Strategy Recommendation
        if hasattr(self, 'current_strategy'):
            strategy = self.current_strategy
            logger.info("Recommended Strategy", event="DEVICE", message="Recommended Strategy", device_type=strategy.get('device_type', 'unknown'))
            logger.info("Strategy Reason", event="DEVICE", message="Strategy Reason", reason=strategy.get('reason', 'No recommendation'))
            
            if strategy.get('warning'):
                logger.warning("Strategy Warning", event="DEVICE", message="Strategy Warning", warning=strategy['warning'])
        
        logger.info("=" * 60, event="DEVICE", message="=" * 60)
