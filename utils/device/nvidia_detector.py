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

import torch
import subprocess
from utils.log.core import PiscesLxCoreLog
from typing import List, Dict, Any

logger = PiscesLxCoreLog("PiscesLx.Utils.Device.NvidiaDetector")

class PiscesLxCoreDeviceNvidiaDetector:
    """
    A detector class for NVIDIA GPUs that supports advanced CUDA capabilities detection.

    This class provides functionality to detect NVIDIA GPUs, gather comprehensive information about them,
    including hardware details and CUDA capabilities, and recommend appropriate strategies based on the GPU information.
    """
    
    def __init__(self):
        """
        Initialize the NVIDIA GPU detector.

        Sets up initial state including GPU information list, CUDA availability flag,
        CUDA version, and structured logger instance.
        """
        self.gpu_info = []  # List to store information about detected GPUs
        self.cuda_available = torch.cuda.is_available()  # Flag indicating whether CUDA is available
        self.cuda_version = torch.version.cuda if self.cuda_available else None  # CUDA version if available, otherwise None

        
    def detect(self) -> List[Dict[str, Any]]:
        """
        Detect NVIDIA GPUs and collect comprehensive information using the `nvidia-smi` command.

        This method attempts to use nvidia-smi to gather detailed GPU information. If that fails,
        it falls back to PyTorch's CUDA API for basic detection.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about each detected NVIDIA GPU.
                                Returns an empty list if CUDA is not available.
        """
        if not self.cuda_available:
            logger.warning("CUDA not available, skipping NVIDIA GPU detection")
            return self.gpu_info
            
        try:
            # Use `nvidia-smi` to get comprehensive GPU information
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw,power.limit,clocks.mem,memory.bus_width', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 11:
                        self.gpu_info.append({
                            'index': int(parts[0]),
                            'name': parts[1].strip(),
                            'memory_total': int(parts[2]),
                            'memory_used': int(parts[3]),
                            'memory_free': int(parts[4]),
                            'temperature': int(parts[5]) if parts[5] != 'N/A' else self._get_gpu_temperature_fallback(int(parts[0])),
                            'gpu_utilization': int(parts[6]) if parts[6] != 'N/A' else 0,
                            'power_draw': float(parts[7]) if parts[7] != 'N/A' else 0.0,
                            'power_limit': float(parts[8]) if parts[8] != 'N/A' else 0.0,
                            'memory_clock': int(parts[9]) if parts[9] != 'N/A' else 0,
                            'memory_bus_width': int(parts[10]) if parts[10] != 'N/A' else 0,
                            'cuda_version': self.cuda_version,
                            'driver_version': self._get_driver_version()
                        })
                        
                logger.info("NVIDIA GPU detection", {"count": len(self.gpu_info), "method": "nvidia-smi"})
            else:
                # Fallback if nvidia-smi returned no data
                logger.debug("nvidia-smi returned no data, falling back to torch.cuda", {"returncode": result.returncode})
                self._detect_via_torch_cuda()
                
        except Exception as e:
            logger.error("nvidia-smi detection failed", error=str(e), error_class=type(e).__name__)
            # Fallback to torch-based detection
            self._detect_via_torch_cuda()
            
    def _detect_via_torch_cuda(self) -> None:
        """
        Fallback method to detect NVIDIA GPUs using the PyTorch CUDA API.

        This method is used when the `nvidia-smi` command is not available or fails.
        It provides basic GPU information through the PyTorch CUDA API.
        """
        try:
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    
                    self.gpu_info.append({
                        'index': i,
                        'name': props.name,
                        'memory_total': props.total_memory // 1024 // 1024,  # Convert to MB
                        'memory_used': 0,  # Not available via PyTorch
                        'memory_free': 0,  # Not available via PyTorch
                        'temperature': self._get_gpu_temperature_fallback(i),  # Try to get via alternative methods
                        'gpu_utilization': 0,  # Not available via PyTorch
                        'power_draw': 0.0,  # Not available via PyTorch
                        'power_limit': 0.0,  # Not available via PyTorch
                        'memory_clock': 0,  # Not available via PyTorch
                        'memory_bus_width': 0,  # Not available via PyTorch
                        'cuda_version': self.cuda_version,
                        'driver_version': 'unknown',
                        'multi_processor_count': props.multi_processor_count,
                        'major': props.major,
                        'minor': props.minor
                    })
                    
                except Exception as e:
                    logger.error("Failed to detect CUDA device", device_index=i, error=str(e))
                    continue
                    
            logger.info("NVIDIA GPU detection", {"count": len(self.gpu_info), "method": "pytorch-cuda"})
            
        except Exception as e:
            logger.error("PyTorch CUDA detection failed", error=str(e), error_class=type(e).__name__)
            
    def _add_cuda_compute_info(self) -> None:
        """
        Add CUDA compute capability information to each GPU in the `gpu_info` list.

        This method enriches the GPU information with compute capability and memory bandwidth details.
        """
        for device_idx, gpu in enumerate(self.gpu_info):
            try:
                # Get compute capability
                major = gpu.get('major', 0)
                minor = gpu.get('minor', 0)
                
                if major > 0 and minor >= 0:
                    gpu['compute_capability'] = f"{major}.{minor}"
                    # GPUs with Volta architecture (7.0) and newer are considered modern
                    gpu['is_modern'] = major >= 7
                    
                    # Calculate memory bandwidth (simplified calculation)
                    memory_bus_width = gpu.get('memory_bus_width', 0)
                    memory_clock_rate = gpu.get('memory_clock', 0)
                    
                    if memory_bus_width > 0 and memory_clock_rate > 0:
                        # Memory bandwidth = bus_width * memory_clock * 2 (DDR) / 8 (bits to bytes) / 1e9 (to GB/s)
                        bandwidth_gbps = (memory_bus_width * memory_clock_rate * 2) / (8 * 1e9)
                        gpu['memory_bandwidth_gbps'] = round(bandwidth_gbps, 1)
                    else:
                        gpu['memory_bandwidth_gbps'] = 0.0
                        
            except Exception as e:
                logger.error("Failed to get compute info for GPU", gpu_index=device_idx, error=str(e))
                gpu['compute_capability'] = 'unknown'
                gpu['is_modern'] = False
                gpu['memory_bandwidth_gbps'] = 0.0
                
    def _get_gpu_temperature_fallback(self, gpu_index: int) -> int:
        """Get GPU temperature using fallback methods when nvidia-smi returns N/A."""
        try:
            # Try nvidia-ml-py first
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                return temp
            except (ImportError, pynvml.NVMLError):
                pass
            
            # Try direct nvidia-smi call for specific GPU
            result = subprocess.run([
                'nvidia-smi', f'--id={gpu_index}', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                temp_str = result.stdout.strip().split('\n')[0]
                if temp_str.isdigit():
                    return int(temp_str)
                    
        except Exception:
            pass
            
        # Final fallback: estimate based on GPU model and utilization
        try:
            # Try to get GPU name and make educated guess
            result = subprocess.run([
                'nvidia-smi', f'--id={gpu_index}', '--query-gpu=name', '--format=csv,noheader'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0].lower()
                # Conservative temperature estimates based on GPU generation
                if 'h100' in gpu_name or 'a100' in gpu_name:
                    return 45  # Modern data center GPUs run cooler
                elif 'v100' in gpu_name or 'rtx' in gpu_name:
                    return 55  # Previous generation
                elif 'p100' in gpu_name or 'gtx' in gpu_name:
                    return 65  # Older generation
                else:
                    return 50  # Default for unknown models
        except Exception:
            pass
            
        return 45  # Conservative default temperature

    def _get_driver_version(self) -> str:
        """
        Get the NVIDIA driver version using nvidia-smi.

        Returns:
            str: The NVIDIA driver version if the command execution is successful, otherwise 'unknown'.
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
                check=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except subprocess.CalledProcessError as e:
            logger.debug("Failed to get NVIDIA driver version", {"error": str(e), "error_class": type(e).__name__})
        except Exception as e:
            logger.debug("Failed to get NVIDIA driver version", {"error": str(e), "error_class": type(e).__name__})
        return 'unknown'
        
    def get_recommended_strategy(self, model_size: str = None) -> Dict[str, Any]:
        """
        Get the recommended strategy based on the capabilities of the detected NVIDIA GPUs.

        This method analyzes the detected GPUs and provides recommendations for:
        - Execution mode (single GPU, multi-GPU, mixed precision)
        - GPU selection and configuration
        - Performance optimization strategies

        Args:
            model_size (str, optional): Size of the model. Currently not used. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the recommended mode, GPU IDs, reason,
                            and other relevant information.
        """
        if not self.gpu_info:
            return {'mode': 'cpu', 'reason': 'No NVIDIA GPU detected'}
            
        # Filter modern GPUs (Volta or newer)
        modern_gpus = [gpu for gpu in self.gpu_info if gpu.get('is_modern', False)]
        
        if len(modern_gpus) == len(self.gpu_info) and len(self.gpu_info) > 1:
            # All GPUs are modern, recommend a multi-GPU strategy
            return {
                'mode': 'multi_gpu',
                'gpu_ids': [gpu['index'] for gpu in self.gpu_info],
                'reason': f'All {len(self.gpu_info)} NVIDIA GPUs are modern (Volta+)',
                'mixed_precision': True,
                'tensor_parallel': True
            }
        elif len(self.gpu_info) == 1:
            # Single GPU case
            gpu = self.gpu_info[0]
            if gpu.get('is_modern', False):
                return {
                    'mode': 'single_gpu',
                    'gpu_ids': [gpu['index']],
                    'reason': f'Single modern NVIDIA GPU: {gpu["name"]}',
                    'mixed_precision': True
                }
            else:
                return {
                    'mode': 'single_gpu_conservative',
                    'gpu_ids': [gpu['index']],
                    'reason': f'Legacy NVIDIA GPU: {gpu["name"]}',
                    'mixed_precision': False
                }
        else:
            # Mixed GPUs case
            return {
                'mode': 'multi_gpu_mixed',
                'gpu_ids': [gpu['index'] for gpu in self.gpu_info],
                'reason': f'Mixed NVIDIA GPUs ({len(modern_gpus)} modern, {len(self.gpu_info)-len(modern_gpus)} legacy)',
                'mixed_precision': len(modern_gpus) > 0,
                'primary_gpu': modern_gpus[0]['index'] if modern_gpus else self.gpu_info[0]['index']
            }
