#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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

import os
import json
import torch
import subprocess
from typing import Dict, List, Tuple, Optional

class GPUManager:
    """
    Intelligent GPU Manager - System-level hardware detection and strategy selection.
    The selection is automatically made based on system hardware status without developer intervention.
    """
    
    def __init__(self):
        self.gpu_info = []
        self.strategy = {}
        self._detect_hardware()
        self._determine_strategy()
    
    def _detect_hardware(self):
        """Perform system-level hardware detection."""
        if not torch.cuda.is_available():
            self.gpu_info = []
            return
            
        try:
            # Use nvidia-smi to get GPU information
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            self.gpu_info.append({
                                'index': int(parts[0]),
                                'name': parts[1],
                                'total_memory': int(parts[2]),
                                'free_memory': int(parts[3]),
                                'used_memory': int(parts[4]),
                                'temperature': int(parts[5]) if parts[5] != 'N/A' else 0,
                                'utilization': int(parts[6]) if parts[6] != 'N/A' else 0
                            })
        except Exception:
            # Fallback: Use torch to get basic information
            for i in range(torch.cuda.device_count()):
                try:
                    total = torch.cuda.get_device_properties(i).total_memory // 1024**2
                    self.gpu_info.append({
                        'index': i,
                        'name': torch.cuda.get_device_name(i),
                        'total_memory': total,
                        'free_memory': total,  # Rough estimate
                        'used_memory': 0,
                        'temperature': 0,
                        'utilization': 0
                    })
                except:
                    continue
    
    def _determine_strategy(self):
        """System determines the training strategy without developer control."""
        if not self.gpu_info:
            # Raise an error when no GPU is detected, CPU training is not allowed
            raise RuntimeError("❌tNo GPU detected. Pisces L1 requires a GPU for training.")
        
        available_gpus = []
        for gpu in self.gpu_info:
            # Calculate available memory (conservative estimate, reserve 20%)
            available_memory = gpu['free_memory'] * 0.8
            if available_memory > 1000:  # At least 1GB available
                available_gpus.append({
                    **gpu,
                    'available_memory': available_memory
                })
        
        if not available_gpus:
            raise RuntimeError("❌\tInsufficient memory on all GPUs. Training cannot proceed.")
        
        # System-level decision logic
        total_gpus = len(available_gpus)
        total_memory = sum(g['available_memory'] for g in available_gpus)
        
        if total_gpus == 1:
            gpu = available_gpus[0]
            memory_gb = gpu['available_memory'] / 1024
            
            if memory_gb >= 16:
                batch_size = 32
            elif memory_gb >= 8:
                batch_size = 16
            elif memory_gb >= 4:
                batch_size = 8
            elif memory_gb >= 2:
                batch_size = 4
            else:
                batch_size = 2
            
            self.strategy = {
                'mode': 'single_gpu',
                'gpu_ids': [gpu['index']],
                'batch_size': batch_size,
                'mixed_precision': True,
                'reason': f'Single GPU {memory_gb:.1f}GB'
            }
            
        else:
            # Multiple GPUs - System automatically decides whether to use distributed training
            memory_gb = total_memory / 1024
            
            # Decide whether to use distributed training based on environment variables
            use_distributed = (
                os.environ.get('RANK') is not None or
                os.environ.get('LOCAL_RANK') is not None
            )
            
            if use_distributed:
                # Distributed training
                per_gpu_memory = min(g['available_memory'] for g in available_gpus) / 1024
                if per_gpu_memory >= 16:
                    batch_size = 16  # Per GPU
                elif per_gpu_memory >= 8:
                    batch_size = 8
                elif per_gpu_memory >= 4:
                    batch_size = 4
                else:
                    batch_size = 2
                
                self.strategy = {
                    'mode': 'distributed',
                    'gpu_ids': [g['index'] for g in available_gpus],
                    'batch_size': batch_size,
                    'mixed_precision': True,
                    'reason': f'Distributed training with {total_gpus} GPUs'
                }
            else:
                # DataParallel
                if memory_gb >= 32:
                    batch_size = 64
                elif memory_gb >= 16:
                    batch_size = 32
                elif memory_gb >= 8:
                    batch_size = 16
                else:
                    batch_size = 8
                
                self.strategy = {
                    'mode': 'multi_gpu',
                    'gpu_ids': [g['index'] for g in available_gpus],
                    'batch_size': batch_size,
                    'mixed_precision': True,
                    'reason': f'DataParallel with {total_gpus} GPUs'
                }
    
    def print_summary(self):
        """Print the system hardware status."""
        print("=" * 50)
        print("🟧\tPisces L1 Hardware Detection Report")
        print("=" * 50)
        
        if not self.gpu_info:
            print("❌tNo GPU detected")
            return
        
        print(f"🟧\tDetected {len(self.gpu_info)} GPUs:")
        for gpu in self.gpu_info:
            memory_gb = gpu['total_memory'] / 1024
            free_gb = gpu['free_memory'] / 1024
            print(f"🟧\tGPU {gpu['index']}: {gpu['name']}")
            print(f"🟧\t\tTotal Memory: {memory_gb:.1f}GB")
            print(f"🟧\t\tFree Memory: {free_gb:.1f}GB")
            print(f"🟧\t\tTemperature: {gpu['temperature']}°C")
            print(f"🟧\t\tUtilization: {gpu['utilization']}%")
        
        print(f"\n🟧\tSystem-determined Strategy: {self.strategy['mode']}")
        print(f"🟧\tReason: {self.strategy['reason']}")
        print(f"🟧\tRecommended Batch Size: {self.strategy['batch_size']}")
        print(f"🟧\tGPUs Used: {self.strategy['gpu_ids']}")
        print("=" * 50)
    
    def get_recommendation(self) -> Dict:
        """Get the system-level recommended configuration."""
        return {
            'strategy': self.strategy,
            'gpu_info': self.gpu_info,
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

    def get_inference_strategy(self) -> Dict:
        """Get the system-level inference strategy - CPU inference is allowed."""
        if not self.gpu_info:
            return {
                'mode': 'cpu',
                'gpu_ids': [],
                'batch_size': 1,
                'mixed_precision': False,
                'reason': 'CPU inference mode'
            }
        
        # Use the training strategy but reduce the batch size for inference
        strategy = self.strategy.copy()
        strategy['batch_size'] = max(1, strategy['batch_size'] // 4)
        return strategy