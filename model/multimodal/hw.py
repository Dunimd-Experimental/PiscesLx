#!/usr/bin/env/python3
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

"""Hardware detection utilities for Yv multimodal agents.

This module provides hardware detection and adaptive configuration components
for the Yv model, enabling automatic workload adaptation based on
available hardware resources.

Module Components:
    1. YvHardwareAdaptiveConfig:
       - Hardware detection and tier classification
       - Adaptive configuration generation
       - Memory and device count analysis

Key Features:
    - CUDA device detection and memory analysis
    - Tier-based configuration profiles (minimal to maximum)
    - Gradient clipping and mixed precision settings
    - Checkpoint segmentation recommendations
    - Fallback to minimal configuration on detection failure

Configuration Tiers:
    - minimal: 1 GPU, <40GB VRAM
    - conservative: 1 GPU, 40-80GB VRAM
    - medium: 2 GPUs, 80-160GB VRAM
    - high: 2 GPUs, 160-320GB VRAM
    - maximum: 4+ GPUs, 320GB+ VRAM

Performance Characteristics:
    - Detection: O(D) where D = number of CUDA devices
    - Configuration lookup: O(1) dictionary access

Usage Example:
    >>> from model.multimodal.hw import YvHardwareAdaptiveConfig
    >>> 
    >>> # Initialize hardware detection
    >>> hw_config = YvHardwareAdaptiveConfig()
    >>> 
    >>> # Get device info
    >>> device_count = hw_config.device_info["device_count"]
    >>> total_memory = hw_config.device_info["total_memory_gb"]
    >>> 
    >>> # Get adaptive configuration
    >>> config = hw_config.adaptive_config
    >>> max_layers = config["max_layers"]

Note:
    Falls back to minimal configuration if hardware detection fails.
    Mixed precision enabled for memory-constrained configurations.
"""

from typing import Dict

class YvHardwareAdaptiveConfig:
    """Hardware adapter that derives configuration tiers from detected devices.
    
    A comprehensive hardware detection and configuration system that analyzes
    available CUDA devices, determines optimal configuration tiers, and generates
    adaptive configuration profiles for workload optimization.
    
    Architecture:
        1. Hardware Detection:
           - CUDA device count analysis
           - Total VRAM calculation
           - Tier classification based on resources
        
        2. Configuration Generation:
           - Tier-based profile selection
           - Gradient clipping settings
           - Mixed precision recommendations
           - Checkpoint segmentation hints
    
    Key Features:
        - Automatic CUDA device detection
        - Memory-based tier classification
        - Adaptive configuration profiles
        - Fallback to minimal configuration on failure
    
    Attributes:
        device_info (Dict[str, Dict]): Raw device detection results including:
            - device_count: Number of CUDA devices
            - total_memory_gb: Total VRAM in GB
            - recommended_config: Tier name (minimal/conservative/medium/high/maximum)
        adaptive_config (Dict[str, Dict]): Profiled configuration containing:
            - max_layers: Maximum transformer layers
            - max_heads: Maximum attention heads
            - max_lstm_layers: Maximum LSTM layers
            - gradient_clip_norm: Gradient clipping threshold
            - use_mixed_precision: Whether to use FP16
            - checkpoint_segments: Number of checkpoint segments
    
    Example:
        >>> hw_config = YvHardwareAdaptiveConfig()
        >>> print(hw_config.device_info["recommended_config"])
        'high'
        >>> print(hw_config.adaptive_config["max_layers"])
        6
    
    Note:
        Tier thresholds:
        - maximum: 4+ GPUs, 320GB+ VRAM
        - high: 2+ GPUs, 160-320GB VRAM
        - medium: 2+ GPUs, 80-160GB VRAM
        - conservative: 1+ GPU, 40-80GB VRAM
        - minimal: <40GB VRAM or detection failure
    """

    def __init__(self):
        """Detect hardware information and build the adaptive configuration.
        
        Initializes hardware detection and generates adaptive configuration
        based on detected device capabilities.
        
        Note:
            Calls _detect_hardware() for device analysis.
            Calls _profile() for configuration generation.
        """
        self.device_info = self._detect_hardware()
        self.adaptive_config = self._profile(self.device_info["recommended_config"])

    def _detect_hardware(self) -> Dict:
        """Detect available CUDA hardware and determine configuration tier.
        
        Analyzes CUDA device count and total VRAM to classify hardware
        into appropriate configuration tiers for workload optimization.
        
        Returns:
            Dict: Hardware information dictionary containing:
                - device_count (int): Number of CUDA devices
                - total_memory_gb (int): Total VRAM in gigabytes
                - recommended_config (str): Tier name
        
        Note:
            Tier classification:
            - maximum: 4+ devices, 320GB+ VRAM
            - high: 2+ devices, 160-320GB VRAM
            - medium: 2+ devices, 80-160GB VRAM
            - conservative: 1+ device, 40-80GB VRAM
            - minimal: <40GB VRAM or detection failure
        """
        try:
            import torch
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            total_mem = 0
            if device_count > 0:
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    total_mem += props.total_memory // (1024**3)
            if device_count >= 4 and total_mem >= 320:
                tier = "maximum"
            elif device_count >= 2 and total_mem >= 80:
                tier = "high" if total_mem >= 160 else "medium"
            elif device_count >= 1 and total_mem >= 40:
                tier = "conservative"
            else:
                tier = "minimal"
            return {
                "device_count": device_count,
                "total_memory_gb": total_mem,
                "recommended_config": tier,
            }
        except Exception:
            return {"device_count": 0, "total_memory_gb": 0, "recommended_config": "minimal"}

    def _profile(self, tier: str) -> Dict:
        """Get adaptive configuration based on hardware tier.
        
        Returns a configuration profile optimized for the specified tier,
        including layer counts, attention heads, and training parameters.
        
        Args:
            tier (str): Configuration tier name. Valid values:
                - "minimal": Memory-constrained environments
                - "conservative": Single GPU with moderate VRAM
                - "medium": Dual GPU setup
                - "high": High-end dual GPU setup
                - "maximum": Multi-GPU cluster
        
        Returns:
            Dict: Configuration dictionary containing:
                - max_layers (int): Maximum transformer layers
                - max_heads (int): Maximum attention heads
                - max_lstm_layers (int): Maximum LSTM layers
                - gradient_clip_norm (float): Gradient clipping threshold
                - use_mixed_precision (bool): FP16 training flag
                - checkpoint_segments (int): Gradient checkpoint segments
        
        Note:
            Falls back to "minimal" profile if tier is not recognized.
        """
        profiles = {
            "minimal":       {"max_layers": 2, "max_heads": 4,  "max_lstm_layers": 1, "gradient_clip_norm": 0.5, "use_mixed_precision": True,  "checkpoint_segments": 4},
            "conservative":  {"max_layers": 3, "max_heads": 8,  "max_lstm_layers": 1, "gradient_clip_norm": 1.0, "use_mixed_precision": True,  "checkpoint_segments": 2},
            "medium":        {"max_layers": 4, "max_heads": 12, "max_lstm_layers": 2, "gradient_clip_norm": 1.5, "use_mixed_precision": False, "checkpoint_segments": 1},
            "high":          {"max_layers": 6, "max_heads": 16, "max_lstm_layers": 3, "gradient_clip_norm": 2.0, "use_mixed_precision": False, "checkpoint_segments": 1},
            "maximum":       {"max_layers": 8, "max_heads": 24, "max_lstm_layers": 4, "gradient_clip_norm": 3.0, "use_mixed_precision": False, "checkpoint_segments": 1},
        }
        return profiles.get(tier, profiles["minimal"])

    def get_gradient_config(self) -> Dict:
        """Get gradient-related configuration parameters.
        
        Extracts gradient clipping, mixed precision, and checkpoint
        settings from the adaptive configuration.
        
        Returns:
            Dict: Gradient configuration containing:
                - max_grad_norm (float): Maximum gradient norm for clipping
                - use_mixed_precision (bool): Whether to use FP16 training
                - checkpoint_segments (int): Number of checkpoint segments
        
        Example:
            >>> hw_config = YvHardwareAdaptiveConfig()
            >>> grad_config = hw_config.get_gradient_config()
            >>> max_norm = grad_config["max_grad_norm"]
        """
        return {
            "max_grad_norm": self.adaptive_config["gradient_clip_norm"],
            "use_mixed_precision": self.adaptive_config["use_mixed_precision"],
            "checkpoint_segments": self.adaptive_config["checkpoint_segments"],
        }
