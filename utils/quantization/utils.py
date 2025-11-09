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

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from utils.log.core import PiscesLxCoreLog
from utils.error import PiscesLxCoreValidationError, PiscesLxCoreIOError

logger = PiscesLxCoreLog("PiscesLx.Core.Quantization.Utils")

def get_device_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Get memory information for the specified device."""
    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda" and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            memory_free = memory_total - memory_allocated
            
            return {
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
                "total_gb": memory_total,
                "free_gb": memory_free,
                "utilization_percent": (memory_allocated / memory_total) * 100 if memory_total > 0 else 0
            }
        else:
            # For CPU, return placeholder values
            return {
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "total_gb": 0.0,
                "free_gb": 0.0,
                "utilization_percent": 0.0
            }
            
    except Exception as e:
        logger.error(f"Failed to get device memory info: {e}")
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": 0.0,
            "free_gb": 0.0,
            "utilization_percent": 0.0
        }

def calculate_model_size(model: nn.Module, return_breakdown: bool = False) -> Union[float, Dict[str, float]]:
    """Calculate the size of a model in MB."""
    try:
        param_size = 0
        buffer_size = 0
        param_breakdown = {}
        buffer_breakdown = {}
        
        for name, param in model.named_parameters():
            param_size_bytes = param.nelement() * param.element_size()
            param_size += param_size_bytes
            if return_breakdown:
                param_breakdown[name] = param_size_bytes / 1024 / 1024  # MB
        
        for name, buffer in model.named_buffers():
            buffer_size_bytes = buffer.nelement() * buffer.element_size()
            buffer_size += buffer_size_bytes
            if return_breakdown:
                buffer_breakdown[name] = buffer_size_bytes / 1024 / 1024  # MB
        
        total_size_mb = (param_size + buffer_size) / 1024 / 1024  # MB
        
        if return_breakdown:
            return {
                "total_mb": total_size_mb,
                "parameters_mb": param_size / 1024 / 1024,
                "buffers_mb": buffer_size / 1024 / 1024,
                "parameter_breakdown": param_breakdown,
                "buffer_breakdown": buffer_breakdown
            }
        else:
            return total_size_mb
            
    except Exception as e:
        logger.error(f"Failed to calculate model size: {e}")
        return 0.0 if not return_breakdown else {"total_mb": 0.0}

def estimate_quantized_size(original_size_mb: float, bits: int, 
                          include_buffers: bool = True) -> float:
    """Estimate the size of a quantized model."""
    try:
        # Assume 32-bit original precision
        original_bits = 32
        compression_ratio = original_bits / bits
        
        # Account for buffers (typically not quantized)
        buffer_ratio = 0.1 if include_buffers else 0.0
        quantized_size_mb = (original_size_mb * (1 - buffer_ratio)) / compression_ratio + \
                           (original_size_mb * buffer_ratio)
        
        return quantized_size_mb
        
    except Exception as e:
        logger.error(f"Failed to estimate quantized size: {e}")
        return original_size_mb

def get_quantization_method_info(method: str) -> Dict[str, Any]:
    """Get information about a quantization method."""
    method_info = {
        "bitsandbytes": {
            "name": "BitsAndBytes",
            "description": "Efficient quantization using bitsandbytes library",
            "supported_bits": [4, 8],
            "training_required": False,
            "calibration_required": False,
            "typical_compression": 2.0,
            "accuracy_preservation": "high",
            "speed_impact": "minimal",
            "memory_efficiency": "high"
        },
        "dynamic": {
            "name": "Dynamic Quantization",
            "description": "PyTorch dynamic quantization",
            "supported_bits": [8, 16],
            "training_required": False,
            "calibration_required": False,
            "typical_compression": 2.0,
            "accuracy_preservation": "medium",
            "speed_impact": "low",
            "memory_efficiency": "high"
        },
        "static": {
            "name": "Static Quantization",
            "description": "PyTorch static quantization with calibration",
            "supported_bits": [8, 16],
            "training_required": False,
            "calibration_required": True,
            "typical_compression": 2.0,
            "accuracy_preservation": "medium",
            "speed_impact": "minimal",
            "memory_efficiency": "high"
        },
        "gptq": {
            "name": "GPTQ",
            "description": "Gradient-based Post-training Quantization",
            "supported_bits": [2, 3, 4, 8],
            "training_required": False,
            "calibration_required": True,
            "typical_compression": 4.0,
            "accuracy_preservation": "high",
            "speed_impact": "low",
            "memory_efficiency": "high"
        },
        "awq": {
            "name": "AWQ",
            "description": "Activation-aware Weight Quantization",
            "supported_bits": [4, 8],
            "training_required": False,
            "calibration_required": False,
            "typical_compression": 4.0,
            "accuracy_preservation": "high",
            "speed_impact": "low",
            "memory_efficiency": "high"
        },
        "squeezellm": {
            "name": "SqueezeLLM",
            "description": "Sparse and dense quantization",
            "supported_bits": [3, 4],
            "training_required": False,
            "calibration_required": True,
            "typical_compression": 6.0,
            "accuracy_preservation": "medium",
            "speed_impact": "medium",
            "memory_efficiency": "very_high"
        },
        "kvcache": {
            "name": "KV Cache Quantization",
            "description": "Quantization of key-value cache",
            "supported_bits": [4, 8],
            "training_required": False,
            "calibration_required": False,
            "typical_compression": 2.0,
            "accuracy_preservation": "high",
            "speed_impact": "minimal",
            "memory_efficiency": "high"
        },
        "bitnet_v2": {
            "name": "BitNet v2",
            "description": "1.58-bit ternary quantization with Hadamard transform",
            "supported_bits": [2],  # 1.58-bit represented as 2-bit
            "training_required": True,
            "calibration_required": False,
            "typical_compression": 16.0,
            "accuracy_preservation": "medium",
            "speed_impact": "minimal",
            "memory_efficiency": "very_high"
        }
    }
    
    return method_info.get(method, {
        "name": "Unknown",
        "description": "Unknown quantization method",
        "supported_bits": [],
        "training_required": False,
        "calibration_required": False,
        "typical_compression": 1.0,
        "accuracy_preservation": "unknown",
        "speed_impact": "unknown",
        "memory_efficiency": "unknown"
    })

def compare_quantization_methods(methods: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Compare multiple quantization methods."""
    if methods is None:
        methods = ["bitsandbytes", "dynamic", "static", "gptq", "awq", "bitnet_v2"]
    
    comparison = {}
    for method in methods:
        comparison[method] = get_quantization_method_info(method)
    
    return comparison

def recommend_quantization_method(
    model_size_mb: float,
    target_compression: float,
    accuracy_priority: str = "high",
    inference_speed_priority: str = "medium",
    hardware_constraints: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Recommend quantization methods based on requirements."""
    try:
        all_methods = ["bitsandbytes", "dynamic", "static", "gptq", "awq", "squeezellm", "bitnet_v2"]
        
        recommendations = []
        
        for method in all_methods:
            info = get_quantization_method_info(method)
            
            # Check if method meets compression target
            if info["typical_compression"] >= target_compression:
                
                # Check accuracy priority
                accuracy_match = (
                    (accuracy_priority == "high" and info["accuracy_preservation"] in ["high"]) or
                    (accuracy_priority == "medium" and info["accuracy_preservation"] in ["high", "medium"]) or
                    (accuracy_priority == "low")
                )
                
                # Check speed priority
                speed_match = (
                    (inference_speed_priority == "high" and info["speed_impact"] in ["minimal", "low"]) or
                    (inference_speed_priority == "medium" and info["speed_impact"] in ["minimal", "low", "medium"]) or
                    (inference_speed_priority == "low")
                )
                
                if accuracy_match and speed_match:
                    # Calculate suitability score
                    score = 0
                    score += info["typical_compression"] / target_compression  # Compression efficiency
                    score += {"high": 3, "medium": 2, "low": 1}.get(info["accuracy_preservation"], 0)  # Accuracy
                    score += {"minimal": 3, "low": 2, "medium": 1}.get(info["speed_impact"], 0)  # Speed
                    score += {"very_high": 4, "high": 3}.get(info["memory_efficiency"], 2)  # Memory efficiency
                    
                    recommendations.append({
                        "method": method,
                        "info": info,
                        "suitability_score": score,
                        "meets_requirements": True
                    })
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to recommend quantization method: {e}")
        return []

def validate_quantization_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate quantization configuration."""
    errors = []
    
    try:
        # Check required fields
        if "method" not in config:
            errors.append("Missing required field: method")
        
        if "bits" not in config:
            errors.append("Missing required field: bits")
        
        # Validate method
        if "method" in config:
            method = config["method"]
            method_info = get_quantization_method_info(method)
            if method_info["name"] == "Unknown":
                errors.append(f"Unknown quantization method: {method}")
        
        # Validate bits
        if "bits" in config:
            bits = config["bits"]
            if not isinstance(bits, int) or bits < 1 or bits > 16:
                errors.append(f"Invalid bits value: {bits}. Must be integer between 1 and 16")
            
            if "method" in config:
                method = config["method"]
                method_info = get_quantization_method_info(method)
                if bits not in method_info.get("supported_bits", []):
                    errors.append(f"Method {method} does not support {bits} bits")
        
        # Validate granularity
        if "granularity" in config:
            granularity = config["granularity"]
            valid_granularities = ["per_tensor", "per_channel", "per_group", "per_token"]
            if granularity not in valid_granularities:
                errors.append(f"Invalid granularity: {granularity}. Must be one of {valid_granularities}")
        
        # Validate group_size
        if "group_size" in config:
            group_size = config["group_size"]
            if not isinstance(group_size, int) or group_size < 0:
                errors.append(f"Invalid group_size: {group_size}. Must be non-negative integer")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Validation error: {e}")
        return False, errors

def safe_quantization_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a safe quantization configuration with defaults."""
    safe_config = config.copy()
    
    # Set safe defaults
    if "bits" not in safe_config:
        safe_config["bits"] = 8
    
    if "granularity" not in safe_config:
        safe_config["granularity"] = "per_tensor"
    
    if "group_size" not in safe_config:
        safe_config["group_size"] = 128
    
    if "num_calibration_samples" not in safe_config:
        safe_config["num_calibration_samples"] = 128
    
    # Validate and fix method-specific issues
    if "method" in safe_config:
        method = safe_config["method"]
        method_info = get_quantization_method_info(method)
        
        # Fix bits if not supported
        if "bits" in safe_config and safe_config["bits"] not in method_info.get("supported_bits", []):
            supported_bits = method_info.get("supported_bits", [8])
            safe_config["bits"] = min(supported_bits, key=lambda x: abs(x - safe_config["bits"]))
            logger.warning(f"Adjusted bits from {config['bits']} to {safe_config['bits']} for method {method}")
    
    return safe_config