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

"""
GPU Health Check Operator

This module provides comprehensive GPU health checking capabilities for the
PiscesL1 inference system. It validates CUDA availability, GPU status, and
performs tensor operation tests to ensure the GPU environment is ready for
deep learning inference operations.

Key Features:
    - CUDA availability detection with version reporting
    - Multi-GPU support with detailed property reporting
    - Memory usage tracking (allocated, cached, total)
    - Tensor operation validation through matrix multiplication
    - Thread-safe execution with standardized result format

Operator Class:
    POPSSGPUCheckOperator: Main operator for GPU health checking

Input Format:
    {
        "test_tensor_ops": bool,      # Whether to perform tensor operation tests (default: True)
        "tensor_size": int,           # Size of test matrix for operations (default: 1000)
        "device_id": Optional[int],   # Specific GPU device to test, None for all (default: None)
        "clear_cache": bool            # Clear CUDA cache before memory check (default: True)
    }

Output Format:
    {
        "cuda_available": bool,         # Whether CUDA is available
        "cuda_version": str,           # CUDA version string
        "gpu_count": int,              # Number of available GPUs
        "gpus": List[Dict],            # List of GPU information dictionaries
        "tensor_ops_success": bool,     # Whether tensor operations passed
        "tensor_ops_device": str,       # Device used for tensor operations
        "memory_info": Optional[Dict],  # Memory statistics for primary GPU
        "status": str                  # Overall health status: "healthy", "warning", "critical"
    }

GPU Information Dictionary:
    {
        "device_id": int,
        "name": str,
        "total_memory_gb": float,
        "compute_capability": str,
        "allocated_memory_gb": float,
        "cached_memory_gb": float
    }

Usage Examples:
    >>> from opss.infer.check import POPSSGPUCheckOperator
    >>> operator = POPSSGPUCheckOperator()
    >>> result = operator.execute({})
    >>> print(result.output["status"])
    healthy

    >>> # Custom tensor test
    >>> result = operator.execute({"tensor_size": 500, "test_tensor_ops": True})

    >>> # Check specific GPU
    >>> result = operator.execute({"device_id": 0})

Author: PiscesL1 Development Team
Version: 1.0.0
"""

import torch
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from utils.opsc.base import PiscesLxBaseOperator
from utils.opsc.interface import PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION


@dataclass
class GPUInfo:
    """Data class representing individual GPU information."""
    device_id: int
    name: str
    total_memory_gb: float
    compute_capability: str
    allocated_memory_gb: float = 0.0
    cached_memory_gb: float = 0.0


@dataclass
class GPUMemoryInfo:
    """Data class representing GPU memory statistics."""
    allocated_gb: float
    cached_gb: float
    total_gb: float


class POPSSGPUCheckOperator(PiscesLxBaseOperator):
    """
    GPU Health Check Operator for Inference System
    
    This operator provides comprehensive GPU health checking capabilities,
    validating CUDA availability, GPU properties, memory usage, and
    performing tensor operation tests to ensure the GPU environment
    is ready for inference operations.
    
    Attributes:
        name: Operator identifier (default: "pisceslx_gpu_check_operator")
        version: Operator version string (default: "1.0.0")
        description: Human-readable description of operator functionality
    
    Input Validation:
        - test_tensor_ops: Must be boolean, defaults to True
        - tensor_size: Must be positive integer, defaults to 1000
        - device_id: Must be None or non-negative integer less than GPU count
        - clear_cache: Must be boolean, defaults to True
    
    Error Handling:
        - Catches all exceptions during GPU detection and testing
        - Returns FAILED status with detailed error message on exception
        - Continues partial execution when possible (e.g., CUDA available but tensor ops fail)
    
    Thread Safety:
        - Uses thread-safe CUDA API calls
        - Protected by base class RLock for concurrent execution
    
    Example:
        >>> operator = POPSSGPUCheckOperator()
        >>> result = operator.execute({})
        >>> if result.is_success():
        ...     print(f"GPU Status: {result.output['status']}")
        ...     print(f"GPUs Found: {result.output['gpu_count']}")
    """
    
    def __init__(self):
        super().__init__()
        self.name = "pisceslx_gpu_check_operator"
        self.version = VERSION
        self.description = "GPU health check operator for inference system validation"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Define input schema for GPU check operator.
        
        Returns:
            JSON Schema defining expected input parameters
        """
        return {
            "type": "object",
            "properties": {
                "test_tensor_ops": {
                    "type": "boolean",
                    "description": "Whether to perform tensor operation tests",
                    "default": True
                },
                "tensor_size": {
                    "type": "integer",
                    "description": "Size of square matrix for tensor operations",
                    "minimum": 100,
                    "maximum": 10000,
                    "default": 1000
                },
                "device_id": {
                    "type": ["integer", "null"],
                    "description": "Specific GPU device to test, None for all devices",
                    "minimum": 0
                },
                "clear_cache": {
                    "type": "boolean",
                    "description": "Clear CUDA cache before memory measurement",
                    "default": True
                }
            },
            "additionalProperties": False
        }
    
    def _execute_impl(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> PiscesLxOperatorResult:
        """
        Execute GPU health checking implementation.
        
        This method performs comprehensive GPU validation including:
        1. CUDA availability and version detection
        2. GPU device enumeration and property reporting
        3. Memory usage analysis
        4. Tensor operation validation
        
        Args:
            inputs: Dictionary containing optional parameters
            **kwargs: Additional keyword arguments
            
        Returns:
            PiscesLxOperatorResult with comprehensive GPU health information
        """
        try:
            test_tensor_ops = inputs.get("test_tensor_ops", True)
            tensor_size = inputs.get("tensor_size", 1000)
            target_device_id = inputs.get("device_id")
            clear_cache = inputs.get("clear_cache", True)
            
            cuda_available = torch.cuda.is_available()
            cuda_version = str(torch.version.cuda) if cuda_available else "N/A"
            gpu_count = torch.cuda.device_count() if cuda_available else 0
            
            gpus: List[Dict[str, Any]] = []
            memory_info: Optional[Dict[str, Any]] = None
            
            if cuda_available:
                for i in range(gpu_count):
                    if target_device_id is not None and i != target_device_id:
                        continue
                    
                    props = torch.cuda.get_device_properties(i)
                    
                    gpu_data = {
                        "device_id": i,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                        "compute_capability": f"{props.major}.{props.minor}"
                    }
                    
                    if i == 0 and clear_cache:
                        torch.cuda.empty_cache()
                    
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    
                    gpu_data["allocated_memory_gb"] = round(allocated, 2)
                    gpu_data["cached_memory_gb"] = round(cached, 2)
                    
                    if i == 0:
                        memory_info = {
                            "allocated_gb": round(allocated, 2),
                            "cached_gb": round(cached, 2),
                            "total_gb": round(props.total_memory / (1024 ** 3), 2)
                        }
                    
                    gpus.append(gpu_data)
            
            tensor_ops_success = False
            tensor_ops_device = "cpu"
            
            if test_tensor_ops and cuda_available:
                try:
                    device = torch.device("cuda")
                    x = torch.randn(tensor_size, tensor_size).to(device)
                    y = torch.randn(tensor_size, tensor_size).to(device)
                    z = torch.mm(x, y)
                    tensor_ops_success = True
                    tensor_ops_device = str(device)
                except Exception:
                    tensor_ops_success = False
            
            overall_status = self._determine_health_status(
                cuda_available=cuda_available,
                gpu_count=gpu_count,
                tensor_ops_success=tensor_ops_success,
                gpus=gpus
            )
            
            output: Dict[str, Any] = {
                "cuda_available": cuda_available,
                "cuda_version": cuda_version,
                "gpu_count": gpu_count,
                "gpus": gpus,
                "tensor_ops_success": tensor_ops_success,
                "tensor_ops_device": tensor_ops_device,
                "memory_info": memory_info,
                "status": overall_status
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=output,
                metadata={
                    "version": self.version,
                    "test_tensor_ops": test_tensor_ops,
                    "tensor_size": tensor_size,
                    "target_device_id": target_device_id
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"GPU check failed: {str(e)}",
                output={
                    "cuda_available": False,
                    "cuda_version": "N/A",
                    "gpu_count": 0,
                    "gpus": [],
                    "tensor_ops_success": False,
                    "tensor_ops_device": "N/A",
                    "memory_info": None,
                    "status": "critical"
                }
            )
    
    def _determine_health_status(
        self,
        cuda_available: bool,
        gpu_count: int,
        tensor_ops_success: bool,
        gpus: List[Dict[str, Any]]
    ) -> str:
        """
        Determine overall GPU health status based on collected metrics.
        
        This method evaluates multiple health indicators and assigns an
        overall status level: healthy, warning, or critical.
        
        Status Criteria:
            - healthy: CUDA available, at least one GPU, tensor ops pass
            - warning: CUDA available but no GPUs detected, or tensor ops fail
            - critical: CUDA not available or major errors
        
        Args:
            cuda_available: Whether CUDA runtime is available
            gpu_count: Number of detected GPUs
            tensor_ops_success: Whether tensor operation tests passed
            gpus: List of GPU information dictionaries
            
        Returns:
            String representing overall health status
        """
        if not cuda_available:
            return "critical"
        
        if gpu_count == 0:
            return "warning"
        
        if not tensor_ops_success:
            return "warning"
        
        for gpu in gpus:
            if gpu.get("allocated_memory_gb", 0) > gpu.get("total_memory_gb", float('inf')):
                return "warning"
        
        return "healthy"
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate input parameters for GPU check operator.
        
        Args:
            inputs: Input dictionary to validate
            
        Returns:
            True if all inputs are valid, False otherwise
        """
        if not isinstance(inputs, dict):
            return False
        
        test_tensor_ops = inputs.get("test_tensor_ops", True)
        if not isinstance(test_tensor_ops, bool):
            return False
        
        tensor_size = inputs.get("tensor_size", 1000)
        if not isinstance(tensor_size, int) or tensor_size < 100 or tensor_size > 10000:
            return False
        
        device_id = inputs.get("device_id")
        if device_id is not None:
            if not isinstance(device_id, int) or device_id < 0:
                return False
        
        clear_cache = inputs.get("clear_cache", True)
        if not isinstance(clear_cache, bool):
            return False
        
        return True
