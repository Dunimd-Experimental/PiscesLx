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

import torch
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager

logger = PiscesLxCoreLog("PiscesLx.Tools.Check")

def check(args=None, extra=None):
    """
    Check GPU availability and status, then test tensor operations.

    Args:
        args (optional): Optional arguments, default is None. Expected to be None or dict-like object.
        extra (optional): Extra information, default is None. Expected to be None, dict or string.

    Returns:
        bool: True if tensor operations are successful, False otherwise.
    """
    # Validate input arguments
    try:
        validate_check_args(args, extra)
    except Exception as e:
        logger.error(f"Invalid check arguments: {e}")

    # Log the start of GPU status check
    logger.success("GPU Status Check")

    # Check PyTorch CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.success(f"PyTorch CUDA available: {cuda_available}")

    if cuda_available:
        # Log CUDA version
        logger.success(f"CUDA version: {torch.version.cuda}")
        # Log the number of available GPUs
        gpu_count = torch.cuda.device_count()
        logger.success(f"Number of GPUs: {gpu_count}")

        for i in range(gpu_count):
            # Get properties of the current GPU
            props = torch.cuda.get_device_properties(i)
            # Log GPU name
            logger.success(f"GPU {i}: {props.name}")
            # Log total GPU memory in GB
            logger.success(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            # Log GPU compute capability
            logger.success(f"  Compute Capability: {props.major}.{props.minor}")

            if i == 0:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Get currently allocated memory in GB
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                # Get currently reserved memory in GB
                cached = torch.cuda.memory_reserved(i) / 1024**3
                # Log allocated memory
                logger.success(f"  Allocated: {allocated:.2f} GB")
                # Log cached memory
                logger.success(f"  Cached: {cached:.2f} GB")
    else:
        # Log that no CUDA-capable GPU was found
        logger.error("No CUDA-capable GPU found")
        # Log that training will fall back to CPU
        logger.error("Training will use CPU (slower but functional)")

    # Log the start of tensor operation testing
    logger.success("=" * 50)
    logger.success("Testing tensor operations...")

    try:
        # Select device: CUDA if available, otherwise CPU
        device = torch.device("cuda" if cuda_available else "cpu")
        # Create random tensors and move them to the selected device
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        # Perform matrix multiplication
        z = torch.mm(x, y)
        # Log successful tensor operations
        logger.success(f"Tensor operations successful on {device}")
        return True
    except Exception as e:
        # Log failed tensor operations
        logger.error(f"Tensor operations failed: {e}")
        return False


def validate_check_args(args=None, extra=None):
    """
    Validate and normalize arguments for the check() function.
    
    Args:
        args (optional): Optional arguments. Expected to be None or a dict-like object.
        extra (optional): Extra information. Expected to be None, a dict, or a string.
    
    Raises:
        ValueError: If args is not None and not a dict-like object, 
                   or if extra is not None, dict, or string.
    """
    if args is not None and not isinstance(args, (dict, object)):
        # Allow any object (e.g., Namespace), but reject obviously wrong types
        raise ValueError("args must be None or an object-like container")
    if extra is not None and not isinstance(extra, (dict, str)):
        raise ValueError("extra must be None, dict or str")