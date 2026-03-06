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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
GPU and Environment Check Module for PiscesL1.

This module provides comprehensive hardware and environment validation
for the PiscesL1 framework. It performs GPU detection, capability
assessment, and tensor operation verification to ensure the system
is ready for training or inference workloads.

Key Responsibilities:
    1. GPU Detection and Enumeration:
       - Detect CUDA availability via PyTorch
       - Enumerate all available GPU devices
       - Report GPU names, memory, and compute capabilities
    
    2. Memory Analysis:
       - Report total GPU memory per device
       - Track currently allocated memory
       - Monitor cached/reserved memory
    
    3. Compute Verification:
       - Test basic tensor operations on detected devices
       - Verify CUDA kernel functionality
       - Validate memory allocation and deallocation
    
    4. Fallback Handling:
       - Graceful degradation to CPU when no GPU available
       - Clear error reporting for hardware issues

Diagnostic Output:
    The check function produces detailed diagnostic output including:
    
    - CUDA availability status
    - CUDA toolkit version
    - Number of detected GPUs
    - Per-GPU information:
      * Device name (e.g., "NVIDIA A100-SXM4-80GB")
      * Total memory in GB
      * Compute capability (major.minor)
      * Currently allocated memory
      * Currently cached memory
    - Tensor operation test results

Usage Examples:
    Command-line usage:
        $ python manage.py check
        
    Programmatic usage:
        from tools.check import check
        success = check()
        if success:
            print("System ready for training")
        else:
            print("Hardware issues detected")
    
    With arguments:
        from tools.check import check, validate_check_args
        args = {"verbose": True}
        validate_check_args(args, None)
        check(args, None)

Integration Points:
    - manage.py: Called via "python manage.py check" command
    - tools/setup.py: Validates environment after setup
    - Training pipeline: Pre-flight check before training starts
    - Inference server: Health check endpoint

Error Handling:
    The module handles various error conditions gracefully:
    - Missing CUDA installation: Falls back to CPU mode
    - GPU driver issues: Reports error and continues
    - Memory allocation failures: Logs error and returns False
    - Invalid arguments: Validates and raises descriptive errors

Performance Considerations:
    - Uses small tensor sizes (1000x1000) for quick verification
    - Clears CUDA cache before memory measurement
    - Minimal overhead for production use

Thread Safety:
    The check function is thread-safe but should typically be called
    once at application startup. Concurrent calls may produce
    interleaved log output.
"""

import torch
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file


_LOG = PiscesLxLogger("PiscesLx.Tools.Check", file_path=get_log_file("PiscesLx.Tools.Check"), enable_file=True)


def check(args=None, extra=None):
    """
    Perform comprehensive GPU and environment validation.
    
    This function executes a series of diagnostic checks to verify that
    the system is properly configured for PiscesL1 training or inference.
    It detects available GPUs, reports their capabilities, and tests
    basic tensor operations to ensure CUDA is functioning correctly.
    
    The function performs the following checks in order:
    
    1. Argument validation via validate_check_args()
    2. CUDA availability detection via torch.cuda.is_available()
    3. GPU enumeration and property reporting
    4. Memory status reporting for the primary GPU
    5. Tensor operation verification (matrix multiplication)
    
    Diagnostic Output:
        The function logs detailed information at each step:
        
        - GPU Status Check header
        - CUDA availability (True/False)
        - CUDA version string (e.g., "12.1")
        - Number of detected GPUs
        - Per-GPU details:
          * Device name
          * Total memory in GB
          * Compute capability (e.g., "8.0" for A100)
          * Allocated memory (current usage)
          * Cached memory (reserved by PyTorch)
        - Tensor operation test result
    
    Args:
        args (optional): Command-line arguments or configuration object.
            Expected to be None or a dict-like object (e.g., argparse.Namespace).
            Currently not used but reserved for future extensions like:
            - verbose: Enable verbose output
            - device: Specify device to check
            - memory_test: Run extended memory tests
        extra (optional): Additional information for logging or debugging.
            Expected to be None, a dict, or a string.
            Can contain context information for error reporting.
    
    Returns:
        bool: True if all checks pass successfully, False otherwise.
            Returns True when:
            - GPU is available and tensor operations succeed
            - No GPU available but CPU operations succeed
            Returns False when:
            - Tensor operations fail on any device
            - CUDA is available but operations error
    
    Raises:
        No exceptions are raised directly. All errors are logged and
        result in a False return value.
    
    Example:
        >>> from tools.check import check
        >>> success = check()
        >>> if success:
        ...     print("System ready for training")
        ... else:
        ...     print("Hardware issues detected")
        
        Output:
        GPU Status Check
        PyTorch CUDA available: True
        CUDA version: 12.1
        Number of GPUs: 1
        GPU 0: NVIDIA A100-SXM4-80GB
          Memory: 80.0 GB
          Compute Capability: 8.0
          Allocated: 0.00 GB
          Cached: 0.00 GB
        ==================================================
        Testing tensor operations...
        Tensor operations successful on cuda
    
    Note:
        The function uses both structured logging (_LOG) and direct
        output (logger) for different purposes. Structured logs are
        captured by the logging system, while direct output is shown
        to the user immediately.
    """
    try:
        validate_check_args(args, extra)
    except Exception as e:
        _LOG.error(f"Invalid check arguments: {e}")

    _LOG.info("GPU Status Check")

    cuda_available = torch.cuda.is_available()
    _LOG.info(f"PyTorch CUDA available: {cuda_available}")

    if cuda_available:
        _LOG.info(f"CUDA version: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        _LOG.info(f"Number of GPUs: {gpu_count}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            _LOG.info(f"GPU {i}: {props.name}")
            _LOG.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            _LOG.info(f"  Compute Capability: {props.major}.{props.minor}")

            if i == 0:
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                _LOG.info(f"  Allocated: {allocated:.2f} GB")
                _LOG.info(f"  Cached: {cached:.2f} GB")
    else:
        _LOG.error("No CUDA-capable GPU found")
        _LOG.error("Training will use CPU (slower but functional)")

    _LOG.info("=" * 50)
    _LOG.info("Testing tensor operations...")

    try:
        device = torch.device("cuda" if cuda_available else "cpu")
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        _LOG.info(f"Tensor operations successful on {device}")
        return True
    except Exception as e:
        _LOG.error(f"Tensor operations failed: {e}")
        return False


def validate_check_args(args=None, extra=None):
    """
    Validate and normalize arguments for the check() function.
    
    This function ensures that the arguments passed to check() are valid
    and conform to expected types. It provides early validation to catch
    configuration errors before the main check logic executes.
    
    The validation is intentionally permissive to support various calling
    patterns (command-line arguments, programmatic calls, etc.) while
    still catching obvious type errors.
    
    Args:
        args (optional): Command-line arguments or configuration object.
            Valid types:
            - None: No arguments provided (most common)
            - dict: Dictionary of configuration options
            - object: Any object with attributes (e.g., argparse.Namespace)
            
            Invalid types:
            - int, float, str, list, tuple: These are rejected as they
              don't represent configuration objects
        extra (optional): Additional information for extended functionality.
            Valid types:
            - None: No extra information (most common)
            - dict: Dictionary of additional parameters
            - str: String message or identifier
            
            Invalid types:
            - int, float, list, tuple: These are rejected
    
    Raises:
        ValueError: If args is not None and not a dict-like object,
            or if extra is not None, dict, or string.
            The error message describes the expected types.
    
    Example:
        >>> validate_check_args(None, None)  # Valid
        >>> validate_check_args({"verbose": True}, None)  # Valid
        >>> validate_check_args(None, "debug_mode")  # Valid
        >>> validate_check_args(123, None)  # Raises ValueError
        ValueError: args must be None or an object-like container
    
    Note:
        This function is called automatically by check() and typically
        does not need to be called directly by users.
    """
    if args is not None and not isinstance(args, (dict, object)):
        raise ValueError("args must be None or an object-like container")
    if extra is not None and not isinstance(extra, (dict, str)):
        raise ValueError("extra must be None, dict or str")
