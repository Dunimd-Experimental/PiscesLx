#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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
from utils.log import RIGHT, ERROR

def check(args=None, extra=None):
    """
    Check GPU availability and status.

    Args:
        args (optional): Optional arguments, default is None.
        extra (optional): Extra information, default is None.

    Returns:
        bool: True if tensor operations are successful, False otherwise.
    """
    # Validate arguments
    try:
        validate_check_args(args, extra)
    except Exception as e:
        ERROR(f"Invalid check arguments: {e}")
    # Log the start of GPU status check
    RIGHT("GPU Status Check")
    # Check if PyTorch CUDA is available
    RIGHT(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        # Log the CUDA version
        RIGHT(f"CUDA version: {torch.version.cuda}")
        # Log the number of available GPUs
        RIGHT(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            # Get the properties of the current GPU
            props = torch.cuda.get_device_properties(i)
            # Log the name of the current GPU
            RIGHT(f"GPU {i}: {props.name}")
            # Log the total memory of the current GPU in GB
            RIGHT(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            # Log the compute capability of the current GPU
            RIGHT(f"  Compute Capability: {props.major}.{props.minor}")
            if i == 0:
                # Empty the CUDA cache
                torch.cuda.empty_cache()
                # Get the currently allocated memory in GB
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                # Get the currently reserved memory in GB
                cached = torch.cuda.memory_reserved(i) / 1024**3
                # Log the allocated memory
                RIGHT(f"  Allocated: {allocated:.2f} GB")
                # Log the cached memory
                RIGHT(f"  Cached: {cached:.2f} GB")
    else:
        # Log that no CUDA-capable GPU is found
        ERROR("No CUDA-capable GPU found")
        # Log that training will use CPU
        ERROR("Training will use CPU (slower but functional)")
    
    # Log the start of tensor operation testing
    RIGHT("=" * 50)
    RIGHT("Testing tensor operations...")
    try:
        # Select the device (CUDA if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create random tensors and move them to the selected device
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        # Perform matrix multiplication
        z = torch.mm(x, y)
        # Log that tensor operations are successful
        RIGHT(f"Tensor operations successful on {device}")
    except Exception as e:
        # Log that tensor operations failed
        ERROR(f"Tensor operations failed: {e}")
        return False
    return True


def validate_check_args(args=None, extra=None):
    """Validate/normalize arguments for tools.check.check().
    args/extra are optional; ensure they are None or dict-like when provided.
    """
    if args is not None and not isinstance(args, (dict, object)):
        # allow any object (Namespace), but reject obvious wrong types
        raise ValueError("args must be None or an object-like container")
    if extra is not None and not isinstance(extra, (dict, str)):
        raise ValueError("extra must be None, dict or str")