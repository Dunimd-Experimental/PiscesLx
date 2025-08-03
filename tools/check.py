#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch
from utils.log import RIGHT, ERROR


def check(args=None, extra=None):
    """Check GPU availability and status"""
    RIGHT("GPU Status Check")
    RIGHT(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        RIGHT(f"CUDA version: {torch.version.cuda}")
        RIGHT(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            RIGHT(f"GPU {i}: {props.name}")
            RIGHT(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            RIGHT(f"  Compute Capability: {props.major}.{props.minor}")
            if i == 0:
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                RIGHT(f"  Allocated: {allocated:.2f} GB")
                RIGHT(f"  Cached: {cached:.2f} GB")
    else:
        ERROR("No CUDA-capable GPU found")
        ERROR("Training will use CPU (slower but functional)")
    RIGHT("=" * 50)
    RIGHT("Testing tensor operations...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        RIGHT(f"Tensor operations successful on {device}")
    except Exception as e:
        ERROR(f"Tensor operations failed: {e}")
        return False
    return True