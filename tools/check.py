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

import os
import sys
import torch


def check(args=None, extra=None):
    """Check GPU availability and status"""
    print("✅\tGPU Status Check")
    print("✅\t" + "=" * 50)
    print(f"✅\tPyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅\tCUDA version: {torch.version.cuda}")
        print(f"✅\tNumber of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"✅\tGPU {i}: {props.name}")
            print(f"✅\t  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"✅\t  Compute Capability: {props.major}.{props.minor}")
            if i == 0:
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"✅\t  Allocated: {allocated:.2f} GB")
                print(f"✅\t  Cached: {cached:.2f} GB")
    else:
        print("❌\tNo CUDA-capable GPU found")
        print("❌\tTraining will use CPU (slower but functional)")
    print("✅\t" + "=" * 50)
    print("✅\tTesting tensor operations...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        print(f"✅\tTensor operations successful on {device}")
    except Exception as e:
        print(f"❌\tTensor operations failed: {e}")
        return False
    return True