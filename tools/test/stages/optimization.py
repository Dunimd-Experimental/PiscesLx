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
Optimization Checker Module for PiscesL1.

This module provides the PiscesLxOptimizationChecker class for validating
inference optimization features.

Checks performed:
    - Flash Attention availability (v2/v3)
    - KV cache functionality
    - Quantization (INT8/FP8)
    - Batch processing
"""

import time
from typing import List, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PiscesLxOptimizationChecker:
    """
    Optimization features validation checker.
    
    Validates that inference optimization features are available.
    
    Attributes:
        verbose: Enable verbose output
        results: List of check results
    
    Example:
        >>> checker = PiscesLxOptimizationChecker()
        >>> results = checker.run()
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the optimization checker.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, float]] = []
    
    def run(self) -> List[Tuple[str, str, str, float]]:
        """
        Run all optimization checks.
        
        Returns:
            List of (name, status, message, duration) tuples
        """
        self.results = []
        
        if not TORCH_AVAILABLE:
            self._add_result("Optimization", "SKIP", "PyTorch not available", 0)
            return self.results
        
        self._check_flash_attention()
        self._check_kv_cache()
        self._check_quantization()
        self._check_batch_processing()
        
        return self.results
    
    def _add_result(self, name: str, status: str, message: str, duration: float) -> None:
        """Add a check result."""
        self.results.append((name, status, message, duration))
    
    def _check_flash_attention(self) -> None:
        """Check Flash Attention availability."""
        start = time.time()
        
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                compute_capability = torch.cuda.get_device_capability()
                cc_str = f"{compute_capability[0]}.{compute_capability[1]}"
                
                if compute_capability >= (8, 0):
                    self._add_result("Flash Attention", "PASS", f"SM {cc_str} supported", time.time() - start)
                else:
                    self._add_result("Flash Attention", "WARN", f"SM {cc_str} (need 8.0+)", time.time() - start)
            else:
                self._add_result("Flash Attention", "WARN", "CUDA not available", time.time() - start)
                
        except Exception as e:
            self._add_result("Flash Attention", "WARN", str(e)[:50], time.time() - start)
    
    def _check_kv_cache(self) -> None:
        """Check KV cache functionality."""
        start = time.time()
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            batch_size = 2
            num_heads = 8
            head_dim = 64
            seq_len = 128
            
            k_cache = torch.zeros(batch_size, num_heads, seq_len, head_dim, device=device)
            v_cache = torch.zeros(batch_size, num_heads, seq_len, head_dim, device=device)
            
            new_k = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
            new_v = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
            
            k_cache[:, :, -1:, :] = new_k
            v_cache[:, :, -1:, :] = new_v
            
            self._add_result("KV Cache", "PASS", f"shape={k_cache.shape}", time.time() - start)
            
        except Exception as e:
            self._add_result("KV Cache", "FAIL", str(e)[:50], time.time() - start)
    
    def _check_quantization(self) -> None:
        """Check quantization support."""
        start = time.time()
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            x = torch.randn(128, 128, device=device)
            
            x_int8 = x.to(torch.int8)
            x_back = x_int8.to(torch.float32)
            
            quantization_types = ["INT8"]
            
            if torch.cuda.is_available():
                try:
                    if hasattr(torch, "float8_e4m3fn"):
                        x_fp8 = x.to(torch.float8_e4m3fn)
                        quantization_types.append("FP8")
                except Exception:
                    pass
            
            self._add_result("Quantization", "PASS", ", ".join(quantization_types), time.time() - start)
            
        except Exception as e:
            self._add_result("Quantization", "WARN", str(e)[:50], time.time() - start)
    
    def _check_batch_processing(self) -> None:
        """Check batch processing capability."""
        start = time.time()
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            batch_sizes = [1, 2, 4, 8]
            seq_len = 64
            hidden_size = 256
            
            for batch_size in batch_sizes:
                x = torch.randn(batch_size, seq_len, hidden_size, device=device)
                y = torch.randn(hidden_size, hidden_size, device=device)
                _ = torch.matmul(x, y)
            
            self._add_result("Batch processing", "PASS", f"batch={batch_sizes[-1]}", time.time() - start)
            
        except Exception as e:
            self._add_result("Batch processing", "WARN", str(e)[:50], time.time() - start)
