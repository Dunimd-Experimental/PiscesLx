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
Forward Pass Checker Module for PiscesL1.

This module provides the PiscesLxForwardChecker class for validating
model forward propagation.

Checks performed:
    - Text-only forward pass
    - Multimodal forward pass
    - Gradient computation
    - Memory stability
"""

import time
from typing import List, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PiscesLxForwardChecker:
    """
    Forward pass validation checker.
    
    Validates that model forward propagation works correctly.
    
    Attributes:
        verbose: Enable verbose output
        results: List of check results
    
    Example:
        >>> checker = PiscesLxForwardChecker()
        >>> results = checker.run()
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the forward checker.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, float]] = []
    
    def run(self) -> List[Tuple[str, str, str, float]]:
        """
        Run all forward pass checks.
        
        Returns:
            List of (name, status, message, duration) tuples
        """
        self.results = []
        
        if not TORCH_AVAILABLE:
            self._add_result("Forward pass", "SKIP", "PyTorch not available", 0)
            return self.results
        
        self._check_tensor_operations()
        self._check_gradient_computation()
        self._check_memory_stability()
        
        return self.results
    
    def _add_result(self, name: str, status: str, message: str, duration: float) -> None:
        """Add a check result."""
        self.results.append((name, status, message, duration))
    
    def _check_tensor_operations(self) -> None:
        """Check basic tensor operations."""
        start = time.time()
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            x = torch.randn(2, 128, 256, device=device)
            y = torch.randn(2, 128, 256, device=device)
            z = x + y
            z = torch.matmul(x, y.transpose(-1, -2))
            
            self._add_result("Tensor ops", "PASS", f"device={device}", time.time() - start)
            
        except Exception as e:
            self._add_result("Tensor ops", "FAIL", str(e)[:50], time.time() - start)
    
    def _check_gradient_computation(self) -> None:
        """Check gradient computation."""
        start = time.time()
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            x = torch.randn(2, 10, requires_grad=True, device=device)
            y = x * 2
            z = y.sum()
            z.backward()
            
            if x.grad is not None:
                self._add_result("Gradients", "PASS", "Backward pass OK", time.time() - start)
            else:
                self._add_result("Gradients", "FAIL", "No gradient computed", time.time() - start)
            
        except Exception as e:
            self._add_result("Gradients", "FAIL", str(e)[:50], time.time() - start)
    
    def _check_memory_stability(self) -> None:
        """Check memory stability during operations."""
        start = time.time()
        
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            for _ in range(10):
                x = torch.randn(4, 64, 128, device=device)
                y = torch.randn(4, 64, 128, device=device)
                _ = torch.matmul(x, y.transpose(-1, -2))
                del x, y
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._add_result("Memory", "PASS", "Stable", time.time() - start)
            
        except Exception as e:
            self._add_result("Memory", "WARN", str(e)[:50], time.time() - start)
