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
Environment Checker Module for PiscesL1.

This module provides the PiscesLxEnvironmentChecker class for validating
the runtime environment including Python version, PyTorch, CUDA, GPU
hardware, and system dependencies.

Checks performed:
    - Python version (>=3.10 required)
    - PyTorch installation and version
    - CUDA availability and version
    - GPU detection and memory
    - System memory
    - Required dependencies
"""

import sys
import time
import platform
from typing import List, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PiscesLxEnvironmentChecker:
    """
    Environment validation checker.
    
    Performs comprehensive checks on the runtime environment to ensure
    all prerequisites are met for running PiscesL1.
    
    Attributes:
        verbose: Enable verbose output
        results: List of (name, status, message, duration) tuples
    
    Example:
        >>> checker = PiscesLxEnvironmentChecker(verbose=True)
        >>> results = checker.run()
        >>> for name, status, msg, dur in results:
        ...     print(f"{name}: {status}")
    """
    
    REQUIRED_PYTHON_VERSION = (3, 10)
    MIN_PYTORCH_VERSION = "2.0.0"
    
    REQUIRED_PACKAGES = [
        "torch",
        "numpy",
        "tqdm",
        "pyyaml",
        "safetensors",
        "transformers",
        "tokenizers",
        "accelerate",
        "datasets",
        "huggingface_hub",
    ]
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the environment checker.
        
        Args:
            verbose: Enable verbose output for detailed logging
        """
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, float]] = []
    
    def run(self) -> List[Tuple[str, str, str, float]]:
        """
        Run all environment checks.
        
        Returns:
            List of (name, status, message, duration) tuples where:
            - name: Check name
            - status: "PASS", "FAIL", "WARN", or "SKIP"
            - message: Details or error message
            - duration: Execution time in seconds
        """
        self.results = []
        
        self._check_python_version()
        self._check_pytorch()
        self._check_cuda()
        self._check_gpu()
        self._check_system_memory()
        self._check_dependencies()
        
        return self.results
    
    def _add_result(self, name: str, status: str, message: str, duration: float) -> None:
        """Add a check result."""
        self.results.append((name, status, message, duration))
    
    def _check_python_version(self) -> None:
        """Check Python version meets minimum requirement."""
        start = time.time()
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major >= self.REQUIRED_PYTHON_VERSION[0] and \
           (version.major > self.REQUIRED_PYTHON_VERSION[0] or 
            version.minor >= self.REQUIRED_PYTHON_VERSION[1]):
            self._add_result("Python", "PASS", version_str, time.time() - start)
        else:
            required = f"{self.REQUIRED_PYTHON_VERSION[0]}.{self.REQUIRED_PYTHON_VERSION[1]}+"
            self._add_result("Python", "FAIL", f"{version_str} (need {required})", time.time() - start)
    
    def _check_pytorch(self) -> None:
        """Check PyTorch installation and version."""
        start = time.time()
        
        if not TORCH_AVAILABLE:
            self._add_result("PyTorch", "FAIL", "Not installed", time.time() - start)
            return
        
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_suffix = "+cu" if cuda_available else ""
        
        self._add_result("PyTorch", "PASS", f"{version}{cuda_suffix}", time.time() - start)
    
    def _check_cuda(self) -> None:
        """Check CUDA availability and version."""
        start = time.time()
        
        if not TORCH_AVAILABLE:
            self._add_result("CUDA", "SKIP", "PyTorch not available", time.time() - start)
            return
        
        if not torch.cuda.is_available():
            self._add_result("CUDA", "WARN", "Not available (CPU mode)", time.time() - start)
            return
        
        cuda_version = torch.version.cuda
        self._add_result("CUDA", "PASS", f"v{cuda_version}", time.time() - start)
    
    def _check_gpu(self) -> None:
        """Check GPU hardware and memory."""
        start = time.time()
        
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            self._add_result("GPU", "SKIP", "CUDA not available", time.time() - start)
            return
        
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            self._add_result("GPU", "WARN", "No GPU detected", time.time() - start)
            return
        
        gpu_info = []
        for i in range(min(gpu_count, 4)):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)
            gpu_info.append(f"{props.name} ({memory_gb:.0f}GB)")
        
        if gpu_count > 4:
            gpu_info.append(f"... +{gpu_count - 4} more")
        
        self._add_result("GPU", "PASS", ", ".join(gpu_info), time.time() - start)
    
    def _check_system_memory(self) -> None:
        """Check system RAM."""
        start = time.time()
        
        if not PSUTIL_AVAILABLE:
            self._add_result("Memory", "SKIP", "psutil not available", time.time() - start)
            return
        
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        
        status = "PASS" if total_gb >= 16 else "WARN"
        self._add_result("Memory", status, f"{total_gb:.0f}GB total, {available_gb:.0f}GB available", time.time() - start)
    
    def _check_dependencies(self) -> None:
        """Check required Python packages."""
        start = time.time()
        
        missing = []
        for package in self.REQUIRED_PACKAGES:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
        
        if missing:
            self._add_result("Dependencies", "WARN", f"Missing: {', '.join(missing)}", time.time() - start)
        else:
            self._add_result("Dependencies", "PASS", f"{len(self.REQUIRED_PACKAGES)} packages", time.time() - start)
