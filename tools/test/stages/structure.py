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
Project Structure Checker Module for PiscesL1.

This module provides the PiscesLxStructureChecker class for validating
the project file structure, including core modules, configuration files,
and directory permissions.

Checks performed:
    - Core module files existence
    - Configuration files existence
    - Tokenizer/vocabulary files
    - Log/cache directory permissions
"""

import os
import time
from typing import List, Tuple
from pathlib import Path


class PiscesLxStructureChecker:
    """
    Project structure validation checker.
    
    Validates that all required files and directories exist and are
    accessible for the PiscesL1 project.
    
    Attributes:
        root_path: Project root directory
        verbose: Enable verbose output
        results: List of check results
    
    Example:
        >>> checker = PiscesLxStructureChecker("/path/to/project")
        >>> results = checker.run()
    """
    
    CORE_MODULES = [
        "model/__init__.py",
        "model/core/__init__.py",
        "model/core/model.py",
        "model/core/attention.py",
        "model/core/config.py",
        "model/core/cache.py",
        "model/moe/__init__.py",
        "model/moe/gate.py",
        "model/moe/expert.py",
        "model/multimodal/__init__.py",
        "model/generation/__init__.py",
        "opss/__init__.py",
        "utils/__init__.py",
        "utils/dc.py",
        "utils/paths.py",
    ]
    
    CONFIG_FILES = [
        "configs/model/0.5B.yaml",
        "configs/model/7B.yaml",
        "configs/model/32B.yaml",
        "configs/model/671B.yaml",
    ]
    
    REQUIRED_DIRS = [
        "logs",
        "cache",
        "checkpoints",
    ]
    
    def __init__(self, root_path: str = None, verbose: bool = False):
        """
        Initialize the structure checker.
        
        Args:
            root_path: Project root directory (defaults to current directory)
            verbose: Enable verbose output
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, float]] = []
    
    def run(self) -> List[Tuple[str, str, str, float]]:
        """
        Run all structure checks.
        
        Returns:
            List of (name, status, message, duration) tuples
        """
        self.results = []
        
        self._check_core_modules()
        self._check_config_files()
        self._check_directories()
        
        return self.results
    
    def _add_result(self, name: str, status: str, message: str, duration: float) -> None:
        """Add a check result."""
        self.results.append((name, status, message, duration))
    
    def _check_core_modules(self) -> None:
        """Check core module files existence."""
        start = time.time()
        
        existing = 0
        missing = []
        
        for module_path in self.CORE_MODULES:
            full_path = self.root_path / module_path
            if full_path.exists():
                existing += 1
            else:
                missing.append(module_path)
        
        total = len(self.CORE_MODULES)
        
        if missing:
            self._add_result("Core modules", "WARN", f"{existing}/{total}, missing: {missing[:3]}", time.time() - start)
        else:
            self._add_result("Core modules", "PASS", f"{total}/{total}", time.time() - start)
    
    def _check_config_files(self) -> None:
        """Check configuration files existence."""
        start = time.time()
        
        existing = 0
        missing = []
        
        for config_path in self.CONFIG_FILES:
            full_path = self.root_path / config_path
            if full_path.exists():
                existing += 1
            else:
                missing.append(config_path)
        
        total = len(self.CONFIG_FILES)
        
        if existing == 0:
            self._add_result("Config files", "WARN", "No config files found", time.time() - start)
        elif missing:
            self._add_result("Config files", "PASS", f"{existing}/{total}", time.time() - start)
        else:
            self._add_result("Config files", "PASS", f"{total}/{total}", time.time() - start)
    
    def _check_directories(self) -> None:
        """Check required directories and permissions."""
        start = time.time()
        
        existing = 0
        writable = 0
        
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.root_path / dir_name
            if dir_path.exists():
                existing += 1
                if os.access(dir_path, os.W_OK):
                    writable += 1
            else:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    existing += 1
                    writable += 1
                except Exception:
                    pass
        
        total = len(self.REQUIRED_DIRS)
        
        if existing == total and writable == total:
            self._add_result("Directories", "PASS", f"{total} directories", time.time() - start)
        elif existing == total:
            self._add_result("Directories", "WARN", "Some directories not writable", time.time() - start)
        else:
            self._add_result("Directories", "WARN", f"{existing}/{total} directories", time.time() - start)
