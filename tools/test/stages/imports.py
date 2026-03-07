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
Module Import Checker for PiscesL1.

This module provides the PiscesLxImportChecker class for validating
that all project modules can be imported correctly.

Checks performed:
    - model.core.* imports
    - model.moe.* imports
    - model.multimodal.* imports
    - model.reasoning.* imports
    - model.generation.* imports
    - opss.* imports
    - tools.* imports
"""

import time
import importlib
from typing import List, Tuple, Dict


class PiscesLxImportChecker:
    """
    Module import validation checker.
    
    Validates that all required modules can be imported successfully.
    
    Attributes:
        verbose: Enable verbose output
        results: List of check results
    
    Example:
        >>> checker = PiscesLxImportChecker()
        >>> results = checker.run()
    """
    
    MODULE_GROUPS: Dict[str, List[str]] = {
        "model.core": [
            "model.core",
            "model.core.model",
            "model.core.config",
            "model.core.attention",
            "model.core.cache",
        ],
        "model.moe": [
            "model.moe",
            "model.moe.gate",
            "model.moe.expert",
        ],
        "model.multimodal": [
            "model.multimodal",
        ],
        "model.generation": [
            "model.generation",
        ],
        "opss": [
            "opss",
        ],
        "tools": [
            "tools",
        ],
        "utils": [
            "utils",
            "utils.dc",
            "utils.paths",
        ],
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the import checker.
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, float]] = []
    
    def run(self) -> List[Tuple[str, str, str, float]]:
        """
        Run all import checks.
        
        Returns:
            List of (name, status, message, duration) tuples
        """
        self.results = []
        
        for group_name, modules in self.MODULE_GROUPS.items():
            self._check_module_group(group_name, modules)
        
        return self.results
    
    def _add_result(self, name: str, status: str, message: str, duration: float) -> None:
        """Add a check result."""
        self.results.append((name, status, message, duration))
    
    def _check_module_group(self, group_name: str, modules: List[str]) -> None:
        """
        Check a group of modules.
        
        Args:
            group_name: Display name for the group
            modules: List of module paths to check
        """
        start = time.time()
        
        success = 0
        failed = []
        
        for module_path in modules:
            try:
                importlib.import_module(module_path)
                success += 1
            except Exception as e:
                failed.append(f"{module_path}: {str(e)[:30]}")
        
        total = len(modules)
        
        if success == total:
            self._add_result(group_name, "PASS", f"{total} modules", time.time() - start)
        elif success == 0:
            self._add_result(group_name, "FAIL", "All imports failed", time.time() - start)
        else:
            self._add_result(group_name, "WARN", f"{success}/{total}, failed: {failed[:2]}", time.time() - start)
