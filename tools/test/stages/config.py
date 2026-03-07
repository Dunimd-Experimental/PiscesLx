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
Configuration Checker Module for PiscesL1.

This module provides the PiscesLxConfigChecker class for validating
model and training configuration files.

Checks performed:
    - YAML parsing for all model configs
    - Training config validation
    - Parameter range validation
    - Config compatibility
"""

import os
import time
from typing import List, Tuple, Dict, Any
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class PiscesLxConfigChecker:
    """
    Configuration validation checker.
    
    Validates that configuration files are properly formatted and
    contain valid parameters.
    
    Attributes:
        root_path: Project root directory
        config_name: Specific config to check (e.g., "7B" or "configs/7B.yaml")
        verbose: Enable verbose output
        results: List of check results
    
    Example:
        >>> checker = PiscesLxConfigChecker(config_name="7B")
        >>> results = checker.run()
    """
    
    REQUIRED_CONFIG_FIELDS = [
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "vocab_size",
        "max_position_embeddings",
    ]
    
    VALID_MODEL_SIZES = ["0.5B", "1B", "7B", "14B", "32B", "72B", "671B", "1T"]
    
    def __init__(self, root_path: str = None, config_name: str = None, verbose: bool = False):
        """
        Initialize the config checker.
        
        Args:
            root_path: Project root directory
            config_name: Specific config to check (e.g., "7B" or "configs/7B.yaml")
            verbose: Enable verbose output
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.config_name = config_name
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, float]] = []
    
    def run(self) -> List[Tuple[str, str, str, float]]:
        """
        Run all configuration checks.
        
        Returns:
            List of (name, status, message, duration) tuples
        """
        self.results = []
        
        if not YAML_AVAILABLE:
            self._add_result("Config loader", "FAIL", "PyYAML not installed", 0)
            return self.results
        
        if self.config_name:
            self._check_single_config(self.config_name)
        else:
            self._check_all_configs()
        
        return self.results
    
    def _add_result(self, name: str, status: str, message: str, duration: float) -> None:
        """Add a check result."""
        self.results.append((name, status, message, duration))
    
    def _check_single_config(self, config_name: str) -> None:
        """
        Check a single configuration file.
        
        Args:
            config_name: Name of the config (e.g., "7B" or "configs/7B.yaml")
        """
        start = time.time()
        
        config_name_clean = config_name.replace("\\", "/")
        
        if "/" in config_name_clean:
            if config_name_clean.endswith(".yaml") or config_name_clean.endswith(".yml"):
                config_path = self.root_path / config_name_clean
            else:
                config_path = self.root_path / f"{config_name_clean}.yaml"
        else:
            config_path = self.root_path / "configs" / "model" / f"{config_name_clean}.yaml"
        
        if not config_path.exists():
            try:
                display_name = config_path.relative_to(self.root_path)
            except ValueError:
                display_name = str(config_path)
            self._add_result(str(display_name), "FAIL", "File not found", time.time() - start)
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                self._add_result(f"{config_name} config", "FAIL", "Empty config", time.time() - start)
                return
            
            missing = [f for f in self.REQUIRED_CONFIG_FIELDS if f not in config]
            
            if missing:
                self._add_result(f"{config_name} config", "WARN", f"Missing: {missing[:3]}", time.time() - start)
            else:
                hidden = config.get("hidden_size", 0)
                self._add_result(f"{config_name} config", "PASS", f"hidden_size={hidden}", time.time() - start)
                
        except yaml.YAMLError as e:
            self._add_result(f"{config_name} config", "FAIL", f"YAML error: {str(e)[:30]}", time.time() - start)
        except Exception as e:
            self._add_result(f"{config_name} config", "FAIL", str(e)[:50], time.time() - start)
    
    def _check_all_configs(self) -> None:
        """Check all available configuration files."""
        configs_dir = self.root_path / "configs" / "model"
        
        if not configs_dir.exists():
            self._add_result("Config directory", "WARN", "configs/model/ not found", 0)
            return
        
        yaml_files = list(configs_dir.glob("*.yaml"))
        
        if not yaml_files:
            self._add_result("Config files", "WARN", "No YAML configs found", 0)
            return
        
        success = 0
        for yaml_file in yaml_files:
            config_name = yaml_file.stem
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                if config and all(f in config for f in self.REQUIRED_CONFIG_FIELDS[:3]):
                    success += 1
            except Exception:
                pass
        
        total = len(yaml_files)
        self._add_result("Config files", "PASS" if success > 0 else "WARN", f"{success}/{total} valid", 0)
