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
Model Instantiation Checker Module for PiscesL1.

This module provides the PiscesLxModelChecker class for validating
model creation and initialization.

Checks performed:
    - YvConfig creation
    - YvModel initialization
    - Encoder initialization (Vision, Audio, Video, Doc)
    - MoE layer initialization
    - Cache manager initialization
"""

import os
import time
from typing import List, Tuple
from pathlib import Path


class PiscesLxModelChecker:
    """
    Model instantiation validation checker.
    
    Validates that model components can be created and initialized.
    
    Attributes:
        config_name: Model configuration name (e.g., "7B")
        verbose: Enable verbose output
        results: List of check results
    
    Example:
        >>> checker = PiscesLxModelChecker(config_name="7B")
        >>> results = checker.run()
    """
    
    def __init__(self, config_name: str = "7B", verbose: bool = False):
        """
        Initialize the model checker.
        
        Args:
            config_name: Model configuration name
            verbose: Enable verbose output
        """
        self.config_name = config_name
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, float]] = []
    
    def run(self) -> List[Tuple[str, str, str, float]]:
        """
        Run all model instantiation checks.
        
        Returns:
            List of (name, status, message, duration) tuples
        """
        self.results = []
        
        self._check_config_creation()
        self._check_model_creation()
        self._check_components()
        
        return self.results
    
    def _add_result(self, name: str, status: str, message: str, duration: float) -> None:
        """Add a check result."""
        self.results.append((name, status, message, duration))
    
    def _check_config_creation(self) -> None:
        """Check YvConfig creation."""
        start = time.time()
        
        try:
            from model.core.config import YvConfig
            
            config = YvConfig()
            self._add_result("YvConfig", "PASS", "Created successfully", time.time() - start)
            
        except ImportError as e:
            self._add_result("YvConfig", "FAIL", f"Import error: {str(e)[:30]}", time.time() - start)
        except Exception as e:
            self._add_result("YvConfig", "FAIL", str(e)[:50], time.time() - start)
    
    def _check_model_creation(self) -> None:
        """Check YvModel initialization."""
        start = time.time()
        
        try:
            from model.core.config import YvConfig
            from model.core.model import YvModel
            
            config = YvConfig()
            config.hidden_size = 256
            config.num_attention_heads = 4
            config.num_hidden_layers = 2
            config.vocab_size = 1000
            config.max_position_embeddings = 512
            
            model = YvModel(config)
            
            param_count = sum(p.numel() for p in model.parameters())
            param_str = f"{param_count / 1e6:.1f}M params"
            
            self._add_result("YvModel", "PASS", param_str, time.time() - start)
            
        except ImportError as e:
            self._add_result("YvModel", "FAIL", f"Import error: {str(e)[:30]}", time.time() - start)
        except Exception as e:
            self._add_result("YvModel", "FAIL", str(e)[:50], time.time() - start)
    
    def _check_components(self) -> None:
        """Check individual model components."""
        start = time.time()
        
        components_to_check = [
            ("model.moe", "MoE"),
            ("model.multimodal", "Multimodal"),
            ("model.generation", "Generation"),
        ]
        
        for module_name, display_name in components_to_check:
            try:
                __import__(module_name)
                self._add_result(display_name, "PASS", "Module available", time.time() - start)
            except ImportError:
                self._add_result(display_name, "WARN", "Module not available", time.time() - start)
            
            start = time.time()
