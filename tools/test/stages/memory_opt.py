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
Memory Optimization Checker Module for PiscesL1.

This module provides the PiscesLxMemoryOptChecker class for validating
memory optimization techniques are correctly configured.

Checks performed:
    - GaLore gradient projection
    - Gradient checkpointing
    - Mixed precision (BF16/FP16)
    - QLoRA/quantization
    - MoE gradient optimization
"""

import os
import time
from typing import List, Tuple, Dict, Any
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class PiscesLxMemoryOptChecker:
    """
    Memory optimization configuration checker.
    
    Validates that memory optimization techniques are correctly
    configured in model and training configurations.
    
    Attributes:
        root_path: Project root directory
        config_name: Model configuration name
        verbose: Enable verbose output
        results: List of check results
    
    Example:
        >>> checker = PiscesLxMemoryOptChecker(config_name="7B")
        >>> results = checker.run()
    """
    
    def __init__(self, root_path: str = None, config_name: str = "7B", verbose: bool = False):
        """
        Initialize the memory optimization checker.
        
        Args:
            root_path: Project root directory
            config_name: Model configuration name
            verbose: Enable verbose output
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.config_name = config_name
        self.verbose = verbose
        self.results: List[Tuple[str, str, str, float]] = []
        self._config: Dict[str, Any] = {}
    
    def run(self) -> List[Tuple[str, str, str, float]]:
        """
        Run all memory optimization checks.
        
        Returns:
            List of (name, status, message, duration) tuples
        """
        self.results = []
        
        if not YAML_AVAILABLE:
            self._add_result("Config loader", "FAIL", "PyYAML not installed", 0)
            return self.results
        
        self._load_config()
        
        if not self._config:
            self._add_result("Config", "FAIL", "Failed to load config", 0)
            return self.results
        
        self._check_galore()
        self._check_gradient_checkpointing()
        self._check_mixed_precision()
        self._check_quantization()
        self._check_lora()
        self._check_moe_optimization()
        
        return self.results
    
    def _add_result(self, name: str, status: str, message: str, duration: float) -> None:
        """Add a check result."""
        self.results.append((name, status, message, duration))
    
    def _load_config(self) -> None:
        """Load model configuration."""
        config_path = self.root_path / "configs" / "model" / f"{self.config_name}.yaml"
        
        if not config_path.exists():
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
        except Exception:
            pass
    
    def _check_galore(self) -> None:
        """Check GaLore gradient projection configuration."""
        start = time.time()
        
        galore_enabled = self._config.get("galore_enabled", False)
        galore_rank = self._config.get("galore_rank", 128)
        
        if galore_enabled:
            self._add_result("GaLore", "PASS", f"enabled, rank={galore_rank}", time.time() - start)
        else:
            self._add_result("GaLore", "WARN", "disabled (enable for 50%+ memory savings)", time.time() - start)
    
    def _check_gradient_checkpointing(self) -> None:
        """Check gradient checkpointing configuration."""
        start = time.time()
        
        training_cfg = self._config.get("training_config", {})
        gc_enabled = training_cfg.get("gradient_checkpointing", False)
        
        if gc_enabled:
            self._add_result("Gradient Checkpointing", "PASS", "enabled (30-50% memory savings)", time.time() - start)
        else:
            self._add_result("Gradient Checkpointing", "WARN", "disabled (enable for large models)", time.time() - start)
    
    def _check_mixed_precision(self) -> None:
        """Check mixed precision configuration."""
        start = time.time()
        
        training_cfg = self._config.get("training_config", {})
        mp = training_cfg.get("mixed_precision", "no")
        
        if mp in ["bf16", "fp16"]:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                if mp == "bf16":
                    bf16_ok = False
                    try:
                        bf16_ok = bool(torch.cuda.is_bf16_supported())
                    except Exception:
                        pass
                    
                    if bf16_ok:
                        self._add_result("Mixed Precision", "PASS", "bf16 enabled (50% memory savings)", time.time() - start)
                    else:
                        self._add_result("Mixed Precision", "WARN", "bf16 not supported, will fallback to fp16", time.time() - start)
                else:
                    self._add_result("Mixed Precision", "PASS", "fp16 enabled (50% memory savings)", time.time() - start)
            else:
                self._add_result("Mixed Precision", "PASS", f"{mp} configured", time.time() - start)
        else:
            self._add_result("Mixed Precision", "WARN", "disabled (enable bf16/fp16 for 50% savings)", time.time() - start)
    
    def _check_quantization(self) -> None:
        """Check quantization configuration."""
        start = time.time()
        
        inference_cfg = self._config.get("inference_config", {})
        quant = inference_cfg.get("force_quant", None)
        
        if quant:
            self._add_result("Quantization", "PASS", f"{quant} enabled (60-75% memory savings)", time.time() - start)
        else:
            self._add_result("Quantization", "INFO", "not configured (optional for training)", time.time() - start)
    
    def _check_lora(self) -> None:
        """Check LoRA configuration."""
        start = time.time()
        
        lora_cfg = self._config.get("lora", {})
        lora_enabled = lora_cfg.get("enabled", False) if isinstance(lora_cfg, dict) else False
        
        if lora_enabled:
            lora_r = lora_cfg.get("r", 8)
            self._add_result("LoRA", "PASS", f"enabled, r={lora_r} (trainable params reduced)", time.time() - start)
        else:
            self._add_result("LoRA", "INFO", "disabled (optional for fine-tuning)", time.time() - start)
    
    def _check_moe_optimization(self) -> None:
        """Check MoE gradient optimization configuration."""
        start = time.time()
        
        training_cfg = self._config.get("training_config", {})
        moe_cfg = training_cfg.get("moe_gradient", {})
        moe_enabled = moe_cfg.get("enabled", False) if isinstance(moe_cfg, dict) else False
        
        moe_num_experts = self._config.get("moe_num_experts", 0)
        
        if moe_num_experts > 0:
            if moe_enabled:
                self._add_result("MoE Optimization", "PASS", f"enabled for {moe_num_experts} experts", time.time() - start)
            else:
                self._add_result("MoE Optimization", "WARN", f"MoE model ({moe_num_experts} experts) but optimization disabled", time.time() - start)
        else:
            self._add_result("MoE Optimization", "INFO", "not a MoE model", time.time() - start)
