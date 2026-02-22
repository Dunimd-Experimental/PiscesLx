#!/usr/bin/env/python3
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

"""
Calibration Operator - Model Quantization Calibration Framework
Based on utils/quantization/calibration.py

This module provides calibration operators for model quantization, including
data loading and activation collection for optimal quantization parameters.

Key Components:
    - CalibrationConfig: Configuration for calibration settings
    - POPSSCalibrationDataLoaderOperator: Loads calibration datasets
    - POPSSActivationCollectorOperator: Collects layer activations for analysis

The calibration process is essential for determining optimal quantization
parameters (scale and zero-point) that minimize accuracy loss.

Example:
    >>> from opss.quantize.calibration import POPSSCalibrationDataLoaderOperator
    >>> loader = POPSSCalibrationDataLoaderOperator()
    >>> result = loader.execute({"config": CalibrationConfig()})
    >>> calibration_batches = result.output["calibration_batches"]
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, List, Union, Tuple
from pathlib import Path
from configs.version import VERSION
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus, PiscesLxOperatorConfig


class CalibrationConfig(PiscesLxOperatorConfig):
    """
    Configuration for quantization calibration.
    
    This configuration class defines settings for calibration data loading
    and activation collection during the quantization process.
    
    Attributes:
        name: Configuration identifier. Default: "quantize.calibration.config"
        dataset_name: Name of calibration dataset. Default: "wikitext"
        num_calibration_samples: Number of samples for calibration. Default: 128
        dataset_config: Optional dataset configuration string. Default: None
        custom_data_path: Path to custom calibration data. Default: None
        sequence_length: Maximum sequence length for tokenization. Default: 512
        batch_size: Batch size for calibration. Default: 1
        calibration_method: Method for calibration. Options: "mse", "percentile", "kl_divergence". Default: "mse"
        percentile: Percentile value for percentile method. Default: 99.99
    """
    name: str = "quantize.calibration.config"
    dataset_name: str = "wikitext"
    num_calibration_samples: int = 128
    dataset_config: Optional[str] = None
    custom_data_path: Optional[str] = None
    sequence_length: int = 512
    batch_size: int = 1
    calibration_method: str = "mse"
    percentile: float = 99.99


class POPSSCalibrationDataLoaderOperator(PiscesLxOperatorInterface):
    """
    Calibration Data Loading Operator.
    
    This operator loads and prepares calibration datasets for quantization.
    It supports multiple dataset sources including HuggingFace datasets
    and custom local files.
    
    Supported Datasets:
        - wikitext: Wikipedia-based text dataset
        - c4: Colossal Clean Crawled Corpus
        - pile: The Pile dataset
        - custom: User-provided data files
    
    Example:
        >>> loader = POPSSCalibrationDataLoaderOperator()
        >>> result = loader.execute({
        ...     "config": CalibrationConfig(
        ...         dataset_name="wikitext",
        ...         num_calibration_samples=256
        ...     )
        ... })
        >>> batches = result.output["calibration_batches"]
    """
    
    def __init__(self):
        """Initialize the calibration data loader operator."""
        super().__init__()
        self.name = "calibration_data_loader"
        self.version = VERSION
        self.supported_datasets = {
            "wikitext": ["wikitext-2-v1", "wikitext-103-v1"],
            "c4": ["en", "realnewslike"],
            "pile": ["plain_text"],
            "custom": []
        }
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Load calibration data.
        
        Loads and preprocesses calibration data from the specified dataset.
        
        Args:
            inputs: Dictionary containing configuration
                - config: Calibration configuration (CalibrationConfig)
        
        Returns:
            PiscesLxOperatorResult: Result containing
                - calibration_batches: List of tokenized batches
                - dataset_info: Information about loaded dataset
        """
        try:
            config = inputs.get("config", CalibrationConfig())
            
            # Load calibration data
            calibration_batches = self._load_calibration_data(config)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "calibration_batches": calibration_batches,
                    "dataset_info": {
                        "name": config.dataset_name,
                        "samples": len(calibration_batches),
                        "sequence_length": config.sequence_length
                    }
                },
                metadata={
                    "version": self.version,
                    "samples_loaded": len(calibration_batches)
                },
                execution_time=0.0,
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                metadata={
                    "version": self.version,
                    "error_type": type(e).__name__
                },
                execution_time=0.0,
            )
    
    def _load_calibration_data(self, config: CalibrationConfig) -> List[Dict[str, torch.Tensor]]:
        """
        Load calibration data from dataset.
        
        Loads data from HuggingFace datasets or custom files and
        tokenizes it for model input.
        
        Args:
            config: Calibration configuration
        
        Returns:
            List of tokenized batches with input_ids and attention_mask
        """
        batches = []
        
        try:
            # Try using HuggingFace datasets
            from datasets import load_dataset
            from model.tokenizer import YvTokenizer
            
            # Load dataset
            if config.dataset_name in self.supported_datasets:
                dataset_config = config.dataset_config or self.supported_datasets[config.dataset_name][0]
                dataset = load_dataset(config.dataset_name, dataset_config, split="train")
            else:
                # Try custom dataset
                if config.custom_data_path:
                    dataset = self._load_custom_dataset(config.custom_data_path)
                else:
                    raise ValueError(f"Unsupported dataset: {config.dataset_name}")
            
            # Prepare tokenizer
            tokenizer = YvTokenizer()
            
            # Process data samples
            processed_count = 0
            for item in dataset:
                if processed_count >= config.num_calibration_samples:
                    break
                
                # Get text content
                text = self._extract_text_from_item(item, config.dataset_name)
                if not text:
                    continue
                
                # Encode text
                ids = tokenizer.encode(text, return_tensors="pt")
                if isinstance(ids, torch.Tensor):
                    input_ids = ids
                else:
                    input_ids = torch.tensor([ids], dtype=torch.long)
                input_ids = input_ids[:, : int(config.sequence_length)]
                if input_ids.shape[1] < int(config.sequence_length):
                    pad_id = int(getattr(tokenizer, "pad_token_id", 0))
                    pad_len = int(config.sequence_length) - int(input_ids.shape[1])
                    input_ids = torch.cat(
                        [input_ids, torch.full((input_ids.shape[0], pad_len), pad_id, dtype=torch.long)],
                        dim=1,
                    )
                attention_mask = (input_ids != int(getattr(tokenizer, "pad_token_id", 0))).to(torch.long)
                
                batches.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                })
                
                processed_count += 1
                
        except ImportError as e:
            raise RuntimeError(f"Missing dependency for calibration data loading: {e}")
        except Exception as e:
            raise
        
        return batches
    
    def _extract_text_from_item(self, item: Dict, dataset_name: str) -> str:
        """
        Extract text from data item.
        
        Handles different dataset formats to extract the text content.
        
        Args:
            item: Data item from dataset
            dataset_name: Name of the dataset
        
        Returns:
            Extracted text string
        """
        if "wikitext" in dataset_name:
            return item.get("text", "")
        elif "c4" in dataset_name:
            return item.get("text", "")
        elif "pile" in dataset_name:
            return item.get("text", "")
        else:
            # Try common field names
            for field in ["text", "content", "sentence", "paragraph"]:
                if field in item:
                    return str(item[field])
            return str(item)  # Final fallback
    
    def _load_custom_dataset(self, data_path: str):
        """
        Load custom dataset from file.
        
        Supports JSON and TXT file formats.
        
        Args:
            data_path: Path to custom data file
        
        Returns:
            Loaded dataset as list of items
        """
        path = Path(data_path)
        if path.suffix == ".json":
            import json
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        elif path.suffix == ".txt":
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return [{"text": line.strip()} for line in lines if line.strip()]
        else:
            raise ValueError(f"Unsupported custom data format: {path.suffix}")


class POPSSActivationCollectorOperator(PiscesLxOperatorInterface):
    """
    Activation Collection Operator.
    
    This operator collects activation statistics from model layers during
    forward passes with calibration data. These statistics are used to
    determine optimal quantization parameters.
    
    The collected statistics include:
        - min: Minimum activation value
        - max: Maximum activation value
        - mean: Mean activation value
        - std: Standard deviation of activations
    
    Example:
        >>> collector = POPSSActivationCollectorOperator()
        >>> result = collector.execute({
        ...     "model": model,
        ...     "calibration_data": calibration_batches
        ... })
        >>> stats = result.output["activation_stats"]
    """
    
    def __init__(self):
        """Initialize the activation collector operator."""
        super().__init__()
        self.name = "activation_collector"
        self.version = VERSION
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Collect layer activations for calibration.
        
        Runs forward passes with calibration data and collects activation
        statistics from specified or all layers.
        
        Args:
            inputs: Dictionary containing collection inputs
                - model: Model to collect activations from (nn.Module)
                - calibration_data: Calibration data batches
                - target_layers: Optional list of layer names to monitor
        
        Returns:
            PiscesLxOperatorResult: Result containing
                - activation_stats: Statistics for each monitored layer
                - collected_layers: List of monitored layer names
                - total_samples: Number of samples processed
        """
        try:
            model = inputs.get("model")
            calibration_data = inputs.get("calibration_data", [])
            target_layers = inputs.get("target_layers")
            
            if not model or not calibration_data:
                raise ValueError("Model and calibration data are required")
            
            # Collect activations
            activation_stats = self._collect_activations(model, calibration_data, target_layers)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "activation_stats": activation_stats,
                    "collected_layers": list(activation_stats.keys()),
                    "total_samples": len(calibration_data)
                },
                metadata={
                    "version": self.version,
                    "layers_monitored": len(activation_stats)
                },
                execution_time=0.0,
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                metadata={
                    "version": self.version,
                    "error_type": type(e).__name__
                },
                execution_time=0.0,
            )
    
    def _collect_activations(self, 
                           model: nn.Module, 
                           calibration_data: List[Dict],
                           target_layers: Optional[List[str]] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Collect activations from model layers.
        
        Registers forward hooks on target layers and collects activation
        statistics during forward passes.
        
        Args:
            model: Neural network model
            calibration_data: Calibration data batches
            target_layers: Optional list of layer names to monitor
        
        Returns:
            Dictionary mapping layer names to activation statistics
        """
        activation_stats = {}
        hooks = []
        
        def register_hooks(module, name):
            def hook_fn(module, input, output):
                if name not in activation_stats:
                    activation_stats[name] = {
                        "min_values": [],
                        "max_values": [],
                        "mean_values": [],
                        "std_values": []
                    }
                
                # Handle different output types
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                elif isinstance(output, tuple) and len(output) > 0:
                    act = output[0].detach() if isinstance(output[0], torch.Tensor) else None
                else:
                    return
                
                if act is not None:
                    activation_stats[name]["min_values"].append(act.min().item())
                    activation_stats[name]["max_values"].append(act.max().item())
                    activation_stats[name]["mean_values"].append(act.mean().item())
                    activation_stats[name]["std_values"].append(act.std().item())
            
            # Register hook for Linear and Conv layers
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Register hooks
        for name, module in model.named_modules():
            if target_layers is None or name in target_layers:
                register_hooks(module, name)
        
        # Run forward passes to collect data
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch in calibration_data:
                # Move data to device
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                # Forward pass
                try:
                    _ = model(**batch)
                except Exception:
                    # Simplified input format attempt
                    if "input_ids" in batch:
                        _ = model(input_ids=batch["input_ids"])
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Compute summary statistics
        for layer_name in activation_stats:
            stats = activation_stats[layer_name]
            activation_stats[layer_name] = {
                "min": min(stats["min_values"]) if stats["min_values"] else 0,
                "max": max(stats["max_values"]) if stats["max_values"] else 1,
                "mean": sum(stats["mean_values"]) / len(stats["mean_values"]) if stats["mean_values"] else 0,
                "std": sum(stats["std_values"]) / len(stats["std_values"]) if stats["std_values"] else 0,
                "sample_count": len(stats["min_values"])
            }
        
        return activation_stats


CalibrationDataLoaderOperator = POPSSCalibrationDataLoaderOperator
ActivationCollectorOperator = POPSSActivationCollectorOperator
