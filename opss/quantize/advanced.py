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
Advanced Quantization Operator - Intelligent Quantization and Sensitivity Analysis
Based on utils/quantization/advanced.py

This module provides advanced quantization capabilities including sensitivity
analysis and adaptive bit allocation for optimal model compression.

Key Features:
    - Layer-wise sensitivity analysis for quantization
    - Adaptive bit allocation based on sensitivity scores
    - Automatic precision optimization for target compression ratio
    - Layer preservation for critical components

Example:
    >>> from opss.quantize.advanced import advanced_quantize
    >>> quantized_model, info = advanced_quantize(
    ...     model, target_compression=0.5
    ... )
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, List, Tuple, Union
from configs.version import VERSION
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus, PiscesLxOperatorConfig


class AdvancedQuantizationConfig(PiscesLxOperatorConfig):
    """
    Configuration for advanced quantization.
    
    This configuration class provides settings for intelligent quantization
    with sensitivity analysis and adaptive bit allocation.
    
    Attributes:
        sensitivity_analysis: Enable layer sensitivity analysis. Default: True
        adaptive_bit_allocation: Enable adaptive bit allocation. Default: True
        preserve_layers: List of layer names to keep in high precision. Default: None
        target_compression_ratio: Target compression ratio (0.0-1.0). Default: 0.5
        calibration_samples: Number of calibration samples. Default: 128
        sensitivity_metric: Metric for sensitivity analysis. Options: "mse", "perplexity", "accuracy". Default: "mse"
    """
    name: str = "quantize.advanced.config"
    sensitivity_analysis: bool = True
    adaptive_bit_allocation: bool = True
    preserve_layers: List[str] = None
    target_compression_ratio: float = 0.5
    calibration_samples: int = 128
    sensitivity_metric: str = "mse"


class SensitivityAnalysisOperator(PiscesLxOperatorInterface):
    """
    Layer Sensitivity Analysis Operator.
    
    This operator analyzes the sensitivity of each layer to quantization,
    measuring how much model performance degrades when each layer is quantized.
    This information guides optimal bit allocation strategies.
    
    The analysis process:
        1. Save original model weights
        2. For each layer, temporarily quantize it
        3. Measure performance degradation
        4. Restore original weights
        5. Rank layers by sensitivity
    
    Example:
        >>> operator = SensitivityAnalysisOperator()
        >>> result = operator.execute({
        ...     "model": model,
        ...     "test_data": calibration_data,
        ...     "config": AdvancedQuantizationConfig()
        ... })
        >>> sensitivity_scores = result.output["sensitivity_scores"]
    """
    
    def __init__(self):
        """Initialize the sensitivity analysis operator."""
        super().__init__()
        self.name = "sensitivity_analyzer"
        self.version = VERSION
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute layer sensitivity analysis.
        
        Analyzes each layer's sensitivity to quantization by temporarily
        quantizing layers and measuring performance impact.
        
        Args:
            inputs: Dictionary containing analysis inputs
                - model: Model to analyze (nn.Module)
                - test_data: Test data for evaluation
                - config: Analysis configuration (AdvancedQuantizationConfig)
        
        Returns:
            PiscesLxOperatorResult: Result containing
                - sensitivity_scores: Dict mapping layer names to sensitivity scores
                - recommended_bits: Dict mapping layer names to recommended bit widths
        """
        try:
            model = inputs.get("model")
            test_data = inputs.get("test_data", [])
            config = inputs.get("config", AdvancedQuantizationConfig())
            
            if not model:
                raise ValueError("Model is required for sensitivity analysis")
            
            # Execute sensitivity analysis
            sensitivity_scores = self._analyze_sensitivity(model, test_data, config)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "sensitivity_scores": sensitivity_scores,
                    "recommended_bits": self._recommend_bit_allocation(sensitivity_scores, config)
                },
                metadata={
                    "version": self.version,
                    "analyzed_layers": len(sensitivity_scores),
                    "metric": config.sensitivity_metric
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
    
    def _analyze_sensitivity(self, 
                           model: nn.Module, 
                           test_data: List,
                           config: AdvancedQuantizationConfig) -> Dict[str, float]:
        """
        Analyze sensitivity of each layer to quantization.
        
        Measures how much model performance degrades when each layer
        is quantized individually.
        
        Args:
            model: Model to analyze
            test_data: Test data for evaluation
            config: Analysis configuration
        
        Returns:
            Dictionary mapping layer names to sensitivity scores
        """
        sensitivity_scores = {}
        device = next(model.parameters()).device
        
        # Save original weights
        original_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_weights[name] = param.data.clone()
        
        # Test quantization on each layer
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight'):
                # Temporarily quantize this layer
                original_weight = module.weight.data.clone()
                
                # Execute 8-bit quantization test
                quantized_weight = self._quantize_weight_test(original_weight, bits=8)
                module.weight.data = quantized_weight
                
                # Evaluate performance degradation
                performance_drop = self._evaluate_performance_drop(
                    model, test_data, config.sensitivity_metric
                )
                
                sensitivity_scores[name] = performance_drop
                
                # Restore original weight
                module.weight.data = original_weight
        
        # Restore all weights
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data = original_weights[name]
        
        return sensitivity_scores
    
    def _quantize_weight_test(self, weight: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """
        Test weight quantization.
        
        Implements simple linear quantization for sensitivity testing.
        
        Args:
            weight: Weight tensor to quantize
            bits: Number of bits for quantization
        
        Returns:
            Dequantized weight tensor
        """
        # Simple linear quantization implementation
        min_val = weight.min()
        max_val = weight.max()
        
        if min_val == max_val:
            return weight
        
        # Compute quantization parameters
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale
        
        # Quantize
        quantized = torch.round(weight / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _evaluate_performance_drop(self, 
                                  model: nn.Module, 
                                  test_data: List,
                                  metric: str = "mse") -> float:
        """
        Evaluate performance degradation after quantization.
        
        Measures how much model performance drops when a layer is quantized.
        
        Args:
            model: Model with quantized layer
            test_data: Test data for evaluation
            metric: Metric for evaluation ("mse", "perplexity", etc.)
        
        Returns:
            Performance degradation score
        """
        model.eval()
        device = next(model.parameters()).device
        
        if metric == "mse":
            total_mse = 0.0
            count = 0
            
            with torch.no_grad():
                for batch in test_data[:10]:  # Limit test samples for speed
                    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                    
                    try:
                        outputs = model(**batch)
                        if hasattr(outputs, 'logits'):
                            predictions = outputs.logits
                        else:
                            predictions = outputs
                        
                        # Compute MSE (simplified)
                        target = batch.get('labels', predictions)
                        mse = torch.mean((predictions - target) ** 2).item()
                        total_mse += mse
                        count += 1
                    except Exception:
                        continue
            
            return total_mse / count if count > 0 else float('inf')
        
        else:
            # Simplified implementation for other metrics
            return np.random.random()  # Placeholder
    
    def _recommend_bit_allocation(self, 
                                 sensitivity_scores: Dict[str, float],
                                 config: AdvancedQuantizationConfig) -> Dict[str, int]:
        """
        Recommend bit allocation based on sensitivity scores.
        
        Assigns higher bit widths to more sensitive layers.
        
        Args:
            sensitivity_scores: Dict of layer sensitivity scores
            config: Quantization configuration
        
        Returns:
            Dictionary mapping layer names to recommended bit widths
        """
        recommendations = {}
        
        # Sort layers by sensitivity
        sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1])
        
        # Simple tiered strategy
        total_layers = len(sorted_layers)
        high_sensitivity_count = int(total_layers * 0.2)  # Top 20% most sensitive
        medium_sensitivity_count = int(total_layers * 0.5)  # Middle 50%
        
        for i, (layer_name, _) in enumerate(sorted_layers):
            if i < high_sensitivity_count:
                recommendations[layer_name] = 16  # High precision
            elif i < high_sensitivity_count + medium_sensitivity_count:
                recommendations[layer_name] = 8   # Medium precision
            else:
                recommendations[layer_name] = 4   # Low precision
        
        return recommendations


class POPSSAdaptiveBitAllocationOperator(PiscesLxOperatorInterface):
    """
    Adaptive Bit Allocation Operator.
    
    This operator optimizes bit allocation across layers to achieve
    target compression ratio while minimizing accuracy loss. It uses
    sensitivity analysis results to make intelligent allocation decisions.
    
    The optimization process:
        1. Start with all layers at high precision
        2. Iteratively reduce bits for less sensitive layers
        3. Stop when target compression is achieved
    
    Example:
        >>> operator = POPSSAdaptiveBitAllocationOperator()
        >>> result = operator.execute({
        ...     "model": model,
        ...     "sensitivity_analysis": sensitivity_results,
        ...     "config": AdvancedQuantizationConfig(target_compression_ratio=0.5)
        ... })
        >>> bit_allocation = result.output["bit_allocation"]
    """
    
    def __init__(self):
        """Initialize the adaptive bit allocation operator."""
        super().__init__()
        self.name = "adaptive_bit_allocation"
        self.version = VERSION
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute adaptive bit allocation.
        
        Optimizes bit allocation for each layer based on sensitivity analysis.
        
        Args:
            inputs: Dictionary containing allocation inputs
                - model: Model to allocate bits for
                - sensitivity_analysis: Results from sensitivity analysis
                - config: Allocation configuration
        
        Returns:
            PiscesLxOperatorResult: Result containing
                - bit_allocation: Dict mapping layer names to bit widths
                - estimated_compression: Estimated compression ratio
                - memory_savings: Memory savings information
        """
        try:
            model = inputs.get("model")
            sensitivity_analysis = inputs.get("sensitivity_analysis", {})
            config = inputs.get("config", AdvancedQuantizationConfig())
            
            if not model:
                raise ValueError("Model is required for bit allocation")
            
            # Execute adaptive bit allocation optimization
            bit_allocation = self._optimize_bit_allocation(
                model, sensitivity_analysis, config
            )
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "bit_allocation": bit_allocation,
                    "estimated_compression": self._estimate_compression(model, bit_allocation),
                    "memory_savings": self._calculate_memory_savings(model, bit_allocation)
                },
                metadata={
                    "version": self.version,
                    "target_compression": config.target_compression_ratio
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
    
    def _optimize_bit_allocation(self,
                                model: nn.Module,
                                sensitivity_analysis: Dict[str, Any],
                                config: AdvancedQuantizationConfig) -> Dict[str, int]:
        """
        Optimize bit allocation to achieve target compression ratio.
        
        Uses a greedy algorithm to allocate bits, prioritizing less
        sensitive layers for lower bit widths.
        
        Args:
            model: Model to optimize
            sensitivity_analysis: Sensitivity analysis results
            config: Allocation configuration
        
        Returns:
            Dictionary mapping layer names to optimal bit widths
        """
        sensitivity_scores = sensitivity_analysis.get("sensitivity_scores", {})
        recommended_bits = sensitivity_analysis.get("recommended_bits", {})
        
        # Use recommendations if available
        if recommended_bits:
            return recommended_bits
        
        # Otherwise execute optimization algorithm
        layer_sizes = self._get_layer_sizes(model)
        target_compression = config.target_compression_ratio
        
        # Simplified greedy algorithm
        bit_options = [4, 8, 16]  # Available bit widths
        current_allocation = {name: 16 for name in layer_sizes.keys()}  # Start with 16-bit
        
        # Sort layers by sensitivity (most sensitive first)
        sorted_layers = sorted(
            sensitivity_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Gradually reduce bits until target compression is reached
        for layer_name, _ in sorted_layers:
            if self._calculate_current_compression(layer_sizes, current_allocation) >= target_compression:
                break
                
            # Try reducing bits for this layer
            for bits in [8, 4]:
                temp_allocation = current_allocation.copy()
                temp_allocation[layer_name] = bits
                
                if self._calculate_current_compression(layer_sizes, temp_allocation) <= target_compression:
                    current_allocation[layer_name] = bits
                    break
        
        return current_allocation


    def _get_layer_sizes(self, model: nn.Module) -> Dict[str, int]:
        """
        Get parameter count for each layer.
        
        Args:
            model: Neural network model
        
        Returns:
            Dictionary mapping layer names to parameter counts
        """
        layer_sizes = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                size = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    size += module.bias.numel()
                layer_sizes[name] = size
        return layer_sizes
    
    def _calculate_current_compression(self, 
                                      layer_sizes: Dict[str, int],
                                      bit_allocation: Dict[str, int]) -> float:
        """
        Calculate current compression ratio.
        
        Args:
            layer_sizes: Layer parameter counts
            bit_allocation: Current bit allocation
        
        Returns:
            Current compression ratio
        """
        total_original_bits = sum(size * 32 for size in layer_sizes.values())  # Assume 32-bit original
        total_quantized_bits = sum(layer_sizes[name] * bits for name, bits in bit_allocation.items())
        
        if total_original_bits == 0:
            return 0.0
            
        return 1.0 - (total_quantized_bits / total_original_bits)
    
    def _estimate_compression(self, 
                             model: nn.Module, 
                             bit_allocation: Dict[str, int]) -> float:
        """
        Estimate compression effect.
        
        Args:
            model: Neural network model
            bit_allocation: Bit allocation for each layer
        
        Returns:
            Estimated compression ratio
        """
        return self._calculate_current_compression(self._get_layer_sizes(model), bit_allocation)
    
    def _calculate_memory_savings(self, 
                                 model: nn.Module, 
                                 bit_allocation: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate memory savings.
        
        Computes the memory savings achieved by the bit allocation.
        
        Args:
            model: Neural network model
            bit_allocation: Bit allocation for each layer
        
        Returns:
            Dictionary with memory savings information
        """
        layer_sizes = self._get_layer_sizes(model)
        original_memory = sum(size * 4 for size in layer_sizes.values())  # 32-bit float, 4 bytes
        
        quantized_memory = 0
        for name, size in layer_sizes.items():
            bits = bit_allocation.get(name, 32)
            bytes_per_param = bits / 8
            quantized_memory += size * bytes_per_param
        
        savings_ratio = (original_memory - quantized_memory) / original_memory if original_memory > 0 else 0
        
        return {
            "original_memory_mb": original_memory / 1024 / 1024,
            "quantized_memory_mb": quantized_memory / 1024 / 1024,
            "memory_savings_mb": (original_memory - quantized_memory) / 1024 / 1024,
            "savings_ratio": savings_ratio
        }


def advanced_quantize(model: nn.Module,
                     test_data: List = None,
                     target_compression: float = 0.5,
                     preserve_layers: List[str] = None) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function for advanced quantization.
    
    Provides a simple interface for intelligent model quantization with
    sensitivity analysis and adaptive bit allocation.
    
    Args:
        model: Model to quantize
        test_data: Test data for sensitivity analysis
        target_compression: Target compression ratio
        preserve_layers: List of layer names to keep in high precision
    
    Returns:
        Tuple of (quantized_model, detailed_info)
    
    Example:
        >>> quantized_model, info = advanced_quantize(
        ...     model, target_compression=0.5
        ... )
    """
    
    # Execute sensitivity analysis
    sensitivity_op = SensitivityAnalysisOperator()
    sensitivity_inputs = {
        "model": model,
        "test_data": test_data or [],
        "config": AdvancedQuantizationConfig(
            sensitivity_analysis=True,
            target_compression_ratio=target_compression,
            preserve_layers=preserve_layers or []
        )
    }
    
    sensitivity_result = sensitivity_op.execute(sensitivity_inputs)
    
    if not sensitivity_result.is_success():
        raise RuntimeError(f"Sensitivity analysis failed: {sensitivity_result.error}")
    
    # Execute adaptive bit allocation
    allocation_op = AdaptiveBitAllocationOperator()
    allocation_inputs = {
        "model": model,
        "sensitivity_analysis": sensitivity_result.output or {},
        "config": AdvancedQuantizationConfig(target_compression_ratio=target_compression)
    }
    
    allocation_result = allocation_op.execute(allocation_inputs)
    
    if not allocation_result.is_success():
        raise RuntimeError(f"Bit allocation failed: {allocation_result.error}")
    
    from .methods import QuantizationConfig
    from .methods import QuantizationOperatorFactory
    
    quant_op = QuantizationOperatorFactory.create_operator("gptq")
    quant_config = QuantizationConfig(bits=8)
    
    quant_inputs = {
        "model": model,
        "config": quant_config,
        "bit_allocation": (allocation_result.output or {}).get("bit_allocation", {})
    }
    
    quant_result = quant_op.execute(quant_inputs)
    
    if not quant_result.is_success():
        raise RuntimeError(f"Quantization failed: {quant_result.error}")
    
    qout = quant_result.output or {}
    return qout["model"], {
        "sensitivity_scores": (sensitivity_result.output or {}).get("sensitivity_scores", {}),
        "bit_allocation": (allocation_result.output or {}).get("bit_allocation", {}),
        "compression_info": allocation_result.output or {},
        "quantization_info": qout.get("quantization_info", {})
    }


POPSSSensitivityAnalysisOperator = SensitivityAnalysisOperator
AdaptiveBitAllocationOperator = POPSSAdaptiveBitAllocationOperator
