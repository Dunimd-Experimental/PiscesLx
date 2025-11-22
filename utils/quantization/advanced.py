#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from .core import QuantizationConfig, QuantizationMetrics
from utils.log.core import PiscesLxCoreLog
from utils.error import PiscesLxCoreValidationError

logger = PiscesLxCoreLog("PiscesLx.Core.Quantization.Advanced")


class SensitivityAnalyzer:
    """Analyzes layer sensitivity to quantization for optimal bit allocation."""
    
    def __init__(self):
        self.sensitivity_cache = {}
    
    def analyze_model_sensitivity(
        self, 
        model: nn.Module, 
        test_inputs: Optional[List[torch.Tensor]] = None,
        bits: int = 8,
        num_samples: int = 10
    ) -> Dict[str, float]:
        """
        Analyze sensitivity of each layer to quantization.
        
        Args:
            model: The model to analyze
            test_inputs: Optional test inputs for sensitivity analysis
            bits: Number of bits for sensitivity testing
            num_samples: Number of samples to use for analysis
            
        Returns:
            Dictionary mapping layer names to sensitivity scores
        """
        try:
            logger.info("starting sensitivity analysis", 
                       bits=bits, num_samples=num_samples)
        except Exception as log_e:
            logger.debug("SENSITIVITY_LOG_ERROR", error=str(log_e))
        
        sensitivity_analysis = {}
        
        # Generate test inputs if not provided
        if test_inputs is None:
            test_inputs = self._generate_test_inputs(model, num_samples)
        
        # Get baseline outputs
        baseline_outputs = self._get_baseline_outputs(model, test_inputs)
        
        # Analyze each layer
        for name, module in model.named_modules():
            if self._is_quantizable_layer(module):
                try:
                    sensitivity = self._analyze_layer_sensitivity(
                        model, module, name, test_inputs, baseline_outputs, bits
                    )
                    sensitivity_analysis[name] = sensitivity
                    
                    logger.debug(f"Layer {name} sensitivity: {sensitivity:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze layer {name}: {e}")
                    sensitivity_analysis[name] = 0.0
        
        try:
            logger.info("sensitivity analysis completed", 
                       layers_analyzed=len(sensitivity_analysis),
                       most_sensitive=max(sensitivity_analysis.items(), 
                                        key=lambda x: x[1])[0] if sensitivity_analysis else "none")
        except Exception as log_e:
            logger.debug("SENSITIVITY_LOG_ERROR", error=str(log_e))
        
        return sensitivity_analysis
    
    def _generate_test_inputs(self, model: nn.Module, num_samples: int) -> List[torch.Tensor]:
        """Generate random test inputs for sensitivity analysis."""
        test_inputs = []
        
        # Try to infer input shape from model
        first_module = next(model.modules())
        if hasattr(first_module, 'weight'):
            input_shape = first_module.weight.shape[1]
        else:
            input_shape = 512  # Default
        
        for i in range(num_samples):
            generator = torch.Generator().manual_seed(42 + i)
            test_input = torch.randn(1, input_shape, generator=generator)
            test_inputs.append(test_input)
        
        return test_inputs
    
    def _get_baseline_outputs(self, model: nn.Module, 
                               test_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Get baseline outputs from the original model."""
        baseline_outputs = []
        model.eval()
        
        with torch.no_grad():
            for test_input in test_inputs:
                try:
                    output = model(test_input)
                    baseline_outputs.append(output.clone())
                except Exception as e:
                    logger.warning(f"Failed to get baseline output: {e}")
                    baseline_outputs.append(torch.zeros_like(test_input))
        
        return baseline_outputs
    
    def _analyze_layer_sensitivity(
        self, 
        model: nn.Module,
        layer: nn.Module, 
        layer_name: str,
        test_inputs: List[torch.Tensor],
        baseline_outputs: List[torch.Tensor],
        bits: int
    ) -> float:
        """Analyze sensitivity of a specific layer."""
        # Create a copy of the layer for quantization testing
        layer_copy = self._copy_layer(layer)
        
        # Apply quantization to the layer copy
        quantized_layer = self._apply_layer_quantization(layer_copy, bits)
        
        # Replace the original layer temporarily
        original_layer = self._replace_layer(model, layer_name, quantized_layer)
        
        try:
            # Get outputs with quantized layer
            quantized_outputs = []
            model.eval()
            
            with torch.no_grad():
                for test_input in test_inputs:
                    try:
                        output = model(test_input)
                        quantized_outputs.append(output.clone())
                    except Exception as e:
                        logger.warning(f"Failed to get quantized output: {e}")
                        quantized_outputs.append(torch.zeros_like(test_input))
            
            # Calculate sensitivity as output difference
            sensitivity = self._calculate_output_difference(
                baseline_outputs, quantized_outputs
            )
            
        finally:
            # Restore original layer
            self._replace_layer(model, layer_name, original_layer)
        
        return sensitivity
    
    def _is_quantizable_layer(self, module: nn.Module) -> bool:
        """Check if a module is quantizable."""
        return isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d))
    
    def _copy_layer(self, layer: nn.Module) -> nn.Module:
        """Create a copy of a layer."""
        import copy
        return copy.deepcopy(layer)
    
    def _apply_layer_quantization(self, layer: nn.Module, bits: int) -> nn.Module:
        """Apply quantization to a layer."""
        # Simple quantization simulation
        if hasattr(layer, 'weight'):
            original_weight = layer.weight.data
            quantized_weight = self._simulate_quantization(original_weight, bits)
            layer.weight.data = quantized_weight
        
        return layer
    
    def _simulate_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Simulate quantization effects."""
        if bits >= 16:
            return tensor
        
        # Apply uniform quantization
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        min_val = tensor.min()
        max_val = tensor.max()
        
        if max_val <= min_val:
            return tensor
        
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale)
        
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def _replace_layer(self, model: nn.Module, layer_name: str, 
                      new_layer: nn.Module) -> nn.Module:
        """Replace a layer in the model."""
        # Navigate to the parent module
        name_parts = layer_name.split('.')
        parent_name = '.'.join(name_parts[:-1]) if len(name_parts) > 1 else ''
        layer_attr = name_parts[-1]
        
        if parent_name:
            parent_module = dict(model.named_modules())[parent_name]
        else:
            parent_module = model
        
        # Store original layer
        original_layer = getattr(parent_module, layer_attr)
        
        # Replace with new layer
        setattr(parent_module, layer_attr, new_layer)
        
        return original_layer
    
    def _calculate_output_difference(self, 
                                   baseline_outputs: List[torch.Tensor],
                                   quantized_outputs: List[torch.Tensor]) -> float:
        """Calculate difference between baseline and quantized outputs."""
        total_diff = 0.0
        count = 0
        
        for base, quant in zip(baseline_outputs, quantized_outputs):
            if base.shape == quant.shape:
                diff = torch.mean(torch.abs(base - quant)).item()
                total_diff += diff
                count += 1
        
        return total_diff / count if count > 0 else 0.0


class MemoryEstimator:
    """Estimates memory usage for quantized models."""
    
    def __init__(self):
        self.activation_ratio = 0.2  # 20% of model size
        self.kv_cache_ratio = 0.1  # 10% of model size
    
    def estimate_memory_usage(
        self, 
        model_config: Any, 
        quantization_config: QuantizationConfig
    ) -> Dict[str, float]:
        """
        Estimate memory usage of a quantized model.
        
        Args:
            model_config: Model configuration
            quantization_config: Quantization configuration
            
        Returns:
            Dictionary containing memory usage estimates
        """
        # Estimate total parameters
        param_count = self._estimate_parameter_count(model_config)
        
        # Calculate model memory
        bytes_per_param = quantization_config.bits / 8
        model_memory_mb = (param_count * bytes_per_param) / (1024 * 1024)
        
        # Calculate activation memory
        activation_memory_mb = model_memory_mb * self.activation_ratio
        
        # Calculate KV-cache memory if enabled
        kv_cache_memory_mb = 0
        if quantization_config.enable_kv_cache_quant:
            kv_cache_memory_mb = self._estimate_kv_cache_memory(
                model_config, quantization_config.kv_cache_bits
            )
        
        # Total memory
        total_memory_mb = model_memory_mb + activation_memory_mb + kv_cache_memory_mb
        
        # Compression ratio
        baseline_memory_mb = (param_count * 4) / (1024 * 1024)  # FP32 baseline
        compression_ratio = baseline_memory_mb / total_memory_mb if total_memory_mb > 0 else 1.0
        
        result = {
            "model_memory_mb": model_memory_mb,
            "activation_memory_mb": activation_memory_mb,
            "kv_cache_memory_mb": kv_cache_memory_mb,
            "total_memory_mb": total_memory_mb,
            "baseline_memory_mb": baseline_memory_mb,
            "compression_ratio": compression_ratio,
            "size_reduction_percentage": (1 - total_memory_mb / baseline_memory_mb) * 100 if baseline_memory_mb > 0 else 0
        }
        
        try:
            logger.info("memory estimation completed",
                       total_memory_mb=f"{total_memory_mb:.1f}",
                       compression_ratio=f"{compression_ratio:.2f}x")
        except Exception as log_e:
            logger.debug("MEMORY_ESTIMATE_LOG_ERROR", error=str(log_e))
        
        return result
    
    def _estimate_parameter_count(self, model_config: Any) -> int:
        """Estimate total parameter count from model configuration."""
        # Get configuration parameters
        vocab_size = getattr(model_config, 'vocab_size', 50000)
        hidden_size = getattr(model_config, 'hidden_size', 768)
        num_layers = getattr(model_config, 'num_hidden_layers', 12)
        intermediate_size = getattr(model_config, 'intermediate_size', hidden_size * 4)
        
        # Embedding parameters
        embedding_params = vocab_size * hidden_size
        
        # Layer parameters (attention + FFN + layer norm)
        attention_params = 4 * hidden_size * hidden_size  # Q, K, V, O
        ffn_params = 2 * hidden_size * intermediate_size  # Two linear layers
        layer_norm_params = 2 * hidden_size  # Two layer norms
        
        layer_params = attention_params + ffn_params + layer_norm_params
        total_layer_params = num_layers * layer_params
        
        # Output layer
        output_params = hidden_size * vocab_size
        
        total_params = embedding_params + total_layer_params + output_params
        
        return total_params
    
    def _estimate_kv_cache_memory(self, model_config: Any, kv_cache_bits: int) -> float:
        """Estimate KV-cache memory usage."""
        max_seq_len = getattr(model_config, 'max_position_embeddings', 2048)
        hidden_size = getattr(model_config, 'hidden_size', 768)
        num_layers = getattr(model_config, 'num_hidden_layers', 12)
        
        # KV cache size per token: 2 (K and V) * hidden_size * num_layers
        bytes_per_element = kv_cache_bits / 8
        kv_cache_size = 2 * max_seq_len * hidden_size * num_layers * bytes_per_element
        
        return kv_cache_size / (1024 * 1024)  # Convert to MB


class OptimalConfigRecommender:
    """Recommends optimal quantization configurations."""
    
    def __init__(self):
        self.memory_estimator = MemoryEstimator()
        self.sensitivity_analyzer = SensitivityAnalyzer()
    
    def recommend_config(
        self,
        model: nn.Module,
        model_config: Any,
        target_memory_mb: Optional[float] = None,
        target_accuracy: Optional[float] = None,
        device_constraints: Optional[Dict[str, Any]] = None,
        calibration_data: Optional[Any] = None
    ) -> QuantizationConfig:
        """
        Recommend optimal quantization configuration.
        
        Args:
            model: The model to quantize
            model_config: Model configuration
            target_memory_mb: Target memory usage in MB
            target_accuracy: Target accuracy retention (0.0-1.0)
            device_constraints: Device constraints dict
            calibration_data: Optional calibration data
            
        Returns:
            Recommended QuantizationConfig
        """
        try:
            logger.info("starting optimal config recommendation",
                       target_memory_mb=target_memory_mb,
                       target_accuracy=target_accuracy)
        except Exception as log_e:
            logger.debug("CONFIG_RECOMMEND_LOG_ERROR", error=str(log_e))
        
        # Start with default configuration
        config = QuantizationConfig()
        
        # Analyze model sensitivity if model is provided
        if model is not None:
            sensitivity_analysis = self.sensitivity_analyzer.analyze_model_sensitivity(
                model, calibration_data
            )
        else:
            sensitivity_analysis = {}
        
        # Optimize based on memory constraints
        if target_memory_mb:
            config = self._optimize_for_memory(
                model_config, config, target_memory_mb, sensitivity_analysis
            )
        
        # Optimize based on accuracy constraints
        if target_accuracy:
            config = self._optimize_for_accuracy(
                model_config, config, target_accuracy, sensitivity_analysis
            )
        
        # Optimize based on device constraints
        if device_constraints:
            config = self._optimize_for_device(
                config, device_constraints
            )
        
        # Final validation
        config = self._validate_recommended_config(config, model_config)
        
        try:
            logger.info("optimal configuration recommended",
                       method=config.method.value,
                       bits=config.bits,
                       enable_kv_cache_quant=config.enable_kv_cache_quant)
        except Exception as log_e:
            logger.debug("CONFIG_RECOMMEND_LOG_ERROR", error=str(log_e))
        
        return config
    
    def _optimize_for_memory(
        self,
        model_config: Any,
        config: QuantizationConfig,
        target_memory_mb: float,
        sensitivity_analysis: Dict[str, float]
    ) -> QuantizationConfig:
        """Optimize configuration for memory constraints."""
        # Test different bit widths
        for bits in [4, 2, 1]:
            test_config = QuantizationConfig(bits=bits)
            estimated_memory = self.memory_estimator.estimate_memory_usage(
                model_config, test_config
            )
            
            if estimated_memory["total_memory_mb"] <= target_memory_mb:
                config.bits = bits
                break
        
        # If still over memory budget, enable KV-cache quantization
        estimated_memory = self.memory_estimator.estimate_memory_usage(
            model_config, config
        )
        
        if estimated_memory["total_memory_mb"] > target_memory_mb:
            config.enable_kv_cache_quant = True
            config.kv_cache_bits = 4  # Use 4-bit for KV cache
        
        return config
    
    def _optimize_for_accuracy(
        self,
        model_config: Any,
        config: QuantizationConfig,
        target_accuracy: float,
        sensitivity_analysis: Dict[str, float]
    ) -> QuantizationConfig:
        """Optimize configuration for accuracy constraints."""
        # If high accuracy required, use conservative quantization
        if target_accuracy > 0.95:
            config.bits = 8
            config.method = "dynamic"  # Dynamic quantization preserves accuracy better
        elif target_accuracy > 0.90:
            config.bits = 4
        else:
            config.bits = 2
        
        return config
    
    def _optimize_for_device(
        self,
        config: QuantizationConfig,
        device_constraints: Dict[str, Any]
    ) -> QuantizationConfig:
        """Optimize configuration for device constraints."""
        device_type = device_constraints.get("type", "cuda")
        memory_gb = device_constraints.get("memory_gb", 16)
        
        if device_type == "cpu":
            # CPU prefers dynamic quantization
            config.method = "dynamic"
            config.bits = 8
        elif memory_gb < 8:
            # Low memory devices need aggressive quantization
            config.bits = 4
            config.enable_kv_cache_quant = True
            config.kv_cache_bits = 2
        elif memory_gb < 16:
            # Medium memory devices
            config.bits = 4
        
        return config
    
    def _validate_recommended_config(
        self, 
        config: QuantizationConfig, 
        model_config: Any
    ) -> QuantizationConfig:
        """Validate and adjust recommended configuration."""
        # Ensure bits are valid
        if config.bits not in [1, 2, 4, 8, 16]:
            config.bits = 8
        
        # Ensure KV-cache bits are valid
        if config.enable_kv_cache_quant:
            if config.kv_cache_bits not in [1, 2, 4, 8]:
                config.kv_cache_bits = 4
        
        # Validate method compatibility
        if config.method == "bitnet_v2" and config.bits != 2:
            config.bits = 2
        
        return config


class AdvancedQuantizationUtils:
    """Advanced utilities for quantization analysis and optimization."""
    
    def __init__(self):
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.memory_estimator = MemoryEstimator()
        self.config_recommender = OptimalConfigRecommender()
    
    def comprehensive_analysis(
        self,
        model: nn.Module,
        model_config: Any,
        test_inputs: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive quantization analysis.
        
        Args:
            model: The model to analyze
            model_config: Model configuration
            test_inputs: Optional test inputs
            
        Returns:
            Comprehensive analysis results
        """
        try:
            logger.info("starting comprehensive quantization analysis")
        except Exception as log_e:
            logger.debug("COMPREHENSIVE_ANALYSIS_LOG_ERROR", error=str(log_e))
        
        results = {}
        
        # Sensitivity analysis
        sensitivity_results = self.sensitivity_analyzer.analyze_model_sensitivity(
            model, test_inputs
        )
        results["sensitivity_analysis"] = sensitivity_results
        
        # Memory estimation for different configurations
        memory_estimates = {}
        for bits in [1, 2, 4, 8]:
            test_config = QuantizationConfig(bits=bits)
            memory_estimates[f"{bits}bit"] = self.memory_estimator.estimate_memory_usage(
                model_config, test_config
            )
        results["memory_estimates"] = memory_estimates
        
        # Optimal configurations for different scenarios
        scenarios = {
            "memory_optimized": {"target_memory_mb": 100},
            "accuracy_optimized": {"target_accuracy": 0.95},
            "balanced": {"target_memory_mb": 500, "target_accuracy": 0.90}
        }
        
        optimal_configs = {}
        for scenario_name, constraints in scenarios.items():
            optimal_configs[scenario_name] = self.config_recommender.recommend_config(
                model, model_config, **constraints
            )
        results["optimal_configs"] = optimal_configs
        
        try:
            logger.info("comprehensive analysis completed",
                       sensitivity_layers=len(sensitivity_results),
                       memory_scenarios=len(memory_estimates),
                       optimal_scenarios=len(optimal_configs))
        except Exception as log_e:
            logger.debug("COMPREHENSIVE_ANALYSIS_LOG_ERROR", error=str(log_e))
        
        return results