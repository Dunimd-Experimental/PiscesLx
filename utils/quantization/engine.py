#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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
from typing import Optional, Any, Dict, List, Union
from .core import QuantizationConfig, QuantizationMetrics
from .methods import (
    BitsAndBytesQuantizer, DynamicQuantizer, StaticQuantizer, 
    GPTQQuantizer, AWQQuantizer, SqueezeLLMQuantizer, KVCacheQuantizer
)
from .bitnet_v2 import BitNetV2Quantizer
from utils.log.core import PiscesLxCoreLog
from utils.error import PiscesLxCoreValidationError, PiscesLxCoreIOError

logger = PiscesLxCoreLog("PiscesLx.Utils.Quantization.Engine")

class QuantizationEngine:
    """Main quantization engine that orchestrates different quantization methods."""
    
    def __init__(self):
        self.quantizers = {
            'bitsandbytes': BitsAndBytesQuantizer(),
            'dynamic': DynamicQuantizer(),
            'static': StaticQuantizer(),
            'gptq': GPTQQuantizer(),
            'awq': AWQQuantizer(),
            'squeezellm': SqueezeLLMQuantizer(),
            'kvcache': KVCacheQuantizer(),
            'bitnet_v2': BitNetV2Quantizer(),
        }
        self.metrics = QuantizationMetrics()
        self.calibration_data = None
    
    def quantize_model(self, model: nn.Module, config: QuantizationConfig, 
                      calibration_data: Optional[Any] = None) -> nn.Module:
        """
        Main quantization method that applies the specified quantization to a model.
        
        Args:
            model: The model to quantize
            config: Quantization configuration
            calibration_data: Optional calibration data for static/GPTQ quantization
            
        Returns:
            Quantized model
        """
        try:
            logger.info("starting model quantization", 
                       method=config.method.value,
                       bits=config.bits,
                       granularity=config.granularity.value)
        except Exception as log_e:
            logger.debug("QUANTIZATION_LOG_ERROR", error=str(log_e))
        
        # Validate configuration
        self._validate_config(config)
        
        # Get the appropriate quantizer
        quantizer = self._get_quantizer(config.method.value)
        if quantizer is None:
            raise PiscesLxCoreValidationError(f"Unsupported quantization method: {config.method.value}")
        
        # Store calibration data for later use
        if calibration_data is not None:
            self.calibration_data = calibration_data
        elif config.calibration_dataset:
            self.calibration_data = self._load_calibration_data(config)
        
        # Apply quantization
        try:
            if config.method.value in ['static', 'gptq']:
                quantized_model = quantizer.quantize(model, config, self.calibration_data)
            else:
                quantized_model = quantizer.quantize(model, config)
            
            # Calculate and log quantization metrics
            self._calculate_metrics(model, quantized_model, config)
            
            logger.info("model quantization completed successfully")
            return quantized_model
            
        except Exception as e:
            logger.error(f"quantization failed: {e}")
            raise PiscesLxCoreIOError(f"Quantization failed: {e}")
    
    def _validate_config(self, config: QuantizationConfig):
        """Validate quantization configuration."""
        if config.bits not in [1, 2, 4, 8, 16]:
            logger.warning(f"Unusual quantization bits: {config.bits}")
        
        if config.method.value == 'gptq' and config.bits not in [2, 3, 4, 8]:
            logger.warning(f"GPTQ typically uses 2, 3, 4, or 8 bits, got {config.bits}")
        
        if config.method.value == 'bitnet_v2' and config.bits != 2:
            logger.warning(f"BitNet v2 uses 1.58-bit quantization, config.bits={config.bits} will be adjusted")
    
    def _get_quantizer(self, method: str):
        """Get the appropriate quantizer for the method."""
        return self.quantizers.get(method)
    
    def _load_calibration_data(self, config: QuantizationConfig) -> Optional[Any]:
        """Load calibration data if specified."""
        if not config.calibration_dataset:
            return None
        
        try:
            # This would typically load from a dataset or file
            logger.info(f"Loading calibration data from {config.calibration_dataset}")
            # Placeholder for actual data loading logic
            return None
        except Exception as e:
            logger.warning(f"Failed to load calibration data: {e}")
            return None
    
    def _calculate_metrics(self, original_model: nn.Module, quantized_model: nn.Module, 
                          config: QuantizationConfig):
        """Calculate quantization metrics."""
        try:
            original_size = self._calculate_model_size(original_model)
            quantized_size = self._calculate_model_size(quantized_model)
            
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
            size_reduction = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
            
            self.metrics.compression_ratio = compression_ratio
            self.metrics.size_reduction_percentage = size_reduction
            self.metrics.quantization_method = config.method.value
            self.metrics.bits = config.bits
            
            logger.info("quantization metrics calculated",
                       compression_ratio=f"{compression_ratio:.2f}x",
                       size_reduction=f"{size_reduction:.1f}%",
                       original_size_mb=f"{original_size:.1f}MB",
                       quantized_size_mb=f"{quantized_size:.1f}MB")
        except Exception as e:
            logger.warning(f"Failed to calculate metrics: {e}")
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        return total_size / 1024 / 1024  # Convert to MB
    
    def get_metrics(self) -> QuantizationMetrics:
        """Get quantization metrics."""
        return self.metrics
    
    def benchmark_quantization(self, model: nn.Module, config: QuantizationConfig,
                              test_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Benchmark quantization performance and accuracy.
        
        Args:
            model: Original model
            config: Quantization configuration
            test_data: Optional test data for accuracy evaluation
            
        Returns:
            Benchmark results dictionary
        """
        try:
            logger.info("starting quantization benchmark")
            
            import time
            
            # Measure original model performance
            original_start = time.time()
            # Placeholder for inference benchmark
            original_inference_time = time.time() - original_start
            
            # Apply quantization
            quantized_model = self.quantize_model(model, config)
            
            # Measure quantized model performance
            quantized_start = time.time()
            # Placeholder for inference benchmark
            quantized_inference_time = time.time() - quantized_start
            
            # Calculate speedup
            speedup = original_inference_time / quantized_inference_time if quantized_inference_time > 0 else 1.0
            
            results = {
                'original_inference_time': original_inference_time,
                'quantized_inference_time': quantized_inference_time,
                'speedup': speedup,
                'compression_ratio': self.metrics.compression_ratio,
                'size_reduction_percentage': self.metrics.size_reduction_percentage,
                'quantization_method': config.method.value,
                'bits': config.bits
            }
            
            logger.info("quantization benchmark completed",
                       speedup=f"{speedup:.2f}x",
                       compression_ratio=f"{self.metrics.compression_ratio:.2f}x")
            
            return results
            
        except Exception as e:
            logger.error(f"benchmark failed: {e}")
            raise PiscesLxCoreIOError(f"Benchmark failed: {e}")

class QuantizationManager:
    """High-level quantization manager for easy model quantization."""
    
    def __init__(self):
        self.engine = QuantizationEngine()
        self.default_config = QuantizationConfig()
    
    def quantize(self, model: nn.Module, method: str = "bitsandbytes", 
                bits: int = 8, **kwargs) -> nn.Module:
        """
        Simple quantization interface.
        
        Args:
            model: Model to quantize
            method: Quantization method (bitsandbytes, dynamic, static, gptq, awq, bitnet_v2)
            bits: Number of bits for quantization
            **kwargs: Additional configuration parameters
            
        Returns:
            Quantized model
        """
        config = QuantizationConfig(
            method=method,
            bits=bits,
            **kwargs
        )
        
        return self.engine.quantize_model(model, config)
    
    def auto_quantize(self, model: nn.Module, target_compression: float = 2.0,
                     preserve_accuracy: bool = True) -> nn.Module:
        """
        Automatically select the best quantization method based on requirements.
        
        Args:
            model: Model to quantize
            target_compression: Target compression ratio
            preserve_accuracy: Whether to prioritize accuracy preservation
            
        Returns:
            Quantized model with optimal method
        """
        try:
            logger.info("starting automatic quantization selection",
                       target_compression=target_compression,
                       preserve_accuracy=preserve_accuracy)
            
            # Define method priorities based on requirements
            if preserve_accuracy and target_compression <= 2.0:
                methods = ["bitsandbytes", "awq", "gptq"]
                bits_options = [8, 4]
            elif preserve_accuracy and target_compression <= 4.0:
                methods = ["awq", "gptq", "bitnet_v2"]
                bits_options = [4, 2]
            else:
                methods = ["bitnet_v2", "gptq", "dynamic"]
                bits_options = [2, 4, 8]
            
            # Try different combinations
            for method in methods:
                for bits in bits_options:
                    try:
                        config = QuantizationConfig(method=method, bits=bits)
                        quantized_model = self.engine.quantize_model(model, config)
                        
                        # Check if compression target is met
                        metrics = self.engine.get_metrics()
                        if metrics.compression_ratio >= target_compression:
                            logger.info(f"Selected quantization: {method} with {bits} bits, "
                                      f"compression ratio: {metrics.compression_ratio:.2f}x")
                            return quantized_model
                            
                    except Exception as e:
                        logger.warning(f"Failed to quantize with {method} {bits} bits: {e}")
                        continue
            
            # Fallback to basic quantization
            logger.warning("Auto-quantization failed, falling back to basic 8-bit quantization")
            return self.quantize(model, method="bitsandbytes", bits=8)
            
        except Exception as e:
            logger.error(f"Auto-quantization failed: {e}")
            raise PiscesLxCoreIOError(f"Auto-quantization failed: {e}")
    
    def get_supported_methods(self) -> List[str]:
        """Get list of supported quantization methods."""
        return list(self.engine.quantizers.keys())
    
    def compare_methods(self, model: nn.Module, methods: List[str] = None,
                       test_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Compare different quantization methods on the same model.
        
        Args:
            model: Model to compare methods on
            methods: List of methods to compare (uses all if None)
            test_data: Optional test data for evaluation
            
        Returns:
            Comparison results
        """
        if methods is None:
            methods = self.get_supported_methods()
        
        results = {}
        
        for method in methods:
            try:
                config = QuantizationConfig(method=method, bits=4)  # Use 4-bit as standard
                quantized_model = self.engine.quantize_model(model, config)
                metrics = self.engine.get_metrics()
                
                results[method] = {
                    'compression_ratio': metrics.compression_ratio,
                    'size_reduction_percentage': metrics.size_reduction_percentage,
                    'quantization_method': metrics.quantization_method,
                    'success': True
                }
                
            except Exception as e:
                results[method] = {
                    'error': str(e),
                    'success': False
                }
        
        return results