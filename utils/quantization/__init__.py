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

from typing import Optional, Any, Dict
import torch

"""
Quantization module for PiscesL1.

This module provides comprehensive quantization capabilities including:
- Multiple quantization methods (BitsAndBytes, GPTQ, AWQ, Dynamic, Static, BitNet v2)
- Calibration and optimization tools
- Performance benchmarking and metrics
- Easy-to-use interfaces for model quantization

The module has been restructured with a modular architecture for better maintainability
and extensibility.
"""

# Import core components
from .core import (
    QuantizationMethod, 
    QuantizationGranularity, 
    QuantizationConfig, 
    QuantizationMetrics
)

# Import quantization methods
from .methods import (
    BitsAndBytesQuantizer,
    DynamicQuantizer,
    StaticQuantizer,
    GPTQQuantizer,
    AWQQuantizer,
    SqueezeLLMQuantizer,
    KVCacheQuantizer
)

# Import BitNet v2
from .bitnet_v2 import (
    STEQuantizer,
    HadamardTransform,
    HBitLinear,
    BitNetV2Config,
    BitNetV2Quantizer
)

# Import engine and manager
from .engine import (
    QuantizationEngine,
    QuantizationManager
)

# Import calibration
from .calibration import (
    CalibrationDataLoader,
    CalibrationProcessor,
    CalibrationMetrics,
    CalibrationManager
)

# Import utilities
from .utils import (
    get_device_memory_info,
    calculate_model_size,
    estimate_quantized_size,
    get_quantization_method_info,
    compare_quantization_methods,
    recommend_quantization_method,
    validate_quantization_config,
    safe_quantization_config
)

# Backward compatibility: maintain the original PiscesLxCoreQuantizer interface
class PiscesLxCoreQuantizer:
    """Backward compatibility wrapper for the original quantizer interface."""
    
    def __init__(self, device_manager: Optional[Any] = None):
        """
        Initialize the quantizer with backward compatibility.
        
        Args:
            device_manager: Optional device manager for compatibility
        """
        self.manager = QuantizationManager()
        self._metrics = QuantizationMetrics()
        self.device_manager = device_manager
    
    def quantize_checkpoint(
        self,
        checkpoint_path: str,
        save_path: str,
        bits: int = 8,
        *,
        model_size: Optional[str] = None,
        config_path: Optional[str] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ) -> QuantizationMetrics:
        """
        Quantize a model checkpoint with backward compatibility.
        
        Args:
            checkpoint_path: Path to the original model checkpoint
            save_path: Path to save the quantized model
            bits: Number of bits for quantization
            model_size: Size of the model (for config inference)
            config_path: Path to the model configuration file
            quantization_config: Quantization configuration
            
        Returns:
            QuantizationMetrics: Metrics evaluating quantization performance
        """
        import os
        import time
        import torch
        from utils.log.core import PiscesLxCoreLog
        
        logger = PiscesLxCoreLog("PiscesLx.Utils.Quantization")
        
        # Use provided config or create default
        config = quantization_config or QuantizationConfig(bits=bits)
        
        start_time = time.time()
        
        try:
            # Load model and checkpoint
            from model import ArcticModel, ArcticConfig
            
            cfg_path = config_path or f"configs/{(model_size or '0.5B').upper()}.json"
            cfg = ArcticConfig.from_json(cfg_path)
            
            model = ArcticModel(cfg)
            state = torch.load(checkpoint_path, map_location="cpu")
            model_state = state["model"] if isinstance(state, dict) and "model" in state else state
            
            # Calculate original model size
            self._metrics.original_size_mb = self._calculate_model_size_mb(model_state)
            
            # Load state dict
            model.load_state_dict(model_state, strict=False)
            
            # Apply quantization
            quantized_model = self.manager.quantize(model, method=config.method.value, bits=config.bits)
            
            # Calculate metrics
            self._metrics.quantized_size_mb = self._calculate_model_size_mb(quantized_model.state_dict())
            self._metrics.compression_ratio = (
                self._metrics.original_size_mb / self._metrics.quantized_size_mb 
                if self._metrics.quantized_size_mb > 0 else 1.0
            )
            self._metrics.quantization_time_seconds = time.time() - start_time
            
            # Save quantized model
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            
            quantized_state = {
                "model": quantized_model.state_dict(),
                "quantization_config": {
                    "method": config.method.value,
                    "bits": config.bits,
                    "granularity": config.granularity.value,
                    "group_size": config.group_size,
                    "symmetric": config.symmetric,
                    "enable_kv_cache_quant": config.enable_kv_cache_quant,
                    "kv_cache_bits": config.kv_cache_bits,
                },
                "metrics": {
                    "original_size_mb": self._metrics.original_size_mb,
                    "quantized_size_mb": self._metrics.quantized_size_mb,
                    "compression_ratio": self._metrics.compression_ratio,
                    "quantization_time_seconds": self._metrics.quantization_time_seconds,
                }
            }
            
            torch.save(quantized_state, save_path)
            
            logger.info(
                "quantization completed successfully",
                path=save_path,
                method=config.method.value,
                original_size_mb=round(self._metrics.original_size_mb, 2),
                quantized_size_mb=round(self._metrics.quantized_size_mb, 2),
                compression_ratio=round(self._metrics.compression_ratio, 2),
                time_seconds=round(self._metrics.quantization_time_seconds, 2)
            )
            
            return self._metrics
            
        except Exception as e:
            logger.error(f"quantization failed: {e}")
            raise
    
    def _calculate_model_size_mb(self, state_dict: Dict[str, torch.Tensor]) -> float:
        """Calculate model size from state dict."""
        total_size = 0
        for tensor in state_dict.values():
            total_size += tensor.numel() * tensor.element_size()
        return total_size / 1024 / 1024  # Convert to MB
    
    def get_metrics(self) -> QuantizationMetrics:
        """Get quantization metrics."""
        return self._metrics

# Maintain backward compatibility for direct imports
__all__ = [
    # Core components
    "QuantizationMethod",
    "QuantizationGranularity", 
    "QuantizationConfig",
    "QuantizationMetrics",
    
    # Quantization methods
    "BitsAndBytesQuantizer",
    "DynamicQuantizer",
    "StaticQuantizer",
    "GPTQQuantizer",
    "AWQQuantizer",
    "SqueezeLLMQuantizer",
    "KVCacheQuantizer",
    
    # BitNet v2
    "STEQuantizer",
    "HadamardTransform",
    "HBitLinear",
    "BitNetV2Config",
    "BitNetV2Quantizer",
    
    # Engine and manager
    "QuantizationEngine",
    "QuantizationManager",
    
    # Calibration
    "CalibrationDataLoader",
    "CalibrationProcessor",
    "CalibrationMetrics",
    "CalibrationManager",
    
    # Utilities
    "get_device_memory_info",
    "calculate_model_size",
    "estimate_quantized_size",
    "get_quantization_method_info",
    "compare_quantization_methods",
    "recommend_quantization_method",
    "validate_quantization_config",
    "safe_quantization_config",
    
    # Backward compatibility
    "PiscesLxCoreQuantizer"
]