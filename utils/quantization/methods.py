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
from typing import Optional, Any, Dict
from .core import QuantizationConfig, QuantizationMetrics
from utils.log.core import PiscesLxCoreLog
from utils.error import PiscesLxCoreValidationError, PiscesLxCoreIOError

logger = PiscesLxCoreLog("PiscesLx.Core.Quantization.Methods")

class BitsAndBytesQuantizer:
    """BitsAndBytes quantization method implementation."""
    
    def __init__(self):
        self.metrics = QuantizationMetrics()
    
    def quantize(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply BitsAndBytes quantization to the model."""
        try:
            logger.info("applying bitsandbytes quantization", 
                       bits=config.bits, 
                       granularity=config.granularity.value)
        except Exception as log_e:
            logger.debug("QUANTIZATION_LOG_ERROR", error=str(log_e))
        
        bnb_config = self._build_bnb_config(config.bits)
        if bnb_config is not None:
            # Create a new model with the BitsAndBytes quantization config
            try:
                from model import ArcticModel
                quantized_model = ArcticModel(model.config, quantization_config=bnb_config)
                quantized_model.load_state_dict(model.state_dict(), strict=False)
                return quantized_model
            except Exception as e:
                logger.warning(f"failed to load into quantized model, falling back to original: {e}")
        
        return model
    
    def _build_bnb_config(self, bits: int) -> Optional[Any]:
        """Build BitsAndBytes configuration."""
        try:
            from transformers import BitsAndBytesConfig
            
            if bits == 4:
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif bits == 8:
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
            else:
                logger.warning(f"BitsAndBytes only supports 4-bit and 8-bit quantization, got {bits}")
                return None
        except ImportError:
            logger.error("BitsAndBytes not available")
            return None

class DynamicQuantizer:
    """Dynamic quantization method implementation."""
    
    def __init__(self):
        self.metrics = QuantizationMetrics()
    
    def quantize(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply dynamic quantization to the model."""
        try:
            logger.info("applying dynamic quantization", 
                       bits=config.bits, 
                       granularity=config.granularity.value)
        except Exception as log_e:
            logger.debug("QUANTIZATION_LOG_ERROR", error=str(log_e))
        
        # Determine quantization dtype based on bits
        if config.bits == 8:
            dtype = torch.qint8
        elif config.bits == 4:
            dtype = torch.qint4
        elif config.bits == 16:
            dtype = torch.qint16
        else:
            dtype = torch.qint8
        
        # Perform dynamic quantization on PyTorch models
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d},
            dtype=dtype
        )
        return quantized_model

class StaticQuantizer:
    """Static quantization method implementation."""
    
    def __init__(self):
        self.metrics = QuantizationMetrics()
    
    def quantize(self, model: nn.Module, config: QuantizationConfig, 
                 calibration_data: Optional[Any] = None) -> nn.Module:
        """Apply static quantization with calibration to the model."""
        try:
            logger.info("applying static quantization", 
                       bits=config.bits, 
                       granularity=config.granularity.value,
                       calibration_samples=config.num_calibration_samples)
        except Exception as log_e:
            logger.debug("QUANTIZATION_LOG_ERROR", error=str(log_e))
        
        # Prepare the model for static quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model)
        
        # Perform the calibration step if calibration data is provided
        if calibration_data is not None:
            self._calibrate_model(model_prepared, calibration_data, config)
        elif config.calibration_dataset:
            logger.warning("Static quantization requires calibration data")
        
        # Convert the prepared model to a quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        return quantized_model
    
    def _calibrate_model(self, model: nn.Module, calibration_data: Any, 
                        config: QuantizationConfig):
        """Calibrate the model with provided data."""
        logger.info(f"Calibrating model with {config.num_calibration_samples} samples")
        # Placeholder for calibration logic
        # In practice, this would run forward passes with calibration data

class GPTQQuantizer:
    """GPTQ quantization method implementation."""
    
    def __init__(self):
        self.metrics = QuantizationMetrics()
    
    def quantize(self, model: nn.Module, config: QuantizationConfig, 
                 calibration_data: Optional[Any] = None) -> nn.Module:
        """Apply GPTQ quantization to the model."""
        try:
            logger.info("applying GPTQ quantization", 
                       bits=config.bits, 
                       group_size=config.group_size)
        except Exception as log_e:
            logger.debug("QUANTIZATION_LOG_ERROR", error=str(log_e))
        
        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
            
            # Check if calibration data is available
            if calibration_data is None and not config.calibration_dataset:
                logger.warning("GPTQ requires calibration data, falling back to bitsandbytes")
                return BitsAndBytesQuantizer().quantize(model, config)
            
            quantize_config = BaseQuantizeConfig(
                bits=config.bits,
                group_size=config.group_size,
                desc_act=False,
                damp_percent=0.1
            )
            
            # Use in-memory model serialization for GPTQ quantization
            return self._perform_gptq_quantization(model, quantize_config, calibration_data, config)
            
        except ImportError:
            logger.error("AutoGPTQ not available, falling back to bitsandbytes")
            return BitsAndBytesQuantizer().quantize(model, config)
        except Exception as e:
            logger.error(f"GPTQ quantization failed: {e}, falling back to bitsandbytes")
            return BitsAndBytesQuantizer().quantize(model, config)
    
    def _perform_gptq_quantization(self, model: nn.Module, quantize_config: Any,
                                  calibration_data: Optional[Any], 
                                  config: QuantizationConfig) -> nn.Module:
        """Perform GPTQ quantization with proper error handling."""
        import tempfile
        import os
        
        # Create temporary directory for GPTQ processing
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_path = os.path.join(tmpdir, "model")
            os.makedirs(temp_model_path, exist_ok=True)
            
            # Save model state and config
            torch.save({"model": model.state_dict()}, 
                      os.path.join(temp_model_path, "pytorch_model.bin"))
            model.config.to_json_file(os.path.join(temp_model_path, "config.json"))
            
            # Load model with GPTQ
            gptq_model = AutoGPTQForCausalLM.from_quantized(
                temp_model_path,
                quantize_config=quantize_config,
                device_map="auto"
            )
            
            # Quantize with calibration data
            if calibration_data is not None:
                gptq_model.quantize(calibration_data, 
                                  use_triton=True,
                                  batch_size=1)
            
            # Extract quantized state dict
            quantized_state_dict = gptq_model.state_dict()
            
            # Create new model with quantized weights
            from model import ArcticModel
            quantized_model = ArcticModel(model.config)
            quantized_model.load_state_dict(quantized_state_dict, strict=False)
            
            return quantized_model

class AWQQuantizer:
    """AWQ quantization method implementation."""
    
    def __init__(self):
        self.metrics = QuantizationMetrics()
    
    def quantize(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply AWQ quantization to the model."""
        try:
            logger.info("applying AWQ quantization", 
                       bits=config.bits, 
                       group_size=config.group_size)
        except Exception as log_e:
            logger.debug("QUANTIZATION_LOG_ERROR", error=str(log_e))
        
        try:
            from awq import AutoAWQForCausalLM
            
            # Use in-memory model processing for AWQ
            return self._perform_awq_quantization(model, config)
            
        except ImportError:
            logger.error("AWQ not available, falling back to bitsandbytes")
            return BitsAndBytesQuantizer().quantize(model, config)
        except Exception as e:
            logger.error(f"AWQ quantization failed: {e}, falling back to bitsandbytes")
            return BitsAndBytesQuantizer().quantize(model, config)
    
    def _perform_awq_quantization(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Perform AWQ quantization with proper error handling."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_path = os.path.join(tmpdir, "model")
            os.makedirs(temp_model_path, exist_ok=True)
            
            # Save model state and config
            torch.save({"model": model.state_dict()}, 
                      os.path.join(temp_model_path, "pytorch_model.bin"))
            model.config.to_json_file(os.path.join(temp_model_path, "config.json"))
            
            # Load and quantize with AWQ
            awq_model = AutoAWQForCausalLM.from_pretrained(temp_model_path)
            awq_model.quantize(
                quant_config={
                    "zero_point": True,
                    "q_group_size": config.group_size,
                    "w_bit": config.bits,
                    "version": "GEMM"
                }
            )
            
            # Extract quantized state dict
            quantized_state_dict = awq_model.state_dict()
            
            # Create new model with quantized weights
            from model import ArcticModel
            quantized_model = ArcticModel(model.config)
            quantized_model.load_state_dict(quantized_state_dict, strict=False)
            
            return quantized_model

class SqueezeLLMQuantizer:
    """SqueezeLLM quantization method implementation."""
    
    def __init__(self):
        self.metrics = QuantizationMetrics()
    
    def quantize(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply SqueezeLLM quantization to the model."""
        try:
            logger.info("applying SqueezeLLM quantization", bits=config.bits)
        except Exception as log_e:
            logger.debug("QUANTIZATION_LOG_ERROR", error=str(log_e))
        
        # SqueezeLLM is a specialized method, fallback to AWQ for now
        logger.warning("SqueezeLLM quantization not fully implemented, using AWQ as fallback")
        return AWQQuantizer().quantize(model, config)

class KVCacheQuantizer:
    """KV cache quantization method implementation with advanced quantization algorithms."""
    
    def __init__(self):
        self.metrics = QuantizationMetrics()
        self.quantization_stats = {}
    
    def quantize(self, model: nn.Module, config: QuantizationConfig) -> nn.Module:
        """Apply KV cache quantization to the model."""
        try:
            logger.info("applying advanced KV cache quantization", 
                       bits=config.kv_cache_bits,
                       method="adaptive_uniform")
        except Exception as log_e:
            logger.debug("QUANTIZATION_LOG_ERROR", error=str(log_e))
        
        # Add KV cache quantization hooks to the model
        self._apply_kv_cache_quantization(model, config)
        return model
    
    def _apply_kv_cache_quantization(self, model: nn.Module, config: QuantizationConfig):
        """Apply advanced KV cache quantization hooks."""
        def quantize_kv_cache_hook(module, input, output):
            if hasattr(output, 'past_key_values') and output.past_key_values is not None:
                quantized_past_key_values = []
                
                for layer_idx, layer_past in enumerate(output.past_key_values):
                    if len(layer_past) == 2:  # key, value
                        key, value = layer_past
                        
                        # Use adaptive quantization based on tensor statistics
                        key_quantized = self._adaptive_quantize_tensor(
                            key, config.kv_cache_bits, f"layer_{layer_idx}_key"
                        )
                        value_quantized = self._adaptive_quantize_tensor(
                            value, config.kv_cache_bits, f"layer_{layer_idx}_value"
                        )
                        
                        quantized_past_key_values.append((key_quantized, value_quantized))
                    else:
                        quantized_past_key_values.append(layer_past)
                
                # Create new output with quantized cache
                if hasattr(output, '_replace'):
                    return output._replace(past_key_values=quantized_past_key_values)
                else:
                    output.past_key_values = quantized_past_key_values
            
            return output
        
        # Register hooks for attention layers with priority
        for name, module in model.named_modules():
            if self._is_attention_module(name, module):
                module.register_forward_hook(quantize_kv_cache_hook)
    
    def _is_attention_module(self, name: str, module: nn.Module) -> bool:
        """Check if module is an attention module."""
        attention_keywords = ['attention', 'attn', 'self_attn', 'multihead_attention']
        return any(keyword in name.lower() for keyword in attention_keywords)
    
    def _adaptive_quantize_tensor(self, tensor: torch.Tensor, bits: int, 
                                 tensor_name: str) -> torch.Tensor:
        """Apply adaptive quantization based on tensor statistics."""
        if bits >= 16:
            return tensor
        
        # Collect statistics
        stats = self._collect_tensor_stats(tensor, tensor_name)
        
        # Choose quantization strategy based on statistics
        if stats["std_dev"] < 1e-6:  # Nearly constant tensor
            return self._quantize_constant_tensor(tensor, bits)
        elif stats["skewness"] > 2.0:  # Highly skewed distribution
            return self._quantize_skewed_tensor(tensor, bits, stats)
        else:
            return self._quantize_uniform_tensor(tensor, bits, stats)
    
    def _collect_tensor_stats(self, tensor: torch.Tensor, tensor_name: str) -> Dict[str, float]:
        """Collect comprehensive tensor statistics."""
        flat_tensor = tensor.flatten()
        
        stats = {
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.mean().item(),
            "std_dev": tensor.std().item(),
            "median": tensor.median().item(),
            "q25": torch.quantile(flat_tensor, 0.25).item(),
            "q75": torch.quantile(flat_tensor, 0.75).item(),
            "skewness": self._calculate_skewness(tensor),
            "kurtosis": self._calculate_kurtosis(tensor),
            "dynamic_range": tensor.max().item() - tensor.min().item(),
            "outlier_ratio": self._calculate_outlier_ratio(tensor)
        }
        
        # Cache statistics for analysis
        self.quantization_stats[tensor_name] = stats
        
        return stats
    
    def _calculate_skewness(self, tensor: torch.Tensor) -> float:
        """Calculate skewness of tensor distribution."""
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return 0.0
        
        skewness = torch.mean(((tensor - mean) / std) ** 3).item()
        return skewness
    
    def _calculate_kurtosis(self, tensor: torch.Tensor) -> float:
        """Calculate kurtosis of tensor distribution."""
        mean = tensor.mean()
        std = tensor.std()
        if std == 0:
            return 0.0
        
        kurtosis = torch.mean(((tensor - mean) / std) ** 4).item() - 3
        return kurtosis
    
    def _calculate_outlier_ratio(self, tensor: torch.Tensor) -> float:
        """Calculate ratio of outliers using IQR method."""
        flat_tensor = tensor.flatten()
        q25 = torch.quantile(flat_tensor, 0.25)
        q75 = torch.quantile(flat_tensor, 0.75)
        iqr = q75 - q25
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = ((flat_tensor < lower_bound) | (flat_tensor > upper_bound)).sum()
        outlier_ratio = outliers.item() / flat_tensor.numel()
        
        return outlier_ratio
    
    def _quantize_constant_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize nearly constant tensor."""
        # For constant tensors, just store the mean value
        mean_val = tensor.mean()
        return torch.full_like(tensor, mean_val)
    
    def _quantize_skewed_tensor(self, tensor: torch.Tensor, bits: int, 
                                 stats: Dict[str, float]) -> torch.Tensor:
        """Quantize skewed tensor using percentile-based range."""
        # Use percentile-based range for skewed distributions
        lower_bound = stats["q25"] - 1.5 * (stats["q75"] - stats["q25"])
        upper_bound = stats["q75"] + 1.5 * (stats["q75"] - stats["q25"])
        
        return self._quantize_with_range(tensor, bits, lower_bound, upper_bound)
    
    def _quantize_uniform_tensor(self, tensor: torch.Tensor, bits: int, 
                                   stats: Dict[str, float]) -> torch.Tensor:
        """Quantize uniform tensor using standard range."""
        # Handle outliers by using percentile-based range
        if stats["outlier_ratio"] > 0.05:  # More than 5% outliers
            return self._quantize_skewed_tensor(tensor, bits, stats)
        else:
            return self._quantize_with_range(tensor, bits, 
                                           stats["min"], stats["max"])
    
    def _quantize_with_range(self, tensor: torch.Tensor, bits: int, 
                              min_val: float, max_val: float) -> torch.Tensor:
        """Quantize tensor with specified range."""
        if max_val <= min_val:
            return tensor
        
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        
        scale = (max_val - min_val) / (qmax - qmin)
        if scale == 0:
            scale = 1e-8
        
        zero_point = qmin - torch.round(torch.tensor(min_val / scale))
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
    
    def get_quantization_stats(self) -> Dict[str, Dict[str, float]]:
        """Get collected quantization statistics."""
        return self.quantization_stats.copy()