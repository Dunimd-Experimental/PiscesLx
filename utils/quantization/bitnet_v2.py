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
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from .core import QuantizationConfig, QuantizationMetrics
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("PiscesLx.Utils.Quantization.BitNetV2")

class STEQuantizer(torch.autograd.Function):
    """Straight-Through Estimator for BitNet v2 quantization."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int = 1) -> torch.Tensor:
        """Forward pass: quantize to {-1, 0, 1} for 1.58bit or specified bit width."""
        if bits == 1:  # 1.58bit: ternary {-1, 0, 1}
            # Scale to [-1, 1] range first
            scale = input.abs().max()
            if scale > 0:
                input_normalized = input / scale
                # Quantize to {-1, 0, 1}
                quantized = torch.sign(input_normalized)
                # Handle exact zeros
                quantized = torch.where(input_normalized.abs() < 0.5, torch.zeros_like(quantized), quantized)
                return quantized * scale
            else:
                return input
        else:
            # General bit quantization
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
            scale = (input.max() - input.min()) / (qmax - qmin)
            zero_point = qmin - torch.round(input.min() / scale)
            
            quantized = torch.round(input / scale + zero_point)
            quantized = torch.clamp(quantized, qmin, qmax)
            
            # Dequantize
            dequantized = scale * (quantized - zero_point)
            return dequantized
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass: straight-through estimator."""
        return grad_output, None

class HadamardTransform:
    """Hadamard transform for reducing outliers in activations."""
    
    @staticmethod
    def hadamard_matrix(n: int) -> torch.Tensor:
        """Generate Hadamard matrix of size n x n."""
        if n == 1:
            return torch.tensor([[1.0]], dtype=torch.float32)
        
        # Ensure n is a power of 2
        if (n & (n - 1)) != 0:
            # Pad to next power of 2
            n = 2 ** math.ceil(math.log2(n))
        
        H = torch.tensor([[1.0]], dtype=torch.float32)
        while H.shape[0] < n:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1)
            ], dim=0) / math.sqrt(2)
        
        return H
    
    @staticmethod
    def apply(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Apply Hadamard transform to input tensor."""
        original_shape = x.shape
        if dim != -1:
            x = x.transpose(dim, -1)
        
        n = x.shape[-1]
        H = HadamardTransform.hadamard_matrix(n).to(x.device, x.dtype)
        
        # Reshape for matrix multiplication
        x_flat = x.reshape(-1, n)
        transformed = torch.matmul(x_flat, H.T)
        
        # Reshape back
        transformed = transformed.reshape(original_shape)
        if dim != -1:
            transformed = transformed.transpose(dim, -1)
        
        return transformed

class HBitLinear(nn.Module):
    """H-BitLinear module for BitNet v2 with Hadamard transform and activation quantization."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 1,
        activation_bits: int = 4,
        use_hadamard: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.use_hadamard = use_hadamard
        
        # Initialize weight with Xavier initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
        # Hadamard transform matrix (lazy initialization)
        self._hadamard_matrix = None
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def get_hadamard_matrix(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get or create Hadamard matrix."""
        if self._hadamard_matrix is None or self._hadamard_matrix.shape[0] != n:
            self._hadamard_matrix = HadamardTransform.hadamard_matrix(n).to(device, dtype)
        return self._hadamard_matrix
    
    def quantize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize weight using STE."""
        return STEQuantizer.apply(weight, self.weight_bits)
    
    def quantize_activation(self, activation: torch.Tensor) -> torch.Tensor:
        """Quantize activation with optional Hadamard transform."""
        if self.use_hadamard:
            # Apply Hadamard transform to reduce outliers
            transformed = HadamardTransform.apply(activation)
            # Quantize the transformed activation
            quantized = STEQuantizer.apply(transformed, self.activation_bits)
            # Inverse Hadamard transform
            H = self.get_hadamard_matrix(activation.shape[-1], activation.device, activation.dtype)
            H_inv = H.T  # Hadamard matrix is orthogonal: H^T = H^-1
            
            # Reshape for matrix multiplication
            original_shape = quantized.shape
            quantized_flat = quantized.reshape(-1, activation.shape[-1])
            dequantized = torch.matmul(quantized_flat, H_inv.T)
            
            return dequantized.reshape(original_shape)
        else:
            return STEQuantizer.apply(activation, self.activation_bits)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        # Quantize weight
        weight_quantized = self.quantize_weight(self.weight)
        
        # Quantize activation (input)
        input_quantized = self.quantize_activation(input)
        
        # Linear operation with quantized weights and activations
        output = F.linear(input_quantized, weight_quantized, self.bias)
        
        return output

@dataclass
class BitNetV2Config:
    """Configuration for BitNet v2 quantization."""
    weight_bits: int = 1  # 1.58bit weights
    activation_bits: int = 4  # 4bit activations
    use_hadamard: bool = True  # Use Hadamard transform
    mixed_precision: bool = False  # Enable mixed precision
    sensitive_layers: list = None  # Layers to exclude from quantization
    calibration_samples: int = 128  # Number of calibration samples
    
    def __post_init__(self):
        if self.sensitive_layers is None:
            self.sensitive_layers = []

class BitNetV2Quantizer:
    """BitNet v2 quantizer with native 4-bit activation quantization."""
    
    def __init__(self, config: Optional[BitNetV2Config] = None):
        self.config = config or BitNetV2Config()
        self.metrics = QuantizationMetrics()
    
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply BitNet v2 quantization to the entire model."""
        logger.info("Applying BitNet v2 quantization", 
                   weight_bits=self.config.weight_bits,
                   activation_bits=self.config.activation_bits,
                   use_hadamard=self.config.use_hadamard)
        
        # Replace Linear layers with H-BitLinear
        self._replace_linear_layers(model)
        
        # Calibrate if needed
        if self.config.calibration_samples > 0:
            self._calibrate_model(model)
        
        return model
    
    def _replace_linear_layers(self, model: nn.Module):
        """Replace Linear layers with H-BitLinear layers."""
        def replace_module(module, name, parent):
            if isinstance(module, nn.Linear) and name not in self.config.sensitive_layers:
                # Create H-BitLinear layer
                hbit_linear = HBitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    weight_bits=self.config.weight_bits,
                    activation_bits=self.config.activation_bits,
                    use_hadamard=self.config.use_hadamard,
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )
                
                # Copy weights (will be quantized during forward pass)
                hbit_linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    hbit_linear.bias.data = module.bias.data.clone()
                
                setattr(parent, name, hbit_linear)
                logger.debug(f"Replaced {name} with H-BitLinear")
        
        # Recursively replace all Linear layers
        for name, module in model.named_modules():
            if '.' in name:
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = dict(model.named_modules())[parent_name]
                replace_module(module, child_name, parent)
            elif name != '':  # Skip the root module
                replace_module(module, name, model)
    
    def _calibrate_model(self, model: nn.Module):
        """Calibrate the quantized model."""
        logger.info("Calibrating BitNet v2 model", samples=self.config.calibration_samples)
        # Placeholder for calibration logic
        # In practice, this would run forward passes with calibration data
        # to determine optimal quantization parameters
    
    def get_metrics(self) -> QuantizationMetrics:
        """Get quantization metrics."""
        return self.metrics