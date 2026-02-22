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

"""
FP4 (4-bit Floating Point) Training Implementation

Complete implementation of native FP4 training for extreme memory efficiency.
FP4 uses only 16 representable values, enabling 50% memory savings over FP8.

Key Innovation:
    - E2M1 format: 1 sign bit + 2 exponent bits + 1 mantissa bit
    - Block-wise quantization for reduced precision loss
    - Stochastic rounding for statistical accuracy
    - FP32 master weights for convergence
    - Mixed precision gradient computation

Reference:
    "Optimizing Large Language Model Training Using FP4 Quantization" (arXiv:2501.17116)
    "Quartet: Native FP4 Training Can Be Optimal for Large Language Models" (2025)

FP4 Format (E2M1):
    - Sign: 1 bit
    - Exponent: 2 bits (bias = 1)
    - Mantissa: 1 bit
    - Representable values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    - Dynamic range: [-6, 6]

Memory Savings:
    - FP32: 32 bits per value
    - FP16/BF16: 16 bits per value
    - FP8: 8 bits per value
    - FP4: 4 bits per value (75% savings vs FP16, 50% vs FP8)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import math
import time
import struct

from configs.version import VERSION
from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig,
)


FP4_E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

FP4_E2M1_MAX = 6.0
FP4_E2M1_MIN = -6.0


@dataclass
class POPSSFP4Config(PiscesLxOperatorConfig):
    """
    FP4 (4-bit Floating Point) Training Configuration.
    
    This configuration controls the FP4 quantization and training parameters
    for extreme memory efficiency.
    
    Attributes:
        fp4_format: FP4 format type (E2M1, E1M2)
        block_size: Block size for block-wise quantization
        stochastic_rounding: Whether to use stochastic rounding
        master_weights_dtype: Data type for master weights (fp32/bf16)
        gradient_accumulation_dtype: Data type for gradient accumulation
        amax_history_length: Length of amax history for dynamic scaling
        use_dynamic_scaling: Whether to use dynamic scaling
        scale_update_interval: Steps between scale updates
        activation_dtype: Data type for activations (fp8/fp16/bf16)
        weight_dtype: Data type for weights (fp4)
        gradient_dtype: Data type for gradients (fp8/fp16)
        compute_dtype: Data type for compute (fp16/bf16/fp32)
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
    """
    name: str = "fp4"
    version: str = VERSION
    
    fp4_format: str = "E2M1"
    block_size: int = 16
    stochastic_rounding: bool = True
    master_weights_dtype: str = "fp32"
    gradient_accumulation_dtype: str = "fp16"
    amax_history_length: int = 256
    use_dynamic_scaling: bool = True
    scale_update_interval: int = 100
    activation_dtype: str = "fp8"
    weight_dtype: str = "fp4"
    gradient_dtype: str = "fp8"
    compute_dtype: str = "bf16"
    enable_gradient_checkpointing: bool = True
    
    model_path: str = "./checkpoints/model"
    output_dir: str = "./checkpoints/fp4_output"
    train_data: str = "./data/train.jsonl"
    batch_size: int = 32
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    max_steps: int = 100000
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    
    checkpoint_interval: int = 5000
    checkpoint_dir: str = "./checkpoints/fp4_checkpoints"
    save_optimizer_state: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if self.fp4_format not in ["E2M1", "E1M2"]:
            raise ValueError(f"Unknown FP4 format: {self.fp4_format}")


class POPSSFP4Quantizer:
    """
    FP4 Quantization and Dequantization.
    
    Provides efficient FP4 quantization with block-wise scaling
    and optional stochastic rounding.
    """
    
    def __init__(
        self,
        fp4_format: str = "E2M1",
        block_size: int = 16,
        stochastic_rounding: bool = True,
    ):
        self.fp4_format = fp4_format
        self.block_size = block_size
        self.stochastic_rounding = stochastic_rounding
        
        if fp4_format == "E2M1":
            self.values = FP4_E2M1_VALUES
            self.max_val = FP4_E2M1_MAX
            self.min_val = FP4_E2M1_MIN
        else:
            self.values = self._compute_e1m2_values()
            self.max_val = 3.0
            self.min_val = -3.0
    
    def _compute_e1m2_values(self) -> torch.Tensor:
        """Compute E1M2 format values (1 exponent bit, 2 mantissa bits)."""
        values = []
        for sign in [0, 1]:
            for exp in range(2):
                for mant in range(4):
                    if exp == 0:
                        v = mant * 0.25
                    else:
                        v = 1.0 + mant * 0.25
                    if sign:
                        v = -v
                    values.append(v)
        return torch.tensor(sorted(set(values)), dtype=torch.float32)
    
    def quantize(
        self,
        tensor: torch.Tensor,
        block_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to FP4.
        
        Args:
            tensor: Input tensor to quantize
            block_size: Block size for quantization (overrides default)
        
        Returns:
            Tuple of (quantized_indices, scale_factors)
        """
        block_size = block_size or self.block_size
        
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()
        
        num_elements = tensor_flat.numel()
        num_blocks = (num_elements + block_size - 1) // block_size
        
        padded_size = num_blocks * block_size
        if padded_size > num_elements:
            padding = torch.zeros(padded_size - num_elements, dtype=tensor.dtype, device=tensor.device)
            tensor_flat = torch.cat([tensor_flat, padding])
        
        tensor_blocks = tensor_flat.view(num_blocks, block_size)
        
        block_max = tensor_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scale_factors = block_max / self.max_val
        
        scaled_blocks = tensor_blocks / scale_factors
        
        scaled_blocks = torch.clamp(scaled_blocks, self.min_val, self.max_val)
        
        if self.stochastic_rounding and tensor.requires_grad:
            quantized_blocks = self._stochastic_quantize(scaled_blocks)
        else:
            quantized_blocks = self._nearest_quantize(scaled_blocks)
        
        quantized_flat = quantized_blocks.flatten()[:num_elements]
        quantized_indices = quantized_flat.view(original_shape)
        
        return quantized_indices, scale_factors.squeeze(-1)
    
    def dequantize(
        self,
        quantized_indices: torch.Tensor,
        scale_factors: torch.Tensor,
        original_shape: Optional[Tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """
        Dequantize FP4 indices back to floating point.
        
        Args:
            quantized_indices: FP4 indices (0-15)
            scale_factors: Per-block scale factors
            original_shape: Original tensor shape (optional)
        
        Returns:
            Dequantized tensor
        """
        indices_flat = quantized_indices.flatten().long()
        
        values_device = self.values.to(quantized_indices.device)
        dequantized_flat = values_device[indices_flat.clamp(0, 15)]
        
        block_size = self.block_size
        num_elements = dequantized_flat.numel()
        num_blocks = (num_elements + block_size - 1) // block_size
        
        padded_size = num_blocks * block_size
        if padded_size > num_elements:
            padding = torch.zeros(padded_size - num_elements, dtype=dequantized_flat.dtype, device=dequantized_flat.device)
            dequantized_flat = torch.cat([dequantized_flat, padding])
        
        dequantized_blocks = dequantized_flat.view(num_blocks, block_size)
        
        if scale_factors.numel() == num_blocks:
            scaled_blocks = dequantized_blocks * scale_factors.unsqueeze(-1)
        else:
            scaled_blocks = dequantized_blocks * scale_factors
        
        result = scaled_blocks.flatten()[:num_elements]
        
        if original_shape:
            result = result.view(original_shape)
        
        return result
    
    def _nearest_quantize(self, scaled_tensor: torch.Tensor) -> torch.Tensor:
        """Nearest neighbor quantization."""
        values_device = self.values.to(scaled_tensor.device)
        
        distances = torch.abs(scaled_tensor.unsqueeze(-1) - values_device)
        indices = torch.argmin(distances, dim=-1)
        
        return indices.float()
    
    def _stochastic_quantize(self, scaled_tensor: torch.Tensor) -> torch.Tensor:
        """Stochastic rounding quantization."""
        values_device = self.values.to(scaled_tensor.device)
        
        distances = torch.abs(scaled_tensor.unsqueeze(-1) - values_device)
        
        _, top2_indices = torch.topk(distances, k=2, largest=False, dim=-1)
        
        idx0 = top2_indices[..., 0]
        idx1 = top2_indices[..., 1]
        
        val0 = values_device[idx0]
        val1 = values_device[idx1]
        
        total_dist = (val1 - val0).abs() + 1e-8
        prob0 = (val1 - scaled_tensor).abs() / total_dist
        
        rand = torch.rand_like(scaled_tensor)
        selected = torch.where(rand < prob0, idx0.float(), idx1.float())
        
        return selected


class POPSSFP4Operator(PiscesLxOperatorInterface):
    """
    FP4 Training Operator.
    
    Provides complete FP4 training functionality including quantization,
    forward/backward passes, and optimization.
    
    Key Features:
        - Native FP4 weight storage
        - Block-wise quantization
        - Stochastic rounding
        - FP32 master weights
        - Mixed precision training
    
    Example:
        >>> config = POPSSFP4Config(block_size=16)
        >>> fp4 = POPSSFP4Operator()
        >>> result = fp4.execute({
        ...     "model": model,
        ...     "dataloader": train_loader,
        ...     "config": config,
        ... })
    """
    
    def __init__(self):
        super().__init__()
        self._name = "fp4"
        self._version = VERSION
        self.quantizer = None
        self.master_weights = {}
        self.amax_history = {}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Native FP4 Training - 16-value extreme quantization"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute FP4 training step.
        
        Args:
            inputs: Dictionary containing:
                - model: Model to train
                - batch: Training batch
                - config: FP4 configuration
                - optimizer: Optimizer
                - step: Current training step
        
        Returns:
            PiscesLxOperatorResult with training statistics
        """
        start_time = self._get_time()
        
        try:
            model = inputs.get("model")
            batch = inputs.get("batch")
            config = inputs.get("config", POPSSFP4Config())
            optimizer = inputs.get("optimizer")
            step = inputs.get("step", 0)
            
            if model is None:
                raise ValueError("Model is required for FP4 training")
            
            if self.quantizer is None:
                self.quantizer = POPSSFP4Quantizer(
                    fp4_format=config.fp4_format,
                    block_size=config.block_size,
                    stochastic_rounding=config.stochastic_rounding,
                )
            
            if not self.master_weights:
                self._initialize_master_weights(model, config)
            
            stats = self._fp4_forward_backward(
                model=model,
                batch=batch,
                config=config,
                step=step,
            )
            
            if optimizer:
                self._fp4_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                )
            
            execution_time = self._get_time() - start_time
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "loss": stats.get("loss", 0.0),
                    "grad_norm": stats.get("grad_norm", 0.0),
                    "scale_factor": stats.get("scale_factor", 1.0),
                    "quantization_error": stats.get("quantization_error", 0.0),
                },
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "algorithm": "FP4",
                    "fp4_format": config.fp4_format,
                    "block_size": config.block_size,
                },
            )
            
        except Exception as e:
            execution_time = self._get_time() - start_time
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "error_type": type(e).__name__,
                },
            )
    
    def _initialize_master_weights(
        self,
        model: nn.Module,
        config: POPSSFP4Config,
    ):
        """Initialize FP32 master weights from model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                master_dtype = torch.float32 if config.master_weights_dtype == "fp32" else torch.bfloat16
                self.master_weights[name] = param.data.clone().to(master_dtype)
                
                self.amax_history[name] = torch.zeros(config.amax_history_length)
    
    def _fp4_forward_backward(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        config: POPSSFP4Config,
        step: int,
    ) -> Dict[str, float]:
        """Perform FP4 forward and backward pass."""
        stats = {}
        
        self._quantize_weights_to_fp4(model, config)
        
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        
        if input_ids is None:
            return {"loss": 0.0, "grad_norm": 0.0}
        
        compute_dtype = self._get_compute_dtype(config.compute_dtype)
        
        with torch.cuda.amp.autocast(enabled=True, dtype=compute_dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs[0]
            
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = outputs.loss if hasattr(outputs, "loss") else torch.tensor(0.0)
        
        stats["loss"] = loss.item()
        
        loss.backward()
        
        self._quantize_gradients_to_fp8(model, config)
        
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        stats["grad_norm"] = total_norm
        
        return stats
    
    def _quantize_weights_to_fp4(
        self,
        model: nn.Module,
        config: POPSSFP4Config,
    ):
        """Quantize master weights to FP4 for forward pass."""
        for name, param in model.named_parameters():
            if name not in self.master_weights:
                continue
            
            master_weight = self.master_weights[name]
            
            indices, scale = self.quantizer.quantize(master_weight, config.block_size)
            
            dequantized = self.quantizer.dequantize(indices, scale, master_weight.shape)
            
            param.data.copy_(dequantized.to(param.dtype))
    
    def _quantize_gradients_to_fp8(
        self,
        model: nn.Module,
        config: POPSSFP4Config,
    ):
        """Quantize gradients to FP8 for memory efficiency."""
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            if config.gradient_dtype == "fp8":
                grad_max = grad.abs().max().clamp(min=1e-8)
                scale = 448.0 / grad_max
                scaled_grad = grad * scale
                
                if config.stochastic_rounding:
                    quantized_grad = self._stochastic_round_to_fp8(scaled_grad)
                else:
                    quantized_grad = scaled_grad.round().clamp(-448, 448)
                
                param.grad.data = quantized_grad / scale
    
    def _stochastic_round_to_fp8(self, tensor: torch.Tensor) -> torch.Tensor:
        """Stochastic rounding to FP8 E4M3 range."""
        rounded = tensor.round()
        
        noise = torch.rand_like(tensor) - 0.5
        stochastic_rounded = rounded + noise.sign()
        
        return stochastic_rounded.clamp(-448, 448)
    
    def _fp4_optimizer_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: POPSSFP4Config,
    ):
        """Perform optimizer step with FP4 weight update."""
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        optimizer.step()
        
        for name, param in model.named_parameters():
            if name in self.master_weights and param.grad is not None:
                self.master_weights[name].copy_(param.data.to(self.master_weights[name].dtype))
        
        optimizer.zero_grad()
    
    def _get_compute_dtype(self, dtype_str: str) -> torch.dtype:
        """Get torch dtype from string."""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)
    
    def _get_time(self) -> float:
        """Get current time in seconds."""
        return time.time()
    
    def save_checkpoint(
        self,
        model: nn.Module,
        path: str,
        config: POPSSFP4Config,
    ):
        """Save FP4 checkpoint with master weights."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "master_weights": self.master_weights,
            "amax_history": self.amax_history,
            "config": config.__dict__,
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(
        self,
        model: nn.Module,
        path: str,
    ):
        """Load FP4 checkpoint with master weights."""
        checkpoint = torch.load(path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        self.master_weights = checkpoint["master_weights"]
        self.amax_history = checkpoint["amax_history"]
        
        return checkpoint.get("config", {})


class POPSSFP4Trainer:
    """
    High-level FP4 Trainer.
    
    Provides complete training loop with FP4 quantization.
    
    Example:
        >>> config = POPSSFP4Config(
        ...     block_size=16,
        ...     stochastic_rounding=True,
        ... )
        >>> trainer = POPSSFP4Trainer(model, config)
        >>> trainer.train(train_loader, num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[POPSSFP4Config] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.config = config or POPSSFP4Config()
        
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optimizer
        
        self.operator = POPSSFP4Operator()
        self.step = 0
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 1,
        save_dir: Optional[str] = None,
        save_every: int = 1000,
    ) -> Dict[str, Any]:
        """
        Train model with FP4 quantization.
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of training epochs
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N steps
        
        Returns:
            Training statistics
        """
        all_losses = []
        
        for epoch in range(num_epochs):
            for batch in dataloader:
                if isinstance(batch, dict):
                    pass
                elif isinstance(batch, (list, tuple)):
                    batch = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1] if len(batch) > 1 else None,
                        "labels": batch[2] if len(batch) > 2 else batch[0],
                    }
                
                result = self.operator.execute({
                    "model": self.model,
                    "batch": batch,
                    "config": self.config,
                    "optimizer": self.optimizer,
                    "step": self.step,
                })
                
                if result.status == PiscesLxOperatorStatus.SUCCESS:
                    all_losses.append(result.output["loss"])
                
                self.step += 1
                
                if save_dir and self.step % save_every == 0:
                    self._save_checkpoint(save_dir)
        
        return {
            "mean_loss": sum(all_losses) / len(all_losses) if all_losses else 0,
            "total_steps": self.step,
        }
    
    def _save_checkpoint(self, save_dir: str):
        """Save training checkpoint."""
        import os
        path = os.path.join(save_dir, f"checkpoint_{self.step}.pt")
        self.operator.save_checkpoint(self.model, path, self.config)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        self.operator.load_checkpoint(self.model, path)


class POPSSFP4Linear(nn.Module):
    """
    FP4 Linear Layer.
    
    A linear layer that stores weights in FP4 format with FP32 master weights.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 16,
        stochastic_rounding: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        self.quantizer = POPSSFP4Quantizer(
            block_size=block_size,
            stochastic_rounding=stochastic_rounding,
        )
        
        self.register_buffer(
            "weight_indices",
            torch.zeros(out_features, in_features // block_size, block_size // 2, dtype=torch.uint8),
        )
        self.register_buffer(
            "weight_scales",
            torch.ones(out_features, in_features // block_size),
        )
        self.master_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)),
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP4 weight dequantization."""
        weight = self.quantizer.dequantize(
            self.weight_indices.flatten().float(),
            self.weight_scales.flatten(),
            (self.out_features, self.in_features),
        )
        
        return F.linear(x, weight, self.bias)
    
    def quantize_weights(self):
        """Quantize master weights to FP4."""
        indices, scales = self.quantizer.quantize(self.master_weight.data, self.block_size)
        
        self.weight_indices.copy_(indices.view(self.weight_indices.shape).to(torch.uint8))
        self.weight_scales.copy_(scales.view(self.weight_scales.shape))
    
    def update_master_weights(self, grad: torch.Tensor, lr: float):
        """Update master weights from gradient."""
        with torch.no_grad():
            self.master_weight.add_(grad, alpha=-lr)
            self.quantize_weights()
