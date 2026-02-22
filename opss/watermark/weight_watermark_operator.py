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
Weight Watermark Operator

This module implements model weight watermarking for AI-generated model provenance.
It enables ownership verification and model attribution through imperceptible
weight modifications during training.

Weight Watermarking Principles:
    1. Select stable weight layers for watermark embedding
    2. Generate codebook from owner_id and seed
    3. Apply regularization loss to align weights with codebook
    4. Verify ownership through correlation analysis

Key Features:
    - Automatic layer selection (Top-N Linear/Conv layers)
    - Codebook generation with cryptographic seeding
    - Training-time regularization for watermark injection
    - Verification with configurable threshold
    - Distributed training synchronization support
    - Minimal impact on model performance

Watermark Injection:
    L_total = L_task + λ × L_watermark
    
    L_watermark = -strength × (w_norm · code)^2
    
    Where:
        w_norm: Normalized weight vector
        code: Codebook vector (±1)
        strength: Watermark strength coefficient

Verification Algorithm:
    1. Select same layers used during training
    2. Compute normalized correlations per layer
    3. Aggregate scores (mean or weighted average)
    4. Compare against threshold (default: 0.02)

Usage Examples:
    >>> from opss.watermark.weight_watermark_operator import PiscesLxWeightWatermarkOperator
    >>> operator = PiscesLxWeightWatermarkOperator()
    >>> 
    >>> # Configure operator
    >>> operator.configure(owner_id="owner123", strength=1e-5)
    >>> 
    >>> # Get regularization loss during training
    >>> loss = operator.get_regularization_loss(model)
    >>> 
    >>> # Verify model ownership
    >>> score, passed = operator.verify(model)

Author: PiscesL1 Development Team
Version: 1.0.0
"""

import math
import hashlib
import random
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

from utils.opsc.base import PiscesLxBaseOperator
from utils.opsc.interface import PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION
from .config import POPSSWatermarkConfig


class POPSSWeightWatermarkOperator(PiscesLxBaseOperator):
    """
    Model weight watermark operator.
    
    This operator provides weight-level watermarking capabilities for model
    provenance and ownership verification. It injects imperceptible patterns
    into model weights during training that can later be extracted for
    model attribution.
    
    Attributes:
        config (POPSSWatermarkConfig): Watermark configuration
        owner_id (str): Unique owner identifier
        seed (int): Random seed for reproducibility
        strength (float): Watermark injection strength
        top_k_layers (int): Number of layers to watermark
        min_layer_size (int): Minimum parameter count for selection
        
    Input Format:
        {
            "action": "regularize" | "verify" | "select_layers" | "configure",
            "model": torch.nn.Module,        # Model for regularization/verification
            "owner_id": str,                  # Owner identifier
            "strength": float,                # Watermark strength
            "threshold": float,               # Verification threshold
            "seed": int                       # Random seed
        }
        
    Output Format:
        {
            "action": str,
            "result": torch.Tensor | Tuple[float, bool],
            "metadata": Dict
        }
    """
    
    def __init__(self, config: Optional[POPSSWatermarkConfig] = None):
        super().__init__()
        self.name = "pisceslx_weight_watermark_operator"
        self.version = VERSION
        self.description = "Weight watermarking for model provenance and ownership verification"
        self.config = config or POPSSWatermarkConfig()
        self.owner_id = self.config.owner_id
        self.seed = hash(self.owner_id) & 0x7FFFFFFF
        self.strength = self.config.watermark_strength
        self.threshold = self.config.verify_threshold
        self.top_k_layers = 8
        self.min_layer_size = 256
        self._selected_layers: List[str] = []
        self._codebooks: Dict[str, torch.Tensor] = {}
        self._layer_stats: Dict[str, Dict] = defaultdict(lambda: {"correlation": 0.0, "magnitude": 0.0})
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["regularize", "verify", "select_layers", "configure", "get_stats"]
                },
                "model": {
                    "type": "object",
                    "description": "PyTorch model for processing"
                },
                "owner_id": {
                    "type": "string",
                    "description": "Unique owner identifier"
                },
                "strength": {
                    "type": "number",
                    "description": "Watermark injection strength"
                },
                "threshold": {
                    "type": "number",
                    "description": "Verification threshold"
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of layers to watermark"
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "result": {"type": "any"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "layers_watermarked": {"type": "integer"},
                        "avg_correlation": {"type": "number"},
                        "verification_passed": {"type": "boolean"},
                        "watermark_score": {"type": "number"}
                    }
                }
            }
        }
    
    def _execute_impl(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        action = inputs.get("action", "regularize")
        
        if action == "configure":
            return self._configure(inputs)
        elif action == "select_layers":
            return self._select_layers(inputs)
        elif action == "regularize":
            return self._regularize(inputs)
        elif action == "verify":
            return self._verify(inputs)
        elif action == "get_stats":
            return self._get_stats(inputs)
        else:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Unknown action: {action}"
            )
    
    def _configure(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        try:
            if inputs.get("owner_id"):
                self.owner_id = inputs["owner_id"]
                self.seed = hash(self.owner_id) & 0x7FFFFFFF
            
            if inputs.get("strength"):
                self.strength = float(inputs["strength"])
                if not 1e-8 <= self.strength <= 1e-2:
                    return PiscesLxOperatorResult(
                        operator_name=self.name,
                        status=PiscesLxOperatorStatus.FAILED,
                        error=f"Strength must be between 1e-8 and 1e-2, got {self.strength}"
                    )
            
            if inputs.get("threshold"):
                self.threshold = float(inputs["threshold"])
                if not 0 < self.threshold < 1:
                    return PiscesLxOperatorResult(
                        operator_name=self.name,
                        status=PiscesLxOperatorStatus.FAILED,
                        error=f"Threshold must be between 0 and 1, got {self.threshold}"
                    )
            
            if inputs.get("top_k"):
                self.top_k_layers = int(inputs["top_k"])
            
            if inputs.get("seed"):
                self.seed = int(inputs["seed"]) & 0x7FFFFFFF
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "configured": True,
                    "owner_id": self.owner_id,
                    "strength": self.strength,
                    "threshold": self.threshold,
                    "seed": self.seed,
                    "top_k_layers": self.top_k_layers
                },
                metadata={"action": "configure"}
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _select_layers(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        model = inputs.get("model")
        
        if not isinstance(model, nn.Module):
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Model must be a torch.nn.Module"
            )
        
        try:
            candidates = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if not any(k in name.lower() for k in ["weight"]):
                    continue
                if param.dim() not in [2, 4]:
                    continue
                if param.numel() < self.min_layer_size:
                    continue
                
                candidates.append((name, param))
            
            candidates.sort(key=lambda x: x[0])
            
            selected = [(name, param) for name, param in candidates[:self.top_k_layers]]
            self._selected_layers = [name for name, _ in selected]
            
            for name, param in selected:
                self._generate_codebook(name, param)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "selected_layers": self._selected_layers,
                    "layer_count": len(self._selected_layers)
                },
                metadata={
                    "action": "select_layers",
                    "candidates_evaluated": len(candidates),
                    "min_size_threshold": self.min_layer_size
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _regularize(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        model = inputs.get("model")
        
        if not isinstance(model, nn.Module):
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Model must be a torch.nn.Module"
            )
        
        try:
            if not self._selected_layers:
                select_result = self._select_layers({"model": model})
                if not select_result.is_success():
                    return select_result
            
            total_loss = torch.tensor(0.0, device=self._get_device(model))
            layer_contributions = {}
            
            for name, param in model.named_parameters():
                if name not in self._selected_layers:
                    continue
                if param.grad is None:
                    continue
                
                codebook = self._codebooks.get(name)
                if codebook is None:
                    self._generate_codebook(name, param)
                    codebook = self._codebooks.get(name)
                
                if codebook is None:
                    continue
                
                w = param.view(-1)
                if w.numel() < self.min_layer_size:
                    continue
                
                w_norm = torch.nn.functional.normalize(w, dim=0)
                code = codebook.to(w.device)
                corr = (w_norm * code).sum()
                
                layer_loss = -self.strength * (corr ** 2)
                total_loss = total_loss + layer_loss
                
                layer_contributions[name] = layer_loss.item()
            
            if isinstance(total_loss, torch.Tensor):
                total_loss = total_loss.detach()
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "regularization_loss": total_loss,
                    "layer_contributions": layer_contributions
                },
                metadata={
                    "action": "regularize",
                    "layers_processed": len(layer_contributions),
                    "strength": self.strength
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _verify(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        model = inputs.get("model")
        
        if not isinstance(model, nn.Module):
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Model must be a torch.nn.Module"
            )
        
        try:
            if not self._selected_layers:
                select_result = self._select_layers({"model": model})
                if not select_result.is_success():
                    return select_result
            
            scores = []
            details = []
            
            for name, param in model.named_parameters():
                if name not in self._selected_layers:
                    continue
                
                codebook = self._codebooks.get(name)
                if codebook is None:
                    self._generate_codebook(name, param)
                    codebook = self._codebooks.get(name)
                
                if codebook is None:
                    continue
                
                w = param.detach().view(-1)
                if w.numel() < self.min_layer_size:
                    continue
                
                w_norm = torch.nn.functional.normalize(w, dim=0)
                code = codebook.to(w.device)
                corr = (w_norm * code).sum().item()
                
                scores.append(corr)
                details.append({
                    "layer": name,
                    "correlation": corr,
                    "passed": corr > self.threshold
                })
            
            if not scores:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="No layers available for verification"
                )
            
            avg_score = sum(scores) / len(scores)
            passed = avg_score > self.threshold
            
            for i, detail in enumerate(details):
                self._layer_stats[detail["layer"]]["correlation"] = scores[i]
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "verification_score": avg_score,
                    "passed": passed,
                    "threshold": self.threshold,
                    "details": details
                },
                metadata={
                    "action": "verify",
                    "layers_evaluated": len(scores),
                    "avg_correlation": avg_score
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _get_stats(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        try:
            stats = {
                "owner_id": self.owner_id,
                "strength": self.strength,
                "threshold": self.threshold,
                "seed": self.seed,
                "top_k_layers": self.top_k_layers,
                "selected_layers": self._selected_layers,
                "layer_statistics": dict(self._layer_stats),
                "codebook_generated": len(self._codebooks) > 0
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=stats,
                metadata={"action": "get_stats"}
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _generate_codebook(self, layer_name: str, param: torch.Tensor) -> torch.Tensor:
        rnd = random.Random(hash((self.owner_id, self.seed, layer_name)))
        dim = min(param.numel(), 65536)
        
        code = torch.tensor(
            [1.0 if rnd.random() > 0.5 else -1.0 for _ in range(dim)],
            dtype=param.dtype,
            device=param.device
        )
        code = code / math.sqrt(dim)
        
        self._codebooks[layer_name] = code
        return code
    
    def _get_device(self, model: nn.Module) -> torch.device:
        try:
            for param in model.parameters():
                if param.device:
                    return param.device
            return torch.device("cpu")
        except Exception:
            return torch.device("cpu")
    
    def configure(self, owner_id: str = None, strength: float = None, 
                  threshold: float = None, seed: int = None) -> None:
        """Configure watermark operator parameters."""
        if owner_id:
            self.owner_id = owner_id
            self.seed = hash(owner_id) & 0x7FFFFFFF
        if strength:
            self.strength = strength
        if threshold:
            self.threshold = threshold
        if seed:
            self.seed = seed & 0x7FFFFFFF
    
    def select_layers(self, model: nn.Module) -> List[str]:
        """Select layers for watermark embedding."""
        result = self._select_layers({"model": model})
        if result.is_success():
            return result.output["selected_layers"]
        raise ValueError(f"Layer selection failed: {result.error}")
    
    def get_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Get watermark regularization loss for training."""
        result = self._regularize({"model": model})
        if result.is_success():
            return result.output["regularization_loss"]
        raise ValueError(f"Regularization failed: {result.error}")
    
    def verify(self, model: nn.Module) -> Tuple[float, bool]:
        """Verify model ownership."""
        result = self._verify({"model": model})
        if result.is_success():
            return result.output["verification_score"], result.output["passed"]
        raise ValueError(f"Verification failed: {result.error}")
    
    def get_watermark_score(self, model: nn.Module) -> float:
        """Get watermark score without threshold comparison."""
        score, _ = self.verify(model)
        return score


def _select_parameter_slices(model: nn.Module, top_k_layers: int = 8, 
                            min_size: int = 256) -> List[torch.nn.Parameter]:
    """Select stable subset of Linear/Conv weights to host watermark.
    
    This selection should be deterministic across runs based on parameter
    names order.
    
    Args:
        model: PyTorch model to select from
        top_k_layers: Number of top layers to select
        min_size: Minimum parameter count for selection
        
    Returns:
        List of selected parameter tensors
    """
    candidates = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name.lower() for k in ["weight"]) and (p.dim() in [2, 4]):
            if p.numel() >= min_size:
                candidates.append((name, p))
    
    candidates.sort(key=lambda x: x[0])
    return [p for _, p in candidates[:top_k_layers]]


def _codebook(owner_id: str, seed: int, dim: int, device=None, dtype=None) -> torch.Tensor:
    """Generate codebook vector from owner_id and seed.
    
    The codebook consists of ±1 values normalized to unit length,
    providing a deterministic embedding direction.
    
    Args:
        owner_id: Unique owner identifier
        seed: Random seed for reproducibility
        dim: Codebook dimension
        device: Target device for tensor
        dtype: Target dtype for tensor
        
    Returns:
        Codebook tensor of shape (dim,)
    """
    rnd = random.Random(hash((owner_id, seed)))
    vec = torch.tensor(
        [1.0 if rnd.random() > 0.5 else -1.0 for _ in range(dim)],
        dtype=dtype or torch.float32,
        device=device
    )
    return vec / math.sqrt(dim)


def watermarked_regularizer(param: torch.Tensor, owner_id: str, seed: int, 
                          strength: float = 1e-5) -> torch.Tensor:
    """Compute regularization loss to align parameter with codebook.
    
    L = strength * (dot(w_flat_norm, code))^2
    
    Minimizing this loss maximizes the correlation between the weight
    vector and the codebook, embedding the watermark.
    
    Args:
        param: Weight parameter tensor
        owner_id: Owner identifier for codebook generation
        seed: Random seed for reproducibility
        strength: Watermark strength coefficient
        
    Returns:
        Regularization loss tensor (scalar)
    """
    w = param.view(-1)
    if w.numel() < 256:
        return w.new_tensor(0.0)
    
    code = _codebook(owner_id, seed, w.numel(), w.device, w.dtype)
    w_norm = torch.nn.functional.normalize(w, dim=0)
    corr = (w_norm * code).sum()
    
    return strength * (-corr)


def verify_weights(model: nn.Module, owner_id: str, seed: int = 42,
                 top_k_layers: int = 8, threshold: float = 0.02) -> Tuple[float, bool]:
    """Aggregate correlation score over selected layers.
    
    Args:
        model: Model to verify
        owner_id: Expected owner identifier
        seed: Random seed for codebook generation
        top_k_layers: Number of layers to evaluate
        threshold: Verification threshold
        
    Returns:
        Tuple of (average_score, passed)
    """
    params = _select_parameter_slices(model, top_k_layers=top_k_layers)
    if not params:
        return 0.0, False
    
    scores = []
    for p in params:
        w = p.detach().view(-1)
        if w.numel() < 256:
            continue
        code = _codebook(owner_id, seed, w.numel(), w.device, w.dtype)
        w_norm = torch.nn.functional.normalize(w, dim=0)
        corr = (w_norm * code).sum().item()
        scores.append(corr)
    
    if not scores:
        return 0.0, False
    
    avg_score = sum(scores) / len(scores)
    passed = avg_score > threshold
    
    return avg_score, passed


def create_weight_watermark_operator(
    config: Optional[POPSSWatermarkConfig] = None
) -> POPSSWeightWatermarkOperator:
    """Factory function to create a weight watermark operator."""
    return POPSSWeightWatermarkOperator(config=config)


PiscesLxWeightWatermarkOperator = POPSSWeightWatermarkOperator


class POPSSWatermarkWeightOperator(PiscesLxBaseOperator):
    """
    Enhanced Weight Watermark Operator with unified function interfaces.
    
    This class combines all weight watermarking functions into a cohesive
    operator with methods for layer selection, codebook generation,
    regularization, and verification.
    
    Methods:
        select_parameter_slices: Select stable weight layers for watermarking
        generate_codebook: Generate cryptographic codebook from owner_id
        get_regularization_loss: Compute watermark regularization loss
        verify: Verify model ownership through correlation analysis
        create: Factory method to create operator instance
    """
    
    def __init__(self, config: Optional[POPSSWatermarkConfig] = None):
        super().__init__()
        self.name = "pisceslx_weight_watermark_operator"
        self.version = VERSION
        self.description = "Model weight watermarking for provenance and ownership verification"
        self.config = config or POPSSWatermarkConfig()
    
    @staticmethod
    def select_parameter_slices(model: nn.Module, top_k_layers: int = 8, 
                               min_layer_size: int = 10000) -> List[torch.Tensor]:
        """Select stable weight layers for watermark embedding."""
        parameter_list = []
        
        for name, param in model.named_parameters():
            if ('weight' in name or 'gate' in name) and param.ndim >= 2:
                if param.numel() >= min_layer_size:
                    parameter_list.append(param)
        
        parameter_list.sort(key=lambda x: x.numel(), reverse=True)
        
        return parameter_list[:top_k_layers]
    
    @staticmethod
    def generate_codebook(owner_id: str, seed: int, dim: int, 
                         device=None, dtype=None) -> torch.Tensor:
        """Generate cryptographic codebook from owner identifier."""
        hash_input = f"{owner_id}_{seed}".encode()
        hash_digest = hashlib.sha256(hash_input).hexdigest()
        
        seed_vals = [int(hash_digest[i:i+8], 16) for i in range(0, 32, 8)]
        rng = random.Random(seed_vals[0])
        
        code = torch.randn(dim, device=device, dtype=dtype)
        code = code / (torch.norm(code) + 1e-8)
        
        sign_pattern = [1 if rng.random() > 0.5 else -1 for _ in range(dim)]
        code = code * torch.tensor(sign_pattern, device=device, dtype=dtype)
        
        return code
    
    def get_regularization_loss(self, param: torch.Tensor, owner_id: str, 
                                seed: int = 42) -> torch.Tensor:
        """Compute regularization loss for watermark embedding."""
        codebook = POPSSWatermarkWeightOperator.generate_codebook(
            owner_id, seed, param.shape[0], 
            device=param.device, dtype=param.dtype
        )
        
        param_flat = param.view(-1)
        if param_flat.shape[0] > codebook.shape[0]:
            codebook = codebook.repeat(param_flat.shape[0] // codebook.shape[0] + 1)[:param_flat.shape[0]]
        else:
            codebook = codebook[:param_flat.shape[0]]
        
        codebook = codebook.to(param.dtype)
        
        dot_product = torch.sum(param_flat * codebook)
        wm_loss = -self.config.watermark_strength * (dot_product ** 2)
        
        return wm_loss
    
    def verify(self, model: nn.Module, owner_id: str, seed: int = 42,
              threshold: float = None) -> Tuple[float, bool]:
        """Verify model ownership through correlation analysis."""
        threshold = threshold or self.config.verify_threshold
        
        selected_params = POPSSWatermarkWeightOperator.select_parameter_slices(
            model, top_k_layers=self.config.top_k_layers
        )
        
        scores = []
        
        for param in selected_params:
            codebook = POPSSWatermarkWeightOperator.generate_codebook(
                owner_id, seed, param.shape[0],
                device=param.device, dtype=param.dtype
            )
            
            param_flat = param.view(-1)
            if param_flat.shape[0] > codebook.shape[0]:
                codebook_expanded = torch.zeros_like(param_flat)
                codebook_expanded[:codebook.shape[0]] = codebook
                codebook = codebook_expanded
            else:
                codebook = codebook[:param_flat.shape[0]].to(param.dtype)
            
            param_normalized = param_flat / (torch.norm(param_flat) + 1e-8)
            score = torch.abs(torch.sum(param_normalized * codebook)).item()
            scores.append(score)
        
        if not scores:
            return 0.0, False
        
        avg_score = sum(scores) / len(scores)
        passed = avg_score > threshold
        
        return avg_score, passed
    
    @classmethod
    def create(cls, config: Optional[POPSSWatermarkConfig] = None) -> 'POPSSWatermarkWeightOperator':
        """Factory method to create a weight watermark operator."""
        return cls(config=config)


__all__ = [
    "POPSSWeightWatermarkOperator",
    "POPSSWatermarkWeightOperator",
    "_select_parameter_slices",
    "_codebook",
    "watermarked_regularizer",
    "verify_weights",
    "create_weight_watermark_operator"
]
