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
K-FAC Natural Gradient Preconditioning Operator

This operator implements Kronecker-Factored Approximate Curvature (K-FAC)
natural gradient optimization for deep neural networks. K-FAC provides
a computationally efficient approximation of the Fisher Information Matrix,
enabling more effective gradient updates by preconditioning with curvature
information.

Key Features:
    - Kronecker-factored Fisher matrix approximation
    - Automatic layer-wise covariance computation
    - Adaptive damping coefficient adjustment
    - Efficient inverse computation via eigen decomposition
    - Support for common layer types (Linear, Conv2d, Embedding)
    - Optional momentum-based updates for stability

Background:
    The Fisher Information Matrix (FIM) describes the local curvature of
    the loss landscape. For large models, computing the exact FIM is
    computationally prohibitive. K-FAC approximates the FIM using the
    Kronecker product of input and output covariances:
    
    F ≈ E[aa^T] ⊗ E[gg^T]
    
    Where:
    - a: Activations (pre-activations or inputs)
    - g: Gradients (output gradients)
    
    This approximation enables:
    1. Efficient storage (O(n^2) → O(n))
    2. Fast inversion via eigendecomposition
    3. Diagonalization in the eigenbasis for stability

Algorithm:
    1. Forward pass: Store activations and outputs
    2. Backward pass: Store gradients
    3. At update frequency:
       a. Compute covariances: A = E[aa^T], G = E[gg^T]
       b. Eigen decomposition: A = V_A Λ_A V_A^T, G = V_G Λ_G V_G^T
       c. Compute inverse in eigenbasis: F⁻¹ ≈ V_A ⊗ V_G (Λ_A ⊗ Λ_G + λI)⁻¹ V_A^T ⊗ V_G^T
       d. Apply preconditioned gradients

Reference:
    - Martens & Grosse (2015). Optimizing Neural Networks with Kronecker-factored
      Approximate Curvature. ICML 2015.
    - Grosse & Martens (2016). A Kronecker-factored Approximate Fisher Matrix
      for Convolution Layers. ICML 2016.
    - Ba et al. (2016). Distributed Second-Order Optimization using
      Kronecker-Factored Approximations.

Dependencies:
    - torch >= 2.0.0
    - numpy (optional, for eigenvalue computations)

Usage Examples:
    >>> from opss.train.kfac import (
    ...     POPSSKFacConfig,
    ...     POPSSKFacOperator
    ... )
    
    >>> config = POPSSKFacConfig(
    ...     damping=0.001,
    ...     update_freq=100,
    ...     diag_first=True,
    ...     adaptive_damping=True
    ... )
    
    >>> operator = POPSSKFacOperator(config)
    >>> result = operator.execute({"model": model})

Integration:
    This operator integrates with PiscesLxTrainingEngine through the
    AdvancedTrainingCoordinator in opss/train/impl.py. It should be
    executed after backward pass and before optimizer.step().

See Also:
    - opss/train/gradient_optimization.py: Generic gradient optimization
    - opss/train/moe_gradient.py: MoE-specific gradient handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import time

from configs.version import VERSION
from utils.dc import PiscesLxLogger

from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig
)


class POPSSKFacLayerType(Enum):
    """
    Classification of layer types for K-FAC computation.
    
    Different layer types require different covariance computation strategies
    and have different parameterization patterns.
    
    Types:
        LINEAR: Fully connected layers
        CONV2D: 2D convolutional layers
        EMBEDDING: Token embedding layers
        LAYER_NORM: Layer normalization (diagonal only)
        BATCH_NORM: Batch normalization (special handling)
        ATTENTION: Multi-head attention (composite)
    """
    LINEAR = "linear"
    CONV2D = "conv2d"
    EMBEDDING = "embedding"
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"
    ATTENTION = "attention"


@dataclass
class POPSSKFacConfig(PiscesLxOperatorConfig):
    """
    Configuration for K-FAC natural gradient preconditioning.
    
    This configuration controls how K-FAC approximations are computed and
    applied, including damping, update frequency, and stability options.
    
    Attributes:
        damping: Base damping coefficient for numerical stability
               (prevents division by near-zero eigenvalues)
        update_freq: Steps between K-FAC matrix updates
                    (higher = faster but less accurate)
        diag_first: Use diagonal approximation for first-order factors
                   (reduces memory, less accurate)
        adaptive_damping: Enable automatic damping adjustment based on
                         gradient curvature
        damping_upper_bound: Maximum allowed damping value
        damping_lower_bound: Minimum allowed damping value
        damping_scale_factor: Factor for adaptive damping adjustment
        momentum: Momentum factor for running average of covariances
                 (0.0 = no momentum, 1.0 = no updates)
        include_bias: Include bias parameters in K-FAC computation
        include_embedding: Include embedding layers in K-FAC computation
        clip_gradients: Apply gradient clipping before K-FAC application
        grad_clip_norm: Maximum gradient norm for clipping
        use_eigen_decomp: Use eigen decomposition for inverse computation
                         (more stable but slower)
        precision_mode: Precision for covariance computation (full, half, auto)
        layer_selection: List of layer patterns to include (empty = all)
        layer_exclusion: List of layer patterns to exclude
        warmup_steps: Steps before K-FAC optimization activates
        verbose_logging: Enable detailed K-FAC statistics logging
        
    Default Values:
        damping: 0.001
        update_freq: 100
        diag_first: True
        adaptive_damping: True
        momentum: 0.95
        include_bias: True
        include_embedding: False
        warmup_steps: 100
        
    Example:
        >>> config = POPSSKFacConfig(
        ...     damping=0.001,
        ...     update_freq=100,
        ...     momentum=0.95,
        ...     adaptive_damping=True
        ... )
    """
    damping: float = 0.001
    update_freq: int = 100
    diag_first: bool = True
    adaptive_damping: bool = True
    damping_upper_bound: float = 1.0
    damping_lower_bound: float = 1e-8
    damping_scale_factor: float = 0.95
    momentum: float = 0.95
    include_bias: bool = True
    include_embedding: bool = False
    clip_gradients: bool = True
    grad_clip_norm: float = 1.0
    use_eigen_decomp: bool = True
    precision_mode: str = "auto"
    layer_selection: List[str] = field(default_factory=list)
    layer_exclusion: List[str] = field(default_factory=lambda: [
        'layernorm', 'batchnorm', 'norm', 'embedding', 'head', 'lm_head'
    ])
    warmup_steps: int = 100
    verbose_logging: bool = False
    
    def __post_init__(self):
        self.name = "kfac_natural_gradient"
        self.version = VERSION


class POPSSKFacOperator(PiscesLxOperatorInterface):
    """
    K-FAC natural gradient preconditioning operator.
    
    This operator implements efficient natural gradient optimization using
    the Kronecker-factored Approximate Curvature matrix. It provides:
    
    1. Automatic Layer Classification:
       - Identifies layer types for appropriate K-FAC computation
       - Supports Linear, Conv2d, LayerNorm (diagonal), Embedding
    
    2. Covariance Computation:
       - Running average of input/output covariances
       - Momentum-based updates for stability
       - Memory-efficient storage
    
    3. Inverse Application:
       - Eigen decomposition for stable inversion
       - Adaptive damping based on gradient statistics
       - Gradient preconditioning in eigenbasis
    
    Attributes:
        config: POPSSKFacConfig instance
        layer_info: Dictionary tracking layer information
        cov_a: Dictionary of input covariances (A = E[aa^T])
        cov_g: Dictionary of output covariances (G = E[gg^T])
        eigen_a: Dictionary of A eigen-decomposition results
        eigen_g: Dictionary of G eigen-decomposition results
        damping: Current damping coefficient
        step_count: Number of steps since initialization
        
    Thread Safety:
        This operator maintains per-instance state. Each training process
        should use its own operator instance to avoid race conditions.
        
    Memory Usage:
        O(n * d²) where n = number of layers, d = layer dimension.
        Diagonal approximations reduce to O(n * d).
    """
    
    def __init__(self, config: Optional[POPSSKFacConfig] = None):
        """
        Initialize the K-FAC operator.
        
        Args:
            config: Optional configuration instance. If None, default config
                   with standard K-FAC parameters is used.
        """
        super().__init__(config)
        self.config = config or POPSSKFacConfig()
        self.layer_info: Dict[str, Dict] = {}
        self.cov_a: Dict[str, torch.Tensor] = {}
        self.cov_g: Dict[str, torch.Tensor] = {}
        self.eigen_a: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.eigen_g: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.damping: float = self.config.damping
        self.step_count: int = 0
        self._is_warmed_up: bool = False
        
        self._logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup operator logger."""
        return get_logger("pisceslx.opss.train.kfac")
    
    @property
    def name(self) -> str:
        """Get operator name identifier."""
        return "kfac_natural_gradient"
    
    @property
    def version(self) -> str:
        """Get operator semantic version."""
        return VERSION
    
    @property
    def description(self) -> str:
        """Get operator description."""
        return (
            "K-FAC natural gradient preconditioning with adaptive damping, "
            "Fisher matrix approximation, and efficient inverse application"
        )
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        Define expected input format for this operator.
        
        Returns:
            Dictionary describing required and optional input fields.
        """
        return {
            "type": "object",
            "required": ["model"],
            "properties": {
                "model": {
                    "type": "torch.nn.Module",
                    "description": "Model for K-FAC gradient preconditioning"
                },
                "step": {
                    "type": "int",
                    "description": "Current training step (for update frequency)"
                },
                "activations": {
                    "type": "dict",
                    "description": "Optional pre-computed activations dict"
                },
                "gradients": {
                    "type": "dict",
                    "description": "Optional pre-computed gradients dict"
                },
                "forward_pass": {
                    "type": "bool",
                    "description": "Flag indicating forward pass phase"
                },
                "backward_pass": {
                    "type": "bool",
                    "description": "Flag indicating backward pass phase"
                },
                "update_kfac": {
                    "type": "bool",
                    "description": "Force K-FAC matrix update this step"
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        """
        Define output format from this operator.
        
        Returns:
            Dictionary describing output fields and their types.
        """
        return {
            "type": "object",
            "properties": {
                "preconditioning_applied": {
                    "type": "bool",
                    "description": "Whether gradient preconditioning was applied"
                },
                "matrices_updated": {
                    "type": "bool",
                    "description": "Whether K-FAC matrices were updated"
                },
                "current_damping": {
                    "type": "float",
                    "description": "Current damping coefficient"
                },
                "num_layers_processed": {
                    "type": "int",
                    "description": "Number of layers with K-FAC applied"
                },
                "gradient_norm_before": {
                    "type": "float",
                    "description": "Gradient norm before preconditioning"
                },
                "gradient_norm_after": {
                    "type": "float",
                    "description": "Gradient norm after preconditioning"
                },
                "curvature_factor": {
                    "type": "float",
                    "description": "Average curvature factor from Fisher"
                },
                "update_efficiency": {
                    "type": "float",
                    "description": "Ratio of actual to expected update"
                }
            }
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """
        Validate input parameters before execution.
        
        Args:
            inputs: Dictionary of input parameters from caller.
            
        Returns:
            True if all required inputs are valid, False otherwise.
        """
        if "model" not in inputs:
            self._logger.error("Missing required input: model")
            return False
        
        model = inputs["model"]
        if not isinstance(model, nn.Module):
            self._logger.error(
                f"Invalid model type: {type(model)}. "
                "Expected torch.nn.Module instance."
            )
            return False
        
        if list(model.parameters()) == []:
            self._logger.error("Model has no parameters")
            return False
        
        self._logger.debug("Input validation passed")
        return True
    
    def _detect_layer_type(
        self,
        module: nn.Module
    ) -> POPSSKFacLayerType:
        """
        Detect the type of a neural network module.
        
        Args:
            module: The module to classify.
            
        Returns:
            POPSSKFacLayerType enum value.
        """
        if isinstance(module, nn.Linear):
            return POPSSKFacLayerType.LINEAR
        elif isinstance(module, nn.Conv2d):
            return POPSSKFacLayerType.CONV2D
        elif isinstance(module, nn.Embedding):
            return POPSSKFacLayerType.EMBEDDING
        elif isinstance(module, nn.LayerNorm):
            return POPSSKFacLayerType.LAYER_NORM
        elif isinstance(module, nn.BatchNorm2d):
            return POPSSKFacLayerType.BATCH_NORM
        else:
            for child in module.children():
                if isinstance(child, nn.MultiheadAttention):
                    return POPSSKFacLayerType.ATTENTION
            return POPSSKFacLayerType.LINEAR
    
    def _register_hooks(
        self,
        model: nn.Module
    ) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register forward and backward hooks for layer activation capture.
        
        Args:
            model: Model to instrument with hooks.
            
        Returns:
            List of hook handles for cleanup.
        """
        handles = []
        
        def forward_hook(module, input, output):
            layer_id = id(module)
            if layer_id not in self.layer_info:
                layer_type = self._detect_layer_type(module)
                self.layer_info[layer_id] = {
                    'type': layer_type,
                    'module': module,
                    'name': module.__class__.__name__
                }
            
            activation = input[0]
            if activation.requires_grad:
                if layer_id not in self.cov_a:
                    self._init_covariance(layer_id, activation)
                
                self._update_input_covariance(layer_id, activation)
        
        def backward_hook(module, grad_input, grad_output):
            layer_id = id(module)
            if layer_id not in self.layer_info:
                return
            
            grad_output_tensor = grad_output[0]
            if grad_output_tensor.requires_grad:
                if layer_id not in self.cov_g:
                    self._init_output_covariance(layer_id, grad_output_tensor)
                
                self._update_output_covariance(layer_id, grad_output_tensor)
        
        for module in model.modules():
            layer_type = self._detect_layer_type(module)
            if layer_type in [POPSSKFacLayerType.LINEAR, POPSSKFacLayerType.CONV2D]:
                handle_f = module.register_forward_hook(forward_hook)
                handle_b = module.register_backward_hook(backward_hook)
                handles.extend([handle_f, handle_b])
        
        return handles
    
    def _init_covariance(
        self,
        layer_id: int,
        activation: torch.Tensor
    ) -> None:
        """
        Initialize input covariance matrix for a layer.
        
        Args:
            layer_id: Unique identifier for the layer.
            activation: Sample activation tensor.
        """
        layer_info = self.layer_info[layer_id]
        layer_type = layer_info['type']
        
        if layer_type == POPSSKFacLayerType.LINEAR:
            dim = activation.shape[-1]
            if self.config.diag_first:
                self.cov_a[layer_id] = torch.zeros(dim, device=activation.device)
            else:
                self.cov_a[layer_id] = torch.eye(dim, device=activation.device)
        
        elif layer_type == POPSSKFacLayerType.CONV2D:
            batch, channels, height, width = activation.shape
            dim = channels * height * width
            if self.config.diag_first:
                self.cov_a[layer_id] = torch.zeros(dim, device=activation.device)
            else:
                self.cov_a[layer_id] = torch.eye(dim, device=activation.device)
        
        layer_info['input_dim'] = dim
    
    def _init_output_covariance(
        self,
        layer_id: int,
        grad_output: torch.Tensor
    ) -> None:
        """
        Initialize output gradient covariance matrix for a layer.
        
        Args:
            layer_id: Unique identifier for the layer.
            grad_output: Sample gradient tensor.
        """
        layer_info = self.layer_info[layer_id]
        layer_type = layer_info['type']
        
        if layer_type == POPSSKFacLayerType.LINEAR:
            dim = grad_output.shape[-1]
            if self.config.diag_first:
                self.cov_g[layer_id] = torch.zeros(dim, device=grad_output.device)
            else:
                self.cov_g[layer_id] = torch.eye(dim, device=grad_output.device)
        
        elif layer_type == POPSSKFacLayerType.CONV2D:
            batch, channels, height, width = grad_output.shape
            dim = channels * height * width
            if self.config.diag_first:
                self.cov_g[layer_id] = torch.zeros(dim, device=grad_output.device)
            else:
                self.cov_g[layer_id] = torch.eye(dim, device=grad_output.device)
        
        layer_info['output_dim'] = dim
    
    def _update_input_covariance(
        self,
        layer_id: int,
        activation: torch.Tensor
    ) -> None:
        """
        Update input covariance with running average.
        
        Args:
            layer_id: Unique identifier for the layer.
            activation: Current activation tensor.
        """
        layer_type = self.layer_info[layer_id]['type']
        
        if layer_type == POPSSKFacLayerType.LINEAR:
            activation = activation.view(-1, activation.shape[-1])
        
        elif layer_type == POPSSKFacLayerType.CONV2D:
            batch, channels, height, width = activation.shape
            activation = activation.view(batch, -1)
        
        if self.config.diag_first:
            ss = (activation ** 2).mean(dim=0)
        else:
            ss = activation.t() @ activation / activation.shape[0]
        
        if layer_id in self.cov_a:
            self.cov_a[layer_id] = (
                self.config.momentum * self.cov_a[layer_id] +
                (1 - self.config.momentum) * ss
            )
        else:
            self.cov_a[layer_id] = ss
    
    def _update_output_covariance(
        self,
        layer_id: int,
        grad_output: torch.Tensor
    ) -> None:
        """
        Update output gradient covariance with running average.
        
        Args:
            layer_id: Unique identifier for the layer.
            grad_output: Current gradient tensor.
        """
        layer_type = self.layer_info[layer_id]['type']
        
        if layer_type == POPSSKFacLayerType.LINEAR:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
        
        elif layer_type == POPSSKFacLayerType.CONV2D:
            batch, channels, height, width = grad_output.shape
            grad_output = grad_output.view(batch, -1)
        
        if self.config.diag_first:
            ss = (grad_output ** 2).mean(dim=0)
        else:
            ss = grad_output.t() @ grad_output / grad_output.shape[0]
        
        if layer_id in self.cov_g:
            self.cov_g[layer_id] = (
                self.config.momentum * self.cov_g[layer_id] +
                (1 - self.config.momentum) * ss
            )
        else:
            self.cov_g[layer_id] = ss
    
    def _compute_eigen_decomposition(
        self,
        cov: torch.Tensor,
        layer_id: int,
        is_input: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigen decomposition of a covariance matrix.
        
        Args:
            cov: Covariance matrix.
            layer_id: Unique identifier for the layer.
            is_input: Whether this is input (A) or output (G) covariance.
            
        Returns:
            Tuple of (eigenvalues, eigenvectors).
        """
        if self.config.use_eigen_decomp:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        else:
            eigenvalues = cov.diag()
            eigenvectors = torch.eye(cov.shape[0], device=cov.device)
        
        storage_key = f"{'a' if is_input else 'g'}_eigen"
        if is_input:
            self.eigen_a[layer_id] = (eigenvalues, eigenvectors)
        else:
            self.eigen_g[layer_id] = (eigenvalues, eigenvectors)
        
        return eigenvalues, eigenvectors
    
    def _apply_preconditioning(
        self,
        model: nn.Module
    ) -> Tuple[float, float]:
        """
        Apply K-FAC preconditioning to model gradients.
        
        Args:
            model: Model whose gradients to precondition.
            
        Returns:
            Tuple of (gradient norm before, gradient norm after).
        """
        total_norm_before = 0.0
        total_norm_after = 0.0
        processed_layers = 0
        
        for layer_id, layer_info in self.layer_info.items():
            if layer_id not in self.cov_a or layer_id not in self.cov_g:
                continue
            
            module = layer_info['module']
            if not hasattr(module, 'weight') or module.weight.grad is None:
                continue
            
            grad = module.weight.grad.data
            
            grad_norm_before = grad.norm(2).item()
            total_norm_before += grad_norm_before ** 2
            
            A_eigval, A_eigvec = self.eigen_a.get(layer_id, (None, None))
            G_eigval, G_eigvec = self.eigen_g.get(layer_id, (None, None))
            
            if A_eigval is None or G_eigval is None:
                A = self.cov_a[layer_id]
                G = self.cov_g[layer_id]
                A_eigval, A_eigvec = self._compute_eigen_decomposition(A, layer_id, True)
                G_eigval, G_eigvec = self._compute_eigen_decomposition(G, layer_id, False)
            
            if self.config.diag_first:
                A_inv = 1.0 / (A_eigval + self.damping)
                G_inv = 1.0 / (G_eigval + self.damping)
                
                grad_flat = grad.view(grad.shape[0], -1)
                grad_transformed = G_inv * (A_eigvec.t() @ grad_flat @ A_eigvec)
                grad.data.copy_(grad_transformed.view_as(grad))
            else:
                A_inv = torch.inverse(A_eigvec @ (A_eigval + self.damping) * A_eigvec.t())
                G_inv = torch.inverse(G_eigvec @ (G_eigval + self.damping) * G_eigvec.t())
                
                grad.data.copy_((A_inv @ grad @ G_inv))
            
            grad_norm_after = grad.norm(2).item()
            total_norm_after += grad_norm_after ** 2
            processed_layers += 1
        
        return total_norm_before ** 0.5, total_norm_after ** 0.5
    
    def _update_adaptive_damping(
        self,
        norm_before: float,
        norm_after: float
    ) -> None:
        """
        Update damping coefficient based on gradient statistics.
        
        Args:
            norm_before: Gradient norm before preconditioning.
            norm_after: Gradient norm after preconditioning.
        """
        if norm_before > 0 and self.config.adaptive_damping:
            ratio = norm_after / (norm_before + 1e-8)
            
            if ratio < 0.5:
                self.damping = min(
                    self.damping * self.config.damping_scale_factor,
                    self.config.damping_upper_bound
                )
            elif ratio > 2.0:
                self.damping = max(
                    self.damping / self.config.damping_scale_factor,
                    self.config.damping_lower_bound
                )
    
    def _compute_curvature_factor(self) -> float:
        """
        Compute average curvature factor from Fisher matrix.
        
        Returns:
            Average of (eigenvalues_A * eigenvalues_G) / (eigenvalues_A * eigenvalues_G + damping).
        """
        total_factor = 0.0
        count = 0
        
        for layer_id in self.eigen_a:
            if layer_id in self.eigen_g:
                A_eig = self.eigen_a[layer_id][0]
                G_eig = self.eigen_g[layer_id][0]
                
                if len(A_eig) == len(G_eig):
                    product = A_eig * G_eig
                    factor = (product / (product + self.damping)).mean()
                    total_factor += factor.item()
                    count += 1
        
        return total_factor / max(count, 1)
    
    def execute(
        self,
        inputs: Dict[str, Any],
        **kwargs
    ) -> PiscesLxOperatorResult:
        """
        Execute K-FAC gradient preconditioning.
        
        This method performs natural gradient optimization using K-FAC
        approximation, including covariance computation, eigen decomposition,
        and gradient preconditioning.
        
        Args:
            inputs: Dictionary containing:
                - model: PyTorch model for K-FAC processing
                - step: Current training step (optional)
                - forward_pass: Flag for forward pass phase (optional)
                - backward_pass: Flag for backward pass phase (optional)
                - update_kfac: Force matrix update this step (optional)
            **kwargs: Additional keyword arguments for flexibility.
            
        Returns:
            PiscesLxOperatorResult containing:
                - Preconditioning results and statistics
                - Updated gradient norms
                - Curvature information
                
        Raises:
            ValueError: If model is invalid.
            RuntimeError: If K-FAC processing fails unexpectedly.
        """
        start_time = time.time()
        
        try:
            if not self.validate_inputs(inputs):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Input validation failed",
                    execution_time=time.time() - start_time
                )
            
            model = inputs["model"]
            current_step = inputs.get("step", self.step_count)
            is_forward = inputs.get("forward_pass", False)
            is_backward = inputs.get("backward_pass", False)
            force_update = inputs.get("update_kfac", False)
            
            self.step_count += 1
            
            if current_step < self.config.warmup_steps:
                self._logger.debug(
                    f"Warmup: skipping K-FAC at step {current_step}"
                )
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output={
                        "preconditioning_applied": False,
                        "matrices_updated": False,
                        "current_damping": self.damping,
                        "num_layers_processed": 0,
                        "gradient_norm_before": 0.0,
                        "gradient_norm_after": 0.0,
                        "curvature_factor": 1.0,
                        "update_efficiency": 1.0,
                        "message": "Warmup phase - no processing applied"
                    },
                    execution_time=time.time() - start_time
                )
            
            should_update = (
                (current_step % self.config.update_freq == 0) or force_update
            )
            
            if should_update:
                for layer_id in self.layer_info:
                    if layer_id in self.cov_a:
                        A = self.cov_a[layer_id]
                        if layer_id not in self.eigen_a:
                            self._compute_eigen_decomposition(A, layer_id, True)
                    
                    if layer_id in self.cov_g:
                        G = self.cov_g[layer_id]
                        if layer_id not in self.eigen_g:
                            self._compute_eigen_decomposition(G, layer_id, False)
            
            preconditioning_applied = False
            norm_before = 0.0
            norm_after = 0.0
            
            if is_backward or should_update:
                preconditioning_applied = True
                
                if self.config.clip_gradients:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.grad_clip_norm
                    )
                    norm_before = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
                
                norm_before, norm_after = self._apply_preconditioning(model)
                
                self._update_adaptive_damping(norm_before, norm_after)
            
            curvature_factor = self._compute_curvature_factor()
            
            update_efficiency = (
                norm_after / (norm_before + 1e-8) if norm_before > 0 else 1.0
            )
            
            output = {
                "preconditioning_applied": preconditioning_applied,
                "matrices_updated": should_update,
                "current_damping": self.damping,
                "num_layers_processed": len(self.layer_info),
                "gradient_norm_before": norm_before,
                "gradient_norm_after": norm_after,
                "curvature_factor": curvature_factor,
                "update_efficiency": update_efficiency,
                "num_covariance_matrices": len(self.cov_a),
                "num_eigen_pairs": len(self.eigen_a)
            }
            
            if self.config.verbose_logging and preconditioning_applied:
                self._logger.info(
                    f"K-FAC Stats: norm_before={norm_before:.6f}, "
                    f"norm_after={norm_after:.6f}, "
                    f"damping={self.damping:.6e}, "
                    f"curvature={curvature_factor:.4f}"
                )
            
            self._logger.info(
                f"K-FAC preprocessing complete: "
                f"layers={len(self.layer_info)}, "
                f"damping={self.damping:.6e}"
            )
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=output,
                execution_time=time.time() - start_time,
                metadata={
                    "step": current_step,
                    "layer_count": len(self.layer_info),
                    "damping": self.damping
                }
            )
            
        except Exception as e:
            error_msg = f"K-FAC execution failed: {str(e)}"
            self._logger.error(error_msg, exc_info=True)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=error_msg,
                execution_time=time.time() - start_time,
                metadata={
                    "exception_type": type(e).__name__
                }
            )
    
    def register_hooks(self, model: nn.Module) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register forward/backward hooks for covariance computation.
        
        Args:
            model: Model to instrument.
            
        Returns:
            List of hook handles for cleanup.
        """
        return self._register_hooks(model)
    
    def get_covariance_matrices(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get current covariance matrices.
        
        Returns:
            Dictionary mapping layer IDs to their covariance matrices.
        """
        return {
            'input': {str(k): v for k, v in self.cov_a.items()},
            'output': {str(k): v for k, v in self.cov_g.items()}
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current K-FAC statistics.
        
        Returns:
            Dictionary containing layer count, damping value, etc.
        """
        return {
            "num_layers": len(self.layer_info),
            "num_covariance_matrices": len(self.cov_a),
            "current_damping": self.damping,
            "step_count": self.step_count,
            "is_warmed_up": self._is_warmed_up
        }
    
    def reset(self) -> None:
        """
        Reset operator state for a new training run.
        
        Clears all covariances, eigen decompositions, and statistics.
        """
        self.layer_info.clear()
        self.cov_a.clear()
        self.cov_g.clear()
        self.eigen_a.clear()
        self.eigen_g.clear()
        self.damping = self.config.damping
        self.step_count = 0
        self._is_warmed_up = False
        self._logger.info("K-FAC operator state reset")


class POPSSKFacFacade:
    """
    Convenience facade for quick K-FAC setup and usage.
    
    This facade provides a simplified interface for common K-FAC use cases,
    automatically handling hook registration and step coordination.
    
    Usage:
        >>> facade = POPSSKFacFacade()
        >>> 
        >>> # During forward pass
        >>> facade.register_hooks(model)
        >>> outputs = model(inputs)
        >>> 
        >>> # During backward pass
        >>> loss.backward()
        >>> facade.precondition_gradients(model, step=current_step)
    """
    
    def __init__(
        self,
        damping: float = 0.001,
        update_freq: int = 100,
        momentum: float = 0.95,
        adaptive_damping: bool = True,
        **kwargs
    ):
        """
        Initialize the facade with K-FAC parameters.
        
        Args:
            damping: Base damping coefficient
            update_freq: Steps between matrix updates
            momentum: Running average momentum
            adaptive_damping: Enable adaptive damping adjustment
            **kwargs: Additional configuration
        """
        config = POPSSKFacConfig(
            damping=damping,
            update_freq=update_freq,
            momentum=momentum,
            adaptive_damping=adaptive_damping,
            **kwargs
        )
        self.operator = POPSSKFacOperator(config)
        self._handles = []
    
    def register_hooks(self, model: nn.Module) -> None:
        """
        Register hooks for covariance computation.
        
        Args:
            model: Model to instrument.
        """
        self._handles = self.operator.register_hooks(model)
        self._logger.info(f"Registered {len(self._handles)//2} layer hooks")
    
    def precondition_gradients(
        self,
        model: nn.Module,
        step: int,
        is_backward: bool = True
    ) -> Dict[str, Any]:
        """
        Apply K-FAC gradient preconditioning.
        
        Args:
            model: Model whose gradients to precondition.
            step: Current training step.
            is_backward: Whether this is after backward pass.
            
        Returns:
            Dictionary of preconditioning statistics.
        """
        result = self.operator.execute({
            "model": model,
            "step": step,
            "backward_pass": is_backward,
            "update_kfac": False
        })
        
        if result.is_success():
            return result.output
        else:
            raise RuntimeError(f"K-FAC failed: {result.error}")
    
    def update_matrices(self, model: nn.Module, step: int) -> Dict[str, Any]:
        """
        Force update of K-FAC matrices.
        
        Args:
            model: Model for covariance computation.
            step: Current training step.
            
        Returns:
            Dictionary of update statistics.
        """
        result = self.operator.execute({
            "model": model,
            "step": step,
            "forward_pass": True,
            "update_kfac": True
        })
        
        if result.is_success():
            return result.output
        else:
            raise RuntimeError(f"K-FAC update failed: {result.error}")
    
    def cleanup(self) -> None:
        """Remove registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get K-FAC operator statistics."""
        return self.operator.get_statistics()


__all__ = [
    "POPSSKFacLayerType",
    "POPSSKFacConfig",
    "POPSSKFacOperator",
    "POPSSKFacFacade",
]
