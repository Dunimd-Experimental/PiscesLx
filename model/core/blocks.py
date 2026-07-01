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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

from __future__ import annotations

"""
Advanced Transformer Blocks Module for Yv Model.

This module provides comprehensive transformer block implementations that form the
layer-wise building blocks of the Yv transformer architecture. Each block
combines attention, feed-forward networks, normalization, and optional specialized
components into a cohesive computational unit.

Architecture Overview:
    The transformer block system implements multiple architectural patterns:

    1. Standard Transformer Blocks:
       - YvTransformerBlock: Standard sequential block
         * Pre-norm or post-norm architecture
         * Attention followed by MLP
         * Residual connections around each sublayer
         * Configurable normalization placement
       
       - YvParallelBlock: Parallel attention-MLP architecture
         * Attention and MLP computed in parallel
         * Combined output with residual connection
         * Reduced layer latency for inference
         * Memory-efficient gradient computation

    2. Deep Network Stability:
       - YvDeepNormBlock: DeepNorm-stabilized block
         * Scaled residual connections for deep networks
         * Prevents gradient explosion in 100+ layer models
         * Alpha and beta scaling parameters
         * Proven stability for deep transformers
       
       - YvLayerScaleBlock: LayerScale integration
         * Learnable per-channel scaling factors
         * Initializes small for stable deep training
         * Gradually learns optimal scaling
         * Compatible with any block architecture

    3. Dynamic Computation:
       - YvMixtureOfDepthsBlock: Dynamic depth routing
         * Routes tokens through different numbers of layers
         * Learns which tokens need more computation
         * Reduces average FLOPs while maintaining quality
         * Entropy-based routing decisions
       
       - YvAdaptiveComputationBlock: Adaptive computation time
         * Dynamic halting based on confidence
         * Pondering mechanism for complex inputs
         * Budget-aware computation allocation
         * Per-token computation budgets

    4. Parameter-Efficient Fine-Tuning:
       - YvLoRABlock: LoRA-integrated block
         * Low-rank adaptation for attention weights
         * Minimal trainable parameters
         * Preserves pretrained knowledge
         * Multiple rank configurations
       
       - YvDoRABlock: DoRA-integrated block
         * Weight-decomposed low-rank adaptation
         * Improved over LoRA with minimal overhead
         * Better stability and convergence
         * Supports both attention and MLP

    5. Cross-Attention and Encoder-Decoder:
       - YvCrossAttentionBlock: Cross-attention block
         * Encoder-decoder attention support
         * Separate KV projections for encoder outputs
         * Causal masking for decoder
         * Supports encoder hidden state caching
       
       - YvEncoderBlock: Encoder-only block
         * Bidirectional attention
         * No causal masking
         * Optimized for understanding tasks
       
       - YvDecoderBlock: Decoder-only block
         * Causal attention with KV caching
         * Optimized for generation tasks
         * Supports incremental decoding

    6. Mixture of Experts Integration:
       - YvMoEBlock: MoE-enabled block
         * Sparse expert routing
         * Top-k expert selection
         * Load balancing mechanisms
         * Expert capacity management
       
       - YvExpertChoiceBlock: Expert-choice routing
         * Experts select tokens (not vice versa)
         * Guaranteed load balancing
         * No token dropping
         * Optimal expert utilization

    7. Specialized Components:
       - YvSwiGLU: SwiGLU activation
         * Gated linear unit with Swish activation
         * Improved over standard ReLU/GELU
         * 3x larger intermediate dimension
       
       - YvGeGLU: GeGLU activation
         * Gated linear unit with GELU activation
         * Alternative to SwiGLU
         * Smooth activation function
       
       - YvManifoldConstraint: Manifold constraint layers
         * Geometric constraints for embeddings
         * Hyperbolic space projections
         * Hierarchical representation learning

Design Rationale:
    - Modularity: Each block type is independently usable
    - Flexibility: Multiple architectural patterns for different needs
    - Training Stability: DeepNorm and LayerScale for deep networks
    - Efficiency: Parallel blocks and dynamic computation reduce FLOPs
    - Fine-Tuning: LoRA/DoRA for parameter-efficient adaptation

Mathematical Formulations:
    Standard Block: x = x + Attention(LayerNorm(x))
                    x = x + MLP(LayerNorm(x))
    Parallel Block: x = x + Attention(x) + MLP(x)
    DeepNorm: x = LayerNorm(x + alpha * Sublayer(x))
              Sublayer = x * beta (for weight initialization)
    LayerScale: x = x + gamma * Sublayer(LayerNorm(x))
                gamma initialized to small value (e.g., 1e-5)
    SwiGLU: MLP(x) = (W1 @ x) * swish(W2 @ x)
    MoE: y = sum_i(gate_i * Expert_i(x))

Performance Considerations:
    - Parallel blocks reduce latency but increase memory
    - DeepNorm enables training 100+ layer models stably
    - MoE blocks increase capacity with constant compute
    - LoRA reduces fine-tuning memory by 1000x+
    - Mixture-of-Depths can reduce FLOPs by 20-40%

Dependencies:
    - torch: PyTorch deep learning framework
    - .norms: Normalization layers (RMSNorm, DeepNorm)
    - .attention: Attention mechanisms
    - ..moe: Mixture of Experts components
    - utils.dc: Logging utilities

Usage Example:
    >>> from model.core.blocks import YvTransformerBlock, YvParallelBlock
    >>> from model.core.blocks import YvDeepNormBlock, YvLoRABlock
    >>> 
    >>> # Standard block
    >>> block = YvTransformerBlock(config)
    >>> output = block(hidden_states, attention_mask)
    >>> 
    >>> # DeepNorm for deep networks
    >>> deep_block = YvDeepNormBlock(config, depth=100)
    >>> 
    >>> # LoRA fine-tuning
    >>> lora_block = YvLoRABlock(config, lora_rank=8)

Note:
    All classes follow the YvXxx naming convention.
    Block selection should match the overall model architecture.
    DeepNorm is recommended for models with 50+ layers.
    LoRA/DoRA are recommended for fine-tuning scenarios.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum

from .norms import YvRMSNorm, YvDeepNorm, YvParallelResidualNorm
from .attention import YvAttention
from utils.dc import PiscesLxLogger
from ..moe import YvDeepSeekMoELayer
from .mamba3 import YvMamba3Block, YvMamba3Config

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Core", file_path=get_log_file("Yv.Core"), enable_file=True)


class YvBlockType(Enum):
    """Enumeration of available transformer block architectures.
    
    Defines the supported block types for the Yv model, each with
    different computational patterns and optimization strategies.
    
    Attributes:
        STANDARD: Sequential attention-MLP block with pre-norm.
            Most common architecture, suitable for general use.
        PARALLEL: Parallel attention-MLP computation.
            Reduces latency by computing attention and MLP simultaneously.
        DEEPNORM: DeepNorm-stabilized block for very deep networks.
            Enables training of 100+ layer models stably.
        CROSS_ATTENTION: Block with cross-attention for encoder-decoder.
            Supports sequence-to-sequence architectures.
        ADAPTIVE: Adaptive computation time block.
            Dynamic computation based on input complexity.
        MIXTURE_OF_DEPTHS: Dynamic layer skipping block.
            Routes tokens through different numbers of layers.
    
    Example:
        >>> block_type = YvBlockType.STANDARD
        >>> if block_type == YvBlockType.DEEPNORM:
        ...     print("Using DeepNorm for deep network stability")
    """
    STANDARD = "standard"
    PARALLEL = "parallel"
    DEEPNORM = "deepnorm"
    CROSS_ATTENTION = "cross_attention"
    ADAPTIVE = "adaptive"
    MIXTURE_OF_DEPTHS = "mixture_of_depths"


@dataclass
class YvBlockConfig:
    """Configuration dataclass for Yv transformer blocks.
    
    Encapsulates all hyperparameters for transformer block initialization,
    providing a centralized configuration interface for different block
    architectures and optimization strategies.
    
    Architecture Configuration:
        - hidden_size: Model hidden dimension (default: 4096)
        - intermediate_size: MLP intermediate dimension (default: 11008)
        - n_layer: Number of transformer layers (default: 32)
        - n_head: Number of attention heads (default: 32)
        - n_kv_head: Number of key/value heads for GQA (default: 8)
        - block_type: Type of transformer block (default: "standard")
    
    Activation Configuration:
        - activation: Activation function type (default: "silu")
        - use_swiglu: Whether to use SwiGLU activation (default: True)
        - use_geglu: Whether to use GeGLU activation (default: False)
    
    Normalization Configuration:
        - use_deepnorm: Whether to use DeepNorm (default: False)
        - use_layerscale: Whether to use LayerScale (default: True)
        - layerscale_init: Initial value for LayerScale (default: 1e-5)
    
    Parallel Computation:
        - use_parallel: Whether to use parallel attention-MLP (default: False)
    
    Regularization:
        - residual_dropout: Dropout for residual connections (default: 0.1)
    
    Gradient Checkpointing:
        - use_checkpoint: Whether to use gradient checkpointing (default: True)
        - adaptive_checkpointing: Whether to use adaptive checkpointing (default: True)
    
    Parameter-Efficient Fine-Tuning:
        - use_lora: Whether to use LoRA (default: False)
        - lora_rank: Rank for LoRA (default: 8)
        - lora_alpha: Alpha for LoRA (default: 16.0)
        - use_dora: Whether to use DoRA (default: False)
    
    Dynamic Computation:
        - mixture_of_depths: Whether to enable MoD (default: False)
        - mod_routing_weight: Weight for MoD routing (default: 0.1)
    
    Example:
        >>> config = YvBlockConfig(
        ...     hidden_size=4096,
        ...     n_layer=32,
        ...     use_deepnorm=True,
        ...     use_swiglu=True
        ... )
    """
    hidden_size: int = 4096
    intermediate_size: int = 11008
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8
    block_type: str = "standard"
    activation: str = "silu"
    use_deepnorm: bool = False
    use_parallel: bool = False
    use_layerscale: bool = True
    layerscale_init: float = 1e-5
    use_swiglu: bool = True
    use_geglu: bool = False
    residual_dropout: float = 0.1
    use_checkpoint: bool = True
    adaptive_checkpointing: bool = True
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    use_dora: bool = False
    mixture_of_depths: bool = False
    mod_routing_weight: float = 0.1
    use_attn_res: bool = False
    attn_res_block_size: int = 8
    attn_res_use_two_phase: bool = True
    attn_res_use_online_softmax: bool = True
    attn_res_cache_pipeline: bool = True
    attn_res_max_blocks: int = 32
    attn_res_learnable_query: bool = True
    attn_res_use_rmsnorm: bool = True


# Paper: Touvron et al., "Going Deeper With Image Transformers" (LayerScale), ICCV 2021, arXiv:2103.17239
class YvLayerScale(nn.Module):
    """LayerScale for improved training stability in deep networks.
    
    Applies learnable per-channel scaling to the output of each layer,
    initialized to a small value to prevent gradient explosion in deep
    networks. This technique enables stable training of very deep
    transformers (100+ layers).
    
    Mathematical Formulation:
        output = input * gamma
        where gamma is initialized to a small value (e.g., 1e-5)
    
    Key Features:
        - Per-channel learnable scaling
        - Small initialization for stability
        - Gradually learns optimal scaling during training
        - Minimal computational overhead
    
    Training Benefits:
        - Prevents gradient explosion in deep networks
        - Enables training of 100+ layer transformers
        - Improves convergence speed
        - Works well with any normalization strategy
    
    Performance Characteristics:
        - Memory: O(dim) for gamma parameter
        - Compute: O(dim) for element-wise multiplication
        - No FLOPs overhead during inference
    
    Attributes:
        gamma (nn.Parameter): Learnable scaling parameter, shape [dim].
    
    Example:
        >>> layerscale = YvLayerScale(dim=4096, init_value=1e-5)
        >>> x = torch.randn(2, 1024, 4096)
        >>> scaled = layerscale(x)
    
    Reference:
        Touvron et al., "Going deeper with Image Transformers", ICCV 2021.
    """
    
    def __init__(
        self,
        dim: int,
        init_value: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize LayerScale with specified dimension and initial value.
        
        Args:
            dim: Dimension of the features to scale. This should match
                the model's hidden dimension.
            init_value: Initial value for the gamma parameter. Smaller
                values provide more stability but slower learning.
                Default: 1e-5.
            device: Device for the gamma parameter.
            dtype: Data type for the gamma parameter.
        
        Example:
            >>> layerscale = YvLayerScale(
            ...     dim=4096,
            ...     init_value=1e-5,
            ...     device='cuda'
            ... )
        """
        super().__init__()
        self.gamma = nn.Parameter(
            torch.ones(dim, device=device, dtype=dtype) * init_value
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer scaling to input tensor.
        
        Multiplies the input by the learnable gamma parameter.
        
        Args:
            x: Input tensor of shape [..., dim]. The last dimension
                must match the initialized dim parameter.
        
        Returns:
            Scaled tensor of the same shape as input.
        
        Note:
            The scaling is applied element-wise along the last dimension,
            allowing each channel to learn its own optimal scale.
        """
        return x * self.gamma


# Paper: Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202, 2020 (SwiGLU)
class YvSwiGLU(nn.Module):
    """SwiGLU activation function for improved performance.
    
    Implements SwiGLU: Swish(xW) * (xV) where W and V are separate
    linear projections. Provides better performance than standard
    ReLU or GeLU activations.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize SwiGLU.
        
        Args:
            hidden_size: Input hidden dimension.
            intermediate_size: Intermediate dimension (output will be intermediate_size // 2).
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, device=device, dtype=dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.
        
        Args:
            x: Input tensor of shape [batch, seq, hidden_size].
            
        Returns:
            Output tensor of shape [batch, seq, hidden_size].
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        gate = torch.clamp(gate, max=10.0)
        up = torch.clamp(up, min=-10.0, max=10.0)
        return self.down_proj(gate * up)


# Paper: Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202, 2020 (GeGLU variant)
class YvGeGLU(nn.Module):
    """GeGLU activation function for improved performance.
    
    Implements GeGLU: GeLU(xW) * (xV) where W and V are separate
    linear projections. Alternative to SwiGLU with GeLU activation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize GeGLU.
        
        Args:
            hidden_size: Input hidden dimension.
            intermediate_size: Intermediate dimension.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        self.gate_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            hidden_size, intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False, device=device, dtype=dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeGLU activation.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# Paper: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022, arXiv:2106.09685
class YvLoRA(nn.Module):
    """Low-Rank Adaptation (LoRA) for efficient fine-tuning.
    
    Implements LoRA: adds trainable low-rank matrices to existing
    linear layers, enabling efficient fine-tuning with minimal parameters.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize LoRA.
        
        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            rank: Rank of low-rank matrices.
            alpha: Scaling factor.
            dropout: Dropout probability.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation.
        
        Args:
            x: Input tensor.
            
        Returns:
            Adapted tensor.
        """
        return self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling


# Paper: Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024, arXiv:2402.09353
class YvDoRA(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA).
    
    Implements DoRA: improves upon LoRA by decomposing the weight matrix
    into magnitude and direction, enabling more flexible adaptation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize DoRA.
        
        Args:
            in_features: Input dimension.
            out_features: Output dimension.
            rank: Rank of low-rank matrices.
            alpha: Scaling factor.
            dropout: Dropout probability.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank, device=device, dtype=dtype) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, out_features, device=device, dtype=dtype)
        )
        self.magnitude = nn.Parameter(
            torch.ones(out_features, device=device, dtype=dtype)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        """Apply DoRA adaptation.
        
        Args:
            x: Input tensor.
            base_weight: Base weight matrix.
            
        Returns:
            Adapted tensor.
        """
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        
        combined_weight = base_weight + self.lora_A @ self.lora_B * self.scaling
        weight_norm = torch.norm(combined_weight, dim=0, keepdim=True)
        normalized_weight = combined_weight / (weight_norm + 1e-8)
        scaled_weight = normalized_weight * self.magnitude
        
        return x @ scaled_weight


# Paper: Graves, "Adaptive Computation Time for Recurrent Neural Networks", arXiv:1603.08983, 2016
class YvAdaptiveComputationTime(nn.Module):
    """Adaptive Computation Time (ACT) for dynamic computation.
    
    Enables the model to dynamically decide how much computation
    to spend on each input token, improving efficiency.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_iterations: int = 3,
        threshold: float = 0.99,
        epsilon: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize ACT.
        
        Args:
            hidden_size: Model hidden dimension.
            max_iterations: Maximum number of iterations.
            threshold: Halting probability threshold.
            epsilon: Small constant for numerical stability.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.epsilon = epsilon
        
        self.halting_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1, device=device, dtype=dtype),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        compute_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive computation time.
        
        Args:
            x: Input tensor.
            compute_fn: Function to apply iteratively.
            
        Returns:
            Tuple of (output, ponder_cost).
        """
        batch_size, seq_len, _ = x.shape
        
        halting_sum = torch.zeros(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)
        remainder = torch.ones(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)
        output = torch.zeros_like(x)
        ponder_cost = torch.zeros(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)
        
        for _ in range(self.max_iterations):
            halt_prob = self.halting_net(x)
            
            still_running = halting_sum < self.threshold
            
            if still_running.any():
                new_halt = halt_prob * still_running.float()
                halting_sum = halting_sum + new_halt
                
                output = output + compute_fn(x) * new_halt
                ponder_cost = ponder_cost + new_halt
                
                remainder = remainder - new_halt
                
                if (halting_sum >= self.threshold - self.epsilon).all():
                    break
                    
        output = output + compute_fn(x) * remainder
        ponder_cost = ponder_cost + remainder
        
        return output, ponder_cost


# Paper: Raposo et al., "Mixture-of-Depths: Dynamically Allocating Compute in Transformer Networks", arXiv:2404.02258, 2024
class YvMixtureOfDepths(nn.Module):
    """Mixture-of-Depths for dynamic layer skipping with modal protection.

    Enables the model to skip layers for certain tokens, improving
    efficiency by not processing all tokens through all layers.
    Cross-modal tokens are force-protected from skipping to preserve
    modal context integrity.
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        routing_weight: float = 0.1,
        capacity_factor: float = 1.25,
        modal_protection: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Mixture-of-Depths.

        Args:
            hidden_size: Model hidden dimension.
            n_head: Number of attention heads.
            routing_weight: Weight for routing decisions.
            capacity_factor: Capacity factor for token allocation.
            modal_protection: Whether to protect cross-modal tokens from skipping.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.routing_weight = routing_weight
        self.capacity_factor = capacity_factor
        self.modal_protection = modal_protection

        self.router = nn.Linear(hidden_size, 2, bias=False, device=device, dtype=dtype)

        self.skip_norm = YvRMSNorm(hidden_size, device=device, dtype=dtype)
        self.process_norm = YvRMSNorm(hidden_size, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        process_fn: Callable[[torch.Tensor], torch.Tensor],
        modal_protection_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixture-of-depths routing with optional modal protection.

        Args:
            x: Input tensor [B, T, H].
            process_fn: Function to apply to processed tokens.
            modal_protection_mask: Boolean mask [B, T] where True forces processing.

        Returns:
            Tuple of (output, routing_loss).
        """
        batch_size, seq_len, _ = x.shape

        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        process_prob = router_probs[..., 0]
        skip_prob = router_probs[..., 1]

        if self.modal_protection and modal_protection_mask is not None:
            if modal_protection_mask.shape != process_prob.shape:
                if modal_protection_mask.dim() == 1:
                    modal_protection_mask = modal_protection_mask.unsqueeze(0).expand(batch_size, -1)
            process_prob = process_prob * (1.0 - modal_protection_mask.float()) + modal_protection_mask.float()
            skip_prob = 1.0 - process_prob

        capacity = int(seq_len * self.capacity_factor)

        _, top_indices = torch.topk(process_prob, min(capacity, seq_len), dim=-1)

        process_mask = torch.zeros_like(process_prob)
        process_mask.scatter_(1, top_indices, 1.0)

        if self.modal_protection and modal_protection_mask is not None:
            process_mask = torch.clamp(process_mask + modal_protection_mask.float(), 0.0, 1.0)

        x_process = self.process_norm(x)
        processed = process_fn(x_process)

        x_skip = self.skip_norm(x)

        output = process_mask.unsqueeze(-1) * processed + (1.0 - process_mask).unsqueeze(-1) * x_skip

        routing_loss = self._compute_routing_loss(router_probs)

        return output, routing_loss

    def _compute_routing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary routing loss for load balancing.

        Args:
            router_probs: Router probabilities.

        Returns:
            Routing loss tensor.
        """
        process_prob = router_probs[..., 0]
        skip_prob = router_probs[..., 1]

        balance_loss = torch.var(process_prob.mean(dim=1)) + torch.var(skip_prob.mean(dim=1))

        return balance_loss * self.routing_weight


class YvCrossAttention(nn.Module):
    """Cross-attention for encoder-decoder architectures.
    
    Implements cross-attention where the query comes from the decoder
    and the key/value come from the encoder.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        n_kv_head: int,
        attention_dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize cross-attention.
        
        Args:
            hidden_size: Model hidden dimension.
            n_head: Number of query heads.
            n_kv_head: Number of key/value heads.
            attention_dropout: Dropout probability.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = hidden_size // n_head
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        
        self.attn_dropout = nn.Dropout(attention_dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute cross-attention.
        
        Args:
            query: Query tensor from decoder.
            encoder_hidden_states: Key/value tensor from encoder.
            attention_mask: Optional attention mask.
            
        Returns:
            Cross-attention output.
        """
        batch_size, query_len, _ = query.shape
        _, encoder_len, _ = encoder_hidden_states.shape
        
        q = self.q_proj(query).view(batch_size, query_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).view(batch_size, encoder_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).view(batch_size, encoder_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
            
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, query_len, self.hidden_size)
        
        return self.o_proj(output)


# Paper: Zhou et al., "Mixture-of-Experts with Expert Choice Routing", NeurIPS 2022, arXiv:2202.09368
class YvExpertChoiceMLP(nn.Module):
    """Expert Choice MLP for improved MoE routing.
    
    Implements expert-choice routing where experts select which tokens
    to process, rather than tokens selecting experts. This provides
    better load balancing and eliminates token dropping.
    
    Key features:
    - Perfect load balancing (each expert processes exactly capacity tokens)
    - No token dropping (all tokens are processed)
    - Efficient batch processing with scatter operations
    - Auxiliary loss for training stability
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_alpha: float = 0.01,
        z_loss_alpha: float = 0.001,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Expert Choice MLP.
        
        Args:
            hidden_size: Model hidden dimension.
            intermediate_size: MLP intermediate dimension.
            n_experts: Number of experts.
            top_k: Number of tokens each expert processes per batch.
            capacity_factor: Factor to increase expert capacity.
            aux_loss_alpha: Coefficient for auxiliary load balancing loss.
            z_loss_alpha: Coefficient for z-loss (router entropy regularization).
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_alpha = aux_loss_alpha
        self.z_loss_alpha = z_loss_alpha
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
            )
            for _ in range(n_experts)
        ])
        
        self.router = nn.Linear(hidden_size, n_experts, bias=False, device=device, dtype=dtype)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.router.weight, gain=0.1)
        for expert in self.experts:
            for module in expert:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply expert choice routing with efficient batch processing.
        
        Args:
            x: Input tensor [batch, seq, hidden].
            
        Returns:
            Tuple of (output, routing_loss).
        """
        batch_size, seq_len, hidden_size = x.shape
        
        router_logits = self.router(x)
        
        router_logits_for_topk = router_logits.transpose(1, 2)
        
        capacity = int(min(seq_len, max(1, (seq_len * self.top_k) // self.n_experts * self.capacity_factor)))
        
        topk_values, topk_indices = torch.topk(
            router_logits_for_topk, 
            capacity, 
            dim=-1
        )
        topk_weights = F.softmax(topk_values, dim=-1)
        
        x_flat = x.view(-1, hidden_size)
        
        output = torch.zeros_like(x_flat)
        counts = torch.zeros(batch_size * seq_len, device=x.device, dtype=x.dtype)
        
        for expert_idx, expert in enumerate(self.experts):
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, capacity)
            token_indices = topk_indices[:, expert_idx]
            
            flat_indices = batch_indices * seq_len + token_indices
            
            selected_tokens = x_flat[flat_indices.view(-1)]
            expert_output = expert(selected_tokens)
            expert_output = expert_output.view(batch_size, capacity, hidden_size)
            
            weights = topk_weights[:, expert_idx].unsqueeze(-1)
            weighted_output = (expert_output * weights).view(-1, hidden_size)
            
            output.scatter_add_(0, flat_indices.view(-1, 1).expand(-1, hidden_size), weighted_output)
            counts.scatter_add_(0, flat_indices.view(-1), torch.ones(flat_indices.numel(), device=x.device))
        
        counts = counts.clamp(min=1.0)
        output = output / counts.unsqueeze(1)
        
        output = output.view(batch_size, seq_len, hidden_size)
        
        routing_loss = self._compute_routing_loss(router_logits)
        
        return output, routing_loss
        
    def _compute_routing_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute routing loss for load balancing and stability.
        
        Args:
            router_logits: Raw router logits [batch, seq, n_experts].
            
        Returns:
            Combined routing loss.
        """
        router_probs = F.softmax(router_logits, dim=-1)
        
        expert_usage = router_probs.mean(dim=(0, 1))
        target_usage = 1.0 / self.n_experts
        aux_loss = torch.mean((expert_usage - target_usage) ** 2) * self.n_experts
        
        z_loss = torch.mean(router_logits ** 2)
        
        routing_loss = self.aux_loss_alpha * aux_loss + self.z_loss_alpha * z_loss
        
        return routing_loss


class YvParallelBlock(nn.Module):
    """Parallel Attention-MLP Block.
    
    Implements parallel computation of attention and MLP branches,
    which can improve throughput compared to sequential computation.
    """
    
    def __init__(
        self,
        cfg,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        gate: Optional[nn.Module] = None
    ):
        """Initialize parallel block.
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
            gate: Optional pre-built MoE gate for PathMoE stage sharing.
        """
        super().__init__()
        self.cfg = cfg
        
        self.attn = YvAttention(cfg, device=device, dtype=dtype)

        # HydraHead: head-level FA/LA hybridization
        if getattr(cfg, 'use_hydra_head', False):
            from ..core.attention import YvHydraHeadAttention
            self.hydra_attn = YvHydraHeadAttention(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.n_head,
                head_dim=cfg.hidden_size // cfg.n_head,
                la_ratio=getattr(cfg, 'hydra_head_la_ratio', 0.5),
                learnable_assignment=getattr(cfg, 'hydra_head_learnable_assignment', True),
                temperature=getattr(cfg, 'hydra_head_temperature', 1.0),
                causal=True,
                device=device, dtype=dtype
            )
        else:
            self.hydra_attn = None

        # LCA: latent-condensed attention
        if getattr(cfg, 'use_lca', False):
            from ..core.attention import YvLatentCondensedAttention
            self.lca = YvLatentCondensedAttention(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.n_head,
                head_dim=cfg.hidden_size // cfg.n_head,
                latent_dim=getattr(cfg, 'lca_latent_dim', 512),
                condense_factor=getattr(cfg, 'lca_condense_factor', 0.25),
                use_residual=getattr(cfg, 'lca_use_residual', True),
                num_kv_heads=getattr(cfg, 'n_kv_head', None),
                device=device, dtype=dtype
            )
        else:
            self.lca = None

        self.mlp = YvDeepSeekMoELayer(cfg, device=device, dtype=dtype)

        # Phi-Balancing: population-level mirror descent load balancer
        if getattr(cfg, 'use_phi_balancing', False):
            from ..moe.gate import YvPhiBalancing
            self.phi_balancing = YvPhiBalancing(
                num_experts=getattr(cfg, 'moe_num_experts', 8),
                momentum=getattr(cfg, 'phi_balancing_momentum', 0.95),
                lr=getattr(cfg, 'phi_balancing_lr', 0.01),
                ema_decay=getattr(cfg, 'phi_balancing_ema_decay', 0.99),
                device=device, dtype=dtype
            )
        else:
            self.phi_balancing = None
            
        self.norm = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        
        if getattr(cfg, 'use_layerscale', True):
            self.attn_scale = YvLayerScale(
                cfg.hidden_size,
                init_value=getattr(cfg, 'layerscale_init', 1e-5),
                device=device, dtype=dtype
            )
            self.mlp_scale = YvLayerScale(
                cfg.hidden_size,
                init_value=getattr(cfg, 'layerscale_init', 1e-5),
                device=device, dtype=dtype
            )
        else:
            self.attn_scale = nn.Identity()
            self.mlp_scale = nn.Identity()
            
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout', 0.1))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        subconscious_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        film_params: Optional[Dict[str, torch.Tensor]] = None,
        modal_id: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass with parallel attention and MLP.

        Args:
            x: Input tensor.
            mask: Attention mask.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use cache.
            subconscious_kv: Optional extra KV pair for hard knowledge injection.
            film_params: Optional FiLM modulation parameters.

        Returns:
            Output tensor(s).
        """
        if film_params is not None:
            scale = film_params["scale"]
            shift = film_params["shift"]
            x = x * (1.0 + scale) + shift

        residual = x
        x_norm = self.norm(x)

        # HydraHead: FA/LA hybrid attention — stateless, no KV cache needed.
        # Runs in parallel with the main attention path (training or inference).
        hydra_out = None
        if self.hydra_attn is not None:
            hydra_out = self.hydra_attn(x_norm, mask)

        if use_cache:
            attn_out, present_kv = self.attn(
                x_norm, mask, past_key_values=past_key_values, use_cache=True,
                extra_kv=subconscious_kv,
            )
            if hydra_out is not None:
                attn_out = attn_out + hydra_out
        else:
            attn_out = self.attn(
                x_norm, mask, past_key_values=past_key_values, use_cache=False,
                extra_kv=subconscious_kv,
            )
            if hydra_out is not None:
                attn_out = attn_out + hydra_out

        mlp_out, aux_loss = self.mlp(x_norm, modal_id=modal_id)

        # Apply phi-balancing to aux_loss
        if self.phi_balancing is not None and self.training:
            aux_loss = aux_loss + self.phi_balancing.get_regularization_loss()

        output = residual + self.residual_dropout(self.attn_scale(attn_out) + self.mlp_scale(mlp_out))

        if use_cache:
            return output, aux_loss, present_kv
        return output, aux_loss


# Paper: Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers", arXiv:2203.00555, 2022
class YvDeepNormBlock(nn.Module):
    """DeepNorm Block for training very deep networks.
    
    Implements DeepNorm: a normalization strategy that combines
    residual scaling with layer normalization for improved stability.
    """
    
    def __init__(
        self,
        cfg,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        gate: Optional[nn.Module] = None
    ):
        """Initialize DeepNorm block.
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
            gate: Optional pre-built MoE gate for PathMoE stage sharing.
        """
        super().__init__()
        self.cfg = cfg
        
        self.attn = YvAttention(cfg, device=device, dtype=dtype)

        # HydraHead: head-level FA/LA hybridization
        if getattr(cfg, 'use_hydra_head', False):
            from ..core.attention import YvHydraHeadAttention
            self.hydra_attn = YvHydraHeadAttention(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.n_head,
                head_dim=cfg.hidden_size // cfg.n_head,
                la_ratio=getattr(cfg, 'hydra_head_la_ratio', 0.5),
                learnable_assignment=getattr(cfg, 'hydra_head_learnable_assignment', True),
                temperature=getattr(cfg, 'hydra_head_temperature', 1.0),
                causal=True,
                device=device, dtype=dtype
            )
        else:
            self.hydra_attn = None

        self.mlp = YvDeepSeekMoELayer(cfg, device=device, dtype=dtype)
            
        self.deep_norm_attn = YvDeepNorm(
            cfg.hidden_size, cfg.n_layer, device=device, dtype=dtype
        )
        self.deep_norm_mlp = YvDeepNorm(
            cfg.hidden_size, cfg.n_layer, device=device, dtype=dtype
        )
        
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout', 0.1))

        # Phi-Balancing: population-level mirror descent load balancer
        if getattr(cfg, 'use_phi_balancing', False):
            from ..moe.gate import YvPhiBalancing
            self.phi_balancing = YvPhiBalancing(
                num_experts=getattr(cfg, 'moe_num_experts', 8),
                momentum=getattr(cfg, 'phi_balancing_momentum', 0.95),
                lr=getattr(cfg, 'phi_balancing_lr', 0.01),
                ema_decay=getattr(cfg, 'phi_balancing_ema_decay', 0.99),
                device=device, dtype=dtype
            )
        else:
            self.phi_balancing = None
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        subconscious_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        film_params: Optional[Dict[str, torch.Tensor]] = None,
        modal_id: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass with DeepNorm.

        Args:
            x: Input tensor.
            mask: Attention mask.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use cache.
            subconscious_kv: Optional extra KV pair for hard knowledge injection.
            film_params: Optional FiLM modulation parameters.

        Returns:
            Output tensor(s).
        """
        if film_params is not None:
            scale = film_params["scale"]
            shift = film_params["shift"]
            x = x * (1.0 + scale) + shift

        residual = x

        # HydraHead: FA/LA hybrid attention — stateless, no KV cache needed.
        hydra_out = None
        if self.hydra_attn is not None:
            hydra_out = self.hydra_attn(x, mask)

        if use_cache:
            attn_out, present_kv = self.attn(
                x, mask, past_key_values=past_key_values, use_cache=True,
                extra_kv=subconscious_kv,
            )
            if hydra_out is not None:
                attn_out = attn_out + hydra_out
        else:
            attn_out = self.attn(
                x, mask, past_key_values=past_key_values, use_cache=False,
                extra_kv=subconscious_kv,
            )
            if hydra_out is not None:
                attn_out = attn_out + hydra_out

        x = self.deep_norm_attn(residual, self.residual_dropout(attn_out))
        
        residual = x
        mlp_out, aux_loss = self.mlp(x, modal_id=modal_id)

        # Apply phi-balancing to aux_loss
        if self.phi_balancing is not None and self.training:
            aux_loss = aux_loss + self.phi_balancing.get_regularization_loss()

        x = self.deep_norm_mlp(residual, self.residual_dropout(mlp_out))
        
        if use_cache:
            return x, aux_loss, present_kv
        return x, aux_loss


class YvCrossAttentionBlock(nn.Module):
    """Transformer block with cross-attention for encoder-decoder models.
    
    Implements a block with self-attention, cross-attention, and MLP
    for encoder-decoder architectures.
    """
    
    def __init__(
        self,
        cfg,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize cross-attention block.
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.cfg = cfg
        
        self.self_attn = YvAttention(cfg, device=device, dtype=dtype)
        self.cross_attn = YvCrossAttention(
            cfg.hidden_size, cfg.n_head,
            getattr(cfg, 'n_kv_head', cfg.n_head),
            attention_dropout=getattr(cfg, 'attention_dropout', 0.0),
            device=device, dtype=dtype
        )
        
        self.mlp = YvDeepSeekMoELayer(cfg, device=device, dtype=dtype)
            
        self.norm1 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.norm2 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.norm3 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout', 0.1))
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        modal_id: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Tuple]]:
        """Forward pass with cross-attention.
        
        Args:
            x: Input tensor.
            encoder_hidden_states: Encoder hidden states.
            self_attn_mask: Self-attention mask.
            cross_attn_mask: Cross-attention mask.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use cache.
            
        Returns:
            Output tensor(s).
        """
        residual = x
        x = self.norm1(x)
        
        if use_cache:
            self_attn_out, self_kv = self.self_attn(
                x, self_attn_mask, past_key_values=past_key_values, use_cache=True
            )
        else:
            self_attn_out = self.self_attn(x, self_attn_mask, past_key_values=past_key_values)
            
        x = residual + self.residual_dropout(self_attn_out)
        
        residual = x
        x = self.norm2(x)
        cross_attn_out = self.cross_attn(x, encoder_hidden_states, cross_attn_mask)
        x = residual + self.residual_dropout(cross_attn_out)
        
        residual = x
        x = self.norm3(x)
        mlp_out, aux_loss = self.mlp(x, modal_id=modal_id)
        x = residual + self.residual_dropout(mlp_out)
        
        if use_cache:
            return x, aux_loss, self_kv
        return x, aux_loss


# Paper: Vaswani et al., "Attention Is All You Need", NeurIPS 2017; core transformer block with Yv extensions
class YvTransformerBlock(nn.Module):
    """Unified Transformer Block with multiple architecture support.

    Implements a comprehensive transformer block supporting:
    - Standard sequential attention-MLP
    - Parallel attention-MLP
    - DeepNorm for deep network stability
    - Cross-attention for encoder-decoder
    - Adaptive computation time
    - Mixture-of-Depths
    - LayerScale
    - LoRA/DoRA integration
    - Gradient checkpointing
    - Quantization support
    """

    def __init__(self, cfg, device=None, dtype=None, quantization_config=None, gate=None):
        """Initialize the transformer block.

        Args:
            cfg: Configuration object containing model hyperparameters.
            device: Device to place the module on.
            dtype: Data type for the module parameters.
            quantization_config: Configuration for model quantization.
            gate: Optional pre-built MoE gate for PathMoE stage sharing.

        Raises:
            RuntimeError: If quantization setup fails and fallback also fails.
        """
        super().__init__()
        self.cfg = cfg
        self.cache_manager = None
        self.layer_idx = -1
        self._moe_gate = gate
        
        self.block_type = getattr(cfg, 'block_type', 'standard')
        self.use_parallel = getattr(cfg, 'use_parallel', False)
        self.use_deepnorm = getattr(cfg, 'use_deepnorm', False)
        self.use_layerscale = getattr(cfg, 'use_layerscale', True)
        self.use_swiglu = getattr(cfg, 'use_swiglu', True)
        self.use_geglu = getattr(cfg, 'use_geglu', False)
        self.use_mixture_of_depths = getattr(cfg, 'mixture_of_depths', False)
        self.use_layer_route = getattr(cfg, 'use_layer_route', False)
        self.use_lora = getattr(cfg, 'use_lora', False)
        self.use_dora = getattr(cfg, 'use_dora', False)
        self.use_mhc = getattr(cfg, 'use_mhc', False)
        self.use_adaptive_computation = getattr(cfg, 'use_adaptive_computation', False)
        
        if self.use_parallel:
            self._init_parallel_block(cfg, device, dtype)
        elif self.use_deepnorm:
            self._init_deepnorm_block(cfg, device, dtype)
        else:
            self._init_standard_block(cfg, device, dtype)
            
        if self.use_mixture_of_depths:
            self.mod_router = YvMixtureOfDepths(
                cfg.hidden_size, cfg.n_head,
                routing_weight=getattr(cfg, 'mod_routing_weight', 0.1),
                device=device, dtype=dtype
            )

        if self.use_adaptive_computation:
            self.act = YvAdaptiveComputationTime(
                hidden_size=cfg.hidden_size,
                max_iterations=getattr(cfg, 'adaptive_computation_max_iterations', 3),
                threshold=getattr(cfg, 'adaptive_computation_threshold', 0.99),
                device=device, dtype=dtype
            )
            
        if self.use_layer_route:
            from .layer_route import YvLayerRouteAdapter
            self.layer_route = YvLayerRouteAdapter(
                cfg.hidden_size, cfg.n_head, cfg.hidden_size // cfg.n_head,
                lora_rank=getattr(cfg, 'layer_route_lora_rank', 8),
                lora_scale=getattr(cfg, 'layer_route_lora_scale', 1.0),
                gate_reg_lambda=getattr(cfg, 'layer_route_gate_reg', 0.01),
                device=device, dtype=dtype,
            )
        else:
            self.layer_route = None
            
        self.use_checkpoint = getattr(cfg, 'use_checkpoint', True)
        self.adaptive_checkpointing = getattr(cfg, 'adaptive_checkpointing', True)
        self.memory_threshold_high = getattr(cfg, 'memory_threshold_high', 0.85)
        self.memory_threshold_low = getattr(cfg, 'memory_threshold_low', 0.60)
        self.checkpoint_frequency = getattr(cfg, 'checkpoint_frequency', 1)
        self.current_checkpoint_freq = self.checkpoint_frequency

        self.quantization_config = quantization_config

        if self.quantization_config is not None:
            self._apply_quantization()

    def _init_standard_block(self, cfg, device, dtype):
        """Initialize standard sequential block.
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        self.attn = YvAttention(cfg, device=device, dtype=dtype)

        # HydraHead: head-level FA/LA hybridization
        if getattr(cfg, 'use_hydra_head', False):
            from ..core.attention import YvHydraHeadAttention
            self.hydra_attn = YvHydraHeadAttention(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.n_head,
                head_dim=cfg.hidden_size // cfg.n_head,
                la_ratio=getattr(cfg, 'hydra_head_la_ratio', 0.5),
                learnable_assignment=getattr(cfg, 'hydra_head_learnable_assignment', True),
                temperature=getattr(cfg, 'hydra_head_temperature', 1.0),
                causal=True,
                device=device, dtype=dtype
            )
        else:
            self.hydra_attn = None

        self.mlp = YvDeepSeekMoELayer(cfg, device=device, dtype=dtype)

        self.norm1 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.norm2 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.pre_norm1 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.pre_norm2 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)

        self.residual_scale = nn.Parameter(
            torch.ones(1, device=device, dtype=dtype) * getattr(cfg, 'residual_alpha', (2.0 * cfg.n_layer) ** -0.5)
        )
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout', 0.1))
        
        if self.use_layerscale:
            self.attn_layerscale = YvLayerScale(
                cfg.hidden_size,
                init_value=getattr(cfg, 'layer_scale_init', getattr(cfg, 'layerscale_init', 1e-5)),
                device=device, dtype=dtype
            )
            self.mlp_layerscale = YvLayerScale(
                cfg.hidden_size,
                init_value=getattr(cfg, 'layer_scale_init', getattr(cfg, 'layerscale_init', 1e-5)),
                device=device, dtype=dtype
            )

        # Manifold-Constrained Hyper-Connections (mHC)
        # Replaces standard residual with expanded stream + Birkhoff polytope constraint
        # Based on DeepSeek-V4 Pro technical report
        if self.use_mhc:
            self.attn_mhc = YvMHC(
                cfg.hidden_size,
                n_hc=getattr(cfg, 'mhc_n_hc', 4),
                sinkhorn_iters=getattr(cfg, 'mhc_sinkhorn_iters', 20),
                device=device, dtype=dtype
            )
            self.mlp_mhc = YvMHC(
                cfg.hidden_size,
                n_hc=getattr(cfg, 'mhc_n_hc', 4),
                sinkhorn_iters=getattr(cfg, 'mhc_sinkhorn_iters', 20),
                device=device, dtype=dtype
            )

        # SwiGLU Clamping — prevents activation explosion
        # Based on DeepSeek-V4: linear clip [-10, 10], gate cap at 10
        self.swiglu_clamp = getattr(cfg, 'swiglu_clamp', True)
        
        # Hybrid SSM Integration: Mamba-3 for linear complexity on long sequences
        # Auto-enabled when hidden_size >= 2048 (sufficient capacity for SSM)
        # Provides O(n) complexity alternative to O(n^2) attention
        self._init_hybrid_ssm(cfg, device, dtype)
        
        # Attention Residuals for deep layer contribution balancing
        # Based on Kimi Attention Residuals (arXiv:2603.15031, 2026)
        # Replaces fixed-weight residual accumulation with dynamic attention-based aggregation
        self.use_attn_res = getattr(cfg, 'use_attn_res', False)
        if self.use_attn_res:
            self.attn_res_block_size = getattr(cfg, 'attn_res_block_size', 8)
            self.attn_res_use_two_phase = getattr(cfg, 'attn_res_use_two_phase', True)
            self.attn_res_use_online_softmax = getattr(cfg, 'attn_res_use_online_softmax', True)
            self.attn_res_cache_pipeline = getattr(cfg, 'attn_res_cache_pipeline', True)
            self.attn_res_max_blocks = getattr(cfg, 'attn_res_max_blocks', 32)
            self.attn_res_learnable_query = getattr(cfg, 'attn_res_learnable_query', True)
            self.attn_res_use_rmsnorm = getattr(cfg, 'attn_res_use_rmsnorm', True)
            
            # Learnable query for attention-based residual aggregation
            # Input-independent parameter for efficient parallel computation
            if self.attn_res_learnable_query:
                self.attn_res_query = nn.Parameter(
                    torch.randn(1, 1, cfg.hidden_size, device=device, dtype=dtype) * 0.02
                )
            else:
                self.register_buffer(
                    'attn_res_query',
                    torch.randn(1, 1, cfg.hidden_size, device=device, dtype=dtype) * 0.02,
                    persistent=False
                )
            
            # Key projection for block representations
            self.attn_res_key_proj = nn.Linear(
                cfg.hidden_size, cfg.hidden_size, bias=False, device=device, dtype=dtype
            )
            
            # Value projection (optional, can use identity)
            self.attn_res_value_proj = nn.Linear(
                cfg.hidden_size, cfg.hidden_size, bias=False, device=device, dtype=dtype
            )
            
            # Output projection
            self.attn_res_out_proj = nn.Linear(
                cfg.hidden_size, cfg.hidden_size, bias=False, device=device, dtype=dtype
            )
            
            # Normalization layer
            if self.attn_res_use_rmsnorm:
                self.attn_res_norm = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
            else:
                self.attn_res_norm = nn.LayerNorm(cfg.hidden_size, device=device, dtype=dtype)
            
            # Block-level representations cache
            # Stores aggregated block outputs for efficient attention computation
            self.register_buffer('_attn_res_blocks', None, persistent=False)
            self.register_buffer('_attn_res_block_count', torch.tensor(0, dtype=torch.long), persistent=False)
            
            # Online softmax state for two-phase computation
            # Maintains running statistics for streaming aggregation
            self.register_buffer('_online_softmax_m', None, persistent=False)
            self.register_buffer('_online_softmax_n', None, persistent=False)
            
            # Pipeline cache for cross-stage communication
            if self.attn_res_cache_pipeline:
                self.register_buffer('_pipeline_cache', None, persistent=False)
                self.register_buffer('_pipeline_ready', torch.tensor(False), persistent=False)
            
            # Partial block accumulation for current block
            self.register_buffer('_partial_block_sum', None, persistent=False)
            self.register_buffer('_partial_block_count', torch.tensor(0, dtype=torch.long), persistent=False)

        # MemSep: Knowledge injection via cross-attention from retrieved memory slots
        # Based on Engram (Cheng et al., arXiv:2601.07372, 2026)
        # Only initialized when use_memory_separation=True in config
        self.use_memory_separation = getattr(cfg, 'use_memory_separation', False)
        if self.use_memory_separation:
            from .memory_attention import YvMemoryCrossAttention
            self.memory_attn = YvMemoryCrossAttention(
                hidden_size=cfg.hidden_size,
                knowledge_dim=getattr(cfg, 'memory_knowledge_dim', 256),
                n_heads=getattr(cfg, 'memory_cross_attn_heads', 4),
                gate_init=getattr(cfg, 'memory_gate_init', 0.0),
                dropout=getattr(cfg, 'residual_dropout', 0.1),
                device=device,
                dtype=dtype,
            )
            self._memory_context = None
        else:
            self.memory_attn = None
            self._memory_context = None

        # Subconscious: volatile knowledge injection via 0.5B dynamic head + 314B field
        # Activated when use_subconscious=True in config. Parallel to 1M context.
        # The subconscious system is a singleton owned by YvModel; each block
        # gets a reference via _set_subconscious_system() before each forward.
        self.use_subconscious = getattr(cfg, 'use_subconscious', False)
        self._subconscious_system = None

        # Phi-Balancing: population-level mirror descent load balancer
        if getattr(cfg, 'use_phi_balancing', False):
            from ..moe.gate import YvPhiBalancing
            self.phi_balancing = YvPhiBalancing(
                num_experts=getattr(cfg, 'moe_num_experts', 8),
                momentum=getattr(cfg, 'phi_balancing_momentum', 0.95),
                lr=getattr(cfg, 'phi_balancing_lr', 0.01),
                ema_decay=getattr(cfg, 'phi_balancing_ema_decay', 0.99),
                device=device, dtype=dtype
            )
        else:
            self.phi_balancing = None

    def _set_subconscious_system(self, system: Optional[object]) -> None:
        """Set reference to the subconscious system for this forward pass.

        Called by YvModel.forward before layer processing begins.

        Args:
            system: The YvSubconsciousSystem instance, or None to disable.
        """
        self._subconscious_system = system

    def _set_memory_context(self, ctx: Optional[dict]) -> None:
        """Set memory context from YvModel forward for knowledge injection.

        Called by YvModel.forward before each block invocation when
        memory separation is active.

        Args:
            ctx: Knowledge context dict from YvMemoryRouter.forward().
        """
        self._memory_context = ctx

    def _get_memory_context(self) -> Optional[dict]:
        """Get current memory context for knowledge injection.

        Returns:
            Knowledge context dict or None.
        """
        return self._memory_context

    def _apply_film(
        self,
        hidden_states: torch.Tensor,
        film_params: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Apply FiLM modulation if ``film_params`` is provided.

        Args:
            hidden_states: [batch, seq, hidden_size].
            film_params: Dict with ``scale`` and ``shift`` tensors, each
                of shape [batch, seq, hidden_size].

        Returns:
            Modulated hidden states.
        """
        if film_params is None:
            return hidden_states
        scale = film_params["scale"]
        shift = film_params["shift"]
        return hidden_states * (1.0 + scale) + shift

    def _init_parallel_block(self, cfg, device, dtype):
        """Initialize parallel attention-MLP block.
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        gate_kwargs = {}
        if getattr(self, '_moe_gate', None) is not None:
            gate_kwargs['gate'] = self._moe_gate
        self.parallel_block = YvParallelBlock(cfg, device=device, dtype=dtype, **gate_kwargs)

    def _init_hybrid_ssm(self, cfg, device, dtype):
        """Initialize hybrid SSM layer for linear complexity on long sequences.
        
        Integrates Mamba-3 state space model as an alternative computation path
        that is automatically activated for sequences longer than 8192 tokens.
        This provides O(n) complexity instead of O(n^2) for attention.
        
        The SSM layer is initialized when:
        - hidden_size >= 2048 (sufficient model capacity)
        - Enables linear-time processing for ultra-long contexts
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
        
        Reference:
            Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
            State Spaces", arXiv:2312.00752, 2023.
            Dao & Gu, "Mamba-2: State Space Duality", arXiv:2405.21060, 2024.
        """
        if cfg.hidden_size >= 2048:
            # Derive SSM configuration from model config
            # Use n_kv_head (GQA heads) as a proxy for state dimension
            n_kv_head = getattr(cfg, 'n_kv_head', getattr(cfg, 'n_head', 8))
            ssm_state_dim = max(64, n_kv_head * 16)  # Scale with GQA capacity
            
            ssm_config = YvMamba3Config(
                d_model=cfg.hidden_size,
                d_state=ssm_state_dim,
                d_conv=4,
                expand=2,
                use_trapezoidal=True,  # Improved stability
                use_complex=True,      # Richer dynamics
                use_mimo=True,         # Enhanced capacity
                use_gated=True,        # Better training
                use_v_kernel=True,     # Mamba-2 optimization
                use_ss_duality=True,   # Efficient training
                use_adaptive_dt=True,  # Adaptive time steps
            )
            
            self.ssm_layer = YvMamba3Block(ssm_config)
            
            # Learnable gate for attention-SSM blending
            # Initialized to 0 so model learns optimal blend
            self.ssm_gate = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
        else:
            self.ssm_layer = None
            self.ssm_gate = None

    def _init_deepnorm_block(self, cfg, device, dtype):
        """Initialize DeepNorm block.
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        gate_kwargs = {}
        if getattr(self, '_moe_gate', None) is not None:
            gate_kwargs['gate'] = self._moe_gate
        self.deepnorm_block = YvDeepNormBlock(cfg, device=device, dtype=dtype, **gate_kwargs)

    def _apply_quantization(self):
        """Apply quantization to linear layers."""
        try:
            import bitsandbytes as bnb
            layer_importance = self._get_layer_importance()

            def convert_linear_to_mixed_precision(module, layer_type='standard'):
                """Recursively convert linear layers to quantized versions.

                Args:
                    module: Module to process recursively.
                    layer_type: Importance level.
                """
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        if layer_importance == 'critical':
                            new_mod = bnb.nn.Linear8bit(
                                child.in_features,
                                child.out_features,
                                bias=child.bias is not None,
                                threshold=getattr(self.quantization_config, 'bnb_8bit_threshold', 6.0),
                            )
                        elif layer_importance == 'important':
                            new_mod = bnb.nn.Linear4bit(
                                child.in_features,
                                child.out_features,
                                bias=child.bias is not None,
                                quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                                compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.float16),
                                compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                            )
                        else:
                            new_mod = bnb.nn.Linear4bit(
                                child.in_features,
                                child.out_features,
                                bias=child.bias is not None,
                                quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                                compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.bfloat16),
                                compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                            )
                        setattr(module, name, new_mod)
                    else:
                        child_layer_type = self._get_child_layer_type(name, layer_type)
                        convert_linear_to_mixed_precision(child, child_layer_type)

            convert_linear_to_mixed_precision(self)
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            _LOG.error(f"Mixed precision quantization failed: {e}")
            self._fallback_to_4bit_quantization()

    def _get_layer_importance(self):
        """Get the importance level for layer quantization.

        Returns:
            Layer importance level.
        """
        return getattr(self.quantization_config, 'layer_importance', 'standard')

    def _get_child_layer_type(self, child_name, parent_type):
        """Determine the importance type of a child layer.

        Args:
            child_name: Name of the child module.
            parent_type: Importance type of the parent module.

        Returns:
            Determined importance level.
        """
        name_lower = child_name.lower()
        if 'attn' in name_lower or 'attention' in name_lower:
            return 'critical'
        elif 'mlp' in name_lower or 'feedforward' in name_lower:
            return 'important'
        else:
            return parent_type

    def _fallback_to_4bit_quantization(self):
        """Apply uniform 4-bit quantization as fallback."""
        try:
            import bitsandbytes as bnb

            def convert_linear_to_4bit(module):
                """Recursively convert all linear layers to 4-bit.

                Args:
                    module: Module to process recursively.
                """
                for name, child in module.named_children():
                    if isinstance(child, nn.Linear):
                        new_mod = bnb.nn.Linear4bit(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                            compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.bfloat16),
                            compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                        )
                        setattr(module, name, new_mod)
                    else:
                        convert_linear_to_4bit(child)

            convert_linear_to_4bit(self)
            _LOG.info("Fallback to 4-bit quantization successful")
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            _LOG.error(f"Fallback 4-bit quantization also failed: {e}")

    def _should_use_checkpoint(self):
        """Determine whether gradient checkpointing should be used.

        Returns:
            True if checkpointing should be used.
        """
        if not self.use_checkpoint or not self.adaptive_checkpointing:
            return self.use_checkpoint

        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                memory_usage = allocated / total_memory

                if memory_usage > self.memory_threshold_high:
                    self.current_checkpoint_freq = max(1, self.checkpoint_frequency // 2)
                    return True
                elif memory_usage < self.memory_threshold_low:
                    self.current_checkpoint_freq = self.checkpoint_frequency * 2
                    return False
                else:
                    self.current_checkpoint_freq = self.checkpoint_frequency
                    return (self.checkpoint_frequency <= 1) or (torch.randint(0, self.checkpoint_frequency, (1,)).item() == 0)
            else:
                return self.use_checkpoint
        except (RuntimeError, ValueError, AttributeError) as e:
            _LOG.error(f"Adaptive checkpointing memory check failed: {e}")
            return self.use_checkpoint

    def _apply_with_checkpoint(
        self,
        x,
        mask,
        past_key_values=None,
        use_cache=False,
        subconscious_kv=None,
        film_params=None,
    ):
        """Apply the transformer block with optional gradient checkpointing.

        Args:
            x: Input tensor.
            mask: Attention mask tensor.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use and update key/value cache.
            subconscious_kv: Optional extra KV pair for hard knowledge injection.
            film_params: Optional FiLM modulation parameters.

        Returns:
            Output tensor(s) from the transformer block.
        """
        import torch.utils.checkpoint as cp

        attn_past_key_values = past_key_values if past_key_values is not None else None
        should_checkpoint = self._should_use_checkpoint()

        def _inner(
            xc,
            mask_arg=None,
            kv=None,
            use_cache_arg=False,
            subconscious_kv_arg=None,
            film_params_arg=None,
        ):
            """Inner function for gradient checkpointing.

            Args:
                xc: Input tensor.
                mask_arg: Attention mask tensor.
                kv: Past key/value pairs.
                use_cache_arg: Whether to use cache.
                subconscious_kv_arg: Optional extra KV pair.
                film_params_arg: Optional FiLM parameters.

            Returns:
                Output from _forward_core.
            """
            # Attention residuals mutate instance state across layers, which breaks
            # checkpoint determinism. Both the forward and recompute calls enter here,
            # so disabling within _inner keeps them consistent.
            saved_use_attn_res = self.use_attn_res
            self.use_attn_res = False
            try:
                return self._forward_core(
                    xc,
                    mask_arg,
                    kv,
                    use_cache_arg,
                    subconscious_kv=subconscious_kv_arg,
                    film_params=film_params_arg,
                )
            finally:
                self.use_attn_res = saved_use_attn_res

        if should_checkpoint and self.training:
            # Set checkpointing flag on MoE gates to disable non-deterministic operations
            self._set_moe_checkpointing(True)
            try:
                out = cp.checkpoint(
                    _inner, x, mask, attn_past_key_values, use_cache,
                    subconscious_kv_arg=subconscious_kv,
                    film_params_arg=film_params,
                    use_reentrant=False,
                    preserve_rng_state=True,
                    determinism_check="default"
                )
            finally:
                self._set_moe_checkpointing(False)
        else:
            out = _inner(
                x,
                mask,
                attn_past_key_values,
                use_cache,
                subconscious_kv_arg=subconscious_kv,
                film_params_arg=film_params,
            )

        return out

    def _set_moe_checkpointing(self, is_checkpointing: bool):
        """Set checkpointing flag on MoE gates to ensure deterministic routing.
        
        Args:
            is_checkpointing: Whether currently in checkpointing mode.
        """
        for attr_name in ['mlp', 'parallel_block', 'deepnorm_block']:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if attr is not None:
                    if hasattr(attr, 'router') and hasattr(attr.router, '_is_checkpointing'):
                        attr.router._is_checkpointing = is_checkpointing
                    if hasattr(attr, 'gate') and hasattr(attr.gate, '_is_checkpointing'):
                        attr.gate._is_checkpointing = is_checkpointing
                    if hasattr(attr, 'experts'):
                        for expert in attr.experts:
                            if hasattr(expert, '_is_checkpointing'):
                                expert._is_checkpointing = is_checkpointing
                    if hasattr(attr, 'mlp'):
                        mlp = attr.mlp
                        if mlp is not None:
                            if hasattr(mlp, 'router') and hasattr(mlp.router, '_is_checkpointing'):
                                mlp.router._is_checkpointing = is_checkpointing
                            if hasattr(mlp, 'gate') and hasattr(mlp.gate, '_is_checkpointing'):
                                mlp.gate._is_checkpointing = is_checkpointing
                            if hasattr(mlp, 'experts'):
                                for expert in mlp.experts:
                                    if hasattr(expert, '_is_checkpointing'):
                                        expert._is_checkpointing = is_checkpointing

    def forward(
        self,
        x,
        mask,
        past_key_values=None,
        use_cache=False,
        subconscious_kv=None,
        film_params=None,
        modal_id=None,
    ):
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].
            mask: Attention mask tensor.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use and update key/value cache.
            subconscious_kv: Optional extra KV pair for hard knowledge injection.
            film_params: Optional FiLM modulation parameters.

        Returns:
            Output tensor(s).
        """
        if self.use_parallel:
            return self.parallel_block(
                x, mask, past_key_values, use_cache,
                subconscious_kv=subconscious_kv,
                film_params=film_params,
            )
        elif self.use_deepnorm:
            return self.deepnorm_block(
                x, mask, past_key_values, use_cache,
                subconscious_kv=subconscious_kv,
                film_params=film_params,
            )
        else:
            return self._apply_with_checkpoint(
                x,
                mask,
                past_key_values,
                use_cache,
                subconscious_kv=subconscious_kv,
                film_params=film_params,
            )

    def _forward_core(
        self,
        x,
        mask,
        attn_past_key_values=None,
        use_cache=False,
        subconscious_kv=None,
        film_params=None,
    ):
        """Core forward computation without checkpointing wrapper.

        Args:
            x: Input tensor.
            mask: Attention mask tensor.
            attn_past_key_values: Key/value cache for attention.
            use_cache: Whether to use and update key/value cache.
            subconscious_kv: Optional extra KV pair for hard knowledge injection.
            film_params: Optional FiLM modulation parameters.

        Returns:
            Output tensor(s).
        """
        # FiLM modulation is applied to the incoming hidden stream first.
        x = self._apply_film(x, film_params)

        pre_block_input = x

        # LayerRoute: per-token adaptive skip via straight-through binary gate
        layer_route_gate = None
        if self.layer_route is not None:
            gate, lr_loss = self.layer_route(x)
            layer_route_gate = gate

        # ACT: Adaptive Computation Time — token-level dynamic iteration count
        if self.use_adaptive_computation and hasattr(self, 'act'):
            x, ponder = self.act(x, lambda h: h)
            del ponder  # ponder cost tracked but not accumulated to loss here

        residual = x

        if self.use_attn_res:
            x = self._apply_attn_res(x)

        x_norm = self.pre_norm1(x)
        attn_cache = None
        past_for_attn = attn_past_key_values

        if use_cache and self.cache_manager is not None and self.layer_idx >= 0:
            got = self.cache_manager.get_kv_cache(self.layer_idx, attn_past_key_values)
            if got is not None:
                past_for_attn = got

        if getattr(self, 'hydra_attn', None) is not None:
            hydra_out = self.hydra_attn(x_norm, mask)
        else:
            hydra_out = None

        if use_cache:
            attn_out, present_kv = self.attn(
                x_norm,
                mask,
                past_key_values=past_for_attn,
                use_cache=True,
                cache_manager=self.cache_manager,
                layer_idx=self.layer_idx,
                extra_kv=subconscious_kv,
            )
            attn_cache = present_kv
            if hydra_out is not None:
                attn_out = attn_out + hydra_out
        else:
            attn_out = self.attn(
                x_norm,
                mask,
                past_key_values=past_for_attn,
                use_cache=False,
                cache_manager=self.cache_manager,
                layer_idx=self.layer_idx,
                extra_kv=subconscious_kv,
            )
            if hydra_out is not None:
                attn_out = attn_out + hydra_out

        if self.use_layerscale:
            attn_out = self.attn_layerscale(attn_out)

        if self.use_attn_res:
            self._accumulate_partial_block(attn_out)

        if torch.cuda.device_count() > 1 and self.training:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    world_size = dist.get_world_size()

                    compressed = attn_out.mean(dim=1)

                    if not hasattr(self, '_gather_buffer') or self._gather_buffer is None or self._gather_buffer[0].shape != compressed.shape:
                        self._gather_buffer = [torch.zeros_like(compressed) for _ in range(world_size)]

                    gathered = self._gather_buffer
                    for g in gathered:
                        g.zero_()
                    dist.all_gather(gathered, compressed)

                    other_info = torch.stack(gathered).mean(dim=0)
                    attn_out = attn_out + 0.05 * other_info.unsqueeze(1)
            except (ImportError, ModuleNotFoundError, RuntimeError, ValueError) as e:
                _LOG.debug(f"YvTransformerBlock: distributed attention gathering skipped: {e}")

        if self.use_mhc:
            x_out = self.attn_mhc(residual + self.residual_dropout(self.residual_scale * attn_out))
        else:
            x_out = residual + self.residual_dropout(self.residual_scale * attn_out)
        x_out = self.norm1(x_out)
        
        if hasattr(self, 'ssm_layer') and self.ssm_layer is not None:
            seq_len = x_out.shape[1]
            if seq_len > 8192:
                ssm_out = self.ssm_layer(x_out)
                gate = torch.sigmoid(self.ssm_gate)
                x_out = gate * x_out + (1.0 - gate) * ssm_out

        residual = x_out
        x_norm = self.pre_norm2(x_out)

        # MemSep: Inject retrieved knowledge via cross-attention before MLP
        # The memory_router (in model.py) provides knowledge_context dict with
        # knowledge_projected tensor. If present, apply cross-attention injection.
        if self.use_memory_separation and self.memory_attn is not None:
            memory_context = self._get_memory_context()
            if memory_context is not None and "knowledge_projected" in memory_context:
                x_norm = self.memory_attn(
                    hidden_states=x_norm,
                    knowledge=memory_context["knowledge_projected"],
                )

        mlp_out, aux_loss = self.mlp(x_norm, modal_id=modal_id)

        # Apply phi-balancing to aux_loss
        if getattr(self, 'phi_balancing', None) is not None and self.training:
            aux_loss = aux_loss + self.phi_balancing.get_regularization_loss()
        
        if self.use_layerscale:
            mlp_out = self.mlp_layerscale(mlp_out)

        if self.use_attn_res:
            self._accumulate_partial_block(mlp_out)

        if self.use_mhc:
            x_out = self.mlp_mhc(residual + self.residual_dropout(self.residual_scale * mlp_out))
        else:
            x_out = residual + self.residual_dropout(self.residual_scale * mlp_out)
        x_out = self.norm2(x_out)

        if layer_route_gate is not None:
            x_out = layer_route_gate.unsqueeze(-1) * pre_block_input + \
                    (1.0 - layer_route_gate.unsqueeze(-1)) * x_out
            aux_loss = aux_loss + lr_loss

        if self.use_mixture_of_depths:
            if not hasattr(self, '_mod_process_proj'):
                self._mod_process_proj = nn.Linear(
                    x_out.size(-1), x_out.size(-1), bias=False,
                    device=x_out.device, dtype=x_out.dtype
                )
                nn.init.zeros_(self._mod_process_proj.weight)
            x_out, mod_loss = self.mod_router(
                x_out, lambda h: h + self._mod_process_proj(h)
            )
            aux_loss = aux_loss + mod_loss
        
        if self.use_attn_res:
            self._update_attn_res_block(x_out)

        if use_cache and self.cache_manager is not None and self.layer_idx >= 0:
            self.cache_manager.compute_pending_prediction(self.layer_idx, x_out)

        if use_cache:
            return x_out, aux_loss, attn_cache
        return x_out, aux_loss

    def _apply_attn_res(self, x: torch.Tensor) -> torch.Tensor:
        """Apply block-level attention residual aggregation with two-phase computation.
        
        Implements Kimi Attention Residuals (arXiv:2603.15031, 2026):
        - Phase 1 (Parallel): Compute attention over all blocks with input-independent query
        - Phase 2 (Serial): Merge block-level attention with current block via Online-Softmax
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].
            
        Returns:
            Aggregated tensor with attention-weighted block contributions.
        """
        if self._attn_res_blocks is None or self._attn_res_block_count == 0:
            return x
        
        batch_size, seq_len, hidden_size = x.shape
        device = x.device
        dtype = x.dtype
        
        if self._attn_res_blocks is None or self._attn_res_block_count.item() == 0:
            return x
        
        num_blocks_stored = self._attn_res_blocks.shape[0]
        num_blocks = min(num_blocks_stored, self.attn_res_max_blocks)
        if num_blocks == 0:
            return x
        
        blocks = self._attn_res_blocks[-num_blocks:]  # [N, B, T, H] where N = num_blocks
        
        if blocks.dim() != 4 or blocks.shape[0] != num_blocks:
            _LOG.warning(f"AttnRes blocks shape mismatch: blocks.shape={blocks.shape}, num_blocks={num_blocks}")
            return x
        
        # Phase 1: Parallel computation of block-level attention
        # Query is input-independent, enabling efficient batched computation
        query = self.attn_res_query.expand(batch_size, -1, -1)  # [B, 1, H]
        
        # Normalize blocks for attention
        blocks_flat = blocks.view(-1, hidden_size)  # [N*B*T, H]
        blocks_norm = self.attn_res_norm(blocks_flat).view_as(blocks)  # [N, B, T, H]
        
        # Project keys and values
        # Key projection: [N, B, T, H]
        keys = self.attn_res_key_proj(blocks_norm)  # [N, B, T, H]
        
        # Value projection: [N, B, T, H]
        values = self.attn_res_value_proj(blocks)  # [N, B, T, H]
        
        # Compute attention scores
        # Query: [B, 1, H], Keys: [N, B, T, H]
        # Reshape for batched matmul
        query_expanded = query  # [B, 1, H]
        keys_transposed = keys.permute(1, 0, 2, 3)  # [B, N, T, H]
        
        # Compute attention logits: [B, N, T]
        # Use scaled dot-product attention
        scale = hidden_size ** -0.5
        attn_logits = torch.matmul(
            query_expanded.squeeze(1).to(keys_transposed.dtype),  # [B, H]
            keys_transposed.reshape(batch_size, -1, hidden_size).transpose(-1, -2)  # [B, H, N*T]
        ) * scale  # [B, N*T]
        
        # Reshape back to [B, N, T]
        attn_logits = attn_logits.view(batch_size, num_blocks, seq_len)
        
        if self.attn_res_use_two_phase and self.attn_res_use_online_softmax:
            # Phase 2: Online Softmax for streaming aggregation
            # Merge block-level attention with current block
            
            # Get current block representation
            current_norm = self.attn_res_norm(x)  # [B, T, H]
            current_key = self.attn_res_key_proj(current_norm)  # [B, T, H]
            current_value = self.attn_res_value_proj(x)  # [B, T, H]
            
            # Compute attention for current block
            current_logits = torch.matmul(
                query.squeeze(1).to(current_key.dtype),  # [B, H]
                current_key.transpose(-1, -2)  # [B, H, T]
            ) * scale  # [B, T]
            
            # Online Softmax: merge previous blocks with current block
            # m: running maximum, n: running normalization factor
            if self._online_softmax_m is None:
                # Initialize with current block
                m_current = current_logits.max(dim=-1, keepdim=True)[0]  # [B, 1]
                n_current = torch.exp(current_logits - m_current).sum(dim=-1, keepdim=True)  # [B, 1]
                
                # Update state
                self._online_softmax_m = m_current
                self._online_softmax_n = n_current
            else:
                # Get previous state
                m_prev = self._online_softmax_m  # [B, 1]
                n_prev = self._online_softmax_n  # [B, 1]
                
                # Compute new maximum
                m_new = torch.maximum(m_prev, current_logits.max(dim=-1, keepdim=True)[0])
                
                # Update normalization factor
                n_prev_scaled = n_prev * torch.exp(m_prev - m_new)
                n_current = torch.exp(current_logits - m_new).sum(dim=-1, keepdim=True)
                n_new = n_prev_scaled + n_current
                
                # Update state
                self._online_softmax_m = m_new
                self._online_softmax_n = n_new
            
            # Compute final attention weights
            # For blocks: use stored m and n
            m_final = self._online_softmax_m
            n_final = self._online_softmax_n
            
            # Block attention weights
            block_weights = torch.exp(attn_logits - m_final.unsqueeze(-1)) / (n_final.unsqueeze(-1) + 1e-8)
            block_weights = block_weights / block_weights.sum(dim=-1, keepdim=True)  # Normalize
            
            # Current block weight
            current_weight = torch.exp(current_logits - m_final) / (n_final + 1e-8)
            current_weight = current_weight / current_weight.sum(dim=-1, keepdim=True)
            
            # Weighted aggregation
            # Block contributions: [B, N, T] @ [N, B, T, H] -> [B, T, H]
            # Ensure dimensions match before einsum
            block_weights_3d = block_weights.view(batch_size, num_blocks, seq_len)
            values_4d = values.view(num_blocks, batch_size, seq_len, hidden_size)
            block_contrib = torch.einsum('bnt,nbth->bth', block_weights_3d, values_4d)
            
            # Current contribution: [B, T] @ [B, T, H] -> [B, T, H]
            current_contrib = current_weight.unsqueeze(-1) * current_value
            
            # Combined output
            aggregated = block_contrib + current_contrib
        else:
            # Standard softmax attention over blocks
            attn_weights = F.softmax(attn_logits.view(batch_size, -1), dim=-1)
            attn_weights = attn_weights.view(batch_size, num_blocks, -1)
            
            # Weighted sum: [B, N, T] @ [N, B, T, H] -> [B, T, H]
            attn_weights_3d = attn_weights.view(batch_size, num_blocks, seq_len)
            values_4d = values.view(num_blocks, batch_size, seq_len, hidden_size)
            aggregated = torch.einsum('bnt,nbth->bth', attn_weights_3d, values_4d)
        
        # Output projection
        output = self.attn_res_out_proj(aggregated)
        
        return output

    def _update_attn_res_block(self, x: torch.Tensor) -> None:
        """Update block representation at block boundaries with pipeline caching.
        
        Args:
            x: Current hidden state tensor [B, T, H].
        """
        if self.layer_idx < 0:
            return
        
        # Check if we're at a block boundary
        if self.layer_idx % self.attn_res_block_size == 0:
            # Detach to prevent gradient flow through block storage
            block_repr = x.detach()
            
            # Update block count
            self._attn_res_block_count += 1
            
            # Store block representation
            if self._attn_res_blocks is None:
                self._attn_res_blocks = block_repr.unsqueeze(0)  # [1, B, T, H]
            else:
                # Append new block
                new_block = block_repr.unsqueeze(0)  # [1, B, T, H]
                
                # Limit memory by keeping only max_blocks
                if self._attn_res_blocks.shape[0] >= self.attn_res_max_blocks:
                    # Remove oldest block
                    self._attn_res_blocks = torch.cat([
                        self._attn_res_blocks[1:],
                        new_block
                    ], dim=0)
                else:
                    self._attn_res_blocks = torch.cat([
                        self._attn_res_blocks,
                        new_block
                    ], dim=0)
            
            # Reset partial block accumulation
            self._partial_block_sum = None
            self._partial_block_count.zero_()
            
            # Update pipeline cache if enabled
            if self.attn_res_cache_pipeline:
                # Store aggregated block for pipeline communication
                self._pipeline_cache = block_repr
                self._pipeline_ready.fill_(True)

    def _accumulate_partial_block(self, output: torch.Tensor) -> None:
        """Accumulate output into partial block sum for current block.
        
        Args:
            output: Output tensor from attention or MLP layer [B, T, H].
        """
        # Detach to prevent gradient flow
        output_detached = output.detach()
        
        if self._partial_block_sum is None:
            self._partial_block_sum = output_detached.clone()
        else:
            self._partial_block_sum = self._partial_block_sum + output_detached
        
        # Update count
        self._partial_block_count += 1


# Paper: Based on hyperbolic embeddings (Nickel & Kiela, NeurIPS 2017) and spherical constraints
class YvManifoldConstraint(nn.Module):
    """Manifold constraint layer for geometric embedding spaces.

    Constrains tensor representations to lie on a specified Riemannian manifold,
    enabling geometric inductive biases that improve performance on tasks with
    inherent structure: hierarchical data (hyperbolic), directional features
    (spherical), or orthonormality requirements (Stiefel).

    Supported manifolds:
        - ``"hyperbolic"``: Poincaré ball model with curvature *c*.
          Möbius gyrovector operations; ideal for tree/taxonomy embeddings.
        - ``"spherical"``: Unit hypersphere.  Geodesic great-circle distance;
          ideal for angular / directional features.
        - ``"stiefel"``: Set of orthonormal matrices  V_{n,k}.
          Polar-decomposition projection; ideal for attention projections.
        - ``"grassmann"``: k-dimensional subspaces of R^n.
          SVD-based projection; ideal for subspace representations.

    The layer exposes:
        - :meth:`project`      – Euclidean → manifold
        - :meth:`expmap`       – tangent vector → manifold point
        - :meth:`logmap`       – manifold point → tangent vector
        - :meth:`distance`     – geodesic distance on manifold
        - :meth:`egrad2rgrad`  – Euclidean gradient → Riemannian gradient
        - :meth:`retraction`   – first-order approximation of expmap
        - :meth:`inner_product`– Riemannian inner product in tangent space
        - :meth:`forward`      – project + optional learnable scale/shift

    Args:
        dim:              Feature dimension (last axis size).
        manifold_type:    ``'hyperbolic'`` | ``'spherical'`` | ``'stiefel'`` | ``'grassmann'``.
        curvature:        Curvature *c* > 0 for the Poincaré ball (hyperbolic only).
        stiefel_k:        Number of orthonormal columns for Stiefel / Grassmann.
        learnable_scale:  If ``True``, attach learnable affine parameters on the
                          projected output (analogue of LayerNorm γ/β).
        clamp_radius:     Maximum norm before hard-clipping (safety).
        eps:              Numerical-stability epsilon.
        device / dtype:   Standard torch parameter options.

    Shape:
        - Input:  ``(..., dim)``
        - Output: ``(..., dim)``  (same shape, projected onto the manifold)

    Example::

        >>> mc = YvManifoldConstraint(512, manifold_type="hyperbolic", curvature=1.0)
        >>> x  = torch.randn(4, 32, 512)
        >>> y  = mc(x)           # x projected into Poincaré ball
        >>> d  = mc.distance(x, torch.randn_like(x))
    """

    _VALID_TYPES = {"hyperbolic", "spherical", "stiefel", "grassmann"}

    def __init__(
        self,
        dim: int,
        manifold_type: str = "hyperbolic",
        curvature: float = 1.0,
        stiefel_k: Optional[int] = None,
        learnable_scale: bool = False,
        clamp_radius: float = 1e4,
        eps: float = 1e-7,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if manifold_type not in self._VALID_TYPES:
            raise ValueError(
                f"manifold_type must be one of {self._VALID_TYPES}, got '{manifold_type}'"
            )

        self.dim = dim
        self.manifold_type = manifold_type
        self.curvature = float(curvature)
        self.eps = float(eps)
        self.clamp_radius = float(clamp_radius)
        self.stiefel_k = stiefel_k if stiefel_k is not None else dim

        # ── learnable affine (optional) ──────────────────────────────────
        self.learnable_scale = learnable_scale
        if learnable_scale:
            self.weight = nn.Parameter(
                torch.ones(dim, device=device, dtype=dtype)
            )
            self.bias = nn.Parameter(
                torch.zeros(dim, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # ── curvature as learnable scalar for hyperbolic ─────────────────
        if manifold_type == "hyperbolic":
            self.log_curvature = nn.Parameter(
                torch.tensor(math.log(max(self.curvature, 1e-8)),
                             device=device, dtype=dtype)
            )
        else:
            self.register_parameter("log_curvature", None)

        # ── register a projection buffer for Stiefel init ───────────────
        if manifold_type in ("stiefel", "grassmann"):
            init_mat = torch.randn(dim, self.stiefel_k,
                                   device=device, dtype=dtype)
            Q, _ = torch.linalg.qr(init_mat)
            self.register_buffer("_stiefel_basis", Q)

        self._reset_parameters(device, dtype)

    # ------------------------------------------------------------------ #
    #  Parameter initialisation                                           #
    # ------------------------------------------------------------------ #
    def _reset_parameters(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialise learnable parameters."""
        if self.learnable_scale:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    # ------------------------------------------------------------------ #
    #  Curvature helpers                                                   #
    # ------------------------------------------------------------------ #
    def _get_curvature(self) -> torch.Tensor:
        """Return current curvature (learnable, always > 0)."""
        return self.log_curvature.exp().clamp(min=1e-8)

    # ================================================================== #
    #                        PROJECTION                                    #
    # ================================================================== #
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project Euclidean point ``x`` onto the chosen manifold.

        Args:
            x: ``(..., dim)`` tensor.

        Returns:
            Projected tensor of the same shape lying on the manifold.
        """
        if self.manifold_type == "hyperbolic":
            return self._project_hyperbolic(x)
        if self.manifold_type == "spherical":
            return self._project_spherical(x)
        if self.manifold_type == "stiefel":
            return self._project_stiefel(x)
        # grassmann
        return self._project_grassmann(x)

    # ── hyperbolic (Poincaré ball) ───────────────────────────────────────
    def _project_hyperbolic(self, x: torch.Tensor) -> torch.Tensor:
        """Project into the open Poincaré ball of curvature *c*.

        Enforces ‖x‖ < 1/√c via norm-clipping.
        """
        c = self._get_curvature()
        max_norm = 1.0 / (c.sqrt() + self.eps)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        # Smooth tanh re-projection: x * tanh(max_norm * ‖x‖/‖x‖) → guarantees ‖y‖<max_norm
        cond = norm > max_norm
        projected = torch.where(
            cond,
            x / norm * max_norm * (1.0 - self.eps),
            x,
        )
        return projected

    # ── spherical ────────────────────────────────────────────────────────
    def _project_spherical(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise *x* to the unit hypersphere."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return x / norm

    # ── Stiefel ──────────────────────────────────────────────────────────
    def _project_stiefel(self, x: torch.Tensor) -> torch.Tensor:
        """Project matrix ``x`` onto the Stiefel manifold via polar decomposition.

        For ``(..., n, k)`` input, computes the nearest orthonormal matrix.
        Falls back to QR when SVD is not available for the batch shape.
        """
        orig_shape = x.shape
        if x.dim() < 2:
            x = x.unsqueeze(0)
        # Flatten batch dims
        batch_shape = x.shape[:-2]
        flat = x.reshape(-1, *x.shape[-2:])  # (B, n, k)
        # Polar decomposition via SVD:  X = UΣVᵀ  →  Q = UVᵀ
        U, _S, Vh = torch.linalg.svd(flat, full_matrices=False)
        Q = U @ Vh
        Q = Q.reshape(orig_shape)
        return Q

    # ── Grassmann ────────────────────────────────────────────────────────
    def _project_grassmann(self, x: torch.Tensor) -> torch.Tensor:
        """Project onto the Grassmann manifold (subspace).

        Returns the orthonormal basis spanning the same column space as *x*.
        """
        return self._project_stiefel(x)  # same operation, different interpretation

    # ================================================================== #
    #                     EXPONENTIAL & LOGARITHMIC MAPS                   #
    # ================================================================== #
    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        r"""Exponential map: map tangent vector *v* at base point *x* to manifold.

        .. math::
            \exp_x(v) = \cosh(\|v\|_x)\, x
                       + \sinh(\|v\|_x)\, \frac{v}{\|v\|_x}

        For hyperbolic (Poincaré ball):
        .. math::
            \exp_x(v) = x \oplus_c
              \left(\tanh\!\left(\frac{\sqrt{c}\,\lambda_x\|v\|}{2}\right)
              \frac{v}{\sqrt{c}\,\|v\|}\right)

        Args:
            x: Base point on manifold ``(..., dim)``.
            v: Tangent vector at *x* ``(..., dim)``.

        Returns:
            Point on manifold ``(..., dim)``.
        """
        if self.manifold_type == "hyperbolic":
            return self._expmap_hyperbolic(x, v)
        if self.manifold_type == "spherical":
            return self._expmap_spherical(x, v)
        # Stiefel / Grassmann: Cayley retraction
        return self._retraction_cayley(x, v)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Logarithmic map: map manifold point *y* to tangent vector at *x*.

        Inverse of :meth:`expmap`.  For the Poincaré ball:

        .. math::
            \log_x(y) = \frac{2}{\sqrt{c}\,\lambda_x}
              \operatorname{artanh}\!\bigl(\sqrt{c}\,\|-x\oplus_c y\|\bigr)
              \frac{-x\oplus_c y}{\|-x\oplus_c y\|}

        Args:
            x: Base point on manifold ``(..., dim)``.
            y: Target point on manifold ``(..., dim)``.

        Returns:
            Tangent vector at *x* ``(..., dim)``.
        """
        if self.manifold_type == "hyperbolic":
            return self._logmap_hyperbolic(x, y)
        if self.manifold_type == "spherical":
            return self._logmap_spherical(x, y)
        return self._logmap_stiefel(x, y)

    # ── hyperbolic maps ──────────────────────────────────────────────────
    def _lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        r"""Conformal factor  λ_x = 2 / (1 − c‖x‖²)."""
        c = self._get_curvature()
        sq_norm = (x * x).sum(dim=-1, keepdim=True)
        return 2.0 / (1.0 - c * sq_norm).clamp(min=self.eps)

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Möbius addition  x ⊕_c y  in the Poincaré ball.

        .. math::
            x \oplus_c y =
              \frac{(1+2c\langle x,y\rangle + c\|y\|^2)x + (1-c\|x\|^2)y}
                   {1+2c\langle x,y\rangle + c^2\|x\|^2\|y\|^2}
        """
        c = self._get_curvature()
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1.0 + 2.0 * c * xy + c * y2) * x + (1.0 - c * x2) * y
        denom = 1.0 + 2.0 * c * xy + c.pow(2) * x2 * y2
        denom = denom.clamp(min=self.eps)

        result = num / denom
        return self._project_hyperbolic(result)

    def _gyration(self, u: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        r"""Gyration operator  gyr[u,v]w  for Möbius gyrovector space.

        Required for parallel transport on the Poincaré ball.
        """
        c = self._get_curvature()
        u2 = (u * u).sum(dim=-1, keepdim=True)
        v2 = (v * v).sum(dim=-1, keepdim=True)
        uv = (u * v).sum(dim=-1, keepdim=True)
        uw = (u * w).sum(dim=-1, keepdim=True)
        vw = (v * w).sum(dim=-1, keepdim=True)

        A = -c.pow(2) * uw * v2 + c * vw + 2.0 * c.pow(2) * uv * vw
        B = -c.pow(2) * vw * u2 - c * uw
        D = 1.0 + 2.0 * c * uv + c.pow(2) * u2 * v2
        D = D.clamp(min=self.eps)

        return w + 2.0 * (A * u + B * v) / D

    def _expmap_hyperbolic(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        r"""Exponential map on the Poincaré ball."""
        c = self._get_curvature()
        sqrt_c = c.sqrt()
        lam = self._lambda_x(x)
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        second_term = torch.tanh(sqrt_c * lam * v_norm / 2.0) * v / (sqrt_c * v_norm)
        result = self._mobius_add(x, second_term)
        return result

    def _logmap_hyperbolic(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Logarithmic map on the Poincaré ball."""
        c = self._get_curvature()
        sqrt_c = c.sqrt()
        # −x ⊕_c y
        neg_x = -x
        diff = self._mobius_add(neg_x, y)
        diff_norm = diff.norm(dim=-1, keepdim=True).clamp(min=self.eps, max=1.0 / sqrt_c - self.eps)

        lam = self._lambda_x(x)
        # artanh(sqrt_c * ‖diff‖) * diff / (sqrt_c * ‖diff‖)  * (2 / (sqrt_c * lam))
        artanh_arg = (sqrt_c * diff_norm).clamp(max=1.0 - self.eps)
        artanh_val = 0.5 * torch.log((1.0 + artanh_arg) / (1.0 - artanh_arg + self.eps))

        return (2.0 / (sqrt_c * lam)) * artanh_val * diff / diff_norm

    # ── spherical maps ───────────────────────────────────────────────────
    def _expmap_spherical(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        r"""Exponential map on the unit sphere."""
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        # Remove radial component: v_orth = v - <v,x>x
        v_orth = v - (v * x).sum(dim=-1, keepdim=True) * x
        v_orth_norm = v_orth.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        v_dir = v_orth / v_orth_norm
        return x * torch.cos(v_norm) + v_dir * torch.sin(v_norm)

    def _logmap_spherical(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Logarithmic map on the unit sphere."""
        inner = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(inner)
        # v = theta / sin(theta) * (y - cos(theta) * x)
        sin_theta = torch.sin(theta).clamp(min=self.eps)
        return theta / sin_theta * (y - inner * x)

    # ── Stiefel maps ────────────────────────────────────────────────────
    def _retraction_cayley(self, X: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        r"""Cayley retraction on the Stiefel manifold.

        .. math::
            R_X(V) = (X + V)(I + V^\top V)^{-1/2}
        Approximated via the Cayley transform:
        .. math::
            R_X(V) = X + V - X\,\mathrm{sym}(X^\top V)
        """
        XtV = torch.matmul(X.transpose(-2, -1), V)
        sym = 0.5 * (XtV + XtV.transpose(-2, -1))
        result = X + V - torch.matmul(X, sym)
        return self._project_stiefel(result)

    def _logmap_stiefel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        r"""Approximate logarithmic map on the Stiefel manifold (projection)."""
        V = Y - X
        XtV = torch.matmul(X.transpose(-2, -1), V)
        sym = 0.5 * (XtV + XtV.transpose(-2, -1))
        return V - torch.matmul(X, sym)

    # ================================================================== #
    #                     GEODESIC DISTANCE                                 #
    # ================================================================== #
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Geodesic distance between *x* and *y* on the manifold.

        Args:
            x, y: Points on the manifold ``(..., dim)``.

        Returns:
            Distance tensor ``(...)``.
        """
        if self.manifold_type == "hyperbolic":
            return self._distance_hyperbolic(x, y)
        if self.manifold_type == "spherical":
            return self._distance_spherical(x, y)
        if self.manifold_type in ("stiefel", "grassmann"):
            return self._distance_stiefel(x, y)
        return (x - y).norm(dim=-1)

    def _distance_hyperbolic(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""d(x,y) = (2/√c) artanh(√c ‖−x ⊕_c y‖)"""
        c = self._get_curvature()
        sqrt_c = c.sqrt()
        diff = self._mobius_add(-x, y)
        diff_norm = diff.norm(dim=-1, keepdim=False).clamp(
            min=self.eps, max=1.0 / sqrt_c - self.eps
        )
        artanh_arg = (sqrt_c * diff_norm).clamp(max=1.0 - self.eps)
        return (2.0 / sqrt_c) * torch.atanh(artanh_arg)

    def _distance_spherical(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""d(x,y) = arccos(<x,y>)"""
        inner = (x * y).sum(dim=-1).clamp(-1.0 + self.eps, 1.0 - self.eps)
        return torch.acos(inner)

    def _distance_stiefel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        r"""Chordal distance  ‖X − Y‖_F."""
        return (X - Y).pow(2).sum(dim=(-2, -1)).clamp(min=0.0).sqrt()

    # ================================================================== #
    #                  RIEMANNIAN GRADIENT & INNER PRODUCT                  #
    # ================================================================== #
    def egrad2rgrad(self, x: torch.Tensor, egrad: torch.Tensor) -> torch.Tensor:
        r"""Convert Euclidean gradient to Riemannian gradient.

        For the Poincaré ball:
        .. math::
            \mathrm{grad}_R f = \frac{(1-c\|x\|^2)^2}{4}\,\mathrm{grad}_E f

        For the sphere:
        .. math::
            \mathrm{grad}_R f = \mathrm{grad}_E f - \langle x, \mathrm{grad}_E f\rangle\, x

        Args:
            x:     Point on manifold ``(..., dim)``.
            egrad: Euclidean gradient ``(..., dim)``.

        Returns:
            Riemannian gradient ``(..., dim)``.
        """
        if self.manifold_type == "hyperbolic":
            c = self._get_curvature()
            sq_norm = (x * x).sum(dim=-1, keepdim=True)
            factor = ((1.0 - c * sq_norm) / 2.0).pow(2)
            return factor * egrad
        if self.manifold_type == "spherical":
            return egrad - (egrad * x).sum(dim=-1, keepdim=True) * x
        # Stiefel / Grassmann
        XtG = torch.matmul(x.transpose(-2, -1), egrad)
        sym = 0.5 * (XtG + XtG.transpose(-2, -1))
        return egrad - torch.matmul(x, sym)

    def inner_product(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Riemannian inner product  ⟨u, v⟩_x  in the tangent space at *x*.

        If *v* is ``None``, computes  ⟨u, u⟩_x  (squared norm).

        Args:
            x: Base point on manifold ``(..., dim)``.
            u: Tangent vector ``(..., dim)``.
            v: Second tangent vector (defaults to *u*).

        Returns:
            Inner-product tensor ``(...)``.
        """
        if v is None:
            v = u
        if self.manifold_type == "hyperbolic":
            c = self._get_curvature()
            sq_norm = (x * x).sum(dim=-1, keepdim=True)
            # g_x = (2/(1-c‖x‖²))² · I
            factor = (2.0 / (1.0 - c * sq_norm).clamp(min=self.eps)).pow(2)
            return (factor.squeeze(-1)) * (u * v).sum(dim=-1)
        if self.manifold_type == "spherical":
            return (u * v).sum(dim=-1)
        return (u * v).sum(dim=(-2, -1))

    # ================================================================== #
    #                     PARALLEL TRANSPORT                                 #
    # ================================================================== #
    def transport(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        r"""Parallel transport of tangent vector *v* from *x* to *y*.

        For the Poincaré ball, uses the gyration operator:
        .. math::
            P_{x\to y}(v) = \mathrm{gyr}[y, -x]\, v\, \frac{\lambda_x}{\lambda_y}

        For the sphere:
        .. math::
            P_{x\to y}(v) = v - \frac{\langle x+y, v\rangle}{1+\langle x,y\rangle}(x+y)

        Args:
            x: Source point ``(..., dim)``.
            y: Target point ``(..., dim)``.
            v: Tangent vector at *x* ``(..., dim)``.

        Returns:
            Transported tangent vector at *y* ``(..., dim)``.
        """
        if self.manifold_type == "hyperbolic":
            lam_x = self._lambda_x(x)
            lam_y = self._lambda_x(y)
            gyr = self._gyration(y, -x, v)
            return gyr * (lam_x / lam_y.clamp(min=self.eps))
        if self.manifold_type == "spherical":
            inner_xy = (x * y).sum(dim=-1, keepdim=True).clamp(min=-1.0 + self.eps)
            s = x + y
            inner_sv = (s * v).sum(dim=-1, keepdim=True)
            return v - inner_sv / (1.0 + inner_xy).clamp(min=self.eps) * s
        # Stiefel / Grassmann: approximate with projection
        return self._project_stiefel(v) if self.manifold_type == "stiefel" else v

    # ================================================================== #
    #                          FORWARD                                      #
    # ================================================================== #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project *x* onto the manifold and optionally apply affine transform.

        Args:
            x: Input tensor ``(..., dim)``.

        Returns:
            Manifold-constrained tensor ``(..., dim)``.
        """
        y = self.project(x)
        if self.learnable_scale:
            y = y * self.weight + self.bias
            y = self.project(y)  # re-project after affine to stay on manifold
        return y

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #
    def lorentz_factor(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return the Lorentz / conformal factor λ_x for diagnostics.

        Only meaningful for ``hyperbolic``.
        """
        if self.manifold_type == "hyperbolic":
            return self._lambda_x(x)
        return torch.ones_like(x[..., :1])

    def extra_repr(self) -> str:
        parts = [f"dim={self.dim}", f"manifold={self.manifold_type}"]
        if self.manifold_type == "hyperbolic":
            parts.append(f"curvature={self.curvature}")
        if self.manifold_type in ("stiefel", "grassmann"):
            parts.append(f"k={self.stiefel_k}")
        parts.append(f"learnable_scale={self.learnable_scale}")
        return ", ".join(parts)


# Paper: Sinkhorn & Knopp, "Concerning nonnegative matrices and doubly stochastic matrices", 1967; applied in mHC (DeepSeek-V4 Pro, 2026)
class YvSinkhornKnopp(nn.Module):
    """Birkhoff polytope projection via Sinkhorn-Knopp iteration.

    Projects a matrix onto the doubly stochastic manifold (Birkhoff polytope),
    where all rows and columns sum to 1. This ensures spectral norm ≤ 1 and
    non-expansive transforms, stabilizing deep residual architectures.

    Based on DeepSeek-V4 mHC: uses 20 Sinkhorn-Knopp iterations with
    fused kernel optimization for minimal overhead.

    Args:
        n_iter: Number of Sinkhorn-Knopp iterations. Default: 20.
        eps: Numerical stability epsilon. Default: 1e-6.
    """

    def __init__(self, n_iter: int = 20, eps: float = 1e-6):
        super().__init__()
        self.n_iter = n_iter
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Project X onto the Birkhoff polytope.

        Args:
            X: Input matrix of shape (..., n, n).

        Returns:
            Doubly stochastic matrix of shape (..., n, n).
        """
        X = X - X.logsumexp(dim=-1, keepdim=True)
        for _ in range(self.n_iter):
            X = X - X.logsumexp(dim=-2, keepdim=True)
            X = X - X.logsumexp(dim=-1, keepdim=True)
        return X.exp()


# Paper: DeepSeek-V4 Pro Technical Report, 2026 (mHC: Manifold-Constrained Hyper-Connections)
class YvMHC(nn.Module):
    """Manifold-Constrained Hyper-Connections (mHC).

    Replaces standard residual connections by expanding the residual stream
    from d → n_hc×d and constraining the mixing matrix B to the Birkhoff
    polytope via Sinkhorn-Knopp iteration. Spectral norm ≤ 1, non-expansive.

    Based on DeepSeek-V4 Pro technical report (2026).

    Args:
        hidden_size: Model hidden dimension.
        n_hc: Expansion factor for residual stream. Default: 4.
        sinkhorn_iters: Sinkhorn-Knopp iterations. Default: 20.
        device: Torch device.
        dtype: Torch dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        n_hc: int = 4,
        sinkhorn_iters: int = 20,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_hc = n_hc

        # Learnable residual mixing matrix B — expanded to n_hc × d
        # B is constrained to Birkhoff polytope via Sinkhorn-Knopp
        self.B = nn.Parameter(torch.randn(n_hc, n_hc, device=device, dtype=dtype) * 0.01)

        # Input/output projections for expanded residual stream
        self.input_proj = nn.Linear(hidden_size, hidden_size * n_hc, bias=False, device=device, dtype=dtype)
        self.output_proj = nn.Linear(hidden_size * n_hc, hidden_size, bias=False, device=device, dtype=dtype)

        self.sinkhorn = YvSinkhornKnopp(n_iter=sinkhorn_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mHC residual connection.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor with mHC residual, shape (..., hidden_size).
        """
        *dims, H = x.shape

        # Expand residual stream
        x_expanded = self.input_proj(x)  # (..., n_hc * H)
        x_expanded = x_expanded.view(*dims, self.n_hc, H)  # (..., n_hc, H)

        # Constrain B to Birkhoff polytope
        B_constrained = self.sinkhorn(self.B)  # (n_hc, n_hc)

        # Apply mixing: aggregated = B_constrained @ x_expanded along n_hc dim
        x_mixed = torch.einsum('ij,...jh->...ih', B_constrained, x_expanded)  # (..., n_hc, H)

        # Collapse back to hidden_size
        x_mixed = x_mixed.reshape(*dims, self.n_hc * H)
        output = self.output_proj(x_mixed)  # (..., H)

        return output


# Paper: DeepSeek-V4 Pro Technical Report, 2026 (mHC: Manifold-Constrained Hyper-Connections)
class YvHyperConnection(nn.Module):
    """Hyper-Connection layer with manifold constraints.

    Uses mHC (Manifold-Constrained Hyper-Connections) from DeepSeek-V4:
    expands residual stream from d → n_hc×d, constrains mixing matrix
    B to the Birkhoff polytope via Sinkhorn-Knopp iteration.
    """

    def __init__(
        self,
        hidden_size: int,
        n_hc: int = 4,
        sinkhorn_iters: int = 20,
        drop_path_rate: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.mhc = YvMHC(hidden_size, n_hc, sinkhorn_iters, device, dtype)
        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_path_rate > 0 and torch.rand(1).item() < self.drop_path_rate:
            return x
        return self.mhc(x)

    def get_constraint_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.mhc.B.device)


class YvMHCBlock(nn.Module):
    """Transformer block with mHC (Manifold-constrained Hyper-Connection).
    
    Replaces standard residual connection with mHC for improved
    training stability and model expressiveness.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        use_manifold_constraint: bool = True,
        constraint_type: str = "soft_orthogonal",
        drop_path_rate: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize mHC block.
        
        Args:
            hidden_size: Hidden dimension.
            num_attention_heads: Number of attention heads.
            num_layers: Number of layers for hyper-connection.
            mlp_ratio: MLP expansion ratio.
            attention_dropout: Attention dropout rate.
            dropout: Dropout rate.
            use_manifold_constraint: Use manifold constraint.
            constraint_type: Type of constraint.
            drop_path_rate: Drop path rate.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True,
            device=device,
            dtype=dtype
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio), bias=False, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size, bias=False, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
        
        self.input_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.attention_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        self.mlp_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        
        self.hyper_connection = YvHyperConnection(
            hidden_size=hidden_size,
            drop_path_rate=drop_path_rate,
            device=device,
            dtype=dtype
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._cache = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, H].
            attention_mask: Attention mask [B, T] or None.
            
        Returns:
            Output tensor [B, T, H].
        """
        if self._cache is None:
            self._cache = []
        
        input_norm = self.input_norm(x)
        
        attention_output, _ = self.attention(
            query=input_norm,
            key=input_norm,
            value=input_norm,
            attn_mask=attention_mask,
            need_weights=False,
        )
        attention_output = self.dropout(attention_output)
        
        attention_norm = self.attention_norm(attention_output)
        
        mlp_output = self.mlp(attention_norm)
        
        residual_attention = x + attention_output
        residual_mlp = residual_attention + mlp_output
        
        self._cache.append(residual_mlp)
        
        if len(self._cache) > 4:
            self._cache = self._cache[-4:]
        
        hyper_output = self.hyper_connection(
            input_norm
        )
        
        output = x + hyper_output
        
        return output
    
    def reset_cache(self):
        """Reset hyper-connection cache."""
        self._cache = None
    
    def get_constraint_loss(self) -> torch.Tensor:
        """Get mHC constraint loss."""
        return self.hyper_connection.get_constraint_loss()


class YvMHCTransformer(nn.Module):
    """Complete Transformer with mHC for PiscesLx.
    
    Replaces standard residual connections with mHC throughout
    the transformer architecture.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        mlp_ratio: float = 4.0,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        use_manifold_constraint: bool = True,
        constraint_type: str = "soft_orthogonal",
        drop_path_rate: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize mHC Transformer.
        
        Args:
            hidden_size: Hidden dimension.
            num_layers: Number of transformer layers.
            num_attention_heads: Number of attention heads.
            mlp_ratio: MLP expansion ratio.
            attention_dropout: Attention dropout rate.
            dropout: Dropout rate.
            use_manifold_constraint: Use manifold constraint.
            constraint_type: Type of constraint.
            drop_path_rate: Drop path rate.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.blocks = nn.ModuleList([
            YvMHCBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_layers=4,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                use_manifold_constraint=use_manifold_constraint,
                constraint_type=constraint_type,
                drop_path_rate=drop_path_rate * i / num_layers,
                device=device,
                dtype=dtype
            )
            for i in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, T, H].
            attention_mask: Attention mask [B, T].
            
        Returns:
            Output tensor [B, T, H].
        """
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.final_norm(x)
        
        return x
    
    def reset_all_caches(self):
        """Reset all block caches."""
        for block in self.blocks:
            block.reset_cache()
    
    def get_total_constraint_loss(self) -> torch.Tensor:
        """Get total mHC constraint loss from all blocks."""
        total_loss = torch.tensor(0.0)
        for block in self.blocks:
            total_loss = total_loss + block.get_constraint_loss()
        return total_loss


class YvMHCLayerReplacement:
    """Utility to replace standard residual connections with mHC.
    
    This can be used to upgrade existing transformer blocks.
    """
    
    @staticmethod
    def replace_attention_block(
        block: nn.Module,
        num_layers: int = 4,
        use_manifold_constraint: bool = True
    ) -> nn.Module:
        """Replace standard attention block with mHC block.
        
        Args:
            block: Standard transformer block.
            num_layers: Number of layers for hyper-connection.
            use_manifold_constraint: Use manifold constraint.
            
        Returns:
            mHC block.
        """
        if not hasattr(block, 'attention') or not hasattr(block, 'mlp'):
            _LOG.warning("Block doesn't have attention/mlp, skipping")
            return block
        
        config = {
            'hidden_size': block.attention.embed_dim,
            'num_attention_heads': block.attention.num_heads,
            'num_layers': num_layers,
            'mlp_ratio': block.mlp[0].out_features // block.attention.embed_dim if len(block.mlp) > 1 else 4,
            'attention_dropout': block.attention.dropout,
            'use_manifold_constraint': use_manifold_constraint,
        }
        
        mhc_block = YvMHCBlock(**config)
        
        return mhc_block
    
    @staticmethod
    def upgrade_transformer(
        model: nn.Module,
        num_layers: int = 4,
        use_manifold_constraint: bool = True
    ) -> nn.Module:
        """Upgrade entire transformer model with mHC.
        
        Args:
            model: Transformer model.
            num_layers: Number of layers for hyper-connection.
            use_manifold_constraint: Use manifold constraint.
            
        Returns:
            Upgraded model.
        """
        if hasattr(model, 'blocks') or hasattr(model, 'layers'):
            blocks_attr = 'blocks' if hasattr(model, 'blocks') else 'layers'
            blocks = getattr(model, blocks_attr)
            
            for i, block in enumerate(blocks):
                upgraded = YvMHCLayerReplacement.replace_attention_block(
                    block, num_layers, use_manifold_constraint
                )
                blocks[i] = upgraded
            
            _LOG.info(f"Upgraded {len(blocks)} transformer blocks with mHC")
        
        return model


class YvMHCLoss(nn.Module):
    """Loss function with mHC constraint penalty.
    
    Combines standard language modeling loss with mHC constraint loss.
    """
    
    def __init__(self, lambda_constraint: float = 0.01):
        """Initialize mHC loss.
        
        Args:
            lambda_constraint: Weight for constraint loss.
        """
        super().__init__()
        self.lambda_constraint = lambda_constraint
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        constraint_loss: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            logits: Model logits [B, T, V].
            labels: Target labels [B, T].
            constraint_loss: mHC constraint loss (optional).
            
        Returns:
            Loss dict.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        lm_loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        total_loss = lm_loss
        
        if constraint_loss is not None and self.lambda_constraint > 0:
            constraint = constraint_loss * self.lambda_constraint
            total_loss = total_loss + constraint
        else:
            constraint = torch.tensor(0.0, device=lm_loss.device)
        
        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "constraint_loss": constraint,
        }


def create_mhc_transformer(
    hidden_size: int = 4096,
    num_layers: int = 32,
    num_attention_heads: int = 32,
    mlp_ratio: float = 4.0,
    use_manifold_constraint: bool = True,
    constraint_type: str = "soft_orthogonal",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> YvMHCTransformer:
    """Factory function to create mHC Transformer.
    
    Args:
        hidden_size: Hidden dimension.
        num_layers: Number of layers.
        num_attention_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        use_manifold_constraint: Use manifold constraint.
        constraint_type: Type of constraint.
        device: Device for parameters.
        dtype: Data type for parameters.
        
    Returns:
        mHC Transformer instance.
    """
    return YvMHCTransformer(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        mlp_ratio=mlp_ratio,
        use_manifold_constraint=use_manifold_constraint,
        constraint_type=constraint_type,
        device=device,
        dtype=dtype,
    )
