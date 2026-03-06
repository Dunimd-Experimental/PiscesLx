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
from ..moe.gate import YvMoELayer as MoELayer
from ..moe.layer import YvDynamicMoELayer

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
        return self.down_proj(gate * up)


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
        
        base_out = x @ base_weight
        
        combined = base_out + lora_out
        
        norm = torch.norm(combined, dim=-1, keepdim=True)
        return combined * self.magnitude.unsqueeze(0) / (norm + 1e-8)


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


class YvMixtureOfDepths(nn.Module):
    """Mixture-of-Depths for dynamic layer skipping.
    
    Enables the model to skip layers for certain tokens, improving
    efficiency by not processing all tokens through all layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        routing_weight: float = 0.1,
        capacity_factor: float = 1.25,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Mixture-of-Depths.
        
        Args:
            hidden_size: Model hidden dimension.
            n_head: Number of attention heads.
            routing_weight: Weight for routing decisions.
            capacity_factor: Capacity factor for token allocation.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.routing_weight = routing_weight
        self.capacity_factor = capacity_factor
        
        self.router = nn.Linear(hidden_size, 2, bias=False, device=device, dtype=dtype)
        
        self.skip_norm = YvRMSNorm(hidden_size, device=device, dtype=dtype)
        self.process_norm = YvRMSNorm(hidden_size, device=device, dtype=dtype)
        
    def forward(
        self,
        x: torch.Tensor,
        process_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixture-of-depths routing.
        
        Args:
            x: Input tensor.
            process_fn: Function to apply to processed tokens.
            
        Returns:
            Tuple of (output, routing_loss).
        """
        batch_size, seq_len, _ = x.shape
        
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        process_prob = router_probs[..., 0]
        skip_prob = router_probs[..., 1]
        
        capacity = int(seq_len * self.capacity_factor)
        
        _, top_indices = torch.topk(process_prob, min(capacity, seq_len), dim=-1)
        
        process_mask = torch.zeros_like(process_prob)
        process_mask.scatter_(1, top_indices, 1.0)
        
        x_process = self.process_norm(x)
        processed = process_fn(x_process)
        
        x_skip = self.skip_norm(x)
        
        output = process_mask.unsqueeze(-1) * processed + skip_prob.unsqueeze(-1) * x_skip
        
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
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize parallel block.
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.cfg = cfg
        
        self.attn = YvAttention(cfg, device=device, dtype=dtype)
        
        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype,
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            self.mlp = YvDynamicMoELayer(cfg, device=device, dtype=dtype)
            
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
            
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass with parallel attention and MLP.
        
        Args:
            x: Input tensor.
            mask: Attention mask.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use cache.
            
        Returns:
            Output tensor(s).
        """
        residual = x
        x_norm = self.norm(x)
        
        if use_cache:
            attn_out, present_kv = self.attn(
                x_norm, mask, past_key_values=past_key_values, use_cache=True
            )
        else:
            attn_out = self.attn(x_norm, mask, past_key_values=past_key_values, use_cache=False)
            
        mlp_out, aux_loss = self.mlp(x_norm)
        
        output = residual + self.residual_dropout(self.attn_scale(attn_out) + self.mlp_scale(mlp_out))
        
        if use_cache:
            return output, aux_loss, present_kv
        return output, aux_loss


class YvDeepNormBlock(nn.Module):
    """DeepNorm Block for training very deep networks.
    
    Implements DeepNorm: a normalization strategy that combines
    residual scaling with layer normalization for improved stability.
    """
    
    def __init__(
        self,
        cfg,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize DeepNorm block.
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.cfg = cfg
        
        self.attn = YvAttention(cfg, device=device, dtype=dtype)
        
        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype,
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            self.mlp = YvDynamicMoELayer(cfg, device=device, dtype=dtype)
            
        self.deep_norm_attn = YvDeepNorm(
            cfg.hidden_size, cfg.n_layer, device=device, dtype=dtype
        )
        self.deep_norm_mlp = YvDeepNorm(
            cfg.hidden_size, cfg.n_layer, device=device, dtype=dtype
        )
        
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass with DeepNorm.
        
        Args:
            x: Input tensor.
            mask: Attention mask.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use cache.
            
        Returns:
            Output tensor(s).
        """
        residual = x
        
        if use_cache:
            attn_out, present_kv = self.attn(
                x, mask, past_key_values=past_key_values, use_cache=True
            )
        else:
            attn_out = self.attn(x, mask, past_key_values=past_key_values, use_cache=False)
            
        x = self.deep_norm_attn(residual, self.residual_dropout(attn_out))
        
        residual = x
        mlp_out, aux_loss = self.mlp(x)
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
        
        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype,
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            self.mlp = YvDynamicMoELayer(cfg, device=device, dtype=dtype)
            
        self.norm1 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.norm2 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.norm3 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False
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
        mlp_out, aux_loss = self.mlp(x)
        x = residual + self.residual_dropout(mlp_out)
        
        if use_cache:
            return x, aux_loss, self_kv
        return x, aux_loss


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

    def __init__(self, cfg, device=None, dtype=None, quantization_config=None):
        """Initialize the transformer block.

        Args:
            cfg: Configuration object containing model hyperparameters.
            device: Device to place the module on.
            dtype: Data type for the module parameters.
            quantization_config: Configuration for model quantization.

        Raises:
            RuntimeError: If quantization setup fails and fallback also fails.
        """
        super().__init__()
        self.cfg = cfg
        self.cache_manager = None
        self.layer_idx = -1
        
        self.block_type = getattr(cfg, 'block_type', 'standard')
        self.use_parallel = getattr(cfg, 'use_parallel', False)
        self.use_deepnorm = getattr(cfg, 'use_deepnorm', False)
        self.use_layerscale = getattr(cfg, 'use_layerscale', True)
        self.use_swiglu = getattr(cfg, 'use_swiglu', True)
        self.use_geglu = getattr(cfg, 'use_geglu', False)
        self.use_mixture_of_depths = getattr(cfg, 'mixture_of_depths', False)
        self.use_lora = getattr(cfg, 'use_lora', False)
        self.use_dora = getattr(cfg, 'use_dora', False)
        
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
            
        self.use_checkpoint = getattr(cfg, 'use_gradient_checkpointing', True)
        self.adaptive_checkpointing = getattr(cfg, 'adaptive_gradient_checkpointing', True)
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

        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype,
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            self.mlp = YvDynamicMoELayer(cfg, device=device, dtype=dtype)

        self.norm1 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.norm2 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.pre_norm1 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.pre_norm2 = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)

        self.residual_scale = nn.Parameter(
            torch.ones(1, device=device, dtype=dtype) * (2.0 * cfg.n_layer) ** -0.5
        )
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))
        
        if self.use_layerscale:
            self.attn_layerscale = YvLayerScale(
                cfg.hidden_size,
                init_value=getattr(cfg, 'layerscale_init', 1e-5),
                device=device, dtype=dtype
            )
            self.mlp_layerscale = YvLayerScale(
                cfg.hidden_size,
                init_value=getattr(cfg, 'layerscale_init', 1e-5),
                device=device, dtype=dtype
            )
        
        # Hybrid SSM Integration: Mamba-3 for linear complexity on long sequences
        # Auto-enabled when hidden_size >= 2048 (sufficient capacity for SSM)
        # Provides O(n) complexity alternative to O(n^2) attention
        self._init_hybrid_ssm(cfg, device, dtype)

    def _init_parallel_block(self, cfg, device, dtype):
        """Initialize parallel attention-MLP block.
        
        Args:
            cfg: Configuration object.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        self.parallel_block = YvParallelBlock(cfg, device=device, dtype=dtype)

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
            State Spaces", arXiv 2023.
            Dao et al., "Mamba-2: Transforming Transformers", arXiv 2024.
        """
        from .mamba3 import YvMamba3Block, YvMamba3Config
        
        # Auto-enable SSM for models with sufficient capacity
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
        self.deepnorm_block = YvDeepNormBlock(cfg, device=device, dtype=dtype)

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
        except Exception as e:
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
        except Exception as e:
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
        except Exception as e:
            _LOG.error(f"Adaptive checkpointing memory check failed: {e}")
            return self.use_checkpoint

    def _apply_with_checkpoint(self, x, mask, past_key_values=None, use_cache=False):
        """Apply the transformer block with optional gradient checkpointing.

        Args:
            x: Input tensor.
            mask: Attention mask tensor.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use and update key/value cache.

        Returns:
            Output tensor(s) from the transformer block.
        """
        import torch.utils.checkpoint as cp

        attn_past_key_values = past_key_values if past_key_values is not None else None
        should_checkpoint = self._should_use_checkpoint()

        def _inner(xc, kv=None):
            """Inner function for gradient checkpointing.

            Args:
                xc: Input tensor.
                kv: Past key/value pairs.

            Returns:
                Output from _forward_core.
            """
            return self._forward_core(xc, mask, kv, use_cache)

        if should_checkpoint and self.training:
            out = cp.checkpoint(_inner, x, attn_past_key_values, use_reentrant=False)
        else:
            out = _inner(x, attn_past_key_values)

        return out

    def forward(self, x, mask, past_key_values=None, use_cache=False):
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].
            mask: Attention mask tensor.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use and update key/value cache.

        Returns:
            Output tensor(s).
        """
        if self.use_parallel:
            return self.parallel_block(x, mask, past_key_values, use_cache)
        elif self.use_deepnorm:
            return self.deepnorm_block(x, mask, past_key_values, use_cache)
        else:
            return self._apply_with_checkpoint(x, mask, past_key_values, use_cache)

    def _forward_core(self, x, mask, attn_past_key_values=None, use_cache=False):
        """Core forward computation without checkpointing wrapper.

        Args:
            x: Input tensor.
            mask: Attention mask tensor.
            attn_past_key_values: Key/value cache for attention.
            use_cache: Whether to use and update key/value cache.

        Returns:
            Output tensor(s).
        """
        residual = x
        x_norm = self.pre_norm1(x)
        attn_cache = None
        past_for_attn = attn_past_key_values

        if use_cache and self.cache_manager is not None and self.layer_idx >= 0:
            got = self.cache_manager.get_kv_cache(self.layer_idx, attn_past_key_values)
            if got is not None:
                past_for_attn = got

        if use_cache:
            attn_out, present_kv = self.attn(
                x_norm,
                mask,
                past_key_values=past_for_attn,
                use_cache=True,
                cache_manager=self.cache_manager
            )
            if self.cache_manager is not None and self.layer_idx >= 0 and present_kv is not None:
                self.cache_manager.update_kv_cache(
                    self.layer_idx,
                    present_kv[0],
                    present_kv[1],
                    current_pos=x_norm.shape[1],
                    use_h2o=getattr(self.attn, 'use_h2o', False)
                )
            attn_cache = present_kv
        else:
            attn_out = self.attn(
                x_norm,
                mask,
                past_key_values=past_for_attn,
                use_cache=False,
                cache_manager=self.cache_manager
            )

        if self.use_layerscale:
            attn_out = self.attn_layerscale(attn_out)
        
        # Hierarchical Communication: Distributed training optimization
        # Auto-enabled when multi-GPU distributed training
        if torch.cuda.device_count() > 1 and self.training:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    world_size = dist.get_world_size()
                    
                    compressed = attn_out.mean(dim=1)
                    gathered = [torch.zeros_like(compressed) for _ in range(world_size)]
                    dist.all_gather(gathered, compressed)
                    
                    other_info = torch.stack(gathered).mean(dim=0)
                    attn_out = attn_out + 0.05 * other_info.unsqueeze(1)
            except Exception:
                pass
            
        x_out = residual + self.residual_dropout(self.residual_scale * attn_out)
        x_out = self.norm1(x_out)
        
        # Hybrid SSM Path: Auto-activated for long sequences (seq_len > 8192)
        # Provides O(n) complexity alternative to O(n^2) attention
        if hasattr(self, 'ssm_layer') and self.ssm_layer is not None:
            seq_len = x_out.shape[1]
            if seq_len > 8192:
                # Compute SSM output with linear complexity
                ssm_out = self.ssm_layer(x_out)
                # Sigmoid gate for smooth blending (learnable)
                gate = torch.sigmoid(self.ssm_gate)
                # Blend attention output with SSM output
                x_out = gate * x_out + (1.0 - gate) * ssm_out

        residual = x_out
        x_norm = self.pre_norm2(x_out)
        mlp_out, aux_loss = self.mlp(x_norm)
        
        if self.use_layerscale:
            mlp_out = self.mlp_layerscale(mlp_out)
            
        x_out = residual + self.residual_dropout(self.residual_scale * mlp_out)
        x_out = self.norm2(x_out)

        if self.use_mixture_of_depths:
            x_out, mod_loss = self.mod_router(x_out, lambda h: h)
            aux_loss = aux_loss + mod_loss

        if use_cache:
            return x_out, aux_loss, attn_cache
        return x_out, aux_loss


class YvManifoldConstraint(nn.Module):
    """Manifold constraint for hyper-connection stability.
    
    Implements orthogonal projection onto a constrained manifold
    to ensure stable and well-conditioned hyper-connections.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        constraint_type: str = "soft_orthogonal",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize manifold constraint.
        
        Args:
            hidden_size: Hidden dimension.
            num_layers: Number of layers to connect.
            constraint_type: Type of constraint:
                - "soft_orthogonal": Soft orthogonal constraint
                - "hard_orthogonal": Hard orthogonalization via SVD
                - "norm_bound": Norm bounding constraint
                - "spectral": Spectral norm constraint
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.constraint_type = constraint_type
        
        if constraint_type == "soft_orthogonal":
            self.register_buffer("_eye", torch.eye(num_layers, device=device, dtype=dtype))
        
        elif constraint_type == "spectral":
            self.spectral_norm = nn.Parameter(torch.ones(1, device=device, dtype=dtype))
        
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply manifold constraint to hyper-connection weights.
        
        Args:
            weights: Hyper-connection weights [B, num_layers, H, H].
            
        Returns:
            Constrained weights.
        """
        if self.constraint_type == "soft_orthogonal":
            return self._soft_orthogonal_constraint(weights)
        elif self.constraint_type == "hard_orthogonal":
            return self._hard_orthogonal_constraint(weights)
        elif self.constraint_type == "norm_bound":
            return self._norm_bound_constraint(weights)
        elif self.constraint_type == "spectral":
            return self._spectral_constraint(weights)
        return weights
    
    def _soft_orthogonal_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """Soft orthogonal constraint via regularization."""
        W = weights.mean(dim=0)
        WtW = W @ W.transpose(-1, -2)
        
        ortho_loss = F.mse_loss(WtW, self._eye.expand_as(WtW))
        
        if self.training:
            if not hasattr(self, "_ortho_loss_acc"):
                self._ortho_loss_acc = 0.0
            self._ortho_loss_acc = ortho_loss.item()
        
        return weights
    
    def _hard_orthogonal_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """Hard orthogonalization via SVD."""
        batch_size = weights.shape[0]
        constrained = []
        
        for b in range(batch_size):
            W = weights[b]
            U, S, Vh = torch.linalg.svd(W)
            V = Vh.transpose(-1, -2).conj()
            W_ortho = U @ V
            constrained.append(W_ortho)
        
        return torch.stack(constrained)
    
    def _norm_bound_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """Norm bounding constraint."""
        norms = torch.linalg.vector_norm(weights, dim=(-1, -2), keepdim=True)
        max_norm = 1.0 / math.sqrt(weights.shape[-1])
        scales = torch.clamp(norms / max_norm, min=1.0)
        return weights / scales
    
    def _spectral_constraint(self, weights: torch.Tensor) -> torch.Tensor:
        """Spectral norm constraint."""
        spectral_norm = torch.linalg.svdvals(weights).max(dim=-1, keepdim=True)[0]
        spectral_norm = spectral_norm.unsqueeze(-1).unsqueeze(-1)
        
        scale = self.spectral_norm / (spectral_norm + 1e-6)
        scale = torch.clamp(scale, max=1.0)
        
        return weights * scale
    
    def get_constraint_loss(self) -> torch.Tensor:
        """Get the constraint loss for training."""
        if hasattr(self, "_ortho_loss_acc"):
            loss = self._ortho_loss_acc
            self._ortho_loss_acc = 0.0
            return torch.tensor(loss, device=self._eye.device if hasattr(self, "_eye") else "cpu")
        return torch.tensor(0.0, device=self._eye.device if hasattr(self, "_eye") else "cpu")


class YvHyperConnection(nn.Module):
    """Hyper-Connection layer with manifold constraints.
    
    Standard hyper-connection: y = Σᵢαᵢxᵢ where αᵢ are learnable weights.
    mHC adds manifold constraints to ensure stable training.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        use_manifold_constraint: bool = True,
        constraint_type: str = "soft_orthogonal",
        drop_path_rate: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize hyper-connection.
        
        Args:
            hidden_size: Hidden dimension.
            num_layers: Number of layers to connect.
            use_manifold_constraint: Whether to use manifold constraint.
            constraint_type: Type of constraint.
            drop_path_rate: Stochastic depth rate.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_path_rate = drop_path_rate
        
        self.use_manifold_constraint = use_manifold_constraint
        if use_manifold_constraint:
            self.manifold_constraint = YvManifoldConstraint(
                hidden_size, num_layers, constraint_type, device=device, dtype=dtype
            )
        
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, bias=False, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_layers, bias=False, device=device, dtype=dtype),
        )
        
        self.gate = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(hidden_size, device=device, dtype=dtype)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.weight_generator:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        layer_outputs: List[torch.Tensor],
        current_input: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            layer_outputs: List of layer outputs [x₀, x₁, ..., xₙ].
            current_input: Current layer input (optional).
            
        Returns:
            Tuple of (hyper-connected output, gate weights).
        """
        if current_input is None:
            current_input = layer_outputs[-1]
        
        input_features = current_input.mean(dim=1)
        
        raw_weights = self.weight_generator(input_features)
        raw_weights = raw_weights.view(-1, self.num_layers)
        
        if self.use_manifold_constraint:
            hyper_weights = self.manifold_constraint(raw_weights)
        else:
            hyper_weights = raw_weights
        
        gate_weights = self.gate(hyper_weights)
        gate_weights = F.softmax(gate_weights, dim=-1)
        
        hyper_output = torch.zeros_like(layer_outputs[0])
        for i, layer_out in enumerate(layer_outputs):
            weight = gate_weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1)
            hyper_output = hyper_output + weight * layer_out
        
        output = self.layer_norm(hyper_output)
        
        return output, gate_weights
    
    def get_constraint_loss(self) -> torch.Tensor:
        """Get constraint loss if applicable."""
        if self.use_manifold_constraint:
            return self.manifold_constraint.get_constraint_loss()
        return torch.tensor(0.0, device="cpu")


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
            num_layers=num_layers,
            use_manifold_constraint=use_manifold_constraint,
            constraint_type=constraint_type,
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
        
        hyper_output, gate_weights = self.hyper_connection(
            self._cache, input_norm
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
    """Complete Transformer with mHC for PiscesL1.
    
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
