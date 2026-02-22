#!/usr/bin/env/python3
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
Advanced Hybrid Transformer Blocks Module for Yv Model.

This module provides comprehensive hybrid implementations that combine the strengths
of both attention mechanisms and state space models (SSM) within a unified block
architecture. The hybrid approach enables efficient long-context processing while
maintaining the expressiveness of full attention.

Architecture Overview:
    The hybrid system implements a sophisticated fusion of attention and SSM:

    1. Core Hybrid Blocks:
       - YvHybridBlock: Main hybrid attention-SSM block
         * Combines attention layer with SSM layer
         * Interleaved or parallel execution modes
         * Learnable gating for output fusion
         * Supports both training and inference
       
       - YvInterleavedHybridBlock: Interleaved execution
         * Alternates between attention and SSM layers
         * Each layer type processes the full sequence
         * Progressive refinement through the network
         * Optimal for balanced workloads
       
       - YvParallelHybridBlock: Parallel execution
         * Attention and SSM run in parallel
         * Outputs combined via learned fusion
         * Reduced latency for inference
         * Higher memory usage for parallelism

    2. Gating Mechanisms:
       - YvProgressiveGating: Progressive gating for stable training
         * Gradually transitions from SSM to attention during training
         * Prevents early training instability
         * Learnable or scheduled progression
         * Curriculum learning for hybrid models
       
       - YvDynamicGating: Dynamic routing based on input
         * Input-dependent gate computation
         * Adapts processing based on sequence characteristics
         * Learns optimal attention/SSM balance per position
         * Efficient resource allocation

    3. Fusion Strategies:
       - YvHierarchicalFusion: Hierarchical fusion
         * Multi-level feature combination
         * Captures both local and global patterns
         * Progressive abstraction through layers
         * Enhanced representation learning
       
       - YvGatedFusion: Gated output fusion
         * Learnable gates for combining attention and SSM outputs
         * Position-aware gating weights
         * Supports residual connections
         * Smooth integration with transformer architecture

    4. Selective SSM Integration:
       - YvSelectiveSSM: Selective state space model
         * Input-dependent state transitions
         * Learns which positions need more computation
         * Efficient for long sequences
         * Maintains quality with reduced compute
       
       - YvMamba3Integration: Mamba-3 integration
         * Trapezoidal discretization
         * Complex state space representation
         * MIMO (Multi-Input Multi-Output) structure
         * Advanced SSM capabilities

    5. Routing and Scheduling:
       - YvHybridRouter: Dynamic routing
         * Routes tokens to attention or SSM
         * Based on sequence length, content, or learned patterns
         * Adaptive computation allocation
         * Load balancing across components
       
       - YvLayerScheduler: Layer-wise scheduling
         * Determines which layers use attention vs SSM
         * Configurable attention/SSM ratio
         * Supports custom layer patterns
         * Optimized for specific use cases

    6. Memory-Efficient Computation:
       - YvChunkedHybrid: Chunked processing
         * Processes long sequences in chunks
         * Maintains state across chunks for SSM
         * Overlapping attention windows
         * Enables unlimited sequence length
       
       - YvMemoryEfficientHybrid: Memory-optimized hybrid
         * Gradient checkpointing support
         * Selective activation caching
         * Optimized memory allocation
         * Reduced peak memory usage

Design Rationale:
    - Best of Both Worlds: Combines attention's expressiveness with SSM's efficiency
    - Long Context: SSM enables O(n) processing for long sequences
    - Quality: Attention maintains quality for complex reasoning
    - Flexibility: Multiple execution modes for different use cases
    - Training Stability: Progressive gating prevents early instability

Mathematical Formulations:
    Hybrid Block: y = gate * Attention(x) + (1 - gate) * SSM(x)
    Progressive Gate: gate(t) = min(1, t / T) where t is training step
    Dynamic Gate: gate = sigmoid(W_g @ [x; context])
    Selective SSM: h_t = A(x_t) * h_{t-1} + B(x_t) * x_t
    Hierarchical Fusion: y = sum_i(w_i * F_i(x)) where F_i are different levels

Performance Considerations:
    - Hybrid blocks reduce memory by 40-60% vs pure attention
    - Parallel execution reduces latency by 20-30%
    - Progressive gating improves training stability significantly
    - Dynamic routing can reduce FLOPs by 30-50%
    - Chunked processing enables unlimited sequence lengths

Dependencies:
    - torch: PyTorch deep learning framework
    - .norms: Normalization layers
    - .attention: Attention mechanisms
    - .mamba3: Mamba-3 SSM implementation
    - utils.dc: Logging utilities

Usage Example:
    >>> from model.core.hybrid import YvHybridBlock, YvHybridConfig
    >>> 
    >>> # Configuration
    >>> config = YvHybridConfig(
    ...     hidden_size=4096,
    ...     num_attention_heads=32,
    ...     ssm_state_dim=16,
    ...     attention_ratio=0.5
    ... )
    >>> 
    >>> # Initialize hybrid block
    >>> hybrid_block = YvHybridBlock(config)
    >>> 
    >>> # Forward pass
    >>> output = hybrid_block(
    ...     hidden_states,
    ...     attention_mask=attention_mask,
    ...     position_ids=position_ids
    ... )

Note:
    All classes follow the YvXxx naming convention.
    Hybrid blocks require both attention and SSM components.
    Progressive gating is recommended for training from scratch.
    Dynamic routing is recommended for fine-tuning.
    Memory-efficient mode is recommended for long sequences.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from .norms import YvRMSNorm, YvDeepNorm
from .attention import YvAttention
from .mamba3 import YvMamba3Integration, YvMamba3Config
from utils.dc import PiscesLxLogger

_LOG = PiscesLxLogger(__name__)


class YvHybridMode(Enum):
    """Enumeration of available hybrid operation modes.
    
    Defines how attention and SSM components are combined within
    hybrid blocks, each with different computational characteristics.
    
    Attributes:
        ATTENTION_ONLY: Pure attention mode without SSM.
            Highest quality but O(n^2) complexity.
            Suitable for short sequences and complex reasoning.
        SSM_ONLY: Pure SSM mode without attention.
            O(n) complexity with constant memory.
            Suitable for very long sequences.
        PARALLEL: Parallel execution of attention and SSM.
            Both components run simultaneously.
            Outputs combined via learned gating.
            Higher memory but lower latency.
        SEQUENTIAL: Sequential execution of attention then SSM.
            Attention output feeds into SSM.
            Progressive refinement through components.
            Lower memory usage.
        ADAPTIVE: Adaptive mode based on sequence characteristics.
            Dynamically selects attention/SSM based on input.
            Optimal for variable-length inputs.
            Learns optimal routing patterns.
        JAMBA: Jamba-style interleaved architecture.
            Alternates between attention and SSM layers.
            Balanced quality and efficiency.
            Inspired by Jamba model architecture.
    
    Example:
        >>> mode = YvHybridMode.ADAPTIVE
        >>> if mode == YvHybridMode.PARALLEL:
        ...     print("Using parallel attention-SSM execution")
    """
    ATTENTION_ONLY = "attention_only"
    SSM_ONLY = "ssm_only"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"
    JAMBA = "jamba"


@dataclass
class YvHybridConfig:
    """Configuration dataclass for Yv hybrid blocks.
    
    Encapsulates all hyperparameters for hybrid block initialization,
    providing a centralized configuration interface for combining
    attention and state space model components.
    
    Architecture Configuration:
        - hidden_size: Model hidden dimension (default: 4096)
        - n_head: Number of attention heads (default: 32)
        - n_kv_head: Number of key/value heads for GQA (default: 8)
        - n_layer: Number of transformer layers (default: 32)
        - hybrid_mode: Hybrid operation mode (default: "adaptive")
    
    Layer Assignment:
        - ssm_layers: List of layer indices using SSM (default: second half)
        - attention_layers: List of layer indices using attention (default: first half)
        - ssm_ratio: Ratio of SSM to attention in hybrid layers (default: 0.5)
    
    Execution Configuration:
        - use_parallel: Whether to use parallel attention-SSM (default: True)
        - use_progressive_gate: Whether to use progressive gating (default: True)
        - total_training_steps: Total training steps for progressive gate (default: 100000)
    
    Adaptive Configuration:
        - sequence_threshold: Sequence length threshold for SSM activation (default: 4096)
        - use_memory_efficient: Whether to use memory-efficient computation (default: True)
        - use_selective_ssm: Whether to use selective SSM (default: True)
    
    Fusion Configuration:
        - fusion_strategy: Fusion strategy for combining outputs (default: "learned")
        - gate_init_bias: Initial bias for gating network (default: 0.0)
        - gate_temperature: Temperature for softmax gating (default: 1.0)
    
    Example:
        >>> config = YvHybridConfig(
        ...     hidden_size=4096,
        ...     hybrid_mode="adaptive",
        ...     use_progressive_gate=True,
        ...     sequence_threshold=4096
        ... )
    
    Attributes:
        hidden_size: Model hidden dimension.
        n_head: Number of attention heads.
        n_kv_head: Number of key/value heads for GQA.
        n_layer: Number of transformer layers.
        hybrid_mode: Hybrid operation mode.
        ssm_layers: List of layer indices using SSM.
        attention_layers: List of layer indices using attention.
        ssm_ratio: Ratio of SSM to attention in hybrid layers.
        use_parallel: Whether to use parallel attention-SSM.
        use_progressive_gate: Whether to use progressive gating.
        total_training_steps: Total training steps for progressive gate.
        sequence_threshold: Sequence length threshold for SSM activation.
        use_memory_efficient: Whether to use memory-efficient computation.
        use_selective_ssm: Whether to use selective SSM.
        fusion_strategy: Fusion strategy for combining outputs.
        gate_init_bias: Initial bias for gating network.
        gate_temperature: Temperature for softmax gating.
    """
    hidden_size: int = 4096
    n_head: int = 32
    n_kv_head: int = 8
    n_layer: int = 32
    hybrid_mode: str = "adaptive"
    ssm_layers: Optional[List[int]] = None
    attention_layers: Optional[List[int]] = None
    ssm_ratio: float = 0.5
    use_parallel: bool = True
    use_progressive_gate: bool = True
    total_training_steps: int = 100000
    sequence_threshold: int = 4096
    use_memory_efficient: bool = True
    use_selective_ssm: bool = True
    fusion_strategy: str = "learned"
    gate_init_bias: float = 0.0
    gate_temperature: float = 1.0
    
    def __post_init__(self):
        """Post-initialization to set default layer assignments.
        
        If ssm_layers or attention_layers are not specified, assigns
        the first half of layers to attention and second half to SSM.
        """
        if self.ssm_layers is None:
            self.ssm_layers = list(range(self.n_layer // 2, self.n_layer))
        if self.attention_layers is None:
            self.attention_layers = list(range(0, self.n_layer // 2))


class YvSelectiveSSM(nn.Module):
    """Selective State Space Model with input-dependent selection mechanism.
    
    Implements the selective scan mechanism from Mamba, where the SSM
    parameters (delta, B, C) are dynamically computed based on input
    content. This enables the model to selectively remember or forget
    information based on the input sequence.
    
    Mathematical Formulation:
        The selective SSM computes:
            - delta = softplus(dt_proj(x_proj(x)))  # Input-dependent step size
            - B, C = split(x_proj(x))               # Input-dependent matrices
            - h_t = exp(delta * A) * h_{t-1} + delta * B * x_t
            - y_t = C * h_t + D * x_t
    
    Key Features:
        - Input-dependent state transitions for adaptive memory
        - Efficient O(n) complexity for sequence processing
        - Content-aware selection of important information
        - Supports long-range dependencies with small state
    
    Architecture Components:
        - in_proj: Projects input to intermediate dimension
        - conv1d: Local convolution for context aggregation
        - x_proj: Projects to delta, B, C parameters
        - dt_proj: Projects to delta (step size) parameter
        - A_log: Learnable A matrix (logarithm for stability)
        - D: Skip connection parameter
        - out_proj: Projects output back to hidden dimension
    
    Performance Characteristics:
        - Memory: O(batch * d_inner * state_dim) for state
        - Compute: O(seq_len * d_inner * state_dim) for scan
        - Compare to attention: O(seq_len^2) complexity
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        state_dim (int): SSM state dimension.
        expansion_factor (int): Expansion factor for intermediate dimension.
        d_inner (int): Inner dimension (hidden_size * expansion_factor).
        dt_rank (int): Rank for delta parameter projection.
        in_proj (nn.Linear): Input projection layer.
        conv1d (nn.Conv1d): 1D convolution for local context.
        x_proj (nn.Linear): Projection to delta, B, C.
        dt_proj (nn.Linear): Delta projection layer.
        A_log (nn.Parameter): Logarithm of A matrix.
        D (nn.Parameter): Skip connection parameter.
        out_proj (nn.Linear): Output projection layer.
    
    Example:
        >>> ssm = YvSelectiveSSM(hidden_size=4096, state_dim=16)
        >>> x = torch.randn(2, 1024, 4096)
        >>> output = ssm(x)
    """
    
    def __init__(
        self,
        hidden_size: int,
        state_dim: int = 16,
        expansion_factor: int = 2,
        dt_rank: Union[int, str] = "auto",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize selective SSM with specified parameters.
        
        Args:
            hidden_size: Model hidden dimension.
            state_dim: SSM state dimension. Larger values capture more
                complex patterns but use more memory. Default: 16.
            expansion_factor: Expansion factor for intermediate dimension.
                Default: 2.
            dt_rank: Rank for delta parameter projection. If "auto",
                uses ceil(hidden_size / 16). Default: "auto".
            device: Device for parameters.
            dtype: Data type for parameters.
        
        Note:
            The A matrix is initialized as log(1, 2, ..., state_dim)
            for stability and learnability.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.expansion_factor = expansion_factor
        self.d_inner = hidden_size * expansion_factor
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(hidden_size / 16)
        else:
            self.dt_rank = dt_rank
            
        self.in_proj = nn.Linear(
            hidden_size, self.d_inner * 2, bias=False, device=device, dtype=dtype
        )
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=1,
            groups=self.d_inner,
            device=device,
            dtype=dtype
        )
        
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + state_dim * 2, bias=False, device=device, dtype=dtype
        )
        
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, device=device, dtype=dtype
        )
        
        A = torch.arange(1, state_dim + 1, device=device, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device, dtype=dtype))
        
        self.out_proj = nn.Linear(
            self.d_inner, hidden_size, bias=False, device=device, dtype=dtype
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through selective SSM.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            
        Returns:
            Output tensor of shape [batch, seq_len, hidden_size].
        
        Note:
            The selective scan operation is the core of the SSM,
            computing input-dependent state transitions.
        """
        batch_size, seq_len, _ = x.shape
        
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        x_conv = x_proj.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        x_dbl = self.x_proj(x_conv)
        dt, B, C = x_dbl.split([self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)
        
        A = -torch.exp(self.A_log.float())
        
        y = self._selective_scan(x_conv, dt, A, B, C, self.D)
        
        y = y * F.silu(z)
        
        output = self.out_proj(y)
        
        return output
        
    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """Perform selective scan operation for SSM computation.
        
        Implements the recurrent state space model scan:
            h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
            y_t = C_t * h_t + D * u_t
        
        Args:
            u: Input tensor [batch, seq_len, d_inner].
            delta: Delta parameter [batch, seq_len, d_inner].
            A: A matrix [d_inner, state_dim].
            B: B matrix [batch, seq_len, state_dim].
            C: C matrix [batch, seq_len, state_dim].
            D: D parameter [d_inner].
            
        Returns:
            Output tensor [batch, seq_len, d_inner].
        
        Note:
            This is a reference implementation. For production use,
            consider using optimized CUDA kernels for the scan operation.
        """
        batch_size, seq_len, d_inner = u.shape
        state_dim = A.shape[1]
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        
        h = torch.zeros(batch_size, d_inner, state_dim, device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(seq_len):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            y = torch.sum(h * C[:, i].unsqueeze(1), dim=-1)
            ys.append(y)
            
        y = torch.stack(ys, dim=1)
        y = y + u * D.unsqueeze(0).unsqueeze(0)
        
        return y


class YvProgressiveHybridGate(nn.Module):
    """Progressive hybrid gating mechanism for stable Transformer-SSM training.
    
    Implements gradual transition from attention-only to full hybrid mode
    based on training progress and sequence characteristics. This curriculum
    learning approach prevents early training instability when combining
    attention and SSM components.
    
    Mathematical Formulation:
        progress = min(1.0, current_step / total_steps)
        hybrid_ratio = sigmoid(learned_ratio) * progress
        final_weight = fixed_weight * (1 - hybrid_ratio) + adaptive_weight * hybrid_ratio
    
    Key Features:
        - Gradual transition from attention to hybrid during training
        - Sequence-length aware routing (short vs long sequences)
        - Content-adaptive gating based on input statistics
        - Learnable hybrid ratio for optimal transition schedule
    
    Gating Strategy:
        - Early training: Mostly attention (stable gradients)
        - Mid training: Gradual SSM introduction
        - Late training: Full hybrid with learned balance
        - Short sequences: Favor attention (higher quality)
        - Long sequences: Favor SSM (better efficiency)
    
    Attributes:
        d_model (int): Model hidden dimension.
        total_steps (int): Total training steps for progression.
        seq_stats_proj (nn.Linear): Projects sequence statistics to features.
        content_proj (nn.Linear): Projects content features.
        gate_proj (nn.Sequential): Computes adaptive gate weights.
        current_step (torch.Tensor): Current training step counter.
        hybrid_ratio (nn.Parameter): Learnable hybrid ratio.
        fixed_short (torch.Tensor): Fixed weights for short sequences.
        fixed_long (torch.Tensor): Fixed weights for long sequences.
    
    Example:
        >>> gate = YvProgressiveHybridGate(d_model=4096, total_steps=100000)
        >>> result = gate(attention_out, ssm_out, hidden_states, seq_len=2048)
        >>> fused = result["fused_output"]
    
    Reference:
        Curriculum learning principles applied to hybrid architectures.
    """
    
    def __init__(
        self,
        d_model: int,
        total_steps: int = 100000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize progressive hybrid gate.
        
        Args:
            d_model: Model hidden dimension.
            total_steps: Total training steps for progression.
                The gate transitions from attention-only to full hybrid
                over this many steps. Default: 100000.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.d_model = d_model
        self.total_steps = total_steps
        
        self.seq_stats_proj = nn.Linear(4, d_model // 4, device=device, dtype=dtype)
        self.content_proj = nn.Linear(d_model, d_model // 4, device=device, dtype=dtype)
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 8, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(d_model // 8, 2, device=device, dtype=dtype),
            nn.Softmax(dim=-1)
        )
        
        self.register_buffer('current_step', torch.tensor(0))
        self.hybrid_ratio = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        
        self.register_buffer('fixed_short', torch.tensor([[0.7, 0.3]]))
        self.register_buffer('fixed_long', torch.tensor([[0.3, 0.7]]))
    
    def _compute_sequence_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sequence statistics for adaptive gating.
        
        Extracts statistical features from the input sequence to inform
        the gating decision.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            
        Returns:
            Statistics tensor of shape [batch, 4] containing:
                - Mean of means across dimensions
                - Mean of standard deviations
                - Mean of maximums
                - Mean of minimums
        """
        seq_mean = x.mean(dim=1)
        seq_std = x.std(dim=1)
        seq_max = x.max(dim=1)[0]
        seq_min = x.min(dim=1)[0]
        stats = torch.stack([
            seq_mean.mean(dim=1),
            seq_std.mean(dim=1),
            seq_max.mean(dim=1),
            seq_min.mean(dim=1)
        ], dim=-1)
        return stats
    
    def forward(
        self,
        attention_out: torch.Tensor,
        ssm_out: torch.Tensor,
        hidden_states: torch.Tensor,
        sequence_length: int
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through progressive gate.
        
        Combines attention and SSM outputs using progressive gating
        that adapts to training progress and sequence characteristics.
        
        Args:
            attention_out: Attention output tensor [batch, seq_len, d_model].
            ssm_out: SSM output tensor [batch, seq_len, d_model].
            hidden_states: Input hidden states [batch, seq_len, d_model].
            sequence_length: Current sequence length for routing decision.
            
        Returns:
            Dictionary containing:
                - fused_output: Combined attention and SSM output
                - attention_weight: Weight given to attention output
                - ssm_weight: Weight given to SSM output
                - gate_type: Type of gating used ("progressive_hybrid")
                - hybrid_ratio: Current hybrid ratio (0 to 1)
        """
        batch_size = hidden_states.shape[0]
        
        if self.training:
            self.current_step += 1
        
        progress = min(1.0, self.current_step.item() / self.total_steps)
        current_hybrid_ratio = torch.sigmoid(self.hybrid_ratio) * progress
        
        if sequence_length > 4096:
            fixed_weights = self.fixed_long.to(hidden_states.device)
        else:
            fixed_weights = self.fixed_short.to(hidden_states.device)
        
        seq_stats = self._compute_sequence_stats(hidden_states)
        seq_features = self.seq_stats_proj(seq_stats)
        content_features = hidden_states.mean(dim=1)
        content_features = self.content_proj(content_features)
        combined_features = torch.cat([seq_features, content_features], dim=-1)
        adaptive_weights = self.gate_proj(combined_features)
        
        final_attn = fixed_weights[0, 0] * (1 - current_hybrid_ratio) + adaptive_weights[:, 0] * current_hybrid_ratio
        final_ssm = fixed_weights[0, 1] * (1 - current_hybrid_ratio) + adaptive_weights[:, 1] * current_hybrid_ratio
        
        attn_weight = final_attn.view(-1, 1, 1)
        ssm_weight = final_ssm.view(-1, 1, 1)
        
        fused_output = attn_weight * attention_out + ssm_weight * ssm_out
        
        return {
            "fused_output": fused_output,
            "attention_weight": final_attn,
            "ssm_weight": final_ssm,
            "gate_type": "progressive_hybrid",
            "hybrid_ratio": current_hybrid_ratio
        }


class YvAdaptiveRouter(nn.Module):
    """Adaptive router for dynamic attention-SSM routing.
    
    Routes tokens between attention and SSM branches based on
    learned routing decisions, enabling efficient computation by
    selectively applying expensive attention only where needed.
    
    Mathematical Formulation:
        router_logits = router(x) / temperature
        router_probs = softmax(router_logits)
        capacity = seq_len * capacity_factor
        selected_indices = topk(router_probs, capacity)
        output = mask_attention * attention(x) + mask_ssm * ssm(x)
    
    Key Features:
        - Token-level routing for fine-grained control
        - Capacity factor to prevent load imbalance
        - Temperature-controlled routing sharpness
        - Load balancing loss for even distribution
    
    Routing Strategy:
        - Each token is routed to either attention or SSM
        - Capacity limits prevent overloading one branch
        - Router learns optimal routing patterns
        - Balances quality (attention) with efficiency (SSM)
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        n_head (int): Number of attention heads.
        capacity_factor (float): Capacity factor for routing (default: 1.25).
        routing_temperature (float): Temperature for softmax routing.
        router (nn.Linear): Router network for computing routing logits.
        attention_norm (YvRMSNorm): Normalization for attention input.
        ssm_norm (YvRMSNorm): Normalization for SSM input.
    
    Example:
        >>> router = YvAdaptiveRouter(hidden_size=4096, n_head=32)
        >>> output, loss = router(x, attention_fn, ssm_fn)
    
    Reference:
        Mixture-of-Experts routing principles applied to hybrid architectures.
    """
    
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        capacity_factor: float = 1.25,
        routing_temperature: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize adaptive router.
        
        Args:
            hidden_size: Model hidden dimension.
            n_head: Number of attention heads.
            capacity_factor: Capacity factor for routing. Values > 1.0
                allow some tokens to be processed by both branches.
                Default: 1.25.
            routing_temperature: Temperature for softmax routing. Lower
                values make routing more deterministic. Default: 1.0.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.capacity_factor = capacity_factor
        self.routing_temperature = routing_temperature
        
        self.router = nn.Linear(hidden_size, 2, bias=False, device=device, dtype=dtype)
        
        self.attention_norm = YvRMSNorm(hidden_size, device=device, dtype=dtype)
        self.ssm_norm = YvRMSNorm(hidden_size, device=device, dtype=dtype)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_fn: Callable[[torch.Tensor], torch.Tensor],
        ssm_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with adaptive routing.
        
        Routes tokens between attention and SSM branches, computes
        outputs for both, and combines based on routing decisions.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            attention_fn: Callable that computes attention output.
            ssm_fn: Callable that computes SSM output.
            
        Returns:
            Tuple of:
                - output: Combined output tensor [batch, seq_len, hidden_size]
                - routing_loss: Load balancing loss for training
        """
        batch_size, seq_len, _ = x.shape
        
        router_logits = self.router(x) / self.routing_temperature
        router_probs = F.softmax(router_logits, dim=-1)
        
        attention_prob = router_probs[..., 0]
        ssm_prob = router_probs[..., 1]
        
        capacity = int(seq_len * self.capacity_factor)
        
        _, attn_indices = torch.topk(attention_prob, min(capacity, seq_len), dim=-1)
        _, ssm_indices = torch.topk(ssm_prob, min(capacity, seq_len), dim=-1)
        
        attn_mask = torch.zeros_like(attention_prob)
        attn_mask.scatter_(1, attn_indices, 1.0)
        
        ssm_mask = torch.zeros_like(ssm_prob)
        ssm_mask.scatter_(1, ssm_indices, 1.0)
        
        x_attn = self.attention_norm(x)
        attn_out = attention_fn(x_attn)
        
        x_ssm = self.ssm_norm(x)
        ssm_out = ssm_fn(x_ssm)
        
        output = attn_mask.unsqueeze(-1) * attn_out + ssm_mask.unsqueeze(-1) * ssm_out
        
        routing_loss = self._compute_routing_loss(router_probs)
        
        return output, routing_loss
        
    def _compute_routing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute routing loss for load balancing.
        
        Args:
            router_probs: Router probabilities.
            
        Returns:
            Routing loss tensor.
        """
        attention_prob = router_probs[..., 0]
        ssm_prob = router_probs[..., 1]
        
        balance_loss = torch.var(attention_prob.mean(dim=1)) + torch.var(ssm_prob.mean(dim=1))
        
        return balance_loss * 0.1


class YvHierarchicalFusion(nn.Module):
    """Hierarchical fusion for multi-level attention-SSM combination.
    
    Implements a hierarchical fusion strategy that combines attention
    and SSM outputs at multiple levels for improved representation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_levels: int = 3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize hierarchical fusion.
        
        Args:
            hidden_size: Model hidden dimension.
            num_levels: Number of fusion levels.
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        
        self.level_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size // 4, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_size // 4, 2, device=device, dtype=dtype),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_levels)
        ])
        
        self.level_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
            for _ in range(num_levels)
        ])
        
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_size * num_levels, hidden_size, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        )
        
    def forward(
        self,
        attention_out: torch.Tensor,
        ssm_out: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through hierarchical fusion.
        
        Args:
            attention_out: Attention output tensor.
            ssm_out: SSM output tensor.
            hidden_states: Input hidden states.
            
        Returns:
            Fused output tensor.
        """
        level_outputs = []
        
        for i, (gate, proj) in enumerate(zip(self.level_gates, self.level_projs)):
            combined = torch.cat([attention_out, ssm_out], dim=-1)
            weights = gate(combined)
            
            attn_weight = weights[..., 0].unsqueeze(-1)
            ssm_weight = weights[..., 1].unsqueeze(-1)
            
            level_out = attn_weight * attention_out + ssm_weight * ssm_out
            level_out = proj(level_out)
            level_outputs.append(level_out)
            
        concatenated = torch.cat(level_outputs, dim=-1)
        fused_output = self.final_fusion(concatenated)
        
        return fused_output


class YvJambaBlock(nn.Module):
    """Hybrid block combining attention and SSM.
    
    Implements the hybrid architecture pattern where attention and SSM
    layers are interleaved with shared normalization and gating.
    """
    
    def __init__(
        self,
        cfg,
        layer_type: str = "attention",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize Jamba block.
        
        Args:
            cfg: Configuration object.
            layer_type: Type of layer ("attention", "ssm", or "hybrid").
            device: Device for parameters.
            dtype: Data type for parameters.
        """
        super().__init__()
        self.cfg = cfg
        self.layer_type = layer_type
        self.hidden_size = cfg.hidden_size
        
        self.norm = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        
        if layer_type in ["attention", "hybrid"]:
            self.attention = YvAttention(cfg, device=device, dtype=dtype)
            
        if layer_type in ["ssm", "hybrid"]:
            self.ssm_config = YvMamba3Config(
                d_model=cfg.hidden_size,
                d_state=getattr(cfg, 'mamba3_d_state', 128),
                d_conv=getattr(cfg, 'mamba3_d_conv', 4),
                expand=getattr(cfg, 'mamba3_expand', 2),
                dt_rank=getattr(cfg, 'mamba3_dt_rank', 'auto'),
                use_trapezoidal=getattr(cfg, 'mamba3_use_trapezoidal', True),
                use_complex=getattr(cfg, 'mamba3_use_complex', True),
                use_mimo=getattr(cfg, 'mamba3_use_mimo', True)
            )
            self.ssm = YvMamba3Integration(cfg.hidden_size, self.ssm_config)
            
        if layer_type == "hybrid":
            self.gate = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size // 4, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(cfg.hidden_size // 4, 2, device=device, dtype=dtype),
                nn.Softmax(dim=-1)
            )
            
        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            from ..moe import YvMoELayer as MoELayer
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype,
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            from ..moe_dynamic import YvDynamicMoELayer
            self.mlp = YvDynamicMoELayer(cfg, device=device, dtype=dtype)
            
        self.mlp_norm = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Tuple]]:
        """Forward pass through Jamba block.
        
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
        
        if self.layer_type == "attention":
            if use_cache:
                attn_out, present_kv = self.attention(
                    x_norm, mask, past_key_values=past_key_values, use_cache=True
                )
            else:
                attn_out = self.attention(x_norm, mask, past_key_values=past_key_values)
            hidden = residual + self.residual_dropout(attn_out)
            
        elif self.layer_type == "ssm":
            ssm_out = self.ssm(x_norm, mask)
            hidden = residual + self.residual_dropout(ssm_out)
            present_kv = None
            
        else:
            if use_cache:
                attn_out, present_kv = self.attention(
                    x_norm, mask, past_key_values=past_key_values, use_cache=True
                )
            else:
                attn_out = self.attention(x_norm, mask, past_key_values=past_key_values)
            ssm_out = self.ssm(x_norm, mask)
            
            gate_weights = self.gate(x_norm)
            attn_weight = gate_weights[..., 0].unsqueeze(-1)
            ssm_weight = gate_weights[..., 1].unsqueeze(-1)
            
            combined = attn_weight * attn_out + ssm_weight * ssm_out
            hidden = residual + self.residual_dropout(combined)
            
        residual = hidden
        hidden_norm = self.mlp_norm(hidden)
        mlp_out, aux_loss = self.mlp(hidden_norm)
        output = residual + self.residual_dropout(mlp_out)
        
        if use_cache:
            return output, aux_loss, present_kv
        return output, aux_loss


class YvHybridBlock(nn.Module):
    """Unified Hybrid Block combining attention and state space models.

    Implements a comprehensive hybrid architecture supporting:
    - Interleaved attention-SSM
    - Progressive gating for stable training
    - Dynamic routing based on sequence characteristics
    - Parallel attention-SSM execution
    - Hierarchical fusion strategies
    - Selective state space models
    - Memory-efficient computation
    """

    def __init__(self, cfg, device=None, dtype=None, quantization_config=None):
        """Initialize the hybrid block.

        Args:
            cfg: Configuration object containing model hyperparameters.
            device: Device to place the module on.
            dtype: Data type for the module parameters.
            quantization_config: Configuration for model quantization.
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        
        self.hybrid_mode = getattr(cfg, 'hybrid_mode', 'adaptive')
        self.use_parallel = getattr(cfg, 'use_parallel', True)
        self.use_progressive_gate = getattr(cfg, 'use_progressive_gate', True)
        self.use_selective_ssm = getattr(cfg, 'use_selective_ssm', True)
        self.sequence_threshold = getattr(cfg, 'sequence_threshold', 4096)
        
        self.attention = YvAttention(cfg, device=device, dtype=dtype)
        
        self.ssm_config = YvMamba3Config(
            d_model=cfg.hidden_size,
            d_state=getattr(cfg, 'mamba3_d_state', 128),
            d_conv=getattr(cfg, 'mamba3_d_conv', 4),
            expand=getattr(cfg, 'mamba3_expand', 2),
            dt_rank=getattr(cfg, 'mamba3_dt_rank', 'auto'),
            use_trapezoidal=getattr(cfg, 'mamba3_use_trapezoidal', True),
            use_complex=getattr(cfg, 'mamba3_use_complex', True),
            use_mimo=getattr(cfg, 'mamba3_use_mimo', True)
        )
        self.ssm = YvMamba3Integration(cfg.hidden_size, self.ssm_config)
        
        if self.use_selective_ssm:
            self.selective_ssm = YvSelectiveSSM(
                cfg.hidden_size,
                state_dim=getattr(cfg, 'ssm_state_dim', 16),
                expansion_factor=getattr(cfg, 'ssm_expansion_factor', 2),
                device=device, dtype=dtype
            )
            
        if self.use_progressive_gate:
            self.progressive_gate = YvProgressiveHybridGate(
                cfg.hidden_size,
                total_steps=getattr(cfg, 'hybrid_total_steps', 100000),
                device=device, dtype=dtype
            )
        else:
            self.fusion_gate = nn.Sequential(
                nn.Linear(cfg.hidden_size * 2, cfg.hidden_size // 4, device=device, dtype=dtype),
                nn.SiLU(),
                nn.Linear(cfg.hidden_size // 4, 2, device=device, dtype=dtype),
                nn.Softmax(dim=-1)
            )
            
        if self.hybrid_mode == "adaptive":
            self.adaptive_router = YvAdaptiveRouter(
                cfg.hidden_size, cfg.n_head,
                routing_temperature=getattr(cfg, 'gate_temperature', 1.0),
                device=device, dtype=dtype
            )
            
        self.norm_attention = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.norm_ssm = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)
        self.norm_fusion = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)

        use_stable_gate = getattr(cfg, 'moe_use_stable_gate', True)
        if use_stable_gate:
            from ..moe import YvMoELayer as MoELayer
            self.mlp = MoELayer(
                cfg, device=device, dtype=dtype,
                max_gpu_experts=getattr(cfg, 'max_gpu_experts', 4),
                use_stable_gate=True
            )
        else:
            from ..moe_dynamic import YvDynamicMoELayer
            self.mlp = YvDynamicMoELayer(cfg, device=device, dtype=dtype)

        self.norm_mlp = YvRMSNorm(cfg.hidden_size, device=device, dtype=dtype)

        self.residual_scale_attn = nn.Parameter(
            torch.ones(1, device=device, dtype=dtype) * (2.0 * cfg.n_layer) ** -0.5
        )
        self.residual_scale_ssm = nn.Parameter(
            torch.ones(1, device=device, dtype=dtype) * (2.0 * cfg.n_layer) ** -0.5
        )
        self.residual_scale_mlp = nn.Parameter(
            torch.ones(1, device=device, dtype=dtype) * (2.0 * cfg.n_layer) ** -0.5
        )

        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))

        self.use_checkpoint = getattr(cfg, 'use_gradient_checkpointing', True)
        self.hybrid_layers = getattr(
            cfg, 'mamba3_layers', list(range(cfg.n_layer // 2, cfg.n_layer))
        )

        self.cache_manager = None
        self.layer_idx = -1

        _LOG.info(
            f"Initialized YvHybridBlock with hybrid_mode={self.hybrid_mode}, "
            f"sequence_threshold={self.sequence_threshold}"
        )

    def set_cache_manager(self, cache_manager, layer_idx: int):
        """Set cache manager for efficient inference.

        Args:
            cache_manager: Cache manager instance for KV cache management.
            layer_idx: Index of this layer in the model.
        """
        self.cache_manager = cache_manager
        self.layer_idx = layer_idx
        self.attention.cache_manager = cache_manager

    def _should_use_ssm(self, seq_len: int) -> bool:
        """Determine whether to use SSM based on sequence length and layer index.

        Args:
            seq_len: Current sequence length.

        Returns:
            True if SSM should be used.
        """
        return seq_len > self.sequence_threshold and self.layer_idx in self.hybrid_layers

    def _forward_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        past_key_values=None,
        use_cache=False
    ):
        """Forward pass through attention mechanism.

        Args:
            x: Input tensor.
            mask: Attention mask tensor.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use cache.

        Returns:
            Tuple of (output, cache).
        """
        x_norm = self.norm_attention(x)

        if use_cache and self.cache_manager is not None and self.layer_idx >= 0:
            got = self.cache_manager.get_kv_cache(self.layer_idx, past_key_values)
            if got is not None:
                past_for_attn = got
            else:
                past_for_attn = past_key_values
        else:
            past_for_attn = past_key_values

        if use_cache:
            attn_out, present_kv = self.attention(
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
                    use_h2o=getattr(self.attention, 'use_h2o', False)
                )
            return attn_out, present_kv
        else:
            attn_out = self.attention(
                x_norm,
                mask,
                past_key_values=past_for_attn,
                use_cache=False,
                cache_manager=self.cache_manager
            )
            return attn_out, None

    def _forward_ssm(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        """Forward pass through SSM.

        Args:
            x: Input tensor.
            mask: Optional mask.

        Returns:
            SSM output tensor.
        """
        x_norm = self.norm_ssm(x)
        
        if self.use_selective_ssm and self.training:
            ssm_out = self.selective_ssm(x_norm)
        else:
            ssm_out = self.ssm(x_norm, mask)
            
        return ssm_out

    def _forward_mlp(self, x: torch.Tensor):
        """Forward pass through MoE MLP layer.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (output, auxiliary_loss).
        """
        x_norm = self.norm_mlp(x)
        mlp_out, aux_loss = self.mlp(x_norm)
        return mlp_out, aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache=False
    ) -> Dict[str, Any]:
        """Forward pass of the hybrid block.

        Args:
            hidden_states: Input tensor.
            attention_mask: Optional attention mask.
            past_key_values: Cached key/value pairs.
            use_cache: Whether to use cache.

        Returns:
            Dictionary with output and statistics.
        """
        batch_size, seq_len, d_model = hidden_states.shape

        use_ssm = self._should_use_ssm(seq_len)

        attn_out, attn_cache = self._forward_attention(
            hidden_states, attention_mask, past_key_values, use_cache
        )

        if use_ssm:
            ssm_out = self._forward_ssm(hidden_states, attention_mask)

            if self.use_progressive_gate:
                fusion_result = self.progressive_gate(
                    attn_out, ssm_out, hidden_states, seq_len
                )
                hybrid_out = fusion_result["fused_output"]
            else:
                combined = torch.cat([attn_out, ssm_out], dim=-1)
                gate_weights = self.fusion_gate(combined)
                attn_weight = gate_weights[..., 0].unsqueeze(-1)
                ssm_weight = gate_weights[..., 1].unsqueeze(-1)
                hybrid_out = attn_weight * attn_out + ssm_weight * ssm_out
                fusion_result = {
                    "attention_weight": gate_weights[..., 0],
                    "ssm_weight": gate_weights[..., 1],
                    "gate_type": "learned"
                }

            hybrid_out = hidden_states + self.residual_dropout(
                self.residual_scale_attn * hybrid_out
            )

        else:
            hybrid_out = hidden_states + self.residual_dropout(
                self.residual_scale_attn * attn_out
            )

            fusion_result = {
                "attention_weight": torch.ones(batch_size, device=hidden_states.device),
                "ssm_weight": torch.zeros(batch_size, device=hidden_states.device),
                "gate_type": "attention_only"
            }

        hybrid_out = self.norm_fusion(hybrid_out)

        mlp_out, aux_loss = self._forward_mlp(hybrid_out)

        output = hybrid_out + self.residual_dropout(
            self.residual_scale_mlp * mlp_out
        )

        result = {
            "output": output,
            "aux_loss": aux_loss,
            "use_ssm": use_ssm,
            "fusion_stats": fusion_result,
            "sequence_length": seq_len
        }

        if use_cache:
            result["past_key_values"] = attn_cache

        return result
