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
Advanced Embedding Module for Yv Model.

This module provides comprehensive embedding implementations that form the input
representation layer of the Yv transformer architecture. It handles the
conversion of raw input tokens and multimodal features into dense vector
representations suitable for processing by the transformer layers.

Architecture Overview:
    The embedding system is designed with a hierarchical structure:

    1. Configuration (YvEmbeddingConfig):
       - Centralized configuration for all embedding parameters
       - Vocabulary size, hidden dimensions, sequence length limits
       - Modality-specific settings for multimodal support

    2. Token Embeddings (YvTokenEmbedding):
       - Standard vocabulary-based token embeddings
       - Optional adaptive scaling for training stability
       - Deep normalization for gradient flow in deep models
       - Padding token handling for variable-length sequences

    3. Position Embeddings:
       - YvSinusoidalPositionEmbedding: Fixed sinusoidal encodings
         * Deterministic, no learned parameters
         * Supports extrapolation to longer sequences
         * Memory efficient with precomputed cache
       
       - YvLearnedPositionEmbedding: Trainable position embeddings
         * Learned position representations
         * Linear interpolation for longer sequences
         * More flexible but requires more memory

    4. Modality Embeddings (YvModalityEmbedding):
       - Modality-aware embeddings for multimodal inputs
       - Supports 6 modalities: text, image, audio, video, document, agentic
       - Learnable modality tokens with projection to hidden dimension
       - Gated fusion for modality integration

    5. Unified Interface (YvUnifiedEmbedding):
       - Combines token, position, and modality embeddings
       - Single entry point for all embedding operations
       - Supports both learned and sinusoidal position encodings
       - Handles embedding scaling and dropout

    6. Multimodal Projection (YvMultimodalEmbeddingProjector):
       - Projects external encoder outputs to model dimension
       - Vision, audio, document, video, and agentic projections
       - Cross-modal alignment with learned transformations

Design Rationale:
    - Modularity: Each embedding type is independently configurable
    - Multimodal Ready: Native support for multiple input modalities
    - Long Context: Position embeddings support 10M+ token sequences
    - Training Stability: Adaptive scaling and deep normalization
    - Memory Efficiency: Optional caching and efficient implementations

Mathematical Formulations:
    Token Embedding: e = Embedding(token_id) * scale + bias
    Sinusoidal PE: PE(pos, 2i) = sin(pos / 10000^(2i/d))
                   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    Modality Embedding: h' = h + gate(h, m) * proj(m)
    Unified: output = dropout(token_emb + pos_emb + modality_emb)

Performance Considerations:
    - Sinusoidal embeddings are faster (no lookup) but less flexible
    - Learned embeddings can capture position-specific patterns
    - Modality embeddings add minimal overhead for multimodal support
    - Adaptive scaling helps with training convergence

Dependencies:
    - torch: PyTorch deep learning framework
    - torch.nn: Neural network modules
    - dataclasses: Configuration data structures

Usage Example:
    >>> from model.core.embedding import YvUnifiedEmbedding, YvEmbeddingConfig
    >>> 
    >>> # Configuration
    >>> config = YvEmbeddingConfig(
    ...     vocab_size=151646,
    ...     hidden_size=4096,
    ...     max_position_embeddings=10485760,
    ...     num_modalities=6
    ... )
    >>> 
    >>> # Initialize embeddings
    >>> embeddings = YvUnifiedEmbedding(config)
    >>> 
    >>> # Text input
    >>> output = embeddings(input_ids, modality="text")
    >>> 
    >>> # Multimodal input
    >>> output = embeddings(input_ids, modality="image")

Note:
    All classes follow the YvXxx naming convention.
    The unified embedding is the primary interface used by YvModel.
    Modality embeddings are automatically applied based on input type.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass


@dataclass
class YvEmbeddingConfig:
    """Configuration for embedding layers.
    
    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of the embedding vectors.
        max_position_embeddings: Maximum sequence length.
        dropout_rate: Dropout probability for embeddings.
        layer_norm_eps: Epsilon for layer normalization.
        use_learned_position: Whether to use learned positional embeddings.
        use_adaptive_scaling: Whether to use adaptive embedding scaling.
        use_deep_norm: Whether to use deep normalization for stability.
        modality_embed_dim: Dimension for modality embeddings.
        num_modalities: Number of supported modalities.
    """
    vocab_size: int = 151646
    hidden_size: int = 2048
    max_position_embeddings: int = 10485760
    dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-6
    use_learned_position: bool = False
    use_adaptive_scaling: bool = True
    use_deep_norm: bool = True
    modality_embed_dim: int = 64
    num_modalities: int = 6


class YvAdaptiveEmbeddingScaling(nn.Module):
    """Adaptive scaling for embedding outputs with learned parameters.
    
    Implements learned scaling factors that adapt based on embedding
    statistics for improved training stability. This addresses the
    variance shift that can occur when embeddings are used directly
    without proper scaling.
    
    Mathematical Formulation:
        output = input * scale + bias
        where scale and bias are learned parameters
    
    Key Features:
        - Per-dimension learned scaling
        - Per-dimension learned bias
        - Improves training convergence
        - Stabilizes gradient flow
    
    Training Benefits:
        - Helps prevent embedding explosion/vanishing
        - Allows model to learn optimal embedding scale
        - Compensates for initialization variance
        - Works well with deep normalization
    
    Attributes:
        hidden_size (int): Dimension of the hidden states.
        scale (nn.Parameter): Learned scaling factors, shape [hidden_size].
        bias (nn.Parameter): Learned bias terms, shape [hidden_size].
    
    Example:
        >>> scaling = YvAdaptiveEmbeddingScaling(hidden_size=4096)
        >>> embeddings = torch.randn(2, 1024, 4096)
        >>> scaled = scaling(embeddings)
    
    Note:
        Initialized with scale=1.0 and bias=0.0 to preserve
        the original embedding values at the start of training.
    """
    
    def __init__(self, hidden_size: int, init_scale: float = 1.0):
        """Initialize adaptive scaling with optional initial scale.
        
        Args:
            hidden_size: Dimension of the hidden states. Determines
                the size of the learned scale and bias parameters.
            init_scale: Initial value for the scaling factors.
                Default: 1.0 to preserve original values initially.
        
        Example:
            >>> scaling = YvAdaptiveEmbeddingScaling(
            ...     hidden_size=4096,
            ...     init_scale=1.0
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = nn.Parameter(torch.ones(hidden_size) * init_scale)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive scaling to input embeddings.
        
        Applies learned per-dimension scaling and bias to the input.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
                Typically the output from an embedding layer.
        
        Returns:
            Scaled tensor of the same shape as input.
        
        Note:
            This operation is equivalent to an element-wise affine
            transformation with learned parameters.
        """
        return x * self.scale + self.bias


class YvSinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings with extrapolation support.
    
    Implements the standard sinusoidal position encoding from the original
    Transformer paper, with support for extrapolation to sequences longer
    than the training length. This encoding is deterministic and requires
    no learned parameters.
    
    Mathematical Formulation:
        PE(pos, 2i) = sin(pos / base^(2i/d))
        PE(pos, 2i+1) = cos(pos / base^(2i/d))
        
        where pos is position, i is dimension index, d is hidden_size
    
    Key Features:
        - No learned parameters (fully deterministic)
        - Precomputed cache for efficiency
        - Extrapolation to longer sequences
        - Relative position information preserved
    
    Position Encoding Properties:
        - Bounded output: [-1, 1] range
        - Unique encoding for each position
        - Linear relative position relationships
        - Multi-scale frequency representation
    
    Performance Characteristics:
        - Memory: O(max_seq_len * hidden_size) for cache
        - Compute: O(seq_len * hidden_size) for encoding
        - No parameters: Reduces model size
    
    Attributes:
        hidden_size (int): Dimension of the embeddings.
        max_position_embeddings (int): Maximum cached sequence length.
        base (int): Base frequency for position encoding.
        inv_freq (torch.Tensor): Precomputed inverse frequencies.
        cos_cached (torch.Tensor): Cached cosine values.
        sin_cached (torch.Tensor): Cached sine values.
    
    Example:
        >>> pos_emb = YvSinusoidalPositionEmbedding(
        ...     hidden_size=4096,
        ...     max_position_embeddings=8192
        ... )
        >>> cos, sin = pos_emb(seq_len=1024)
        >>> # Apply to queries and keys for rotary attention
    
    Reference:
        Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 8192,
        base: int = 10000,
        device: Optional[torch.device] = None
    ):
        """Initialize sinusoidal position embeddings.
        
        Args:
            hidden_size: Dimension of the embeddings. Must match
                the model's hidden dimension.
            max_position_embeddings: Maximum sequence length to cache.
                Sequences longer than this will trigger cache rebuild.
                Default: 8192.
            base: Base frequency for computing position encodings.
                Higher values give slower frequency decay. Default: 10000.
            device: Device to place the embedding buffers.
        
        Example:
            >>> pos_emb = YvSinusoidalPositionEmbedding(
            ...     hidden_size=4096,
            ...     max_position_embeddings=10485760,  # 10M tokens
            ...     base=10000
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, hidden_size, 2).float() / hidden_size))
        self.register_buffer("inv_freq", inv_freq)
        
        self._build_cache(max_position_embeddings, device)
        
    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Build position embedding cache for efficient lookup.
        
        Precomputes cosine and sine values for all positions up to
        seq_len to avoid repeated computation during forward passes.
        
        Args:
            seq_len: Maximum sequence length to cache.
            device: Device for the cache tensors.
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        
    def forward(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ) -> tuple:
        """Get positional embeddings for a sequence.
        
        Returns precomputed cosine and sine values for positions 0 to seq_len-1.
        
        Args:
            seq_len: Length of the sequence to encode.
            device: Target device for the output tensors.
            dtype: Target data type for the output tensors.
        
        Returns:
            Tuple of (cos, sin) tensors, each of shape [seq_len, hidden_size].
        
        Note:
            If seq_len exceeds max_position_embeddings, the cache is
            automatically rebuilt to accommodate the longer sequence.
        """
        if seq_len > self.max_position_embeddings:
            self._build_cache(seq_len, device)
            
        cos = self.cos_cached[:seq_len].to(dtype)
        sin = self.sin_cached[:seq_len].to(dtype)
        return cos, sin


class YvLearnedPositionEmbedding(nn.Module):
    """Learned positional embeddings with interpolation for long sequences.
    
    Supports interpolation for sequences longer than the training length
    using linear interpolation of the learned embeddings. This allows the
    model to handle longer sequences at inference time while maintaining
    the benefits of learned position representations.
    
    Mathematical Formulation:
        For position p <= max_position:
            emb(p) = Embedding(p)
        For position p > max_position:
            scaled_p = p * (max_position / actual_max)
            emb(p) = Embedding(round(scaled_p))
    
    Key Features:
        - Learned position representations
        - Linear interpolation for extrapolation
        - More flexible than sinusoidal encoding
        - Can capture position-specific patterns
    
    Comparison with Sinusoidal:
        - Learned: More flexible, can capture task-specific patterns
        - Sinusoidal: No parameters, better extrapolation, deterministic
    
    Use Cases:
        - Tasks with position-specific patterns
        - Models where position semantics matter
        - Fine-tuning on specific sequence lengths
    
    Performance Characteristics:
        - Memory: O(max_position * hidden_size) for embedding table
        - Compute: O(seq_len) for lookup
        - Parameters: max_position * hidden_size
    
    Attributes:
        hidden_size (int): Dimension of the embeddings.
        max_position_embeddings (int): Maximum sequence length.
        embedding (nn.Embedding): Learned position embedding table.
    
    Example:
        >>> pos_emb = YvLearnedPositionEmbedding(
        ...     hidden_size=4096,
        ...     max_position_embeddings=8192
        ... )
        >>> position_ids = torch.arange(1024).unsqueeze(0)
        >>> embeddings = pos_emb(position_ids)
    
    Note:
        When using interpolation, the quality may degrade for sequences
        much longer than the training length. Consider sinusoidal or
        YaRN embeddings for better long-context support.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 8192,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize learned position embeddings.
        
        Args:
            hidden_size: Dimension of the embeddings. Must match
                the model's hidden dimension.
            max_position_embeddings: Maximum sequence length supported
                by the embedding table. Default: 8192.
            device: Device for the embedding parameters.
            dtype: Data type for the embedding parameters.
        
        Example:
            >>> pos_emb = YvLearnedPositionEmbedding(
            ...     hidden_size=4096,
            ...     max_position_embeddings=8192,
            ...     device='cuda'
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
        self.embedding = nn.Embedding(
            max_position_embeddings,
            hidden_size,
            device=device,
            dtype=dtype
        )
        
    def forward(
        self,
        position_ids: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Get positional embeddings for given positions.
        
        Looks up learned embeddings for the specified positions,
        with interpolation for positions beyond the training range.
        
        Args:
            position_ids: Position indices tensor of shape [batch, seq_len].
                If None, positions 0 to seq_len-1 are used.
            seq_len: Optional sequence length. Required when position_ids
                is None to generate default position indices.
        
        Returns:
            Positional embedding tensor of shape [batch, seq_len, hidden_size].
        
        Note:
            For positions beyond max_position_embeddings, linear interpolation
            is used to map positions to the available embedding range.
        """
        if position_ids is None:
            assert seq_len is not None, "seq_len required when position_ids is None"
            position_ids = torch.arange(seq_len, device=self.embedding.weight.device)
            position_ids = position_ids.unsqueeze(0)
            
        max_pos = position_ids.max().item() + 1
        if max_pos > self.max_position_embeddings:
            return self._interpolate(position_ids, max_pos)
            
        return self.embedding(position_ids)
        
    def _interpolate(self, position_ids: torch.Tensor, max_pos: int) -> torch.Tensor:
        """Interpolate embeddings for sequences longer than training length.
        
        Uses linear scaling to map positions to the available embedding range.
        This is a simple but effective approach for handling longer sequences.
        
        Args:
            position_ids: Position indices to interpolate.
            max_pos: Maximum position value in the input.
        
        Returns:
            Interpolated embeddings for the given positions.
        
        Note:
            This interpolation method compresses the position range,
            which may affect the model's ability to distinguish positions
            at very long sequence lengths.
        """
        scale = self.max_position_embeddings / max_pos
        scaled_ids = (position_ids.float() * scale).long()
        scaled_ids = scaled_ids.clamp(0, self.max_position_embeddings - 1)
        return self.embedding(scaled_ids)


class YvModalityEmbedding(nn.Module):
    """Modality-aware embeddings for multimodal input support.
    
    Provides learnable embeddings for different input modalities
    (text, image, audio, video, document, agentic) with projection
    to the model's hidden dimension. Uses a gated fusion mechanism
    to integrate modality information with the input representations.
    
    Supported Modalities:
        - text (0): Standard text input
        - image (1): Visual content (single images)
        - audio (2): Audio signals and speech
        - video (3): Video sequences
        - document (4): Structured documents
        - agentic (5): Agent-specific inputs
    
    Mathematical Formulation:
        modality_emb = Embedding(modality_id)
        projected = Linear(modality_emb)
        gate = Sigmoid(Linear(concat(hidden, modality_emb)))
        output = hidden + gate * projected
    
    Key Features:
        - 6 predefined modality types
        - Gated fusion for modality integration
        - Learnable modality representations
        - Projection to model dimension
    
    Architecture Integration:
        Modality embeddings are added to the token embeddings after
        position encoding. The gated mechanism allows the model to
        learn how much modality information to incorporate.
    
    Use Cases:
        - Multimodal models (vision-language, audio-text)
        - Document understanding with layout information
        - Agent systems with action embeddings
        - Cross-modal retrieval and generation
    
    Attributes:
        hidden_size (int): Model hidden dimension.
        modality_embed_dim (int): Dimension for modality embeddings.
        num_modalities (int): Number of supported modalities.
        modality_embedding (nn.Embedding): Modality type embeddings.
        modality_projection (nn.Linear): Projects modality embeddings to hidden_size.
        modality_gate (nn.Sequential): Gating network for fusion.
    
    Example:
        >>> modality_emb = YvModalityEmbedding(
        ...     hidden_size=4096,
        ...     modality_embed_dim=64,
        ...     num_modalities=6
        ... )
        >>> hidden = torch.randn(2, 1024, 4096)
        >>> output = modality_emb(hidden, modality="image")
    """
    
    MODALITY_IDS = {
        "text": 0,
        "image": 1,
        "audio": 2,
        "video": 3,
        "document": 4,
        "agentic": 5,
    }
    
    def __init__(
        self,
        hidden_size: int,
        modality_embed_dim: int = 64,
        num_modalities: int = 6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize modality embeddings.
        
        Args:
            hidden_size: Model hidden dimension for projection target.
            modality_embed_dim: Dimension for internal modality embeddings.
                Default: 64.
            num_modalities: Number of supported modality types. Default: 6.
            device: Device for embedding parameters.
            dtype: Data type for embedding parameters.
        
        Example:
            >>> modality_emb = YvModalityEmbedding(
            ...     hidden_size=4096,
            ...     modality_embed_dim=64,
            ...     num_modalities=6
            ... )
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.modality_embed_dim = modality_embed_dim
        self.num_modalities = num_modalities
        
        self.modality_embedding = nn.Embedding(
            num_modalities,
            modality_embed_dim,
            device=device,
            dtype=dtype
        )
        
        self.modality_projection = nn.Linear(
            modality_embed_dim,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype
        )
        
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_size + modality_embed_dim, hidden_size),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        modality: str = "text"
    ) -> torch.Tensor:
        """Add modality embeddings to hidden states.
        
        Applies gated fusion of modality information with the input.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            modality: Modality type string. Must be one of: "text", "image",
                "audio", "video", "document", "agentic". Default: "text".
        
        Returns:
            Tensor with modality information added, same shape as input.
        
        Note:
            The gated mechanism allows the model to learn how much
            modality information to incorporate at each position.
        """
        modality_id = self.MODALITY_IDS.get(modality, 0)
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        modality_ids = torch.full(
            (batch_size, seq_len),
            modality_id,
            dtype=torch.long,
            device=device
        )
        
        modality_embed = self.modality_embedding(modality_ids)
        modality_proj = self.modality_projection(modality_embed)
        
        gate_input = torch.cat([hidden_states, modality_embed], dim=-1)
        gate = self.modality_gate(gate_input)
        
        return hidden_states + gate * modality_proj
        
    def get_modality_embedding(self, modality: str) -> torch.Tensor:
        """Get the embedding vector for a specific modality.
        
        Retrieves the learned embedding for a modality type without
        applying projection or gating.
        
        Args:
            modality: Modality type string.
        
        Returns:
            Modality embedding tensor of shape [modality_embed_dim].
        """
        modality_id = self.MODALITY_IDS.get(modality, 0)
        return self.modality_embedding.weight[modality_id]


class YvTokenEmbedding(nn.Module):
    """Token embedding layer with optional adaptive scaling and normalization.
    
    Provides standard token embeddings with optional adaptive scaling
    and deep normalization for improved training stability. This forms
    the first layer of the transformer, converting token IDs to dense
    vector representations.
    
    Mathematical Formulation:
        emb = Embedding(token_id)
        if adaptive_scaling:
            emb = emb * scale + bias
        if deep_norm:
            emb = LayerNorm(emb)
    
    Key Features:
        - Standard vocabulary-based token embeddings
        - Optional adaptive scaling for training stability
        - Optional deep normalization for gradient flow
        - Padding token handling for variable-length sequences
    
    Training Stability:
        The combination of adaptive scaling and layer normalization
        helps prevent gradient issues in deep models:
        - Adaptive scaling: Learns optimal embedding magnitudes
        - Layer normalization: Stabilizes activation distributions
    
    Memory Considerations:
        - Embedding table: vocab_size * hidden_size
        - Adaptive scaling: 2 * hidden_size (scale + bias)
        - Layer normalization: 2 * hidden_size (weight + bias)
    
    Attributes:
        vocab_size (int): Size of the vocabulary.
        hidden_size (int): Dimension of embedding vectors.
        padding_idx (Optional[int]): Index of padding token.
        embedding (nn.Embedding): Token embedding table.
        adaptive_scaling (Optional[YvAdaptiveEmbeddingScaling]):
            Optional adaptive scaling module.
        layer_norm (Optional[nn.LayerNorm]): Optional layer normalization.
    
    Example:
        >>> token_emb = YvTokenEmbedding(
        ...     vocab_size=151646,
        ...     hidden_size=4096,
        ...     use_adaptive_scaling=True,
        ...     use_deep_norm=True
        ... )
        >>> input_ids = torch.randint(0, 151646, (2, 1024))
        >>> embeddings = token_emb(input_ids)
    
    Note:
        Padding token embeddings are initialized to zero and excluded
        from gradient computation to prevent them from affecting training.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int] = None,
        use_adaptive_scaling: bool = True,
        use_deep_norm: bool = True,
        layer_norm_eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize token embeddings with optional normalization.
        
        Args:
            vocab_size: Size of the vocabulary. Determines the size
                of the embedding table.
            hidden_size: Dimension of embedding vectors. Must match
                the model's hidden dimension.
            padding_idx: Index of padding token. Embeddings at this
                index will be zeroed and excluded from gradients.
            use_adaptive_scaling: Whether to apply learned scaling
                to embeddings. Default: True.
            use_deep_norm: Whether to apply layer normalization to
                embeddings. Default: True.
            layer_norm_eps: Epsilon for layer normalization stability.
                Default: 1e-6.
            device: Device for embedding parameters.
            dtype: Data type for embedding parameters.
        
        Example:
            >>> token_emb = YvTokenEmbedding(
            ...     vocab_size=151646,
            ...     hidden_size=4096,
            ...     padding_idx=0,
            ...     use_adaptive_scaling=True
            ... )
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=padding_idx,
            device=device,
            dtype=dtype
        )
        
        if use_adaptive_scaling:
            self.adaptive_scaling = YvAdaptiveEmbeddingScaling(hidden_size)
        else:
            self.adaptive_scaling = None
            
        if use_deep_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)
        else:
            self.layer_norm = None
            
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embedding weights with normal distribution.
        
        Uses a standard normal distribution with mean 0 and std 0.02.
        Padding token embeddings are initialized to zero.
        """
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.padding_idx], 0)
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings for input token IDs.
        
        Looks up embeddings for each token ID and optionally applies
        adaptive scaling and layer normalization.
        
        Args:
            input_ids: Token ID tensor of shape [batch, seq_len].
                Values should be in range [0, vocab_size).
        
        Returns:
            Embedding tensor of shape [batch, seq_len, hidden_size].
        
        Note:
            The output is ready for addition with position embeddings
            or direct use in transformer layers.
        """
        embeddings = self.embedding(input_ids)
        
        if self.adaptive_scaling is not None:
            embeddings = self.adaptive_scaling(embeddings)
            
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
            
        return embeddings


class YvUnifiedEmbedding(nn.Module):
    """Unified embedding module combining token, position, and modality embeddings.
    
    This is the main embedding class used by the Yv model, providing
    a comprehensive embedding solution that integrates multiple embedding
    types into a single, cohesive interface.
    
    Architecture Components:
        1. Token Embeddings: Vocabulary-based token representations
        2. Position Embeddings: Learned or sinusoidal position encodings
        3. Modality Embeddings: Multimodal input type information
        4. Embedding Scaling: Optional scaling for training stability
    
    Mathematical Formulation:
        token_emb = TokenEmbedding(input_ids) * scale
        if learned_position:
            pos_emb = PositionEmbedding(position_ids)
            combined = token_emb + pos_emb
        else:
            combined = apply_rotary(token_emb, cos, sin)
        if modality != "text":
            combined = ModalityEmbedding(combined, modality)
        output = dropout(combined)
    
    Key Features:
        - Unified interface for all embedding operations
        - Supports both learned and sinusoidal position encodings
        - Native multimodal support with 6 modality types
        - Configurable adaptive scaling and normalization
        - Rotary position embedding support
    
    Configuration Options:
        - use_learned_position: Toggle between learned/sinusoidal PE
        - use_adaptive_scaling: Enable learned embedding scaling
        - use_deep_norm: Enable layer normalization
        - num_modalities: Number of supported modality types
    
    Performance Characteristics:
        - Memory: vocab_size * hidden_size + position_embeddings
        - Compute: O(batch * seq_len * hidden_size)
        - Supports gradient checkpointing for memory efficiency
    
    Attributes:
        config (YvEmbeddingConfig): Embedding configuration.
        token_embedding (YvTokenEmbedding): Token embedding layer.
        position_embedding: Position embedding layer (learned or sinusoidal).
        modality_embedding (YvModalityEmbedding): Modality embedding layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        embedding_scale (nn.Parameter): Learned embedding scale factor.
    
    Example:
        >>> config = YvEmbeddingConfig(
        ...     vocab_size=151646,
        ...     hidden_size=4096,
        ...     max_position_embeddings=10485760
        ... )
        >>> embeddings = YvUnifiedEmbedding(config)
        >>> output = embeddings(input_ids, modality="text")
    
    Note:
        This class is typically instantiated by YvModel and should
        not be used directly unless building a custom architecture.
    """
    
    def __init__(self, config: YvEmbeddingConfig, device=None, dtype=None):
        """Initialize unified embeddings with configuration.
        
        Args:
            config: YvEmbeddingConfig instance containing all
                embedding parameters including vocab_size, hidden_size,
                position embedding settings, and modality configuration.
            device: Device for all embedding parameters.
            dtype: Data type for all embedding parameters.
        
        Example:
            >>> config = YvEmbeddingConfig(
            ...     vocab_size=151646,
            ...     hidden_size=4096,
            ...     use_learned_position=False
            ... )
            >>> embeddings = YvUnifiedEmbedding(config, device='cuda')
        """
        super().__init__()
        self.config = config
        
        self.token_embedding = YvTokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            use_adaptive_scaling=config.use_adaptive_scaling,
            use_deep_norm=config.use_deep_norm,
            layer_norm_eps=config.layer_norm_eps,
            device=device,
            dtype=dtype
        )
        
        if config.use_learned_position:
            self.position_embedding = YvLearnedPositionEmbedding(
                hidden_size=config.hidden_size,
                max_position_embeddings=config.max_position_embeddings,
                device=device,
                dtype=dtype
            )
        else:
            self.position_embedding = YvSinusoidalPositionEmbedding(
                hidden_size=config.hidden_size,
                max_position_embeddings=config.max_position_embeddings,
                device=device
            )
            
        self.modality_embedding = YvModalityEmbedding(
            hidden_size=config.hidden_size,
            modality_embed_dim=config.modality_embed_dim,
            num_modalities=config.num_modalities,
            device=device,
            dtype=dtype
        )
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
        self.embedding_scale = nn.Parameter(
            torch.ones(config.hidden_size) * (config.hidden_size ** 0.5)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        modality: str = "text",
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute unified embeddings for input tokens.
        
        Combines token, position, and modality embeddings into a unified
        representation suitable for transformer processing.
        
        Args:
            input_ids: Token IDs of shape [batch, seq_len]. Values should
                be in range [0, vocab_size).
            position_ids: Optional position indices of shape [batch, seq_len].
                If None, positions 0 to seq_len-1 are used.
            modality: Modality type for the input. Must be one of: "text",
                "image", "audio", "video", "document", "agentic".
                Default: "text".
            attention_mask: Optional attention mask. Currently not used
                directly in embedding computation but included for API
                compatibility.
        
        Returns:
            Dictionary containing:
                - embeddings: Combined embedding tensor [batch, seq_len, hidden_size]
                - position_embeddings: Position embedding tensor (None for sinusoidal)
                - token_embeddings: Token embedding tensor before position addition
        
        Note:
            For sinusoidal position embeddings, rotary position encoding is
            applied directly to the token embeddings rather than added.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        token_embeds = self.token_embedding(input_ids)
        token_embeds = token_embeds * self.embedding_scale
        
        if isinstance(self.position_embedding, YvLearnedPositionEmbedding):
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_embeds = self.position_embedding(position_ids, seq_len)
        else:
            cos, sin = self.position_embedding(seq_len, device, token_embeds.dtype)
            position_embeds = self._apply_rotary_embedding(token_embeds, cos, sin)
            
        embeddings = token_embeds + position_embeds
        
        if modality != "text":
            embeddings = self.modality_embedding(embeddings, modality)
            
        embeddings = self.dropout(embeddings)
        
        return {
            "embeddings": embeddings,
            "position_embeddings": position_embeds if isinstance(self.position_embedding, YvLearnedPositionEmbedding) else None,
            "token_embeddings": token_embeds,
        }
        
    def _apply_rotary_embedding(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary position embeddings to input tensor.
        
        Implements the rotary position embedding transformation for
        sinusoidal position encoding mode.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size].
            cos: Cosine values of shape [max_seq_len, hidden_size].
            sin: Sine values of shape [max_seq_len, hidden_size].
        
        Returns:
            Tensor with rotary embeddings applied, same shape as input.
        
        Note:
            This is a simplified rotary implementation. For full RoPE
            support with YaRN scaling, use YvYaRNRotaryEmbedding.
        """
        seq_len = x.shape[1]
        cos = cos[:seq_len].unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0)
        
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        rotated = torch.cat([
            x1 * cos[..., ::2] - x2 * sin[..., ::2],
            x1 * sin[..., 1::2] + x2 * cos[..., 1::2]
        ], dim=-1)
        
        return rotated
        
    def resize_token_embeddings(self, new_vocab_size: int) -> nn.Embedding:
        """Resize the token embedding matrix for vocabulary changes.
        
        Creates a new embedding table with the specified size and copies
        weights from the old table. Useful for adding new tokens or
        reducing vocabulary size.
        
        Args:
            new_vocab_size: New vocabulary size. Can be larger or smaller
                than the current size.
        
        Returns:
            The new embedding layer.
        
        Note:
            When increasing size, new embeddings are initialized with
            the default initialization. When decreasing, embeddings
            beyond the new size are discarded.
        """
        old_embeddings = self.token_embedding.embedding
        new_embeddings = nn.Embedding(
            new_vocab_size,
            self.config.hidden_size,
            padding_idx=old_embeddings.padding_idx,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype
        )
        
        num_to_copy = min(old_embeddings.num_embeddings, new_vocab_size)
        new_embeddings.weight.data[:num_to_copy] = old_embeddings.weight.data[:num_to_copy]
        
        self.token_embedding.embedding = new_embeddings
        self.token_embedding.vocab_size = new_vocab_size
        self.config.vocab_size = new_vocab_size
        
        return new_embeddings


class YvMultimodalEmbeddingProjector(nn.Module):
    """Projects multimodal features to the model's embedding space.
    
    Handles projection of features from different modalities (vision, audio,
    video, document, agentic) to the unified embedding dimension. Each
    modality has a dedicated projection network with layer normalization
    and GELU activation for robust cross-modal alignment.
    
    Supported Modalities:
        - vision/image: Visual features from vision encoders
        - audio: Audio features from audio encoders
        - video: Spatiotemporal features from video encoders
        - document/doc: Document features from document encoders
        - agentic: Agent-specific features from action encoders
    
    Architecture:
        Each projection consists of:
        1. Linear projection from source dimension to hidden_size
        2. Layer normalization for stable training
        3. GELU activation for non-linearity
        4. Linear projection for final alignment
    
    Mathematical Formulation:
        projected = Linear_2(GELU(LayerNorm(Linear_1(features))))
    
    Key Features:
        - Modality-specific projection networks
        - Layer normalization for training stability
        - GELU activation for smooth gradients
        - Consistent output dimension across modalities
    
    Use Cases:
        - Vision-language models (CLIP-style encoders)
        - Audio-text models (speech recognition/translation)
        - Video understanding (video encoders)
        - Document AI (layout-aware encoders)
        - Agent systems (action embeddings)
    
    Performance Characteristics:
        - Memory: O(sum(modality_dims) * hidden_size) for projections
        - Compute: O(batch * seq_len * hidden_size) per modality
        - Each projection has ~2 * modality_dim * hidden_size parameters
    
    Attributes:
        vision_proj (nn.Sequential): Vision/image projection network.
        audio_proj (nn.Sequential): Audio projection network.
        video_proj (nn.Sequential): Video projection network.
        doc_proj (nn.Sequential): Document projection network.
        agentic_proj (nn.Sequential): Agentic projection network.
    
    Example:
        >>> projector = YvMultimodalEmbeddingProjector(
        ...     hidden_size=4096,
        ...     vision_dim=768,
        ...     audio_dim=512
        ... )
        >>> vision_features = torch.randn(2, 196, 768)  # 14x14 patches
        >>> projected = projector(vision_features, modality="vision")
    
    Note:
        The projection networks are designed to be lightweight while
        providing sufficient capacity for cross-modal alignment.
    """
    
    def __init__(
        self,
        hidden_size: int,
        vision_dim: int = 768,
        audio_dim: int = 512,
        video_dim: int = 1024,
        doc_dim: int = 768,
        agentic_dim: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize multimodal projectors for all supported modalities.
        
        Creates dedicated projection networks for each modality type,
        each consisting of a two-layer MLP with layer normalization
        and GELU activation.
        
        Args:
            hidden_size: Target hidden dimension for all projections.
                This should match the model's hidden dimension.
            vision_dim: Vision encoder output dimension. Default: 768
                (e.g., CLIP ViT-B/16 output dimension).
            audio_dim: Audio encoder output dimension. Default: 512
                (e.g., Whisper base model dimension).
            video_dim: Video encoder output dimension. Default: 1024
                (e.g., VideoMAE or similar encoder dimension).
            doc_dim: Document encoder output dimension. Default: 768
                (e.g., LayoutLM output dimension).
            agentic_dim: Agentic encoder output dimension. Default: 512
                (e.g., action embedding dimension).
            device: Device for all projection parameters.
            dtype: Data type for all projection parameters.
        
        Example:
            >>> projector = YvMultimodalEmbeddingProjector(
            ...     hidden_size=4096,
            ...     vision_dim=768,  # CLIP ViT-B/16
            ...     audio_dim=512,   # Whisper base
            ...     video_dim=1024,  # VideoMAE
            ...     doc_dim=768      # LayoutLM
            ... )
        """
        super().__init__()
        
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_size, bias=False, device=device, dtype=dtype),
            nn.LayerNorm(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_size, bias=False, device=device, dtype=dtype),
            nn.LayerNorm(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        )
        
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_size, bias=False, device=device, dtype=dtype),
            nn.LayerNorm(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        )
        
        self.doc_proj = nn.Sequential(
            nn.Linear(doc_dim, hidden_size, bias=False, device=device, dtype=dtype),
            nn.LayerNorm(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        )
        
        self.agentic_proj = nn.Sequential(
            nn.Linear(agentic_dim, hidden_size, bias=False, device=device, dtype=dtype),
            nn.LayerNorm(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        modality: str
    ) -> torch.Tensor:
        """Project features from a specific modality to embedding space.
        
        Selects the appropriate projection network based on the modality
        and applies it to the input features.
        
        Args:
            features: Input features tensor of shape [batch, seq_len, source_dim]
                where source_dim matches the corresponding modality dimension.
            modality: Modality type string. Must be one of:
                - "vision" or "image": For visual features
                - "audio": For audio features
                - "video": For video features
                - "document" or "doc": For document features
                - "agentic": For agent action features
        
        Returns:
            Projected features tensor of shape [batch, seq_len, hidden_size].
        
        Raises:
            ValueError: If modality is not recognized.
        
        Example:
            >>> # Vision features from CLIP (14x14 patches)
            >>> vision_features = torch.randn(2, 196, 768)
            >>> projected = projector(vision_features, modality="vision")
            >>> print(projected.shape)  # [2, 196, 4096]
        """
        modality_projectors = {
            "vision": self.vision_proj,
            "image": self.vision_proj,
            "audio": self.audio_proj,
            "video": self.video_proj,
            "document": self.doc_proj,
            "doc": self.doc_proj,
            "agentic": self.agentic_proj,
        }
        
        projector = modality_projectors.get(modality)
        if projector is None:
            raise ValueError(f"Unknown modality: {modality}")
            
        return projector(features)
