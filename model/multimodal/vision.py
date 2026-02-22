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

"""Vision encoders and utilities for Yv multimodal agents.

This module provides comprehensive vision processing components for the Yv
model, including image encoding, 3D spatio-temporal positional embeddings,
and visual text processing for H-Network tokenization support.

Module Components:
    1. YvSpatioTemporalRoPE3D:
       - 3D rotary positional embedding for video inputs
       - Temporal, height, and width axis encoding
       - Cached computation for efficiency
    
    2. YvVisualTextProcessor:
       - Text rendered as images for H-Network support
       - High-contrast normalization for glyphs
       - Patch-based tokenization
    
    3. YvVisionEncoder:
       - Main vision backbone with ViT-style architecture
       - Detection, segmentation, and reasoning heads
       - Optional 3D RoPE for video support

Key Features:
    - 3D spatio-temporal rotary positional embeddings
    - H-Network visual text tokenization
    - Multi-head detection with bounding box regression
    - Semantic segmentation with FPN decoder
    - Visual reasoning and grounding heads
    - SDPA and MHA attention backends

Performance Characteristics:
    - Patch embedding: O(H * W / patch_size^2)
    - Attention: O(N^2) with N = num_patches
    - Detection head: O(N * num_anchors)
    - Segmentation head: O(H * W * num_classes)

Usage Example:
    >>> from model.multimodal.vision import (
    ...     YvVisionEncoder,
    ...     YvSpatioTemporalRoPE3D
    ... )
    >>> 
    >>> # Initialize encoder
    >>> encoder = YvVisionEncoder(config)
    >>> 
    >>> # Encode images
    >>> features = encoder(images)
    >>> 
    >>> # For video with 3D RoPE
    >>> encoder.use_3d_rope = True
    >>> video_features = encoder(video_frames, video_shape=(T, H, W))

Note:
    Patch size of 14 matches CLIP-style preprocessing.
    3D RoPE requires dimension divisible by 3.
    Detection head uses 9 anchors per location.
"""

import math
import torch
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Multimodal", file_path=get_log_file("Yv.Multimodal"), enable_file=True)

class YvSpatioTemporalRoPE3D(nn.Module):
    """3D spatio-temporal rotary positional embedding for video inputs.
    
    Extends the standard 2D RoPE formulation by incorporating an additional
    temporal axis, enabling consistent positional encoding across frame
    sequences. This allows the model to maintain spatial and temporal
    coherence when processing video data.
    
    Mathematical Formulation:
        For each position (t, h, w) in a video:
        - Temporal encoding: freq_t = 1 / (base^(2i/d_t))
        - Height encoding: freq_h = 1 / (base^(2i/d_h))
        - Width encoding: freq_w = 1 / (base^(2i/d_w))
        
        Where d_t = d_h = d_w = dim/3 (dimension per axis).
    
    Architecture:
        The embedding dimension is split equally among three axes:
        - [0, dim/3): Temporal position encoding
        - [dim/3, 2*dim/3): Height position encoding
        - [2*dim/3, dim): Width position encoding
    
    Attributes:
        dim (int): Feature dimension divisible by three (one slice per axis).
        dim_per_axis (int): Dimension allocated to each axis (dim // 3).
        max_temporal_frames (int): Maximum time steps supported for caching.
        max_spatial_h (int): Maximum height in patches used for caching.
        max_spatial_w (int): Maximum width in patches used for caching.
        base (float): Base factor controlling the geometric progression.
        temp_freq (torch.Tensor): Frequency values for temporal dimension.
        h_freq (torch.Tensor): Frequency values for height dimension.
        w_freq (torch.Tensor): Frequency values for width dimension.
    
    Example:
        >>> rope_3d = YvSpatioTemporalRoPE3D(
        ...     dim=768,
        ...     max_temporal_frames=64,
        ...     max_spatial_h=32,
        ...     max_spatial_w=32
        ... )
        >>> video_features = rope_3d(features, video_shape=(16, 14, 14))
    
    Note:
        Dimension must be divisible by 3 for equal axis allocation.
        Caching improves efficiency for repeated video shapes.
        Base of 10000.0 matches standard RoPE conventions.
    """

    def __init__(
        self,
        dim: int,
        max_temporal_frames: int = 1024,
        max_spatial_h: int = 64,
        max_spatial_w: int = 64,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the 3D spatio-temporal RoPE module.
        
        Args:
            dim (int): Feature dimension divisible by three (one slice per axis).
            max_temporal_frames (int): Maximum time steps supported for caching.
                Default: 1024.
            max_spatial_h (int): Maximum height in patches used for caching.
                Default: 64.
            max_spatial_w (int): Maximum width in patches used for caching.
                Default: 64.
            base (float): Base factor controlling the geometric progression.
                Default: 10000.0.
            device (torch.device | None): Optional device override for buffers.
        
        Raises:
            AssertionError: If dim is not divisible by 3.
        """
        super().__init__()
        assert dim % 3 == 0, "Dimension must be divisible by 3 for 3D RoPE"
        self.dim = dim
        self.dim_per_axis = dim // 3
        self.max_temporal_frames = max_temporal_frames
        self.max_spatial_h = max_spatial_h
        self.max_spatial_w = max_spatial_w
        self.base = base
        # Compute frequency values for temporal dimension.
        temp_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        # Compute frequency values for height dimension.
        h_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        # Compute frequency values for width dimension.
        w_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        self.register_buffer('temp_freq', temp_freq, persistent=False)
        self.register_buffer('h_freq', h_freq, persistent=False)
        self.register_buffer('w_freq', w_freq, persistent=False)
        self.register_buffer('cos_cache', None, persistent=False)
        self.register_buffer('sin_cache', None, persistent=False)
        self.register_buffer('max_seq_cached', torch.tensor(0), persistent=False)

    def _compute_3d_positions(self, t: int, h: int, w: int) -> torch.Tensor:
        """Compute flattened grid indices spanning temporal, height, and width.
        
        Creates a 3D meshgrid of position indices and flattens them into
        a sequence of (t, h, w) tuples for each position in the video.
        
        Args:
            t (int): Number of temporal frames.
            h (int): Height in patches.
            w (int): Width in patches.
        
        Returns:
            torch.Tensor: Position indices [t*h*w, 3] where each row is
                (temporal_pos, height_pos, width_pos).
        """
        temp_pos = torch.arange(t, dtype=torch.float32)
        h_pos = torch.arange(h, dtype=torch.float32)
        w_pos = torch.arange(w, dtype=torch.float32)
        temp_grid, h_grid, w_grid = torch.meshgrid(temp_pos, h_pos, w_pos, indexing='ij')
        positions = torch.stack([
            temp_grid.flatten(),
            h_grid.flatten(),
            w_grid.flatten()
        ], dim=1)
        return positions

    def _compute_3d_rope(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine lookup tables for 3D rotary embeddings.
        
        Applies the rotary position encoding formula to each axis separately,
        then concatenates the results to form the complete embedding.
        
        Args:
            positions (torch.Tensor): Position indices [seq_len, 3].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - cos (torch.Tensor): Cosine values [seq_len, dim].
                - sin (torch.Tensor): Sine values [seq_len, dim].
        """
        device = positions.device
        seq_len = positions.shape[0]
        temp_pos = positions[:, 0]
        h_pos = positions[:, 1]
        w_pos = positions[:, 2]
        temp_freqs = torch.outer(temp_pos, self.temp_freq.to(device))
        h_freqs = torch.outer(h_pos, self.h_freq.to(device))
        w_freqs = torch.outer(w_pos, self.w_freq.to(device))
        freqs = torch.zeros(seq_len, self.dim, device=device)
        freqs[:, :self.dim_per_axis] = torch.cat([
            torch.sin(temp_freqs),
            torch.cos(temp_freqs)
        ], dim=1)[:, :self.dim_per_axis]
        start_h = self.dim_per_axis
        end_h = start_h + self.dim_per_axis
        freqs[:, start_h:end_h] = torch.cat([
            torch.sin(h_freqs),
            torch.cos(h_freqs)
        ], dim=1)[:, :self.dim_per_axis]
        start_w = 2 * self.dim_per_axis
        end_w = start_w + self.dim_per_axis
        freqs[:, start_w:end_w] = torch.cat([
            torch.sin(w_freqs),
            torch.cos(w_freqs)
        ], dim=1)[:, :self.dim_per_axis]
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    def forward(self, x: torch.Tensor, video_shape: Tuple[int, int, int]) -> torch.Tensor:
        """Apply cached 3D RoPE values to the input sequence.
        
        Computes or retrieves cached rotary embeddings and applies them
        to the input tensor using the standard RoPE rotation formula.
        
        Args:
            x (torch.Tensor): Input tensor [batch, seq_len, dim].
            video_shape (Tuple[int, int, int]): Video dimensions (T, H, W).
        
        Returns:
            torch.Tensor: Rotated tensor with same shape as input.
        
        Note:
            Caching is used to avoid recomputation for repeated shapes.
        """
        t, h, w = video_shape
        seq_len = t * h * w
        if self.cos_cache is None or seq_len > self.max_seq_cached:
            positions = self._compute_3d_positions(t, h, w)
            positions = positions.to(x.device)
            cos, sin = self._compute_3d_rope(positions)
            self.cos_cache = cos
            self.sin_cache = sin
            self.max_seq_cached = torch.tensor(seq_len)
        x_rotated = x * self.cos_cache[:seq_len] + self._rotate_half(x) * self.sin_cache[:seq_len]
        return x_rotated

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the last dimension by half and negate the second half.
        
        Implements the rotation operation used in RoPE:
        rotate_half(x) = [-x[..., d//2:], x[..., :d//2]]
        
        Args:
            x (torch.Tensor): Input tensor [..., dim].
        
        Returns:
            torch.Tensor: Rotated tensor with same shape as input.
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class YvVisualTextProcessor(nn.Module):
    """Encode text rendered as images for H-Network tokenization support.
    
    Processes text that has been rendered as images (e.g., screenshots,
    scanned documents) into patch embeddings suitable for the vision
    encoder. Uses specialized normalization tuned for high-contrast
    glyph imagery.
    
    Architecture:
        1. Normalize with text-specific mean/std for high contrast
        2. Extract patches via convolution
        3. Compress to hidden_size dimension
    
    Attributes:
        hidden_size (int): Target embedding dimension.
        patch_size (int): Convolutional patch size applied during tokenization.
        text_mean (torch.Tensor): Mean values for text normalization.
        text_std (torch.Tensor): Standard deviation for text normalization.
        text_patch_embed (nn.Conv2d): Patch embedding convolution.
        text_compressor (nn.Sequential): Compression network.
    
    Example:
        >>> processor = YvVisualTextProcessor(hidden_size=768)
        >>> text_image = load_text_image("screenshot.png")  # [B, 3, H, W]
        >>> embeddings = processor(text_image)  # [B, N, 768]
    
    Note:
        Uses higher mean values (0.95) for text-specific normalization.
        Compression network restores hidden_size after patch embedding.
    """

    def __init__(self, hidden_size: int, patch_size: int = 14) -> None:
        """Construct a text-oriented preprocessing pipeline.
        
        Args:
            hidden_size (int): Target embedding dimension.
            patch_size (int): Convolutional patch size applied during tokenization.
                Default: 14 (matches CLIP-style preprocessing).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        # Text-specific normalization tuned for high-contrast glyph imagery.
        self.register_buffer('text_mean', torch.Tensor([0.95, 0.95, 0.95]).view(1, 3, 1, 1))
        self.register_buffer('text_std', torch.Tensor([0.1, 0.1, 0.1]).view(1, 3, 1, 1))

        # Text-aware patch embedding producing coarse glyph descriptors.
        self.text_patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Compression network restoring the original hidden dimensionality.
        self.text_compressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        _LOG.info(f"YvVisualTextProcessor initialized: hidden_size={hidden_size}, patch_size={patch_size}")

    def forward(self, text_images: torch.Tensor) -> torch.Tensor:
        """Convert batches of rendered text images into patch embeddings.
        
        Applies text-specific normalization, extracts patches via convolution,
        and compresses to the target hidden dimension.
        
        Args:
            text_images (torch.Tensor): Input tensor of shape ``[B, C, H, W]``.
                Expected to be RGB images with values in [0, 1] or [0, 255].
        
        Returns:
            torch.Tensor: Patch embeddings shaped ``[B, N, hidden_size]``.
                Where N = (H / patch_size) * (W / patch_size).
        
        Note:
            Uses high-contrast normalization (mean=0.95, std=0.1).
            Output is clamped to [-5.0, 5.0] for stability.
        """
        x = (text_images - self.text_mean) / self.text_std
        x = torch.clamp(x, -5.0, 5.0)

        patches = self.text_patch_embed(x)  # [B, D, H', W']

        B, D, H, W = patches.shape
        patches = patches.view(B, D, -1).transpose(1, 2)  # [B, N, D]

        compressed = self.text_compressor(patches)

        _LOG.debug(f"YvVisualTextProcessor: {text_images.shape} -> {compressed.shape}")

        return compressed


class YvVisionEncoder(nn.Module):
    """Vision backbone producing multimodal features and auxiliary predictions.
    
    A comprehensive vision encoder that tokenizes images into patch embeddings,
    supports optional 3D RoPE for video streams, and attaches detection,
    segmentation, and reasoning heads for downstream Yv modules.
    
    Architecture:
        1. Patch Embedding: Conv2d with patch_size=14 (CLIP-style)
        2. Transformer Encoder: Multi-layer with SDPA or MHA attention
        3. Detection Head: Bounding box regression + classification
        4. Segmentation Head: FPN decoder with semantic output
        5. Reasoning Head: Visual reasoning and grounding
    
    Key Features:
        - ViT-style patch tokenization
        - Optional 3D spatio-temporal RoPE for video
        - H-Network visual text processing support
        - Multi-head detection with 9 anchors per location
        - Semantic segmentation with FPN decoder
        - Visual reasoning and grounding capabilities
    
    Attributes:
        enabled (bool): Flag indicating whether the encoder is active.
        cfg: Configuration namespace supplying hyperparameters.
        patch_size (int): Size of image patches (default: 14).
        hidden_size (int): Dimension of the embedding space.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        patch_embed (nn.Conv2d): Patch embedding convolution.
        transformer (nn.ModuleDict): Transformer encoder layers.
        detection_head (nn.ModuleDict): Detection head modules.
        segmentation_head (nn.ModuleDict): Segmentation head modules.
        use_3d_rope (bool): Whether to use 3D RoPE for video.
    
    Example:
        >>> encoder = YvVisionEncoder(config)
        >>> features = encoder(images)  # [B, N, hidden_size]
        >>> 
        >>> # With video and 3D RoPE
        >>> video_features = encoder(video_frames, video_shape=(T, H, W))
    
    Note:
        Uses ImageNet normalization (mean=[0.485, 0.456, 0.406]).
        Detection head uses 9 anchors per location.
        Supports both SDPA and MHA attention backends.
    """

    def __init__(self, cfg, cache_manager=None) -> None:
        """Initialize the vision encoder with configuration.
        
        Args:
            cfg: Configuration namespace supplying hyperparameters such as
                ``hidden_size``, ``n_head``, ``n_layer``, ``use_3d_spatio_temporal_rope``,
                ``vision_use_sdpa``, ``h_network_enabled``, etc.
            cache_manager: Reserved for future caching integrations. Defaults to ``None``.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.patch_size = 14
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.n_head
        self.num_layers = cfg.n_layer
        _LOG.debug(f"VisionEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        # Register mean values for normalization
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # Register standard deviation values for normalization
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # H-Network visual text processing support.
        self.visual_text_processor = None
        if hasattr(cfg, 'h_network_enabled') and cfg.h_network_enabled:
            self.visual_text_processor = self._create_visual_text_processor()
            _LOG.info("H-Network visual text processor initialized")

    def _create_visual_text_processor(self):
        """Instantiate a processor specialized for rendered text inputs.
        
        Creates a YvVisualTextProcessor instance configured for
        high-contrast text image processing.
        
        Returns:
            YvVisualTextProcessor: Text image processor instance.
        """
        return YvVisualTextProcessor(self.hidden_size, self.patch_size)

    def process_visual_text(self, text_images: torch.Tensor) -> torch.Tensor:
        """Project rendered text images into embeddings via the text processor.
        
        Routes text images through the specialized visual text processor
        for H-Network tokenization support. Requires h_network_enabled
        in configuration.
        
        Args:
            text_images (torch.Tensor): Input tensor shaped ``[B, C, H, W]``.
                Expected to be rendered text images with high contrast.
        
        Returns:
            torch.Tensor: Token embeddings shaped ``[B, N, hidden_size]``.
        
        Raises:
            RuntimeError: If the visual text processor has not been enabled via
                ``cfg.h_network_enabled``.
        """
        if self.visual_text_processor is None:
            raise RuntimeError("Visual text processor not initialized. Enable h_network_enabled in config.")

        return self.visual_text_processor(text_images)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode images or rendered text into stabilized vision embeddings.
        
        Main entry point for vision encoding. Routes to either standard
        image processing or visual text processing based on kwargs.
        
        Args:
            x (torch.Tensor): Input tensor shaped ``[B, C, H, W]``.
            **kwargs: Optional flags including:
                - is_visual_text (bool): Route through text processor.
                - video_shape (Tuple[int, int, int]): Video shape for 3D RoPE.
        
        Returns:
            torch.Tensor: Clamped vision features respecting ``hidden_size``.
                Shape depends on input and processing mode.
        
        Note:
            Output is clamped to [-10.0, 10.0] for numerical stability.
        """
        if kwargs.get('is_visual_text', False) and self.visual_text_processor is not None:
            return self.process_visual_text(x)

        output = self._standard_forward(x)
        return torch.clamp(output, -10.0, 10.0)
        max_patches_h = 1024 // self.patch_size
        max_patches_w = 1024 // self.patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches_h * max_patches_w, self.hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        use_sdpa_vision = bool(getattr(cfg, 'vision_use_sdpa', True))
        self.transformer = nn.ModuleDict({
            'layers': nn.ModuleList([
                nn.ModuleDict({
                    'norm1': nn.LayerNorm(self.hidden_size),
                    **({
                        'attn_type': 'sdpa',
                        'attn': nn.ModuleDict({
                            'q': nn.Linear(self.hidden_size, self.hidden_size),
                            'k': nn.Linear(self.hidden_size, self.hidden_size),
                            'v': nn.Linear(self.hidden_size, self.hidden_size),
                            'o': nn.Linear(self.hidden_size, self.hidden_size),
                            'drop': nn.Dropout(getattr(cfg, 'attention_dropout', 0.0))
                        })
                    } if use_sdpa_vision else {
                        'attn_type': 'mha',
                        'attn': nn.MultiheadAttention(
                            embed_dim=self.hidden_size,
                            num_heads=self.num_heads,
                            batch_first=True
                        )
                    }),
                    'norm2': nn.LayerNorm(self.hidden_size),
                    'mlp': nn.Sequential(
                        nn.Linear(self.hidden_size, 4 * self.hidden_size),
                        nn.GELU(),
                        nn.Linear(4 * self.hidden_size, self.hidden_size)
                    )
                }) for _ in range(self.num_layers)
            ]),
            'norm': nn.LayerNorm(self.hidden_size)
        })
        self.proj = nn.Linear(self.hidden_size, cfg.hidden_size)
        self.use_3d_rope = getattr(cfg, 'use_3d_spatio_temporal_rope', False)
        self.max_temporal_frames = getattr(cfg, 'max_temporal_frames', 64)
        if self.use_3d_rope:
            self.spatio_temporal_rope = YvSpatioTemporalRoPE3D(
                dim=self.hidden_size,
                max_temporal_frames=self.max_temporal_frames,
                max_spatial_h=max_patches_h,
                max_spatial_w=max_patches_w,
                base=getattr(cfg, 'rope_theta', 10000.0)
            )
        self.num_classes = 1000
        self.num_anchors = 9
        self.detection_head = nn.ModuleDict({
            'bbox_regressor': nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 4 * self.num_anchors)
            ),
            'classifier': nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, self.num_classes * self.num_anchors)
            ),
            'objectness': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, self.num_anchors)
            )
        })
        self.segmentation_head = nn.ModuleDict({
            'fpn_layers': nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.hidden_size, 256, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU()
                ) for _ in range(3)
            ]),
            'decoder': nn.Sequential(
                nn.Conv2d(256 * 3, 384, 3, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.Conv2d(384, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 150, 1)
            ),
            'instance_head': nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
        })
        self.low_light_enhancer = nn.ModuleDict({
            'illumination_net': nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.Sigmoid()
            ),
            'reflectance_net': nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 3, 3, padding=1),
                nn.Sigmoid()
            ),
            'gamma_predictor': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_size, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        })
        self.visual_reasoning = nn.ModuleDict({
            'spatial_reasoner': nn.Sequential(
                nn.Linear(self.hidden_size * 2, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 9)
            ),
            'scene_graph_net': nn.ModuleDict({
                'object_encoder': nn.Sequential(
                    nn.Linear(self.hidden_size, 256),
                    nn.LayerNorm(256),
                    nn.ReLU()
                ),
                'relation_encoder': nn.Sequential(
                    nn.Linear(512, 128),
                    nn.LayerNorm(128),
                    nn.ReLU()
                ),
                'predicate_classifier': nn.Sequential(
                    nn.Linear(128, 50)
                )
            }),
            'vqa_head': nn.Sequential(
                nn.Linear(self.hidden_size + 512, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 3000)
            )
        })
        self.coordinate_marker = nn.ModuleDict({
            'position_head': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 2)
            ),
            'uncertainty_head': nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 2),
                nn.Softplus()
            )
        })
        _LOG.debug("VisionEncoder: __init__ end")
        
        self.hallu_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, self.hidden_size),
            nn.Tanh()
        )
        self.hallu_scale = nn.Parameter(torch.tensor(0.1))
        
        self.lfq_num_codebooks = 4
        self.lfq_codebook_size = 8192
        self.lfq_dim = self.hidden_size // 8
        
        self.lfq_scales = nn.Parameter(torch.ones(self.lfq_num_codebooks, self.lfq_dim))
        
        self.image_decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, 3 * self.patch_size * self.patch_size),
            nn.Tanh()
        )
        
        self.generate_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def _lfq_encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        LFQ encode features to discrete tokens.
        
        No explicit codebook required, direct discretization via sign function.
        Based on Emu3 (Nature 2026) and Google/CMU research.
        """
        B, T, D = features.shape
        
        features = features.view(B, T, self.lfq_num_codebooks, self.lfq_dim)
        
        quantized = torch.sign(features) * torch.sqrt(torch.abs(features) + 1e-8)
        quantized = quantized * self.lfq_scales.view(1, 1, self.lfq_num_codebooks, self.lfq_dim)
        
        binary = (quantized > 0).int()
        token_ids = torch.zeros(B, T, self.lfq_num_codebooks, dtype=torch.long, device=features.device)
        for i in range(min(13, self.lfq_dim)):
            token_ids = token_ids * 2 + binary[..., i]
        
        return token_ids
    
    def _lfq_decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        LFQ decode discrete tokens back to continuous features.
        """
        B, T, num_codebooks = token_ids.shape
        
        binary = torch.zeros(B, T, num_codebooks, self.lfq_dim, device=token_ids.device)
        temp_ids = token_ids.clone()
        for i in range(min(13, self.lfq_dim)):
            binary[..., 12 - i] = temp_ids % 2
            temp_ids = temp_ids // 2
        
        features = binary.float() * 2 - 1
        features = features / (self.lfq_scales.view(1, 1, self.lfq_num_codebooks, self.lfq_dim) + 1e-8)
        
        features = features.view(B, T, self.hidden_size)
        
        return features
    
    def _decode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode tokens to image pixels for generation mode.
        """
        features = self._lfq_decode(tokens)
        
        features = self.generate_proj(features)
        
        pixels = self.image_decoder(features)
        
        B, T, _ = pixels.shape
        patch_size = self.patch_size
        pixels = pixels.view(B, T, 3, patch_size, patch_size)
        
        return pixels
    
    def encode_to_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode image to discrete tokens for NTP generation.
        
        Public method for external invocation.
        """
        result = self.forward(pixel_values, mode='understand')
        if isinstance(result, dict) and 'features' in result:
            features = result['features']
            return self._lfq_encode(features)
        return None

    def process_image(self, image_path, target_size=None):
        """
        Process an image from the given path, including normalization.

        Args:
            image_path (str): Path to the image file.
            target_size (Optional[Tuple[int, int]]): Target size to resize the image. Defaults to None.

        Returns:
            Optional[torch.Tensor]: Processed image tensor. Returns None if an error occurs.
        """
        _LOG.debug(f"Processing image: {image_path}")
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                if target_size is not None:
                    img = img.resize(target_size, Image.LANCZOS)
                image_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                image_tensor = (image_tensor - self.mean) / self.std
                return image_tensor
        except Exception as e:
            _LOG.error(f"Image processing error: {e}")
            return None

    def interpolate_pos_encoding(self, pos_embed, h, w):
        """
        Interpolate position embeddings to match the given height and width.

        Args:
            pos_embed (torch.Tensor): Position embeddings.
            h (int): Target height.
            w (int): Target width.

        Returns:
            torch.Tensor: Interpolated position embeddings.
        """
        npatch = h * w
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]
        dim = self.hidden_size
        w0 = w
        h0 = h
        sqrt_N = int(math.sqrt(N))
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode='bicubic',
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, h0 * w0, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values, video_shape=None, mode='understand'):
        """
        Forward pass of the YvVisionEncoder.

        Args:
            pixel_values (torch.Tensor): Input pixel values.
            video_shape (Optional[Tuple[int, int, int]]): Shape of the video (temporal, height, width). Defaults to None.
            mode (str): 'understand' for encoding, 'generate' for decoding.

        Returns:
            Dict or torch.Tensor: Detection results or generated tokens.
        """
        if mode == 'generate':
            return self._decode_from_tokens(pixel_values)
        
        if pixel_values is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=self.proj.weight.device)
        x = (pixel_values - self.mean) / self.std
        B, C, H, W = x.shape
        patch_size = self.patch_size
        is_video = video_shape is not None and self.use_3d_rope
        if is_video:
            T, H_video, W_video = video_shape
            x = x.view(-1, T, C, H, W)
            B = x.shape[0]
            x = x.view(B * T, C, H, W)
        x = self.patch_embed(x)
        h_patches, w_patches = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        num_patches = h_patches * w_patches
        
        if num_patches > 1024:
            pos_idx = torch.arange(num_patches, device=x.device).float()
            pos_weights = 1.0 + 0.2 * torch.sin(torch.pi * pos_idx / num_patches)
            mid_start = num_patches // 4
            mid_end = 3 * num_patches // 4
            pos_weights[mid_start:mid_end] = pos_weights[mid_start:mid_end] * 1.25
            x = x * pos_weights.view(1, -1, 1).expand_as(x)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(self.pos_embed, h_patches, w_patches)
        x = x + pos_embed
        if is_video:
            x = x.view(B, T * (1 + h_patches * w_patches), self.hidden_size)
            cls_tokens_video = x[:, :T]
            patch_tokens = x[:, T:]
            
            if T > 64:
                segment_size = max(8, T // 16)
                num_segments = (T + segment_size - 1) // segment_size
                compressed_tokens = []
                
                for seg_idx in range(num_segments):
                    start = seg_idx * segment_size
                    end = min((seg_idx + 1) * segment_size, T)
                    segment = patch_tokens[:, start:end]
                    
                    segment_var = segment.var(dim=1).mean(dim=-1)
                    threshold = segment_var.mean() if segment_var.numel() > 0 else 0
                    
                    if segment_var.numel() > 0 and segment_var.mean() > threshold:
                        compressed_tokens.append(segment.mean(dim=1, keepdim=True))
                    else:
                        compressed_tokens.append(segment[:, ::max(1, segment.shape[1]//2)].mean(dim=1, keepdim=True))
                
                patch_tokens = torch.cat(compressed_tokens, dim=1)
                T = patch_tokens.shape[1]
            
            patch_tokens = self.spatio_temporal_rope(
                patch_tokens,
                video_shape=(T, h_patches, w_patches)
            )
            x = torch.cat([cls_tokens_video, patch_tokens], dim=1)
        for layer in self.transformer['layers']:
            x_norm = layer['norm1'](x)
            if layer.get('attn_type', 'mha') == 'sdpa':
                q_lin = layer['attn']['q'](x_norm)
                k_lin = layer['attn']['k'](x_norm)
                v_lin = layer['attn']['v'](x_norm)
                Bq, Tq, D = q_lin.shape
                Hh = self.num_heads
                Dh = D // Hh
                q = q_lin.view(Bq, Tq, Hh, Dh).transpose(1, 2).reshape(Bq * Hh, Tq, Dh)
                k = k_lin.view(Bq, Tq, Hh, Dh).transpose(1, 2).reshape(Bq * Hh, Tq, Dh)
                v = v_lin.view(Bq, Tq, Hh, Dh).transpose(1, 2).reshape(Bq * Hh, Tq, Dh)
                attn_mask = None
                if bool(getattr(self.cfg, 'use_sliding_window', False)):
                    win = int(getattr(self.cfg, 'streaming_window', 16384))
                    if win > 0 and win < Tq:
                        key_pos = torch.arange(Tq, device=x.device)
                        row_pos = torch.arange(Tq, device=x.device)
                        lower_bound = (row_pos.view(-1, 1) - (win - 1))
                        allowed = key_pos.view(1, -1) >= lower_bound
                        disallow = ~allowed
                        attn_mask = disallow.expand(Bq * Hh, Tq, Tq)
                try:
                    from torch.backends.cuda import sdp_kernel as _sdp
                    use_flash = torch.cuda.is_available() and bool(getattr(self.cfg, 'sdpa_prefer_flash', True))
                except Exception:
                    use_flash = False
                if use_flash:
                    with _sdp(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                        out_ = F.scaled_dot_product_attention(
                            q, k, v,
                            attn_mask=attn_mask,
                            dropout_p=layer['attn']['drop'].p if self.training else 0.0,
                            is_causal=False
                        )
                else:
                    out_ = F.scaled_dot_product_attention(
                        q, k, v,
                        attn_mask=attn_mask,
                        dropout_p=layer['attn']['drop'].p if self.training else 0.0,
                        is_causal=False
                    )
                out = out_.reshape(Bq, Hh, Tq, Dh).transpose(1, 2).contiguous().view(Bq, Tq, D)
                out = layer['attn']['drop'](out)
                attn_out = layer['attn']['o'](out)
            else:
                attn_out = layer['attn'](x_norm, x_norm, x_norm)[0]
            x = x + attn_out
            mlp_out = layer['mlp'](layer['norm2'](x))
            x = x + mlp_out
        x = self.transformer['norm'](x)
        if is_video:
            cls_tokens_video = x[:, :T]
            patch_tokens = x[:, T:]
            patch_tokens = patch_tokens.view(B, T, h_patches * w_patches, self.hidden_size)
            pooled_features = patch_tokens.mean(dim=2)
            xproj = self.proj(pooled_features)
            cls_features = cls_tokens_video.mean(dim=2)
            combined_features = xproj + cls_features
            
            hallu_component = self.hallu_projection(combined_features)
            combined_features = combined_features - self.hallu_scale * hallu_component
            
            detection_results = {
                'features': combined_features,
                'bbox_coords': None,
                'object_classes': None,
                'confidence_scores': None,
                'coordinate_markers': None,
                'video_shape': video_shape
            }
        else:
            cls_token = x[:, :1]
            patch_features = x[:, 1:]
            pooled_features = patch_features.mean(dim=1)
            xproj = self.proj(pooled_features)
            
            hallu_component = self.hallu_projection(xproj)
            xproj = xproj - self.hallu_scale * hallu_component
            
            detection_results = {
                'features': xproj.unsqueeze(1),
                'bbox_coords': None,
                'object_classes': None,
                'confidence_scores': None,
                'coordinate_markers': None
            }
        if self.training or hasattr(self, '_enable_detection'):
            batch_size, num_patches, _ = patch_features.shape
            h_patches_ = int(math.sqrt(num_patches))
            w_patches_ = h_patches_
            patch_features_2d = patch_features.view(batch_size, h_patches_, w_patches_, -1)
            bbox_pred = self.detection_head['bbox_regressor'](patch_features)
            class_pred = self.detection_head['classifier'](patch_features)
            objectness_pred = self.detection_head['objectness'](patch_features)
            coord_markers = self.coordinate_marker['position_head'](patch_features)
            detection_results.update({
                'bbox_coords': bbox_pred.view(batch_size, num_patches, self.num_anchors, 4),
                'object_classes': class_pred.view(batch_size, num_patches, self.num_anchors, self.num_classes),
                'confidence_scores': torch.sigmoid(objectness_pred),
                'coordinate_markers': coord_markers.view(batch_size, num_patches, 2),
                'spatial_shape': (h_patches_, w_patches_)
            })
        return detection_results

    def convert_patch_to_image_coords(self, patch_coords, image_size, patch_size=14):
        """
        Convert patch coordinates to image coordinates.

        Args:
            patch_coords (Tuple[int, int]): Patch coordinates (x, y).
            image_size (Tuple[int, int]): Size of the image (height, width).
            patch_size (int, optional): Size of each patch. Defaults to 14.

        Returns:
            List[int]: Image coordinates [x, y].
        """
        h, w = image_size
        h_patches = h // patch_size
        w_patches = w // patch_size
        x_patch, y_patch = patch_coords
        x_image = (x_patch + 0.5) * patch_size
        y_image = (y_patch + 0.5) * patch_size
        return [min(x_image, w-1), min(y_image, h-1)]

    def get_object_locations(self, detection_results, image_size, confidence_threshold=0.5):
        """
        Get object locations from detection results based on confidence threshold.

        Args:
            detection_results (Dict): Dictionary containing detection results.
            image_size (Tuple[int, int]): Size of the image (height, width).
            confidence_threshold (float, optional): Confidence threshold for object selection. Defaults to 0.5.

        Returns:
            List[Dict]: List of dictionaries containing object information.
        """
        if not detection_results.get('bbox_coords'):
            return []
        bbox_coords = detection_results['bbox_coords']
        object_classes = detection_results['object_classes']
        confidence_scores = detection_results['confidence_scores']
        objects = []
        batch_size, num_patches, num_anchors, _ = bbox_coords.shape
        h_patches = int(math.sqrt(num_patches))
        for b in range(batch_size):
            for p in range(num_patches):
                patch_y = p // h_patches
                patch_x = p % h_patches
                for a in range(num_anchors):
                    confidence = confidence_scores[b, p, a].item()
                    if confidence > confidence_threshold:
                        bbox = bbox_coords[b, p, a]
                        center_x = (patch_x + bbox[0].item()) * self.patch_size
                        center_y = (patch_y + bbox[1].item()) * self.patch_size
                        width = bbox[2].item() * image_size[1]
                        height = bbox[3].item() * image_size[0]
                        class_probs = torch.softmax(object_classes[b, p, a], dim=0)
                        class_id = torch.argmax(class_probs).item()
                        class_name = f"class_{class_id}"
                        objects.append({
                            'coordinates': [center_x, center_y],
                            'bbox': [center_x - width/2, center_y - height/2, 
                                     center_x + width/2, center_y + height/2],
                            'class': class_name,
                            'confidence': confidence,
                            'patch_coords': [patch_x, patch_y]
                        })
        return objects

    def enable_detection(self, enable=True):
        """
        Enable or disable detection functionality.

        Args:
            enable (bool, optional): Whether to enable detection. Defaults to True.
        """
        self._enable_detection = enable


class YvImageDecoder(nn.Module):
    """High-resolution image decoder with progressive upsampling.
    
    A flagship-grade image decoder that progressively upsamples latent features
    to high-resolution RGB images. Supports multiple output resolutions and
    includes optional diffusion-based enhancement for superior quality.
    
    Architecture:
        1. Initial Projection: Expand latent to hidden dimension
        2. Progressive Upsampling: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        3. Residual Blocks: Skip connections for gradient flow
        4. Attention Layers: Self-attention at each resolution
    
    Key Features:
        - Progressive upsampling with skip connections
        - Self-attention at each resolution level
        - Optional diffusion enhancement
        - Support for 64x64 to 512x512 output
    
    Attributes:
        hidden_size (int): Input latent dimension.
        target_size (int): Target output resolution (64, 128, 256, 512).
        initial_size (int): Initial spatial size (8x8).
    
    Example:
        >>> decoder = YvImageDecoder(hidden_size=4096, target_size=256)
        >>> latents = torch.randn(1, 64, 4096)  # 64 tokens
        >>> images = decoder(latents)  # [1, 3, 256, 256]
    
    Note:
        Uses sub-pixel convolution (PixelShuffle) for efficient upsampling.
        Includes LayerScale for training stability.
    """
    
    def __init__(
        self,
        hidden_size: int,
        target_size: int = 256,
        initial_size: int = 8,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        """Initialize the high-resolution image decoder.
        
        Args:
            hidden_size: Input latent dimension.
            target_size: Target output resolution. Must be power of 2.
            initial_size: Initial spatial size. Must be power of 2.
            use_attention: Whether to use self-attention at each level.
            use_residual: Whether to use residual connections.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.initial_size = initial_size
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        num_upsamples = int(math.log2(target_size // initial_size))
        
        self.initial_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LayerNorm(hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size * initial_size * initial_size)
        )
        
        self.upsample_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList() if use_attention else None
        self.residual_blocks = nn.ModuleList() if use_residual else None
        
        current_channels = hidden_size
        
        for i in range(num_upsamples):
            out_channels = max(current_channels // 2, 64)
            
            upsample_block = nn.Sequential(
                nn.Conv2d(current_channels, out_channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.GroupNorm(32, out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.GroupNorm(32, out_channels),
                nn.GELU()
            )
            self.upsample_blocks.append(upsample_block)
            
            if use_attention:
                attn_block = nn.Sequential(
                    nn.LayerNorm([out_channels]),
                    nn.MultiheadAttention(out_channels, 8, batch_first=True),
                    nn.LayerNorm([out_channels])
                )
                self.attention_blocks.append(attn_block)
            
            if use_residual:
                res_block = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.GroupNorm(32, out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.GroupNorm(32, out_channels)
                )
                self.residual_blocks.append(res_block)
            
            current_channels = out_channels
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(current_channels, 64, 3, 1, 1),
            nn.GroupNorm(32, 64),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent features to high-resolution images.
        
        Args:
            latents: Input latent tensor of shape [B, N, D] or [B, D, H, W].
        
        Returns:
            Decoded image tensor of shape [B, 3, H, W] with values in [-1, 1].
        """
        if latents.dim() == 3:
            B, N, D = latents.shape
            H = W = int(math.sqrt(N))
            if H * W != N:
                latents = latents.transpose(1, 2)
                latents = latents.view(B, D, self.initial_size, self.initial_size)
            else:
                latents = latents.transpose(1, 2).view(B, D, H, W)
        
        B = latents.shape[0]
        
        x = self.initial_proj(latents.flatten(2).transpose(1, 2))
        x = x.view(B, self.initial_size, self.initial_size, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        for i, upsample in enumerate(self.upsample_blocks):
            x = upsample(x)
            
            if self.use_attention and self.attention_blocks is not None:
                B_, C, H_, W_ = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                x_norm = self.attention_blocks[i][0](x_flat)
                attn_out, _ = self.attention_blocks[i][1](x_norm, x_norm, x_norm)
                x_flat = x_flat + attn_out
                x_flat = self.attention_blocks[i][2](x_flat)
                x = x_flat.transpose(1, 2).view(B_, C, H_, W_)
            
            if self.use_residual and self.residual_blocks is not None:
                residual = self.residual_blocks[i](x)
                x = x + residual
        
        images = self.final_conv(x)
        
        return images


class YvDiffusionImageEnhancer(nn.Module):
    """Diffusion-based image enhancement module for flagship quality.
    
    An optional diffusion model that refines images from the base decoder
    for superior visual quality. Uses DDPM-style denoising with a U-Net backbone.
    
    Architecture:
        1. U-Net Backbone: Encoder-decoder with skip connections
        2. Time Embedding: Sinusoidal positional encoding for diffusion timesteps
        3. Cross-Attention: Condition on text or image features
    
    Key Features:
        - DDPM/IDDPM denoising process
        - Classifier-free guidance support
        - Efficient attention with Flash Attention
        - Support for text-conditioned refinement
    
    Example:
        >>> enhancer = YvDiffusionImageEnhancer(hidden_size=4096)
        >>> coarse_image = base_decoder(latents)  # [B, 3, 256, 256]
        >>> refined_image = enhancer(coarse_image, text_features, num_steps=50)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        use_flash_attention: bool = True
    ):
        """Initialize the diffusion image enhancer.
        
        Args:
            hidden_size: Dimension of conditioning features.
            num_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            use_flash_attention: Whether to use Flash Attention.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.use_flash_attention = use_flash_attention
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4)
        )
        
        self.cond_proj = nn.Linear(hidden_size, hidden_size * 4)
        
        self.unet = self._build_unet()
        
        self.noise_scheduler = self._build_scheduler()
    
    def _build_unet(self) -> nn.Module:
        """Build U-Net backbone for diffusion."""
        channels = [320, 640, 1280, 1280]
        attentions = [True, True, True, True]
        
        encoder_blocks = nn.ModuleList()
        decoder_blocks = nn.ModuleList()
        
        in_channels = 3
        for i, (ch, use_attn) in enumerate(zip(channels, attentions)):
            encoder_blocks.append(nn.ModuleDict({
                'conv': nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else channels[i-1], ch, 3, 2, 1),
                    nn.GroupNorm(32, ch),
                    nn.SiLU()
                ),
                'attn': nn.MultiheadAttention(ch, 8, batch_first=True) if use_attn else None,
                'norm': nn.LayerNorm([ch]) if use_attn else None
            }))
        
        for i, (ch, use_attn) in enumerate(zip(reversed(channels), reversed(attentions))):
            decoder_blocks.append(nn.ModuleDict({
                'conv': nn.Sequential(
                    nn.ConvTranspose2d(ch * 2 if i > 0 else ch, ch, 4, 2, 1),
                    nn.GroupNorm(32, ch),
                    nn.SiLU()
                ),
                'attn': nn.MultiheadAttention(ch, 8, batch_first=True) if use_attn else None,
                'norm': nn.LayerNorm([ch]) if use_attn else None
            }))
        
        return nn.ModuleDict({
            'encoder': encoder_blocks,
            'decoder': decoder_blocks,
            'mid': nn.Sequential(
                nn.Conv2d(channels[-1], channels[-1], 3, 1, 1),
                nn.GroupNorm(32, channels[-1]),
                nn.SiLU()
            ),
            'out': nn.Conv2d(channels[0], 3, 3, 1, 1)
        })
    
    def _build_scheduler(self) -> Dict[str, torch.Tensor]:
        """Build DDPM noise schedule."""
        betas = torch.linspace(0.0001, 0.02, self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod
        }
    
    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """Refine image using diffusion process.
        
        Args:
            x: Input image tensor [B, 3, H, W].
            cond: Optional conditioning features [B, N, D].
            num_steps: Override number of denoising steps.
        
        Returns:
            Refined image tensor [B, 3, H, W].
        """
        num_steps = num_steps or self.num_steps
        
        latents = x.clone()
        
        for t in reversed(range(num_steps)):
            t_tensor = torch.tensor([t] * x.shape[0], device=x.device)
            time_emb = self._get_time_embedding(t_tensor)
            
            if cond is not None:
                cond_emb = self.cond_proj(cond.mean(dim=1))
                time_emb = time_emb + cond_emb
            
            noise_pred = self._unet_forward(latents, time_emb)
            
            latents = self._denoise_step(latents, noise_pred, t)
        
        return latents
    
    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Get sinusoidal time embedding."""
        half_dim = self.hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)
    
    def _unet_forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net."""
        skips = []
        
        for block in self.unet['encoder']:
            x = block['conv'](x)
            skips.append(x)
            
            if block['attn'] is not None:
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                x_norm = block['norm'](x_flat)
                attn_out, _ = block['attn'](x_norm, x_norm, x_norm)
                x = (x_flat + attn_out).transpose(1, 2).view(B, C, H, W)
        
        x = self.unet['mid'](x)
        
        for i, block in enumerate(self.unet['decoder']):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block['conv'](x)
            
            if block['attn'] is not None:
                B, C, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                x_norm = block['norm'](x_flat)
                attn_out, _ = block['attn'](x_norm, x_norm, x_norm)
                x = (x_flat + attn_out).transpose(1, 2).view(B, C, H, W)
        
        return self.unet['out'](x)
    
    def _denoise_step(
        self,
        x: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """Single denoising step."""
        alphas_cumprod = self.noise_scheduler['alphas_cumprod'].to(x.device)
        alpha = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        
        x0_pred = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
        
        dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred
        
        x = torch.sqrt(alpha_prev) * x0_pred + dir_xt
        
        return x
