#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import uuid
import json
import time
import torch
import psutil
import weakref
import asyncio
import threading
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from enum import Enum
from utils import DEBUG, ERROR
from datetime import datetime
import torch.nn.functional as F
from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Any, Callable, Union

class SpatioTemporalRoPE3D(nn.Module):
    """
    3D Spatio-Temporal Rotary Position Embedding for video frames.
    Extends 2D RoPE to include temporal dimension for video understanding.
    """
    def __init__(self, dim: int, max_temporal_frames: int = 1024, 
                 max_spatial_h: int = 64, max_spatial_w: int = 64,
                 base: float = 10000.0, device: Optional[torch.device] = None):
        """
        Initialize 3D Spatio-Temporal RoPE.
        
        Args:
            dim (int): Dimension of the embeddings (must be divisible by 3)
            max_temporal_frames (int): Maximum number of temporal frames
            max_spatial_h (int): Maximum spatial height in patches
            max_spatial_w (int): Maximum spatial width in patches
            base (float): Base value for frequency calculation
            device (torch.device): Device to place tensors on
        """
        super().__init__()
        assert dim % 3 == 0, "Dimension must be divisible by 3 for 3D RoPE"
        
        self.dim = dim
        self.dim_per_axis = dim // 3  # Split dimension across 3 axes
        self.max_temporal_frames = max_temporal_frames
        self.max_spatial_h = max_spatial_h
        self.max_spatial_w = max_spatial_w
        self.base = base
        
        # Frequency factors for temporal, height, and width dimensions
        temp_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        h_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        w_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        
        # Register frequency buffers
        self.register_buffer('temp_freq', temp_freq, persistent=False)
        self.register_buffer('h_freq', h_freq, persistent=False)
        self.register_buffer('w_freq', w_freq, persistent=False)
        
        # Cache for 3D position embeddings
        self.register_buffer('cos_cache', None, persistent=False)
        self.register_buffer('sin_cache', None, persistent=False)
        self.register_buffer('max_seq_cached', torch.tensor(0), persistent=False)
    
    def _compute_3d_positions(self, t: int, h: int, w: int) -> torch.Tensor:
        """Compute 3D positions for temporal, height, and width dimensions."""
        # Create 3D coordinate grids
        temp_pos = torch.arange(t, dtype=torch.float32)
        h_pos = torch.arange(h, dtype=torch.float32)
        w_pos = torch.arange(w, dtype=torch.float32)
        
        # Create 3D meshgrid
        temp_grid, h_grid, w_grid = torch.meshgrid(temp_pos, h_pos, w_pos, indexing='ij')
        
        # Flatten to sequence format
        positions = torch.stack([
            temp_grid.flatten(),  # Temporal positions
            h_grid.flatten(),      # Height positions
            w_grid.flatten()       # Width positions
        ], dim=1)  # [T*H*W, 3]
        
        return positions
    
    def _compute_3d_rope(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 3D RoPE embeddings for given positions."""
        device = positions.device
        seq_len = positions.shape[0]
        
        # Split positions by dimension
        temp_pos = positions[:, 0]  # Temporal
        h_pos = positions[:, 1]     # Height
        w_pos = positions[:, 2]     # Width
        
        # Compute frequencies for each dimension
        temp_freqs = torch.outer(temp_pos, self.temp_freq.to(device))
        h_freqs = torch.outer(h_pos, self.h_freq.to(device))
        w_freqs = torch.outer(w_pos, self.w_freq.to(device))
        
        # Combine frequencies
        freqs = torch.zeros(seq_len, self.dim, device=device)
        
        # Fill temporal frequencies
        freqs[:, :self.dim_per_axis] = torch.cat([
            torch.sin(temp_freqs),
            torch.cos(temp_freqs)
        ], dim=1)[:, :self.dim_per_axis]
        
        # Fill height frequencies
        start_h = self.dim_per_axis
        end_h = start_h + self.dim_per_axis
        freqs[:, start_h:end_h] = torch.cat([
            torch.sin(h_freqs),
            torch.cos(h_freqs)
        ], dim=1)[:, :self.dim_per_axis]
        
        # Fill width frequencies
        start_w = 2 * self.dim_per_axis
        end_w = start_w + self.dim_per_axis
        freqs[:, start_w:end_w] = torch.cat([
            torch.sin(w_freqs),
            torch.cos(w_freqs)
        ], dim=1)[:, :self.dim_per_axis]
        
        # Create rotation matrices
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return cos, sin
    
    def forward(self, x: torch.Tensor, video_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Apply 3D Spatio-Temporal RoPE to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T*H*W, dim]
            video_shape (Tuple[int, int, int]): Shape as (T, H, W)
        
        Returns:
            torch.Tensor: Tensor with 3D RoPE applied
        """
        t, h, w = video_shape
        seq_len = t * h * w
        
        # Check if we need to recompute caches
        if self.cos_cache is None or seq_len > self.max_seq_cached:
            positions = self._compute_3d_positions(t, h, w)
            positions = positions.to(x.device)
            
            cos, sin = self._compute_3d_rope(positions)
            
            # Cache the results
            self.cos_cache = cos
            self.sin_cache = sin
            self.max_seq_cached = torch.tensor(seq_len)
        
        # Apply rotation
        x_rotated = x * self.cos_cache[:seq_len] + self._rotate_half(x) * self.sin_cache[:seq_len]
        
        return x_rotated
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the dimensions."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class VisionEncoder(nn.Module):
    """
    Unified Vision Encoder with NaViT native resolution support and SigLIP-style architecture.
    Supports arbitrary image resolutions and aspect ratios with efficient patch processing.
    """
    def __init__(self, cfg, cache_manager=None):
        """
        Initialize the VisionEncoder with NaViT-style native resolution support.

        Args:
            cfg: Configuration object containing parameters such as hidden size, etc.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.patch_size = 14
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.n_head
        self.num_layers = cfg.n_layer
        
        DEBUG(f"VisionEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Image preprocessing: register mean and std for normalization (ImageNet stats)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Patch embedding for any resolution (NaViT-style)
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # 2D positional embedding with native resolution support (up to 1024x1024)
        max_patches_h = 1024 // self.patch_size
        max_patches_w = 1024 // self.patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches_h * max_patches_w, self.hidden_size))
        
        # Classification token (SigLIP-style)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        
        # Enhanced Transformer encoder with pre-norm architecture
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
        
        # Final projection layer
        self.proj = nn.Linear(self.hidden_size, cfg.hidden_size)
        
        # 3D Spatio-Temporal RoPE for video understanding
        self.use_3d_rope = getattr(cfg, 'use_3d_spatio_temporal_rope', False)
        self.max_temporal_frames = getattr(cfg, 'max_temporal_frames', 64)
        
        if self.use_3d_rope:
            self.spatio_temporal_rope = SpatioTemporalRoPE3D(
                dim=self.hidden_size,
                max_temporal_frames=self.max_temporal_frames,
                max_spatial_h=max_patches_h,
                max_spatial_w=max_patches_w,
                base=getattr(cfg, 'rope_theta', 10000.0)
            )
        
        # Object detection and coordinate marking
        self.num_classes = 1000  # Common object classes
        self.num_anchors = 9  # Anchor boxes per location
        
        # Enhanced detection head with gradient safety measures
        self.detection_head = nn.ModuleDict({
            'bbox_regressor': nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.LayerNorm(512),  # Gradient stabilization
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),  # Additional normalization
                nn.SiLU(),
                nn.Linear(256, 4 * self.num_anchors)  # [x, y, w, h] for each anchor
            ),
            'classifier': nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.LayerNorm(512),  # Gradient stabilization
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),  # Additional normalization
                nn.SiLU(),
                nn.Linear(256, self.num_classes * self.num_anchors)
            ),
            'objectness': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.LayerNorm(256),  # Gradient stabilization
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),  # Additional normalization
                nn.SiLU(),
                nn.Linear(128, self.num_anchors)  # Objectness score
            )
        })
        
        # Gradient-Safe Semantic Segmentation Head
        self.segmentation_head = nn.ModuleDict({
            # Simplified multi-scale feature pyramid
            'fpn_layers': nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.hidden_size, 256, 1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),  # Changed from SiLU to ReLU for gradient stability
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU()
                ) for _ in range(3)  # Reduced from 4 to 3 layers
            ]),
            
            # Simplified decoder with gradient clipping
            'decoder': nn.Sequential(
                nn.Conv2d(256 * 3, 384, 3, padding=1),  # Adjusted for 3 pyramid levels
                nn.BatchNorm2d(384),
                nn.ReLU(),
                nn.Conv2d(384, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 150, 1)  # 150 semantic classes
            ),
            
            # Simplified instance segmentation
            'instance_head': nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),  # Reduced complexity
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 1, 1),
                nn.Sigmoid()
            )
        })
        
        # Gradient-Safe Low-light Enhancement Module
        self.low_light_enhancer = nn.ModuleDict({
            # Simplified illumination estimation
            'illumination_net': nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1),
                nn.Sigmoid()  # Illumination map [0, 1]
            ),
            
            # Simplified reflectance estimation
            'reflectance_net': nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1),  # RGB + Illumination
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 3, 3, padding=1),
                nn.Sigmoid()  # Reflectance map
            ),
            
            # Gradient-safe gamma correction
            'gamma_predictor': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_size, 64),  # Reduced from 256
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Gamma value [0, 1] -> map to [0.5, 2.0]
            )
        })
        
        # Gradient-Safe Visual Reasoning Module
        self.visual_reasoning = nn.ModuleDict({
            # Simplified spatial relationship understanding
            'spatial_reasoner': nn.Sequential(
                nn.Linear(self.hidden_size * 2, 256),  # Pairwise features
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 9)  # 9 spatial relations
            ),
            
            # Simplified scene graph generation
            'scene_graph_net': nn.ModuleDict({
                'object_encoder': nn.Sequential(
                    nn.Linear(self.hidden_size, 256),
                    nn.LayerNorm(256),
                    nn.ReLU()
                ),
                'relation_encoder': nn.Sequential(
                    nn.Linear(512, 128),  # Concatenated object features
                    nn.LayerNorm(128),
                    nn.ReLU()
                ),
                'predicate_classifier': nn.Sequential(
                    nn.Linear(128, 50)  # 50 common predicates
                )
            }),
            
            # Simplified VQA head
            'vqa_head': nn.Sequential(
                nn.Linear(self.hidden_size + 512, 512),  # Visual + Text features
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 3000)  # VQA vocabulary
            )
        })
        
        # Enhanced coordinate marker with uncertainty estimation
        self.coordinate_marker = nn.ModuleDict({
            'position_head': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 2)  # [x_center, y_center] normalized coordinates
            ),
            'uncertainty_head': nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 2),  # Position uncertainty [σx, σy]
                nn.Softplus()  # Ensure positive uncertainty
            )
        })
        
        DEBUG("VisionEncoder: __init__ end")
    
    def process_image(self, image_path, target_size=None):
        """
        Process an image from the given path with native resolution support.
        
        Args:
            image_path (str): Path to the image file.
            target_size (tuple, optional): Target size (H, W) for resizing. If None, use native resolution.
            
        Returns:
            torch.Tensor: Processed image tensor, or None if an error occurs.
        """
        DEBUG(f"Processing image: {image_path}")
        try:
            # Use context manager to ensure file handle is properly closed
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                
                # Native resolution processing - no forced resizing
                if target_size is not None:
                    img = img.resize(target_size, Image.LANCZOS)
                
                image_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                image_tensor = (image_tensor - self.mean) / self.std
                return image_tensor
        except Exception as e:
            ERROR(f"Image processing error: {e}")
            return None

    def interpolate_pos_encoding(self, pos_embed, h, w):
        """Interpolate positional encoding for arbitrary image sizes (NaViT-style)."""
        npatch = h * w
        N = pos_embed.shape[1] - 1  # Remove cls token
        
        if npatch == N:
            return pos_embed
        
        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]
        
        dim = self.hidden_size
        w0 = w
        h0 = h
        
        # Reshape and interpolate
        sqrt_N = int(math.sqrt(N))
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode='bicubic',
            align_corners=False
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, h0 * w0, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values, video_shape=None):
        """
        Forward pass of the unified VisionEncoder with NaViT native resolution support and SigLIP-style processing.
        Supports both 2D images and 3D video frames with spatio-temporal encoding.
        
        Args:
            pixel_values (torch.Tensor): Input image/video pixel values.
                - For images: shape (B, C, H, W)
                - For videos: shape (B*T, C, H, W) when video_shape is provided
            video_shape (Tuple[int, int, int], optional): Video shape as (T, H, W) for 3D RoPE.
                If provided, enables 3D spatio-temporal encoding for video frames.
            
        Returns:
            Dict: Encoded features with structure:
                - 'features': Tensor of shape (B, 1, hidden_size) for images or (B, T, hidden_size) for videos
                - Additional detection outputs when training
        """
        if pixel_values is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=self.proj.weight.device)
        
        # Image preprocessing: normalize pixel values (SigLIP-style)
        x = (pixel_values - self.mean) / self.std
        
        # Native resolution processing (NaViT-style)
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        # Handle video input for 3D spatio-temporal encoding
        is_video = video_shape is not None and self.use_3d_rope
        if is_video:
            T, H_video, W_video = video_shape
            # Reshape from (B*T, C, H, W) to (B, T, C, H, W) then flatten batch and temporal dimensions
            x = x.view(-1, T, C, H, W)
            B = x.shape[0]  # Actual batch size
            x = x.view(B * T, C, H, W)  # Process all frames at once
        
        # Dynamic patch embedding for any resolution
        x = self.patch_embed(x)  # (B, hidden_size, H//patch_size, W//patch_size)
        h_patches, w_patches = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # Add classification token (SigLIP-style)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Dynamic position embedding interpolation (NaViT-style)
        pos_embed = self.interpolate_pos_encoding(self.pos_embed, h_patches, w_patches)
        x = x + pos_embed
        
        # Apply 3D Spatio-Temporal RoPE for video frames
        if is_video:
            # Reshape for 3D RoPE: (B*T, 1+num_patches, hidden_size) -> (B, T*(1+num_patches), hidden_size)
            x = x.view(B, T * (1 + h_patches * w_patches), self.hidden_size)
            
            # Apply 3D spatio-temporal encoding to patch tokens (skip CLS tokens)
            cls_tokens_video = x[:, :T]  # CLS tokens for each frame
            patch_tokens = x[:, T:]      # All patch tokens
            
            # Apply 3D RoPE to patch tokens
            patch_tokens = self.spatio_temporal_rope(
                patch_tokens, 
                video_shape=(T, h_patches, w_patches)
            )
            
            # Reconstruct sequence
            x = torch.cat([cls_tokens_video, patch_tokens], dim=1)
        
        # Enhanced Transformer encoder (SigLIP + NaViT fusion)
        for layer in self.transformer['layers']:
            # Pre-norm self-attention
            x_norm = layer['norm1'](x)
            if layer.get('attn_type', 'mha') == 'sdpa':
                # SDPA/Flash backend if available, non-causal encoder attention
                q_lin = layer['attn']['q'](x_norm)
                k_lin = layer['attn']['k'](x_norm)
                v_lin = layer['attn']['v'](x_norm)
                B, T, D = q_lin.shape
                H = self.num_heads
                Dh = D // H
                q = q_lin.view(B, T, H, Dh).transpose(1, 2).reshape(B * H, T, Dh)
                k = k_lin.view(B, T, H, Dh).transpose(1, 2).reshape(B * H, T, Dh)
                v = v_lin.view(B, T, H, Dh).transpose(1, 2).reshape(B * H, T, Dh)

                attn_mask = None
                if bool(getattr(self.cfg, 'use_sliding_window', False)):
                    win = int(getattr(self.cfg, 'streaming_window', 16384))
                    if win > 0 and win < T:
                        key_pos = torch.arange(T, device=x.device)
                        row_pos = torch.arange(T, device=x.device)
                        lower_bound = (row_pos.view(-1, 1) - (win - 1))
                        allowed = key_pos.view(1, -1) >= lower_bound  # [T,S]
                        disallow = ~allowed  # True means mask
                        attn_mask = disallow.expand(B * H, T, T)

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
                out = out_.reshape(B, H, T, Dh).transpose(1, 2).contiguous().view(B, T, D)
                out = layer['attn']['drop'](out)
                attn_out = layer['attn']['o'](out)
            else:
                attn_out = layer['attn'](x_norm, x_norm, x_norm)[0]

            x = x + attn_out

            # Pre-norm MLP
            mlp_out = layer['mlp'](layer['norm2'](x))
            x = x + mlp_out
        
        # Final normalization
        x = self.transformer['norm'](x)
        
        # Handle video vs image output differently
        if is_video:
            # For video: separate CLS and patch tokens per frame
            cls_tokens_video = x[:, :T]  # [B, T, hidden_size]
            patch_tokens = x[:, T:]      # [B, T*num_patches, hidden_size]
            
            # Reshape patch tokens for pooling
            patch_tokens = patch_tokens.view(B, T, h_patches * w_patches, self.hidden_size)
            
            # Pool features per frame
            pooled_features = patch_tokens.mean(dim=2)  # [B, T, hidden_size]
            x = self.proj(pooled_features)  # [B, T, cfg.hidden_size]
            
            # Use frame-level CLS tokens as additional features
            cls_features = cls_tokens_video.mean(dim=2)  # [B, T, hidden_size]
            combined_features = x + cls_features
            
            detection_results = {
                'features': combined_features,  # [B, T, hidden_size] for video
                'bbox_coords': None,
                'object_classes': None,
                'confidence_scores': None,
                'coordinate_markers': None,
                'video_shape': video_shape
            }
        else:
            # Original image processing
            cls_token = x[:, :1]  # (B, 1, hidden_size)
            patch_features = x[:, 1:]  # (B, num_patches, hidden_size)
            
            # Global average pooling (SigLIP-style) and projection
            pooled_features = patch_features.mean(dim=1)  # Average over all patches
            x = self.proj(pooled_features)
            
            detection_results = {
                'features': x.unsqueeze(1),  # (B, 1, hidden_size) for images
                'bbox_coords': None,
                'object_classes': None,
                'confidence_scores': None,
                'coordinate_markers': None
            }
        
        if self.training or hasattr(self, '_enable_detection'):
            # Generate detection predictions
            batch_size, num_patches, _ = patch_features.shape
            h_patches = int(math.sqrt(num_patches))
            w_patches = h_patches
            
            # Spatial reshape for detection
            patch_features_2d = patch_features.view(batch_size, h_patches, w_patches, -1)
            
            # Detection head forward pass
            bbox_pred = self.detection_head['bbox_regressor'](patch_features)
            class_pred = self.detection_head['classifier'](patch_features)
            objectness_pred = self.detection_head['objectness'](patch_features)
            
            # Coordinate markers for precise location
            coord_markers = self.coordinate_marker(patch_features)
            
            # Process detection results
            detection_results.update({
                'bbox_coords': bbox_pred.view(batch_size, num_patches, self.num_anchors, 4),
                'object_classes': class_pred.view(batch_size, num_patches, self.num_anchors, self.num_classes),
                'confidence_scores': torch.sigmoid(objectness_pred),
                'coordinate_markers': coord_markers.view(batch_size, num_patches, 2),
                'spatial_shape': (h_patches, w_patches)
            })
        
        return detection_results
    
    def convert_patch_to_image_coords(self, patch_coords, image_size, patch_size=14):
        """
        Convert patch coordinates to actual image coordinates.
        
        Args:
            patch_coords: [x_patch, y_patch] Patch grid coordinates.
            image_size: (H, W) Original image size.
            patch_size: Patch size, default is 14.
            
        Returns:
            [x_image, y_image] Actual image coordinates.
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
        Get the coordinates of detected objects.
        
        Args:
            detection_results: Output of VisionEncoder.
            image_size: (H, W) Original image size.
            confidence_threshold: Confidence threshold.
            
        Returns:
            List[Dict]: A list of objects containing coordinates and classes.
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
                        # Get bounding box coordinates [dx, dy, dw, dh]
                        bbox = bbox_coords[b, p, a]
                        
                        # Convert to actual coordinates
                        center_x = (patch_x + bbox[0].item()) * self.patch_size
                        center_y = (patch_y + bbox[1].item()) * self.patch_size
                        width = bbox[2].item() * image_size[1]
                        height = bbox[3].item() * image_size[0]
                        
                        # Get the class
                        class_probs = torch.softmax(object_classes[b, p, a], dim=0)
                        class_id = torch.argmax(class_probs).item()
                        class_name = f"class_{class_id}"  # Should be mapped to class names in actual applications
                        
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
        """Enable/disable coordinate marking functionality."""
        self._enable_detection = enable

class AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.sampling_rate = 16000
        self.n_mels = 128
        self.n_fft = 1024
        self.hop_length = 512
        self.win_length = 1024
        
        DEBUG(f"AudioEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Mel filter bank
        self.mel_filters = self._create_mel_filters()
        
        # Enhanced audio feature extraction with multi-scale processing
        self.conv_layers = nn.ModuleDict({
            # Multi-scale temporal convolution
            'temporal_conv': nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=k, stride=2, padding=k//2),
                    nn.BatchNorm1d(32),
                    nn.SiLU(),
                    nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(64),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool1d(128)
                ) for k in [3, 5, 7, 11]  # Multi-scale kernels
            ]),
            
            # Spectral feature extraction
            'spectral_conv': nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((16, 16))
            )
        })
        
        # Gradient-Safe Speaker Separation Module
        self.speaker_separation = nn.ModuleDict({
            # Simplified speaker encoder
            'speaker_encoder': nn.Sequential(
                nn.Linear(128 * 16 * 16, 256),  # Reduced from 512
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)  # Speaker embedding dimension
            ),
            
            # Simplified voice activity detection
            'vad_network': nn.Sequential(
                nn.Conv1d(128, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 1, kernel_size=1),
                nn.Sigmoid()  # Voice activity probability
            ),
            
            # Reduced speaker mask generation (2 speakers instead of 4)
            'separation_masks': nn.ModuleList([
                nn.Sequential(
                    nn.Linear(128 + 128, 128),  # Audio + Speaker embedding
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.Sigmoid()  # Separation mask for speaker i
                ) for _ in range(2)  # Support up to 2 speakers for stability
            ]),
            
            # Simplified speaker verification
            'speaker_verifier': nn.Sequential(
                nn.Linear(256, 64),  # Concatenated speaker embeddings
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Same speaker probability
            )
        })
        
        # Gradient-Safe Music Understanding Module
        self.music_understanding = nn.ModuleDict({
            # Simplified instrument recognition
            'instrument_classifier': nn.Sequential(
                nn.Linear(128 * 16 * 16, 512),  # Reduced from 1024
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 128)  # 128 instrument classes
            ),
            
            # Simplified genre classification
            'genre_classifier': nn.Sequential(
                nn.Linear(128 * 16 * 16, 256),  # Reduced from 512
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 20)  # 20 major genres
            ),
            
            # Simplified tempo estimation
            'rhythm_analyzer': nn.ModuleDict({
                'tempo_estimator': nn.Sequential(
                    nn.Conv1d(128, 64, kernel_size=7, padding=3),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()  # Normalized tempo [0, 1] -> [60, 200] BPM
                ),
                'beat_tracker': nn.Sequential(
                    nn.Conv1d(128, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(32, 1, kernel_size=1),
                    nn.Sigmoid()  # Beat probability sequence
                )
            })
        })
        
        # Comprehensive Emotion Recognition System
        self.emotion_recognition = nn.ModuleDict({
            # Speech emotion analysis
            'speech_emotion': nn.ModuleDict({
                'prosodic_features': nn.Sequential(
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                    nn.Linear(32, 8)  # Prosodic feature vector
                ),
                'spectral_features': nn.Sequential(
                    nn.Linear(128 * 16 * 16, 512),
                    nn.SiLU(),
                    nn.Linear(512, 256),
                    nn.SiLU(),
                    nn.Linear(256, 32)  # Spectral feature vector
                ),
                'emotion_classifier': nn.Sequential(
                    nn.Linear(40, 128),  # 8 prosodic + 32 spectral
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.LayerNorm(64),
                    nn.SiLU(),
                    nn.Linear(64, 7)  # 7 basic emotions
                )
            }),
            
            # Arousal and valence prediction
            'arousal_valence': nn.Sequential(
                nn.Linear(128 * 16 * 16, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 2),  # Arousal and valence scores
                nn.Tanh()  # [-1, 1] range
            ),
            
            # Emotional intensity estimation
            'intensity_estimator': nn.Sequential(
                nn.Linear(128 * 16 * 16, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Intensity [0, 1]
            )
        })
        
        # Enhanced projection with multi-task learning
        self.proj = nn.ModuleDict({
            'main_proj': nn.Sequential(
                nn.Linear(128 * 16 * 16, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.hidden_size, cfg.hidden_size)
            ),
            
            # Task-specific projections
            'speaker_proj': nn.Linear(128, cfg.hidden_size // 4),
            'music_proj': nn.Linear(128, cfg.hidden_size // 4),
            'emotion_proj': nn.Linear(7 + 2 + 1, cfg.hidden_size // 4),  # emotion + arousal/valence + intensity
            
            # Multi-task fusion
            'task_fusion': nn.Sequential(
                nn.Linear(cfg.hidden_size + 3 * (cfg.hidden_size // 4), cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU()
            )
        })
        
        DEBUG("AudioEncoder: __init__ end")
    
    def _create_mel_filters(self):
        """Create Mel filter bank."""
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sampling_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sampling_rate).astype(int)
        
        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(1, self.n_mels + 1):
            left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]
            for j in range(left, center):
                filters[i-1, j] = (j - left) / (center - left)
            for j in range(center, right):
                filters[i-1, j] = (right - j) / (right - center)
        
        return nn.Parameter(torch.from_numpy(filters).float(), requires_grad=False)
    
    def _stft(self, audio):
        """Short-time Fourier transform."""
        window = torch.hann_window(self.win_length)
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length, 
            win_length=self.win_length, window=window, return_complex=True
        )
        magnitude = torch.abs(stft)
        # Handle mono audio dimension issue
        if magnitude.dim() == 2:
            magnitude = magnitude.unsqueeze(0)
        return magnitude
    
    def _mel_spectrogram(self, audio):
        """Calculate Mel spectrogram."""
        stft = self._stft(audio)
        mel_spec = torch.matmul(self.mel_filters, stft)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
        return log_mel_spec
    
    def process_audio(self, audio_path):
        """
        Process an audio file.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            torch.Tensor: Processed audio features.
        """
        DEBUG(f"Processing audio: {audio_path}")
        try:
            # Load audio using librosa
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            audio = torch.from_numpy(audio).float()
            
            # Calculate Mel spectrogram
            mel_spec = self._mel_spectrogram(audio)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
            
            return mel_spec.unsqueeze(0)
        except Exception as e:
            ERROR(f"Audio processing error: {e}")
            return torch.zeros(1, self.n_mels, 64)
    
    def forward(self, audio_input):
        """
        Forward pass.
        Args:
            audio_input: Can be an audio tensor or a dictionary.
        Returns:
            torch.Tensor: Encoded audio features.
        """
        if audio_input is None:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.cfg.hidden_size, device=device)
        
        # Process input format
        if isinstance(audio_input, dict):
            audio_tensor = audio_input.get('input_values', audio_input.get('audio', None))
        else:
            audio_tensor = audio_input
        
        if audio_tensor is None:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.cfg.hidden_size, device=device)
        
        # Ensure correct shape
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(1)
        
        # Calculate Mel spectrogram
        if audio_tensor.dim() == 1:
            mel_spec = self._mel_spectrogram(audio_tensor)
        else:
            mel_spec = audio_tensor
        
        # Convolution processing
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
        x = self.conv_layers(mel_spec)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        
        return x.unsqueeze(1)

class DocEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.vocab_size = 50000
        self.max_length = 512
        
        DEBUG(f"DocEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Enhanced text encoder with multi-language support
        self.text_encoder = nn.ModuleDict({
            'embedding': nn.Embedding(self.vocab_size, cfg.hidden_size),
            'positional_encoding': nn.Embedding(self.max_length, cfg.hidden_size),
            'layer_norm': nn.LayerNorm(cfg.hidden_size),
            'dropout': nn.Dropout(0.1),
            
            # Multi-language text processing
            'language_detector': nn.Sequential(
                nn.Linear(cfg.hidden_size, 256),
                nn.SiLU(),
                nn.Linear(256, 100)  # 100 language classes
            ),
            
            # Script-aware encoding (Latin, Chinese, Arabic, etc.)
            'script_encoders': nn.ModuleDict({
                'latin': nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size, nhead=cfg.n_head // 4, 
                    dim_feedforward=cfg.hidden_size * 2, batch_first=True
                ),
                'chinese': nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size, nhead=cfg.n_head // 4,
                    dim_feedforward=cfg.hidden_size * 2, batch_first=True
                ),
                'arabic': nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size, nhead=cfg.n_head // 4,
                    dim_feedforward=cfg.hidden_size * 2, batch_first=True
                )
            })
        })
        
        # Revolutionary Layout Encoder with geometric understanding
        self.layout_encoder = nn.ModuleDict({
            # Enhanced spatial encoding with geometric features
            'spatial_encoder': nn.Sequential(
                nn.Linear(8, cfg.hidden_size // 2),  # [x0, y0, x1, y1, w, h, cx, cy]
                nn.LayerNorm(cfg.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(cfg.hidden_size // 2, cfg.hidden_size // 4),
                nn.LayerNorm(cfg.hidden_size // 4),
                nn.SiLU()
            ),
            
            # Reading order prediction
            'reading_order': nn.Sequential(
                nn.Linear(cfg.hidden_size // 4, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 1)  # Reading order score
            ),
            
            # Layout classification
            'layout_classifier': nn.Sequential(
                nn.Linear(cfg.hidden_size // 4, 128),
                nn.SiLU(),
                nn.Linear(128, 15)  # 15 layout types (paragraph, title, list, etc.)
            ),
            
            # Geometric relationship understanding
            'geometric_reasoner': nn.Sequential(
                nn.Linear(cfg.hidden_size // 2, 256),  # Pairwise layout features
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 9)  # 9 geometric relations
            )
        })
        
        # Advanced Table Understanding Module
        self.table_understanding = nn.ModuleDict({
            # Table structure detection
            'structure_detector': nn.ModuleDict({
                'row_detector': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 256),
                    nn.SiLU(),
                    nn.Linear(256, 128),
                    nn.SiLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()  # Row boundary probability
                ),
                'column_detector': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 256),
                    nn.SiLU(),
                    nn.Linear(256, 128),
                    nn.SiLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()  # Column boundary probability
                ),
                'cell_classifier': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 256),
                    nn.SiLU(),
                    nn.Linear(256, 6)  # Header, data, merged, empty, etc.
                )
            }),
            
            # Table content understanding
            'content_analyzer': nn.ModuleDict({
                'data_type_classifier': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 128),
                    nn.SiLU(),
                    nn.Linear(128, 8)  # Number, date, text, currency, etc.
                ),
                'numerical_analyzer': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 64),
                    nn.SiLU(),
                    nn.Linear(64, 4)  # Sum, average, trend, outlier
                ),
                'semantic_encoder': nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size, nhead=cfg.n_head // 4,
                    dim_feedforward=cfg.hidden_size * 2, batch_first=True
                )
            }),
            
            # Table QA and reasoning
            'table_qa': nn.Sequential(
                nn.Linear(cfg.hidden_size * 2, 512),  # Table + Question features
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, cfg.hidden_size)  # Answer representation
            )
        })
        
        # Handwriting Recognition Module
        self.handwriting_recognition = nn.ModuleDict({
            # Stroke sequence modeling
            'stroke_encoder': nn.LSTM(
                input_size=3,  # [x, y, pressure]
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            ),
            
            # Character recognition from strokes
            'char_recognizer': nn.Sequential(
                nn.Linear(256, 512),  # Bidirectional LSTM output
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 10000)  # Large character vocabulary
            ),
            
            # Handwriting style analysis
            'style_analyzer': nn.Sequential(
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 20)  # Writing style features
            ),
            
            # Text line segmentation
            'line_segmenter': nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()  # Line boundary probability
            ),
            
            # Word-level recognition
            'word_recognizer': nn.Sequential(
                nn.Linear(256, 512),
                nn.SiLU(),
                nn.Linear(512, 1000)  # Common word vocabulary
            )
        })
        
        # Enhanced Document-level Feature Fusion
        self.doc_fusion = nn.ModuleDict({
            # Multi-modal attention for text+layout fusion
            'text_layout_attention': nn.MultiheadAttention(
                embed_dim=cfg.hidden_size,
                num_heads=cfg.n_head // 4,
                batch_first=True,
                dropout=0.1
            ),
            
            # Hierarchical document structure modeling
            'hierarchy_encoder': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size,
                    nhead=cfg.n_head // 2,
                    dim_feedforward=cfg.hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=3
            ),
            
            # Document type classification
            'doc_type_classifier': nn.Sequential(
                nn.Linear(cfg.hidden_size, 256),
                nn.SiLU(),
                nn.Linear(256, 20)  # Invoice, receipt, form, etc.
            ),
            
            # Information extraction heads
            'extraction_heads': nn.ModuleDict({
                'entity_extractor': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 256),
                    nn.SiLU(),
                    nn.Linear(256, 50)  # Named entities
                ),
                'key_value_extractor': nn.Sequential(
                    nn.Linear(cfg.hidden_size * 2, 256),
                    nn.SiLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()  # Key-value pair probability
                )
            }),
            
            # Final fusion layer
            'final_fusion': nn.Sequential(
                nn.Linear(cfg.hidden_size + cfg.hidden_size // 4, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU(),
                nn.Dropout(0.1)
            )
        })
        
        # Enhanced final projection with multi-task learning
        self.final_proj = nn.ModuleDict({
            'main_projection': nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU()
            ),
            
            # Task-specific projections
            'table_proj': nn.Linear(cfg.hidden_size, cfg.hidden_size // 4),
            'handwriting_proj': nn.Linear(256, cfg.hidden_size // 4),
            'layout_proj': nn.Linear(cfg.hidden_size // 4, cfg.hidden_size // 4),
            
            # Multi-task integration
            'task_integration': nn.Sequential(
                nn.Linear(cfg.hidden_size + 3 * (cfg.hidden_size // 4), cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU()
            )
        })
        
        DEBUG("DocEncoder: __init__ end")
    
    def _tokenize_text(self, text):
        """Simple text tokenization and encoding."""
        if isinstance(text, str):
            # Simple character-level tokenization
            tokens = [hash(c) % self.vocab_size for c in text[:self.max_length]]
            tokens += [0] * (self.max_length - len(tokens))
            return torch.tensor(tokens)
        return text
    
    def _encode_layout(self, layout):
        """Encode layout information."""
        if layout is None:
            # Default layout: full page
            layout = torch.tensor([[0, 0, 1, 1]])
        
        if layout.dim() == 1:
            layout = layout.unsqueeze(0)
        
        return self.layout_encoder(layout.float())
    
    def forward(self, doc_input):
        """
        Forward pass.
        Args:
            doc_input: Can be a text string, token IDs, or a dictionary.
        Returns:
            torch.Tensor: Encoded document features.
        """
        if doc_input is None:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.cfg.hidden_size, device=device)
        
        # Process input format
        if isinstance(doc_input, dict):
            text_input = doc_input.get('input_ids', doc_input.get('text', None))
            layout_input = doc_input.get('layout', None)
        elif isinstance(doc_input, str):
            text_input = doc_input
            layout_input = None
        else:
            text_input = doc_input
            layout_input = None
        
        if text_input is None:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.cfg.hidden_size, device=device)
        
        # Text encoding
        text_tokens = self._tokenize_text(text_input)
        if text_tokens.dim() == 1:
            text_tokens = text_tokens.unsqueeze(0)
        
        text_features = self.text_encoder(text_tokens)
        text_features = text_features.mean(dim=1)  # Average pooling
        
        # Layout encoding
        layout_features = self._encode_layout(layout_input)
        layout_features = layout_features.mean(dim=1)  # Average pooling
        
        # Fusion of text and layout features
        combined = torch.cat([text_features, layout_features], dim=-1)
        doc_features = self.doc_fusion(combined)
        doc_features = self.final_proj(doc_features)
        
        return doc_features.unsqueeze(1)

class VideoEncoder(nn.Module):
    """
    A video encoder module that processes video frames using a vision encoder and performs temporal pooling.
    """
    def __init__(self, cfg):
        """
        Initialize the VideoEncoder with 3D spatio-temporal encoding support.

        Args:
            cfg: Configuration object containing parameters such as hidden size.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        DEBUG(f"VideoEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Enhanced video encoding with 3D spatio-temporal support
        self.use_3d_rope = getattr(cfg, 'use_3d_spatio_temporal_rope', False)
        self.frame_encoder = VisionEncoder(cfg)
        
        # Enhanced video encoding with 3D spatio-temporal support
        self.use_3d_rope = getattr(cfg, 'use_3d_spatio_temporal_rope', False)
        self.frame_encoder = VisionEncoder(cfg)
        
        if self.use_3d_rope:
            # Revolutionary 3D-aware temporal processing with multi-scale analysis
            self.temporal_processing = nn.ModuleDict({
                # Multi-scale 3D convolutions for temporal feature extraction
                'temporal_conv_3d': nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(cfg.hidden_size, 256, kernel_size=(t, 3, 3), 
                                 stride=(1, 1, 1), padding=(t//2, 1, 1)),
                        nn.BatchNorm3d(256),
                        nn.SiLU(),
                        nn.Conv3d(256, 512, kernel_size=(3, 3, 3), 
                                 stride=(1, 1, 1), padding=(1, 1, 1)),
                        nn.BatchNorm3d(512),
                        nn.SiLU(),
                        nn.AdaptiveAvgPool3d((8, 8, 8))
                    ) for t in [3, 5, 7]  # Multi-temporal scales
                ]),
                
                # Enhanced temporal projection with residual connections
                'temporal_proj': nn.Sequential(
                    nn.Linear(cfg.hidden_size, cfg.hidden_size),
                    nn.LayerNorm(cfg.hidden_size),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(cfg.hidden_size, cfg.hidden_size),
                    nn.LayerNorm(cfg.hidden_size)
                ),
                
                # 3D-aware temporal modeling with attention
                'temporal_conv': nn.Sequential(
                    nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=5, padding=2),
                    nn.BatchNorm1d(cfg.hidden_size),
                    nn.SiLU(),
                    nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1),
                    nn.BatchNorm1d(cfg.hidden_size),
                    nn.SiLU()
                ),
                
                # Multi-head temporal attention with positional encoding
                'temporal_attention': nn.MultiheadAttention(
                    cfg.hidden_size, cfg.n_head, batch_first=True, dropout=0.1
                ),
                
                # Long-range temporal dependency modeling with gradient safety
                'lstm_temporal': nn.LSTM(
                    cfg.hidden_size, cfg.hidden_size, num_layers=1,  # Reduced from 2 to 1
                    batch_first=True, dropout=0.0, bidirectional=False  # Simplified
                ),
                
                # Temporal fusion gate
                'temporal_fusion': nn.Sequential(
                    nn.Linear(cfg.hidden_size * 3, cfg.hidden_size * 2),  # Conv + Attention + LSTM
                    nn.SiLU(),
                    nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
                    nn.Sigmoid()
                )
            })
            
            # Revolutionary Action Recognition Module
            self.action_recognition = nn.ModuleDict({
                # Spatial-temporal feature extractor
                'spatiotemporal_features': nn.Sequential(
                    nn.Conv3d(cfg.hidden_size, 512, kernel_size=(3, 7, 7), 
                             stride=(1, 2, 2), padding=(1, 3, 3)),
                    nn.BatchNorm3d(512),
                    nn.SiLU(),
                    nn.Conv3d(512, 256, kernel_size=(3, 5, 5), 
                             stride=(1, 2, 2), padding=(1, 2, 2)),
                    nn.BatchNorm3d(256),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool3d((8, 4, 4))
                ),
                
                # Motion flow estimation
                'motion_estimator': nn.ModuleDict({
                    'flow_net': nn.Sequential(
                        nn.Conv3d(cfg.hidden_size, 128, kernel_size=(2, 3, 3), 
                                 stride=(1, 1, 1), padding=(0, 1, 1)),
                        nn.SiLU(),
                        nn.Conv3d(128, 64, kernel_size=(1, 3, 3), 
                                 stride=(1, 1, 1), padding=(0, 1, 1)),
                        nn.SiLU(),
                        nn.Conv3d(64, 2, kernel_size=(1, 1, 1))  # Optical flow [dx, dy]
                    ),
                    'motion_magnitude': nn.Sequential(
                        nn.Conv3d(2, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                        nn.SiLU(),
                        nn.Conv3d(32, 1, kernel_size=(1, 1, 1)),
                        nn.Sigmoid()  # Motion intensity
                    )
                }),
                
                # Action classification head
                'action_classifier': nn.Sequential(
                    nn.Linear(256 * 8 * 4 * 4 + 64, 1024),  # Spatiotemporal + motion features
                    nn.SiLU(),
                    nn.Dropout(0.2),
                    nn.Linear(1024, 512),
                    nn.LayerNorm(512),
                    nn.SiLU(),
                    nn.Linear(512, 400)  # 400 action classes (Kinetics-400)
                ),
                
                # Temporal action localization
                'action_localization': nn.Sequential(
                    nn.Conv1d(cfg.hidden_size, 256, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv1d(256, 128, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv1d(128, 3, kernel_size=1)  # Start, end, confidence
                )
            }),
            
            # Advanced Scene Understanding Module
            self.scene_understanding = nn.ModuleDict({
                # Scene classification
                'scene_classifier': nn.Sequential(
                    nn.AdaptiveAvgPool3d((1, 7, 7)),
                    nn.Flatten(),
                    nn.Linear(cfg.hidden_size * 49, 1024),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(1024, 512),
                    nn.LayerNorm(512),
                    nn.SiLU(),
                    nn.Linear(512, 365)  # Places365 scene categories
                ),
                
                # Object tracking across frames
                'object_tracker': nn.ModuleDict({
                    'feature_correlator': nn.Sequential(
                        nn.Linear(cfg.hidden_size * 2, 512),  # Frame t and t+1 features
                        nn.SiLU(),
                        nn.Linear(512, 256),
                        nn.SiLU(),
                        nn.Linear(256, 1),
                        nn.Sigmoid()  # Correlation score
                    ),
                    'trajectory_predictor': nn.LSTM(
                        4, 64, num_layers=2, batch_first=True, dropout=0.1  # [x, y, w, h] -> trajectory
                    ),
                    'occlusion_detector': nn.Sequential(
                        nn.Linear(cfg.hidden_size, 128),
                        nn.SiLU(),
                        nn.Linear(128, 1),
                        nn.Sigmoid()  # Occlusion probability
                    )
                }),
                
                # Temporal event detection
                'event_detector': nn.Sequential(
                    nn.Conv1d(cfg.hidden_size, 256, kernel_size=5, padding=2),
                    nn.SiLU(),
                    nn.Conv1d(256, 128, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv1d(128, 20, kernel_size=1)  # 20 event types
                ),
                
                # Video summarization
                'summarizer': nn.ModuleDict({
                    'importance_scorer': nn.Sequential(
                        nn.Linear(cfg.hidden_size, 256),
                        nn.SiLU(),
                        nn.Linear(256, 1),
                        nn.Sigmoid()  # Frame importance score
                    ),
                    'diversity_encoder': nn.Sequential(
                        nn.Linear(cfg.hidden_size, 128),
                        nn.SiLU(),
                        nn.Linear(128, 64)  # Diversity feature vector
                    )
                })
            })
            
        else:
            # Legacy 2D temporal processing with basic enhancements
            self.temporal_processing = nn.ModuleDict({
                'temporal_proj': nn.Sequential(
                    nn.Linear(cfg.hidden_size, cfg.hidden_size),
                    nn.LayerNorm(cfg.hidden_size),
                    nn.SiLU(),
                    nn.Linear(cfg.hidden_size, cfg.hidden_size)
                ),
                
                'temporal_conv': nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1),
                'temporal_attention': nn.MultiheadAttention(cfg.hidden_size, cfg.n_head, batch_first=True)
            })
        
        DEBUG("VideoEncoder: __init__ end")
    
    def forward(self, video_frames):
        """
        Forward pass of the VideoEncoder with 3D spatio-temporal encoding support.

        Args:
            video_frames (torch.Tensor): Input video frames of shape (B, T, C, H, W).

        Returns:
            torch.Tensor: Encoded video features of shape (B, 1, hidden_size).
        """
        if video_frames is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=self.cfg.device)
        
        batch_size, num_frames, channels, height, width = video_frames.shape
        
        if self.use_3d_rope:
            # Use 3D-aware video processing
            frames_flat = video_frames.view(-1, channels, height, width)
            
            # Pass video shape to enable 3D spatio-temporal encoding
            video_shape = (num_frames, height, width)
            frame_features = self.frame_encoder(frames_flat, video_shape=video_shape)
            
            # Enhanced temporal processing with 3D features
            video_features = frame_features['features']  # [B, T, hidden_size]
            
            # Apply temporal modeling
            video_features = video_features.transpose(1, 2)  # [B, hidden_size, T]
            video_features = self.temporal_conv(video_features)
            video_features = video_features.transpose(1, 2)  # [B, T, hidden_size]
            
            # Temporal attention
            video_features, _ = self.temporal_attention(
                video_features, video_features, video_features
            )
            
            # Global pooling and projection
            video_features = video_features.mean(dim=1)  # [B, hidden_size]
            video_features = self.temporal_proj(video_features)
            
        else:
            # Legacy 2D processing
            frames_flat = video_frames.view(-1, channels, height, width)
            frame_features = self.frame_encoder(frames_flat)
            
            # Reshape back to temporal sequence
            frame_features = frame_features.view(batch_size, num_frames, -1, self.cfg.hidden_size)
            
            # Temporal pooling (simple average)
            video_features = frame_features.mean(dim=2)  # Average across spatial dimensions
            video_features = self.temporal_proj(video_features.mean(dim=1))  # Average across frames
        
        return video_features.unsqueeze(1)

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentAction:
    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""

@dataclass
class AgentObservation:
    modality: str  # "text", "image", "audio", "tool_result"
    content: Any
    metadata: Dict[str, Any]

@dataclass
class AgentMemory:
    observations: List[AgentObservation]
    actions: List[AgentAction]
    reflections: List[str]
    
    def __post_init__(self):
        """Initialize additional attributes for enhanced memory management"""
        self.embeddings: List[torch.Tensor] = []  # Semantic embeddings for retrieval
        self.importance_scores: List[float] = []  # Importance scores for compression
        self.max_memory_size = 1000  # Maximum memory capacity
        self.compression_threshold = 0.7  # Threshold for memory compression
    
    def add_observation(self, observation: AgentObservation):
        self.observations.append(observation)
        # Generate semantic embedding using actual encoder
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(str(observation.content), convert_to_tensor=True)
            
            # Enhanced importance calculation based on content analysis
            content_str = str(observation.content)
            length_factor = min(len(content_str) / 100.0, 1.0)
            unique_words = len(set(content_str.lower().split()))
            total_words = max(len(content_str.split()), 1)
            complexity_factor = unique_words / total_words
            keyword_factor = sum(1 for word in content_str.lower().split() 
                               if word in ['important', 'critical', 'urgent', 'key', 'essential']) * 0.1
            
            importance = min(1.0, (length_factor * 0.3 + complexity_factor * 0.5 + keyword_factor * 0.2))
            
        except Exception:
            # Fallback to structured random embedding
            import hashlib
            content_hash = int(hashlib.md5(str(observation.content).encode()).hexdigest(), 16)
            torch.manual_seed(content_hash % 2147483647)
            embedding = torch.randn(768)
            importance = min(1.0, len(str(observation.content)) / 100.0)
        
        self.embeddings.append(embedding)
        self.importance_scores.append(importance)
        
        # Trigger compression if memory too large
        if len(self.observations) > self.max_memory_size:
            self.compress_memory()
    
    def add_action(self, action: AgentAction):
        self.actions.append(action)
        # Generate embedding for action
        embedding = torch.randn(768)
        self.embeddings.append(embedding)
        importance = action.confidence  # Use action confidence as importance
        self.importance_scores.append(importance)
    
    def add_reflection(self, reflection: str):
        self.reflections.append(reflection)
        # Generate embedding for reflection
        embedding = torch.randn(768)
        self.embeddings.append(embedding)
        importance = min(1.0, len(reflection) / 200.0)
        self.importance_scores.append(importance)
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search through memory using cosine similarity with enhanced relevance"""
        if not self.embeddings:
            return []
        
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(query, convert_to_tensor=True)
        except Exception:
            # Fallback to structured query embedding
            import hashlib
            query_hash = int(hashlib.md5(query.encode()).hexdigest(), 16)
            torch.manual_seed(query_hash % 2147483647)
            query_embedding = torch.randn(768)
        
        # Calculate similarities with enhanced scoring
        similarities = []
        for i, embedding in enumerate(self.embeddings):
            semantic_similarity = torch.cosine_similarity(query_embedding.unsqueeze(0), embedding.unsqueeze(0)).item()
            
            # Boost importance-based relevance
            importance_boost = self.importance_scores[i] * 0.2
            
            # Time-based decay (more recent = higher relevance)
            time_decay = 1.0 - (i / max(len(self.embeddings), 1)) * 0.1
            
            final_score = semantic_similarity + importance_boost + time_decay
            similarities.append((i, final_score))
        
        # Sort by enhanced similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for idx, similarity in similarities[:k]:
            if idx < len(self.observations):
                results.append({
                    "type": "observation",
                    "content": self.observations[idx],
                    "similarity": similarity,
                    "index": idx,
                    "importance": self.importance_scores[idx]
                })
            elif idx < len(self.observations) + len(self.actions):
                action_idx = idx - len(self.observations)
                results.append({
                    "type": "action",
                    "content": self.actions[action_idx],
                    "similarity": similarity,
                    "index": idx,
                    "importance": self.importance_scores[idx]
                })
            else:
                reflection_idx = idx - len(self.observations) - len(self.actions)
                results.append({
                    "type": "reflection",
                    "content": self.reflections[reflection_idx],
                    "similarity": similarity,
                    "index": idx,
                    "importance": self.importance_scores[idx]
                })
        
        return results
    
    def compress_memory(self):
        """Intelligent memory compression based on importance scores"""
        if not self.importance_scores:
            return
        
        # Calculate threshold for low-importance memories
        threshold = sorted(self.importance_scores)[int(len(self.importance_scores) * self.compression_threshold)]
        
        # Identify indices to keep
        keep_indices = [i for i, score in enumerate(self.importance_scores) 
                       if score >= threshold]
        
        # Compress memories
        self.observations = [self.observations[i] for i in keep_indices if i < len(self.observations)]
        self.actions = [self.actions[i] for i in keep_indices 
                       if len(self.observations) <= i < len(self.observations) + len(self.actions)]
        self.reflections = [self.reflections[i] for i in keep_indices 
                         if i >= len(self.observations) + len(self.actions)]
        self.embeddings = [self.embeddings[i] for i in keep_indices]
        self.importance_scores = [self.importance_scores[i] for i in keep_indices]
    
    def get_context_with_retrieval(self, query: str = None, k: int = 5) -> Dict[str, Any]:
        """Get context with semantic retrieval and compression"""
        if query:
            relevant_memories = self.semantic_search(query, k)
            return {
                "relevant_memories": relevant_memories,
                "total_count": len(self.observations) + len(self.actions) + len(self.reflections)
            }
        else:
            return self.get_recent_context(k)
    
    def get_recent_context(self, k: int = 5) -> Dict[str, List]:
        """Get recent context with enhanced information"""
        return {
            "recent_observations": self.observations[-k:],
            "recent_actions": self.actions[-k:],
            "recent_reflections": self.reflections[-k:],
            "total_count": len(self.observations) + len(self.actions) + len(self.reflections),
            "memory_summary": {
                "observations": len(self.observations),
                "actions": len(self.actions),
                "reflections": len(self.reflections)
            }
        }

class MCPMessageType(Enum):
    """MCP message types for agent communication"""
    OBSERVATION = "observation"
    ACTION = "action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    CAPABILITY_REGISTER = "capability_register"
    HEARTBEAT = "heartbeat"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"

@dataclass
class GenerationCondition:
    """Condition for multi-modal generation"""
    text_prompt: str = ""
    emotion_vector: Optional[torch.Tensor] = None
    style_params: Optional[Dict[str, float]] = None
    generation_params: Optional[Dict[str, Any]] = None

@dataclass
class MCPMessage:
    """Standardized MCP message format"""
    message_type: str
    agent_id: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: str = ""
    priority: str = "normal"

class MCPToolRegistry:
    """Registry for managing MCP tools and capabilities"""
    
    def __init__(self, agent_id: str, message_handler: Callable):
        self.agent_id = agent_id
        self.message_handler = message_handler
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, Dict[str, Any]] = {}
    
    async def register_capability(self, name: str, description: str, 
                                  parameters: Dict[str, Any], handler: Callable):
        """Register a new capability"""
        self.capabilities[name] = {
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
    
    async def handle_tool_call(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool call"""
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            return await tool["handler"](**kwargs)
        else:
            raise ValueError(f"Tool {tool_name} not found")

class TreeSearchReasoner:
    """Tree search reasoning module for advanced planning"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = 5
        self.max_width = 3
    
    async def search(self, problem: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform tree search for complex reasoning"""
        # Simplified tree search implementation
        return [{"solution": "tree_search_result", "confidence": 0.8}]

class PiscesReasoner(nn.Module):
    """Enhanced Pisces L1 reasoning module with multi-step CoT and memory integration"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        
        # Optimized multi-step CoT reasoning layers (reduced redundancy)
        # Merged similar reasoning layers and added dynamic depth control
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.n_head,
                dim_feedforward=cfg.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)  # Optimized: reduced from 4 to 3 layers
        ])
        
        # Dynamic reasoning depth controller
        self.depth_controller = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Enhanced reasoning components
        self.thinking_head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.difficulty_head = nn.Linear(cfg.hidden_size, 5)  # Enhanced: 5 difficulty levels
        self.reflection_head = nn.Linear(cfg.hidden_size, 4)  # Enhanced: 4 reflection types
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)
        
        # Memory integration for multi-turn context
        self.memory_key_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.memory_value_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=cfg.hidden_size,
            num_heads=cfg.n_head,
            dropout=0.1,
            batch_first=True
        )
        
        # Hierarchical reflection components
        self.error_analyzer = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size // 2, 3)  # error_type, severity, confidence
        )
        
        self.correction_head = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(cfg.hidden_size)
    
    def _calculate_problem_complexity(self, hidden_states):
        """
        Calculate problem complexity score for PiscesReasoner.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            
        Returns:
            float: Complexity score between 0 and 1
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Factor 1: Sequence length complexity
        length_complexity = min(seq_len / 256, 1.0)  # Lower threshold for reasoning
        
        # Factor 2: Semantic diversity across the reasoning sequence
        mean_state = hidden_states.mean(dim=1, keepdim=True)
        semantic_variance = torch.var(hidden_states - mean_state, dim=[1,2]).mean()
        diversity_complexity = torch.sigmoid(semantic_variance * 15)  # Higher sensitivity
        
        # Factor 3: Reasoning pattern complexity (gradient-based)
        if hidden_states.requires_grad:
            grad_magnitude = torch.norm(hidden_states.grad) if hidden_states.grad is not None else torch.tensor(0.0)
            reasoning_complexity = torch.sigmoid(grad_magnitude * 5)
        else:
            reasoning_complexity = torch.tensor(0.5)
        
        # Combine factors with reasoning-specific weights
        complexity = (length_complexity * 0.4 + 
                     diversity_complexity * 0.4 + 
                     reasoning_complexity * 0.2)
        
        return complexity.item()
    
    def _calculate_abstraction_gain(self, prev_states, curr_states):
        """
        Calculate abstraction gain for reasoning layers.
        
        Args:
            prev_states: Previous layer hidden states
            curr_states: Current layer hidden states
            
        Returns:
            float: Abstraction gain score
        """
        # Measure reasoning improvement using multiple metrics
        prev_pooled = F.normalize(prev_states.mean(dim=1), p=2, dim=-1)
        curr_pooled = F.normalize(curr_states.mean(dim=1), p=2, dim=-1)
        
        # Semantic advancement
        semantic_advance = 1 - F.cosine_similarity(prev_pooled, curr_pooled, dim=-1).mean()
        
        # Information consolidation (reduction in variance indicates abstraction)
        prev_info = torch.norm(torch.var(prev_states, dim=1), p=2)
        curr_info = torch.norm(torch.var(curr_states, dim=1), p=2)
        consolidation_ratio = max(0, (prev_info - curr_info) / (prev_info + 1e-8))
        
        # Reasoning coherence (smoothness of state transitions)
        state_diff = torch.norm(curr_states - prev_states, p=2, dim=-1).mean()
        coherence_score = torch.sigmoid(1.0 / (state_diff + 0.1))
        
        # Combine metrics with reasoning-specific weights
        abstraction_gain = semantic_advance * 0.5 + consolidation_ratio * 0.3 + coherence_score * 0.2
        return abstraction_gain.item()
        
        # Special tokens for CoT
        self.register_buffer('start_thinking_token', torch.tensor([50256]))
        self.register_buffer('end_thinking_token', torch.tensor([50257]))
        self.register_buffer('reflection_token', torch.tensor([50258]))
    
    def forward(self, input_ids=None, attention_mask=None, memory_context=None, **kwargs):
        """Enhanced forward pass with memory integration and multi-step reasoning"""
        
        # Handle input
        if torch.is_tensor(input_ids):
            hidden_states = input_ids
        else:
            hidden_states = torch.randn(1, 1, self.cfg.hidden_size)
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Memory context integration for multi-turn dialogue
        if memory_context is not None and len(memory_context) > 0:
            # Convert memory to tensor
            memory_tensor = torch.stack(memory_context).unsqueeze(0) if isinstance(memory_context[0], torch.Tensor) else torch.tensor(memory_context).unsqueeze(0)
            
            # Project memory to key and value spaces
            memory_keys = self.memory_key_proj(memory_tensor)
            memory_values = self.memory_value_proj(memory_tensor)
            
            # Cross-attention: current context attends to memory
            memory_out, attention_weights = self.memory_attention(
                hidden_states, memory_keys, memory_values
            )
            
            # Residual connection
            hidden_states = hidden_states + memory_out
        else:
            attention_weights = None
        
        # Multi-step CoT reasoning with adaptive depth
        reasoning_states = self.layer_norm(hidden_states)
        reasoning_steps = []
        
        # Calculate problem complexity for dynamic depth adjustment
        complexity_score = self._calculate_problem_complexity(reasoning_states)
        
        # Dynamic depth selection based on complexity
        if complexity_score < 0.3:
            # Simple problems: use minimal depth
            num_layers = 1
        elif complexity_score < 0.6:
            # Medium complexity: use moderate depth
            num_layers = 2
        else:
            # Complex problems: use full depth
            num_layers = len(self.reasoning_layers)
        
        # Adaptive reasoning with early termination
        for i, layer in enumerate(self.reasoning_layers[:num_layers]):
            prev_states = reasoning_states.clone()
            reasoning_states = layer(reasoning_states)
            reasoning_steps.append(reasoning_states.clone())
            
            # Check abstraction gain for early termination
            if i > 0:
                abstraction_gain = self._calculate_abstraction_gain(prev_states, reasoning_states)
                if abstraction_gain < 0.1:  # Minimal improvement
                    break
        
        # Get final reasoning state for outputs
        final_state = reasoning_states[:, -1, :]
        
        # Enhanced outputs
        thinking_logits = self.thinking_head(final_state)
        difficulty_logits = self.difficulty_head(final_state)
        reflection_logits = self.reflection_head(final_state)
        confidence_score = torch.sigmoid(self.confidence_head(final_state))
        
        # Hierarchical reflection analysis
        error_analysis = self.error_analyzer(final_state)
        correction_input = torch.cat([final_state, error_analysis], dim=-1)
        correction_logits = self.correction_head(correction_input)
        
        return {
            "thinking_logits": thinking_logits,
            "difficulty_logits": difficulty_logits,
            "reflection_logits": reflection_logits,
            "confidence_score": confidence_score,
            "reasoning_states": reasoning_states,
            "reasoning_steps": reasoning_steps,
            "correction_logits": correction_logits,
            "attention_weights": attention_weights,
            "final_state": final_state
        }

class MCPProtocol:
    """MCP protocol implementation for agent communication"""
    
    @staticmethod
    def create_message(
        message_type: MCPMessageType,
        agent_id: str,
        payload: Dict[str, Any],
        correlation_id: str = ""
    ) -> MCPMessage:
        return MCPMessage(
            message_type=message_type.value,
            agent_id=agent_id,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )

class PiscesAgent(nn.Module):
    """
    Native Pisces L1 Agent with integrated reasoning, perception, and action capabilities.
    Fully migrated from agent.py to multimodal.py for unified architecture.
    
    Features:
    1. Unified multimodal perception (text, image, audio)
    2. Advanced reasoning with CoT and self-reflection
    3. Tool use and environment interaction
    4. Persistent memory and context management
    5. End-to-end trainable architecture
    6. Full MCP protocol support
    """
    
    def __init__(self, cfg, tokenizer=None, model=None, agent_id: str = None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        import weakref
        self._model_ref = None  # weak reference placeholder
        self.agent_id = agent_id or str(uuid.uuid4())
        
        # Use provided model or create new one
        if model is not None:
            # Store a weak reference (callable) to avoid registering model as a submodule.
            self._base_model_ref = weakref.ref(model)
            self._model_ref = self._base_model_ref
        else:
            # Stand-alone agent: no linked PiscesModel to avoid cycles.
            self._base_model_ref = None
            self._model_ref = None
            self._reasoner = PiscesReasoner(cfg)
            self._vision_encoder = VisionEncoder(cfg)
            self._audio_encoder = AudioEncoder(cfg)
        
        self.tree_reasoner = TreeSearchReasoner(None, tokenizer) if tokenizer else None

        # MCP Agent infrastructure
        self.memory = AgentMemory([], [], [])
        self.mcp_tools = MCPToolRegistry(self.agent_id, self._handle_mcp_message)
        self.state = AgentState.IDLE
        self.mcp_peers: Dict[str, Dict[str, Any]] = {}
        self.mcp_capabilities: Dict[str, Dict[str, Any]] = {}
        
        # Coordinate marking support
        self._coordinate_detection_enabled = True
        
        # Action prediction heads (now MCP-aware)
        self.action_type_head = nn.Linear(cfg.hidden_size, 10)  # 10 action types
        self.action_param_head = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)
        
        # MCP message handlers
        self.mcp_handlers = {
            MCPMessageType.OBSERVATION.value: self._handle_observation,
            MCPMessageType.ACTION.value: self._handle_action,
            MCPMessageType.TOOL_CALL.value: self._handle_tool_call,
            MCPMessageType.TOOL_RESULT.value: self._handle_tool_result,
            MCPMessageType.CAPABILITY_REGISTER.value: self._handle_capability_register,
            MCPMessageType.SYNC_REQUEST.value: self._handle_sync_request,
            MCPMessageType.SYNC_RESPONSE.value: self._handle_sync_response,
        }

    @property
    def base_model(self):
        return self._base_model_ref() if self._base_model_ref else None

    @property
    def reasoner(self):
        if self._base_model_ref:
            return self._base_model_ref().reasoner
        return self._reasoner

    @property
    def vision_encoder(self):
        if self._base_model_ref:
            return self._base_model_ref().vision
        return self._vision_encoder

    @property
    def audio_encoder(self):
        if self._base_model_ref:
            return self._base_model_ref().audio
        return self._audio_encoder

    async def register_capability(self, name: str, description: str, 
                                  parameters: Dict[str, Any], handler: Callable):
        """Register a capability via MCP protocol"""
        await self.mcp_tools.register_capability(name, description, parameters, handler)

    async def _handle_mcp_message(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP messages"""
        handler = self.mcp_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            return MCPProtocol.create_message(
                MCPMessageType.STATE_UPDATE,
                self.agent_id,
                {"error": f"Unknown message type: {message.message_type}"}
            )

    async def _handle_observation(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP observation messages"""
        observation_data = message.payload
        observation = AgentObservation(
            modality=observation_data["modality"],
            content=observation_data["content"],
            metadata=observation_data.get("metadata", {})
        )
        
        self.memory.add_observation(observation)
        obs_embedding = self.process_observation(observation)
        
        return MCPProtocol.create_message(
            MCPMessageType.OBSERVATION,
            self.agent_id,
            {
                "status": "processed",
                "embedding_shape": list(obs_embedding.shape) if torch.is_tensor(obs_embedding) else None,
                "observation_id": str(uuid.uuid4())
            }
        )

    async def _handle_action(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP action messages"""
        action_data = message.payload
        context = {
            "observation": action_data.get("observation"),
            "available_tools": list(self.mcp_capabilities.keys())
        }
        
        action = await self.plan_action(context)
        self.memory.add_action(action)
        
        return MCPProtocol.create_message(
            MCPMessageType.ACTION,
            self.agent_id,
            {
                "action": asdict(action),
                "agent_state": self.state.value
            }
        )

    async def _handle_tool_call(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP tool call messages"""
        tool_data = message.payload
        tool_name = tool_data["tool_name"]
        parameters = tool_data["parameters"]
        
        if tool_name in self.mcp_capabilities:
            handler = self.mcp_capabilities[tool_name]["handler"]
            result = await handler(**parameters)
            
            return MCPProtocol.create_message(
                MCPMessageType.TOOL_RESULT,
                self.agent_id,
                {
                    "tool_name": tool_name,
                    "result": result,
                    "success": True
                }
            )
        else:
            return MCPProtocol.create_message(
                MCPMessageType.TOOL_RESULT,
                self.agent_id,
                {
                    "tool_name": tool_name,
                    "error": f"Tool {tool_name} not found",
                    "success": False
                }
            )

    async def _handle_tool_result(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP tool result messages"""
        result_data = message.payload
        
        observation = AgentObservation(
            modality="tool_result",
            content=result_data,
            metadata={"source": message.agent_id}
        )
        
        self.memory.add_observation(observation)
        
        return MCPProtocol.create_message(
            MCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "tool_result_processed", "result_id": str(uuid.uuid4())}
        )

    async def _handle_capability_register(self, message: MCPMessage) -> MCPMessage:
        """Handle capability registration from other agents"""
        capability_data = message.payload
        capability_name = capability_data["capability"]
        
        if message.agent_id != self.agent_id:
            self.mcp_capabilities[f"{message.agent_id}.{capability_name}"] = capability_data
        
        return MCPProtocol.create_message(
            MCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "capability_registered", "capability": capability_name}
        )

    async def _handle_sync_request(self, message: MCPMessage) -> MCPMessage:
        """Handle synchronization requests"""
        sync_type = message.payload.get("type")
        
        if sync_type == "capability_discovery":
            return MCPProtocol.create_message(
                MCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "capabilities",
                    "capabilities": list(self.mcp_capabilities.keys())
                }
            )
        elif sync_type == "state_sync":
            return MCPProtocol.create_message(
                MCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "state",
                    "state": self.state.value,
                    "memory_summary": self._summarize_memory()
                }
            )
        
        return MCPProtocol.create_message(
            MCPMessageType.SYNC_RESPONSE,
            self.agent_id,
            {"error": "Unknown sync type"}
        )

    async def _handle_sync_response(self, message: MCPMessage) -> MCPMessage:
        """Handle synchronization responses"""
        response_data = message.payload
        
        if response_data.get("type") == "capabilities":
            self.mcp_peers[message.agent_id] = {
                "capabilities": response_data.get("capabilities", [])
            }
        
        return MCPProtocol.create_message(
            MCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "sync_completed", "peer_id": message.agent_id}
        )

    def process_observation(self, observation: AgentObservation) -> torch.Tensor:
        """Process multimodal observations into unified representation"""
        if observation.modality == "text":
            if hasattr(self, 'tokenizer') and self.tokenizer:
                tokens = self.tokenizer.encode(str(observation.content), return_tensors="pt")
                if self.base_model:
                    return self.base_model.embed_tokens(tokens)
                else:
                    return torch.randn(1, tokens.size(1), self.cfg.hidden_size)
            else:
                return torch.zeros(1, 1, self.cfg.hidden_size)
        
        elif observation.modality == "image":
            if isinstance(observation.content, str):  # image path
                image_tensor = self.vision_encoder.process_image(observation.content)
                if image_tensor is not None:
                    image_tensor = image_tensor.unsqueeze(0)
                    return self.vision_encoder(image_tensor)
            elif torch.is_tensor(observation.content):
                return self.vision_encoder(observation.content)
        
        elif observation.modality == "audio":
            if isinstance(observation.content, str):  # audio path
                audio_tensor = self.audio_encoder.process_audio(observation.content)
                if audio_tensor is not None:
                    return self.audio_encoder(audio_tensor)
            elif torch.is_tensor(observation.content):
                return self.audio_encoder(observation.content)
        
        elif observation.modality == "tool_result":
            # Convert tool results to embedding
            import json
            result_str = json.dumps(observation.content)
            if hasattr(self, 'tokenizer') and self.tokenizer:
                tokens = self.tokenizer.encode(result_str, return_tensors="pt")
                if self.base_model:
                    return self.base_model.embed_tokens(tokens)
                else:
                    return torch.randn(1, tokens.size(1), self.cfg.hidden_size)
        
        # Fallback to zero tensor
        return torch.zeros(1, 1, self.cfg.hidden_size)

    async def plan_action(self, context: Dict[str, Any]) -> AgentAction:
        """Generate action based on current context and enhanced reasoning"""
        # Get enhanced context from memory with semantic search
        memory_context = self.memory.get_context_with_retrieval(
            query=str(context), 
            k=5, 
            include_compressed=True
        )
        
        # Encode query for semantic search
        query_embedding = self._encode_query(str(context))
        
        # Perform semantic memory retrieval
        relevant_memories = self.memory.semantic_search(
            query_embedding=query_embedding,
            k=3,
            threshold=0.7
        )
        
        # Extract memory keys and values for enhanced reasoning
        memory_keys = self._extract_memory_keys(relevant_memories)
        memory_values = self._extract_memory_values(relevant_memories)
        
        # Prepare enhanced reasoning input
        enhanced_input = self._prepare_enhanced_reasoning_input(
            context=context,
            memory_context=memory_context,
            memory_keys=memory_keys,
            memory_values=memory_values,
            query_embedding=query_embedding
        )
        
        # Use enhanced reasoner for multi-step CoT reasoning
        with torch.no_grad():
            if self.base_model and hasattr(self, 'reasoner'):
                # Enhanced reasoning with memory context
                reasoning_output = self.reasoner(
                    enhanced_input,
                    memory_context=memory_context.get("embeddings", None)
                )
                
                # Multi-step CoT processing
                thinking_logits = reasoning_output.get("thinking_logits", reasoning_output.get("logits"))
                difficulty_logits = reasoning_output.get("difficulty_logits")
                reflection_logits = reasoning_output.get("reflection_logits")
                confidence_logits = reasoning_output.get("confidence_logits")
                
                # Enhanced action prediction
                action_logits = self.action_type_head(thinking_logits[:, -1])
                action_probs = torch.softmax(action_logits, dim=-1)
                action_type_idx = torch.argmax(action_probs, dim=-1).item()
                
                # Enhanced confidence calculation with reflection
                base_confidence = torch.sigmoid(self.confidence_head(thinking_logits[:, -1])).item()
                reflection_confidence = torch.sigmoid(reflection_logits[:, -1]).item() if reflection_logits is not None else base_confidence
                confidence = (base_confidence + reflection_confidence) / 2
                
                # Difficulty-aware reasoning
                if difficulty_logits is not None:
                    difficulty = torch.softmax(difficulty_logits[:, -1], dim=-1)
                    difficulty_level = torch.argmax(difficulty, dim=-1).item()
                else:
                    difficulty_level = 2  # Default medium
                
            else:
                # Fallback for stand-alone mode
                action_logits = self.action_type_head(torch.randn(1, self.cfg.hidden_size))
                action_probs = torch.softmax(action_logits, dim=-1)
                action_type_idx = torch.argmax(action_probs, dim=-1).item()
                confidence = 0.5
                difficulty_level = 2
            
            # Enhanced action types with deep thinking
            action_types = [
                "respond", "use_tool", "ask_clarification", "reflect", 
                "search_memory", "plan_next", "wait", "verify", 
                "correct_action", "explore", "deep_think", "summarize"
            ]
            action_type = action_types[action_type_idx]
            
            # Generate enhanced action parameters
            if self.base_model and hasattr(self, 'reasoner'):
                param_embedding = self.action_param_head(thinking_logits[:, -1])
            else:
                param_embedding = self.action_param_head(torch.randn(1, self.cfg.hidden_size))
            
            action_params = self._decode_enhanced_action_params(
                param_embedding, 
                action_type, 
                difficulty_level,
                confidence
            )
            
            # Generate detailed reasoning trace
            reasoning_trace = self._generate_reasoning_trace(
                context=context,
                memory_summary=memory_context.get("memory_summary", {}),
                action_type=action_type,
                confidence=confidence,
                difficulty_level=difficulty_level,
                relevant_memories=relevant_memories
            )
            
            return AgentAction(
                action_type=action_type,
                parameters=action_params,
                confidence=confidence,
                reasoning=reasoning_trace
            )

    def _prepare_reasoning_input(self, context: Dict[str, Any], memory_context: Dict[str, List]) -> Dict[str, Any]:
        """Prepare input for the reasoner"""
        return {
            "context": context,
            "memory": memory_context,
            "agent_state": self.state.value
        }

    def _decode_action_params(self, param_embedding: torch.Tensor, action_type: str) -> Dict[str, Any]:
        """Decode action parameters from embedding"""
        return {
            "embedding": param_embedding.detach().cpu().numpy().tolist(),
            "decoded_from": action_type,
            "confidence": torch.sigmoid(self.confidence_head(param_embedding)).item()
        }

    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode query string into semantic embedding"""
        if hasattr(self, 'tokenizer') and self.tokenizer and self.base_model:
            tokens = self.tokenizer.encode(query, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                embeddings = self.base_model.embed_tokens(tokens)
                # Use mean pooling for query embedding
                query_embedding = embeddings.mean(dim=1)
                return query_embedding
        else:
            # Fallback: use simple hash-based encoding
            import hashlib
            hash_obj = hashlib.md5(query.encode())
            hash_bytes = hash_obj.digest()
            hash_tensor = torch.tensor([int(b) for b in hash_bytes], dtype=torch.float32)
            # Normalize to hidden size
            query_embedding = hash_tensor.unsqueeze(0) / 255.0
            if query_embedding.size(-1) != self.cfg.hidden_size:
                # Pad or truncate to match hidden size
                if query_embedding.size(-1) < self.cfg.hidden_size:
                    padding = torch.zeros(1, self.cfg.hidden_size - query_embedding.size(-1))
                    query_embedding = torch.cat([query_embedding, padding], dim=-1)
                else:
                    query_embedding = query_embedding[:, :self.cfg.hidden_size]
            return query_embedding

    def _extract_memory_keys(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Extract memory keys from retrieved memories"""
        keys = []
        for memory in memories:
            if "content" in memory:
                content = str(memory["content"])
                # Extract key phrases (simplified)
                key_phrases = content.split()[:10]  # First 10 words as key
                keys.append(" ".join(key_phrases))
            else:
                keys.append("memory_entry")
        return keys

    def _extract_memory_values(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Extract memory values from retrieved memories"""
        values = []
        for memory in memories:
            if "content" in memory:
                values.append(str(memory["content"]))
            else:
                values.append(str(memory))
        return values

    def _prepare_enhanced_reasoning_input(self, context: Dict[str, Any], 
                                        memory_context: Dict[str, List],
                                        memory_keys: List[str],
                                        memory_values: List[str],
                                        query_embedding: torch.Tensor) -> Dict[str, Any]:
        """Prepare enhanced input for multi-step CoT reasoning"""
        return {
            "context": context,
            "memory_context": memory_context,
            "memory_keys": memory_keys,
            "memory_values": memory_values,
            "query_embedding": query_embedding,
            "agent_state": self.state.value,
            "timestamp": str(uuid.uuid4())[:8]  # Simple timestamp
        }

    def _decode_enhanced_action_params(self, param_embedding: torch.Tensor, 
                                     action_type: str, 
                                     difficulty_level: int,
                                     confidence: float) -> Dict[str, Any]:
        """Decode enhanced action parameters with difficulty and confidence"""
        params = {
            "embedding": param_embedding.detach().cpu().numpy().tolist(),
            "action_type": action_type,
            "difficulty_level": difficulty_level,
            "confidence": confidence,
            "timestamp": str(uuid.uuid4())[:8]
        }
        
        # Add action-specific parameters
        if action_type == "use_tool":
            params.update({
                "tool_name": "default_tool",
                "tool_parameters": {},
                "retry_count": 0
            })
        elif action_type == "reflect":
            params.update({
                "reflection_type": "self_analysis",
                "focus_areas": ["accuracy", "efficiency", "completeness"]
            })
        elif action_type == "search_memory":
            params.update({
                "search_query": "relevant_memories",
                "max_results": 5,
                "include_compressed": True
            })
        elif action_type == "deep_think":
            params.update({
                "thinking_steps": min(4 + difficulty_level, 8),
                "exploration_depth": min(2 + difficulty_level // 2, 5),
                "validation_required": confidence < 0.7
            })
        
        return params

    def _generate_reasoning_trace(self, context: Dict[str, Any],
                                memory_summary: Dict[str, int],
                                action_type: str,
                                confidence: float,
                                difficulty_level: int,
                                relevant_memories: List[Dict[str, Any]]) -> str:
        """Generate detailed reasoning trace for transparency"""
        trace_parts = []
        
        # Context analysis
        trace_parts.append(f"Context Analysis: Processing {len(str(context))} characters of input")
        
        # Memory integration
        total_memories = memory_summary.get("total_count", 0)
        retrieved_count = len(relevant_memories)
        trace_parts.append(f"Memory Integration: Retrieved {retrieved_count} relevant memories from {total_memories} total")
        
        # Difficulty assessment
        difficulty_labels = ["very_easy", "easy", "medium", "hard", "very_hard"]
        difficulty_label = difficulty_labels[min(difficulty_level, len(difficulty_labels)-1)]
        trace_parts.append(f"Difficulty Assessment: {difficulty_label} (level {difficulty_level})")
        
        # Action selection reasoning
        trace_parts.append(f"Action Selection: Chose '{action_type}' with {confidence:.2f} confidence")
        
        # Memory influence
        if relevant_memories:
            memory_types = [mem.get("type", "unknown") for mem in relevant_memories]
            type_counts = {}
            for t in memory_types:
                type_counts[t] = type_counts.get(t, 0) + 1
            trace_parts.append(f"Memory Influence: {type_counts}")
        
        # Reasoning summary
        trace_parts.append("Reasoning Complete: Enhanced CoT with semantic memory integration")
        
        return " | ".join(trace_parts)

    def _summarize_memory(self) -> Dict[str, int]:
        """Summarize memory for state sync"""
        return {
            "observations": len(self.memory.observations),
            "actions": len(self.memory.actions),
            "reflections": len(self.memory.reflections)
        }

    def detect_objects(self, image_input: Union[str, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        Detect objects in image and return their coordinates.
        
        Args:
            image_input: Image path string, tensor, or numpy array
            
        Returns:
            Dict containing detected objects with coordinates:
            {
                "objects": [
                    {
                        "class": str,
                        "confidence": float,
                        "coordinates": [x_center, y_center],
                        "bbox": [x_min, y_min, x_max, y_max]
                    }
                ],
                "image_size": [width, height],
                "num_objects": int
            }
        """
        if not self._coordinate_detection_enabled:
            return {"objects": [], "image_size": [0, 0], "num_objects": 0}
        
        try:
            # Process image through vision encoder
            if isinstance(image_input, str):
                image_tensor = self.vision_encoder.process_image(image_input)
                if image_tensor is None:
                    return {"objects": [], "image_size": [0, 0], "num_objects": 0}
                image_tensor = image_tensor.unsqueeze(0)
            elif isinstance(image_input, np.ndarray):
                image_tensor = torch.from_numpy(image_input).float()
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.unsqueeze(0)
            else:
                image_tensor = image_input
            
            # Get detection results from vision encoder
            with torch.no_grad():
                detection_results = self.vision_encoder(image_tensor)
            
            if "detection_results" not in detection_results:
                return {"objects": [], "image_size": [0, 0], "num_objects": 0}
            
            results = detection_results["detection_results"]
            objects = []
            
            # Process bounding boxes and coordinates
            if "boxes" in results and "labels" in results:
                boxes = results["boxes"].cpu().numpy()
                labels = results["labels"].cpu().numpy()
                scores = results.get("scores", torch.ones(len(boxes))).cpu().numpy()
                
                # Convert to image coordinates
                img_coords = self.vision_encoder.convert_patch_to_image_coords(boxes)
                
                for i, (box, label, score) in enumerate(zip(img_coords, labels, scores)):
                    if score > 0.5:  # Confidence threshold
                        x_min, y_min, x_max, y_max = box
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        
                        objects.append({
                            "class": f"class_{label}",
                            "confidence": float(score),
                            "coordinates": [float(x_center), float(y_center)],
                            "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)]
                        })
            
            # Get image dimensions
            if isinstance(image_input, str):
                from PIL import Image
                with Image.open(image_input) as img:
                    width, height = img.size
            else:
                # Assume square image for tensor inputs
                width = height = 224
            
            return {
                "objects": objects,
                "image_size": [width, height],
                "num_objects": len(objects)
            }
            
        except Exception as e:
            return {"objects": [], "image_size": [0, 0], "num_objects": 0, "error": str(e)}

    def get_coordinates(self, image_input: Union[str, torch.Tensor, np.ndarray], 
                         target_object: str = None) -> List[List[float]]:
        """
        Get coordinates of detected objects or specific target.
        
        Args:
            image_input: Image to analyze
            target_object: Optional target object name to filter results
            
        Returns:
            List of [x, y] coordinates for detected objects
        """
        detection_results = self.detect_objects(image_input)
        
        coordinates = []
        for obj in detection_results.get("objects", []):
            if target_object is None or target_object.lower() in obj["class"].lower():
                coordinates.append(obj["coordinates"])
        
        return coordinates

    def point_to_object(self, image_input: Union[str, torch.Tensor, np.ndarray], 
                       object_description: str) -> Dict[str, Any]:
        """
        Point to a specific object in the image based on description.
        
        Args:
            image_input: Image to analyze
            object_description: Description of the object to find
            
        Returns:
            Dict with object information and pointing coordinates
        """
        detection_results = self.detect_objects(image_input)
        
        # Simple matching based on description
        best_match = None
        highest_confidence = 0
        
        for obj in detection_results.get("objects", []):
            # Simple keyword matching for now
            obj_class = obj["class"].lower()
            description_lower = object_description.lower()
            
            if any(keyword in obj_class or obj_class in keyword 
                   for keyword in description_lower.split()):
                if obj["confidence"] > highest_confidence:
                    best_match = obj
                    highest_confidence = obj["confidence"]
        
        if best_match:
            return {
                "found": True,
                "object": best_match,
                "point_coordinates": best_match["coordinates"],
                "message": f"Found {object_description} at coordinates {best_match['coordinates']}"
            }
        else:
            return {
                "found": False,
                "point_coordinates": None,
                "message": f"Could not find {object_description} in the image"
            }

    def enable_coordinate_detection(self, enabled: bool = True):
        """Enable or disable coordinate detection functionality"""
        self._coordinate_detection_enabled = enabled
        if hasattr(self.vision_encoder, 'enable_detection'):
            self.vision_encoder.enable_detection(enabled)

class AgentEncoder(nn.Module):
    """
    Legacy Agent encoder - now replaced by PiscesAgent for comprehensive agent functionality.
    Maintained for backward compatibility.
    """
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.pisces_agent = PiscesAgent(cfg)
        
    def forward(self, agent_input):
        """Forward pass using PiscesAgent"""
        return self.pisces_agent.process_observation(agent_input)
    
    def encode_observation(self, observation):
        """Encode multimodal observations like PiscesAgent"""
        if observation['modality'] == "text":
            tokens = observation['content']
            if isinstance(tokens, str):
                # Handle string input
                tokens = torch.tensor([hash(tokens) % self.cfg.vocab_size])
            return self.obs_text_encoder(tokens)
        
        elif observation['modality'] == "image":
            return self.obs_image_encoder(observation['content'])
        
        elif observation['modality'] == "audio":
            return self.obs_audio_encoder(observation['content'])
        
        elif observation['modality'] == "tool_result":
            # Encode tool results as text
            result_str = str(observation['content'])
            tokens = torch.tensor([hash(result_str) % self.cfg.vocab_size])
            return self.obs_text_encoder(tokens)
        
        else:
            return torch.zeros(1, 1, self.cfg.hidden_size)
    
    def encode_memory(self, memory_data):
        """Encode agent memory like PiscesAgent memory system"""
        # Encode observations memory
        obs_features = []
        for obs in memory_data.get('observations', []):
            obs_feat = self.encode_observation(obs)
            obs_features.append(obs_feat)
        
        if obs_features:
            obs_tensor = torch.stack(obs_features, dim=1)
            obs_memory, _ = self.memory_encoder['obs_memory'](obs_tensor)
        else:
            obs_memory = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Encode actions memory
        action_features = []
        for action in memory_data.get('actions', []):
            action_str = str(action)
            tokens = torch.tensor([hash(action_str) % self.cfg.vocab_size])
            action_feat = self.obs_text_encoder(tokens)
            action_features.append(action_feat)
        
        if action_features:
            action_tensor = torch.stack(action_features, dim=1)
            action_memory, _ = self.memory_encoder['action_memory'](action_tensor)
        else:
            action_memory = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Encode reflections memory
        reflection_features = []
        for reflection in memory_data.get('reflections', []):
            tokens = torch.tensor([hash(str(reflection)) % self.cfg.vocab_size])
            ref_feat = self.obs_text_encoder(tokens)
            reflection_features.append(ref_feat)
        
        if reflection_features:
            ref_tensor = torch.stack(reflection_features, dim=1)
            reflection_memory, _ = self.memory_encoder['reflection_memory'](ref_tensor)
        else:
            reflection_memory = torch.zeros(1, 1, self.cfg.hidden_size)
        
        return obs_memory, action_memory, reflection_memory
    
    def forward(self, agent_input):
        """
        Forward pass of the comprehensive AgentEncoder.
        
        Args:
            agent_input (dict): Dictionary containing:
                - observations: List of agent observations
                - actions: List of agent actions
                - reflections: List of agent reflections
                - current_state: Current agent state tensor
                - task_context: Current task description
        
        Returns:
            torch.Tensor: Comprehensive agent features including state, memory, and predictions.
        """
        # Encode observations
        obs_features = []
        for obs in agent_input.get('observations', []):
            obs_feat = self.encode_observation(obs)
            obs_features.append(obs_feat)
        
        # Encode memory
        memory_data = {
            'observations': agent_input.get('observations', []),
            'actions': agent_input.get('actions', []),
            'reflections': agent_input.get('reflections', [])
        }
        obs_memory, action_memory, reflection_memory = self.encode_memory(memory_data)
        
        # Encode current state
        if 'current_state' in agent_input:
            state_feat = self.state_encoder(agent_input['current_state'])
        else:
            state_feat = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Encode task context
        if 'task_context' in agent_input:
            task_tokens = torch.tensor([hash(str(agent_input['task_context'])) % self.cfg.vocab_size])
            task_feat = self.obs_text_encoder(task_tokens)
        else:
            task_feat = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Combine all features using attention
        combined_features = torch.cat([
            obs_memory[:, -1:],  # Latest observation
            action_memory[:, -1:],  # Latest action
            reflection_memory[:, -1:],  # Latest reflection
            state_feat,
            task_feat
        ], dim=1)
        
        # Apply cross-modal attention
        attended_features, _ = self.agent_attention(combined_features, combined_features, combined_features)
        
        # Predict action type and parameters (like PiscesAgent)
        action_logits = self.action_type_head(attended_features.mean(dim=1))
        action_params = self.action_param_head(attended_features.mean(dim=1))
        confidence = torch.sigmoid(self.confidence_head(attended_features.mean(dim=1)))
        
        # Final comprehensive encoding
        all_features = torch.cat([
            attended_features.mean(dim=1),
            action_params,
            confidence
        ], dim=-1)
        
        return self.final_proj(all_features).unsqueeze(1)

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module for enhanced multimodal fusion.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.n_head
        self.hidden_size = cfg.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape

        # Project to Q, K, V
        Q = self.q_proj(self.norm1(query)).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(self.norm2(key)).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Optional xFormers fast path (mirrors usage style in modeling_aurora.Attention)
        try:
            from xformers.ops import memory_efficient_attention  # type: ignore
            _use_xformers = True
        except ImportError:
            _use_xformers = False

        if _use_xformers and mask is None:
            # xFormers expects [B, H, T, D]
            out = memory_efficient_attention(Q, K, V)  # -> [B, H, T_q, D]
            out = out.transpose(1, 2).contiguous().view(B, T_q, self.hidden_size)
        else:
            # Scaled dot-product attention (PyTorch fallback)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, V)
            out = out.transpose(1, 2).contiguous().view(B, T_q, self.hidden_size)

        return self.o_proj(out)

class MemoryManager:
    """
    Advanced memory management system for Pisces L1 model.
    Provides dynamic memory monitoring, garbage collection, and leak prevention.
    """
    def __init__(self, device='cuda', memory_threshold=0.85, cleanup_interval=30, enable_background=True):
        """
        Initialize memory manager.
        
        Args:
            device: Target device ('cuda' or 'cpu')
            memory_threshold: Memory usage threshold for triggering cleanup (0-1)
            cleanup_interval: Interval in seconds for automatic cleanup
        """
        self.device = device
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        self.enable_background = enable_background
        
        # Memory tracking
        self.tensor_registry = weakref.WeakValueDictionary()
        self.intermediate_tensors = defaultdict(list)
        self.memory_stats = {
            'peak_memory': 0,
            'cleanup_count': 0,
            'leak_preventions': 0,
            'total_freed': 0
        }
        
        # Threading for background cleanup
        self.cleanup_thread = None
        self.stop_cleanup = threading.Event()
        self.monitor_active = False
        
        # GPU-specific tracking
        if device == 'cuda' and torch.cuda.is_available():
            self.gpu_id = torch.cuda.current_device()
            self.initial_memory = torch.cuda.memory_allocated(self.gpu_id)
        else:
            self.gpu_id = None
            self.initial_memory = 0
    
    def start_monitoring(self):
        """Start background memory monitoring and cleanup."""
        if not self.monitor_active and self.enable_background:
            self.monitor_active = True
            self.stop_cleanup.clear()
            self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
            self.cleanup_thread.start()
            DEBUG(f"Memory monitoring started on {self.device}")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if self.monitor_active:
            self.monitor_active = False
            self.stop_cleanup.set()
            if self.cleanup_thread:
                self.cleanup_thread.join(timeout=5)
            self._emergency_cleanup()
            DEBUG("Memory monitoring stopped")
    
    def register_tensor(self, tensor, name=None, is_intermediate=True):
        """
        Register a tensor for memory tracking.
        
        Args:
            tensor: Tensor to register
            name: Optional name for tracking
            is_intermediate: Whether this is an intermediate tensor that can be cleaned
        """
        if tensor is None:
            return tensor
            
        tensor_id = id(tensor)
        self.tensor_registry[tensor_id] = tensor
        
        if is_intermediate and name:
            self.intermediate_tensors[name].append(weakref.ref(tensor))
            
        return tensor
    
    def track_memory(self, operation_name="operation"):
        """
        Context manager for tracking memory usage of operations.
        
        Args:
            operation_name: Name of the operation for logging
        """
        class MemoryTracker:
            def __init__(tracker_self):
                tracker_self.operation_name = operation_name
                tracker_self.start_memory = self.get_memory_usage()
                tracker_self.start_time = time.time()
            
            def __enter__(tracker_self):
                return tracker_self
            
            def __exit__(tracker_self, exc_type, exc_val, exc_tb):
                end_memory = self.get_memory_usage()
                end_time = time.time()
                
                memory_diff = end_memory - tracker_self.start_memory
                if abs(memory_diff) > 100 * 1024 * 1024:  # 100MB threshold
                    DEBUG(f"Memory change in {tracker_self.operation_name}: {memory_diff / 1024**2:.2f}MB "
                          f"({end_memory / 1024**2:.2f}MB total)")
                
                # Auto-cleanup if memory usage is high
                if end_memory > self.get_memory_limit() * self.memory_threshold:
                    self.cleanup_intermediate_tensors()
        
        return MemoryTracker()
    
    def get_memory_usage(self):
        """Get current memory usage in bytes."""
        if self.device == 'cuda' and torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.gpu_id)
        else:
            # CPU memory estimation
            process = psutil.Process()
            return process.memory_info().rss
    
    def get_memory_limit(self):
        """Get total available memory in bytes."""
        if self.device == 'cuda' and torch.cuda.is_available():
            return torch.cuda.get_device_properties(self.gpu_id).total_memory
        else:
            return psutil.virtual_memory().total
    
    def cleanup_intermediate_tensors(self, force=False):
        """
        Clean up intermediate tensors to free memory.
        
        Args:
            force: Force cleanup even if memory usage is low
        """
        current_usage = self.get_memory_usage()
        memory_limit = self.get_memory_limit()
        
        if force or (current_usage > memory_limit * self.memory_threshold):
            freed_count = 0
            freed_memory = 0
            
            # Clean registered intermediate tensors
            for name, tensor_refs in list(self.intermediate_tensors.items()):
                alive_tensors = []
                for tensor_ref in tensor_refs:
                    tensor = tensor_ref()
                    if tensor is not None:
                        freed_memory += tensor.numel() * tensor.element_size()
                        del tensor
                        freed_count += 1
                    else:
                        alive_tensors.append(tensor_ref)
                
                self.intermediate_tensors[name] = alive_tensors
            
            # Force garbage collection
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            self.memory_stats['cleanup_count'] += 1
            self.memory_stats['total_freed'] += freed_memory
            
            new_usage = self.get_memory_usage()
            DEBUG(f"Memory cleanup: freed {freed_count} tensors, "
                  f"{freed_memory / 1024**2:.2f}MB, "
                  f"usage: {current_usage / 1024**2:.2f}MB -> {new_usage / 1024**2:.2f}MB")
    
    def _background_cleanup(self):
        """Background thread for periodic memory cleanup."""
        while not self.stop_cleanup.wait(self.cleanup_interval):
            try:
                self.cleanup_intermediate_tensors()
                self._detect_memory_leaks()
            except Exception as e:
                ERROR(f"Background cleanup error: {e}")
    
    def _detect_memory_leaks(self):
        """Detect potential memory leaks by monitoring tensor lifecycle."""
        current_usage = self.get_memory_usage()
        
        # Update peak memory
        if current_usage > self.memory_stats['peak_memory']:
            self.memory_stats['peak_memory'] = current_usage
        
        # Check for suspicious memory patterns
        memory_limit = self.get_memory_limit()
        if current_usage > memory_limit * 0.95:  # 95% threshold
            self.memory_stats['leak_preventions'] += 1
            ERROR(f"High memory usage detected: {current_usage / 1024**3:.2f}GB / "
                  f"{memory_limit / 1024**3:.2f}GB")
            self._emergency_cleanup()
    
    def _emergency_cleanup(self):
        """Emergency cleanup when memory is critically low."""
        DEBUG("Performing emergency memory cleanup...")
        
        # Clear all intermediate tensors
        self.intermediate_tensors.clear()
        self.tensor_registry.clear()
        
        # Force garbage collection and cache clearing
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_stats(self):
        """Get memory usage statistics."""
        return {
            **self.memory_stats,
            'current_usage_mb': self.get_memory_usage() / 1024**2,
            'memory_limit_mb': self.get_memory_limit() / 1024**2,
            'usage_percentage': (self.get_memory_usage() / self.get_memory_limit()) * 100
        }


class HardwareAdaptiveConfig:
    """
    Intelligent hardware detection and adaptive configuration for Arctic Architecture.
    Automatically adjusts tensor dimensions, layer depths, and gradient strategies based on available hardware.
    """
    
    def __init__(self):
        self.device_info = self._detect_hardware()
        self.adaptive_config = self._generate_adaptive_config()
        
        # Initialize memory manager
        self.memory_manager = MemoryManager()
        self.memory_manager.start_monitoring()
    
    def _detect_hardware(self):
        """Detect available hardware and compute capabilities."""
        import torch
        
        device_info = {
            'device_count': 0,
            'total_memory': 0,
            'device_names': [],
            'compute_capability': [],
            'is_cluster': False,
            'recommended_config': 'conservative'
        }
        
        if torch.cuda.is_available():
            device_info['device_count'] = torch.cuda.device_count()
            
            for i in range(device_info['device_count']):
                props = torch.cuda.get_device_properties(i)
                device_info['device_names'].append(props.name)
                device_info['total_memory'] += props.total_memory // (1024**3)  # GB
                device_info['compute_capability'].append((props.major, props.minor))
            
            # Determine configuration based on hardware
            if device_info['device_count'] >= 4:  # Multi-GPU cluster
                device_info['is_cluster'] = True
                if device_info['total_memory'] >= 320:  # 4x80GB A100 or better
                    device_info['recommended_config'] = 'maximum'
                elif device_info['total_memory'] >= 160:  # 4x40GB A100
                    device_info['recommended_config'] = 'high'
                else:
                    device_info['recommended_config'] = 'medium'
            elif device_info['device_count'] >= 2:  # Dual GPU
                if device_info['total_memory'] >= 160:  # 2x80GB A100
                    device_info['recommended_config'] = 'high'
                elif device_info['total_memory'] >= 80:   # 2x40GB A100
                    device_info['recommended_config'] = 'medium'
                else:
                    device_info['recommended_config'] = 'conservative'
            else:  # Single GPU
                if device_info['total_memory'] >= 80:    # Single 80GB A100
                    device_info['recommended_config'] = 'medium'
                elif device_info['total_memory'] >= 40:   # Single 40GB A100
                    device_info['recommended_config'] = 'conservative'
                else:
                    device_info['recommended_config'] = 'minimal'
        
        return device_info
    
    def _generate_adaptive_config(self):
        """Generate adaptive configuration based on detected hardware."""
        config_profiles = {
            'minimal': {  # RTX 4090, V100 16GB
                'max_layers': 2,
                'max_heads': 4,
                'max_lstm_layers': 1,
                'gradient_clip_norm': 0.5,
                'use_mixed_precision': True,
                'checkpoint_segments': 4
            },
            'conservative': {  # Single A100 40GB, V100×2
                'max_layers': 3,
                'max_heads': 8,
                'max_lstm_layers': 1,
                'gradient_clip_norm': 1.0,
                'use_mixed_precision': True,
                'checkpoint_segments': 2
            },
            'medium': {  # Single A100 80GB, A100×2 40GB
                'max_layers': 4,
                'max_heads': 12,
                'max_lstm_layers': 2,
                'gradient_clip_norm': 1.5,
                'use_mixed_precision': False,
                'checkpoint_segments': 1
            },
            'high': {  # A100×2 80GB, A100×4 40GB
                'max_layers': 6,
                'max_heads': 16,
                'max_lstm_layers': 3,
                'gradient_clip_norm': 2.0,
                'use_mixed_precision': False,
                'checkpoint_segments': 1
            },
            'maximum': {  # A100×4+ 80GB, H100 cluster
                'max_layers': 8,
                'max_heads': 24,
                'max_lstm_layers': 4,
                'gradient_clip_norm': 3.0,
                'use_mixed_precision': False,
                'checkpoint_segments': 1
            }
        }
        
        return config_profiles[self.device_info['recommended_config']]
    

    
    def get_gradient_config(self):
        """Get gradient configuration based on hardware."""
        return {
            'max_grad_norm': self.adaptive_config['gradient_clip_norm'],
            'use_mixed_precision': self.adaptive_config['use_mixed_precision'],
            'checkpoint_segments': self.adaptive_config['checkpoint_segments']
        }
    
    def __del__(self):
        """Cleanup memory manager on deletion."""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.stop_monitoring()
    
class DynamicModalFusion(nn.Module):
    """
    Advanced Arctic Architecture Modal Fusion with hierarchical attention and adaptive weighting.
    Implements multi-level fusion with temporal consistency and cross-modal reinforcement.
    """
    def __init__(self, cfg):
        """
        Initialize the DynamicModalFusion module.

        Args:
            cfg: Configuration object containing parameters such as hidden size, number of heads, etc.
        """
        super().__init__()
        self.cfg = cfg
        # Number of modalities: text, image, audio, video, document, agent
        self.num_modalities = 6
        self.hidden_size = cfg.hidden_size
        
        # Weight cache for common modality combinations
        self.weight_cache = {}
        self.cache_size_limit = 1000  # Limit cache size to prevent memory overflow
        self.cache_hit_threshold = 0.95  # Similarity threshold for cache hits
        
        # Initialize memory manager for dynamic memory optimization
        # If an external cache manager is provided, disable background thread to avoid interference
        self.cache_manager = cache_manager
        self.memory_manager = MemoryManager(enable_background=(cache_manager is None))
        self.memory_manager.start_monitoring()
        DEBUG("DynamicModalFusion initialized with advanced memory management")

        # Core module for native multimodal processing: Unified tokenization layer
        self.unified_tokenizer = nn.ModuleDict({
            # Text is already in token form, so use identity mapping
            'text': nn.Identity(),
            # Convert images into patches using 2D convolution
            'image': nn.Conv2d(3, self.hidden_size, 16, 16),
            # Convert audio into patches using 1D convolution
            'audio': nn.Conv1d(1, self.hidden_size, 16, 16),
            # Convert videos into 3D patches using 3D convolution
            'video': nn.Conv3d(3, self.hidden_size, (2, 16, 16), (2, 16, 16)),
            # Project document features to the hidden size
            'document': nn.Linear(768, self.hidden_size),
            # Convert handwritten data into patches using 2D convolution
            'handwriting': nn.Conv2d(1, self.hidden_size, 16, 16)
        })

        # Unified positional encoding for all modalities
        self.unified_pos_embed = nn.Parameter(torch.randn(1, 16384, self.hidden_size) * 0.02)

        # Modality type embedding
        self.modality_tokens = nn.Embedding(6, self.hidden_size)

        # Native multimodal attention for cross-modal token-level fusion
        self.native_attention = nn.ModuleDict({
            'cross_modal': nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=cfg.n_head,
                batch_first=True,
                dropout=0.1
            ),
            'self_modal': nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=cfg.n_head // 2,
                batch_first=True,
                dropout=0.1
            )
        })

        # Dual gating mechanism: Integration of understanding and generation
        self.modality_gates = nn.ModuleDict({
            # Unified understanding gate for cross-modal information fusion
            'understanding': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            ),
            # Generation gates for independent generation control of each modality
            'gen_text': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.Sigmoid()
            ),
            'gen_image': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.Sigmoid()
            ),
            'gen_audio': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.Sigmoid()
            ),
            'gen_video': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.Sigmoid()
            ),
            'gen_document': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.Sigmoid()
            ),
            'gen_agent': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.Sigmoid()
            )
        })

        # Initialize hardware-adaptive configuration
        self.hw_config = HardwareAdaptiveConfig()


        self.max_layers = self.hw_config.adaptive_config['max_layers']
        self.max_heads = min(self.hw_config.adaptive_config['max_heads'], cfg.n_head)

        # Gradient configuration
        self.gradient_config = self.hw_config.get_gradient_config()
        self.max_grad_norm = self.gradient_config['max_grad_norm']

        # === Enhanced Hierarchical Fusion Architecture ===
        # Multi-level weight prediction with attention redistribution
        self.primary_weight_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_modalities, self.hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(getattr(cfg, 'fusion_dropout', 0.1)),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.num_modalities),
            nn.Softmax(dim=-1)
        )

        # Revolutionary Multi-Scale Weight Refinement Network
        self.multi_scale_weight_refiner = nn.ModuleDict({
            # Local weight refinement
            'local_refiner': nn.Sequential(
                nn.Linear(self.num_modalities * 2, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.num_modalities),
                nn.Sigmoid()
            ),
            # Global weight refinement
            'global_refiner': nn.Sequential(
                nn.Linear(self.num_modalities * 3, self.hidden_size),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.LayerNorm(self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.num_modalities),
                nn.Sigmoid()
            ),
            # Temporal weight refinement using LSTM
            'temporal_refiner': nn.LSTM(
                input_size=self.num_modalities,
                hidden_size=self.num_modalities * 2,
                num_layers=3,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            )
        })

        # Advanced Temporal Consistency Enforcer with Memory
        # Register memory_bank parameter separately
        # Temporal memory for storing modality information
        self.memory_bank = nn.Parameter(torch.randn(100, self.num_modalities))

        self.temporal_consistency = nn.ModuleDict({
            # LSTM for maintaining temporal consistency
            'consistency_lstm': nn.LSTM(
                input_size=self.num_modalities,
                hidden_size=self.num_modalities * 2,
                num_layers=3,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            ),
            # Attention mechanism for memory interaction
            'memory_attention': nn.MultiheadAttention(
                embed_dim=self.num_modalities,
                num_heads=min(self.num_modalities, 8),
                batch_first=True,
                dropout=0.1
            ),
            # Score calculation for temporal consistency
            'consistency_scorer': nn.Sequential(
                nn.Linear(self.num_modalities * 4, self.hidden_size),  # LSTM + Memory
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.num_modalities),
                nn.Sigmoid()
            )
        })

        # Revolutionary Cross-modal Reinforcement Attention Network
        self.reinforcement_attention = nn.ModuleDict({
            # Primary attention for cross-modal interaction
            'primary_attention': nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=cfg.n_head // 2,
                batch_first=True,
                dropout=0.1
            ),
            # Secondary attention for further refinement
            'secondary_attention': nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=cfg.n_head // 4,
                batch_first=True,
                dropout=0.1
            ),
            # Gate for reinforcement control
            'reinforcement_gate': nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Sigmoid()
            ),
            # Module for fusing primary and secondary attention
            'attention_fusion': nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size)
            )
        })

        # Super-Enhanced Context Encoder with Hierarchical Processing
        self.context_encoder = nn.ModuleDict({
            # Micro-context encoding
            'micro_context': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.num_modalities)
            ),
            # Local context encoding
            'local_context': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.num_modalities)
            ),
            # Global context encoding
            'global_context': nn.Sequential(
                nn.Linear(self.hidden_size * self.num_modalities, self.hidden_size * 2),
                nn.LayerNorm(self.hidden_size * 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.num_modalities)
            ),
            # Cross-modal context encoding
            'cross_modal_context': nn.Sequential(
                nn.Linear(self.hidden_size * 15, self.hidden_size * 4),  # 15 cross-modal pairs
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                nn.LayerNorm(self.hidden_size * 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size * 2, self.num_modalities)
            ),
            # Semantic context encoding using Transformer
            'semantic_context': nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=cfg.n_head // 4,
                dim_feedforward=self.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            )
        })

        # Hardware-Adaptive Cross-modal Attention Layers
        # Adjust number of cross-modal pairs based on hardware capacity
        if self.hw_config.device_info['recommended_config'] in ['maximum', 'high']:
            # Full 15-layer cross-modal attention for high-end hardware
            cross_modal_pairs = [
                'text_image', 'text_audio', 'text_video', 'text_document', 'text_agent',
                'image_audio', 'image_video', 'image_document', 'image_agent',
                'audio_video', 'audio_document', 'audio_agent',
                'video_document', 'video_agent', 'document_agent'
            ]
        elif self.hw_config.device_info['recommended_config'] == 'medium':
            # Reduced to 9 key cross-modal pairs for medium hardware
            cross_modal_pairs = [
                'text_image', 'text_audio', 'text_video', 'text_agent',
                'image_audio', 'image_video', 'image_agent',
                'audio_video', 'video_agent'
            ]
        else:
            # Minimal 6 cross-modal pairs for conservative hardware
            cross_modal_pairs = [
                'text_image', 'text_audio', 'text_agent',
                'image_audio', 'image_agent', 'audio_agent'
            ]

        # Store the pair configuration and use a single shared attention to reduce redundancy
        self.cross_modal_pairs = cross_modal_pairs
        self.shared_cross_modal_attn = CrossModalAttention(cfg)

        # Modality-specific enhancement networks
        self.modality_enhancers = nn.ModuleDict({
            modal: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.SiLU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            ) for modal in ['text', 'image', 'audio', 'video', 'document', 'agent']
        })

        # Hardware-Adaptive Fusion Gate System
        num_cross_modal_pairs = len(self.cross_modal_pairs)

        # Adjust gate complexity based on hardware
        if self.hw_config.device_info['recommended_config'] in ['maximum', 'high']:
            # Complex multi-level gating for high-end hardware
            self.adaptive_fusion_gate = nn.ModuleDict({
                'primary': nn.Sequential(
                    nn.Linear(self.hidden_size * (self.num_modalities + num_cross_modal_pairs), self.hidden_size * 3),
                    nn.SiLU(),
                    nn.Dropout(getattr(cfg, 'fusion_dropout', 0.1)),
                    nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                    nn.LayerNorm(self.hidden_size * 2),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.Sigmoid()
                ),
                'quality_aware': nn.Sequential(
                    nn.Linear(self.num_modalities * 2, self.hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.Sigmoid()
                ),
                'temporal_gate': nn.Sequential(
                    nn.Linear(self.num_modalities, self.hidden_size // 2),
                    nn.SiLU(),
                    nn.Linear(self.hidden_size // 2, self.hidden_size),
                    nn.Sigmoid()
                )
            })
        else:
            # Simplified gating for lower-end hardware
            self.adaptive_fusion_gate = nn.ModuleDict({
                'primary': nn.Sequential(
                    nn.Linear(self.hidden_size * self.num_modalities, self.hidden_size),
                    nn.LayerNorm(self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.Sigmoid()
                ),
                'quality_aware': nn.Sequential(
                    nn.Linear(self.num_modalities, self.hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size // 2, self.hidden_size),
                    nn.Sigmoid()
                )
            })
        # Unified tokenization configuration
        self.token_config = {
            'image_patch_size': 16,
            'audio_patch_size': 16,
            'video_patch_size': 16,
            'max_patches_per_modality': 256
        }

        # Native multimodal attention configuration
        self.cross_modal_attention = nn.ModuleDict({
            # Unified query/key/value projection
            'q_proj': nn.Linear(self.hidden_size, self.hidden_size),
            'k_proj': nn.Linear(self.hidden_size, self.hidden_size),
            'v_proj': nn.Linear(self.hidden_size, self.hidden_size),
            'output_proj': nn.Linear(self.hidden_size, self.hidden_size),
            'layer_norm': nn.LayerNorm(self.hidden_size),
            'attention_dropout': nn.Dropout(0.1)
        })

        # Modality type embedding
        self.modal_type_embedding = nn.Embedding(self.num_modalities, self.hidden_size)
        
        # Unified positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.token_config['max_patches_per_modality'] * self.num_modalities, self.hidden_size) * 0.02
        )

        # Dual gating mechanism
        self.dual_gating = nn.ModuleDict({
            'understanding_gate': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.Sigmoid()
            ),
            'generation_gate': nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(self.hidden_size // 2, self.hidden_size),
                nn.Sigmoid()
            )
        })

        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def _get_modality_quality_scores(self, features):
        """
        Calculate quality scores for each modality based on content.

        Args:
            features (dict): Dictionary containing features from different modalities.

        Returns:
            dict: Quality scores for each modality.
        """
        quality_scores = {}
        device = next(iter(features.values())).device if features else torch.device('cpu')
        
        for modality, feat in features.items():
            if feat is not None and feat.numel() > 0:
                # Calculate the mean of features along the sequence dimension
                feat_mean = feat.mean(dim=1)  # Shape: [batch, hidden]
                # Calculate the variance of the mean features across the feature dimension
                feat_var = torch.var(feat_mean, dim=-1)
                # Normalize the variance to [0, 1] using sigmoid function
                quality = torch.sigmoid(feat_var).mean()
                quality_scores[modality] = quality
            else:
                # Assign a low quality score for missing modalities
                quality_scores[modality] = torch.tensor(0.1, device=device)
        
        return quality_scores
    
    def _get_cached_weights(self, modal_features, quality_scores):
        """
        Get cached weights for common modality combinations.
        
        Args:
            modal_features (dict): Dictionary of modal features.
            quality_scores (dict): Quality scores for each modality.
            
        Returns:
            torch.Tensor or None: Cached weights if found, None otherwise.
        """
        if not self.weight_cache:
            return None
            
        # Create a signature for the current modality combination
        modal_signature = []
        for modal in ['text', 'image', 'audio', 'video', 'document', 'agent']:
            if modal in modal_features and modal_features[modal] is not None:
                modal_signature.append(f"{modal}:1")
            else:
                modal_signature.append(f"{modal}:0")
        
        # Add quality score information
        quality_signature = []
        for modal in ['text', 'image', 'audio', 'video', 'document', 'agent']:
            if modal in quality_scores:
                quality_val = quality_scores[modal].item() if torch.is_tensor(quality_scores[modal]) else float(quality_scores[modal])
                quality_signature.append(f"{modal}:{quality_val:.2f}")
            else:
                quality_signature.append(f"{modal}:0.00")
        
        full_signature = "_".join(modal_signature + quality_signature)
        
        # Check for similar combinations in cache
        if full_signature in self.weight_cache:
            return self.weight_cache[full_signature]
            
        # Check for similar patterns with relaxed matching
        for cached_sig, cached_weights in self.weight_cache.items():
            if self._is_similar_signature(full_signature, cached_sig):
                return cached_weights
                
        return None
    
    def _cache_weights(self, modal_features, quality_scores, weights):
        """
        Cache weights for common modality combinations.
        
        Args:
            modal_features (dict): Dictionary of modal features.
            quality_scores (dict): Quality scores for each modality.
            weights (torch.Tensor): Weights to cache.
        """
        # Create a signature for the current modality combination
        modal_signature = []
        for modal in ['text', 'image', 'audio', 'video', 'document', 'agent']:
            if modal in modal_features and modal_features[modal] is not None:
                modal_signature.append(f"{modal}:1")
            else:
                modal_signature.append(f"{modal}:0")
        
        # Add quality score information
        quality_signature = []
        for modal in ['text', 'image', 'audio', 'video', 'document', 'agent']:
            if modal in quality_scores:
                quality_val = quality_scores[modal].item() if torch.is_tensor(quality_scores[modal]) else float(quality_scores[modal])
                quality_signature.append(f"{modal}:{quality_val:.2f}")
            else:
                quality_signature.append(f"{modal}:0.00")
        
        full_signature = "_".join(modal_signature + quality_signature)
        
        # Cache the weights
        self.weight_cache[full_signature] = weights.detach().clone()
        
        # Limit cache size to prevent memory overflow
        if len(self.weight_cache) > self.cache_size_limit:
            # Remove oldest entries
            oldest_keys = list(self.weight_cache.keys())[:len(self.weight_cache) - self.cache_size_limit + 100]
            for key in oldest_keys:
                del self.weight_cache[key]
    
    def _is_similar_signature(self, sig1, sig2):
        """
        Check if two signatures are similar enough for cache matching.
        
        Args:
            sig1 (str): First signature.
            sig2 (str): Second signature.
            
        Returns:
            bool: True if signatures are similar, False otherwise.
        """
        # Split signatures into modal and quality parts
        parts1 = sig1.split('_')
        parts2 = sig2.split('_')
        
        if len(parts1) != len(parts2):
            return False
            
        # Check modal parts (exact match)
        modal_parts1 = parts1[:6]
        modal_parts2 = parts2[:6]
        if modal_parts1 != modal_parts2:
            return False
            
        # Check quality parts (relaxed matching)
        quality_parts1 = parts1[6:]
        quality_parts2 = parts2[6:]
        
        similar_count = 0
        for q1, q2 in zip(quality_parts1, quality_parts2):
            # Extract quality values
            val1 = float(q1.split(':')[1])
            val2 = float(q2.split(':')[1])
            
            # Check if quality values are similar
            if abs(val1 - val2) < 0.1:
                similar_count += 1
        
        return similar_count >= 5  # At least 5 out of 6 quality scores should be similar

    def _predict_enhanced_weights(self, modal_features, quality_scores, temporal_context=None):
        """
        Enhanced weight prediction with multi-level context and temporal consistency.

        Args:
            modal_features (dict): Dictionary of modal features.
            quality_scores (dict): Quality scores for each modality.
            temporal_context (torch.Tensor, optional): Optional temporal context for consistency.

        Returns:
            tuple: Enhanced dynamic weights with temporal consistency and a dictionary of intermediate results.
        """
        # Register input tensors for memory monitoring
        if hasattr(self, 'memory_manager'):
            for modal_name, feat in modal_features.items():
                if feat is not None:
                    self.memory_manager.register_tensor(feat, f"modal_feature_{modal_name}")
        
        # Extract and normalize features
        feature_list = []
        modal_names = ['text', 'image', 'audio', 'video', 'document', 'agent']
        batch_size = 1
        
        # Determine the batch size from the first valid feature
        for modal in modal_names:
            feat = modal_features.get(modal)
            if feat is not None and feat.numel() > 0:
                batch_size = feat.shape[0]
                break
        
        for modal in modal_names:
            feat = modal_features.get(modal, torch.zeros(batch_size, 1, self.hidden_size, device=next(self.parameters()).device))
            if feat.dim() == 3:
                # Pool the sequence dimension by taking the mean
                feat = feat.mean(dim=1)
            elif feat.dim() == 2:
                feat = feat
            else:
                # Flatten the feature tensor starting from the second dimension
                feat = feat.flatten(start_dim=1)
            feature_list.append(feat)
        
        # Concatenate all features along the last dimension
        all_features = torch.cat(feature_list, dim=-1)
        
        # Register concatenated features for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(all_features, "concatenated_features")
        
        # Check cache for existing weights
        cached_weights = self._get_cached_weights(modal_features, quality_scores)
        if cached_weights is not None:
            # Use cached weights for efficiency
            final_weights = cached_weights
            primary_weights = cached_weights  # For return compatibility
        else:
            # Predict primary weights using the primary weight predictor
            primary_weights = self.primary_weight_predictor(all_features)
        
        # Encode local context for each modality
        local_contexts = []
        for i, feat in enumerate(feature_list):
            local_ctx = self.context_encoder['local'](feat)
            local_contexts.append(local_ctx)
        
        # Encode global context from all modalities
        global_context = self.context_encoder['global'](all_features)
        
        # Combine primary weights with quality scores
        quality_tensor = torch.zeros(batch_size, self.num_modalities, device=next(self.parameters()).device)
        for i, modal in enumerate(modal_names):
            if modal in quality_scores:
                quality_val = quality_scores[modal]
                if torch.is_tensor(quality_val):
                    quality_tensor[:, i] = quality_val.expand(batch_size)
                else:
                    quality_tensor[:, i] = float(quality_val)
            else:
                # Set default quality score if not available
                quality_tensor[:, i] = 0.5
        
        # Perform quality-aware refinement
        weight_quality_combined = torch.cat([primary_weights, quality_tensor], dim=-1)
        refinement_multipliers = self.weight_refiner(weight_quality_combined)
        
        # Apply refinement to primary weights
        refined_weights = primary_weights * refinement_multipliers
        
        # Ensure temporal consistency if temporal context is available
        if temporal_context is not None:
            # Add a temporal dimension to the refined weights
            temporal_input = refined_weights.unsqueeze(1)
            # Apply LSTM to maintain temporal consistency
            consistent_weights, _ = self.temporal_consistency['consistency_lstm'](temporal_input)
            refined_weights = consistent_weights.squeeze(1)
        
        # Normalize the refined weights using softmax
        final_weights = F.softmax(refined_weights, dim=-1)
        
        # Cache the computed weights for future use
        if cached_weights is None:
            self._cache_weights(modal_features, quality_scores, final_weights)
        
        # Register output tensors for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(final_weights, "final_weights")
            self.memory_manager.register_tensor(primary_weights, "primary_weights")
            self.memory_manager.register_tensor(refined_weights, "refined_weights")
        
        return final_weights, {
            'primary': primary_weights,
            'quality_scores': quality_tensor,
            'refinement': refinement_multipliers,
            'local_contexts': local_contexts,
            'global_context': global_context
        }

    def _hierarchical_cross_modal_attention(self, modal_features):
        """
        Apply hierarchical cross-modal attention with all modality pairs.

        Args:
            modal_features (dict): Dictionary of modal features.

        Returns:
            tuple: Enhanced cross-modal features and cross-attention results.
        """
        # Register input tensors for memory monitoring
        if hasattr(self, 'memory_manager'):
            for modal_name, feat in modal_features.items():
                if feat is not None:
                    self.memory_manager.register_tensor(feat, f"cross_modal_input_{modal_name}")
        
        modal_names = ['text', 'image', 'audio', 'video', 'document', 'agent']
        enhanced_features = {}
        cross_attentions = {}
        
        # First enhance individual modalities
        for modal in modal_names:
            feat = modal_features.get(modal)
            if feat is not None and feat.numel() > 0:
                # Enhance the modality features by taking the mean and passing through the modality enhancer
                enhanced_features[modal] = self.modality_enhancers[modal](feat.mean(dim=1))
            else:
                # Initialize zero tensor for missing modalities
                enhanced_features[modal] = torch.zeros(1, self.hidden_size, device=next(self.parameters()).device)
        
        # Register enhanced features for memory monitoring
        if hasattr(self, 'memory_manager'):
            for modal_name, enhanced_feat in enhanced_features.items():
                self.memory_manager.register_tensor(enhanced_feat, f"enhanced_feature_{modal_name}")
        
        # Compute all cross-modal attention pairs
        modality_pairs = [
            ('text', 'image'), ('text', 'audio'), ('text', 'video'), 
            ('text', 'document'), ('text', 'agent'), ('image', 'audio'),
            ('image', 'video'), ('image', 'document'), ('image', 'agent'),
            ('audio', 'video'), ('audio', 'document'), ('audio', 'agent'),
            ('video', 'document'), ('video', 'agent'), ('document', 'agent')
        ]
        
        for mod1, mod2 in modality_pairs:
            key = f'{mod1}_{mod2}'
            if key in getattr(self, 'cross_modal_pairs', []):
                # Prepare features for cross-attention by adding a sequence dimension
                feat1 = enhanced_features[mod1].unsqueeze(1)
                feat2 = enhanced_features[mod2].unsqueeze(1)

                # Apply cross-modal attention via the shared attention module
                cross_att = self.shared_cross_modal_attn(feat1, feat2, feat2)
                cross_attentions[key] = cross_att

                # Register cross-attention output for memory monitoring
                if hasattr(self, 'memory_manager'):
                    self.memory_manager.register_tensor(cross_att, f"cross_attention_{key}")
        
        return enhanced_features, cross_attentions

    def forward(self, modal_features):
        """
        Native multimodal fusion forward pass.

        Implement true token-level native multimodal fusion. All modalities are unified into tokens and cross-modal fusion is performed at the token level.

        Args:
            modal_features (dict): Dictionary containing features from 6 modalities:
                'text': Text tokens [batch, seq_len, hidden]
                'image': Image features [batch, channels, height, width]
                'audio': Audio features [batch, channels, time, freq]
                'video': Video features [batch, channels, frames, height, width]
                'document': Document features [batch, seq_len, hidden]
                'agent': Agent features [batch, seq_len, hidden]

        Returns:
            torch.Tensor: Fused multimodal representation with native multimodal fusion [batch, 1, hidden]
        """
        # Register input tensors for memory monitoring
        if hasattr(self, 'memory_manager'):
            for modal_name, modal_tensor in modal_features.items():
                if modal_tensor is not None:
                    self.memory_manager.register_tensor(modal_tensor, f"forward_input_{modal_name}")
        
        # Hardware-adaptive gradient safety pre-check
        if self.training:
            # Clear the gradient of the gradient monitor
            self.gradient_monitor.grad = None
            
            # Apply adaptive mixed precision based on hardware configuration
            if self.gradient_config['use_mixed_precision']:
                with torch.cuda.amp.autocast():
                    result = self._forward_impl(modal_features)
            else:
                result = self._forward_impl(modal_features)
        else:
            result = self._forward_impl(modal_features)
        
        # Register output tensor for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(result, "forward_output")
        
        return result
    
    def _forward_impl(self, modal_features):
        """
        Native multimodal Early Fusion implementation.

        Args:
            modal_features (dict): Dictionary containing features from different modalities.

        Returns:
            torch.Tensor: Global multimodal representation after fusion.
        """
        # Native Early Fusion: Unify all modalities into tokens
        unified_tokens = []
        token_type_ids = []  # Modality type identifiers
        
        modal_names = ['text', 'image', 'audio', 'video', 'document', 'agent']
        
        # Unified patch encoding
        for idx, modal in enumerate(modal_names):
            feat = modal_features.get(modal)
            if feat is not None and feat.numel() > 0:
                # Native multimodal encoder (similar to MetaCLIP)
                if modal == 'text':
                    # Pass text tokens directly
                    tokens = feat
                elif modal == 'image':
                    # Patchify images into 16x16 patches
                    patches = self.unified_tokenizer['image'](feat)  # Shape: [B, C, H, W] -> [B, hidden, H/16, W/16]
                    tokens = patches.flatten(2).transpose(1, 2)  # Shape: [B, num_patches, hidden]
                elif modal == 'audio':
                    # Patchify audio spectrograms
                    spec = self.unified_tokenizer['audio'](feat)  # Shape: [B, 1, T] -> [B, hidden, T']
                    tokens = spec.transpose(1, 2)  # Shape: [B, T', hidden]
                elif modal == 'video':
                    # 3D patchify videos into spatio-temporal patches
                    batch_size, channels, frames, height, width = feat.shape
                    feat = feat.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
                    patches = self.unified_tokenizer['video'](feat)
                    tokens = patches.flatten(2).transpose(1, 2)
                    tokens = tokens.reshape(batch_size, frames * tokens.shape[1], self.hidden_size)
                elif modal == 'document':
                    # Layout-aware encoding for documents
                    tokens = self.unified_tokenizer['document'](feat)
                elif modal == 'agent':
                    # State encoding for agents
                    tokens = self.unified_tokenizer['agent'](feat)
                
                # Unified positional encoding (absolute + relative)
                seq_len = tokens.shape[1]
                pos_embed = self.unified_pos_embed[:, :seq_len, :]
                tokens = tokens + pos_embed
                
                # Modality type embedding (similar to segment embedding)
                modal_embed = self.modality_tokens(torch.tensor(idx, device=tokens.device))
                tokens = tokens + modal_embed.unsqueeze(0).unsqueeze(0)
                
                unified_tokens.append(tokens)
                token_type_ids.extend([idx] * seq_len)
        
        if not unified_tokens:
            return torch.zeros(1, 1, self.hidden_size, device=next(self.parameters()).device)
        
        # Native multimodal sequence: Concatenate all tokens
        multimodal_sequence = torch.cat(unified_tokens, dim=1)
        token_type_tensor = torch.tensor(token_type_ids, device=multimodal_sequence.device)
        
        # Register tensors for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(multimodal_sequence, "multimodal_sequence")
            self.memory_manager.register_tensor(token_type_tensor, "token_type_tensor")
        
        # Unified attention mechanism
        # Native cross-modal attention: All tokens attend to each other
        attended_sequence, _ = self.native_attention['cross_modal'](
            multimodal_sequence, multimodal_sequence, multimodal_sequence
        )
        
        # Register attention output for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(attended_sequence, "attended_sequence")
        
        # Modality-aware gating (understanding + generation dual control)
        # Understanding gate: Integrate cross-modal information
        understanding_gate = torch.sigmoid(
            self.modality_gates['understanding'](attended_sequence.mean(dim=1))
        ).unsqueeze(1)
        
        # Generation gates: Control generation for each modality
        generation_gates = {}
        for modal in modal_names:
            generation_gates[modal] = torch.sigmoid(
                self.modality_gates[f'gen_{modal}'](attended_sequence.mean(dim=1))
            ).unsqueeze(1)
        
        # Dual fusion output
        # Understanding output: Unified multimodal representation
        understanding_output = attended_sequence * understanding_gate
        
        # Generation output: Independent generation paths for each modality
        generation_outputs = {}
        current_pos = 0
        for idx, tokens in enumerate(unified_tokens):
            if tokens is not None:
                seq_len = tokens.shape[1]
                modal_tokens = attended_sequence[:, current_pos:current_pos+seq_len, :]
                
                # Enhance generation for specific modalities
                gen_gate = generation_gates[modal_names[idx]]
                enhanced_tokens = modal_tokens * gen_gate
                
                # Save the modality-specific output
                generation_outputs[modal_names[idx]] = enhanced_tokens
                current_pos += seq_len
        
        # Final native fusion: Combine understanding and generation representations
        # Understanding representation: Global multimodal understanding
        global_representation = understanding_output.mean(dim=1, keepdim=True)
        
        # Register final representations for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(global_representation, "global_representation")
            for modal_name, modal_output in generation_outputs.items():
                self.memory_manager.register_tensor(modal_output, f"generation_output_{modal_name}")
        
        # Generation representation: Specific generation capabilities for each modality
        # Use unified cache manager if available, otherwise use local cache
        if hasattr(self, 'cache_manager'):
            for modality, cache_data in generation_outputs.items():
                self.cache_manager.set_generation_cache(modality, cache_data)
        else:
            self.generation_cache = generation_outputs  # Local cache fallback
        
        # Return the native fused representation
        return global_representation
    
    def generate_modality(self, target_modal, prompt_tokens=None, temperature=1.0, top_k=None):
        """
        Generate content for a specified modality based on unified multimodal understanding.
        This is a core feature of PiscesL1: integration of understanding and generation,
        where all modalities share the same semantic space.

        Args:
            target_modal (str): Target generation modality ('text', 'image', 'audio', 'video', 'document', 'agent').
            prompt_tokens (torch.Tensor, optional): Optional prompt tokens [batch, seq_len, hidden].
            temperature (float, optional): Generation temperature control. Defaults to 1.0.
            top_k (int, optional): Top-k sampling. Defaults to None.

        Returns:
            torch.Tensor: Modality-specific generation representation [batch, seq_len, hidden].
        """
        # Get generation cache from unified cache manager or local cache
        if hasattr(self, 'cache_manager'):
            generation_cache = {}
            for modality in ['text', 'image', 'audio', 'video', 'document', 'agent']:
                cached_data = self.cache_manager.get_generation_cache(modality)
                if cached_data is not None:
                    generation_cache[modality] = cached_data
        else:
            generation_cache = getattr(self, 'generation_cache', {})
        
        if not generation_cache:
            raise ValueError("Must call forward() first to build generation cache")
        
        # Register input tensors for memory monitoring
        if hasattr(self, 'memory_manager'):
            if prompt_tokens is not None:
                self.memory_manager.register_tensor(prompt_tokens, f"gen_prompt_{target_modal}")
        
        # Native generation: Modality-specific generation based on unified understanding
        device = next(self.parameters()).device
        
        # Get the base generation representation
        if target_modal in generation_cache:
            base_generation = generation_cache[target_modal]
        else:
            # Use the general understanding representation if no specific modality is available
            base_generation = self.cached_understanding if hasattr(self, 'cached_understanding') else torch.zeros(1, 1, self.hidden_size, device=device)
        
        # Register base generation tensor for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(base_generation, f"gen_base_{target_modal}")
        
        # Conditional generation: Guided by prompts
        if prompt_tokens is not None:
            # Fuse the prompt with the base generation representation
            prompt_mean = prompt_tokens.mean(dim=1, keepdim=True)
            
            # Use the understanding gate for conditional control
            condition_gate = self.modality_gates['understanding'](prompt_mean.squeeze(1))
            conditioned = base_generation * condition_gate.unsqueeze(1)
            
            # Add prompt information
            prompt_weight = 0.3
            enhanced_generation = conditioned + prompt_mean * prompt_weight
        else:
            enhanced_generation = base_generation
        
        # Temperature control
        if temperature != 1.0:
            enhanced_generation = enhanced_generation / temperature
        
        # Top-k processing (if applicable)
        if top_k is not None and enhanced_generation.shape[-1] > top_k:
            # Select top-k values along the feature dimension
            top_values, top_indices = torch.topk(enhanced_generation, k=top_k, dim=-1)
            mask = torch.zeros_like(enhanced_generation)
            mask.scatter_(-1, top_indices, top_values)
            enhanced_generation = mask
        
        # Register final generation output for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(enhanced_generation, f"gen_output_{target_modal}")
        
        return enhanced_generation
    
    def generate_cross_modal(self, source_modal, target_modal, source_tokens=None):
        """
        Implement true cross-modal understanding-generation, e.g., text -> image, image -> text, etc.

        Args:
            source_modal (str): Source modality.
            target_modal (str): Target modality.
            source_tokens (torch.Tensor, optional): Source modality tokens.

        Returns:
            torch.Tensor: Cross-modal generation representation.
        """
        # First perform unified understanding
        dummy_features = {source_modal: source_tokens} if source_tokens is not None else {}
        unified_rep = self.forward(dummy_features)
        
        # Then generate content for the target modality
        return self.generate_modality(target_modal, prompt_tokens=source_tokens)
    
    def generate_image(self, condition: GenerationCondition, steps: int = 50) -> torch.Tensor:
        """Generate image from conditions"""
        latent = self.encode_conditions(condition)
        
        # Register input latent for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(latent, "gen_image_latent")
        
        # Diffusion process (simplified)
        noise = torch.randn(1, 3, 512, 512)
        for t in range(steps):
            timestep = torch.tensor([t / steps])
            time_emb = self.time_embedding(timestep.view(1, 1).expand(1, self.hidden_size))
            conditioned_latent = latent + time_emb
            
            # Generate step
            decoded = self.image_decoder(conditioned_latent)
            noise = noise * 0.98 + decoded * 0.02
        
        final_image = torch.clamp(noise, -1, 1)
        
        # Register final output for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(final_image, "gen_image_output")
        
        return final_image
    
    def generate_video(self, condition: GenerationCondition, frames: int = 16, steps: int = 50) -> torch.Tensor:
        """Generate video from conditions"""
        latent = self.encode_conditions(condition)
        
        # Register input latent for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(latent, "gen_video_latent")
        
        # Temporal consistency
        video_latent = latent.unsqueeze(1).expand(1, frames, self.hidden_size)
        
        # Register video latent for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(video_latent, "gen_video_temporal_latent")
        
        # Diffusion process
        noise = torch.randn(1, 3, frames, 64, 64)
        for t in range(steps):
            timestep = torch.tensor([t / steps])
            time_emb = self.time_embedding(timestep.view(1, 1).expand(1, self.hidden_size))
            conditioned_latent = video_latent + time_emb.unsqueeze(1)
            
            # Generate step with temporal smoothing
            decoded = self.video_decoder(conditioned_latent.reshape(-1, self.hidden_size))
            decoded = decoded.view(1, 3, frames, 64, 64)
            noise = noise * 0.98 + decoded * 0.02
        
        final_video = torch.clamp(noise, -1, 1)
        
        # Register final output for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(final_video, "gen_video_output")
        
        return final_video
    
    def generate_audio(self, condition: GenerationCondition, duration: float = 3.0, steps: int = 50) -> torch.Tensor:
        """Generate audio from conditions with style control"""
        latent = self.encode_conditions(condition)
        
        # Register input latent for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(latent, "gen_audio_latent")
        
        # Calculate sequence length (44.1kHz * duration)
        seq_len = int(44100 * duration / 256)  # Downsampled
        
        # Diffusion process with style control
        noise = torch.randn(1, 1, seq_len)
        for t in range(steps):
            timestep = torch.tensor([t / steps])
            time_emb = self.time_embedding(timestep.view(1, 1).expand(1, self.hidden_size))
            conditioned_latent = latent + time_emb
            
            # Generate step
            decoded = self.audio_decoder(conditioned_latent)
            decoded = F.interpolate(decoded, size=seq_len)
            
            # Apply audio style control
            if condition.style_params:
                decoded = self.apply_audio_style(decoded, condition.style_params)
            
            noise = noise * 0.98 + decoded * 0.02
        
        final_audio = torch.clamp(noise, -1, 1)
        
        # Register final output for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(final_audio, "gen_audio_output")
        
        return final_audio
    
    def generate_document(self, condition: GenerationCondition, max_length: int = 1000, steps: int = 50) -> Dict[str, torch.Tensor]:
        """Generate structured documents with full formatting control"""
        latent = self.encode_conditions(condition)
        
        # Register input latent for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(latent, "gen_document_latent")
        
        # Document generation with structure and style
        document_outputs = {}
        
        # Generate content tokens
        content_tokens = torch.randn(1, max_length, 8192)
        for t in range(steps):
            timestep = torch.tensor([t / steps])
            time_emb = self.time_embedding(timestep.view(1, 1).expand(1, self.hidden_size))
            conditioned_latent = latent + time_emb
            
            # Generate content step
            content_step = self.document_decoder['content'](conditioned_latent)
            content_tokens = content_tokens * 0.95 + content_step.unsqueeze(1).expand(-1, max_length, -1) * 0.05
        
        document_outputs['content'] = torch.clamp(content_tokens, -1, 1)
        
        # Register content tokens for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(document_outputs['content'], "gen_document_content")
        
        # Generate structure
        structure = self.document_decoder['structure'](latent)
        document_outputs['structure'] = structure
        
        # Register structure for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(document_outputs['structure'], "gen_document_structure")
        
        # Generate style
        style = self.document_decoder['style'](latent)
        document_outputs['style'] = style
        
        # Register style for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(document_outputs['style'], "gen_document_style")
        
        # Generate format control
        format_control = self.document_decoder['format'](latent)
        document_outputs['format'] = format_control
        
        # Register format control for memory monitoring
        if hasattr(self, 'memory_manager'):
            self.memory_manager.register_tensor(document_outputs['format'], "gen_document_format")
        
        return document_outputs
    
    def apply_audio_style(self, audio: torch.Tensor, style_params: Dict[str, float]) -> torch.Tensor:
        """Apply audio style controls"""
        if not style_params:
            return audio
        
        # Apply pitch shift
        if 'pitch' in style_params:
            pitch_factor = style_params['pitch']
            # Simplified pitch control via interpolation
            audio = F.interpolate(audio, scale_factor=1 + (pitch_factor - 0.5) * 0.5)
        
        # Apply tempo control
        if 'tempo' in style_params:
            tempo_factor = style_params['tempo']
            audio = F.interpolate(audio, scale_factor=1 + (tempo_factor - 0.5) * 0.3)
        
        # Apply volume control
        if 'volume' in style_params:
            volume_factor = style_params['volume']
            audio = audio * volume_factor
        
        return torch.clamp(audio, -1, 1)
    
    def apply_document_style(self, document: Dict[str, torch.Tensor], style_params: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Apply document style controls"""
        if not style_params:
            return document
        
        # Apply formality control
        if 'formality' in style_params:
            formality = style_params['formality']
            document['content'] = document['content'] * (0.5 + formality * 0.5)
        # Apply tone control
        if 'tone' in style_params:
            tone = style_params['tone']
            tone_embedding = torch.tensor([tone] * 256)
            document['style'] = (document['style'] + tone_embedding) / 2
        
        return document
    
    def apply_style_control(self, generated: torch.Tensor, style_params: Dict[str, float]) -> torch.Tensor:
        """Apply style controls to generated content"""
        if not style_params:
            return generated
        
        # Convert style parameters to latent
        style_vector = torch.tensor([[
            style_params.get('brightness', 0.5),
            style_params.get('contrast', 0.5),
            style_params.get('saturation', 0.5),
            style_params.get('sharpness', 0.5)
        ]])
        
        # Apply style adjustments
        adjusted = generated
        if 'brightness' in style_params:
            adjusted = adjusted + (style_params['brightness'] - 0.5) * 0.2
        if 'contrast' in style_params:
            adjusted = adjusted * (1 + (style_params['contrast'] - 0.5) * 0.5)
        
        return torch.clamp(adjusted, -1, 1)
    
    def __del__(self):
        """Cleanup memory manager on deletion."""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.stop_monitoring()

class MultiModalGenerator:
    """
    High-level interface for multi-modal generation
    Integrates with existing multimodal components
    """
    
    def __init__(self, cfg, vision_encoder=None, audio_encoder=None):
        self.cfg = cfg
        self.unified_gen = UnifiedGeneration(cfg)
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        
    def generate_from_text(self, text: str, modality: str = 'image', **kwargs) -> torch.Tensor:
        """Generate content from text prompt with 2025 mandatory watermark"""
        from tools.watermark import watermark_manager
        
        condition = GenerationCondition()
        condition.text_prompt = text
        condition.generation_params = kwargs
        
        metadata = {
            "prompt": text,
            "modality": modality,
            "params": kwargs,
            "timestamp": str(pd.Timestamp.now()),
            "user_id": kwargs.get("user_id", "anonymous"),
            "generation_method": "text_to_" + modality
        }
        
        if modality == 'image':
            result = self.unified_gen.generate_image(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif modality == 'video':
            result = self.unified_gen.generate_video(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif modality == 'audio':
            result = self.unified_gen.generate_audio(condition)
            return watermark_manager.add_watermark(result, metadata)
            combined_condition = torch.cat(condition_features, dim=-1)
            raise ValueError(f"Unsupported modality: {modality}")
    
    def generate_from_emotion(self, emotion: str, modality: str = 'image', **kwargs) -> torch.Tensor:
        """Generate content from emotion with 2025 mandatory watermark"""
        from tools.watermark import watermark_manager
        
        # Map emotion to vector with valence/arousal
        emotion_map = {
            'happy': [1, 0, 0, 0, 0, 0, 0, 0.8, 0.7, 0.9],  # + valence, arousal, intensity
            'sad': [0, 1, 0, 0, 0, 0, 0, 0.2, 0.3, 0.8],
            'angry': [0, 0, 1, 0, 0, 0, 0, 0.1, 0.9, 0.95],
            'surprised': [0, 0, 0, 1, 0, 0, 0, 0.6, 0.8, 0.85],
            'fear': [0, 0, 0, 0, 1, 0, 0, 0.1, 0.8, 0.9],
            'disgust': [0, 0, 0, 0, 0, 1, 0, 0.2, 0.6, 0.7],
            'neutral': [0, 0, 0, 0, 0, 0, 1, 0.5, 0.5, 0.5],
            'excited': [0.7, 0, 0, 0.3, 0, 0, 0, 0.9, 0.8, 0.95],
            'melancholy': [0, 0.8, 0, 0, 0.2, 0, 0, 0.3, 0.4, 0.7]
        }
        
        condition = GenerationCondition()
        condition.emotion_vector = torch.tensor([emotion_map.get(emotion.lower(), emotion_map['neutral'])], dtype=torch.float32)
        condition.generation_params = kwargs
        condition.style_params = kwargs.get('style', {})
        metadata = {
            "emotion": emotion,
            "modality": modality,
            "params": kwargs,
            "timestamp": str(pd.Timestamp.now()),
            "user_id": kwargs.get("user_id", "anonymous"),
            "generation_method": "emotion_to_" + modality
        }
        
        if modality == 'image':
            result = self.unified_gen.generate_image(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif modality == 'video':
            result = self.unified_gen.generate_video(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif modality == 'audio':
            result = self.unified_gen.generate_audio(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif modality == 'document':
            result = self.unified_gen.generate_document(condition)
            return watermark_manager.add_watermark(result, metadata)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def cross_modal_generate(self, source_modality: str, target_modality: str, 
                           input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Cross-modal generation (e.g., image to audio, text to video) with 2025 mandatory watermark"""
        from tools.watermark import watermark_manager
        
        condition = GenerationCondition()
        
        # Encode source modality
        if source_modality == 'image' and self.vision_encoder:
            condition.image_reference = input_data
        elif source_modality == 'audio' and self.audio_encoder:
            condition.audio_reference = input_data
        elif source_modality == 'video':
            condition.video_reference = input_data
        elif source_modality == 'document':
            condition.document_reference = input_data
        
        condition.generation_params = kwargs
        condition.style_params = kwargs.get('style', {})
        
        metadata = {
            "source_modality": source_modality,
            "target_modality": target_modality,
            "params": kwargs,
            "timestamp": str(pd.Timestamp.now()),
            "user_id": kwargs.get("user_id", "anonymous"),
            "generation_method": f"cross_modal_{source_modality}_to_{target_modality}"
        }
        
        # Generate target modality with watermark
        if target_modality == 'image':
            result = self.unified_gen.generate_image(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif target_modality == 'video':
            result = self.unified_gen.generate_video(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif target_modality == 'audio':
            result = self.unified_gen.generate_audio(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif target_modality == 'document':
            result = self.unified_gen.generate_document(condition)
            return watermark_manager.add_watermark(result, metadata)
        else:
            raise ValueError(f"Unsupported target modality: {target_modality}")
    
    def multimodal_fusion_generate(self, inputs: Dict[str, torch.Tensor], 
                                 target_modality: str, **kwargs) -> torch.Tensor:
        """Generate using multimodal fusion of all input modalities with 2025 mandatory watermark"""
        from tools.watermark import watermark_manager
        
        condition = GenerationCondition()
        
        # Set all available references
        for modality, data in inputs.items():
            if modality == 'image':
                condition.image_reference = data
            elif modality == 'audio':
                condition.audio_reference = data
            elif modality == 'video':
                condition.video_reference = data
            elif modality == 'document':
                condition.document_reference = data
        
        condition.generation_params = kwargs
        condition.style_params = kwargs.get('style', {})
        
        metadata = {
            "fusion_modalities": list(inputs.keys()),
            "target_modality": target_modality,
            "params": kwargs,
            "timestamp": str(pd.Timestamp.now()),
            "user_id": kwargs.get("user_id", "anonymous"),
            "generation_method": f"multimodal_fusion_to_{target_modality}"
        }
        
        # Generate with multimodal fusion and watermark
        if target_modality == 'image':
            result = self.unified_gen.generate_image(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif target_modality == 'video':
            result = self.unified_gen.generate_video(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif target_modality == 'audio':
            result = self.unified_gen.generate_audio(condition)
            return watermark_manager.add_watermark(result, metadata)
        elif target_modality == 'document':
            result = self.unified_gen.generate_document(condition)
            return watermark_manager.add_watermark(result, metadata)
        else:
            raise ValueError(f"Unsupported target modality: {target_modality}")


class MCPGenerationServer:
    """
    Enhanced MCP server for multi-modal generation with streaming support
    Exposes all generation capabilities via MCP protocol
    """
    
    def __init__(self, generator: MultiModalGenerator):
        self.generator = generator
        
    async def handle_generate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP generation requests with enhanced error handling"""
        try:
            modality = request.get('modality', 'image')
            prompt = request.get('prompt', '')
            emotion = request.get('emotion', 'neutral')
            style = request.get('style', {})
            stream = request.get('stream', False)
            
            if stream:
                # Stream generation progress
                return self.generate_streaming(modality, prompt, emotion, style, request)
            
            # Standard generation
            if prompt:
                if modality == 'document':
                    result = self.generator.unified_gen.generate_document(
                        GenerationCondition(text_prompt=prompt, style_params=style)
                    )
                else:
                    result = self.generator.generate_from_text(prompt, modality, style=style)
            else:
                if modality == 'document':
                    result = self.generator.unified_gen.generate_document(
                        GenerationCondition(emotion_vector=torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0.5, 0.5, 0.5]]), style_params=style)
                    )
                else:
                    result = self.generator.generate_from_emotion(emotion, modality, style=style)
            
            metadata = {
                "modality": modality,
                "prompt": prompt,
                "emotion": emotion,
                "style": style,
                "timestamp": str(pd.Timestamp.now()),
                "user_id": request.get('user_id', 'anonymous'),
                "generation_method": "mcp_generation",
                "request_id": str(uuid.uuid4())
            }
            
            from tools.watermark import watermark_manager
            watermarked_result = watermark_manager.add_watermark(result, metadata)
            
            # Enhanced response with metadata and watermark info
            response = {
                'success': True,
                'modality': modality,
                'timestamp': str(pd.Timestamp.now()),
                'metadata': {
                    'prompt': prompt,
                    'emotion': emotion,
                    'style': style,
                    'generation_params': request,
                    'watermark': {
                        'applied': True,
                        'method': 'hidden_watermark_only',
                        'content_type': modality
                    }
                }
            }
            
            # Handle different output formats
            if modality == 'document':
                response['data'] = {
                    'content': watermarked_result['content'].detach().cpu().numpy().tolist(),
                    'structure': watermarked_result['structure'].detach().cpu().numpy().tolist(),
                    'style': watermarked_result['style'].detach().cpu().numpy().tolist(),
                    'format': result['format'].detach().cpu().numpy().tolist()
                }
                response['shape'] = {
                    'content': list(result['content'].shape),
                    'structure': list(result['structure'].shape),
                    'style': list(result['style'].shape),
                    'format': list(result['format'].shape)
                }
            else:
                result_np = watermarked_result.detach().cpu().numpy()
                response['data'] = result_np.tolist()
                response['shape'] = list(result_np.shape)
                
            return response
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': str(pd.Timestamp.now())
            }
    
    async def generate_streaming(self, modality: str, prompt: str, emotion: str, style: Dict, request: Dict):
        """Stream generation progress for real-time updates"""
        steps = request.get('steps', 50)
        
        async def progress_stream():
            for step in range(steps):
                progress = {
                    'step': step + 1,
                    'total_steps': steps,
                    'progress': (step + 1) / steps,
                    'modality': modality
                }
                yield progress
                await asyncio.sleep(0.1)
        
        return {'type': 'stream', 'generator': progress_stream()}
    
    def register_endpoints(self, server):
        """Register all MCP endpoints including document generation"""
        
        @server.call_tool()
        async def generate_image(prompt: str, emotion: str = 'neutral', **style):
            return await self.handle_generate_request({
                'modality': 'image',
                'prompt': prompt,
                'emotion': emotion,
                'style': style
            })
        
        @server.call_tool()
        async def generate_video(prompt: str, emotion: str = 'neutral', frames: int = 16, **style):
            return await self.handle_generate_request({
                'modality': 'video',
                'prompt': prompt,
                'emotion': emotion,
                'frames': frames,
                'style': style
            })
        
        @server.call_tool()
        async def generate_audio(prompt: str, emotion: str = 'neutral', duration: float = 3.0, **style):
            return await self.handle_generate_request({
                'modality': 'audio',
                'prompt': prompt,
                'emotion': emotion,
                'duration': duration,
                'style': style
            })
        
        @server.call_tool()
        async def generate_document(prompt: str, max_length: int = 1000, emotion: str = 'neutral', **style):
            return await self.handle_generate_request({
                'modality': 'document',
                'prompt': prompt,
                'max_length': max_length,
                'emotion': emotion,
                'style': style
            })
        
        @server.call_tool()
        async def cross_modal_generate(source_modality: str, target_modality: str, input_data: List, **kwargs):
            """Cross-modal generation endpoint with 2025 mandatory watermark"""
            from tools.watermark import watermark_manager
            
            try:
                input_tensor = torch.tensor(input_data)
                
                metadata = {
                    "source_modality": source_modality,
                    "target_modality": target_modality,
                    "params": kwargs,
                    "timestamp": str(pd.Timestamp.now()),
                    "user_id": kwargs.get("user_id", "anonymous"),
                    "generation_method": f"mcp_cross_modal_{source_modality}_to_{target_modality}"
                }
                
                result = self.generator.cross_modal_generate(source_modality, target_modality, input_tensor, **kwargs)
                
                watermarked_result = watermark_manager.add_watermark(result, metadata)
                
                return {
                    'success': True,
                    'source_modality': source_modality,
                    'target_modality': target_modality,
                    'watermark': {
                        'applied': True,
                        'method': 'hidden_watermark_only',
                        'content_type': target_modality
                    },
                    'data': watermarked_result.detach().cpu().numpy().tolist(),
                    'shape': list(watermarked_result.shape)
                }
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        @server.call_tool()
        async def multimodal_fusion_generate(target_modality: str, **inputs):
            """Multimodal fusion generation endpoint with 2025 mandatory watermark"""
            from tools.watermark import watermark_manager
            
            try:
                # Parse input tensors
                parsed_inputs = {}
                for modality, data in inputs.items():
                    if modality != 'target_modality':
                        parsed_inputs[modality] = torch.tensor(data)
                
                metadata = {
                    "fusion_modalities": list(parsed_inputs.keys()),
                    "target_modality": target_modality,
                    "params": inputs,
                    "timestamp": str(pd.Timestamp.now()),
                    "user_id": inputs.get("user_id", "anonymous"),
                    "generation_method": f"mcp_multimodal_fusion_to_{target_modality}"
                }
                
                result = self.generator.multimodal_fusion_generate(parsed_inputs, target_modality, **inputs)
                
                watermarked_result = watermark_manager.add_watermark(result, metadata)
                
                return {
                    'success': True,
                    'target_modality': target_modality,
                    'modalities_used': list(parsed_inputs.keys()),
                    'watermark': {
                        'applied': True,
                        'method': 'hidden_watermark_only',
                        'content_type': target_modality
                    },
                    'data': watermarked_result.detach().cpu().numpy().tolist(),
                    'shape': list(watermarked_result.shape)
                }
            except Exception as e:
                return {'success': False, 'error': str(e)}
                
# Export all multimodal components including the new agent system
__all__ = [
    # Vision encoders
    'VisionEncoder',
    'ImageProcessor',
    
    # Audio encoders  
    'AudioEncoder',
    'AudioProcessor',
    
    # Video encoders
    'VideoEncoder',
    'FrameEncoder',
    
    # Document encoders
    'DocumentEncoder',
    'DocumentProcessor',
    
    # Agent system (fully migrated from agent.py)
    'PiscesAgent',
    'AgentState',
    'AgentAction',
    'AgentObservation', 
    'AgentMemory',
    'MCPMessageType',
    'MCPMessage',
    'MCPProtocol',
    'MCPToolRegistry',
    'TreeSearchReasoner',
    'PiscesReasoner',
    'AgentEncoder',  # Legacy wrapper maintained for compatibility
    
    # Cross-modal components
    'CrossModalAttention',
    'DynamicModalFusion',  # Unified dynamic modal fusion with native multimodal fusion capabilities
    
    # Unified generation system
    'UnifiedGeneration',
    'MultiModalGenerator',
    'GenerationCondition',
    'MCPGenerationServer',
    
    # Enhanced generation components
    'DocumentDecoder',
    'CrossModalGenerator',
    'MultimodalFusionGenerator',
    'StreamingGenerationServer'
]