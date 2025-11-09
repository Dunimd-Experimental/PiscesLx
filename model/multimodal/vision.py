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

"""Vision encoders and utilities for Arctic multimodal agents.

This module houses components that transform raw images and video frames into
feature representations consumed across the Pisces multimodal stack. It
includes optional 3D rotary positional embeddings, a vision encoder with
auxiliary detection heads, and helpers for text rendered as images.
"""

import math
import torch
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
from utils.log.core import PiscesLxCoreLog
from typing import Any, Dict, List, Optional, Tuple

logger = PiscesLxCoreLog("Arctic.Core.Multimodal", file_path="logs/ArcticCore.log")

class ArcticSpatioTemporalRoPE3D(nn.Module):
    """3D spatio-temporal rotary positional embedding for video inputs.

    The module extends the standard 2D RoPE formulation by incorporating an
    additional temporal axis, enabling consistent positional encoding across
    frame sequences.

    Args:
        dim (int): Feature dimension divisible by three (one slice per axis).
        max_temporal_frames (int): Maximum time steps supported for caching.
        max_spatial_h (int): Maximum height in patches used for caching.
        max_spatial_w (int): Maximum width in patches used for caching.
        base (float): Base factor controlling the geometric progression.
        device (torch.device | None): Optional device override for buffers.
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
        """Return flattened grid indices spanning temporal, height, and width."""
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
        """Compute cosine and sine lookup tables associated with 3D positions."""
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
        """Apply cached 3D RoPE values to the input sequence."""
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
        """
        Rotate the last dimension of the input tensor by half and negate the second half.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Rotated tensor.
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class ArcticVisualTextProcessor(nn.Module):
    """Encode text rendered as images for H-Network tokenization support."""

    def __init__(self, hidden_size: int, patch_size: int = 14) -> None:
        """Construct a text-oriented preprocessing pipeline.

        Args:
            hidden_size (int): Target embedding dimension.
            patch_size (int): Convolutional patch size applied during tokenization.
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

        logger.info(f"ArcticVisualTextProcessor initialized: hidden_size={hidden_size}, patch_size={patch_size}")

    def forward(self, text_images: torch.Tensor) -> torch.Tensor:
        """Convert batches of rendered text images into patch embeddings.

        Args:
            text_images (torch.Tensor): Input tensor of shape ``[B, C, H, W]``.

        Returns:
            torch.Tensor: Patch embeddings shaped ``[B, N, hidden_size]``.
        """
        # Normalize for text (higher contrast) with stability bounds.
        x = (text_images - self.text_mean) / self.text_std
        # Clamp to prevent extreme values in text processing.
        x = torch.clamp(x, -5.0, 5.0)

        # Extract patches using the text-specific convolution.
        patches = self.text_patch_embed(x)  # [B, D, H', W']

        # Flatten spatial dimensions into sequences.
        B, D, H, W = patches.shape
        patches = patches.view(B, D, -1).transpose(1, 2)  # [B, N, D]

        # Apply text-specific compression restoring the ``hidden_size`` dimension.
        compressed = self.text_compressor(patches)

        logger.debug(f"ArcticVisualTextProcessor: {text_images.shape} -> {compressed.shape}")

        return compressed


class ArcticVisionEncoder(nn.Module):
    """Vision backbone producing multimodal features and auxiliary predictions.

    The encoder tokenizes images into patch embeddings, supports optional 3D
    RoPE for video streams, and attaches detection, segmentation, and reasoning
    heads used by downstream Arctic modules.

    Args:
        cfg: Configuration namespace supplying hyperparameters such as
            ``hidden_size``, ``n_head``, and ``n_layer``.
        cache_manager: Reserved for future caching integrations. Defaults to
            ``None``.
    """

    def __init__(self, cfg, cache_manager=None) -> None:
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.patch_size = 14
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.n_head
        self.num_layers = cfg.n_layer
        logger.debug(f"VisionEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
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
            logger.info("H-Network visual text processor initialized")

    def _create_visual_text_processor(self):
        """Instantiate a processor specialized for rendered text inputs."""
        return ArcticVisualTextProcessor(self.hidden_size, self.patch_size)

    def process_visual_text(self, text_images: torch.Tensor) -> torch.Tensor:
        """Project rendered text images into embeddings via the text processor.

        Args:
            text_images (torch.Tensor): Input tensor shaped ``[B, C, H, W]``.

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

        Args:
            x (torch.Tensor): Input tensor shaped ``[B, C, H, W]``.
            **kwargs: Optional flags including ``is_visual_text`` to route through
                the text processor.

        Returns:
            torch.Tensor: Clamped vision features respecting ``hidden_size``.
        """
        if kwargs.get('is_visual_text', False) and self.visual_text_processor is not None:
            return self.process_visual_text(x)

        # Standard vision processing with output stability.
        output = self._standard_forward(x)
        # Clamp output to prevent extreme values.
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
            self.spatio_temporal_rope = ArcticSpatioTemporalRoPE3D(
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
        logger.debug("VisionEncoder: __init__ end")

    def process_image(self, image_path, target_size=None):
        """
        Process an image from the given path, including normalization.

        Args:
            image_path (str): Path to the image file.
            target_size (Optional[Tuple[int, int]]): Target size to resize the image. Defaults to None.

        Returns:
            Optional[torch.Tensor]: Processed image tensor. Returns None if an error occurs.
        """
        logger.debug(f"Processing image: {image_path}")
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                if target_size is not None:
                    img = img.resize(target_size, Image.LANCZOS)
                image_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
                image_tensor = (image_tensor - self.mean) / self.std
                return image_tensor
        except Exception as e:
            logger.error(f"Image processing error: {e}")
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

    def forward(self, pixel_values, video_shape=None):
        """
        Forward pass of the ArcticVisionEncoder.

        Args:
            pixel_values (torch.Tensor): Input pixel values.
            video_shape (Optional[Tuple[int, int, int]]): Shape of the video (temporal, height, width). Defaults to None.

        Returns:
            Dict: A dictionary containing detection results.
        """
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
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(self.pos_embed, h_patches, w_patches)
        x = x + pos_embed
        if is_video:
            x = x.view(B, T * (1 + h_patches * w_patches), self.hidden_size)
            cls_tokens_video = x[:, :T]
            patch_tokens = x[:, T:]
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
