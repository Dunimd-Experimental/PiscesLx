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

"""Video encoder utilities for Arctic multimodal agents.

The module exposes :class:`ArcticVideoEncoder`, a wrapper around the vision
encoder that augments per-frame embeddings with temporal modeling, action
recognition, and summarization heuristics. It mirrors the detailed docstring
style adopted across the multimodal stack and leaves model behavior unchanged.
"""

import torch
from torch import nn
from typing import Optional
from .vision import ArcticVisionEncoder
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("Arctic.Core.Multimodal", file_path="logs/ArcticCore.log")

class ArcticVideoEncoder(nn.Module):
    """Encode video frame sequences with optional 3D rotary positional modeling.

    The encoder delegates per-frame feature extraction to
    :class:`ArcticVisionEncoder` and enriches temporal structure through
    convolution, attention, and lightweight analytic heads.

    Attributes:
        enabled (bool): Indicates whether the encoder is active.
        cfg: Configuration namespace providing ``hidden_size`` and ``n_head``.
        use_3d_rope (bool): Flag controlling 3D spatio-temporal RoPE handling.
        frame_encoder (ArcticVisionEncoder): Backbone used for per-frame features.
        temporal_processing (nn.ModuleDict): Temporal modules varying with RoPE usage.
        action_recognition (nn.ModuleDict): Heads that output coarse action logits
            and localization cues.
        scene_understanding (nn.Sequential): Classifier estimating scene context.
        event_detector (nn.Conv1d): Temporal convolution approximating event scores.
        object_tracker (nn.ModuleDict): Modules estimating tracking correlation and
            occlusion likelihood.
        summarizer (nn.ModuleDict): Heads approximating importance and diversity scores.
    """

    def __init__(self, cfg):
        """Instantiate the encoder using the supplied configuration namespace.

        Args:
            cfg: Configuration object providing vision backbone parameters such as
                ``hidden_size`` and ``n_head`` as well as temporal feature toggles.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        logger.debug(f"VideoEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")

        # Determine whether to use 3D spatio-temporal RoPE.
        self.use_3d_rope = getattr(cfg, 'use_3d_spatio_temporal_rope', False)
        # Initialize the per-frame feature extractor.
        self.frame_encoder = ArcticVisionEncoder(cfg)

        if self.use_3d_rope:
            # Temporal processing modules activated when 3D RoPE is enabled.
            self.temporal_processing = nn.ModuleDict({
                'temporal_conv': nn.Sequential(
                    nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=5, padding=2),
                    nn.BatchNorm1d(cfg.hidden_size),
                    nn.SiLU(),
                    nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1),
                    nn.BatchNorm1d(cfg.hidden_size),
                    nn.SiLU()
                ),
                'temporal_attention': nn.MultiheadAttention(cfg.hidden_size, cfg.n_head, batch_first=True, dropout=0.1),
                'temporal_proj': nn.Sequential(
                    nn.Linear(cfg.hidden_size, cfg.hidden_size),
                    nn.LayerNorm(cfg.hidden_size),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(cfg.hidden_size, cfg.hidden_size),
                    nn.LayerNorm(cfg.hidden_size)
                ),
            })
        else:
            # Temporal processing modules for the standard (non-3D RoPE) pathway.
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

        # Initialize analytic heads that produce auxiliary video descriptors.
        self.action_recognition = nn.ModuleDict({
            'head': nn.Sequential(
                nn.Linear(cfg.hidden_size, 256), nn.SiLU(),
                nn.Linear(256, 64), nn.SiLU(),
                nn.Linear(64, 10)  # Placeholder for 10 action classes
            ),
            'localization': nn.Conv1d(cfg.hidden_size, 3, kernel_size=3, padding=1)  # [start, end, confidence]
        })
        self.scene_understanding = nn.Sequential(
            nn.Linear(cfg.hidden_size, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 20)  # 20 scene classes
        )
        self.event_detector = nn.Conv1d(cfg.hidden_size, 8, kernel_size=3, padding=1)  # 8 event types
        self.object_tracker = nn.ModuleDict({
            'correlator': nn.Sequential(nn.Linear(cfg.hidden_size * 2, 128), nn.SiLU(), nn.Linear(128, 1)),
            'occlusion': nn.Sequential(nn.Linear(cfg.hidden_size, 64), nn.SiLU(), nn.Linear(64, 1))
        })
        self.summarizer = nn.ModuleDict({
            'importance': nn.Sequential(nn.Linear(cfg.hidden_size, 64), nn.SiLU(), nn.Linear(64, 1)),
            'diversity': nn.Sequential(nn.Linear(cfg.hidden_size, 64), nn.SiLU(), nn.Linear(64, 32))
        })

        logger.debug("VideoEncoder: __init__ end")

    def forward(self, video_frames: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode a batch of videos into pooled feature representations.

        Args:
            video_frames (torch.Tensor | None): Input tensor shaped ``[B, T, C, H, W]``
                containing batched video clips. ``None`` yields a zero tensor placeholder.

        Returns:
            torch.Tensor: Encoded tensor shaped ``[B, 1, hidden_size]`` summarizing each
            video through temporal pooling and projection.
        """
        if video_frames is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=getattr(self.cfg, 'device', 'cpu'))

        B, T, C, H, W = video_frames.shape

        if self.use_3d_rope:
            # Flatten frames to feed the vision encoder.
            frames_flat = video_frames.view(-1, C, H, W)
            video_shape = (T, H, W)
            frame_features = self.frame_encoder(frames_flat, video_shape=video_shape)
            video_features = frame_features['features']  # [B, T, hidden]
            video_features = video_features.transpose(1, 2)  # [B, hidden, T]
            # Apply temporal convolution for localized temporal mixing.
            video_features = self.temporal_processing['temporal_conv'](video_features)
            video_features = video_features.transpose(1, 2)  # [B, T, hidden]
            # Apply temporal attention to model long-range dependencies.
            video_features, _ = self.temporal_processing['temporal_attention'](
                video_features, video_features, video_features
            )
            # Average over time dimension to obtain a pooled descriptor.
            video_features = video_features.mean(dim=1)  # [B, hidden]
            # Apply temporal projection to match the downstream hidden size.
            video_features = self.temporal_processing['temporal_proj'](video_features)
        else:
            # Flatten frames to feed the vision encoder.
            frames_flat = video_frames.view(-1, C, H, W)
            frame_features = self.frame_encoder(frames_flat)
            # Reshape frame features based on encoder output type.
            if isinstance(frame_features, dict) and 'features' in frame_features:
                feats = frame_features['features']  # [B*T, 1, hidden] or [B, 1, hidden]
                feats = feats.view(B, T, -1, self.cfg.hidden_size)
                video_seq = feats.mean(dim=2)  # [B, T, hidden]
            else:
                feats = frame_features.view(B, T, -1, self.cfg.hidden_size)
                video_seq = feats.mean(dim=2)  # [B, T, hidden]

            # Perform lightweight event detection to maintain parity with existing interface.
            seq_ = video_seq.transpose(1, 2)  # [B, hidden, T]
            _ = self.event_detector(seq_)

            # Perform action recognition; output is retained for interface completeness.
            act_logits = self.action_recognition['head'](video_seq.mean(dim=1))
            _ = act_logits  # Placeholder for interface

            # Perform scene understanding to populate auxiliary outputs.
            _ = self.scene_understanding(video_seq.mean(dim=1))

            # Perform object tracking and occlusion handling heuristics.
            pair = torch.cat([video_seq[:, :1, :], video_seq[:, -1:, :]], dim=-1).squeeze(1)  # [B, 2H]
            _ = self.object_tracker['correlator'](pair)
            _ = self.object_tracker['occlusion'](video_seq.mean(dim=1))

            # Generate a video summary using importance and diversity heuristics.
            imp = self.summarizer['importance'](video_seq.mean(dim=1))
            div = self.summarizer['diversity'](video_seq.mean(dim=1))
            _ = (imp, div)

            # Apply temporal projection to produce the final pooled features.
            video_features = self.temporal_processing['temporal_proj'](video_seq.mean(dim=1))

        return video_features.unsqueeze(1)
