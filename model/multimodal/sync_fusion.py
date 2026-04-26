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

"""SyncFusion: Audio-Video Synchronous Understanding for Yv Models.

Based on ICLR 2025 JavisGPT. Aligns temporal features across audio and video
for synchronized event understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class YvSyncFusion(nn.Module):
    """Audio-video synchronous fusion module.

    Aligns temporal features between audio and video modalities
    to capture synchronized events.

    Attributes:
        hidden_size (int): Feature dimension.
        num_temporal_bins (int): Number of temporal alignment bins.
        temporal_aligner (nn.Linear): Aligns temporal features.
        cross_attention (nn.MultiheadAttention): Cross-modal attention.

    Example:
        >>> fusion = YvSyncFusion(hidden_size=4096, num_temporal_bins=16)
        >>> audio_feat = torch.randn(2, 100, 4096)
        >>> video_feat = torch.randn(2, 50, 4096)
        >>> fused = fusion(audio_feat, video_feat)
    """

    def __init__(
        self,
        hidden_size: int,
        num_temporal_bins: int = 16,
        num_heads: int = 8,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_temporal_bins = num_temporal_bins

        # Temporal alignment projections
        self.audio_temporal = nn.Linear(
            hidden_size, num_temporal_bins * hidden_size, bias=False, device=device, dtype=dtype
        )
        self.video_temporal = nn.Linear(
            hidden_size, num_temporal_bins * hidden_size, bias=False, device=device, dtype=dtype
        )

        # Cross-modal attention for synchronization
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            device=device,
            dtype=dtype
        )

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # Output projection
        self.output_proj = nn.Linear(
            hidden_size * 2, hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse audio and video with temporal synchronization.

        Args:
            audio_features: Audio features [batch, audio_len, hidden].
            video_features: Video features [batch, video_len, hidden].
            audio_mask: Optional audio mask.
            video_mask: Optional video mask.

        Returns:
            Fused features [batch, max_len, hidden].
        """
        batch_size = audio_features.shape[0]

        # Temporal binning
        audio_temp = self.audio_temporal(audio_features.mean(dim=1))
        audio_temp = audio_temp.view(batch_size, self.num_temporal_bins, self.hidden_size)

        video_temp = self.video_temporal(video_features.mean(dim=1))
        video_temp = video_temp.view(batch_size, self.num_temporal_bins, self.hidden_size)

        # Cross-modal attention
        audio_aligned, _ = self.cross_attention(
            query=audio_temp,
            key=video_temp,
            value=video_temp
        )

        video_aligned, _ = self.cross_attention(
            query=video_temp,
            key=audio_temp,
            value=audio_temp
        )

        # Combine aligned features
        combined = torch.cat([audio_aligned, video_aligned], dim=-1)
        gate = self.fusion_gate(combined)

        fused = gate * audio_aligned + (1 - gate) * video_aligned

        # Project to output dimension
        output = self.output_proj(
            torch.cat([fused.mean(dim=1, keepdim=True).expand(-1, max(audio_features.shape[1], video_features.shape[1]), -1),
                       fused.mean(dim=1, keepdim=True).expand(-1, max(audio_features.shape[1], video_features.shape[1]), -1)], dim=-1)
        ) if audio_features.shape[1] != video_features.shape[1] else self.output_proj(
            torch.cat([audio_features, video_features], dim=-1)
        )

        # Return properly sized output
        max_len = max(audio_features.shape[1], video_features.shape[1])
        return fused.mean(dim=1, keepdim=True).expand(-1, max_len, -1)


class YvCoupledMambaFusion(nn.Module):
    """Coupled Mamba state-space model for multimodal fusion.

    Each modality has its own SSM state with cross-modal coupling.
    """

    def __init__(
        self,
        hidden_size: int,
        num_modalities: int = 6,
        coupling_strength: float = 0.3,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.coupling_strength = coupling_strength

        # Per-modality state projections
        self.modality_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False, device=device, dtype=dtype)
            for _ in range(num_modalities)
        ])

        # Cross-modal coupling weights
        self.coupling_weights = nn.Parameter(
            torch.randn(num_modalities, num_modalities, device=device, dtype=dtype) * 0.02
        )

        # Output fusion
        self.fusion_proj = nn.Linear(
            hidden_size * num_modalities, hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(
        self,
        modality_features: dict
    ) -> torch.Tensor:
        """Fuse multimodal features with coupled SSM states.

        Args:
            modality_features: Dict mapping modality name to features.

        Returns:
            Fused features [batch, seq, hidden].
        """
        # Project each modality
        projected = []
        for i, (name, feat) in enumerate(modality_features.items()):
            if i >= self.num_modalities:
                break
            proj = self.modality_projections[i](feat)
            projected.append(proj)

        if not projected:
            return None

        # Apply cross-modal coupling
        coupled = []
        for i in range(len(projected)):
            coupled_state = projected[i]
            for j in range(len(projected)):
                if i != j:
                    weight = torch.sigmoid(self.coupling_weights[i, j])
                    coupled_state = coupled_state + self.coupling_strength * weight * projected[j]
            coupled.append(coupled_state)

        # Pad or truncate to same length
        max_len = max(c.shape[1] for c in coupled)
        aligned = []
        for c in coupled:
            if c.shape[1] < max_len:
                padding = torch.zeros(c.shape[0], max_len - c.shape[1], c.shape[2], device=c.device, dtype=c.dtype)
                c = torch.cat([c, padding], dim=1)
            elif c.shape[1] > max_len:
                c = c[:, :max_len, :]
            aligned.append(c)

        # Concatenate and fuse
        fused = torch.cat(aligned, dim=-1)
        output = self.fusion_proj(fused)

        return output
