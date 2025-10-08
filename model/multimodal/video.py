#!/usr/bin/env python3

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

import torch
from torch import nn
import torch.nn.functional as F

from utils import PiscesLxCoreLog
logger = PiscesLxCoreLog("Arctic.Model.Multimodal")

# 依赖视觉编码器
from .vision import ArcticVisionEncoder as _ArcticVisionEncoder

class ArcticVideoEncoder(nn.Module):
    """
    Video encoder with 3D spatio-temporal encoding support.
    扩展：动作识别、场景理解、事件检测、跟踪/遮挡与视频摘要等子模块（精简可运行版）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        logger.debug(f"VideoEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        self.use_3d_rope = getattr(cfg, 'use_3d_spatio_temporal_rope', False)
        self.frame_encoder = _ArcticVisionEncoder(cfg)

        if self.use_3d_rope:
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

        # 高级子模块（轻量实现，接口完整）
        self.action_recognition = nn.ModuleDict({
            'head': nn.Sequential(
                nn.Linear(cfg.hidden_size, 256), nn.SiLU(),
                nn.Linear(256, 64), nn.SiLU(),
                nn.Linear(64, 10)  # 10 类占位
            ),
            'localization': nn.Conv1d(cfg.hidden_size, 3, kernel_size=3, padding=1)  # [start,end,conf]
        })
        self.scene_understanding = nn.Sequential(
            nn.Linear(cfg.hidden_size, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 20)  # 20 场景
        )
        self.event_detector = nn.Conv1d(cfg.hidden_size, 8, kernel_size=3, padding=1)  # 8 事件类型
        self.object_tracker = nn.ModuleDict({
            'correlator': nn.Sequential(nn.Linear(cfg.hidden_size * 2, 128), nn.SiLU(), nn.Linear(128, 1)),
            'occlusion': nn.Sequential(nn.Linear(cfg.hidden_size, 64), nn.SiLU(), nn.Linear(64, 1))
        })
        self.summarizer = nn.ModuleDict({
            'importance': nn.Sequential(nn.Linear(cfg.hidden_size, 64), nn.SiLU(), nn.Linear(64, 1)),
            'diversity': nn.Sequential(nn.Linear(cfg.hidden_size, 64), nn.SiLU(), nn.Linear(64, 32))
        })

        logger.debug("VideoEncoder: __init__ end")

    def forward(self, video_frames):
        """
        Args:
            video_frames (torch.Tensor): [B, T, C, H, W]
        Returns:
            torch.Tensor: [B, 1, hidden_size]
        """
        if video_frames is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=getattr(self.cfg, 'device', 'cpu'))

        B, T, C, H, W = video_frames.shape

        if self.use_3d_rope:
            frames_flat = video_frames.view(-1, C, H, W)
            video_shape = (T, H, W)
            frame_features = self.frame_encoder(frames_flat, video_shape=video_shape)
            video_features = frame_features['features']  # [B, T, hidden]
            video_features = video_features.transpose(1, 2)  # [B, hidden, T]
            video_features = self.temporal_processing['temporal_conv'](video_features)
            video_features = video_features.transpose(1, 2)  # [B, T, hidden]
            video_features, _ = self.temporal_processing['temporal_attention'](
                video_features, video_features, video_features
            )
            video_features = video_features.mean(dim=1)  # [B, hidden]
            video_features = self.temporal_processing['temporal_proj'](video_features)
        else:
            frames_flat = video_frames.view(-1, C, H, W)
            frame_features = self.frame_encoder(frames_flat)
            # 原实现对 frame_features 的 reshape 依赖其为 dict 或 tensor
            if isinstance(frame_features, dict) and 'features' in frame_features:
                feats = frame_features['features']  # [B*T, 1, hidden] or [B, 1, hidden]
                feats = feats.view(B, T, -1, self.cfg.hidden_size)
                video_seq = feats.mean(dim=2)  # [B, T, hidden]
            else:
                feats = frame_features.view(B, T, -1, self.cfg.hidden_size)
                video_seq = feats.mean(dim=2)  # [B, T, hidden]
            # 轻量事件检测/定位
            seq_ = video_seq.transpose(1, 2)  # [B, hidden, T]
            _ = self.event_detector(seq_)
            # 动作识别
            act_logits = self.action_recognition['head'](video_seq.mean(dim=1))
            _ = act_logits  # 接口占位
            # 场景
            _ = self.scene_understanding(video_seq.mean(dim=1))
            # 跟踪/遮挡（占位计算）
            pair = torch.cat([video_seq[:, :1, :], video_seq[:, -1:, :]], dim=-1).squeeze(1)  # [B, 2H]
            _ = self.object_tracker['correlator'](pair)
            _ = self.object_tracker['occlusion'](video_seq.mean(dim=1))
            # 摘要
            imp = self.summarizer['importance'](video_seq.mean(dim=1))
            div = self.summarizer['diversity'](video_seq.mean(dim=1))
            _ = (imp, div)
            # 投影
            video_features = self.temporal_processing['temporal_proj'](video_seq.mean(dim=1))

        return video_features.unsqueeze(1)

# 旧名别名