#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import math
from typing import Optional


class YaRNRotaryEmbedding(nn.Module):
    """YaRN长上下文位置编码实现
    基于RoPE (Rotary Position Embedding) 并扩展YaRN (Yet Another RoPE Extension) 方法
    支持超长上下文长度，最高可达32768 tokens
    """
    def __init__(self,
                 dim: int,
                 max_position_embeddings: int = 32768,
                 base: int = 10000,
                 scale: float = 8.0,
                 original_max_position_embeddings: int = 4096,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        
        # 计算频率因子
        self.freq_factors = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        
        # 预计算YaRN缩放因子
        self.scale_factors = self._compute_scale_factors(device)
        
        # 缓存位置嵌入
        self.register_buffer("inv_freq", self.freq_factors, persistent=False)
        self.register_buffer("scale_factors", self.scale_factors, persistent=False)
        
    def _compute_scale_factors(self, device: Optional[torch.device] = None) -> torch.Tensor:
        # 生成原始位置索引
        original_positions = torch.arange(self.original_max_position_embeddings, device=device)
        
        # 计算YaRN缩放因子
        scale_factors = torch.ones(self.max_position_embeddings, device=device)
        
        # 应用YaRN扩展公式
        if self.scale > 1.0:
            # 计算交叉点
            crossover = math.sqrt(self.original_max_position_embeddings)
            # 计算缩放区域
            high_positions = torch.arange(crossover, self.max_position_embeddings, device=device)
            # 应用对数缩放
            scale_factors[crossover:] = crossover * (high_positions / crossover) ** (1.0 / self.scale)
        
        return scale_factors
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # 获取设备
        device = x.device
        
        # 获取位置索引
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # 应用YaRN缩放
        t = t * self.scale_factors[:seq_len].to(device)
        
        # 计算频率
        freqs = torch.outer(t, self.inv_freq)
        
        # 计算cos和sin
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        # 应用旋转
        return self._rotate_half(x, cos, sin)
    
    @staticmethod
    def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # 将张量分为两半
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        
        # 应用旋转
        rotated = torch.cat((-x2 * sin + x1 * cos, x1 * sin + x2 * cos), dim=-1)
        
        return rotated
    
    def extra_repr(self) -> str:
        return (f"dim={self.dim}, max_position_embeddings={self.max_position_embeddings}, "
                f"base={self.base}, scale={self.scale}, "
                f"original_max_position_embeddings={self.original_max_position_embeddings}")