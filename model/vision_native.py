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
from einops import rearrange
import numpy as np
from PIL import Image

class NativeSiglipVisionEncoder(nn.Module):
    """原生实现的SigLIP级视觉编码器，基于谷歌SigLIP 2架构改进"""
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.image_size = 384
        self.patch_size = 14
        self.hidden_size = 1152  # 对应400M参数规模
        self.num_heads = 18
        self.num_layers = 24
        
        print(f"🟧	NativeSiglipVisionEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # 图像预处理
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Patch嵌入
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # 位置嵌入
        num_patches = (self.image_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, self.hidden_size))
        
        # Transformer编码器
        self.transformer = nn.ModuleDict({
            'layers': nn.ModuleList([
                nn.ModuleDict({
                    'norm1': nn.LayerNorm(self.hidden_size),
                    'attn': nn.MultiheadAttention(
                        embed_dim=self.hidden_size,
                        num_heads=self.num_heads,
                        batch_first=True
                    ),
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
        
        # 投影层
        self.proj = nn.Linear(self.hidden_size, cfg.hidden_size)
        
        print("🟧	NativeSiglipVisionEncoder: __init__ end")
    
    def process_image(self, image_path):
        """Process image data"""
        print(f"🟧	Processing image: {image_path}")
        try:
            # 使用PIL读取图像并转换为张量
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            image = (image - self.mean) / self.std
            return image
        except Exception as e:
            print(f"❌	Image processing error: {e}")
            return None
    
    def forward(self, pixel_values):
        if pixel_values is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=self.proj.weight.device)
        
        # 标准化输入
        x = (pixel_values - self.mean) / self.std
        
        # Patch嵌入
        x = self.patch_embed(x)  # (B, 1152, 27, 27)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # 添加位置嵌入
        x = x + self.pos_embed
        
        # Transformer编码器
        for layer in self.transformer['layers']:
            # 自注意力
            attn_out = layer['attn'](layer['norm1'](x), layer['norm1'](x), layer['norm1'](x))[0]
            x = x + attn_out
            # MLP
            mlp_out = layer['mlp'](layer['norm2'](x))
            x = x + mlp_out
        
        # 最终归一化
        x = self.transformer['norm'](x)
        
        # 全局平均池化并投影
        x = self.proj(x.mean(dim=1))
        return x.unsqueeze(1)  # (B, 1, hidden_size)