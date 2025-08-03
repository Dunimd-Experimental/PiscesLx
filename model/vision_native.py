#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces L1.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from einops import rearrange
from utils.log import  DEBUG, ERROR

class NativeSiglipVisionEncoder(nn.Module):
    """A native implementation of the SigLIP-level vision encoder, improved based on Google's SigLIP 2 architecture."""
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.image_size = 384
        self.patch_size = 14
        self.hidden_size = 1152  # Corresponding to a parameter scale of 400M
        self.num_heads = 18
        self.num_layers = 24
        
        DEBUG(f"NativeSiglipVisionEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Image preprocessing
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Positional embedding
        num_patches = (self.image_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, self.hidden_size))
        
        # Transformer encoder
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
        
        # Projection layer
        self.proj = nn.Linear(self.hidden_size, cfg.hidden_size)
        
        DEBUG("NativeSiglipVisionEncoder: __init__ end")
    
    def process_image(self, image_path):
        """
        Process image data from the given file path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Processed image tensor, or None if an error occurs.
        """
        DEBUG(f"Processing image: {image_path}")
        try:
            # Read the image using PIL and convert it to a tensor
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            image = (image - self.mean) / self.std
            return image
        except Exception as e:
            ERROR(f"Image processing error: {e}")
            return None
    
    def forward(self, pixel_values):
        """
        Forward pass of the NativeSiglipVisionEncoder.

        Args:
            pixel_values (torch.Tensor): Input pixel values.

        Returns:
            torch.Tensor: Output tensor with shape (B, 1, hidden_size).
        """
        if pixel_values is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=self.proj.weight.device)
        
        # Normalize the input
        x = (pixel_values - self.mean) / self.std
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 1152, 27, 27)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        for layer in self.transformer['layers']:
            # Self-attention
            attn_out = layer['attn'](layer['norm1'](x), layer['norm1'](x), layer['norm1'](x))[0]
            x = x + attn_out
            # MLP
            mlp_out = layer['mlp'](layer['norm2'](x))
            x = x + mlp_out
        
        # Final normalization
        x = self.transformer['norm'](x)
        
        # Global average pooling and projection
        x = self.proj(x.mean(dim=1))
        return x.unsqueeze(1)  # (B, 1, hidden_size)