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
from torch import nn
from PIL import Image
import torch.nn.functional as F
from utils.log import DEBUG, ERROR
from transformers import ASTFeatureExtractor

class VisionEncoder(nn.Module):
    """
    A vision encoder module that processes image data and encodes it using a Transformer architecture.
    """
    def __init__(self, cfg):
        """
        Initialize the VisionEncoder.

        Args:
            cfg: Configuration object containing parameters such as image resolution, hidden size, etc.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.image_size = cfg.image_res
        self.patch_size = 14
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.n_head
        self.num_layers = cfg.n_layer
        
        DEBUG(f"VisionEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Image preprocessing: register mean and std for normalization
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Patch embedding: convert image patches to embeddings
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # Position embedding: add positional information to patches
        num_patches = (self.image_size // self.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, self.hidden_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        
        # Transformer encoder: process patches using multiple Transformer layers
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
        
        # Projection layer: project the output to the model's hidden dimension
        self.proj = nn.Linear(self.hidden_size, cfg.hidden_size)
        
        DEBUG("VisionEncoder: __init__ end")
    
    def process_image(self, image_path):
        """
        Process an image from the given path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Processed image tensor, or None if an error occurs.
        """
        DEBUG(f"Processing image: {image_path}")
        try:
            # Read images using PIL and convert them into tensors
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
        Forward pass of the VisionEncoder.

        Args:
            pixel_values (torch.Tensor): Input image pixel values.

        Returns:
            torch.Tensor: Encoded image features.
        """
        if pixel_values is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=self.proj.weight.device)
        
        # Image preprocessing: normalize pixel values
        x = (pixel_values - self.mean) / self.std
        
        # Dynamic resolution processing (NaViT style)
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        # If the input resolution is not a multiple of patch_size, adjust it
        if H % patch_size != 0 or W % patch_size != 0:
            new_H = ((H + patch_size - 1) // patch_size) * patch_size
            new_W = ((W + patch_size - 1) // patch_size) * patch_size
            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
            H, W = new_H, new_W
        
        # Patch embedding: convert image to patch embeddings
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add classification token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Dynamic position embedding
        num_patches = (H // patch_size) * (W // patch_size)
        pos_embed = self.pos_embed[:, :num_patches+1]
        x = x + pos_embed
        
        # Transformer encoder: process patches through Transformer layers
        for layer in self.transformer['layers']:
            # Self-attention
            x = x + layer['attn'](layer['norm1'](x), layer['norm1'](x), layer['norm1'](x))[0]
            # MLP
            x = x + layer['mlp'](layer['norm2'](x))
        
        # Final normalization
        x = self.transformer['norm'](x)
        
        # Projection to model hidden dimension
        x = self.proj(x)
        return x


class AudioEncoder(nn.Module):
    """
    An audio encoder module that processes audio data and encodes it using a convolutional network.
    """
    def __init__(self, cfg):
        """
        Initialize the AudioEncoder.

        Args:
            cfg: Configuration object containing parameters such as hidden size.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        DEBUG(f"AudioEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        self.processor = ASTFeatureExtractor()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=8, stride=8)
        
        self.proj = nn.Sequential(
            nn.Linear(64 * 128, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size)
        )
        
        DEBUG("AudioEncoder: __init__ end")
    
    def process_audio(self, audio_path):
        """
        Process audio from the given path.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Processed audio tensor, or None if an error occurs.
        """
        DEBUG(f"Processing audio: {audio_path}")
        try:
            audio = self.processor(audio=audio_path, return_tensors="pt")
            return audio['input_values'][0]
        except Exception as e:
            ERROR(f"Audio processing error: {e}")
            return None
    
    def forward(self, audio_input):
        """
        Forward pass of the AudioEncoder.

        Args:
            audio_input (dict): Dictionary containing audio input values.

        Returns:
            torch.Tensor: Encoded audio features.
        """
        if audio_input is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=audio_input.device)
        
        x = self.conv1(audio_input['input_values'].unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x.unsqueeze(1)


class DocEncoder(nn.Module):
    """
    A document encoder module that processes document input and projects it to a hidden dimension.
    """
    def __init__(self, cfg):
        """
        Initialize the DocEncoder.

        Args:
            cfg: Configuration object containing parameters such as hidden size.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        DEBUG(f"DocEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        self.doc_proj = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size)
        )
        
        DEBUG("DocEncoder: __init__ end")
    
    def forward(self, doc_input):
        """
        Forward pass of the DocEncoder.

        Args:
            doc_input (dict): Dictionary containing document input IDs.

        Returns:
            torch.Tensor: Encoded document features.
        """
        x = self.doc_proj(doc_input['input_ids'])
        return x.unsqueeze(1)


class VideoEncoder(nn.Module):
    """
    A video encoder module that processes video frames using a vision encoder and performs temporal pooling.
    """
    def __init__(self, cfg):
        """
        Initialize the VideoEncoder.

        Args:
            cfg: Configuration object containing parameters such as hidden size.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        DEBUG(f"VideoEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Simple video encoding using frame-based approach
        self.frame_encoder = VisionEncoder(cfg)
        self.temporal_proj = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size)
        )
        
        DEBUG("VideoEncoder: __init__ end")
    
    def forward(self, video_frames):
        """
        Forward pass of the VideoEncoder.

        Args:
            video_frames (torch.Tensor): Input video frames.

        Returns:
            torch.Tensor: Encoded video features.
        """
        if video_frames is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=self.cfg.device)
        
        # Process each frame with vision encoder
        batch_size, num_frames, channels, height, width = video_frames.shape
        frames_flat = video_frames.view(-1, channels, height, width)
        frame_features = self.frame_encoder(frames_flat)
        
        # Reshape back to temporal sequence
        frame_features = frame_features.view(batch_size, num_frames, -1, self.cfg.hidden_size)
        
        # Temporal pooling (simple average)
        video_features = frame_features.mean(dim=2)  # Average across spatial dimensions
        video_features = self.temporal_proj(video_features.mean(dim=1))  # Average across frames
        
        return video_features.unsqueeze(1)