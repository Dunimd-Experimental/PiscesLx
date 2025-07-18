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
from torch import nn
from transformers import CLIPImageProcessor, ASTFeatureExtractor


class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        print(f"🟧\tVisionEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        self.processor = CLIPImageProcessor()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.proj = nn.Sequential(
            nn.Linear(64 * 56 * 56, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size)
        )
        
        print("🟧\tVisionEncoder: __init__ end")
    
    def process_image(self, image_path):
        """Process image data"""
        print(f"🟧\tProcessing image: {image_path}")
        try:
            image = self.processor(images=image_path, return_tensors="pt")
            return image['pixel_values'][0]
        except Exception as e:
            print(f"❌\tImage processing error: {e}")
            return None
    
    def forward(self, pixel_values):
        if pixel_values is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=pixel_values.device)
        
        x = self.conv1(pixel_values)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x.unsqueeze(1)


class AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        print(f"🟧\tAudioEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        self.processor = ASTFeatureExtractor()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=8, stride=8)
        
        self.proj = nn.Sequential(
            nn.Linear(64 * 128, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size)
        )
        
        print("🟧\tAudioEncoder: __init__ end")
    
    def process_audio(self, audio_path):
        print(f"🟧\tProcessing audio: {audio_path}")
        try:
            audio = self.processor(audio=audio_path, return_tensors="pt")
            return audio['input_values'][0]
        except Exception as e:
            print(f"❌\tAudio processing error: {e}")
            return None
    
    def forward(self, audio_input):
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
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        print(f"🟧\tDocEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        self.doc_proj = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size)
        )
        
        print("🟧\tDocEncoder: __init__ end")
    
    def forward(self, doc_input):
        x = self.doc_proj(doc_input['input_ids'])
        return x.unsqueeze(1)