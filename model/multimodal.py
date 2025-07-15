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
        print("[DEBUG] VisionEncoder: __init__ start (disabled)")
        self.enabled = False
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # Placeholder
        print("[DEBUG] VisionEncoder: __init__ end (disabled)")
    
    def forward(self, pixel_values):
        return torch.zeros(pixel_values.shape[0], 1, self.proj.out_features, device=pixel_values.device)


class AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("[DEBUG] AudioEncoder: __init__ start (disabled)")
        self.enabled = False
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # Placeholder
        print("[DEBUG] AudioEncoder: __init__ end (disabled)")
    
    def forward(self, audio_input):
        return torch.zeros(audio_input['input_values'].shape[0], 1, self.proj.out_features, device=audio_input['input_values'].device)


class DocEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print("[DEBUG] DocEncoder: __init__ start (disabled)")
        self.enabled = False
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)  # Placeholder
        print("[DEBUG] DocEncoder: __init__ end (disabled)")
    
    def forward(self, doc_input):
        return torch.zeros(doc_input['input_ids'].shape[0], 1, self.proj.out_features, device=doc_input['input_ids'].device)