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

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ArcticAudioEncoder(nn.Module):
    """Advanced audio encoder with multi-task learning capabilities."""
    
    def __init__(self, cfg):
        """Initialize the ArcticAudioEncoder.
        
        Args:
            cfg: Configuration object containing hyperparameters.
        """
        super().__init__()
        self.cfg = cfg
        self.enabled = True
        self.hidden_size = cfg.hidden_size
        
        # Audio processing parameters
        self.sampling_rate = getattr(cfg, 'audio_sampling_rate', 16000)
        self.n_mels = getattr(cfg, 'audio_n_mels', 128)
        self.n_fft = getattr(cfg, 'audio_n_fft', 1024)
        self.hop_length = getattr(cfg, 'audio_hop_length', 512)
        self.win_length = getattr(cfg, 'audio_win_length', 1024)
        
        logger.debug("AudioEncoder: __init__ start")
        
        # Create Mel filter bank
        self.mel_filters = self._create_mel_filters()
        
        # Spectral convolution layers
        self.conv_layers = nn.ModuleDict({
            # Spectral convolution for main feature extraction
            'spectral_conv': nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((16, 16))
            ),
            # Multi-task learning modules
            'prosodic_features': nn.Sequential(
                nn.Linear(128 * 16 * 16, 512),
                nn.SiLU(),
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 32)  # Output prosodic feature vector of dimension 32
            ),
            'spectral_features': nn.Sequential(
                nn.Linear(128 * 16 * 16, 512),
                nn.SiLU(),
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 32)  # Output spectral feature vector of dimension 32
            ),
            'emotion_classifier': nn.Sequential(
                nn.Linear(40, 128),  # Input combines prosodic and spectral features
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Linear(64, 7)  # Output classification for 7 basic emotions
            )
        })
        
        # Arousal and valence prediction
        self.arousal_valence = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 2),  # Output arousal and valence scores
            nn.Tanh()  # Output values in range [-1, 1]
        )
        
        # Emotional intensity estimation
        self.intensity_estimator = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output intensity in range [0, 1]
        )
        
        # Enhanced projection with multi-task learning
        self.proj = nn.ModuleDict({
            'main_proj': nn.Sequential(
                nn.Linear(128 * 16 * 16, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.hidden_size, cfg.hidden_size)
            ),
            'speaker_proj': nn.Linear(128, cfg.hidden_size // 4),
            'music_proj': nn.Linear(128, cfg.hidden_size // 4),
            'emotion_proj': nn.Linear(7 + 2 + 1, cfg.hidden_size // 4),  # Combine emotion, arousal/valence, and intensity
            'task_fusion': nn.Sequential(
                nn.Linear(cfg.hidden_size + 3 * (cfg.hidden_size // 4), cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU()
            )
        })
        
        logger.debug("AudioEncoder: __init__ end")

    def _create_mel_filters(self):
        """Create a Mel filter bank."""
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sampling_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.n_mels + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sampling_rate).astype(int)

        filters = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for i in range(1, self.n_mels + 1):
            left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]
            for j in range(left, center):
                filters[i-1, j] = (j - left) / (center - left)
            for j in range(center, right):
                filters[i-1, j] = (right - j) / (right - center)

        return nn.Parameter(torch.from_numpy(filters).float(), requires_grad=False)

    def _stft(self, audio):
        """Perform a short-time Fourier transform (STFT) on the input audio."""
        window = torch.hann_window(self.win_length)
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True
        )
        magnitude = torch.abs(stft)
        # Handle mono audio by adding a batch dimension if needed
        if magnitude.dim() == 2:
            magnitude = magnitude.unsqueeze(0)
        return magnitude

    def _mel_spectrogram(self, audio):
        """Calculate the log Mel spectrogram of the input audio."""
        stft = self._stft(audio)
        mel_spec = torch.matmul(self.mel_filters, stft)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
        return log_mel_spec

    def forward(self, audio_input):
        """Perform a forward pass through the audio encoder.
        
        Args:
            audio_input: Input audio data. Can be a tensor or a dictionary.
            
        Returns:
            torch.Tensor: Encoded audio features.
        """
        if audio_input is None:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.cfg.hidden_size, device=device)

        # Extract audio tensor from input
        if isinstance(audio_input, dict):
            audio_tensor = audio_input.get('input_values', audio_input.get('audio', None))
        else:
            audio_tensor = audio_input

        if audio_tensor is None:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.cfg.hidden_size, device=device)

        # Ensure the audio tensor has the correct shape
        if audio_tensor.dim() == 3:
            audio_tensor = audio_tensor.squeeze(1)

        # Calculate log Mel spectrogram if input is raw audio
        if audio_tensor.dim() == 1:
            mel_spec = self._mel_spectrogram(audio_tensor)
        else:
            mel_spec = audio_tensor

        # Prepare the input for spectral convolution
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)

        # Apply spectral convolution
        x2d = self.conv_layers['spectral_conv'](mel_spec)
        x = x2d.view(x2d.size(0), -1)
        x = self.proj['main_proj'](x)
        # Apply output stability bounds
        x = torch.clamp(x, -10.0, 10.0)

        return x.unsqueeze(1)