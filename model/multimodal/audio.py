#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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
import numpy as np
from torch import nn
from typing import Dict, Any
import torch.nn.functional as F
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("Arctic.Core.Multimodal")

class ArcticAudioEncoder(nn.Module):
    """A PyTorch module for audio encoding with multi-task capabilities.

    This encoder processes audio inputs to extract various features including speaker information,
    music understanding, and emotion recognition. It supports multi-scale feature extraction and
    multi-task learning.
    """

    def __init__(self, cfg):
        """Initialize the ArcticAudioEncoder module.

        Args:
            cfg: Configuration object containing hyperparameters. Expected to have 'hidden_size' attribute.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.sampling_rate = 16000
        self.n_mels = 128
        self.n_fft = 1024
        self.hop_length = 512
        self.win_length = 1024

        logger.debug(f"AudioEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")

        # Create Mel filter bank
        self.mel_filters = self._create_mel_filters()

        # Module for enhanced audio feature extraction using multi-scale processing
        self.conv_layers = nn.ModuleDict({
            # Module list for multi-scale temporal convolution
            'temporal_conv': nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=k, stride=2, padding=k//2),
                    nn.BatchNorm1d(32),
                    nn.SiLU(),
                    nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm1d(64),
                    nn.SiLU(),
                    nn.AdaptiveAvgPool1d(128)
                ) for k in [3, 5, 7, 11]  # Using different kernel sizes for multi-scale processing
            ]),
            # Sequential module for spectral feature extraction
            'spectral_conv': nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
                nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((16, 16))
            )
        })

        # Module for gradient-safe speaker separation
        self.speaker_separation = nn.ModuleDict({
            # Sequential module for simplified speaker encoding
            'speaker_encoder': nn.Sequential(
                nn.Linear(128 * 16 * 16, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)  # Output speaker embedding of dimension 128
            ),
            # Sequential module for simplified voice activity detection
            'vad_network': nn.Sequential(
                nn.Conv1d(128, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 1, kernel_size=1),
                nn.Sigmoid()  # Output voice activity probability
            ),
            # Module list for generating separation masks for up to 2 speakers
            'separation_masks': nn.ModuleList([
                nn.Sequential(
                    nn.Linear(128 + 128, 128),  # Input combines audio and speaker embeddings
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.Sigmoid()  # Output separation mask for each speaker
                ) for _ in range(2)
            ]),
            # Sequential module for simplified speaker verification
            'speaker_verifier': nn.Sequential(
                nn.Linear(256, 64),  # Input is concatenated speaker embeddings
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output probability of speakers being the same
            )
        })

        # Module for gradient-safe music understanding
        self.music_understanding = nn.ModuleDict({
            # Sequential module for simplified instrument recognition
            'instrument_classifier': nn.Sequential(
                nn.Linear(128 * 16 * 16, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 128)  # Output classification for 128 instrument classes
            ),
            # Sequential module for simplified genre classification
            'genre_classifier': nn.Sequential(
                nn.Linear(128 * 16 * 16, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 20)  # Output classification for 20 major genres
            ),
            # Module for simplified rhythm analysis
            'rhythm_analyzer': nn.ModuleDict({
                # Sequential module for tempo estimation
                'tempo_estimator': nn.Sequential(
                    nn.Conv1d(128, 64, kernel_size=7, padding=3),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()  # Output normalized tempo in range [0, 1]
                ),
                # Sequential module for beat tracking
                'beat_tracker': nn.Sequential(
                    nn.Conv1d(128, 32, kernel_size=3, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(32, 1, kernel_size=1),
                    nn.Sigmoid()  # Output beat probability sequence
                )
            })
        })

        # Module for comprehensive emotion recognition
        self.emotion_recognition = nn.ModuleDict({
            # Module for speech emotion analysis
            'speech_emotion': nn.ModuleDict({
                # Sequential module for extracting prosodic features
                'prosodic_features': nn.Sequential(
                    nn.Linear(128, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                    nn.Linear(32, 8)  # Output prosodic feature vector of dimension 8
                ),
                # Sequential module for extracting spectral features
                'spectral_features': nn.Sequential(
                    nn.Linear(128 * 16 * 16, 512),
                    nn.SiLU(),
                    nn.Linear(512, 256),
                    nn.SiLU(),
                    nn.Linear(256, 32)  # Output spectral feature vector of dimension 32
                ),
                # Sequential module for emotion classification
                'emotion_classifier': nn.Sequential(
                    nn.Linear(40, 128),  # Input combines prosodic and spectral features
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.LayerNorm(64),
                    nn.SiLU(),
                    nn.Linear(64, 7)  # Output classification for 7 basic emotions
                )
            }),
            # Sequential module for predicting arousal and valence
            'arousal_valence': nn.Sequential(
                nn.Linear(128 * 16 * 16, 256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 2),  # Output arousal and valence scores
                nn.Tanh()  # Output values in range [-1, 1]
            ),
            # Sequential module for estimating emotional intensity
            'intensity_estimator': nn.Sequential(
                nn.Linear(128 * 16 * 16, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Output intensity in range [0, 1]
            )
        })

        # Module for enhanced projection with multi-task learning
        self.proj = nn.ModuleDict({
            # Sequential module for main projection
            'main_proj': nn.Sequential(
                nn.Linear(128 * 16 * 16, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(cfg.hidden_size, cfg.hidden_size)
            ),
            # Linear layer for speaker-specific projection
            'speaker_proj': nn.Linear(128, cfg.hidden_size // 4),
            # Linear layer for music-specific projection
            'music_proj': nn.Linear(128, cfg.hidden_size // 4),
            # Linear layer for emotion-specific projection
            'emotion_proj': nn.Linear(7 + 2 + 1, cfg.hidden_size // 4),  # Combine emotion, arousal/valence, and intensity
            # Sequential module for multi-task fusion
            'task_fusion': nn.Sequential(
                nn.Linear(cfg.hidden_size + 3 * (cfg.hidden_size // 4), cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU()
            )
        })

        logger.debug("AudioEncoder: __init__ end")

    def _create_mel_filters(self):
        """Create a Mel filter bank.

        Returns:
            nn.Parameter: A non-trainable parameter containing the Mel filter bank.
        """
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
        """Perform a short-time Fourier transform (STFT) on the input audio.

        Args:
            audio (torch.Tensor): Input audio tensor.

        Returns:
            torch.Tensor: Magnitude spectrogram obtained from the STFT.
        """
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
        """Calculate the log Mel spectrogram of the input audio.

        Args:
            audio (torch.Tensor): Input audio tensor.

        Returns:
            torch.Tensor: Log Mel spectrogram.
        """
        stft = self._stft(audio)
        mel_spec = torch.matmul(self.mel_filters, stft)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
        return log_mel_spec

    def process_audio(self, audio_path):
        """Process an audio file and return its normalized log Mel spectrogram.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Processed and normalized log Mel spectrogram with an added batch dimension.
        """
        logger.debug(f"Processing audio: {audio_path}")
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            audio = torch.from_numpy(audio).float()

            # Calculate log Mel spectrogram
            mel_spec = self._mel_spectrogram(audio)

            # Normalize the spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

            return mel_spec.unsqueeze(0)
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return torch.zeros(1, self.n_mels, 64)

    def forward(self, audio_input):
        """Perform a forward pass through the audio encoder.

        Args:
            audio_input (torch.Tensor or dict): Input audio data. Can be a tensor or a dictionary 
                containing 'input_values' or 'audio' keys.

        Returns:
            torch.Tensor: Encoded audio features with shape (batch_size, 1, hidden_size).
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

        return x.unsqueeze(1)