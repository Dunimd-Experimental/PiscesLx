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

from __future__ import annotations

"""
Modality scanner for PiscesL1 datasets.

Detects which of the four supported modalities (text, image, audio, video) a dataset
carries, so the training pipeline can route it. Document content is NOT a modality: it
reaches the model through the agent+text path (docs/prompts/UPGRADE_03_ARCH.md 3.3.1).
"""

from typing import Optional, Set
import torch
from torch.utils.data import Dataset, DataLoader

_MODALITY_SIGNALS = {
    'image': {'image', 'pixel_values', 'images', 'img', 'img_path', 'image_path', 'picture', 'pic'},
    'video': {'video', 'video_frames', 'video_path', 'mp4', 'avi'},
    'audio': {'audio', 'audio_values', 'audio_input', 'audio_path', 'wav', 'waveform', 'sound'},
}

class PiscesLxModalityScanner:
    """Operations formerly exposed as module-level functions of modality_scanner.

    Standard §6.2 requires functionality be exposed through a class, so each became a static method here.
"""

    @staticmethod
    def _scan_hf_dataset(ds) -> Set[str]:
        """Scan HuggingFace Dataset features/columns for modality signals."""
        modalities = {'text'}
        col_names = set()
        if hasattr(ds, 'column_names'):
            cols = ds.column_names
            col_names = set(cols) if isinstance(cols, (list, tuple)) else {cols}
        elif hasattr(ds, 'features'):
            col_names = set(ds.features.keys())
        for col in col_names:
            for modality, signals in _MODALITY_SIGNALS.items():
                if col.lower() in signals or col.lower().startswith(tuple(signals)):
                    modalities.add(modality)
        return modalities

    @staticmethod
    def _scan_sample(sample) -> Set[str]:
        """Scan a single sample dict for modality keys."""
        modalities = {'text'}
        if isinstance(sample, dict):
            for key in sample:
                val = sample[key]
                if val is None:
                    continue
                if isinstance(val, torch.Tensor) and val.numel() == 0:
                    continue
                key_lower = key.lower()
                for modality, signals in _MODALITY_SIGNALS.items():
                    if key_lower in signals:
                        modalities.add(modality)
        return modalities

    @staticmethod
    def scan_dataset(dataset, num_samples: int = 3) -> Set[str]:
        """Scan a dataset to detect which modalities are present.

        Args:
            dataset: HuggingFace Dataset, PyTorch Dataset, or DataLoader.
            num_samples: Number of samples to inspect for key-based detection.

        Returns:
            Set of modality names: {'text'}, {'text', 'image'}, etc.
        """
        # Try HF Dataset schema first (fast, no data loading)
        if hasattr(dataset, 'column_names') or hasattr(dataset, 'features'):
            modalities = PiscesLxModalityScanner._scan_hf_dataset(dataset)
            if len(modalities) > 1:
                return modalities

        # Fallback: sample a few items to check keys
        modalities = {'text'}
        try:
            if isinstance(dataset, DataLoader):
                for i, batch in enumerate(dataset):
                    if i >= num_samples:
                        break
                    if isinstance(batch, dict):
                        for key in batch:
                            key_lower = key.lower()
                            for modality, signals in _MODALITY_SIGNALS.items():
                                if key_lower in signals:
                                    modalities.add(modality)
            elif hasattr(dataset, '__getitem__'):
                for i in range(min(num_samples, len(dataset))):
                    sample = dataset[i]
                    modalities |= PiscesLxModalityScanner._scan_sample(sample)
            elif hasattr(dataset, '__iter__'):
                for i, sample in enumerate(dataset):
                    if i >= num_samples:
                        break
                    modalities |= PiscesLxModalityScanner._scan_sample(sample)
        except Exception:
            pass

        return modalities

    @staticmethod
    def scan_dataloader_keys(dataloader: DataLoader, num_batches: int = 1) -> Set[str]:
        """Scan DataLoader output batch keys without consuming the data."""
        modalities = {'text'}
        try:
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                if isinstance(batch, dict):
                    for key in batch:
                        key_lower = key.lower()
                        for modality, signals in _MODALITY_SIGNALS.items():
                            if key_lower in signals:
                                modalities.add(modality)
        except Exception:
            pass
        return modalities
