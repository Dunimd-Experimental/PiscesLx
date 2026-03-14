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

import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
from model.tokenizer import YvTokenizer
from utils.paths import get_cache_dir
from model.multimodal import YvVisionEncoder as VisionEncoder, YvAudioEncoder as AudioEncoder, YvDocEncoder as DocEncoder, YvVideoEncoder as VideoEncoder

IMAGE_KEYS = [
    "image", "img_path", "image_path", "picture", "pic",
    "img", "images", "img_file", "image_file", "photo",
    "screenshot", "frame", "frames", "visual", "visual_input"
]
AUDIO_KEYS = [
    "audio", "audio_path", "wav", "sound", 
    "audio_file", "audio_input", "speech", "voice",
    "waveform", "spectrogram", "mel", "audio_data"
]
DOC_KEYS = [
    "doc", "document", "doc_path", "pdf",
    "document_path", "doc_file", "text_file", "txt",
    "markdown", "md", "html", "document_input", "file_path"
]
VIDEO_KEYS = [
    "video", "video_path", "mp4", "avi", "mov", "mkv",
    "video_file", "video_input", "clip", "movie", "footage",
    "video_data", "frames_path", "video_frames_path"
]

class Dataset:
    """A dataset class for Pisces that supports both text and multimodal data.

    This class provides a unified interface for handling different types of data
    including text, images, audio, and video. It supports various data formats
    and provides efficient data loading and processing capabilities.
    """

    def __init__(self, name: str, subset: Optional[str] = None, split: str = "train", config: Optional[Dict[str, Any]] = None, cache_dir: Optional[str] = None, max_samples: Optional[int] = None):
        """Initialize the Dataset instance.

        Args:
            name: Name of the dataset.
            subset: Subset of the dataset (optional).
            split: Data split to use (e.g., 'train', 'test', 'validation').
            config: Configuration dictionary (optional). Supports:
                - custom_image_keys: List of custom keys for image data
                - custom_audio_keys: List of custom keys for audio data
                - custom_doc_keys: List of custom keys for document data
                - custom_video_keys: List of custom keys for video data
                - force_enable_vision: Force enable vision encoder
                - force_enable_audio: Force enable audio encoder
                - force_enable_doc: Force enable doc encoder
                - force_enable_video: Force enable video encoder
            cache_dir: Directory to cache the dataset (optional).
            max_samples: Maximum number of samples to load (optional).
        """
        
        self.subset = subset
        self.split = split
        self.config = config or {}
        try:
            from types import SimpleNamespace
            if isinstance(self.config, dict):
                self.config = SimpleNamespace(**self.config)
        except Exception:
            pass
        self.max_samples = max_samples
        
        # Allow custom key names from config
        self._image_keys = getattr(self.config, 'custom_image_keys', None) or IMAGE_KEYS
        self._audio_keys = getattr(self.config, 'custom_audio_keys', None) or AUDIO_KEYS
        self._doc_keys = getattr(self.config, 'custom_doc_keys', None) or DOC_KEYS
        self._video_keys = getattr(self.config, 'custom_video_keys', None) or VIDEO_KEYS
        
        # Force enable flags for modalities (useful when keys don't match)
        self._force_vision = getattr(self.config, 'force_enable_vision', False)
        self._force_audio = getattr(self.config, 'force_enable_audio', False)
        self._force_doc = getattr(self.config, 'force_enable_doc', False)
        self._force_video = getattr(self.config, 'force_enable_video', False)

        data_cache = cache_dir or get_cache_dir("data_cache")
        cache_path = os.path.join(str(data_cache), self.subset)

        # Check if the dataset cache path exists
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Dataset cache not found at {cache_path}. Please run downloader to prepare local cache.")

        # Load dataset from the cache path
        ds = load_from_disk(cache_path)
        if isinstance(ds, dict) and self.split in ds:
            ds = ds[self.split]
        self.ds = ds

        # Limit the number of samples if max_samples is specified
        if self.max_samples is not None and len(self.ds) > self.max_samples:
            self.ds = self.ds.select(range(self.max_samples))

        # Initialize the tokenizer and modality encoders
        self.tokenizer = YvTokenizer()
        # Lazy initialization: Do NOT create multimodal encoders at init time.
        # They will be created on-demand or use the model's encoders during training.
        # This prevents OOM during model initialization on low-VRAM systems.
        self._vision_encoder = None
        self._audio_encoder = None
        self._doc_encoder = None
        self._video_encoder = None

    @property
    def vision_encoder(self):
        """Lazy-loaded vision encoder."""
        if self._vision_encoder is None and self.config:
            self._vision_encoder = VisionEncoder(self.config)
        return self._vision_encoder

    @property
    def audio_encoder(self):
        """Lazy-loaded audio encoder."""
        if self._audio_encoder is None and self.config:
            self._audio_encoder = AudioEncoder(self.config)
        return self._audio_encoder

    @property
    def doc_encoder(self):
        """Lazy-loaded doc encoder."""
        if self._doc_encoder is None and self.config:
            self._doc_encoder = DocEncoder(self.config)
        return self._doc_encoder

    @property
    def video_encoder(self):
        """Lazy-loaded video encoder."""
        if self._video_encoder is None and self.config:
            self._video_encoder = VideoEncoder(self.config)
        return self._video_encoder

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve and process a single sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing the processed sample data.
        """
        item = self.ds[idx]
        text = self._extract_text(item)
        if not text:
            text = "<empty>"
        try:
            ids = self.tokenizer.encode(text, return_tensors="pt")[0]
            vocab_size = len(self.tokenizer)
            ids = torch.clamp(ids, 0, vocab_size - 1)
        except Exception:
            ids = torch.tensor([0], dtype=torch.long)
        
        # Optimized: Only create/access encoders when data actually exists for that modality
        # This prevents unnecessary encoder initialization for modalities not present in the data
        # Uses custom keys from config and respects force_enable flags
        pixel_values = self._process_mm_lazy(item, self._image_keys, "vision_encoder", "image", self._force_vision)
        audio_input = self._process_mm_lazy(item, self._audio_keys, "audio_encoder", "audio", self._force_audio)
        doc_input = self._process_mm_lazy(item, self._doc_keys, "doc_encoder", "document", self._force_doc)
        video_frames = self._process_mm_lazy(item, self._video_keys, "video_encoder", "video", self._force_video)
        
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "pixel_values": pixel_values,
            "audio_input": audio_input if audio_input is not None else {"input_values": None},
            "doc_input": doc_input,
            "video_frames": video_frames,
        }

    def _process_mm_lazy(self, item: Dict[str, Any], keys: list, encoder_attr: str, kind: str, force_enable: bool = False) -> Optional[Any]:
        """Process multi-modal data with lazy encoder initialization.

        Only creates the encoder when the data actually contains this modality,
        or when force_enable is True.

        Args:
            item: A dictionary representing a dataset item.
            keys: A list of keys to search for in the item.
            encoder_attr: The attribute name of the encoder (e.g., "vision_encoder").
            kind: The type of data to process, e.g., "image", "audio", "document", "video".
            force_enable: Force create encoder even if no data found (default: False).

        Returns:
            The processed data if successful, otherwise None.
        """
        # First check if data exists for this modality
        path = None
        for key in keys:
            value = item.get(key) if isinstance(item, dict) else None
            if isinstance(value, str) and value.strip():
                path = value.strip()
                break

        # If no data found and not forced, skip encoder creation
        if not path and not force_enable:
            return None

        # Data exists or forced, now get/create the encoder (lazy initialization)
        encoder = getattr(self, encoder_attr)
        if not encoder or not getattr(encoder, "enabled", False):
            return None
        
        # If forced but no path, return None (encoder created but no data to process)
        if not path:
            return None

        try:
            if kind == "image":
                return encoder.process_image(path)
            if kind == "audio":
                return encoder.process_audio(path)
            if kind == "document":
                return encoder.process_doc(path)
            if kind == "video":
                return encoder.process_video(path)
        except Exception:
            pass
        return None

    def _extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from a dataset item through multiple strategies.

        Args:
            item (Dict[str, Any]): A dictionary representing a dataset item.

        Returns:
            str: The extracted text. Returns an empty string if no text is found.
        """
        from tools.data import TEXT_FIELD_KEYS
        if isinstance(item, dict):
            # Try to find text using predefined text field keys
            for key in TEXT_FIELD_KEYS:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            # Try to extract text from conversation data
            conversations = item.get("conversations")
            if isinstance(conversations, list) and conversations:
                text_parts = []
                for turn in conversations:
                    if isinstance(turn, dict):
                        content = turn.get("value") or turn.get("content") or turn.get("text")
                        if content and str(content).strip():
                            role = turn.get("from", turn.get("role", ""))
                            text_parts.append(f"{role}: {content}" if role else str(content))
                if text_parts:
                    return "\n".join(text_parts)

            # Fallback strategy: find any non-empty string value
            for value in item.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""
