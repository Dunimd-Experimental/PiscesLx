#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
from model.tokenizer import get_tokenizer
from utils import PiscesLxCoreCacheManagerFacade
from model.multimodal import RuchbahVisionEncoder as VisionEncoder, RuchbahAudioEncoder as AudioEncoder, RuchbahDocEncoder as DocEncoder, RuchbahVideoEncoder as VideoEncoder

IMAGE_KEYS = ["image", "img_path", "image_path", "picture", "pic"]
AUDIO_KEYS = ["audio", "audio_path", "wav", "sound"]
DOC_KEYS = ["doc", "document", "doc_path", "pdf"]
VIDEO_KEYS = ["video", "video_path", "mp4", "avi", "mov", "mkv"]

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
            config: Configuration dictionary (optional).
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

        # Get the instance of cache manager and create data cache directory
        cache = PiscesLxCoreCacheManagerFacade.get_instance()
        data_cache = cache.get_cache_dir("data_cache")
        cache_path = os.path.join(data_cache, self.subset)

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
        self.tokenizer = get_tokenizer()
        self.vision_encoder = VisionEncoder(self.config) if self.config else None
        self.audio_encoder = AudioEncoder(self.config) if self.config else None
        self.doc_encoder = DocEncoder(self.config) if self.config else None
        self.video_encoder = VideoEncoder(self.config) if self.config else None

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
        pixel_values = self._process_mm(item, IMAGE_KEYS, self.vision_encoder, "image")
        audio_input = self._process_mm(item, AUDIO_KEYS, self.audio_encoder, "audio")
        doc_input = self._process_mm(item, DOC_KEYS, self.doc_encoder, "document")
        video_frames = self._process_mm(item, VIDEO_KEYS, self.video_encoder, "video")
        return {
            "input_ids": ids,
            "labels": ids.clone(),
            "pixel_values": pixel_values,
            "audio_input": audio_input if audio_input is not None else {"input_values": None},
            "doc_input": doc_input,
            "video_frames": video_frames,
        }

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

    def _process_mm(self, item: Dict[str, Any], keys: list, encoder, kind: str) -> Optional[Any]:
        """Process multi-modal data from a dataset item.

        Args:
            item (Dict[str, Any]): A dictionary representing a dataset item.
            keys (list): A list of keys to search for in the item.
            encoder: The encoder to process the data.
            kind (str): The type of data to process, e.g., "image", "audio", "document", "video".

        Returns:
            Optional[Any]: The processed data if successful, otherwise None.
        """
        if not encoder or not getattr(encoder, "enabled", False):
            return None

        # Find the first valid path from the given keys
        path = None
        for key in keys:
            value = item.get(key) if isinstance(item, dict) else None
            if isinstance(value, str) and value.strip():
                path = value.strip()
                break

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
