#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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

import os
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
from utils import PiscesLxCoreCacheManagerFacade
from model import get_tokenizer, VisionEncoder, AudioEncoder, DocEncoder, VideoEncoder

IMAGE_KEYS = ["image", "img_path", "image_path", "picture", "pic"]
AUDIO_KEYS = ["audio", "audio_path", "wav", "sound"]
DOC_KEYS = ["doc", "document", "doc_path", "pdf"]
VIDEO_KEYS = ["video", "video_path", "mp4", "avi", "mov", "mkv"]

class PiscesDataset(Dataset):
    """A custom dataset class for Pisces, designed to load and process multi-modal data."""

    def __init__(self, subset: str = "tiny", split: str = "train", config: Optional[Dict[str, Any]] = None, max_samples: Optional[int] = None):
        """Initialize the PiscesDataset instance.

        Args:
            subset (str, optional): The subset of the dataset to load. Defaults to "tiny".
            split (str, optional): The data split to load (e.g., "train", "val", "test"). Defaults to "train".
            config (Optional[Dict[str, Any]], optional): Configuration dictionary for the dataset. Defaults to None.
            max_samples (Optional[int], optional): Maximum number of samples to load. If None, load all samples. Defaults to None.

        Raises:
            FileNotFoundError: If the dataset cache directory does not exist.
        """
        
        self.subset = subset
        self.split = split
        self.config = config or {}

        # Get or create the data cache directory
        cache = PiscesLxCoreCacheManagerFacade.instance()
        data_cache = cache.get_or_create_cache_dir("data_cache")
        cache_path = os.path.join(data_cache, subset)

        # Check if the cache path exists
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Dataset cache not found at {cache_path}. Please run downloader to prepare local cache.")

        # Load the dataset from disk
        ds = load_from_disk(cache_path)
        if isinstance(ds, dict) and split in ds:
            ds = ds[split]
        self.ds = ds

        # Limit the number of samples if max_samples is specified
        if max_samples is not None and len(self.ds) > max_samples:
            self.ds = self.ds.select(range(max_samples))

        # Initialize the tokenizer and encoders
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
        """Retrieve a single sample from the dataset at the specified index.

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
            vocab = len(self.tokenizer)
            ids = torch.clamp(ids, 0, vocab - 1)
        except Exception as e:
            pass
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
        """Extract text from a dataset item.

        Args:
            item (Dict[str, Any]): A dictionary representing a dataset item.

        Returns:
            str: The extracted text. If no text is found, an empty string is returned.
        """
        from data import TEXT_FIELD_KEYS
        if isinstance(item, dict):
            # Try to find text in predefined text field keys
            for k in TEXT_FIELD_KEYS:
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

            # Try to extract text from conversations
            conv = item.get("conversations")
            if isinstance(conv, list) and conv:
                acc = []
                for turn in conv:
                    if isinstance(turn, dict):
                        content = turn.get("value") or turn.get("content") or turn.get("text")
                        if content and str(content).strip():
                            role = turn.get("from", turn.get("role", ""))
                            acc.append(f"{role}: {content}" if role else str(content))
                if acc:
                    return "\n".join(acc)

            # Fallback: find any non-empty string value
            for v in item.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return ""

    def _process_mm(self, item: Dict[str, Any], keys, encoder, kind: str) -> Optional[Any]:
        """Process multi-modal data from a dataset item.

        Args:
            item (Dict[str, Any]): A dictionary representing a dataset item.
            keys (list): A list of keys to search for in the item.
            encoder: The encoder to process the data.
            kind (str): The type of data to process (e.g., "image", "audio", "document", "video").

        Returns:
            Optional[Any]: The processed data if successful, otherwise None.
        """
        if not encoder or not getattr(encoder, "enabled", False):
            return None

        # Find the first valid path from the keys
        p = None
        for k in keys:
            v = item.get(k) if isinstance(item, dict) else None
            if isinstance(v, str) and v.strip():
                p = v.strip()
                break

        if not p:
            return None

        try:
            if kind == "image":
                return encoder.process_image(p)
            if kind == "audio":
                return encoder.process_audio(p)
            if kind == "document":
                return encoder.process_doc(p)
            if kind == "video":
                return encoder.process_video(p)
        except Exception as e:
            pass
        return None