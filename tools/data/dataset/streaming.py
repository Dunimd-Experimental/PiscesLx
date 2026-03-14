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
import json
import torch
from .memory import PiscesLxToolsMemoryMonitor
from torch.utils.data import IterableDataset
from typing import Iterator, Dict, List, Optional
from model.tokenizer import YvTokenizer
from model.multimodal import YvVisionEncoder, YvAudioEncoder, YvDocEncoder, YvVideoEncoder

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

class PiscesLxToolsDataLargeScaleStreamingDataset(IterableDataset):
    """An iterable dataset for large-scale streaming data processing.
    
    This dataset supports reading data from multiple sources, including directories and files,
    and processing various file formats such as JSONL, JSON, and TXT. It also provides support
    for multimodal data processing.
    """
    def __init__(self, data_sources: List[str], config: Optional[dict] = None):
        """Initialize the PiscesLxToolsLargeScaleStreamingDataset.

        Args:
            data_sources (List[str]): List of paths to data sources, which can be files or directories.
            config (Optional[dict]): Configuration dictionary. Supports:
                - custom_image_keys: List of custom keys for image data
                - custom_audio_keys: List of custom keys for audio data
                - custom_doc_keys: List of custom keys for document data
                - custom_video_keys: List of custom keys for video data
                - force_enable_vision: Force enable vision encoder
                - force_enable_audio: Force enable audio encoder
                - force_enable_doc: Force enable doc encoder
                - force_enable_video: Force enable video encoder
        """
        super().__init__()
        
        self.data_sources = data_sources
        self.tokenizer = YvTokenizer()
        self.config = config or {}
        self.memory = PiscesLxToolsMemoryMonitor(threshold_gb=8.0)
        # Lazy initialization: Do NOT create multimodal encoders at init time.
        # They will be created on-demand or use the model's encoders during training.
        # This prevents OOM during model initialization on low-VRAM systems.
        self._vision_encoder = None
        self._audio_encoder = None
        self._doc_encoder = None
        self._video_encoder = None
        
        # Allow custom key names from config
        self._image_keys = self.config.get('custom_image_keys', None) or IMAGE_KEYS
        self._audio_keys = self.config.get('custom_audio_keys', None) or AUDIO_KEYS
        self._doc_keys = self.config.get('custom_doc_keys', None) or DOC_KEYS
        self._video_keys = self.config.get('custom_video_keys', None) or VIDEO_KEYS
        
        # Force enable flags for modalities (useful when keys don't match)
        self._force_vision = self.config.get('force_enable_vision', False)
        self._force_audio = self.config.get('force_enable_audio', False)
        self._force_doc = self.config.get('force_enable_doc', False)
        self._force_video = self.config.get('force_enable_video', False)
        
        self._index: List[Dict] = self._build_index()

    @property
    def vision_encoder(self):
        """Lazy-loaded vision encoder."""
        if self._vision_encoder is None and self.config:
            self._vision_encoder = YvVisionEncoder(self.config)
        return self._vision_encoder

    @property
    def audio_encoder(self):
        """Lazy-loaded audio encoder."""
        if self._audio_encoder is None and self.config:
            self._audio_encoder = YvAudioEncoder(self.config)
        return self._audio_encoder

    @property
    def doc_encoder(self):
        """Lazy-loaded doc encoder."""
        if self._doc_encoder is None and self.config:
            self._doc_encoder = YvDocEncoder(self.config)
        return self._doc_encoder

    @property
    def video_encoder(self):
        """Lazy-loaded video encoder."""
        if self._video_encoder is None and self.config:
            self._video_encoder = YvVideoEncoder(self.config)
        return self._video_encoder

    def _build_index(self) -> List[Dict]:
        """Build an index of all valid data files.

        Traverse through the provided data sources, collect paths of all JSONL, JSON, and TXT files.

        Returns:
            List[Dict]: A list of dictionaries, each containing a 'path' key pointing to a data file.
        """
        idx: List[Dict] = []
        for src in self.data_sources:
            if os.path.isdir(src):
                for root, _, files in os.walk(src):
                    for f in files:
                        if f.endswith((".jsonl", ".json", ".txt")):
                            p = os.path.join(root, f)
                            idx.append({"path": p})
            elif os.path.isfile(src):
                idx.append({"path": src})
        return idx

    def __iter__(self) -> Iterator[Dict]:
        """Return an iterator over the dataset.

        Yields data samples from all indexed files.

        Returns:
            Iterator[Dict]: An iterator that yields processed data samples.
        """
        for fi in self._index:
            yield from self._iter_file(fi["path"])

    def _iter_file(self, path: str) -> Iterator[Dict]:
        """Iterate over a single data file and yield processed samples.

        Supports JSONL, JSON, and TXT file formats.

        Args:
            path (str): Path to the data file.

        Returns:
            Iterator[Dict]: An iterator that yields processed data samples from the file.
        """
        try:
            if path.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as f:
                    for ln, line in enumerate(f):
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        yield self._process_one(obj, f"{path}:{ln}")
            elif path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    for i, it in enumerate(obj):
                        yield self._process_one(it, f"{path}:{i}")
                elif isinstance(obj, dict):
                    yield self._process_one(obj, path)
            elif path.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    for ln, line in enumerate(f):
                        text = line.strip()
                        if not text:
                            continue
                        yield self._process_one({"text": text}, f"{path}:{ln}")
        except Exception as e:
            pass

    def _process_one(self, raw: Dict, sid: str) -> Dict:
        """Process a single raw data sample.

        Tokenize the text in the sample, extract multimodal data, and format the output.

        Args:
            raw (Dict): Raw data sample.
            sid (str): Sample ID.

        Returns:
            Dict: Processed data sample containing input IDs, labels, sample ID, and multimodal data.
        """
        try:
            from tools.data import TEXT_FIELD_KEYS
            text = ""
            if isinstance(raw, dict):
                for k in TEXT_FIELD_KEYS:
                    v = raw.get(k)
                    if isinstance(v, str) and v.strip():
                        text = v.strip()
                        break
            if not text:
                text = "<empty>"
            ids = self.tokenizer.encode(text, return_tensors="pt")[0]
            vocab = len(self.tokenizer)
            ids = torch.clamp(ids, 0, vocab - 1)
            mm = self._extract_mm(raw)
            return {"input_ids": ids, "labels": ids.clone(), "sample_id": sid, **mm}
        except Exception as e:
            return {"input_ids": torch.tensor([0], dtype=torch.long), "labels": torch.tensor([0], dtype=torch.long), "sample_id": sid}

    def _extract_mm(self, sample: Dict) -> Dict:
        """Extract multimodal data from a sample.

        Process image, audio, document, and video data only when the data exists.
        Uses lazy encoder initialization to avoid creating encoders for unused modalities.
        Supports custom keys and force_enable flags from config.

        Args:
            sample (Dict): Data sample containing multimodal data.

        Returns:
            Dict: A dictionary containing processed multimodal data.
        """
        out = {"pixel_values": None, "audio_input": None, "doc_input": None, "video_frames": None}
        try:
            # Optimized: Check if data exists BEFORE accessing encoder (lazy init)
            # Uses custom keys from config
            ip = self._first_valid(sample, self._image_keys)
            if ip or self._force_vision:
                encoder = self.vision_encoder
                if encoder and encoder.enabled and ip:
                    out["pixel_values"] = encoder.process_image(ip)
            
            ap = self._first_valid(sample, self._audio_keys)
            if ap or self._force_audio:
                encoder = self.audio_encoder
                if encoder and encoder.enabled and ap:
                    out["audio_input"] = encoder.process_audio(ap)
            
            dp = self._first_valid(sample, self._doc_keys)
            if dp or self._force_doc:
                encoder = self.doc_encoder
                if encoder and encoder.enabled and dp:
                    out["doc_input"] = encoder.process_doc(dp)
            
            vp = self._first_valid(sample, self._video_keys)
            if (vp or self._force_video) and self.memory.check_memory()["system_available_gb"] > 4.0:
                encoder = self.video_encoder
                if encoder and encoder.enabled and vp:
                    out["video_frames"] = encoder.process_video(vp)
        except Exception as e:
            pass
        return out

    @staticmethod
    def _first_valid(item: Dict, keys: List[str]) -> Optional[str]:
        """Find the first valid string value for the given keys in a dictionary.

        Args:
            item (Dict): Dictionary to search in.
            keys (List[str]): List of keys to check.

        Returns:
            Optional[str]: The first valid string value found, or None if no valid value is found.
        """
        for k in keys:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None
