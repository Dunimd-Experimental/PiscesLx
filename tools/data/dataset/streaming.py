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
import json
import torch
from .memory import PiscesLxToolsMemoryMonitor
from torch.utils.data import IterableDataset
from typing import Iterator, Dict, List, Optional
from model.tokenizer import get_tokenizer
from model.multimodal import RuchbahVisionEncoder, RuchbahAudioEncoder, RuchbahDocEncoder, RuchbahVideoEncoder

IMAGE_KEYS = ["image", "img_path", "image_path", "picture", "pic"]
AUDIO_KEYS = ["audio", "audio_path", "wav", "sound"]
DOC_KEYS = ["doc", "document", "doc_path", "pdf"]
VIDEO_KEYS = ["video", "video_path", "mp4", "avi", "mov", "mkv"]

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
            config (Optional[dict]): Configuration dictionary. If None, default settings will be used.
        """
        super().__init__()
        
        self.data_sources = data_sources
        self.tokenizer = get_tokenizer()
        self.config = config or {}
        self.memory = PiscesLxToolsMemoryMonitor(threshold_gb=8.0)
        # Initialize multimodal encoders if config is provided
        self.vision_encoder = RuchbahVisionEncoder(config) if config else None
        self.audio_encoder = RuchbahAudioEncoder(config) if config else None
        self.doc_encoder = RuchbahDocEncoder(config) if config else None
        self.video_encoder = RuchbahVideoEncoder(config) if config else None
        self._index: List[Dict] = self._build_index()

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

        Process image, audio, document, and video data if the corresponding encoders are enabled.

        Args:
            sample (Dict): Data sample containing multimodal data.

        Returns:
            Dict: A dictionary containing processed multimodal data.
        """
        out = {"pixel_values": None, "audio_input": None, "doc_input": None, "video_frames": None}
        try:
            if self.vision_encoder and self.vision_encoder.enabled:
                ip = self._first_valid(sample, IMAGE_KEYS)
                if ip: out["pixel_values"] = self.vision_encoder.process_image(ip)
            if self.audio_encoder and self.audio_encoder.enabled:
                ap = self._first_valid(sample, AUDIO_KEYS)
                if ap: out["audio_input"] = self.audio_encoder.process_audio(ap)
            if self.doc_encoder and self.doc_encoder.enabled:
                dp = self._first_valid(sample, DOC_KEYS)
                if dp: out["doc_input"] = self.doc_encoder.process_doc(dp)
            if self.video_encoder and self.video_encoder.enabled:
                vp = self._first_valid(sample, VIDEO_KEYS)
                if vp and self.memory.check_memory()["system_available_gb"] > 4.0:
                    out["video_frames"] = self.video_encoder.process_video(vp)
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