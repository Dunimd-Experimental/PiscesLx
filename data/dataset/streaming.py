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
import json
import torch
from .memory import MemoryMonitor
from torch.utils.data import IterableDataset
from typing import Iterator, Dict, List, Optional
from model import get_tokenizer, VisionEncoder, AudioEncoder, DocEncoder, VideoEncoder



IMAGE_KEYS = ["image", "img_path", "image_path", "picture", "pic"]
AUDIO_KEYS = ["audio", "audio_path", "wav", "sound"]
DOC_KEYS = ["doc", "document", "doc_path", "pdf"]
VIDEO_KEYS = ["video", "video_path", "mp4", "avi", "mov", "mkv"]

class LargeScaleStreamingDataset(IterableDataset):
    def __init__(self, data_sources: List[str], config: Optional[dict] = None):
        super().__init__()
        
        self.data_sources = data_sources
        self.tokenizer = get_tokenizer()
        self.config = config or {}
        self.memory = MemoryMonitor(threshold_gb=8.0)
        # multimodal encoders are optional
        self.vision_encoder = VisionEncoder(config) if config else None
        self.audio_encoder = AudioEncoder(config) if config else None
        self.doc_encoder = DocEncoder(config) if config else None
        self.video_encoder = VideoEncoder(config) if config else None
        self._index: List[Dict] = self._build_index()

    def _build_index(self) -> List[Dict]:
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
        pass
        return idx

    def __iter__(self) -> Iterator[Dict]:
        for fi in self._index:
            yield from self._iter_file(fi["path"])

    def _iter_file(self, path: str) -> Iterator[Dict]:
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
        try:
            from data import TEXT_FIELD_KEYS
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
            pass
            return {"input_ids": torch.tensor([0], dtype=torch.long), "labels": torch.tensor([0], dtype=torch.long), "sample_id": sid}

    def _extract_mm(self, sample: Dict) -> Dict:
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
        for k in keys:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None