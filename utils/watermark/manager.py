#!/usr/bin/env/python3

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

from typing import Dict, Any, Optional, Union
import json
import torch

from utils.watermark.protocol import verify_payload
from utils.watermark.text import TextWatermarker
from utils.watermark.image import ImageWatermarker
from utils.watermark.audio import AudioWatermarker
from utils.watermark.video import VideoWatermarker
from utils.watermark.detection import decision

class PiscesWatermarkManager:
    def __init__(self, model_id: str = "PiscesL1-1.5B", version: str = "1.0.0", config: Dict[str, Any] = None):
        self.model_id = model_id
        self.version = version
        self.config = config or self._load_config()
        self.text_wm = TextWatermarker(model_id, version, self.config)
        self.image_wm = ImageWatermarker(model_id, version, self.config)
        self.audio_wm = AudioWatermarker(model_id, version, self.config)
        self.video_wm = VideoWatermarker(model_id, version, self.config)

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open('configs/watermark.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _detect_content_type(self, content: Union[str, torch.Tensor]) -> str:
        if isinstance(content, str):
            return "text"
        if isinstance(content, torch.Tensor):
            shape = content.shape
            if len(shape) == 3 and shape[0] in [1, 3, 4]:
                return "image"
            if len(shape) == 1 or (len(shape) == 2 and shape[0] == 1):
                return "audio"
            if len(shape) == 4:
                return "video"
        return "unknown"

    def add_watermark(self, content: Union[str, torch.Tensor], metadata: Dict[str, Any]) -> Union[str, torch.Tensor]:
        ct = self._detect_content_type(content)
        if ct == "text":
            return self.text_wm.embed(content, metadata)
        if ct == "image":
            return self.image_wm.embed(content, metadata)
        if ct == "audio":
            return self.audio_wm.embed(content, metadata)
        if ct == "video":
            return self.video_wm.embed(content, metadata)
        return content

    def check_watermark(self, content: Union[str, torch.Tensor]) -> Optional[Dict[str, Any]]:
        if isinstance(content, str):
            return self.text_wm.extract(content)
        # 图像/音频P0仅返回占位信息，P1会加入真正检测器
        if isinstance(content, torch.Tensor):
            ct = self._detect_content_type(content)
            if ct == "image":
                # 无需payload也能打分
                return self.image_wm.detect_score(content, None)
            if ct == "audio":
                return self.audio_wm.detect_score(content, None)
            if ct == "video":
                return self.video_wm.detect_score(content, None)
        return None

    def verify_watermark(self, content: Union[str, torch.Tensor]) -> bool:
        if isinstance(content, str):
            payload = self.text_wm.extract(content)
            if not payload:
                return False
            ok_sig = verify_payload(payload, self.config) and payload.get("model_id") == self.model_id
            return bool(ok_sig)
        if isinstance(content, torch.Tensor):
            ct = self._detect_content_type(content)
            if ct == "image":
                res = self.image_wm.detect_score(content, None)
                return decision(res["score"], self.config)
            if ct == "audio":
                res = self.audio_wm.detect_score(content, None)
                return decision(res["score"], self.config)
            if ct == "video":
                res = self.video_wm.detect_score(content, None)
                return decision(res["score"], self.config)
        return False