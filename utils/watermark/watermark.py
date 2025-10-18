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

import json
import hashlib
import torch
from typing import Dict, Any, Optional, Union
from utils.watermark.protocol import verify_payload
from utils.watermark.manager import PiscesWatermarkManager

class PiscesLxWatermark:
    """
    Backward-compatible facade. Internally delegates to modular watermarkers.
    """
    def __init__(self, model_id: str = "PiscesL1-1.5B", version: str = "1.0.0"):
        self.model_id = model_id
        self.version = version
        try:
            with open('configs/watermark.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except Exception:
            self.config = {}
        self._mgr = PiscesWatermarkManager(model_id=model_id, version=version, config=self.config)

    # Text
    def embed_text_watermark(self, content: str, metadata: Dict[str, Any]) -> str:
        return self._mgr.add_watermark(content, metadata)

    def extract_text_watermark(self, text: str) -> Optional[Dict[str, Any]]:
        return self._mgr.check_watermark(text)

    # Image
    def embed_image_watermark(self, image_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        return self._mgr.add_watermark(image_tensor, metadata)

    # Audio
    def embed_audio_watermark(self, audio_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        return self._mgr.add_watermark(audio_tensor, metadata)

    def verify_watermark(self, content: Union[str, torch.Tensor], expected_metadata: Dict[str, Any] = None) -> bool:
        if isinstance(content, str):
            payload = self.extract_text_watermark(content)
            return bool(payload) and verify_payload(payload, self.config) and payload.get("model_id") == self.model_id
        return True

class PiscesLxWatermarkManager(PiscesWatermarkManager):
    """
    Backward-compatible alias for previous manager class name.
    """
    pass