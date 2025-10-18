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

from typing import Dict, Any
import json
import hashlib
import torch

from utils.watermark.protocol import create_payload, sign_payload
from utils.watermark.detection import pack_result

# 图像不可见水印：中频域嵌入，支持位串冗余与检测评分（可配置强度与通道）。

class ImageWatermarker:
    def __init__(self, model_id: str, version: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.version = version
        self.config = config or {}

    def _image_cfg(self) -> Dict[str, Any]:
        return (self.config.get("modalities", {})
                .get("image", {}))

    def _visible_cfg(self) -> Dict[str, Any]:
        return self._image_cfg().get("visible", {}) or {}

    def embed(self, image_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        import torch.fft as fft

        image_cfg = self._image_cfg()
        wm_cfg = image_cfg.get("watermarking", {}) or {}
        visible_cfg = self._visible_cfg()

        user_hash = hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:32]
        content_hash = hashlib.sha256(str(image_tensor.shape).encode()).hexdigest()[:32]
        base = create_payload(
            issuer="PiscesL1",
            model_id=self.model_id,
            content_hash=content_hash,
            tenant=metadata.get("tenant"),
            user_hash=user_hash,
            extra={"version": self.version, "type": "image"}
        )
        payload = sign_payload(base, self.config)
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)

        # 中频区域乘法嵌入（含位串冗余）
        c, h, w = image_tensor.shape
        dct_strength = wm_cfg.get("dct_strength", 0.08)
        bits_str = "".join(format(ord(ch), "08b") for ch in payload_str)
        redundancy = int(wm_cfg.get("redundancy", 3))
        bits = list("".join(b * max(1, redundancy) for b in bits_str))
        bit_idx = 0

        out = image_tensor.clone()
        for ch in range(min(c, 3)):
            F = fft.fft2(out[ch])
            hs, he = h // 4, 3 * h // 4
            ws, we = w // 4, 3 * w // 4
            for i in range(hs, he, 2):
                for j in range(ws, we, 2):
                    if bit_idx >= len(bits):
                        break
                    b = int(bits[bit_idx])
                    F[i, j] *= (1 + b * dct_strength)
                    bit_idx += 1
                if bit_idx >= len(bits):
                    break
            out[ch] = torch.real(fft.ifft2(F))

        # 可见角标叠加
        text = visible_cfg.get("text", "AI-Generated")
        opacity = float(visible_cfg.get("opacity", 0.1))
        margin = int(visible_cfg.get("margin", 10))
        char_w = int(visible_cfg.get("char_width", 12))
        text_h = int(visible_cfg.get("font_size", 24))
        text_w = len(text) * char_w
        pos = visible_cfg.get("position", "bottom_right")

        if pos == "bottom_left":
            sy, sx = max(0, h - text_h - margin), margin
        elif pos == "top_right":
            sy, sx = margin, max(0, w - text_w - margin)
        elif pos == "top_left":
            sy, sx = margin, margin
        else:
            sy, sx = max(0, h - text_h - margin), max(0, w - text_w - margin)

        bg = visible_cfg.get("color_rgb", [1.0, 1.0, 1.0])
        for y in range(sy, min(h, sy + text_h)):
            for x in range(sx, min(w, sx + text_w)):
                for ch in range(min(c, len(bg))):
                    out[ch, y, x] = (1 - opacity) * out[ch, y, x] + opacity * float(bg[ch])
        return out

    def detect_score(self, image_tensor: torch.Tensor, payload_str: str = None) -> Dict[str, Any]:
        # 简化相关性评分：取中频区域与载荷比特的响应均值，映射到[0,1]
        import torch.fft as fft
        if payload_str:
            bits_str = "".join(format(ord(c), "08b") for c in payload_str)
        else:
            import hashlib
            seed = int(hashlib.sha256(self.model_id.encode()).hexdigest()[:8], 16)
            bits_str = "".join("1" if ((i * 131 + seed) % 2) else "0" for i in range(4096))
        if len(bits_str) == 0:
            return pack_result("image", 0.0)
        bits = torch.tensor([int(b) for b in bits_str], dtype=torch.float32)
        c, h, w = image_tensor.shape
        F = fft.fft2(image_tensor[0])
        hs, he = h // 4, 3 * h // 4
        ws, we = w // 4, 3 * w // 4
        region = torch.real(F[hs:he:2, ws:we:2]).flatten()
        m = min(region.numel(), bits.numel())
        if m == 0:
            score = 0.0
        else:
            score = float(torch.sigmoid((region[:m] * (bits[:m] * 2 - 1)).mean()))
        return pack_result("image", score)