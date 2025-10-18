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

# 音频不可见水印：高频带扩频注入，载荷为签名JSON，支持位串冗余与检测评分。

class AudioWatermarker:
    def __init__(self, model_id: str, version: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.version = version
        self.config = config or {}

    def _audio_cfg(self) -> Dict[str, Any]:
        return (self.config.get("modalities", {})
                .get("audio", {}))

    def embed(self, audio_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        import torch.fft as fft

        audio_cfg = self._audio_cfg()
        wm_cfg = audio_cfg.get("watermarking", {}) or {}

        user_hash = hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:32]
        content_hash = hashlib.sha256(str(audio_tensor.shape).encode()).hexdigest()[:32]
        base = create_payload(
            issuer="PiscesL1",
            model_id=self.model_id,
            content_hash=content_hash,
            tenant=metadata.get("tenant"),
            user_hash=user_hash,
            extra={"version": self.version, "type": "audio"}
        )
        payload = sign_payload(base, self.config)
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)

        if len(audio_tensor.shape) > 1:
            audio = audio_tensor.view(-1).clone()
        else:
            audio = audio_tensor.clone()

        # 频域扩频注入（占位）
        sr = int(wm_cfg.get("sample_rate", 44100))
        min_f, max_f = wm_cfg.get("frequency_range", [18000, 20000])
        amp = float(wm_cfg.get("amplitude", 0.01))
        spread = int(wm_cfg.get("spread_factor", 10))

        spec = fft.fft(audio)
        freqs = torch.fft.fftfreq(len(audio), 1 / sr)
        mask = (freqs >= min_f) & (freqs <= max_f)
        bits_str = "".join(format(ord(ch), "08b") for ch in payload_str)
        redundancy = int(wm_cfg.get("redundancy", 3))
        bits = "".join(b * max(1, redundancy) for b in bits_str)
        bidx = 0
        idxs = torch.where(mask)[0]
        for k in idxs:
            if bidx >= len(bits):
                break
            bit = int(bits[bidx])
            for off in range(spread):
                if k + off < len(spec):
                    spec[k + off] += bit * amp * (1 - off / spread)
            bidx += 1
        out = torch.real(fft.ifft(spec)).view(audio_tensor.shape)
        return out

    def detect_score(self, audio_tensor: torch.Tensor, payload_str: str = None) -> Dict[str, Any]:
        # 频域目标带与载荷比特相关性评分（占位简化版）
        import torch.fft as fft
        audio_cfg = self._audio_cfg()
        wm_cfg = audio_cfg.get("watermarking", {}) or {}
        sr = int(wm_cfg.get("sample_rate", 44100))
        min_f, max_f = wm_cfg.get("frequency_range", [18000, 20000])

        if payload_str:
            bits_str = "".join(format(ord(c), "08b") for c in payload_str)
        else:
            import hashlib
            seed = int(hashlib.sha256(self.model_id.encode()).hexdigest()[:8], 16)
            bits_str = "".join("1" if ((i * 197 + seed) % 2) else "0" for i in range(4096))
        if len(bits_str) == 0:
            return pack_result("audio", 0.0)
        bits = torch.tensor([int(b) for b in bits_str], dtype=torch.float32)

        if len(audio_tensor.shape) > 1:
            audio = audio_tensor.view(-1)
        else:
            audio = audio_tensor

        spec = fft.fft(audio)
        freqs = torch.fft.fftfreq(len(audio), 1 / sr)
        mask = (freqs >= min_f) & (freqs <= max_f)
        region = torch.real(spec[mask]).abs().flatten()
        m = min(region.numel(), bits.numel())
        if m == 0:
            score = 0.0
        else:
            score = float(torch.sigmoid((region[:m] * (bits[:m] * 2 - 1)).mean()))
        return pack_result("audio", score)