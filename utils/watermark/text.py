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

import hashlib
import json
from typing import Dict, Any, Optional
from utils.watermark.protocol import create_payload, sign_payload

ZERO_WIDTH_FALLBACK = ["\u200B", "\u200C", "\u200D", "\uFEFF"]

from utils.watermark.text_lexical import apply_watermark_logits, detection_score
from utils.watermark.detection import pack_result, decision

class TextWatermarker:
    def __init__(self, model_id: str, version: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.version = version
        self.config = config or {}
        self._zw_map = {str(i): ZERO_WIDTH_FALLBACK[i] for i in range(4)}
        self._rev_map = {v: k for k, v in self._zw_map.items()}

    def _visible_prefix(self) -> Optional[str]:
        return (self.config.get("modalities", {})
                .get("text", {})
                .get("visible", {})
                .get("prefix"))

    def embed(self, content: str, metadata: Dict[str, Any]) -> str:
        # 受控采样水印应在生成时调用 apply_watermark_logits() 实现主通道不可见标记。
        # 零宽字符通道作为兜底编码链路，避免极端情况下丢失来源指纹。
        if not content or not content.strip():
            return content

        # 生成统一的最小载荷并签名
        user_hash = hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:32]
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]
        base = create_payload(
            issuer="PiscesL1",
            model_id=self.model_id,
            content_hash=content_hash,
            tenant=metadata.get("tenant"),
            user_hash=user_hash,
            extra={"version": self.version, "type": "text"}
        )
        payload = sign_payload(base, self.config)

        # 零宽编码通道：将签名载荷序列化为比特，按定位规则插入零宽字符（兜底链路）
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
        bits = "".join(format(ord(c), "08b") for c in payload_str)

        watermarked = list(content)
        step = max(1, len(content) // max(1, len(bits)))
        idx = 0
        for b in bits:
            if idx >= len(watermarked):
                break
            if b in self._zw_map:
                watermarked.insert(idx, self._zw_map[b])
            idx += step

        # 可见前缀（法规披露）
        prefix = self._visible_prefix()
        out = "".join(watermarked)
        if prefix:
            out = f"[{prefix}]{out}"
        return out

    def extract(self, text: str) -> Optional[Dict[str, Any]]:
        # 从零宽字符兜底通道恢复编码载荷，并配合同步签名校验。
        chars = [c for c in text if c in self._rev_map]
        if not chars:
            return None
        bits = "".join(self._rev_map[c] for c in chars)
        # 以8位切片尝试还原
        by = []
        for i in range(0, len(bits), 8):
            b = bits[i:i+8]
            if len(b) == 8:
                try:
                    by.append(chr(int(b, 2)))
                except ValueError:
                    break
        if not by:
            return None
        try:
            s = "".join(by)
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # 采样器集成入口（供推理管线调用）：根据seed对logits做轻量加权
    def bias_logits(self, logits, vocab_size: int, seed: int, boost: float = 0.15):
        return apply_watermark_logits(logits, vocab_size, seed, boost)

    # 文本检测分数：需要token序列和seed（由载荷/会话派生）
    def detect_score(self, token_ids, vocab_size: int, seed: int) -> Dict[str, Any]:
        score = detection_score(token_ids, vocab_size, seed)
        return pack_result("text", score)