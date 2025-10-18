from typing import Dict, Any
import json
import hashlib
import torch

from utils.watermark.protocol import create_payload, sign_payload
from utils.watermark.detection import pack_result

class VideoWatermarker:
    def __init__(self, model_id: str, version: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.version = version
        self.config = config or {}

    def _video_cfg(self) -> Dict[str, Any]:
        return (self.config.get("modalities", {})
                .get("video", {}) or {})

    def _visible_cfg(self) -> Dict[str, Any]:
        return self._video_cfg().get("visible_overlay", {}) or {}

    def embed(self, video_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        video_tensor shape: [C, T, H, W]
        不可见：逐帧中频域嵌入，位串冗余与时间扩展（每n帧重复载荷片段）
        可见：角标覆盖
        """
        import torch.fft as fft
        assert len(video_tensor.shape) == 4, "video tensor must be [C, T, H, W]"
        c, t, h, w = video_tensor.shape

        vcfg = self._video_cfg()
        interval = int(vcfg.get("frame_interval", 5))
        kboost = float(vcfg.get("keyframe_boost", 2.0))
        visible_cfg = self._visible_cfg()

        user_hash = hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:32]
        content_hash = hashlib.sha256(str(video_tensor.shape).encode()).hexdigest()[:32]
        base = create_payload(
            issuer="PiscesL1",
            model_id=self.model_id,
            content_hash=content_hash,
            tenant=metadata.get("tenant"),
            user_hash=user_hash,
            extra={"version": self.version, "type": "video"}
        )
        payload = sign_payload(base, self.config)
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        bits_str = "".join(format(ord(ch), "08b") for ch in payload_str)
        redundancy = int(vcfg.get("redundancy", 3))
        bits = list("".join(b * max(1, redundancy) for b in bits_str))

        out = video_tensor.clone()
        bit_idx = 0
        for frame_idx in range(t):
            # 中频嵌入
            for ch in range(min(c, 3)):
                F = fft.fft2(out[ch, frame_idx])
                hs, he = h // 4, 3 * h // 4
                ws, we = w // 4, 3 * w // 4
                strength = 0.06 * (kboost if frame_idx % interval == 0 else 1.0)
                for i in range(hs, he, 2):
                    for j in range(ws, we, 2):
                        if bit_idx >= len(bits):
                            bit_idx = 0  # 时间扩展循环使用
                        b = int(bits[bit_idx])
                        F[i, j] *= (1 + b * strength)
                        bit_idx += 1
                out[ch, frame_idx] = torch.real(fft.ifft2(F))

            # 可见角标
            text = visible_cfg.get("text", "AI-Generated")
            opacity = float(visible_cfg.get("opacity", 0.7))
            margin = 10
            char_w = 12
            text_h = 24
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

            bg = [1.0, 1.0, 1.0]
            for y in range(sy, min(h, sy + text_h)):
                for x in range(sx, min(w, sx + text_w)):
                    for ch in range(min(c, len(bg))):
                        out[ch, frame_idx, y, x] = (1 - opacity) * out[ch, frame_idx, y, x] + opacity * float(bg[ch])
        return out

    def detect_score(self, video_tensor: torch.Tensor, payload_str: str = None) -> Dict[str, Any]:
        """
        检测：对关键帧提取中频响应，与比特序列（或model_id派生模式）相关性均值→分数
        """
        import torch.fft as fft
        assert len(video_tensor.shape) == 4, "video tensor must be [C, T, H, W]"
        c, t, h, w = video_tensor.shape
        vcfg = self._video_cfg()
        interval = int(vcfg.get("frame_interval", 5))

        if not payload_str:
            # 缺省模式：用model_id派生伪比特模式，保证无需外部信息也能打分
            seed = int(hashlib.sha256(self.model_id.encode()).hexdigest()[:8], 16)
            pseudo = "".join("1" if ((i * 131 + seed) % 2) else "0" for i in range(1024))
            bits_str = pseudo
        else:
            bits_str = "".join(format(ord(ch), "08b") for ch in payload_str)

        if len(bits_str) == 0:
            return pack_result("video", 0.0)
        import torch
        bits = torch.tensor([int(b) for b in bits_str], dtype=torch.float32)

        scores = []
        for frame_idx in range(0, t, max(1, interval)):
            F = fft.fft2(video_tensor[0, frame_idx])
            hs, he = h // 4, 3 * h // 4
            ws, we = w // 4, 3 * w // 4
            region = torch.real(F[hs:he:2, ws:we:2]).flatten()
            m = min(region.numel(), bits.numel())
            if m > 0:
                s = float(torch.sigmoid((region[:m] * (bits[:m] * 2 - 1)).mean()))
                scores.append(s)
        score = float(sum(scores) / len(scores)) if scores else 0.0
        return pack_result("video", score)