#!/usr/bin/env python3

# Moved from tools/watermark.py into utils/watermark/content.py
# NOTE: Future work will refactor protocols (SYNC+LEN+CRC+ECC) and add extractors for image/audio.

import re
import json
import time
import uuid
import torch
import random
import base64
import secrets
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Union, List

# Unified text framing protocol (SYNC+LEN+CRC)
from .protocol import frame_payload, extract_from_bits
from .dct import embed_bits_in_dct, extract_bits_from_dct

class PiscesWatermark:
    """
    An AI content hidden watermark system compliant with national standards.
    Supports zero-width character watermarks for text, frequency domain watermarks for images, 
    and LSB watermarks for audio.
    """
    
    def __init__(self, model_id: str = "PiscesL1-1.5B", version: str = None):
        # Initialize model ID and version
        self.model_id = model_id
        from configs.version import PVERSION
        self.version = version or PVERSION
        # Generate a unique watermark key
        self.watermark_key = self._generate_watermark_key()
        # Load watermark configuration from JSON file
        self.config = self._load_watermark_config()
        # Initialize zero-width character mapping from configuration
        self.zero_width_chars = self._get_zero_width_mapping()
        # Create reverse mapping from zero-width characters to bits
        self.char_to_bits = {v: k for k, v in self.zero_width_chars.items()}
    
    def _load_watermark_config(self) -> Dict[str, Any]:
        from configs.version import PVERSION
        try:
            with open('configs/watermark.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Update compliance_version placeholder from PVERSION
            if "watermark_2025" in config and "compliance_version" in config["watermark_2025"] and config["watermark_2025"]["compliance_version"] == "{{VERSION}}":
                config["watermark_2025"]["compliance_version"] = PVERSION
                return config
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "watermark_2025": {
                    "compliance_version": "{{VERSION}}",
                    "text": {
                        "zero_width": {
                            "force_density": 0.33,
                            "characters": ["\u200B", "\u200C", "\u200D", "\uFEFF", "\u2060", "\u2061", "\u2062", "\u2063"],
                            "layers": ["user_id", "timestamp", "model_hash", "session_id", "content_hash"],
                            "redundancy": 3
                        }
                    },
                    "image": {
                        "hidden": {
                            "dct_strength": 0.08,
                            "channels": ["Y", "U", "V"]
                        },
                        "visible": {
                            "text": "AI-Generated",
                            "position": "bottom_right",
                            "opacity": 0.8,
                            "font_size": 24
                        }
                    },
                    "audio": {
                        "hidden": {
                            "lsb_bits": 6,
                            "frequency_range": [18000, 20000]
                        }
                    }
                }
            }

    def _get_zero_width_mapping(self) -> Dict[str, str]:
        """Return 2-bit symbol mapping using first 4 zero-width characters.
        '00'->ch0, '01'->ch1, '10'->ch2, '11'->ch3
        """
        chars = self.config["watermark_2025"]["text"]["zero_width"]["characters"]
        chars4 = (chars + ["\u200B"])[:4]
        return {
            "00": chars4[0],
            "01": chars4[1],
            "10": chars4[2],
            "11": chars4[3],
        }

    def _generate_watermark_key(self) -> str:
        timestamp = str(int(time.time()))
        unique_id = str(uuid.uuid4())[:8]
        return f"{self.model_id}_{timestamp}_{unique_id}"
    
    def _create_watermark_payload(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.model_id,
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": str(uuid.uuid4()),
            "user_id": hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:16],
            "prompt_hash": hashlib.sha256(metadata.get("prompt", "").encode()).hexdigest()[:16],
            "generation_params": metadata.get("params", {}),
            "compliance": {
                "standard": "GB/T 45225-2024",
                "type": "hidden_watermark",
                "detectable": True,
                "removable": False
            }
        }
        return payload
    
    def _encode_watermark_bits(self, payload: Dict[str, Any]) -> str:
        json_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        checksum = hashlib.md5(json_str.encode()).hexdigest()[:8]
        combined = f"{json_str}|{checksum}"
        encoded = base64.b64encode(combined.encode()).decode()
        binary_str = ''.join(format(ord(c), '08b') for c in encoded)
        return binary_str

    def _encode_watermark_bits_2025(self, payload: Dict[str, Any]) -> str:
        from configs.version import PVERSION
        payload["encoding_version"] = PVERSION
        payload["encoding_time"] = int(time.time() * 1000)
        json_str = json.dumps(payload, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
        md5_hash = hashlib.md5(json_str.encode()).hexdigest()
        sha256_hash = hashlib.sha256(json_str.encode()).hexdigest()
        combined = f"{json_str}|{md5_hash}|{sha256_hash[:16]}"
        encoded = base64.b64encode(combined.encode('utf-8')).decode('ascii')
        binary_str = ''.join(format(ord(c), '08b') for c in encoded)
        redundant_binary = ""
        for i in range(0, len(binary_str), 8):
            byte = binary_str[i:i+8]
            if len(byte) == 8:
                parity = str(byte.count('1') % 2)
                redundant_binary += byte + parity
            else:
                redundant_binary += byte
        return redundant_binary
    
    def embed_text_watermark(self, content: str, metadata: Dict[str, Any]) -> str:
        if not content or not content.strip():
            return content
        text_config = self.config["watermark_2025"]["text"]
        zero_width_config = text_config["zero_width"]
        compliance_version = self.config["watermark_2025"]["compliance_version"]
        key = self._generate_watermark_key()
        payload = {
            "standard": compliance_version,
            "model": self.model_id,
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": str(uuid.uuid4()),
            "user_id": hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:32],
            "content_hash": hashlib.sha256(content.encode()).hexdigest()[:32],
            "trace_chain": str(uuid.uuid4()),
            "compliance": {
                "standard": compliance_version,
                "type": "mandatory_text_watermark",
                "detectable": True,
                "removable": False,
                "tamper_proof": zero_width_config.get("tamper_proof", True)
            }
        }
        # 1) Frame payload bits with SYNC+LEN+CRC
        framed_bits = frame_payload(payload)
        # 2) Map bits to 2-bit symbols and corresponding zero-width chars
        def bits_to_zwc(bitstream: str) -> List[str]:
            # pad to even length
            if len(bitstream) % 2 == 1:
                bitstream += '0'
            symbols = [bitstream[i:i+2] for i in range(0, len(bitstream), 2)]
            return [self.zero_width_chars[s] for s in symbols]

        zwc_stream = bits_to_zwc(framed_bits)
        redundancy = zero_width_config.get("redundancy", 3)
        watermarked_content = content

        # 3) Pseudo-random distribute zero-width chars across layers and positions
        idx = 0
        for layer in range(redundancy):
            positions: List[int] = []
            for i, _ch in enumerate(content):
                # sparse selection keyed by (key, layer, i)
                if (hashlib.md5(f"{key}_{layer}_{i}".encode()).hexdigest())[:2] == '00':
                    positions.append(i)
            temp = list(watermarked_content)
            for pos in sorted(positions, reverse=True):
                if idx < len(zwc_stream):
                    temp.insert(pos, zwc_stream[idx])
                    idx += 1
            watermarked_content = ''.join(temp)
            if idx >= len(zwc_stream):
                break
        return watermarked_content
    
    def extract_text_watermark(self, text: str) -> Optional[Dict[str, Any]]:
        """Collect zero-width chars, map back to 2-bit symbols, deframe with protocol."""
        # Build reverse map: char -> 2-bit symbol
        rev_map: Dict[str, str] = {v: k for k, v in self.zero_width_chars.items()}
        symbols: List[str] = []
        for ch in text:
            sym = rev_map.get(ch)
            if sym is not None:
                symbols.append(sym)
        if not symbols:
            return None
        bitstream = ''.join(symbols)  # each symbol is 2 bits
        return extract_from_bits(bitstream)
    
    @staticmethod
    def _ecc_repeat3_encode(bits: str) -> str:
        return ''.join(ch * 3 for ch in bits)

    @staticmethod
    def _ecc_repeat3_decode(bits: str) -> str:
        if len(bits) < 3:
            return ''
        out = []
        for i in range(0, len(bits) - 2, 3):
            tri = bits[i:i+3]
            if tri.count('1') >= 2:
                out.append('1')
            elif tri.count('0') >= 2:
                out.append('0')
        return ''.join(out)

    def embed_image_watermark(self, image_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        import torch.fft as fft
        image_config = self.config["watermark_2025"]["image"]
        compliance_version = self.config["watermark_2025"]["compliance_version"]
        hidden = image_config["hidden"]
        visible = image_config["visible"]

        # Build payload and frame bits
        payload = {
            "standard": compliance_version,
            "model": self.model_id,
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": str(uuid.uuid4()),
            "user_id": hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:32],
            "content_hash": hashlib.sha256(str(image_tensor.shape).encode()).hexdigest()[:32],
            "trace_chain": str(uuid.uuid4()),
            "type": "image_fft_pairs",
        }
        bits = frame_payload(payload)
        # ECC for image if configured
        if str(hidden.get("ecc", "")).lower() == "repeat3":
            bits = self._ecc_repeat3_encode(bits)

        c, h, w = image_tensor.shape
        strength = float(hidden.get("dct_strength", 0.05))
        block_band = str(hidden.get("embed_band", "mid")).lower()
        key = str(hidden.get("key", "pisces2025"))

        rng = int(hashlib.sha256((key + str(h) + "x" + str(w)).encode()).hexdigest(), 16)
        torch.manual_seed(rng & 0x7FFFFFFF)

        # Define mid-band box
        ih0, ih1 = h // 4, 3 * h // 4
        iw0, iw1 = w // 4, 3 * w // 4
        if block_band == "low":
            ih0, ih1 = h // 6, h // 2
            iw0, iw1 = w // 6, w // 2
        elif block_band == "high":
            ih0, ih1 = h // 2, 5 * h // 6
            iw0, iw1 = w // 2, 5 * w // 6

        out = image_tensor.clone()
        sym_pairs_per_channel = max(1, (len(bits) // max(1, c)) + 8)
        bit_idx = 0

        for ch in range(c):
            F = fft.fft2(image_tensor[ch])
            # Embed by enforcing pair-wise magnitude ordering to represent bits
            count = 0
            for i in range(ih0, ih1 - 1):
                if count >= sym_pairs_per_channel or bit_idx >= len(bits):
                    break
                for j in range(iw0, iw1 - 1):
                    if count >= sym_pairs_per_channel or bit_idx >= len(bits):
                        break
                    b = 1 if bits[bit_idx] == '1' else 0
                    a = F[i, j]
                    b2 = F[i + 1, j + 1]
                    mag_a = torch.abs(a)
                    mag_b = torch.abs(b2)
                    # target relation: bit 1 -> mag_a > mag_b, bit 0 -> mag_b > mag_a
                    delta = strength * (0.1 + 0.9 * torch.rand(1, device=mag_a.device))
                    if b == 1:
                        if mag_a <= mag_b + delta:
                            scale = (mag_b + delta + 1e-8) / (mag_a + 1e-8)
                            F[i, j] = a * scale
                    else:
                        if mag_b <= mag_a + delta:
                            scale = (mag_a + delta + 1e-8) / (mag_b + 1e-8)
                            F[i + 1, j + 1] = b2 * scale
                    bit_idx += 1
                    count += 1
            out[ch] = torch.real(fft.ifft2(F))

        # Optional visible overlay
        watermarked_image = self._add_visible_watermark(out) if visible.get("force_enabled", True) else out
        return watermarked_image
    
    def extract_image_watermark(self, image_tensor: torch.Tensor) -> Optional[Dict[str, Any]]:
        import torch
        import torch.fft as fft  # fallback use
        hidden = self.config["watermark_2025"]["image"]["hidden"]
        c, h, w = image_tensor.shape
        band = str(hidden.get("embed_band", "mid")).lower()
        block = int(hidden.get("block_size", 8))
        key = str(hidden.get("key", "pisces2025"))

        # try DCT-based extraction
        try:
            # Estimate an upper bound for bit count using number of blocks per channel
            blocks_per_ch = (h // block) * (w // block)
            collected: List[str] = []
            for ch in range(c):
                seed = int(hashlib.sha256((f"{key}:{ch}:{h}x{w}").encode()).hexdigest(), 16) & 0x7FFFFFFF
                bits_ch = extract_bits_from_dct(image_tensor[ch], blocks_per_ch * 2, block, band, seed)
                collected.append(bits_ch)
            bitstream = ''.join(collected)
        except Exception:
            # Fallback: FFT pair relation sweep (as before)
            ih0, ih1 = h // 4, 3 * h // 4
            iw0, iw1 = w // 4, 3 * w // 4
            if band == "low":
                ih0, ih1 = h // 6, h // 2
                iw0, iw1 = w // 6, w // 2
            elif band == "high":
                ih0, ih1 = h // 2, 5 * h // 6
                iw0, iw1 = w // 2, 5 * w // 6
            symbols: List[str] = []
            per_ch_limit = (h * w) // 64
            for ch in range(c):
                F = fft.fft2(image_tensor[ch])
                count = 0
                for i in range(ih0, ih1 - 1):
                    if count >= per_ch_limit:
                        break
                    for j in range(iw0, iw1 - 1):
                        if count >= per_ch_limit:
                            break
                        a = torch.abs(F[i, j])
                        b2 = torch.abs(F[i + 1, j + 1])
                        symbols.append('1' if a > b2 else '0')
                        count += 1
            bitstream = ''.join(symbols)

        # ECC decode if configured
        if str(hidden.get("ecc", "")).lower() == "repeat3":
            bitstream = self._ecc_repeat3_decode(bitstream)
        return extract_from_bits(bitstream)
    
    def _add_visible_watermark(self, image_tensor: torch.Tensor) -> torch.Tensor:
        import torch
        c, h, w = image_tensor.shape
        image_config = self.config["watermark_2025"]["image"]
        visible_config = image_config["visible"]
        watermark_text = visible_config.get("text", "AI-generated-PiscesL1")
        text_height = visible_config.get("text_height", 24)
        text_width = len(watermark_text) * visible_config.get("char_width", 12)
        position = visible_config.get("position", "bottom_right")
        margin = visible_config.get("margin", 10)
        if position == "bottom_right":
            start_y = max(0, h - text_height - margin)
            start_x = max(0, w - text_width - margin)
        elif position == "bottom_left":
            start_y = max(0, h - text_height - margin)
            start_x = margin
        elif position == "top_right":
            start_y = margin
            start_x = max(0, w - text_width - margin)
        elif position == "top_left":
            start_y = margin
            start_x = margin
        else:
            start_y = max(0, h - text_height - margin)
            start_x = max(0, w - text_width - margin)
        watermarked = image_tensor.clone()
        opacity = visible_config.get("opacity", 0.1)
        background_color = visible_config.get("background_color", [1.0, 1.0, 1.0])
        for y in range(start_y, min(h, start_y + text_height)):
            for x in range(start_x, min(w, start_x + text_width)):
                if 0 <= y < h and 0 <= x < w:
                    for channel in range(min(c, len(background_color))):
                        watermarked[channel, y, x] = (
                            (1 - opacity) * watermarked[channel, y, x] + 
                            opacity * background_color[channel]
                        )
        return watermarked
    
    def embed_audio_watermark(self, audio_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        import torch
        audio_cfg = self.config["watermark_2025"]["audio"]
        compliance_version = self.config["watermark_2025"]["compliance_version"]
        hidden = audio_cfg["hidden"]
        stft_cfg = hidden.get("stft", {})
        sr = int(stft_cfg.get("sample_rate", 44100))
        min_f = float(stft_cfg.get("min_freq", 18000))
        max_f = float(stft_cfg.get("max_freq", 20000))
        amp = float(stft_cfg.get("amplitude", 0.01))
        spread = int(stft_cfg.get("spread_factor", 10))
        key = str(hidden.get("key", "pisces2025_audio"))

        # Payload -> framed bits
        payload = {
            "standard": compliance_version,
            "model": self.model_id,
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": str(uuid.uuid4()),
            "user_id": hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:32],
            "content_hash": hashlib.sha256(str(audio_tensor.shape).encode()).hexdigest()[:32],
            "trace_chain": str(uuid.uuid4()),
            "type": "audio_stft_dsss",
        }
        bits = frame_payload(payload)

        x = audio_tensor.flatten()
        n_fft = 1024
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)

        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(x.device)
        band_mask = (freqs >= min_f) & (freqs <= max_f)
        idx_band = torch.where(band_mask)[0]
        if idx_band.numel() == 0:
            return audio_tensor

        # PN sequence from key
        seed = int(hashlib.sha256((key).encode()).hexdigest(), 16) & 0x7FFFFFFF
        torch.manual_seed(seed)
        pn = torch.sign(torch.rand(spread, device=x.device) - 0.5)  # +/-1 chips

        bit_idx = 0
        for t in range(X.shape[1]):
            if bit_idx >= len(bits):
                break
            b = 1 if bits[bit_idx] == '1' else -1
            # spread across band bins using PN chips
            chips = pn[:min(spread, idx_band.numel())] * b
            X[idx_band[:chips.numel()], t] = X[idx_band[:chips.numel()], t] * (1.0 + amp * chips)
            if (t % 2) == 1:  # advance bit every 2 frames (can tune)
                bit_idx += 1

        y = torch.istft(X, n_fft=n_fft, hop_length=hop, window=win, length=x.numel())
        return y.view(audio_tensor.shape)
    
    def extract_audio_watermark(self, audio_tensor: torch.Tensor, sample_rate: int = 44100) -> Optional[Dict[str, Any]]:
        import torch
        hidden = self.config["watermark_2025"]["audio"]["hidden"]
        stft_cfg = hidden.get("stft", {})
        sr = int(stft_cfg.get("sample_rate", sample_rate))
        min_f = float(stft_cfg.get("min_freq", 18000))
        max_f = float(stft_cfg.get("max_freq", 20000))
        spread = int(stft_cfg.get("spread_factor", 10))
        key = str(hidden.get("key", "pisces2025_audio"))

        x = audio_tensor.flatten()
        n_fft = 1024
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)

        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(x.device)
        band_mask = (freqs >= min_f) & (freqs <= max_f)
        idx_band = torch.where(band_mask)[0]
        if idx_band.numel() == 0:
            return None

        seed = int(hashlib.sha256((key).encode()).hexdigest(), 16) & 0x7FFFFFFF
        torch.manual_seed(seed)
        pn = torch.sign(torch.rand(spread, device=x.device) - 0.5)

        # Correlate per 2 frames to vote on bit
        votes: List[str] = []
        for t in range(0, X.shape[1] - 1, 2):
            mag = torch.abs(X[idx_band, t])
            chips = pn[:min(spread, idx_band.numel())]
            # Simple projection
            corr = (mag[:chips.numel()] * chips).sum()
            votes.append('1' if corr > 0 else '0')
        bitstream = ''.join(votes)
        # Majority vote decode for audio repeat3
        bitstream = self._ecc_repeat3_decode(bitstream)
        return extract_from_bits(bitstream)
    
    def _add_spectrum_watermark(self, audio: torch.Tensor, payload: str, spectrum_config: Dict[str, Any] = None) -> torch.Tensor:
        import torch.fft as fft
        if spectrum_config is None:
            spectrum_config = {}
        sample_rate = spectrum_config.get("sample_rate", 44100)
        min_freq = spectrum_config.get("min_freq", 18000)
        max_freq = spectrum_config.get("max_freq", 20000)
        amplitude = spectrum_config.get("amplitude", 0.01)
        spread_factor = spectrum_config.get("spread_factor", 10)
        spectrum = fft.fft(audio)
        freq_bins = torch.fft.fftfreq(len(audio), 1/sample_rate)
        target_freqs = (freq_bins >= min_freq) & (freq_bins <= max_freq)
        if torch.sum(target_freqs) > 0:
            watermark_bits = ''.join(format(ord(c), '08b') for c in payload)
            target_indices = torch.where(target_freqs)[0]
            for i, idx in enumerate(target_indices[:len(watermark_bits)]):
                bit = int(watermark_bits[i % len(watermark_bits)])
                for offset in range(spread_factor):
                    if idx + offset < len(spectrum):
                        spectrum[idx + offset] += bit * amplitude * (1 - offset/spread_factor)
        return torch.real(fft.ifft(spectrum))
    
    def _add_voice_watermark(self, audio: torch.Tensor) -> torch.Tensor:
        import torch
        sample_rate = 44100
        watermark_duration = 2.0
        watermark_samples = int(watermark_duration * sample_rate)
        if len(audio) < watermark_samples:
            return audio
        t = torch.linspace(0, watermark_duration, watermark_samples)
        base_freq = 220
        voice_signal = (
            0.1 * torch.sin(2 * torch.pi * base_freq * t) +
            0.05 * torch.sin(2 * torch.pi * base_freq * 2 * t) +
            0.03 * torch.sin(2 * torch.pi * base_freq * 3 * t)
        )
        envelope = torch.ones_like(t)
        fade_samples = int(0.1 * sample_rate)
        envelope[:fade_samples] = torch.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = torch.linspace(1, 0, fade_samples)
        voice_signal *= envelope
        watermarked = audio.clone()
        watermarked[:watermark_samples] += voice_signal
        return watermarked
    
    def verify_watermark(self, content: Union[str, torch.Tensor], expected_metadata: Dict[str, Any]) -> bool:
        if isinstance(content, str):
            extracted = self.extract_text_watermark(content)
        elif isinstance(content, torch.Tensor):
            extracted = None
        else:
            return False
        if not extracted:
            return False
        return (
            extracted.get("model") == self.model_id and
            extracted.get("compliance", {}).get("standard") == "GB/T 45225-2024"
        )

class WatermarkManager:
    """Watermark manager integrated into the inference process."""
    
    def __init__(self, model_id: str = "PiscesL1-1.5B"):
        self.watermark = PiscesWatermark(model_id)
        self.enabled = True
        self.force_enabled = True  
    
    def add_watermark(self, content: Union[str, torch.Tensor], metadata: Dict[str, Any]) -> Union[str, torch.Tensor]:
        if content is None:
            return content
        enhanced_metadata = {
            **metadata,
            "user_id": metadata.get("user_id", "anonymous"),
            "ip": metadata.get("ip", "unknown"),
            "device": metadata.get("device", "unknown"),
            "session_start": metadata.get("session_start", datetime.utcnow().isoformat()),
            "generation_count": metadata.get("generation_count", 1),
            "content_type": self._detect_content_type(content)
        }
        try:
            if isinstance(content, str):
                return self.watermark.embed_text_watermark(content, enhanced_metadata)
            elif isinstance(content, torch.Tensor):
                content_type = self._detect_content_type(content)
                if content_type == "image":
                    return self.watermark.embed_image_watermark(content, enhanced_metadata)
                elif content_type == "audio":
                    return self.watermark.embed_audio_watermark(content, enhanced_metadata)
                elif content_type == "video":
                    return self._embed_video_watermark(content, enhanced_metadata)
        except Exception:
            if isinstance(content, str):
                return f"[AI生成]{content}"
            return content
        return content
    
    def _detect_content_type(self, content: Union[str, torch.Tensor]) -> str:
        if isinstance(content, str):
            return "text"
        elif isinstance(content, torch.Tensor):
            shape = content.shape
            if len(shape) == 3 and shape[0] in [1, 3, 4]:
                return "image"
            elif len(shape) == 1 or (len(shape) == 2 and shape[0] == 1):
                return "audio"
            elif len(shape) == 4:
                return "video"
        return "unknown"
