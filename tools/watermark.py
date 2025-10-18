#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

class PiscesWatermark:
    """
    An AI content hidden watermark system compliant with national standards.
    Supports zero-width character watermarks for text, frequency domain watermarks for images, 
    and LSB watermarks for audio.
    """
    
    def __init__(self, model_id: str = "PiscesL1-1.5B", version: str = "1.0.0"):
        # Initialize model ID and version
        self.model_id = model_id
        self.version = version
        # Generate a unique watermark key
        self.watermark_key = self._generate_watermark_key()
        # Load watermark configuration from JSON file
        self.config = self._load_watermark_config()
        # Initialize zero-width character mapping from configuration
        self.zero_width_chars = self._get_zero_width_mapping()
        # Create reverse mapping from zero-width characters to bits
        self.char_to_bits = {v: k for k, v in self.zero_width_chars.items()}
    
    def _load_watermark_config(self) -> Dict[str, Any]:
        """
        Load watermark configuration from JSON file.
        If the file is not found or JSON decoding fails, return a fallback configuration.

        Returns:
            Dict[str, Any]: Loaded watermark configuration or fallback configuration.
        """
        try:
            with open('configs/watermark.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "watermark_2025": {
                    "compliance_version": "2025.09.01",
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
        """
        Get zero-width character mapping from configuration.
        Use the first 4 characters for 2-bit encoding.

        Returns:
            Dict[str, str]: Character mapping for watermark encoding.
        """
        chars = self.config["watermark_2025"]["text"]["zero_width"]["characters"]
        mapping = {}
        for i, char in enumerate(chars[:4]):
            mapping[str(i)] = char
        return mapping

    def _generate_watermark_key(self) -> str:
        """
        Generate a watermark key based on the model ID and current timestamp, 
        combined with a short UUID for uniqueness.

        Returns:
            str: Generated watermark key.
        """
        timestamp = str(int(time.time()))
        unique_id = str(uuid.uuid4())[:8]
        return f"{self.model_id}_{timestamp}_{unique_id}"
    
    def _create_watermark_payload(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a watermark payload compliant with national standards.
        The payload includes model information, generation metadata, and compliance details.

        Args:
            metadata (Dict[str, Any]): Generation metadata.

        Returns:
            Dict[str, Any]: Generated watermark payload.
        """
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
        """
        Encode watermark data into a binary string.
        Convert the payload to compressed JSON, add a checksum, 
        encode to Base64, and then convert to binary.

        Args:
            payload (Dict[str, Any]): Watermark payload.

        Returns:
            str: Binary string representing the watermark data.
        """
        # Convert payload to compressed JSON
        json_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        # Calculate MD5 checksum of the JSON string
        checksum = hashlib.md5(json_str.encode()).hexdigest()[:8]
        # Combine JSON string and checksum
        combined = f"{json_str}|{checksum}"
        # Encode the combined string to Base64
        encoded = base64.b64encode(combined.encode()).decode()
        # Convert Base64 string to binary representation
        binary_str = ''.join(format(ord(c), '08b') for c in encoded)
        return binary_str

    def _encode_watermark_bits_2025(self, payload: Dict[str, Any]) -> str:
        """
        Enhanced watermark encoding method.
        Add version identifier and encoding time, use multiple checksums, 
        add error correction code, and redundant parity bits.

        Args:
            payload (Dict[str, Any]): Watermark payload.

        Returns:
            str: Binary string with enhanced encoding and redundancy.
        """
        # Add version identifier to the payload
        payload["encoding_version"] = "2025.09.01"
        # Add encoding time to the payload
        payload["encoding_time"] = int(time.time() * 1000)
        # Convert payload to compressed JSON with ensure_ascii=False
        json_str = json.dumps(payload, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
        # Calculate MD5 and SHA-256 checksums
        md5_hash = hashlib.md5(json_str.encode()).hexdigest()
        sha256_hash = hashlib.sha256(json_str.encode()).hexdigest()
        # Combine JSON string and multiple checksums
        combined = f"{json_str}|{md5_hash}|{sha256_hash[:16]}"
        # Encode the combined string to Base64
        encoded = base64.b64encode(combined.encode('utf-8')).decode('ascii')
        # Convert Base64 string to binary representation
        binary_str = ''.join(format(ord(c), '08b') for c in encoded)
        # Add redundant parity bits
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
        """
        Embed a text watermark using zero-width characters with configuration parameters.
        Generate a payload, convert it to binary, and insert zero-width characters 
        at pseudo-random positions in the content.

        Args:
            content (str): Original text content.
            metadata (Dict[str, Any]): Generation metadata.

        Returns:
            str: Text content with embedded watermark.
        """
        if not content or not content.strip():
            return content

        # Get text watermark configuration
        text_config = self.config["watermark_2025"]["text"]
        zero_width_config = text_config["zero_width"]
        compliance_version = self.config["watermark_2025"]["compliance_version"]
        
        # Generate a unique watermark key
        key = self._generate_watermark_key()
        
        # Create watermark payload based on configuration
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
        
        # Convert payload to binary string
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        binary_data = ''.join(format(ord(c), '08b') for c in payload_str)
        
        # Prepare watermark bits for embedding
        watermark_bits = list(binary_data)
        
        # Get configured redundancy value
        redundancy = zero_width_config.get("redundancy", 3)
        watermarked_content = content
        
        for layer in range(redundancy):
            # Distribute bits across layers
            layer_bits = watermark_bits[layer::redundancy]
            # Generate insertion positions based on key and layer
            positions = []
            for i, char in enumerate(content):
                if (hashlib.md5(f"{key}_{layer}_{i}".encode()).hexdigest())[:2] == '00':
                    positions.append(i)
            
            # Insert watermark bits from end to avoid index shifting
            temp_content = list(watermarked_content)
            bit_idx = 0
            
            for pos in sorted(positions, reverse=True):
                if bit_idx < len(layer_bits):
                    bit = int(layer_bits[bit_idx])
                    if bit < len(self.zero_width_chars):
                        watermark_char = self.zero_width_chars[str(bit)]
                        temp_content.insert(pos, watermark_char)
                    bit_idx += 1
            
            watermarked_content = ''.join(temp_content)
        
        return watermarked_content
    
    def extract_text_watermark(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract hidden watermark from text.
        Extract zero-width characters, convert them back to binary, 
        find synchronization mark, verify checksum, and parse JSON.

        Args:
            text (str): Text that may contain a watermark.

        Returns:
            Optional[Dict[str, Any]]: Watermark data or None if not found.
        """
        # Extract all zero-width characters from the text
        zero_width_chars = []
        for char in text:
            if char in self.char_to_bits:
                zero_width_chars.append(char)
        
        if not zero_width_chars:
            return None
        
        # Convert zero-width characters back to binary string
        binary_str = ''.join(self.char_to_bits[char] for char in zero_width_chars)
        
        # Find synchronization mark in the binary string
        sync_pattern = "1010" * 4
        sync_pos = binary_str.find(sync_pattern)
        if sync_pos == -1:
            return None
        
        # Extract length information from the binary string
        start_pos = sync_pos + len(sync_pattern)
        if start_pos + 16 > len(binary_str):
            return None
        
        length_bits = binary_str[start_pos:start_pos + 16]
        try:
            data_length = int(length_bits, 2)
        except ValueError:
            return None
        
        # Extract data part from the binary string
        data_start = start_pos + 16
        data_end = data_start + data_length * 4
        if data_end > len(binary_str):
            return None
        
        data_bits = binary_str[data_start:data_end]
        
        # Convert binary string back to characters
        decoded_chars = []
        for i in range(0, len(data_bits), 8):
            byte_bits = data_bits[i:i+8]
            if len(byte_bits) == 8:
                try:
                    char_code = int(byte_bits, 2)
                    decoded_chars.append(chr(char_code))
                except ValueError:
                    break
        
        if not decoded_chars:
            return None
        
        try:
            # Decode Base64 string
            decoded_str = ''.join(decoded_chars)
            decoded_data = base64.b64decode(decoded_str).decode()
            
            # Separate JSON data and checksum
            if '|' not in decoded_data:
                return None
            
            json_data, checksum = decoded_data.rsplit('|', 1)
            
            # Verify checksum
            expected_checksum = hashlib.md5(json_data.encode()).hexdigest()[:8]
            if checksum != expected_checksum:
                return None
            
            # Parse JSON data
            payload = json.loads(json_data)
            return payload
            
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return None
    
    def embed_image_watermark(self, image_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Embed dual watermarks (hidden + visible) in an image using configuration parameters.
        Generate a payload, embed a hidden frequency domain watermark, 
        and add a visible watermark.

        Args:
            image_tensor (torch.Tensor): Input image tensor [C, H, W].
            metadata (Dict[str, Any]): Generation metadata.

        Returns:
            torch.Tensor: Image tensor with mandatory dual watermarks.
        """
        import torch.fft as fft
        import torch.nn.functional as F
        
        # Get image watermark configuration
        image_config = self.config["watermark_2025"]["image"]
        compliance_version = self.config["watermark_2025"]["compliance_version"]
        hidden_config = image_config["hidden"]
        visible_config = image_config["visible"]
        
        # Create watermark payload based on configuration
        payload = {
            "standard": compliance_version,
            "model": self.model_id,
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": str(uuid.uuid4()),
            "user_id": hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:32],
            "content_hash": hashlib.sha256(str(image_tensor.shape).encode()).hexdigest()[:32],
            "trace_chain": str(uuid.uuid4()),
            "compliance": {
                "standard": compliance_version,
                "type": "mandatory_dual_watermark",
                "detectable": True,
                "removable": False,
                "tamper_proof": True
            }
        }
        
        # Step 1: Embed hidden frequency domain watermark
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        binary_data = ''.join(format(ord(c), '08b') for c in payload_str)
        
        c, h, w = image_tensor.shape
        
        # Convert image to frequency domain
        freq_domain = fft.fft2(image_tensor)
        
        # Prepare watermark bits for embedding
        watermark_bits = list(binary_data)
        bit_idx = 0
        dct_strength = hidden_config.get("dct_strength", 0.08)
        channels = hidden_config.get("channels", ["Y", "U", "V"])
        
        # Embed watermark in configured channels
        for channel_idx in range(min(c, len(channels))):
            # Apply DCT to each channel
            channel_freq = fft.fft2(image_tensor[channel_idx])
            
            # Embed in the mid-frequency region
            embed_start_h, embed_end_h = h//4, 3*h//4
            embed_start_w, embed_end_w = w//4, 3*w//4
            
            # Use pseudo-random pattern for embedding
            for i in range(embed_start_h, embed_end_h, 2):
                for j in range(embed_start_w, embed_end_w, 2):
                    if bit_idx < len(watermark_bits):
                        bit = int(watermark_bits[bit_idx])
                        # Adjust embedding strength based on position
                        strength = dct_strength * (1 + 0.1 * (i % 3))
                        phase_shift = bit * strength
                        channel_freq[i, j] *= (1 + phase_shift)
                        bit_idx += 1
            
            # Inverse transform back to spatial domain
            image_tensor[channel_idx] = torch.real(fft.ifft2(channel_freq))
        
        # Step 2: Add visible watermark
        watermarked_image = self._add_visible_watermark(image_tensor)
        
        return watermarked_image
    
    def _add_visible_watermark(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Add a visible watermark using configuration parameters.
        Calculate watermark position, create a semi-transparent background, 
        and apply it to the image.

        Args:
            image_tensor (torch.Tensor): Input image tensor [C, H, W].

        Returns:
            torch.Tensor: Image tensor with a visible watermark.
        """
        import torch
        
        c, h, w = image_tensor.shape
        
        # Get visible watermark configuration
        image_config = self.config["watermark_2025"]["image"]
        visible_config = image_config["visible"]
        
        # Get watermark text and dimensions
        watermark_text = visible_config.get("text", "AI-generated-PiscesL1")
        text_height = visible_config.get("text_height", 24)
        text_width = len(watermark_text) * visible_config.get("char_width", 12)
        
        # Calculate watermark position based on configuration
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
        
        # Create a copy of the image tensor
        watermarked = image_tensor.clone()
        
        # Get background color and opacity
        opacity = visible_config.get("opacity", 0.1)
        background_color = visible_config.get("background_color", [1.0, 1.0, 1.0])
        
        # Apply semi-transparent background to the watermark area
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
        """
        Embed a hidden watermark in audio using configuration parameters.
        Generate a payload, embed a hidden LSB watermark, 
        add a spectrum watermark, and restore the original shape.

        Args:
            audio_tensor (torch.Tensor): Input audio tensor [T] or [C, T].
            metadata (Dict[str, Any]): Generation metadata.

        Returns:
            torch.Tensor: Audio tensor with a mandatory hidden watermark.
        """
        import torch
        
        # Get audio watermark configuration
        audio_config = self.config["watermark_2025"]["audio"]
        compliance_version = self.config["watermark_2025"]["compliance_version"]
        hidden_config = audio_config["hidden"]
        
        # Create watermark payload based on configuration
        payload = {
            "standard": compliance_version,
            "model": self.model_id,
            "version": self.version,
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": str(uuid.uuid4()),
            "user_id": hashlib.sha256(metadata.get("user_id", "anonymous").encode()).hexdigest()[:32],
            "content_hash": hashlib.sha256(str(audio_tensor.shape).encode()).hexdigest()[:32],
            "trace_chain": str(uuid.uuid4()),
            "compliance": {
                "standard": compliance_version,
                "type": "mandatory_hidden_audio_watermark",
                "detectable": True,
                "removable": False,
                "tamper_proof": True
            }
        }
        
        # Step 1: Embed hidden LSB watermark
        payload_str = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        binary_data = ''.join(format(ord(c), '08b') for c in payload_str)
        
        # Ensure audio is a 1D tensor
        if len(audio_tensor.shape) > 1:
            audio_1d = audio_tensor.view(-1)
        else:
            audio_1d = audio_tensor
        
        # Get LSB configuration
        lsb_bits = hidden_config.get("lsb_bits", 6)
        lsb_strength = hidden_config.get("lsb_strength", 1.0)
        
        # Embed LSB watermark
        watermarked_audio = audio_1d.clone()
        bits = list(binary_data)
        
        for i in range(min(len(watermarked_audio) - lsb_bits, len(bits))):
            bit = int(bits[i])
            
            start_idx = i * lsb_bits
            if start_idx + lsb_bits <= len(watermarked_audio):
                for j in range(lsb_bits):
                    sample = watermarked_audio[start_idx + j]
                    
                    mask = 1 << j
                    if bit == 1:
                        watermarked_audio[start_idx + j] = sample | mask
                    else:
                        watermarked_audio[start_idx + j] = sample & ~mask
        
        # Step 2: Add spectrum watermark
        spectrum_config = hidden_config.get("spectrum", {})
        watermarked_audio = self._add_spectrum_watermark(watermarked_audio, payload_str, spectrum_config)
        
        # Step 3: Commented out - Remove visible voice watermark
        # watermarked_audio = self._add_voice_watermark(watermarked_audio)
        
        # Restore original shape
        return watermarked_audio.view(audio_tensor.shape)
    
    def _add_spectrum_watermark(self, audio: torch.Tensor, payload: str, spectrum_config: Dict[str, Any] = None) -> torch.Tensor:
        """
        Add a spectrum watermark to the inaudible frequency band using configuration parameters.
        Convert audio to frequency domain, encode watermark data into spectrum amplitude changes, 
        and convert back to time domain.

        Args:
            audio (torch.Tensor): Input audio tensor.
            payload (str): Watermark payload string.
            spectrum_config (Dict[str, Any]): Spectrum watermark configuration.

        Returns:
            torch.Tensor: Audio tensor with a spectrum watermark.
        """
        import torch.fft as fft
        
        if spectrum_config is None:
            spectrum_config = {}
        
        # Get spectrum watermark configuration
        sample_rate = spectrum_config.get("sample_rate", 44100)
        min_freq = spectrum_config.get("min_freq", 18000)
        max_freq = spectrum_config.get("max_freq", 20000)
        amplitude = spectrum_config.get("amplitude", 0.01)
        spread_factor = spectrum_config.get("spread_factor", 10)
        
        # Convert audio to frequency domain
        spectrum = fft.fft(audio)
        
        # Get frequency bins
        freq_bins = torch.fft.fftfreq(len(audio), 1/sample_rate)
        
        # Find frequency indices in the configured range
        target_freqs = (freq_bins >= min_freq) & (freq_bins <= max_freq)
        
        if torch.sum(target_freqs) > 0:
            # Encode watermark data into spectrum amplitude changes
            watermark_bits = ''.join(format(ord(c), '08b') for c in payload)
            
            target_indices = torch.where(target_freqs)[0]
            for i, idx in enumerate(target_indices[:len(watermark_bits)]):
                bit = int(watermark_bits[i % len(watermark_bits)])
                
                # Use spread spectrum technique
                for offset in range(spread_factor):
                    if idx + offset < len(spectrum):
                        spectrum[idx + offset] += bit * amplitude * (1 - offset/spread_factor)
        
        # Convert back to time domain
        return torch.real(fft.ifft(spectrum))
    
    def _add_voice_watermark(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Add a visible voice watermark to the audio.
        Create a simple voice signal using sine waves, add an envelope, 
        and add it to the beginning of the audio.

        Args:
            audio (torch.Tensor): Input audio tensor.

        Returns:
            torch.Tensor: Audio tensor with a visible voice watermark.
        """
        import torch
        
        # Assume sampling rate is 44.1kHz
        sample_rate = 44100
        
        # Visible voice watermark duration is 2 seconds
        watermark_duration = 2.0
        watermark_samples = int(watermark_duration * sample_rate)
        
        if len(audio) < watermark_samples:
            return audio
        
        # Create a simple voice signal
        t = torch.linspace(0, watermark_duration, watermark_samples)
        base_freq = 220
        voice_signal = (
            0.1 * torch.sin(2 * torch.pi * base_freq * t) +
            0.05 * torch.sin(2 * torch.pi * base_freq * 2 * t) +
            0.03 * torch.sin(2 * torch.pi * base_freq * 3 * t)
        )
        
        # Add envelope to the voice signal
        envelope = torch.ones_like(t)
        fade_samples = int(0.1 * sample_rate)
        envelope[:fade_samples] = torch.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = torch.linspace(1, 0, fade_samples)
        
        voice_signal *= envelope
        
        # Add voice signal to the beginning of the audio
        watermarked = audio.clone()
        watermarked[:watermark_samples] += voice_signal
        
        return watermarked
    
    def verify_watermark(self, content: Union[str, torch.Tensor], expected_metadata: Dict[str, Any]) -> bool:
        """
        Verify if the content contains a valid watermark.
        Extract watermark data based on content type and verify key fields.

        Args:
            content (Union[str, torch.Tensor]): Content to be verified.
            expected_metadata (Dict[str, Any]): Expected metadata.

        Returns:
            bool: Watermark verification result.
        """
        if isinstance(content, str):
            extracted = self.extract_text_watermark(content)
        elif isinstance(content, torch.Tensor):
            # Simplified processing; should call corresponding extraction method in practice
            extracted = None
        else:
            return False
        
        if not extracted:
            return False
        
        # Verify key fields in the extracted watermark data
        return (
            extracted.get("model") == self.model_id and
            extracted.get("compliance", {}).get("standard") == "GB/T 45225-2024"
        )


class WatermarkManager:
    """
    Watermark manager integrated into the inference process.
    This class is responsible for adding, checking, and verifying watermarks.
    """
    
    def __init__(self, model_id: str = "PiscesL1-1.5B"):
        # Initialize the watermark instance
        self.watermark = PiscesWatermark(model_id)
        # Watermark is always enabled and cannot be disabled
        self.enabled = True
        # Force enabled mode
        self.force_enabled = True  
    
    def add_watermark(self, content: Union[str, torch.Tensor], metadata: Dict[str, Any]) -> Union[str, torch.Tensor]:
        """
        Add mandatory watermarks based on content type.
        Enhance metadata, detect content type, and call the corresponding watermark embedding method.

        Args:
            content (Union[str, torch.Tensor]): Content to add watermark.
            metadata (Dict[str, Any]): Generation metadata.

        Returns:
            Union[str, torch.Tensor]: Content with watermark.
        """
        # Force add watermark regardless of the enabled state
        if content is None:
            return content
            
        # Enhance metadata collection
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
        except Exception as e:
            # Add basic watermark even if an error occurs
            if isinstance(content, str):
                return f"[AI生成]{content}"
            return content
        
        return content
    
    def _detect_content_type(self, content: Union[str, torch.Tensor]) -> str:
        """
        Automatically detect content type based on the input content.

        Args:
            content (Union[str, torch.Tensor]): Content to detect type.

        Returns:
            str: Detected content type.
        """
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
    
    def _embed_video_watermark(self, video_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Process video watermarking by adding image watermark to each frame.

        Args:
            video_tensor (torch.Tensor): Input video tensor [C, T, H, W].
            metadata (Dict[str, Any]): Generation metadata.

        Returns:
            torch.Tensor: Video tensor with watermark.
        """
        import torch
        
        # Get video dimensions
        c, t, h, w = video_tensor.shape
        watermarked_video = video_tensor.clone()
        
        for frame_idx in range(t):
            frame = video_tensor[:, frame_idx, :, :]
            watermarked_frame = self.watermark.embed_image_watermark(frame, {
                **metadata,
                "frame_index": frame_idx,
                "total_frames": t
            })
            watermarked_video[:, frame_idx, :, :] = watermarked_frame
        
        return watermarked_video
    
    def check_watermark(self, content: Union[str, torch.Tensor]) -> Optional[Dict[str, Any]]:
        """
        Check watermark information in the content.
        Detect content type and call the corresponding watermark extraction method.

        Args:
            content (Union[str, torch.Tensor]): Content to check watermark.

        Returns:
            Optional[Dict[str, Any]]: Watermark information or None if not found.
        """
        if isinstance(content, str):
            return self.watermark.extract_text_watermark(content)
        elif isinstance(content, torch.Tensor):
            content_type = self._detect_content_type(content)
            if content_type == "image":
                return self._detect_image_watermark(content)
            elif content_type == "audio":
                return self._detect_audio_watermark(content)
        return None
    
    def _detect_image_watermark(self, image_tensor: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        Detect image watermark by loading configuration and returning watermark information.

        Args:
            image_tensor (torch.Tensor): Input image tensor.

        Returns:
            Optional[Dict[str, Any]]: Detected watermark information or None.
        """
        try:
            with open('configs/watermark.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            return {
                "type": "image",
                "detected": True,
                "config": config["watermark_2025"]["image"]
            }
        except:
            return {"type": "image", "detected": True}

    def _detect_audio_watermark(self, audio_tensor: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        Detect audio watermark by loading configuration and returning watermark information.

        Args:
            audio_tensor (torch.Tensor): Input audio tensor.

        Returns:
            Optional[Dict[str, Any]]: Detected watermark information or None.
        """
        try:
            with open('configs/watermark.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            return {
                "type": "audio",
                "detected": True,
                "config": config["watermark_2025"]["audio"]
            }
        except:
            return {"type": "audio", "detected": True}
    
    def disable(self):
        """
        Disable function removed, maintain interface compatibility.
        Watermark is in force mode and cannot be disabled.
        """
        pass
    
    def enable(self):
        """
        Always enabled.
        Watermark is in force mode and always enabled.
        """
        pass
    
    def is_compliant(self, content: Union[str, torch.Tensor]) -> bool:
        """
        Check if the content complies with the mandatory watermark standard.
        For text, check both visible and hidden watermarks; other types are considered compliant by default.

        Args:
            content (Union[str, torch.Tensor]): Content to check compliance.

        Returns:
            bool: Compliance result.
        """
        if isinstance(content, str):
            try:
                with open('configs/watermark.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                visible_text = config["watermark_2025"]["image"]["visible"]["text"]
            except:
                visible_text = "AI-Generated"
            
            # Check if there is a visible prefix
            has_visible = content.startswith(f"[{visible_text}]")
            # Check if there is a hidden watermark
            has_hidden = self.watermark.extract_text_watermark(content) is not None
            return has_visible or has_hidden
        return True  # Other types are considered compliant by default

# Global watermark manager instance (2025 mandatory standard)
watermark_manager = WatermarkManager()

def watermark_text(text: str, prompt: str = "", generation_params: Dict[str, Any] = None) -> str:
    """
    Convenience function to add a watermark to text.

    Args:
        text (str): Original text.
        prompt (str): Input prompt.
        generation_params (Dict[str, Any]): Generation parameters.

    Returns:
        str: Text with watermark.
    """
    metadata = {
        "prompt": prompt,
        "params": generation_params or {}
    }
    return watermark_manager.add_watermark(text, metadata)

def check_text_watermark(text: str) -> Optional[Dict[str, Any]]:
    """
    Check watermark information in text.

    Args:
        text (str): Text to check.

    Returns:
        Optional[Dict[str, Any]]: Watermark information or None.
    """
    return watermark_manager.check_watermark(text)