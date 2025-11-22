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

import json
import zlib
from typing import Optional, Dict

SYNC_HEADER = 0xA5F0A5F0  # 32-bit sync word
LEN_BITS = 16             # payload length field in bytes (0..65535)

def _int_to_bits(value: int, bit_len: int) -> str:
    return ''.join('1' if (value >> i) & 1 else '0' for i in reversed(range(bit_len)))

def _bits_to_int(bits: str) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | (1 if b == '1' else 0)
    return v

def _bytes_to_bits(data: bytes) -> str:
    return ''.join(f"{b:08b}" for b in data)

def _bits_to_bytes(bits: str) -> Optional[bytes]:
    if len(bits) % 8 != 0:
        return None
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(int(bits[i:i+8], 2))
    return bytes(out)

def frame_payload(payload: Dict) -> str:
    """Frame JSON payload into a bitstring with SYNC+LEN+PAYLOAD+CRC32."""
    payload_bytes = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode('utf-8')
    if len(payload_bytes) >= 2 ** LEN_BITS:
        raise ValueError("payload too large for LEN field")
    crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
    frame = bytearray()
    frame += SYNC_HEADER.to_bytes(4, byteorder='big')
    frame += len(payload_bytes).to_bytes(2, byteorder='big')
    frame += payload_bytes
    frame += crc.to_bytes(4, byteorder='big')
    return _bytes_to_bits(bytes(frame))

def extract_from_bits(bitstream: str) -> Optional[Dict]:
    """Search for a valid frame in bitstream and return JSON payload dict if found."""
    # Convert sync header to bits for search
    sync_bits = _int_to_bits(SYNC_HEADER, 32)
    pos = bitstream.find(sync_bits)
    if pos < 0:
        return None
    # After sync: need at least LEN + CRC even if no payload
    min_tail = LEN_BITS + 32
    if pos + 32 + min_tail > len(bitstream):
        return None
    # Read LEN (bytes)
    len_bits = bitstream[pos + 32: pos + 32 + LEN_BITS]
    payload_len = _bits_to_int(len_bits)
    # Compute byte-aligned start of payload after LEN
    payload_bits_start = pos + 32 + LEN_BITS
    payload_bits_len = payload_len * 8
    crc_bits_start = payload_bits_start + payload_bits_len
    crc_bits_end = crc_bits_start + 32
    if crc_bits_end > len(bitstream):
        return None
    payload_bits = bitstream[payload_bits_start: payload_bits_start + payload_bits_len]
    crc_bits = bitstream[crc_bits_start: crc_bits_end]
    payload_bytes = _bits_to_bytes(payload_bits)
    if payload_bytes is None:
        return None
    crc_val = _bits_to_int(crc_bits)
    if (zlib.crc32(payload_bytes) & 0xFFFFFFFF) != crc_val:
        return None
    try:
        return json.loads(payload_bytes.decode('utf-8'))
    except Exception:
        return None
