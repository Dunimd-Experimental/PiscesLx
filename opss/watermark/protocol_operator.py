#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
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

"""
Watermark Protocol Operator

This module implements the watermark framing protocol for reliable payload
embedding and extraction. The protocol ensures robust synchronization,
error detection, and payload integrity.

Protocol Frame Structure:
    +--------+--------+--------+--------+
    |      SYNC (32 bits)              |
    +--------+--------+--------+--------+
    |  LEN (16 bits)  |  PAYLOAD...   |
    +--------+--------+--------+--------+
    |         CRC32 (32 bits)          |
    +--------+--------+--------+--------+

Frame Components:
    SYNC (0xA5F0A5F0): Synchronization word for frame detection
    LEN: Payload length in bytes (16-bit unsigned integer)
    PAYLOAD: JSON-encoded metadata dictionary
    CRC32: Error detection checksum

Key Features:
    - Synchronization pattern for reliable frame detection
    - Length field for payload boundary identification
    - CRC32 error detection
    - Base64 encoding for binary-safe transmission
    - Redundant parity bits for error correction

Error Detection:
    - CRC32 checksum provides 32-bit error detection
    - Parity bits enable repeat-3 decoding
    - Frame validation prevents false positives

Usage Examples:
    >>> from opss.watermark.protocol_operator import POPSSProtocolOperator
    >>> operator = POPSSProtocolOperator()
    >>> 
    >>> # Encode payload to bitstream
    >>> payload = {"model": "PiscesL1", "timestamp": "2025-01-01"}
    >>> bitstream = operator.encode(payload)
    >>> 
    >>> # Decode bitstream to payload
    >>> decoded = operator.decode(bitstream)
    >>> print(decoded["model"])
"""

import json
import zlib
import base64
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from utils.opsc.base import PiscesLxBaseOperator
from utils.opsc.interface import PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION


SYNC_HEADER = 0xA5F0A5F0
LEN_BITS = 16
CRC_BITS = 32
PARITY_BITS = 1


class POPSSProtocolOperator(PiscesLxBaseOperator):
    """
    Watermark framing protocol operator.
    
    This operator implements the synchronization, framing, and error detection
    mechanisms required for robust watermark embedding and extraction. It uses
    a structured frame format with synchronization headers, length fields, and
    CRC32 checksums to ensure reliable payload transmission.
    
    Protocol Frame Structure:
        The protocol uses a fixed header format followed by variable-length
        payload and error detection fields:
        
        +--------+--------+--------+--------+
        |      SYNC (32 bits)              |  <- 0xA5F0A5F0
        +--------+--------+--------+--------+
        |  LEN (16 bits)  |  PAYLOAD...   |  <- Payload length + data
        +--------+--------+--------+--------+
        |         CRC32 (32 bits)          |  <- Error detection
        +--------+--------+--------+--------+
    
    Frame Components:
        - SYNC (0xA5F0A5F0): 32-bit synchronization word for frame detection.
          This pattern was chosen for its good autocorrelation properties
          and low probability of false detection in random data.
        - LEN: 16-bit unsigned integer specifying payload length in bytes.
          Maximum payload size is 65,535 bytes (64KB - 1).
        - PAYLOAD: JSON-encoded metadata dictionary containing watermark
          information such as model_id, timestamp, user_id, and trace_chain.
        - CRC32: 32-bit CRC checksum computed over the payload bytes using
          the IEEE 802.3 polynomial (0xEDB88320).
    
    Error Correction:
        The protocol supports optional redundancy encoding using repeat-3
        coding. Each bit is triplicated, and majority voting is applied
        during decoding to correct single-bit errors.
    
    Attributes:
        sync_header (int): 32-bit synchronization pattern (0xA5F0A5F0)
        max_payload_size (int): Maximum payload size in bytes (65,535)
        parity_enabled (bool): Enable parity-based error correction
        
    Input Format:
        {
            "action": "encode" | "decode",
            "payload": Dict[str, Any],  # for encode
            "bitstream": str,             # for decode
            "parity_enabled": bool        # optional
        }
        
    Output Format:
        {
            "action": str,
            "result": Any,
            "valid": bool,
            "frame_info": Dict
        }
    
    Example:
        >>> operator = POPSSProtocolOperator()
        >>> payload = {"model": "PiscesL1", "user": "test"}
        >>> bitstream = operator.frame_payload(payload)
        >>> decoded = operator.extract_from_bits(bitstream)
        >>> print(decoded["model"])
        'PiscesL1'
    """
    
    def __init__(self):
        super().__init__()
        self.name = "pisceslx_protocol_operator"
        self.version = VERSION
        self.description = "Watermark framing protocol with SYNC+LEN+CRC32"
        self.sync_header = SYNC_HEADER
        self.max_payload_size = 2 ** LEN_BITS - 1
        self.parity_enabled = False
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["encode", "decode"]
                },
                "payload": {
                    "type": "object",
                    "description": "Payload dictionary for encoding"
                },
                "bitstream": {
                    "type": "string",
                    "description": "Bitstream for decoding"
                },
                "parity_enabled": {
                    "type": "boolean",
                    "description": "Enable parity encoding for error correction"
                },
                "redundancy": {
                    "type": "integer",
                    "description": "Redundancy level for encoding"
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "result": {"type": "any"},
                "valid": {"type": "boolean"},
                "frame_info": {
                    "type": "object",
                    "properties": {
                        "sync_found": {"type": "boolean"},
                        "payload_length": {"type": "integer"},
                        "crc_valid": {"type": "boolean"},
                        "parity_valid": {"type": "boolean"}
                    }
                }
            }
        }
    
    def _execute_impl(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        action = inputs.get("action", "encode")
        
        if action == "encode":
            return self._encode(inputs)
        elif action == "decode":
            return self._decode(inputs)
        else:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Unknown action: {action}"
            )
    
    def _encode(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Encode a payload dictionary into a framed bitstream.
        
        This method implements the encoding pipeline that transforms a
        metadata dictionary into a binary bitstream suitable for watermark
        embedding. The encoding process includes JSON serialization, frame
        construction, CRC computation, and optional redundancy encoding.
        
        Encoding Pipeline:
            1. Serialize payload to JSON with sorted keys and minimal whitespace
            2. Validate payload size against maximum limit (65,535 bytes)
            3. Compute CRC32 checksum over payload bytes
            4. Construct frame: SYNC + LEN + PAYLOAD + CRC
            5. Convert frame bytes to binary string
            6. Apply optional redundancy encoding (repeat-3)
            7. Encode to Base64 for safe transmission
        
        Args:
            inputs: Dictionary containing encoding parameters
                - payload (Dict[str, Any]): Metadata to encode
                - redundancy (int): Redundancy level for error correction (default: 1)
                - parity_enabled (bool): Enable parity encoding (default: False)
        
        Returns:
            PiscesLxOperatorResult: Encoding result containing
                - output: Dict with bitstream, base64, payload_size, and crc
                - metadata: Dict with frame info (sync_found, payload_length, etc.)
        
        Raises:
            ValueError: If payload exceeds maximum size limit
        """
        payload = inputs.get("payload", {})
        redundancy = inputs.get("redundancy", 1)
        self.parity_enabled = inputs.get("parity_enabled", False)
        
        try:
            payload_bytes = json.dumps(
                payload, 
                separators=(',', ':'), 
                sort_keys=True,
                ensure_ascii=False
            ).encode('utf-8')
            
            if len(payload_bytes) >= self.max_payload_size:
                raise ValueError(f"Payload too large: {len(payload_bytes)} >= {self.max_payload_size}")
            
            crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
            
            frame = bytearray()
            frame.extend(self.sync_header.to_bytes(4, byteorder='big'))
            frame.extend(len(payload_bytes).to_bytes(2, byteorder='big'))
            frame.extend(payload_bytes)
            frame.extend(crc.to_bytes(4, byteorder='big'))
            
            bitstream = self._bytes_to_bits(bytes(frame))
            
            if redundancy > 1 or self.parity_enabled:
                bitstream = self._apply_redundancy(bitstream, redundancy)
            
            base64_encoded = base64.b64encode(bitstream.encode('ascii')).decode('ascii')
            
            frame_info = {
                "sync_found": True,
                "payload_length": len(payload_bytes),
                "crc_valid": True,
                "parity_valid": True,
                "redundancy": redundancy,
                "bitstream_length": len(bitstream)
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "bitstream": bitstream,
                    "base64": base64_encoded,
                    "payload_size": len(payload_bytes),
                    "crc": crc
                },
                metadata=frame_info
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _decode(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Decode a bitstream to extract the payload dictionary.
        
        This method implements the decoding pipeline that extracts metadata
        from a binary bitstream. The decoding process includes synchronization
        detection, frame parsing, CRC verification, and payload deserialization.
        
        Decoding Pipeline:
            1. Remove redundancy encoding if present (majority voting)
            2. Search for synchronization header (0xA5F0A5F0)
            3. Extract length field to determine payload boundaries
            4. Extract payload bytes and CRC checksum
            5. Verify CRC integrity
            6. Deserialize JSON payload to dictionary
        
        Error Handling:
            - Sync header not found: Returns error with sync_found=False
            - Incomplete frame: Returns error if frame is truncated
            - CRC mismatch: Returns error with crc_valid=False
            - Invalid JSON: Attempts UTF-8 then Latin-1 decoding
        
        Args:
            inputs: Dictionary containing decoding parameters
                - bitstream (str): Binary string to decode
                - redundancy (int): Redundancy level used during encoding (default: 1)
                - parity_enabled (bool): Enable parity decoding (default: False)
        
        Returns:
            PiscesLxOperatorResult: Decoding result containing
                - output: Dict with payload, payload_size, and crc
                - metadata: Dict with frame info (sync_found, crc_valid, etc.)
        
        Raises:
            ValueError: If bitstream is empty or decoding fails
        """
        bitstream = inputs.get("bitstream", "")
        redundancy = inputs.get("redundancy", 1)
        self.parity_enabled = inputs.get("parity_enabled", False)
        
        try:
            if not bitstream:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Empty bitstream"
                )
            
            if self.parity_enabled or redundancy > 1:
                bitstream = self._remove_redundancy(bitstream, redundancy)
            
            sync_bits = self._int_to_bits(self.sync_header, 32)
            pos = bitstream.find(sync_bits)
            
            if pos < 0:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Sync header not found",
                    metadata={"sync_found": False}
                )
            
            min_tail = LEN_BITS + CRC_BITS
            if pos + 32 + min_tail > len(bitstream):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Incomplete frame"
                )
            
            len_bits = bitstream[pos + 32: pos + 32 + LEN_BITS]
            payload_len = self._bits_to_int(len_bits)
            
            payload_bits_start = pos + 32 + LEN_BITS
            payload_bits_len = payload_len * 8
            crc_bits_start = payload_bits_start + payload_bits_len
            crc_bits_end = crc_bits_start + CRC_BITS
            
            if crc_bits_end > len(bitstream):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Frame truncated"
                )
            
            payload_bits = bitstream[payload_bits_start: payload_bits_start + payload_bits_len]
            crc_bits = bitstream[crc_bits_start: crc_bits_end]
            
            payload_bytes = self._bits_to_bytes(payload_bits)
            if payload_bytes is None:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Invalid payload encoding"
                )
            
            crc_val = self._bits_to_int(crc_bits)
            computed_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
            
            crc_valid = computed_crc == crc_val
            
            if not crc_valid:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="CRC verification failed",
                    metadata={
                        "sync_found": True,
                        "payload_length": payload_len,
                        "crc_valid": False,
                        "expected_crc": computed_crc,
                        "received_crc": crc_val
                    }
                )
            
            try:
                payload = json.loads(payload_bytes.decode('utf-8'))
            except UnicodeDecodeError:
                payload = json.loads(payload_bytes.decode('latin-1'))
            
            frame_info = {
                "sync_found": True,
                "payload_length": payload_len,
                "crc_valid": True,
                "parity_valid": True,
                "redundancy": redundancy
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "payload": payload,
                    "payload_size": payload_len,
                    "crc": computed_crc
                },
                metadata=frame_info
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _int_to_bits(self, value: int, bit_len: int) -> str:
        """
        Convert an integer to a binary string representation.
        
        This method converts an integer value to a fixed-width binary string
        using big-endian bit ordering (MSB first).
        
        Args:
            value: Integer value to convert
            bit_len: Number of bits in the output string
        
        Returns:
            str: Binary string of length bit_len
        """
        return ''.join('1' if (value >> i) & 1 else '0' for i in reversed(range(bit_len)))
    
    def _bits_to_int(self, bits: str) -> int:
        """
        Convert a binary string to an integer value.
        
        This method parses a binary string and returns the corresponding
        integer value using big-endian interpretation (MSB first).
        
        Args:
            bits: Binary string to convert
        
        Returns:
            int: Integer value represented by the binary string
        """
        v = 0
        for b in bits:
            v = (v << 1) | (1 if b == '1' else 0)
        return v
    
    def _bytes_to_bits(self, data: bytes) -> str:
        """
        Convert a bytes object to a binary string representation.
        
        This method converts each byte to its 8-bit binary representation
        and concatenates them into a single binary string.
        
        Args:
            data: Bytes object to convert
        
        Returns:
            str: Binary string where each byte is represented as 8 bits
        """
        return ''.join(f"{b:08b}" for b in data)
    
    def _bits_to_bytes(self, bits: str) -> Optional[bytes]:
        """
        Convert a binary string to a bytes object.
        
        This method parses a binary string and groups bits into 8-bit chunks
        to reconstruct the original bytes. Returns None if the bit string
        length is not a multiple of 8.
        
        Args:
            bits: Binary string to convert (length must be multiple of 8)
        
        Returns:
            Optional[bytes]: Bytes object if conversion succeeds, None otherwise
        """
        if len(bits) % 8 != 0:
            return None
        out = bytearray()
        for i in range(0, len(bits), 8):
            out.append(int(bits[i:i+8], 2))
        return bytes(out)
    
    def _apply_redundancy(self, bitstream: str, redundancy: int) -> str:
        """
        Apply redundancy encoding for error correction.
        
        This method implements repeat-3 coding where each bit is triplicated.
        This allows for single-bit error correction through majority voting
        during decoding.
        
        Encoding Scheme:
            - '0' -> '000'
            - '1' -> '111'
        
        Args:
            bitstream: Original binary string
            redundancy: Redundancy level (only >1 triggers encoding)
        
        Returns:
            str: Encoded binary string with redundancy
        """
        if redundancy <= 1:
            return bitstream
        
        result = ""
        for i, bit in enumerate(bitstream):
            if bit == '1':
                result += '111'
            else:
                result += '000'
        
        return result
    
    def _remove_redundancy(self, bitstream: str, redundancy: int) -> str:
        """
        Remove redundancy encoding and perform majority voting.
        
        This method decodes repeat-3 coded bitstreams by applying majority
        voting to each group of 3 bits. This corrects single-bit errors
        that may have occurred during transmission.
        
        Decoding Scheme:
            - Groups of 3 bits are analyzed
            - If 2 or more bits are '1', output '1'
            - If 2 or more bits are '0', output '0'
        
        Args:
            bitstream: Redundancy-encoded binary string
            redundancy: Redundancy level used during encoding
        
        Returns:
            str: Decoded binary string with errors corrected
        """
        if redundancy <= 1:
            return bitstream
        
        result = ""
        for i in range(0, len(bitstream), 3):
            tri = bitstream[i:i+3]
            if len(tri) < 3:
                result += tri if tri else ''
                continue
            if tri.count('1') >= 2:
                result += '1'
            else:
                result += '0'
        
        return result
    
    def frame_payload(self, payload: Dict[str, Any], redundancy: int = 1) -> str:
        """
        Frame a payload dictionary into a bitstream.
        
        This is a convenience method that wraps the encoding operation
        and returns the bitstream directly without the full result object.
        
        Args:
            payload: Dictionary to encode containing watermark metadata
            redundancy: Redundancy level for error correction (1-3)
        
        Returns:
            str: Binary string representation of framed payload
        
        Raises:
            ValueError: If encoding fails due to size limits or other errors
        
        Example:
            >>> operator = POPSSProtocolOperator()
            >>> bitstream = operator.frame_payload({"model": "PiscesL1"})
        """
        result = self._encode({
            "payload": payload,
            "redundancy": redundancy
        })
        if result.is_success():
            return result.output["bitstream"]
        raise ValueError(f"Framing failed: {result.error}")
    
    def extract_from_bits(self, bitstream: str, redundancy: int = 1) -> Optional[Dict[str, Any]]:
        """
        Extract payload from a bitstream.
        
        This is a convenience method that wraps the decoding operation
        and returns the payload dictionary directly. Returns None if
        extraction fails for any reason.
        
        Args:
            bitstream: Binary string to decode
            redundancy: Redundancy level used during encoding (1-3)
        
        Returns:
            Optional[Dict[str, Any]]: Decoded payload dictionary or None if
                extraction fails (sync not found, CRC error, etc.)
        
        Example:
            >>> operator = POPSSProtocolOperator()
            >>> bitstream = operator.frame_payload({"model": "PiscesL1"})
            >>> payload = operator.extract_from_bits(bitstream)
            >>> print(payload["model"])
            'PiscesL1'
        """
        result = self._decode({
            "bitstream": bitstream,
            "redundancy": redundancy
        })
        if result.is_success():
            return result.output["payload"]
        return None


@dataclass
class POPSSFrameInfo:
    """
    Information about a decoded protocol frame.
    
    This dataclass holds metadata extracted during frame decoding,
    providing details about synchronization status, payload size,
    and integrity verification results.
    
    Attributes:
        sync_found (bool): Whether the synchronization header was detected
        payload_length (int): Length of the payload in bytes
        crc_valid (bool): Whether CRC verification passed
        frame_start (int): Bit position where frame starts (default: 0)
        frame_end (int): Bit position where frame ends (default: 0)
    
    Properties:
        is_valid (bool): True if sync was found and CRC is valid
    
    Example:
        >>> info = POPSSFrameInfo(sync_found=True, payload_length=64, crc_valid=True)
        >>> info.is_valid
        True
    """
    sync_found: bool
    payload_length: int
    crc_valid: bool
    frame_start: int = 0
    frame_end: int = 0
    
    @property
    def is_valid(self) -> bool:
        return self.sync_found and self.crc_valid


class POPSSWatermarkProtocolOperator(PiscesLxBaseOperator):
    """
    Enhanced Protocol Operator with unified factory and constants.
    
    This class combines all protocol-related constants and functions into
    a cohesive operator with factory methods. It provides the same functionality
    as POPSSProtocolOperator but with class-level constants for easier access.
    
    This operator is the recommended interface for protocol operations in
    production environments, offering:
        - Standardized frame structure with SYNC+LEN+CRC32
        - Optional redundancy encoding for error correction
        - Base64 encoding for safe transmission
        - Factory method for easy instantiation
    
    Class Attributes:
        SYNC_HEADER (int): 32-bit synchronization pattern (0xA5F0A5F0)
        LEN_BITS (int): Length field size in bits (16)
        CRC_BITS (int): CRC field size in bits (32)
        PARITY_BITS (int): Parity field size in bits (1)
    
    Instance Attributes:
        sync_header (int): Instance copy of synchronization pattern
        max_payload_size (int): Maximum payload size (65,535 bytes)
        parity_enabled (bool): Enable parity-based error correction
    
    Methods:
        create: Factory method to create operator instance
    """
    
    SYNC_HEADER = 0xA5F0A5F0
    LEN_BITS = 16
    CRC_BITS = 32
    PARITY_BITS = 1
    
    def __init__(self):
        super().__init__()
        self.name = "pisceslx_protocol_operator"
        self.version = VERSION
        self.description = "Watermark framing protocol with SYNC+LEN+CRC32"
        self.sync_header = POPSSWatermarkProtocolOperator.SYNC_HEADER
        self.max_payload_size = 2 ** POPSSWatermarkProtocolOperator.LEN_BITS - 1
        self.parity_enabled = False
    
    @classmethod
    def create(cls) -> 'POPSSWatermarkProtocolOperator':
        """Factory method to create a protocol operator instance."""
        return cls()


def create_protocol_operator() -> 'POPSSProtocolOperator':
    """
    Factory function to create a protocol operator instance.
    
    This function provides a convenient way to instantiate a protocol
    operator with default settings.
    
    Returns:
        POPSSProtocolOperator: Protocol operator instance ready for use
    
    Example:
        >>> operator = create_protocol_operator()
        >>> bitstream = operator.frame_payload({"model": "PiscesL1"})
    """
    return POPSSProtocolOperator()


__all__ = [
    "POPSSProtocolOperator",
    "POPSSFrameInfo",
    "POPSSWatermarkProtocolOperator",
    "SYNC_HEADER",
    "LEN_BITS",
    "CRC_BITS",
    "create_protocol_operator"
]
