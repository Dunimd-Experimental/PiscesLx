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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
PiscesLx Watermark Detection Core Operators

This module implements flagship watermark detection operators based on OPSC architecture,
supporting comprehensive detection across text, image, audio, video, and model weights.

Supported Content Types:
    - Text: Zero-width character encoding detection
    - Image: DCT frequency domain watermark detection
    - Audio: STFT ultrasonic band watermark detection
    - Video: Frame-by-frame multi-modal detection
    - Model Weights: Codebook correlation verification

Key Features:
    - Full integration with opss.watermark orchestrator
    - Multi-jurisdiction compliance validation
    - Audit trail logging for forensic analysis
    - Batch processing capabilities
    - Tamper evidence detection

Usage Examples:
    >>> from tools.wmc import PiscesLxWatermarkDetectionOperator
    >>> detector = PiscesLxWatermarkDetectionOperator()
    >>> 
    >>> # Detect text watermark
    >>> result = detector.detect_text_watermark("AI-generated text...")
    >>> 
    >>> # Detect video watermark
    >>> video_result = detector.detect_video_watermark(video_tensor)
    >>> 
    >>> # Verify model weights
    >>> model_result = detector.detect_model_watermark(model)

Author: PiscesL1 Development Team
Version: 1.0.0
"""

import struct
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import time
from datetime import datetime
import numpy as np

from utils.opsc.base import PiscesLxTransformOperator
from utils.opsc.interface import PiscesLxOperatorConfig
from utils.opsc.registry import PiscesLxOperatorRegistrar
from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Wmc", file_path=get_log_file("PiscesLx.Tools.Wmc"), enable_file=True)


@PiscesLxOperatorRegistrar()
class PiscesLxWatermarkDetectionOperator(PiscesLxTransformOperator):
    """
    PiscesLx Flagship Watermark Detection Operator
    
    Comprehensive watermark detection supporting text, image, audio, video,
    and model weight watermarks with full opss.watermark orchestrator integration.
    
    Attributes:
        strict_mode (bool): Raise exceptions on detection failures
        enable_compliance_check (bool): Enable jurisdiction compliance validation
        orchestrator: opss.watermark orchestrator instance
        detection_stats (Dict): Detection statistics tracking
        
    Supported Detection Types:
        - text: Zero-width character and protocol-based detection
        - image: DCT frequency domain extraction
        - audio: STFT ultrasonic band extraction
        - video: Multi-frame multi-modal detection
        - model: Weight codebook correlation verification
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, 
                 strict_mode: bool = False,
                 enable_compliance_check: bool = True):
        super().__init__(config)
        if config is not None:
            params = getattr(config, "parameters", {}) or {}
            strict_mode = bool(params.get("strict_mode", strict_mode))
            enable_compliance_check = bool(params.get("enable_compliance_check", enable_compliance_check))
        self.strict_mode = strict_mode
        self.enable_compliance_check = enable_compliance_check
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'compliance_passed': 0,
            'compliance_failed': 0,
            'by_type': {
                'text': 0, 'image': 0, 'audio': 0, 'video': 0, 'model': 0
            }
        }
        
        self._setup_detection_components()
        
        _LOG.info("PiscesLxWatermarkDetectionOperator initialized")
    
    def _setup_detection_components(self):
        """Setup detection components with opss.watermark integration."""
        self.orchestrator = None
        self.content_operator = None
        self.weight_operator = None
        self.compliance_operator = None
        self.audit_operator = None
        self.protocol_operator = None
        self.dct_operator = None
        
        try:
            from opss.watermark import (
                POPSSWatermarkOrchestrator,
                POPSSWatermarkContentOperator,
                POPSSWatermarkWeightOperator,
                POPSSWatermarkComplianceOperator,
                POPSSWatermarkAuditOperator,
                POPSSWatermarkProtocolOperator,
                POPSSWatermarkDCTOperator,
                POPSSWatermarkConfig
            )
            
            self.config_instance = POPSSWatermarkConfig()
            self.orchestrator = POPSSWatermarkOrchestrator(self.config_instance)
            self.content_operator = POPSSWatermarkContentOperator(self.config_instance)
            self.weight_operator = POPSSWatermarkWeightOperator(self.config_instance)
            self.compliance_operator = POPSSWatermarkComplianceOperator(self.config_instance) if self.enable_compliance_check else None
            self.audit_operator = POPSSWatermarkAuditOperator(self.config_instance)
            self.protocol_operator = POPSSWatermarkProtocolOperator()
            self.dct_operator = POPSSWatermarkDCTOperator()
            
            _LOG.info("OPSS watermark operators successfully loaded")
            
        except ImportError as e:
            _LOG.warning(f"OPSS watermark operators not available: {e}")
            self._setup_fallback_detection()
    
    def _setup_fallback_detection(self):
        """Setup fallback detection methods."""
        self.zero_width_chars = ["\u200B", "\u200C", "\u200D", "\uFEFF"]
        self.char_mapping = {
            "00": self.zero_width_chars[0],
            "01": self.zero_width_chars[1], 
            "10": self.zero_width_chars[2],
            "11": self.zero_width_chars[3]
        }
        self.reverse_mapping = {v: k for k, v in self.char_mapping.items()}
        
        _LOG.info("Fallback detection methods initialized")
    
    def detect_text_watermark(self, text: str, 
                              jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
        """
        Detect watermark in text content.
        
        Args:
            text: Text content to analyze
            jurisdiction: Target jurisdiction for compliance (CN, EU, US, UK, JP, KR, GLOBAL)
            
        Returns:
            Detection result dictionary with watermark info and compliance status
        """
        start_time = time.time()
        
        result = {
            'watermark_detected': False,
            'watermark_info': None,
            'compliance_status': 'unknown',
            'detection_confidence': 0.0,
            'jurisdiction': jurisdiction,
            'content_type': 'text',
            'processing_time': 0.0,
            'error': None
        }
        
        try:
            if self.content_operator:
                extract_result = self.content_operator._extract_from_text(text)
                if extract_result.is_success():
                    output = extract_result.output
                    if output.get('watermark_found'):
                        result['watermark_detected'] = True
                        result['watermark_info'] = output.get('payload')
                        result['detection_confidence'] = 0.95
            else:
                fallback_result = self._fallback_text_detection(text)
                if fallback_result:
                    result['watermark_detected'] = True
                    result['watermark_info'] = fallback_result
                    result['detection_confidence'] = 0.85
            
            if result['watermark_detected'] and self.compliance_operator:
                compliance_result = self.compliance_operator._validate({
                    "content_type": "text",
                    "jurisdiction": jurisdiction
                })
                if compliance_result.is_success():
                    result['compliance_status'] = compliance_result.output.get('compliance_status', 'unknown')
                    if result['compliance_status'] == 'compliant':
                        self.detection_stats['compliance_passed'] += 1
                    else:
                        self.detection_stats['compliance_failed'] += 1
            
            self._update_stats('text', result['watermark_detected'])
            self._log_audit('text', result['watermark_detected'], start_time, jurisdiction)
            
        except Exception as e:
            result['error'] = str(e)
            result['compliance_status'] = 'error'
            _LOG.error(f"Text watermark detection failed: {e}")
            if self.strict_mode:
                raise
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def detect_image_watermark(self, image: torch.Tensor,
                               jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
        """
        Detect watermark in image content.
        
        Args:
            image: Image tensor (C, H, W) or (B, C, H, W)
            jurisdiction: Target jurisdiction for compliance
            
        Returns:
            Detection result dictionary
        """
        start_time = time.time()
        
        result = {
            'watermark_detected': False,
            'watermark_info': None,
            'compliance_status': 'unknown',
            'detection_confidence': 0.0,
            'jurisdiction': jurisdiction,
            'content_type': 'image',
            'processing_time': 0.0,
            'error': None
        }
        
        try:
            if not isinstance(image, torch.Tensor):
                raise ValueError("Image must be a torch.Tensor")
            
            if self.content_operator:
                extract_result = self.content_operator._extract_from_image(image)
                if extract_result.is_success():
                    output = extract_result.output
                    if output.get('watermark_found'):
                        result['watermark_detected'] = True
                        result['watermark_info'] = output.get('payload')
                        result['detection_confidence'] = 0.90
            else:
                fallback_result = self._fallback_image_detection(image)
                if fallback_result:
                    result['watermark_detected'] = True
                    result['watermark_info'] = fallback_result
                    result['detection_confidence'] = 0.75
            
            if result['watermark_detected'] and self.compliance_operator:
                compliance_result = self.compliance_operator._validate({
                    "content_type": "image",
                    "jurisdiction": jurisdiction
                })
                if compliance_result.is_success():
                    result['compliance_status'] = compliance_result.output.get('compliance_status', 'unknown')
            
            self._update_stats('image', result['watermark_detected'])
            self._log_audit('image', result['watermark_detected'], start_time, jurisdiction)
            
        except Exception as e:
            result['error'] = str(e)
            _LOG.error(f"Image watermark detection failed: {e}")
            if self.strict_mode:
                raise
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def detect_audio_watermark(self, audio: torch.Tensor,
                               sample_rate: int = 44100,
                               jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
        """
        Detect watermark in audio content.
        
        Args:
            audio: Audio tensor (samples,) or (channels, samples)
            sample_rate: Audio sample rate
            jurisdiction: Target jurisdiction for compliance
            
        Returns:
            Detection result dictionary
        """
        start_time = time.time()
        
        result = {
            'watermark_detected': False,
            'watermark_info': None,
            'compliance_status': 'unknown',
            'detection_confidence': 0.0,
            'jurisdiction': jurisdiction,
            'content_type': 'audio',
            'processing_time': 0.0,
            'error': None
        }
        
        try:
            if not isinstance(audio, torch.Tensor):
                raise ValueError("Audio must be a torch.Tensor")
            
            if self.content_operator:
                extract_result = self.content_operator._extract_from_audio(audio)
                if extract_result.is_success():
                    output = extract_result.output
                    if output.get('watermark_found'):
                        result['watermark_detected'] = True
                        result['watermark_info'] = output.get('payload')
                        result['detection_confidence'] = 0.88
            else:
                fallback_result = self._fallback_audio_detection(audio, sample_rate)
                if fallback_result:
                    result['watermark_detected'] = True
                    result['watermark_info'] = fallback_result
                    result['detection_confidence'] = 0.70
            
            if result['watermark_detected'] and self.compliance_operator:
                compliance_result = self.compliance_operator._validate({
                    "content_type": "audio",
                    "jurisdiction": jurisdiction
                })
                if compliance_result.is_success():
                    result['compliance_status'] = compliance_result.output.get('compliance_status', 'unknown')
            
            self._update_stats('audio', result['watermark_detected'])
            self._log_audit('audio', result['watermark_detected'], start_time, jurisdiction)
            
        except Exception as e:
            result['error'] = str(e)
            _LOG.error(f"Audio watermark detection failed: {e}")
            if self.strict_mode:
                raise
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def detect_video_watermark(self, video: torch.Tensor,
                               jurisdiction: str = "GLOBAL",
                               frame_sample_rate: int = 30) -> Dict[str, Any]:
        """
        Detect watermark in video content using multi-frame analysis.
        
        This method performs comprehensive video watermark detection by:
        1. Sampling frames at specified intervals
        2. Detecting image watermarks in each frame
        3. Extracting audio track watermarks (if present)
        4. Aggregating results with temporal consistency analysis
        
        Args:
            video: Video tensor (T, C, H, W) or (B, T, C, H, W)
            jurisdiction: Target jurisdiction for compliance
            frame_sample_rate: Sample every N frames for detection
            
        Returns:
            Detection result dictionary with frame-level and aggregated results
        """
        start_time = time.time()
        
        result = {
            'watermark_detected': False,
            'watermark_info': None,
            'compliance_status': 'unknown',
            'detection_confidence': 0.0,
            'jurisdiction': jurisdiction,
            'content_type': 'video',
            'processing_time': 0.0,
            'frame_results': [],
            'audio_result': None,
            'aggregated_payload': None,
            'temporal_consistency': 0.0,
            'error': None
        }
        
        try:
            if not isinstance(video, torch.Tensor):
                raise ValueError("Video must be a torch.Tensor")
            
            if video.dim() == 5:
                video = video.squeeze(0)
            
            if video.dim() != 4:
                raise ValueError(f"Video tensor must be (T, C, H, W), got shape {video.shape}")
            
            num_frames = video.shape[0]
            frame_indices = list(range(0, num_frames, frame_sample_rate))
            
            frame_detections = []
            payload_votes = {}
            
            for idx in frame_indices:
                frame = video[idx]
                frame_result = self.detect_image_watermark(frame, jurisdiction)
                frame_result['frame_index'] = idx
                frame_detections.append(frame_result)
                
                if frame_result['watermark_detected'] and frame_result['watermark_info']:
                    payload_key = self._get_payload_signature(frame_result['watermark_info'])
                    payload_votes[payload_key] = payload_votes.get(payload_key, 0) + 1
            
            result['frame_results'] = frame_detections
            
            detected_frames = sum(1 for f in frame_detections if f['watermark_detected'])
            result['temporal_consistency'] = detected_frames / len(frame_indices) if frame_indices else 0.0
            
            if payload_votes:
                best_payload_sig = max(payload_votes, key=payload_votes.get)
                vote_ratio = payload_votes[best_payload_sig] / detected_frames if detected_frames > 0 else 0
                
                if vote_ratio >= 0.5:
                    for f in frame_detections:
                        if f['watermark_detected']:
                            sig = self._get_payload_signature(f['watermark_info'])
                            if sig == best_payload_sig:
                                result['watermark_detected'] = True
                                result['watermark_info'] = f['watermark_info']
                                result['aggregated_payload'] = f['watermark_info']
                                result['detection_confidence'] = result['temporal_consistency'] * 0.95
                                break
            
            if result['watermark_detected'] and self.compliance_operator:
                compliance_result = self.compliance_operator._validate({
                    "content_type": "video",
                    "jurisdiction": jurisdiction
                })
                if compliance_result.is_success():
                    result['compliance_status'] = compliance_result.output.get('compliance_status', 'unknown')
            
            self._update_stats('video', result['watermark_detected'])
            self._log_audit('video', result['watermark_detected'], start_time, jurisdiction, 
                           extra={'frames_analyzed': len(frame_indices), 'temporal_consistency': result['temporal_consistency']})
            
        except Exception as e:
            result['error'] = str(e)
            _LOG.error(f"Video watermark detection failed: {e}")
            if self.strict_mode:
                raise
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def detect_model_watermark(self, model: nn.Module,
                               expected_owner: Optional[str] = None,
                               jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
        """
        Detect watermark in model weights.
        
        Args:
            model: PyTorch model to analyze
            expected_owner: Expected owner identifier for verification
            jurisdiction: Target jurisdiction for compliance
            
        Returns:
            Detection result dictionary with verification details
        """
        start_time = time.time()
        
        result = {
            'watermark_detected': False,
            'model_verified': False,
            'owner_info': None,
            'owner_match': False,
            'tampering_detected': False,
            'verification_confidence': 0.0,
            'jurisdiction': jurisdiction,
            'content_type': 'model',
            'processing_time': 0.0,
            'error': None
        }
        
        try:
            if self.weight_operator:
                verify_result = self.weight_operator._verify({
                    "model": model,
                    "owner_id": expected_owner
                })
                if verify_result.is_success():
                    output = verify_result.output
                    result['watermark_detected'] = output.get('verified', False)
                    result['model_verified'] = output.get('verified', False)
                    result['verification_confidence'] = output.get('score', 0.0)
                    result['owner_info'] = expected_owner
                    if expected_owner:
                        result['owner_match'] = output.get('owner_match', False)
            else:
                analysis_result = self._analyze_model_weights(model)
                result.update(analysis_result)
            
            if result['watermark_detected'] and self.compliance_operator:
                compliance_result = self.compliance_operator._validate({
                    "content_type": "weight",
                    "jurisdiction": jurisdiction
                })
                if compliance_result.is_success():
                    result['compliance_status'] = compliance_result.output.get('compliance_status', 'unknown')
            
            self._update_stats('model', result['watermark_detected'])
            self._log_audit('model', result['watermark_detected'], start_time, jurisdiction,
                           extra={'model_params': sum(p.numel() for p in model.parameters())})
            
        except Exception as e:
            result['error'] = str(e)
            _LOG.error(f"Model watermark detection failed: {e}")
            if self.strict_mode:
                raise
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def detect_multimodal_watermark(self, content: Dict[str, Any],
                                    jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
        """
        Detect watermarks across multiple content types in a single call.
        
        Args:
            content: Dictionary with keys 'text', 'image', 'audio', 'video', 'model'
            jurisdiction: Target jurisdiction for compliance
            
        Returns:
            Aggregated detection results across all content types
        """
        start_time = time.time()
        
        result = {
            'watermark_detected': False,
            'detection_results': {},
            'compliance_status': 'unknown',
            'jurisdiction': jurisdiction,
            'content_type': 'multimodal',
            'processing_time': 0.0,
            'error': None
        }
        
        try:
            if 'text' in content and content['text']:
                result['detection_results']['text'] = self.detect_text_watermark(
                    content['text'], jurisdiction
                )
            
            if 'image' in content and content['image'] is not None:
                result['detection_results']['image'] = self.detect_image_watermark(
                    content['image'], jurisdiction
                )
            
            if 'audio' in content and content['audio'] is not None:
                result['detection_results']['audio'] = self.detect_audio_watermark(
                    content['audio'], 
                    content.get('sample_rate', 44100),
                    jurisdiction
                )
            
            if 'video' in content and content['video'] is not None:
                result['detection_results']['video'] = self.detect_video_watermark(
                    content['video'], jurisdiction
                )
            
            if 'model' in content and content['model'] is not None:
                result['detection_results']['model'] = self.detect_model_watermark(
                    content['model'],
                    content.get('expected_owner'),
                    jurisdiction
                )
            
            detected_any = any(
                r.get('watermark_detected', False) 
                for r in result['detection_results'].values()
            )
            result['watermark_detected'] = detected_any
            
            compliant_count = sum(
                1 for r in result['detection_results'].values()
                if r.get('compliance_status') == 'compliant'
            )
            total_checked = len(result['detection_results'])
            result['compliance_status'] = 'compliant' if compliant_count == total_checked else 'partial' if compliant_count > 0 else 'unknown'
            
        except Exception as e:
            result['error'] = str(e)
            _LOG.error(f"Multimodal watermark detection failed: {e}")
            if self.strict_mode:
                raise
        
        result['processing_time'] = time.time() - start_time
        return result
    
    def _get_payload_signature(self, payload: Any) -> str:
        """Generate a signature for payload comparison."""
        if isinstance(payload, dict):
            return json.dumps(payload, sort_keys=True, default=str)[:128]
        return str(payload)[:128]
    
    def _update_stats(self, content_type: str, detected: bool):
        """Update detection statistics."""
        self.detection_stats['total_detections'] += 1
        self.detection_stats['by_type'][content_type] = self.detection_stats['by_type'].get(content_type, 0) + 1
        if detected:
            self.detection_stats['successful_detections'] += 1
        else:
            self.detection_stats['failed_detections'] += 1
    
    def _log_audit(self, content_type: str, detected: bool, start_time: float,
                   jurisdiction: str, extra: Optional[Dict] = None):
        """Log audit trail for detection operation."""
        if self.audit_operator:
            try:
                self.audit_operator._log_operation({
                    "operation": "detect",
                    "content_type": content_type,
                    "result": "success" if detected else "not_found",
                    "jurisdiction": jurisdiction,
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "processing_time": time.time() - start_time,
                        **(extra or {})
                    }
                })
            except Exception:
                pass
    
    def _fallback_text_detection(self, text: str) -> Optional[Dict[str, Any]]:
        """Fallback text watermark detection using zero-width characters."""
        try:
            symbols = []
            for ch in text:
                if ch in self.reverse_mapping:
                    symbols.append(self.reverse_mapping[ch])
            
            if not symbols:
                return None
            
            bitstream = ''.join(symbols)
            binary_data = ''.join(['0' if b == '0' else '1' for b in bitstream])
            
            byte_data = bytearray()
            for i in range(0, len(binary_data), 8):
                byte_chunk = binary_data[i:i+8]
                if len(byte_chunk) == 8:
                    byte_data.append(int(byte_chunk, 2))
            
            import zlib
            decompressed = zlib.decompress(bytes(byte_data))
            payload_str = decompressed.decode('utf-8')
            payload = json.loads(payload_str)
            
            return {
                'valid': True,
                'payload': payload,
                'extraction_method': 'fallback_zero_width',
                'confidence': 0.85
            }
            
        except Exception as e:
            _LOG.debug(f"Fallback text detection failed: {e}")
            return None
    
    def _fallback_image_detection(self, image: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Fallback image watermark detection using DCT analysis."""
        try:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            if self.dct_operator:
                extract_result = self.dct_operator._extract({
                    "image": image.squeeze(0),
                    "num_bits": 256,
                    "seed": 12345,
                    "band": "mid"
                })
                if extract_result.is_success():
                    bits = extract_result.output.get("bits", "")
                    if bits:
                        return {
                            'valid': True,
                            'bitstream': bits,
                            'extraction_method': 'dct_fallback',
                            'confidence': 0.75
                        }
            return None
        except Exception as e:
            _LOG.debug(f"Fallback image detection failed: {e}")
            return None
    
    def _fallback_audio_detection(self, audio: torch.Tensor, sample_rate: int) -> Optional[Dict[str, Any]]:
        """Fallback audio watermark detection using STFT analysis."""
        try:
            min_freq = 18000
            max_freq = 20000
            
            x = audio.flatten()
            n_fft = 1024
            hop = n_fft // 4
            win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
            X = torch.stft(x, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
            
            freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(x.device)
            band_mask = (freqs >= min_freq) & (freqs <= max_freq)
            idx_band = torch.where(band_mask)[0]
            
            if idx_band.numel() == 0:
                return None
            
            energy = torch.abs(X[idx_band, :]).mean()
            if energy > 0.001:
                return {
                    'valid': True,
                    'ultrasonic_energy': float(energy),
                    'extraction_method': 'stft_fallback',
                    'confidence': 0.70
                }
            return None
        except Exception as e:
            _LOG.debug(f"Fallback audio detection failed: {e}")
            return None
    
    def _analyze_model_weights(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model weights for watermark signatures."""
        try:
            all_weights = []
            for param in model.parameters():
                if param.requires_grad:
                    all_weights.extend(param.data.flatten().tolist())
            
            if not all_weights:
                return {'watermark_detected': False, 'error': 'No trainable parameters'}
            
            weights_array = np.array(all_weights)
            
            mean_weight = np.mean(weights_array)
            std_weight = np.std(weights_array)
            skewness = np.mean(((weights_array - mean_weight) / (std_weight + 1e-8)) ** 3)
            
            watermark_score = self._calculate_watermark_score(mean_weight, std_weight, skewness)
            
            return {
                'watermark_detected': watermark_score > 0.7,
                'verification_confidence': watermark_score,
                'weight_statistics': {
                    'mean': float(mean_weight),
                    'std': float(std_weight),
                    'skewness': float(skewness)
                },
                'analysis_method': 'statistical_heuristics'
            }
            
        except Exception as e:
            _LOG.debug(f"Model weight analysis failed: {e}")
            return {'watermark_detected': False, 'error': str(e)}
    
    def _calculate_watermark_score(self, mean: float, std: float, skewness: float) -> float:
        """Calculate watermark confidence score from weight statistics."""
        score = 0.0
        
        if abs(mean) < 0.1:
            score += 0.3
        
        if 0.05 < std < 0.3:
            score += 0.3
        
        if abs(skewness) < 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def batch_detect(self, items: List[Any], 
                     content_type: str = "text",
                     jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
        """
        Batch detect watermarks across multiple items.
        
        Args:
            items: List of content items to analyze
            content_type: Type of content ('text', 'image', 'audio', 'video', 'model')
            jurisdiction: Target jurisdiction for compliance
            
        Returns:
            Batch detection results with summary statistics
        """
        results = []
        start_time = time.time()
        
        detect_method = {
            'text': self.detect_text_watermark,
            'image': self.detect_image_watermark,
            'audio': lambda x: self.detect_audio_watermark(x, jurisdiction=jurisdiction),
            'video': lambda x: self.detect_video_watermark(x, jurisdiction=jurisdiction),
            'model': lambda x: self.detect_model_watermark(x, jurisdiction=jurisdiction)
        }.get(content_type, self.detect_text_watermark)
        
        for i, item in enumerate(items):
            if content_type == 'text':
                detection_result = detect_method(item, jurisdiction)
            else:
                detection_result = detect_method(item)
            
            detection_result['batch_index'] = i
            if content_type == 'text' and isinstance(item, str):
                detection_result['item_preview'] = item[:50] + "..." if len(item) > 50 else item
            results.append(detection_result)
        
        total_items = len(results)
        detected_items = sum(1 for r in results if r.get('watermark_detected'))
        compliant_items = sum(1 for r in results if r.get('compliance_status') == 'compliant')
        
        summary = {
            'total_items': total_items,
            'detected_items': detected_items,
            'compliant_items': compliant_items,
            'detection_rate': detected_items / total_items if total_items > 0 else 0,
            'compliance_rate': compliant_items / total_items if total_items > 0 else 0,
            'individual_results': results,
            'processing_time': time.time() - start_time,
            'jurisdiction': jurisdiction,
            'content_type': content_type
        }
        
        _LOG.info(f"Batch detection completed: {detected_items}/{total_items} items detected")
        return summary
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        total_attempts = self.detection_stats['total_detections']
        success_rate = (
            self.detection_stats['successful_detections'] / max(total_attempts, 1)
        )
        compliance_total = self.detection_stats['compliance_passed'] + self.detection_stats['compliance_failed']
        compliance_rate = (
            self.detection_stats['compliance_passed'] / max(compliance_total, 1)
        )
        
        return {
            'total_detections': total_attempts,
            'successful_detections': self.detection_stats['successful_detections'],
            'failed_detections': self.detection_stats['failed_detections'],
            'success_rate': success_rate,
            'compliance_passed': self.detection_stats['compliance_passed'],
            'compliance_failed': self.detection_stats['compliance_failed'],
            'compliance_rate': compliance_rate,
            'by_content_type': self.detection_stats['by_type']
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        return isinstance(inputs, dict)
    
    def transform(self, data: Any) -> Any:
        if not isinstance(data, dict):
            raise TypeError("WMC watermark detection expects a dict input")
        
        mode = data.get("mode") or data.get("type") or "text"
        jurisdiction = str(data.get("jurisdiction", "GLOBAL"))
        
        if mode == "text":
            return self.detect_text_watermark(
                text=str(data.get("text", "")),
                jurisdiction=jurisdiction
            )
        elif mode == "image":
            return self.detect_image_watermark(
                image=data.get("image"),
                jurisdiction=jurisdiction
            )
        elif mode == "audio":
            return self.detect_audio_watermark(
                audio=data.get("audio"),
                sample_rate=data.get("sample_rate", 44100),
                jurisdiction=jurisdiction
            )
        elif mode == "video":
            return self.detect_video_watermark(
                video=data.get("video"),
                jurisdiction=jurisdiction,
                frame_sample_rate=data.get("frame_sample_rate", 30)
            )
        elif mode in ("model", "weights"):
            return self.detect_model_watermark(
                model=data.get("model"),
                expected_owner=data.get("expected_owner"),
                jurisdiction=jurisdiction
            )
        elif mode == "multimodal":
            return self.detect_multimodal_watermark(
                content=data.get("content", {}),
                jurisdiction=jurisdiction
            )
        
        raise ValueError(f"Unsupported detection mode: {mode}")


@PiscesLxOperatorRegistrar()
class PiscesLxModelWatermarkVerificationOperator(PiscesLxTransformOperator):
    """
    Model Watermark Verification Operator
    
    Specialized operator for deep verification of model weight watermarks
    with configurable sensitivity levels.
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, 
                 sensitivity_level: str = "medium"):
        super().__init__(config)
        if config is not None:
            params = getattr(config, "parameters", {}) or {}
            sensitivity_level = str(params.get("sensitivity_level", sensitivity_level))
        self.sensitivity_level = sensitivity_level
        self.verification_thresholds = self._get_sensitivity_thresholds()
        self.weight_operator = None
        
        try:
            from opss.watermark import POPSSWatermarkWeightOperator
            self.weight_operator = POPSSWatermarkWeightOperator()
            _LOG.info("Weight operator loaded for model verification")
        except ImportError:
            _LOG.warning("Weight operator not available, using heuristic methods")
        
        _LOG.info(f"Model watermark verification operator initialized with {sensitivity_level} sensitivity")
    
    def _get_sensitivity_thresholds(self) -> Dict[str, float]:
        thresholds = {
            'low': {'correlation': 0.6, 'confidence': 0.7},
            'medium': {'correlation': 0.75, 'confidence': 0.85},
            'high': {'correlation': 0.9, 'confidence': 0.95}
        }
        return thresholds.get(self.sensitivity_level, thresholds['medium'])
    
    def verify_model_integrity(self, model: nn.Module,
                               reference_signature: Optional[bytes] = None,
                               owner_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify model integrity and ownership.
        
        Args:
            model: PyTorch model to verify
            reference_signature: Expected model signature for comparison
            owner_id: Expected owner identifier
            
        Returns:
            Verification result with integrity status and confidence
        """
        result = {
            'integrity_verified': False,
            'tampering_detected': False,
            'confidence_score': 0.0,
            'owner_verified': False,
            'anomalies': [],
            'error': None
        }
        
        try:
            if self.weight_operator and owner_id:
                verify_result = self.weight_operator._verify({
                    "model": model,
                    "owner_id": owner_id
                })
                if verify_result.is_success():
                    output = verify_result.output
                    result['integrity_verified'] = output.get('verified', False)
                    result['confidence_score'] = output.get('score', 0.0)
                    result['owner_verified'] = output.get('owner_match', False)
                    return result
            
            model_fingerprint = self._compute_model_fingerprint(model)
            
            if reference_signature:
                correlation = self._compare_signatures(model_fingerprint, reference_signature)
                result['confidence_score'] = correlation
                
                if correlation >= self.verification_thresholds['correlation']:
                    result['integrity_verified'] = True
                else:
                    result['tampering_detected'] = True
                    result['anomalies'].append('signature_mismatch')
            else:
                heuristic_result = self._heuristic_integrity_check(model_fingerprint)
                result.update(heuristic_result)
                
        except Exception as e:
            result['error'] = str(e)
            _LOG.error(f"Model integrity verification failed: {e}")
        
        return result
    
    def _compute_model_fingerprint(self, model: nn.Module) -> bytes:
        """Compute model fingerprint from key layer weights."""
        import hashlib
        
        key_layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                if hasattr(module, 'weight') and module.weight is not None:
                    key_layers.append(module.weight.data.cpu().numpy())
        
        fingerprint_data = b""
        for layer_weights in key_layers[:10]:
            mean = np.mean(layer_weights)
            std = np.std(layer_weights)
            fingerprint_data += struct.pack('ff', float(mean), float(std))
        
        return hashlib.sha256(fingerprint_data).digest()
    
    def _compare_signatures(self, current: bytes, reference: bytes) -> float:
        """Compare signature similarity using Hamming distance."""
        if len(current) != len(reference):
            return 0.0
        
        matches = sum(c == r for c, r in zip(current, reference))
        return matches / len(current)
    
    def _heuristic_integrity_check(self, fingerprint: bytes) -> Dict[str, Any]:
        """Perform heuristic integrity check on fingerprint."""
        fingerprint_array = np.frombuffer(fingerprint, dtype=np.uint8)
        
        mean_byte = np.mean(fingerprint_array)
        std_byte = np.std(fingerprint_array)
        
        confidence = 0.5
        
        if 100 < mean_byte < 156:
            confidence += 0.2
        
        if std_byte > 20:
            confidence += 0.2
        
        entropy = self._calculate_entropy(fingerprint_array)
        if entropy > 6.0:
            confidence += 0.1
        
        confidence = min(confidence, 1.0)
        
        return {
            'integrity_verified': confidence >= self.verification_thresholds['confidence'],
            'confidence_score': confidence,
            'heuristic_checks': {
                'byte_mean': float(mean_byte),
                'byte_std': float(std_byte),
                'entropy': entropy
            }
        }
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        from collections import Counter
        
        counter = Counter(data)
        total = len(data)
        probabilities = [count / total for count in counter.values()]
        
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        return isinstance(inputs, dict)
    
    def transform(self, data: Any) -> Any:
        if not isinstance(data, dict):
            raise TypeError("WMC model verification expects a dict input")
        model = data.get("model")
        if model is None:
            raise ValueError("Missing 'model' for model integrity verification")
        return self.verify_model_integrity(
            model=model, 
            reference_signature=data.get("reference_signature"),
            owner_id=data.get("owner_id")
        )


@PiscesLxOperatorRegistrar()
class PiscesLxContentIntegrityCheckOperator(PiscesLxTransformOperator):
    """
    Content Integrity Check Operator
    
    Detects content tampering, corruption, and suspicious modifications
    across text, binary, and tensor content types.
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, 
                 check_depth: str = "standard"):
        super().__init__(config)
        if config is not None:
            params = getattr(config, "parameters", {}) or {}
            check_depth = str(params.get("check_depth", check_depth))
        self.check_depth = check_depth
        self.integrity_patterns = self._load_integrity_patterns()
        
        _LOG.info(f"Content integrity check operator initialized with {check_depth} depth")
    
    def _load_integrity_patterns(self) -> Dict[str, Any]:
        """Load integrity check patterns."""
        return {
            'suspicious_patterns': [
                r'<script[^>]*>',
                r'javascript:',
                r'on\w+\s*=',
                r'data:text/html',
            ],
            'structural_indicators': [
                'inconsistent_formatting',
                'unexpected_encoding',
                'metadata_mismatch'
            ]
        }
    
    def check_content_integrity(self, content: Union[str, bytes, torch.Tensor],
                                content_type: str = "text") -> Dict[str, Any]:
        """
        Check content integrity for tampering or corruption.
        
        Args:
            content: Content to analyze
            content_type: Type of content ('text', 'binary', 'tensor')
            
        Returns:
            Integrity check result with detected anomalies
        """
        result = {
            'integrity_verified': True,
            'tampering_detected': False,
            'suspicious_elements': [],
            'confidence_score': 1.0,
            'checks_performed': [],
            'error': None
        }
        
        try:
            if content_type == "text":
                text_content = content if isinstance(content, str) else content.decode('utf-8')
                self._check_text_integrity(text_content, result)
            elif content_type == "binary":
                self._check_binary_integrity(content, result)
            elif content_type == "tensor":
                self._check_tensor_integrity(content, result)
            
            suspicious_count = len(result['suspicious_elements'])
            if suspicious_count > 0:
                result['integrity_verified'] = False
                result['tampering_detected'] = True
                result['confidence_score'] = max(0.0, 1.0 - (suspicious_count * 0.1))
            
        except Exception as e:
            result['error'] = str(e)
            result['integrity_verified'] = False
            _LOG.error(f"Content integrity check failed: {e}")
        
        return result
    
    def _check_text_integrity(self, text: str, result: Dict[str, Any]):
        """Check text content integrity."""
        import re
        
        for pattern in self.integrity_patterns['suspicious_patterns']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                result['suspicious_elements'].extend(matches)
                result['checks_performed'].append(f'pattern_check_{pattern}')
        
        if self.check_depth in ['deep', 'standard']:
            structural_issues = self._check_structural_integrity(text)
            result['suspicious_elements'].extend(structural_issues)
            result['checks_performed'].append('structural_analysis')
    
    def _check_structural_integrity(self, text: str) -> List[str]:
        """Check structural integrity of text."""
        issues = []
        
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            issues.append('non_ascii_characters')
        
        lines = text.split('\n')
        if len(lines) > 1:
            line_lengths = [len(line.strip()) for line in lines if line.strip()]
            if line_lengths:
                avg_length = sum(line_lengths) / len(line_lengths)
                length_variance = sum((l - avg_length) ** 2 for l in line_lengths) / len(line_lengths)
                if length_variance > (avg_length * 0.5) ** 2:
                    issues.append('inconsistent_line_lengths')
        
        return issues
    
    def _check_binary_integrity(self, content: bytes, result: Dict[str, Any]):
        """Check binary content integrity."""
        if len(content) >= 4:
            header = content[:4]
            known_signatures = {
                b'\x89PNG': 'PNG image',
                b'\xFF\xD8\xFF': 'JPEG image',
                b'RIFF': 'RIFF container'
            }
            
            matched_signature = None
            for sig, desc in known_signatures.items():
                if content.startswith(sig):
                    matched_signature = desc
                    break
            
            if not matched_signature:
                result['suspicious_elements'].append('unknown_file_format')
            
            result['checks_performed'].append('file_header_analysis')
    
    def _check_tensor_integrity(self, tensor: torch.Tensor, result: Dict[str, Any]):
        """Check tensor content integrity."""
        if torch.isnan(tensor).any():
            result['suspicious_elements'].append('nan_values')
        
        if torch.isinf(tensor).any():
            result['suspicious_elements'].append('inf_values')
        
        result['checks_performed'].append('tensor_value_check')
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        return isinstance(inputs, dict)
    
    def transform(self, data: Any) -> Any:
        if not isinstance(data, dict):
            raise TypeError("WMC content integrity check expects a dict input")
        content = data.get("content")
        if content is None:
            raise ValueError("Missing 'content' for content integrity check")
        return self.check_content_integrity(
            content=content,
            content_type=str(data.get("content_type", "text"))
        )


__all__ = [
    "PiscesLxWatermarkDetectionOperator",
    "PiscesLxModelWatermarkVerificationOperator",
    "PiscesLxContentIntegrityCheckOperator"
]
