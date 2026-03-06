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
PiscesLx WMC Check API

This module provides high-level API functions for watermark detection
across text, image, audio, video, and model weight content.

API Functions:
    - detect_watermark: Detect watermark in text content
    - batch_detect: Batch detect watermarks from file
    - detect_image_watermark: Detect watermark from image file
    - detect_audio_watermark: Detect watermark from audio file
    - detect_video_watermark: Detect watermark from video tensor
    - detect_model_watermark: Detect watermark in model weights
    - detect_multimodal_watermark: Detect watermarks across multiple content types

Usage Examples:
    >>> from tools.wmc.check import detect_watermark, detect_video_watermark
    >>> 
    >>> # Detect text watermark
    >>> result = detect_watermark("AI-generated text...", verbose=True)
    >>> 
    >>> # Detect video watermark
    >>> video_result = detect_video_watermark(video_tensor, verbose=True)

Author: PiscesL1 Development Team
Version: 1.0.0
"""

import os
import json
from typing import Dict, Any, Optional, List, Union

from utils.dc import PiscesLxLogger
from .core import PiscesLxWatermarkDetectionOperator

from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Wmc", file_path=get_log_file("PiscesLx.Tools.Wmc"), enable_file=True)


def _validate_detect_args(text: str, verbose: bool):
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("text must be a non-empty string")
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")


def _validate_batch_detect_args(file_path: str, verbose: bool):
    if not isinstance(file_path, str) or file_path.strip() == "":
        raise ValueError("file_path must be a non-empty string")
    if not os.path.exists(file_path):
        raise ValueError(f"file_path not found: {file_path}")
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")


def detect_watermark(text: str, 
                     verbose: bool = False,
                     jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
    """
    Detect hidden watermark information in the specified text.
    
    Args:
        text: Text content to analyze for watermarks
        verbose: Enable verbose logging output
        jurisdiction: Target jurisdiction for compliance (CN, EU, US, UK, JP, KR, GLOBAL)
        
    Returns:
        Dictionary containing:
            - watermark_detected: bool - Whether watermark was found
            - watermark_info: Optional[dict] - Extracted watermark payload
            - compliance_status: str - Compliance status (unknown/compliant/non_compliant/error)
            - detection_confidence: float - Detection confidence score
            - error: Optional[str] - Error message if detection failed
            
    Example:
        >>> result = detect_watermark("AI-generated content...", verbose=True)
        >>> if result["watermark_detected"]:
        ...     print(f"Found watermark: {result['watermark_info']}")
    """
    _validate_detect_args(text, verbose)

    result: Dict[str, Any] = {
        "watermark_detected": False,
        "watermark_info": None,
        "compliance_status": "unknown",
        "detection_confidence": 0.0,
        "error": None,
    }

    try:
        detector = PiscesLxWatermarkDetectionOperator(enable_compliance_check=True)
        det = detector.detect_text_watermark(text, jurisdiction=jurisdiction)
        result["watermark_detected"] = bool(det.get("watermark_detected"))
        result["watermark_info"] = det.get("watermark_info")
        result["compliance_status"] = det.get("compliance_status", "unknown")
        result["detection_confidence"] = det.get("detection_confidence", 0.0)
        if verbose:
            _LOG.info("wmc.detect", watermark_detected=result["watermark_detected"], 
                       compliance_status=result["compliance_status"])
    except Exception as e:
        result["error"] = str(e)
        result["compliance_status"] = "error"
        if verbose:
            _LOG.error("wmc.detect_error", error=str(e))

    return result


def batch_detect(file_path: str, 
                 verbose: bool = False,
                 jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
    """
    Batch detect watermarks line-by-line in a text file.
    
    Args:
        file_path: Path to text file containing content to analyze
        verbose: Enable verbose logging output
        jurisdiction: Target jurisdiction for compliance
        
    Returns:
        Dictionary containing:
            - total_lines: int - Total lines processed
            - detected_lines: int - Lines with detected watermarks
            - compliant_lines: int - Lines with compliant watermarks
            - detection_rate: float - Detection rate (0.0-1.0)
            - compliance_rate: float - Compliance rate (0.0-1.0)
            - detailed_results: List[Dict] - Per-line detection results
            
    Example:
        >>> result = batch_detect("output.txt", verbose=True)
        >>> print(f"Detection rate: {result['detection_rate']:.2%}")
    """
    _validate_batch_detect_args(file_path, verbose)

    detected_lines = 0
    compliant_lines = 0
    total_lines = 0
    results = []

    detector = PiscesLxWatermarkDetectionOperator(enable_compliance_check=True)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            text = line.strip()
            if not text:
                continue
            det = detector.detect_text_watermark(text, jurisdiction=jurisdiction)
            payload = det.get("watermark_info")
            detected = bool(det.get("watermark_detected"))
            compliant = False
            std = None
            if detected:
                detected_lines += 1
                if isinstance(payload, dict):
                    std = payload.get("standard") or payload.get("compliance_status")
                compliant = det.get("compliance_status") == "compliant"
                if compliant:
                    compliant_lines += 1
            results.append({
                "line": total_lines,
                "detected": detected,
                "compliant": compliant,
                "standard": std,
            })

    summary = {
        "total_lines": total_lines,
        "detected_lines": detected_lines,
        "compliant_lines": compliant_lines,
        "detection_rate": (detected_lines / total_lines) if total_lines else 0.0,
        "compliance_rate": (compliant_lines / total_lines) if total_lines else 0.0,
        "detailed_results": results,
    }

    if verbose:
        _LOG.info("Batch Detection Results")
        print(f"\tTotal Lines: {total_lines}")
        print(f"\tLines with Watermark Detected: {detected_lines}")
        print(f"\tCompliant Lines: {compliant_lines}")
        print(f"\tDetection Rate: {summary['detection_rate']:.2%}")
        print(f"\tCompliance Rate: {summary['compliance_rate']:.2%}")

    return summary


def _load_image_as_tensor(path: str):
    """Load image file as tensor for watermark detection."""
    try:
        from PIL import Image
        import torch
        import numpy as np
        img = Image.open(path).convert('RGB')
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return t
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return None


def _load_audio_as_tensor(path: str):
    """Load audio file as tensor for watermark detection."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        return wav, int(sr)
    except Exception:
        try:
            import wave
            import struct
            import torch
            import numpy as np
            import array
            
            with wave.open(path, 'rb') as wf:
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                raw = wf.readframes(n_frames)
                fmt = {1:'b', 2:'h', 4:'i'}.get(sampwidth)
                if fmt is None:
                    raise RuntimeError("Unsupported sample width")
                arr = array.array(fmt, raw)
                np_arr = np.asarray(arr, dtype=float)
                if n_channels > 1:
                    np_arr = np_arr.reshape(-1, n_channels).T
                else:
                    np_arr = np_arr.reshape(1, -1)
                wav = torch.from_numpy(np_arr).float() / (2**(8*sampwidth-1))
                return wav, int(framerate)
        except Exception as e:
            _LOG.error(f"Failed to load audio: {e}")
            return None, None


def _load_video_as_tensor(path: str, max_frames: int = 300):
    """Load video file as tensor for watermark detection."""
    try:
        import torch
        import numpy as np
        
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            frames = []
            frame_count = 0
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frames.append(frame_tensor)
                frame_count += 1
            
            cap.release()
            
            if frames:
                video_tensor = torch.stack(frames)
                return video_tensor, frame_count
            return None, 0
            
        except ImportError:
            try:
                import imageio
                reader = imageio.get_reader(path)
                frames = []
                frame_count = 0
                
                for frame in reader:
                    if frame_count >= max_frames:
                        break
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(frame_tensor)
                    frame_count += 1
                
                reader.close()
                
                if frames:
                    video_tensor = torch.stack(frames)
                    return video_tensor, frame_count
                return None, 0
                
            except ImportError:
                _LOG.error("No video library available (install opencv-python or imageio)")
                return None, 0
                
    except Exception as e:
        _LOG.error(f"Failed to load video: {e}")
        return None, 0


def detect_image_watermark(image_path: str, 
                           verbose: bool = False,
                           jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
    """
    Detect watermark payload from an image file.
    
    Args:
        image_path: Path to image file (PNG, JPEG, etc.)
        verbose: Enable verbose logging output
        jurisdiction: Target jurisdiction for compliance
        
    Returns:
        Dictionary containing:
            - file: str - Image file path
            - watermark_detected: bool - Whether watermark was found
            - watermark_info: Optional[dict] - Extracted watermark payload
            - detection_confidence: float - Detection confidence score
            - compliance_status: str - Compliance status
            - error: Optional[str] - Error message if detection failed
            
    Example:
        >>> result = detect_image_watermark("generated_image.png", verbose=True)
        >>> if result["watermark_detected"]:
        ...     print(f"Watermark found with confidence: {result['detection_confidence']}")
    """
    img_t = _load_image_as_tensor(image_path)
    result = {
        "file": image_path, 
        "watermark_detected": False, 
        "watermark_info": None,
        "detection_confidence": 0.0,
        "compliance_status": "unknown",
        "error": None
    }
    
    if img_t is None:
        result["error"] = "image_load_failed"
        return result
    
    try:
        detector = PiscesLxWatermarkDetectionOperator(enable_compliance_check=True)
        det_result = detector.detect_image_watermark(img_t, jurisdiction=jurisdiction)
        result["watermark_detected"] = det_result.get("watermark_detected", False)
        result["watermark_info"] = det_result.get("watermark_info")
        result["detection_confidence"] = det_result.get("detection_confidence", 0.0)
        result["compliance_status"] = det_result.get("compliance_status", "unknown")
        
        if verbose:
            if result["watermark_detected"]:
                _LOG.info("Image watermark detected", event="wmc.img.detect")
            else:
                _LOG.info("No image watermark detected")
                
    except Exception as e:
        result["error"] = str(e)
        
    return result


def detect_audio_watermark(audio_path: str, 
                           verbose: bool = False,
                           jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
    """
    Detect watermark payload from an audio file.
    
    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        verbose: Enable verbose logging output
        jurisdiction: Target jurisdiction for compliance
        
    Returns:
        Dictionary containing:
            - file: str - Audio file path
            - watermark_detected: bool - Whether watermark was found
            - watermark_info: Optional[dict] - Extracted watermark payload
            - detection_confidence: float - Detection confidence score
            - compliance_status: str - Compliance status
            - sample_rate: int - Audio sample rate
            - error: Optional[str] - Error message if detection failed
            
    Example:
        >>> result = detect_audio_watermark("generated_audio.wav", verbose=True)
        >>> if result["watermark_detected"]:
        ...     print(f"Watermark found: {result['watermark_info']}")
    """
    wav, sr = _load_audio_as_tensor(audio_path)
    result = {
        "file": audio_path, 
        "watermark_detected": False, 
        "watermark_info": None,
        "detection_confidence": 0.0,
        "compliance_status": "unknown",
        "sample_rate": sr,
        "error": None
    }
    
    if wav is None or sr is None:
        result["error"] = "audio_load_failed"
        return result
    
    try:
        detector = PiscesLxWatermarkDetectionOperator(enable_compliance_check=True)
        det_result = detector.detect_audio_watermark(wav, sample_rate=sr, jurisdiction=jurisdiction)
        result["watermark_detected"] = det_result.get("watermark_detected", False)
        result["watermark_info"] = det_result.get("watermark_info")
        result["detection_confidence"] = det_result.get("detection_confidence", 0.0)
        result["compliance_status"] = det_result.get("compliance_status", "unknown")
        
        if verbose:
            if result["watermark_detected"]:
                _LOG.info("Audio watermark detected", event="wmc.audio.detect")
            else:
                _LOG.info("No audio watermark detected")
                
    except Exception as e:
        result["error"] = str(e)
        
    return result


def detect_video_watermark(video_path: str,
                           verbose: bool = False,
                           jurisdiction: str = "GLOBAL",
                           frame_sample_rate: int = 30,
                           max_frames: int = 300) -> Dict[str, Any]:
    """
    Detect watermark payload from a video file.
    
    This function performs multi-frame video watermark detection by:
    1. Loading video frames at specified sample rate
    2. Detecting image watermarks in sampled frames
    3. Aggregating results with temporal consistency analysis
    
    Args:
        video_path: Path to video file (MP4, AVI, etc.)
        verbose: Enable verbose logging output
        jurisdiction: Target jurisdiction for compliance
        frame_sample_rate: Sample every N frames for detection
        max_frames: Maximum number of frames to process
        
    Returns:
        Dictionary containing:
            - file: str - Video file path
            - watermark_detected: bool - Whether watermark was found
            - watermark_info: Optional[dict] - Extracted watermark payload
            - detection_confidence: float - Detection confidence score
            - compliance_status: str - Compliance status
            - temporal_consistency: float - Consistency across frames
            - frames_analyzed: int - Number of frames analyzed
            - frame_results: List[Dict] - Per-frame detection results
            - error: Optional[str] - Error message if detection failed
            
    Example:
        >>> result = detect_video_watermark("generated_video.mp4", verbose=True)
        >>> if result["watermark_detected"]:
        ...     print(f"Temporal consistency: {result['temporal_consistency']:.2%}")
    """
    video_tensor, frame_count = _load_video_as_tensor(video_path, max_frames)
    
    result = {
        "file": video_path,
        "watermark_detected": False,
        "watermark_info": None,
        "detection_confidence": 0.0,
        "compliance_status": "unknown",
        "temporal_consistency": 0.0,
        "frames_analyzed": 0,
        "frame_results": [],
        "error": None
    }
    
    if video_tensor is None:
        result["error"] = "video_load_failed"
        return result
    
    try:
        detector = PiscesLxWatermarkDetectionOperator(enable_compliance_check=True)
        det_result = detector.detect_video_watermark(
            video_tensor, 
            jurisdiction=jurisdiction,
            frame_sample_rate=frame_sample_rate
        )
        
        result["watermark_detected"] = det_result.get("watermark_detected", False)
        result["watermark_info"] = det_result.get("watermark_info")
        result["detection_confidence"] = det_result.get("detection_confidence", 0.0)
        result["compliance_status"] = det_result.get("compliance_status", "unknown")
        result["temporal_consistency"] = det_result.get("temporal_consistency", 0.0)
        result["frames_analyzed"] = len(det_result.get("frame_results", []))
        result["frame_results"] = det_result.get("frame_results", [])
        
        if verbose:
            if result["watermark_detected"]:
                _LOG.info(
                    f"Video watermark detected in {result['frames_analyzed']} frames",
                    event="wmc.video.detect"
                )
            else:
                _LOG.info("No video watermark detected")
                
    except Exception as e:
        result["error"] = str(e)
        
    return result


def detect_model_watermark(model_path: str,
                           verbose: bool = False,
                           expected_owner: Optional[str] = None,
                           jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
    """
    Detect watermark in model weights from a checkpoint file.
    
    Args:
        model_path: Path to model checkpoint file (.pt, .pth, .bin)
        verbose: Enable verbose logging output
        expected_owner: Expected owner identifier for verification
        jurisdiction: Target jurisdiction for compliance
        
    Returns:
        Dictionary containing:
            - file: str - Model file path
            - watermark_detected: bool - Whether watermark was found
            - model_verified: bool - Whether model integrity verified
            - owner_match: bool - Whether owner matches expected
            - verification_confidence: float - Verification confidence score
            - compliance_status: str - Compliance status
            - error: Optional[str] - Error message if detection failed
            
    Example:
        >>> result = detect_model_watermark("model.pt", expected_owner="PiscesL1", verbose=True)
        >>> if result["watermark_detected"]:
        ...     print(f"Owner verified: {result['owner_match']}")
    """
    result = {
        "file": model_path,
        "watermark_detected": False,
        "model_verified": False,
        "owner_match": False,
        "verification_confidence": 0.0,
        "compliance_status": "unknown",
        "error": None
    }
    
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        class DummyModel(torch.nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                for key, value in state_dict.items():
                    setattr(self, key.replace('.', '_'), torch.nn.Parameter(value))
        
        model = DummyModel(state_dict)
        
        detector = PiscesLxWatermarkDetectionOperator(enable_compliance_check=True)
        det_result = detector.detect_model_watermark(
            model, 
            expected_owner=expected_owner,
            jurisdiction=jurisdiction
        )
        
        result["watermark_detected"] = det_result.get("watermark_detected", False)
        result["model_verified"] = det_result.get("model_verified", False)
        result["owner_match"] = det_result.get("owner_match", False)
        result["verification_confidence"] = det_result.get("verification_confidence", 0.0)
        result["compliance_status"] = det_result.get("compliance_status", "unknown")
        
        if verbose:
            if result["watermark_detected"]:
                logger.info(
                    f"Model watermark detected with confidence: {result['verification_confidence']:.2f}",
                    event="wmc.model.detect"
                )
            else:
                logger.info("No model watermark detected")
                
    except Exception as e:
        result["error"] = str(e)
        
    return result


def detect_multimodal_watermark(content: Dict[str, Any],
                                 verbose: bool = False,
                                 jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
    """
    Detect watermarks across multiple content types in a single call.
    
    Args:
        content: Dictionary with keys 'text', 'image', 'audio', 'video', 'model'
                 Each value should be the appropriate content or file path
        verbose: Enable verbose logging output
        jurisdiction: Target jurisdiction for compliance
        
    Returns:
        Dictionary containing:
            - watermark_detected: bool - Whether any watermark was found
            - detection_results: Dict - Per-content-type detection results
            - compliance_status: str - Overall compliance status
            - error: Optional[str] - Error message if detection failed
            
    Example:
        >>> content = {
        ...     "text": "AI-generated text...",
        ...     "image": image_tensor,
        ...     "audio": audio_tensor
        ... }
        >>> result = detect_multimodal_watermark(content, verbose=True)
    """
    result = {
        "watermark_detected": False,
        "detection_results": {},
        "compliance_status": "unknown",
        "error": None
    }
    
    try:
        detector = PiscesLxWatermarkDetectionOperator(enable_compliance_check=True)
        
        processed_content = {}
        
        if 'text' in content and content['text']:
            processed_content['text'] = content['text']
        
        if 'image' in content:
            if isinstance(content['image'], str):
                img_tensor = _load_image_as_tensor(content['image'])
                if img_tensor is not None:
                    processed_content['image'] = img_tensor
            else:
                processed_content['image'] = content['image']
        
        if 'audio' in content:
            if isinstance(content['audio'], str):
                wav, sr = _load_audio_as_tensor(content['audio'])
                if wav is not None:
                    processed_content['audio'] = wav
                    processed_content['sample_rate'] = sr
            else:
                processed_content['audio'] = content['audio']
        
        if 'video' in content:
            if isinstance(content['video'], str):
                video_tensor, _ = _load_video_as_tensor(content['video'])
                if video_tensor is not None:
                    processed_content['video'] = video_tensor
            else:
                processed_content['video'] = content['video']
        
        det_result = detector.detect_multimodal_watermark(processed_content, jurisdiction=jurisdiction)
        
        result["watermark_detected"] = det_result.get("watermark_detected", False)
        result["detection_results"] = det_result.get("detection_results", {})
        result["compliance_status"] = det_result.get("compliance_status", "unknown")
        
        if verbose:
            detected_types = [k for k, v in result["detection_results"].items() 
                            if v.get("watermark_detected")]
            if detected_types:
                logger.info(
                    f"Multimodal watermarks detected in: {', '.join(detected_types)}",
                    event="wmc.multimodal.detect"
                )
            else:
                logger.info("No multimodal watermarks detected")
                
    except Exception as e:
        result["error"] = str(e)
        
    return result


__all__ = [
    "detect_watermark",
    "batch_detect",
    "detect_image_watermark",
    "detect_audio_watermark",
    "detect_video_watermark",
    "detect_model_watermark",
    "detect_multimodal_watermark"
]
