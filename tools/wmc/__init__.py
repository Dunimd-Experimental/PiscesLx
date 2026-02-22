#!/usr/bin/env/python3
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
PiscesLx WMC (Watermark Check) Toolkit

This module provides flagship watermark detection capabilities based on OPSC architecture,
supporting comprehensive detection across text, image, audio, video, and model weights.

Core Operators:
    - PiscesLxWatermarkDetectionOperator: Main detection operator for all content types
    - PiscesLxModelWatermarkVerificationOperator: Specialized model weight verification
    - PiscesLxContentIntegrityCheckOperator: Content tampering detection

API Functions:
    - detect_watermark: Detect watermark in text content
    - batch_detect: Batch detect watermarks from file
    - detect_image_watermark: Detect watermark from image file
    - detect_audio_watermark: Detect watermark from audio file
    - detect_video_watermark: Detect watermark from video file
    - detect_model_watermark: Detect watermark in model weights
    - detect_multimodal_watermark: Detect watermarks across multiple content types

Supported Content Types:
    - Text: Zero-width character encoding detection
    - Image: DCT frequency domain watermark detection
    - Audio: STFT ultrasonic band watermark detection
    - Video: Frame-by-frame multi-modal detection
    - Model Weights: Codebook correlation verification

Supported Jurisdictions:
    - CN (China): GB/T 45225-2024
    - EU (European Union): AI Act 2024
    - US (United States): NIST AI RMF 1.0
    - UK (United Kingdom): AI Safety Act 2024
    - JP (Japan): AI Guidelines 2024
    - KR (South Korea): AI Act 2024
    - GLOBAL: ISO/IEC 27090

Usage Examples:
    >>> from tools.wmc import PiscesLxWatermarkDetectionOperator, detect_watermark
    >>> 
    >>> # Using operator directly
    >>> detector = PiscesLxWatermarkDetectionOperator()
    >>> result = detector.detect_text_watermark("AI-generated text...")
    >>> 
    >>> # Using API function
    >>> result = detect_watermark("AI-generated text...", verbose=True)
    >>> 
    >>> # Video watermark detection
    >>> video_result = detect_video_watermark("video.mp4", verbose=True)

Author: PiscesL1 Development Team
Version: 1.0.0
"""

from .core import (
    PiscesLxWatermarkDetectionOperator,
    PiscesLxModelWatermarkVerificationOperator,
    PiscesLxContentIntegrityCheckOperator
)

from .check import (
    detect_watermark,
    batch_detect,
    detect_image_watermark,
    detect_audio_watermark,
    detect_video_watermark,
    detect_model_watermark,
    detect_multimodal_watermark
)

from configs.version import VERSION, AUTHOR

__version__ = VERSION
__author__ = AUTHOR

__all__ = [
    "PiscesLxWatermarkDetectionOperator",
    "PiscesLxModelWatermarkVerificationOperator",
    "PiscesLxContentIntegrityCheckOperator",
    "detect_watermark",
    "batch_detect",
    "detect_image_watermark",
    "detect_audio_watermark",
    "detect_video_watermark",
    "detect_model_watermark",
    "detect_multimodal_watermark"
]
