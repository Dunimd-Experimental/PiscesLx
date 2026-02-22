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
WMC Toolkit Validation Test Suite

This module provides comprehensive validation tests for the watermark detection
system, covering text, image, audio, video, and model weight detection.

Test Categories:
    - Text watermark detection
    - Image watermark detection
    - Audio watermark detection
    - Video watermark detection
    - Model weight watermark verification
    - Content integrity checking
    - Batch processing
    - Error handling
    - Compliance checking

Usage:
    >>> from tools.wmc.validation import validate_wmc_toolkit
    >>> results = validate_wmc_toolkit()
    >>> print(f"Passed: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")

Author: PiscesL1 Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List
import tempfile
import os
from pathlib import Path
from datetime import datetime
import json

from utils.opsc.test_suite import BaseTestSuite
from utils.dc import PiscesLxLogger

from .core import (
    PiscesLxWatermarkDetectionOperator,
    PiscesLxModelWatermarkVerificationOperator,
    PiscesLxContentIntegrityCheckOperator
)


_LOG = PiscesLxLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple test model for watermark verification tests."""
    
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
    
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask.float()
        transformed = self.transformer(embedded, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        logits = self.classifier(transformed)
        return logits


class WMCTestSuite(BaseTestSuite):
    """WMC Toolkit Test Suite"""
    
    def __init__(self):
        super().__init__("PiscesLx WMC Toolkit")
        self.temp_dir = None
    
    def setUp(self):
        """Initialize test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="pisceslx_wmc_test_")
        logger.info(f"Test environment created at {self.temp_dir}")
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
        _LOG.info("Test environment cleaned up")
    
    def test_01_watermark_detection_operator(self):
        """Test watermark detection operator initialization and basic detection."""
        _LOG.info("Testing watermark detection operator...")
        
        detector = PiscesLxWatermarkDetectionOperator(
            strict_mode=False,
            enable_compliance_check=True
        )
        
        test_text = "This is a test text without watermark."
        result = detector.detect_text_watermark(test_text, jurisdiction="GLOBAL")
        
        self.assertIn('watermark_detected', result)
        self.assertIn('compliance_status', result)
        self.assertIn('processing_time', result)
        self.assertIn('content_type', result)
        self.assertFalse(result['watermark_detected'])
        self.assertEqual(result['content_type'], 'text')
        
        stats = detector.get_detection_statistics()
        self.assertIn('total_detections', stats)
        self.assertIn('by_content_type', stats)
        self.assertEqual(stats['total_detections'], 1)
        
        _LOG.info("Watermark detection operator test passed")
    
    def test_02_image_watermark_detection(self):
        """Test image watermark detection functionality."""
        _LOG.info("Testing image watermark detection...")
        
        detector = PiscesLxWatermarkDetectionOperator()
        
        test_image = torch.rand(3, 224, 224)
        result = detector.detect_image_watermark(test_image, jurisdiction="GLOBAL")
        
        self.assertIn('watermark_detected', result)
        self.assertIn('content_type', result)
        self.assertEqual(result['content_type'], 'image')
        self.assertIn('detection_confidence', result)
        
        _LOG.info("Image watermark detection test passed")
    
    def test_03_audio_watermark_detection(self):
        """Test audio watermark detection functionality."""
        _LOG.info("Testing audio watermark detection...")
        
        detector = PiscesLxWatermarkDetectionOperator()
        
        test_audio = torch.randn(1, 44100)
        result = detector.detect_audio_watermark(test_audio, sample_rate=44100, jurisdiction="GLOBAL")
        
        self.assertIn('watermark_detected', result)
        self.assertIn('content_type', result)
        self.assertEqual(result['content_type'], 'audio')
        
        _LOG.info("Audio watermark detection test passed")
    
    def test_04_video_watermark_detection(self):
        """Test video watermark detection with multi-frame analysis."""
        _LOG.info("Testing video watermark detection...")
        
        detector = PiscesLxWatermarkDetectionOperator()
        
        test_video = torch.rand(60, 3, 224, 224)
        result = detector.detect_video_watermark(test_video, jurisdiction="GLOBAL", frame_sample_rate=10)
        
        self.assertIn('watermark_detected', result)
        self.assertIn('content_type', result)
        self.assertEqual(result['content_type'], 'video')
        self.assertIn('frame_results', result)
        self.assertIn('temporal_consistency', result)
        self.assertIn('frames_analyzed', result)
        
        self.assertIsInstance(result['frame_results'], list)
        self.assertGreater(len(result['frame_results']), 0)
        
        for frame_result in result['frame_results']:
            self.assertIn('frame_index', frame_result)
            self.assertIn('watermark_detected', frame_result)
        
        _LOG.info("Video watermark detection test passed")
    
    def test_05_model_watermark_verification(self):
        """Test model watermark verification."""
        _LOG.info("Testing model watermark verification...")
        
        verifier = PiscesLxModelWatermarkVerificationOperator(sensitivity_level="medium")
        
        test_model = SimpleTestModel(vocab_size=100, hidden_size=64)
        
        result = verifier.verify_model_integrity(test_model)
        
        self.assertIn('integrity_verified', result)
        self.assertIn('confidence_score', result)
        self.assertIn('tampering_detected', result)
        
        self.assertIsInstance(result['confidence_score'], float)
        self.assertGreaterEqual(result['confidence_score'], 0.0)
        self.assertLessEqual(result['confidence_score'], 1.0)
        
        _LOG.info("Model watermark verification test passed")
    
    def test_06_model_watermark_detection(self):
        """Test model watermark detection through main operator."""
        _LOG.info("Testing model watermark detection through main operator...")
        
        detector = PiscesLxWatermarkDetectionOperator()
        
        test_model = SimpleTestModel(vocab_size=50, hidden_size=32)
        
        result = detector.detect_model_watermark(test_model)
        
        self.assertIn('watermark_detected', result)
        self.assertIn('model_verified', result)
        self.assertIn('verification_confidence', result)
        self.assertIn('content_type', result)
        self.assertEqual(result['content_type'], 'model')
        
        _LOG.info("Model watermark detection test passed")
    
    def test_07_content_integrity_check(self):
        """Test content integrity checking."""
        _LOG.info("Testing content integrity check...")
        
        integrity_checker = PiscesLxContentIntegrityCheckOperator(check_depth="standard")
        
        normal_text = "This is normal content without suspicious elements."
        result = integrity_checker.check_content_integrity(normal_text, content_type="text")
        
        self.assertIn('integrity_verified', result)
        self.assertIn('suspicious_elements', result)
        self.assertTrue(result['integrity_verified'])
        self.assertEqual(len(result['suspicious_elements']), 0)
        
        suspicious_text = "Normal text <script>alert('xss')</script> more text"
        result = integrity_checker.check_content_integrity(suspicious_text, content_type="text")
        
        self.assertIn('suspicious_elements', result)
        self.assertGreater(len(result['suspicious_elements']), 0)
        
        _LOG.info("Content integrity check test passed")
    
    def test_08_batch_detection(self):
        """Test batch detection functionality."""
        _LOG.info("Testing batch detection...")
        
        detector = PiscesLxWatermarkDetectionOperator()
        
        test_texts = [
            "First test text without watermark.",
            "Second test text also clean.",
            "Third text for batch processing."
        ]
        
        batch_result = detector.batch_detect(test_texts, content_type="text", jurisdiction="GLOBAL")
        
        self.assertIn('total_items', batch_result)
        self.assertIn('detected_items', batch_result)
        self.assertIn('individual_results', batch_result)
        self.assertIn('content_type', batch_result)
        self.assertEqual(batch_result['total_items'], 3)
        self.assertEqual(len(batch_result['individual_results']), 3)
        self.assertEqual(batch_result['content_type'], 'text')
        
        for individual_result in batch_result['individual_results']:
            self.assertIn('watermark_detected', individual_result)
            self.assertIn('batch_index', individual_result)
        
        _LOG.info("Batch detection test passed")
    
    def test_09_multimodal_detection(self):
        """Test multimodal watermark detection."""
        _LOG.info("Testing multimodal watermark detection...")
        
        detector = PiscesLxWatermarkDetectionOperator()
        
        content = {
            'text': "Test text for multimodal detection",
            'image': torch.rand(3, 64, 64),
            'audio': torch.randn(1, 22050)
        }
        
        result = detector.detect_multimodal_watermark(content, jurisdiction="GLOBAL")
        
        self.assertIn('watermark_detected', result)
        self.assertIn('detection_results', result)
        self.assertIn('content_type', result)
        self.assertEqual(result['content_type'], 'multimodal')
        
        self.assertIn('text', result['detection_results'])
        self.assertIn('image', result['detection_results'])
        self.assertIn('audio', result['detection_results'])
        
        _LOG.info("Multimodal detection test passed")
    
    def test_10_detection_statistics(self):
        """Test detection statistics tracking."""
        _LOG.info("Testing detection statistics...")
        
        detector = PiscesLxWatermarkDetectionOperator()
        
        texts = ["Test 1", "Test 2", "Test 3"]
        for text in texts:
            detector.detect_text_watermark(text)
        
        stats = detector.get_detection_statistics()
        
        self.assertIn('total_detections', stats)
        self.assertIn('successful_detections', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('by_content_type', stats)
        self.assertEqual(stats['total_detections'], 3)
        
        expected_success_rate = stats['successful_detections'] / max(stats['total_detections'], 1)
        self.assertAlmostEqual(stats['success_rate'], expected_success_rate, places=6)
        
        _LOG.info("Detection statistics test passed")
    
    def test_11_compliance_checking(self):
        """Test compliance checking for different jurisdictions."""
        _LOG.info("Testing compliance checking...")
        
        detector = PiscesLxWatermarkDetectionOperator(
            enable_compliance_check=True
        )
        
        jurisdictions = ['CN', 'EU', 'US', 'UK', 'GLOBAL']
        
        for jurisdiction in jurisdictions:
            test_text = f"Test text for {jurisdiction} compliance."
            result = detector.detect_text_watermark(test_text, jurisdiction=jurisdiction)
            
            self.assertIn('compliance_status', result)
            self.assertIn('jurisdiction', result)
            self.assertEqual(result['jurisdiction'], jurisdiction)
        
        _LOG.info("Compliance checking test passed")
    
    def test_12_error_handling(self):
        """Test error handling mechanisms."""
        _LOG.info("Testing error handling...")
        
        detector = PiscesLxWatermarkDetectionOperator(strict_mode=False)
        
        result = detector.detect_text_watermark("")
        self.assertIn('error', result)
        self.assertFalse(result['watermark_detected'])
        
        try:
            result = detector.detect_image_watermark("not a tensor")
            self.assertIn('error', result)
        except Exception as e:
            _LOG.info(f"Expected error caught: {e}")
        
        _LOG.info("Error handling test passed")
    
    def test_13_tensor_integrity_check(self):
        """Test tensor content integrity checking."""
        _LOG.info("Testing tensor integrity check...")
        
        integrity_checker = PiscesLxContentIntegrityCheckOperator()
        
        normal_tensor = torch.randn(3, 64, 64)
        result = integrity_checker.check_content_integrity(normal_tensor, content_type="tensor")
        
        self.assertIn('integrity_verified', result)
        self.assertIn('checks_performed', result)
        
        nan_tensor = torch.randn(3, 64, 64)
        nan_tensor[0, 0, 0] = float('nan')
        result = integrity_checker.check_content_integrity(nan_tensor, content_type="tensor")
        
        self.assertIn('nan_values', result['suspicious_elements'])
        
        _LOG.info("Tensor integrity check test passed")
    
    def test_14_video_temporal_consistency(self):
        """Test video temporal consistency analysis."""
        _LOG.info("Testing video temporal consistency analysis...")
        
        detector = PiscesLxWatermarkDetectionOperator()
        
        test_video = torch.rand(100, 3, 112, 112)
        result = detector.detect_video_watermark(test_video, frame_sample_rate=5)
        
        self.assertIn('temporal_consistency', result)
        self.assertGreaterEqual(result['temporal_consistency'], 0.0)
        self.assertLessEqual(result['temporal_consistency'], 1.0)
        
        _LOG.info("Video temporal consistency test passed")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all validation tests."""
        _LOG.info("Starting comprehensive WMC toolkit validation...")
        
        test_results = {}
        passed_tests = 0
        total_tests = 14
        
        try:
            self.setUp()
            
            test_methods = [
                self.test_01_watermark_detection_operator,
                self.test_02_image_watermark_detection,
                self.test_03_audio_watermark_detection,
                self.test_04_video_watermark_detection,
                self.test_05_model_watermark_verification,
                self.test_06_model_watermark_detection,
                self.test_07_content_integrity_check,
                self.test_08_batch_detection,
                self.test_09_multimodal_detection,
                self.test_10_detection_statistics,
                self.test_11_compliance_checking,
                self.test_12_error_handling,
                self.test_13_tensor_integrity_check,
                self.test_14_video_temporal_consistency
            ]
            
            for i, test_method in enumerate(test_methods, 1):
                test_name = f"test_{i:02d}"
                try:
                    test_method()
                    test_results[test_name] = "PASSED"
                    passed_tests += 1
                    _LOG.info(f"✓ {test_method.__doc__}")
                except Exception as e:
                    test_results[test_name] = f"FAILED: {str(e)}"
                    _LOG.error(f"✗ {test_method.__doc__}: {e}")
            
            self.tearDown()
            
        except Exception as e:
            _LOG.error(f"Test suite failed: {e}")
            test_results['suite'] = f"FAILED: {str(e)}"
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'details': test_results,
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'temp_dir': self.temp_dir
            }
        }
        
        _LOG.info(f"WMC toolkit validation completed: {passed_tests}/{total_tests} tests passed")
        return report


def validate_wmc_toolkit() -> Dict[str, Any]:
    """Main entry point for WMC toolkit validation."""
    validator = WMCTestSuite()
    return validator.run_all_tests()


if __name__ == "__main__":
    results = validate_wmc_toolkit()
    print(json.dumps(results, indent=2, ensure_ascii=False))
