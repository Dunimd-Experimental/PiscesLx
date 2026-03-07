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
CMU OCR Reader - Optical Character Recognition Module

This module provides OCR capabilities for text extraction from screenshots,
supporting multiple OCR engines and languages.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMURectangle,
    POPSSCMUScreenState,
)

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Perception.OCR", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Perception.OCR"), enable_file=True)

_HAS_TESSERACT = False
_HAS_PYTESSERACT = False
_HAS_EASYOCR = False
_HAS_PIL = False

try:
    import pytesseract
    _HAS_PYTESSERACT = True
except ImportError:
    _LOG.warning("pytesseract_not_available", message="Install pytesseract for OCR support: pip install pytesseract")

try:
    import easyocr
    _HAS_EASYOCR = True
except ImportError:
    _LOG.warning("easyocr_not_available", message="Install easyocr for advanced OCR: pip install easyocr")

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _LOG.warning("pillow_not_available", message="Install Pillow for image processing: pip install Pillow")


@dataclass
class POPSSCMUTextBlock:
    """Detected text block with position."""
    text: str
    confidence: float
    bounds: POPSSCMURectangle
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bounds": {
                "x": self.bounds.x,
                "y": self.bounds.y,
                "width": self.bounds.width,
                "height": self.bounds.height,
            },
            "language": self.language,
        }


class POPSSCMUOCRReader:
    """
    OCR reader supporting multiple engines.
    
    Provides text extraction capabilities:
        - Tesseract OCR
        - EasyOCR
        - Multi-language support
        - Text localization
    
    Attributes:
        engine: OCR engine type ("tesseract", "easyocr")
        languages: Supported languages
        _reader: OCR reader instance
    """
    
    def __init__(
        self,
        engine: str = "tesseract",
        languages: List[str] = None,
    ):
        self.engine = engine
        self.languages = languages or ["en"]
        self._reader: Optional[Any] = None
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize OCR engine."""
        if self.engine == "easyocr" and _HAS_EASYOCR:
            try:
                self._reader = easyocr.Reader(self.languages)
                _LOG.info("easyocr_initialized", languages=self.languages)
            except Exception as e:
                _LOG.error("easyocr_init_failed", error=str(e))
                self.engine = "tesseract"
        
        if self.engine == "tesseract":
            if _HAS_PYTESSERACT:
                _LOG.info("tesseract_initialized")
            else:
                _LOG.warning("no_ocr_engine_available")
    
    async def extract_text(
        self,
        image: Any,
        region: Optional[POPSSCMURectangle] = None,
    ) -> str:
        """
        Extract text from image.
        
        Args:
            image: Image to process
            region: Optional region to extract from
        
        Returns:
            str: Extracted text
        """
        if region:
            image = self._crop_image(image, region)
        
        if self.engine == "easyocr" and self._reader:
            return await self._extract_easyocr(image)
        elif self.engine == "tesseract" and _HAS_PYTESSERACT:
            return await self._extract_tesseract(image)
        
        return ""
    
    async def extract_text_blocks(
        self,
        image: Any,
        region: Optional[POPSSCMURectangle] = None,
    ) -> List[POPSSCMUTextBlock]:
        """
        Extract text blocks with positions.
        
        Args:
            image: Image to process
            region: Optional region to extract from
        
        Returns:
            List[POPSSCMUTextBlock]: Detected text blocks
        """
        if region:
            image = self._crop_image(image, region)
        
        if self.engine == "easyocr" and self._reader:
            return await self._extract_blocks_easyocr(image)
        elif self.engine == "tesseract" and _HAS_PYTESSERACT:
            return await self._extract_blocks_tesseract(image)
        
        return []
    
    async def _extract_tesseract(self, image: Any) -> str:
        """Extract text using Tesseract."""
        try:
            if _HAS_PIL and isinstance(image, Image.Image):
                text = pytesseract.image_to_string(image)
            else:
                text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            _LOG.error("tesseract_extraction_failed", error=str(e))
            return ""
    
    async def _extract_easyocr(self, image: Any) -> str:
        """Extract text using EasyOCR."""
        try:
            if _HAS_PIL and isinstance(image, Image.Image):
                import numpy as np
                image = np.array(image)
            
            results = self._reader.readtext(image)
            texts = [r[1] for r in results]
            return " ".join(texts)
        except Exception as e:
            _LOG.error("easyocr_extraction_failed", error=str(e))
            return ""
    
    async def _extract_blocks_tesseract(
        self,
        image: Any,
    ) -> List[POPSSCMUTextBlock]:
        """Extract text blocks using Tesseract."""
        blocks = []
        
        try:
            if _HAS_PIL and isinstance(image, Image.Image):
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            else:
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if not text:
                    continue
                
                confidence = float(data['conf'][i]) / 100.0 if data['conf'][i] != '-1' else 0.0
                
                block = POPSSCMUTextBlock(
                    text=text,
                    confidence=confidence,
                    bounds=POPSSCMURectangle(
                        x=data['left'][i],
                        y=data['top'][i],
                        width=data['width'][i],
                        height=data['height'][i],
                    ),
                )
                blocks.append(block)
                
        except Exception as e:
            _LOG.error("tesseract_block_extraction_failed", error=str(e))
        
        return blocks
    
    async def _extract_blocks_easyocr(
        self,
        image: Any,
    ) -> List[POPSSCMUTextBlock]:
        """Extract text blocks using EasyOCR."""
        blocks = []
        
        try:
            if _HAS_PIL and isinstance(image, Image.Image):
                import numpy as np
                image = np.array(image)
            
            results = self._reader.readtext(image)
            
            for bbox, text, confidence in results:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                
                x = min(x_coords)
                y = min(y_coords)
                width = max(x_coords) - x
                height = max(y_coords) - y
                
                block = POPSSCMUTextBlock(
                    text=text,
                    confidence=confidence,
                    bounds=POPSSCMURectangle(x=x, y=y, width=width, height=height),
                )
                blocks.append(block)
                
        except Exception as e:
            _LOG.error("easyocr_block_extraction_failed", error=str(e))
        
        return blocks
    
    def _crop_image(
        self,
        image: Any,
        region: POPSSCMURectangle,
    ) -> Any:
        """Crop image to region."""
        if _HAS_PIL and isinstance(image, Image.Image):
            return image.crop((
                int(region.x),
                int(region.y),
                int(region.x + region.width),
                int(region.y + region.height),
            ))
        return image
    
    async def find_text(
        self,
        image: Any,
        search_text: str,
        exact_match: bool = False,
    ) -> List[POPSSCMUTextBlock]:
        """Find specific text in image."""
        blocks = await self.extract_text_blocks(image)
        results = []
        
        search_lower = search_text.lower()
        
        for block in blocks:
            if exact_match:
                if block.text.lower() == search_lower:
                    results.append(block)
            else:
                if search_lower in block.text.lower():
                    results.append(block)
        
        return results
    
    def set_languages(self, languages: List[str]) -> None:
        """Set OCR languages."""
        self.languages = languages
        
        if self.engine == "easyocr" and _HAS_EASYOCR:
            try:
                self._reader = easyocr.Reader(languages)
                _LOG.info("ocr_languages_updated", languages=languages)
            except Exception as e:
                _LOG.error("ocr_language_update_failed", error=str(e))
