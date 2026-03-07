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
CMU Element Detector - UI Element Detection Module

This module provides UI element detection capabilities for the Computer Use Agent,
using visual analysis and pattern matching to identify interactive elements.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUElement,
    POPSSCMUElementState,
    POPSSCMURectangle,
    POPSSCMUCoordinate,
    POPSSCMUScreenState,
    POPSSCMUTarget,
)

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Perception.ElementDetector", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Perception.ElementDetector"), enable_file=True)

_HAS_CV2 = False
_HAS_NUMPY = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _LOG.warning("opencv_not_available")

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _LOG.warning("numpy_not_available")


@dataclass
class POPSSCMUDetectionResult:
    """Element detection result."""
    element: POPSSCMUElement
    confidence: float
    detection_method: str
    raw_data: Dict[str, Any] = field(default_factory=dict)


class POPSSCMUElementDetector:
    """
    UI element detector using visual analysis.
    
    Provides element detection capabilities:
        - Button detection
        - Input field detection
        - Text element detection
        - Icon matching
        - Template matching
    
    Attributes:
        confidence_threshold: Minimum confidence for detection
        template_cache: Cache for loaded templates
    """
    
    ELEMENT_PATTERNS = {
        "button": [
            r"(?i)button",
            r"(?i)btn",
            r"(?i)submit",
            r"(?i)cancel",
            r"(?i)ok",
            r"(?i)confirm",
            r"(?i)save",
            r"(?i)delete",
            r"(?i)close",
        ],
        "input": [
            r"(?i)input",
            r"(?i)text\s*field",
            r"(?i)search",
            r"(?i)email",
            r"(?i)password",
            r"(?i)username",
            r"(?i)enter",
            r"(?i)type",
        ],
        "link": [
            r"(?i)link",
            r"(?i)click\s*here",
            r"(?i)learn\s*more",
            r"(?i)read\s*more",
            r"(?i)see\s*more",
        ],
        "checkbox": [
            r"(?i)checkbox",
            r"(?i)check",
            r"(?i)agree",
            r"(?i)accept",
            r"(?i)remember\s*me",
        ],
        "dropdown": [
            r"(?i)dropdown",
            r"(?i)select",
            r"(?i)choose",
            r"(?i)options",
        ],
    }
    
    def __init__(
        self,
        confidence_threshold: float = 0.8,
        vision_encoder: Optional[Any] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.vision_encoder = vision_encoder
        self.template_cache: Dict[str, Any] = {}
        
        _LOG.info("element_detector_initialized")
    
    async def detect_elements(
        self,
        screen_state: POPSSCMUScreenState,
    ) -> List[POPSSCMUElement]:
        """
        Detect all UI elements in screen state.
        
        Args:
            screen_state: Screen state to analyze
        
        Returns:
            List[POPSSCMUElement]: Detected elements
        """
        elements = []
        
        if self.vision_encoder:
            try:
                vision_elements = await self._detect_with_vision_encoder(screen_state)
                elements.extend(vision_elements)
            except Exception as e:
                _LOG.error("vision_detection_failed", error=str(e))
        
        if _HAS_CV2 and screen_state.image_data:
            try:
                cv_elements = await self._detect_with_opencv(screen_state)
                elements.extend(cv_elements)
            except Exception as e:
                _LOG.error("opencv_detection_failed", error=str(e))
        
        return self._deduplicate_elements(elements)
    
    async def _detect_with_vision_encoder(
        self,
        screen_state: POPSSCMUScreenState,
    ) -> List[POPSSCMUElement]:
        """Detect elements using vision encoder."""
        elements = []
        
        if not self.vision_encoder or not screen_state.image_data:
            return elements
        
        try:
            if hasattr(self.vision_encoder, 'detect_objects'):
                detections = await self.vision_encoder.detect_objects(
                    screen_state.image_data,
                    confidence_threshold=self.confidence_threshold,
                )
                
                for det in detections:
                    element = POPSSCMUElement(
                        element_type=det.get("label", "unknown"),
                        bounds=POPSSCMURectangle(
                            x=det.get("x", 0),
                            y=det.get("y", 0),
                            width=det.get("width", 0),
                            height=det.get("height", 0),
                        ),
                        confidence=det.get("confidence", 0.0),
                        text=det.get("text", ""),
                    )
                    elements.append(element)
        except Exception as e:
            _LOG.error("vision_encoder_detection_failed", error=str(e))
        
        return elements
    
    async def _detect_with_opencv(
        self,
        screen_state: POPSSCMUScreenState,
    ) -> List[POPSSCMUElement]:
        """Detect elements using OpenCV."""
        elements = []
        
        if not _HAS_CV2 or not _HAS_NUMPY:
            return elements
        
        try:
            if _HAS_PIL:
                import PIL.Image
                if isinstance(screen_state.image_data, PIL.Image.Image):
                    img_array = np.array(screen_state.image_data)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = screen_state.image_data
            else:
                return elements
            
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                if w < 20 or h < 20:
                    continue
                if w > screen_state.width * 0.9 or h > screen_state.height * 0.9:
                    continue
                
                element = POPSSCMUElement(
                    element_type="region",
                    bounds=POPSSCMURectangle(x=x, y=y, width=w, height=h),
                    confidence=0.5,
                )
                elements.append(element)
                
        except Exception as e:
            _LOG.error("opencv_processing_failed", error=str(e))
        
        return elements
    
    async def find_element(
        self,
        screen_state: POPSSCMUScreenState,
        description: str,
    ) -> Optional[POPSSCMUElement]:
        """
        Find element by description.
        
        Args:
            screen_state: Screen state to search
            description: Element description
        
        Returns:
            Optional[POPSSCMUElement]: Found element or None
        """
        elements = await self.detect_elements(screen_state)
        
        description_lower = description.lower()
        
        for element_type, patterns in self.ELEMENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, description_lower):
                    for element in elements:
                        if element.element_type == element_type:
                            return element
        
        for element in elements:
            if element.text and description_lower in element.text.lower():
                return element
        
        return None
    
    async def find_element_by_text(
        self,
        screen_state: POPSSCMUScreenState,
        text: str,
        exact_match: bool = False,
    ) -> Optional[POPSSCMUElement]:
        """Find element by text content."""
        elements = await self.detect_elements(screen_state)
        
        for element in elements:
            if not element.text:
                continue
            
            if exact_match:
                if element.text == text:
                    return element
            else:
                if text.lower() in element.text.lower():
                    return element
        
        return None
    
    async def find_element_by_position(
        self,
        screen_state: POPSSCMUScreenState,
        x: float,
        y: float,
    ) -> Optional[POPSSCMUElement]:
        """Find element at position."""
        elements = await self.detect_elements(screen_state)
        coord = POPSSCMUCoordinate(x=x, y=y)
        
        for element in elements:
            if element.bounds.contains(coord):
                return element
        
        return None
    
    async def find_all_by_type(
        self,
        screen_state: POPSSCMUScreenState,
        element_type: str,
    ) -> List[POPSSCMUElement]:
        """Find all elements of a specific type."""
        elements = await self.detect_elements(screen_state)
        return [e for e in elements if e.element_type == element_type]
    
    async def match_template(
        self,
        screen_state: POPSSCMUScreenState,
        template_path: str,
        threshold: Optional[float] = None,
    ) -> List[Tuple[float, POPSSCMURectangle]]:
        """Match template image in screen."""
        if not _HAS_CV2:
            return []
        
        threshold = threshold or self.confidence_threshold
        matches = []
        
        try:
            if template_path not in self.template_cache:
                self.template_cache[template_path] = cv2.imread(template_path, cv2.IMREAD_COLOR)
            
            template = self.template_cache[template_path]
            if template is None:
                return []
            
            if _HAS_PIL:
                import PIL.Image
                if isinstance(screen_state.image_data, PIL.Image.Image):
                    img_array = np.array(screen_state.image_data)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = screen_state.image_data
            else:
                return []
            
            result = cv2.matchTemplate(img_bgr, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            
            h, w = template.shape[:2]
            
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                rect = POPSSCMURectangle(x=pt[0], y=pt[1], width=w, height=h)
                matches.append((float(confidence), rect))
                
        except Exception as e:
            _LOG.error("template_matching_failed", error=str(e))
        
        return matches
    
    def _deduplicate_elements(
        self,
        elements: List[POPSSCMUElement],
    ) -> List[POPSSCMUElement]:
        """Remove duplicate elements."""
        seen = set()
        unique = []
        
        for element in elements:
            key = (
                element.element_type,
                int(element.bounds.x),
                int(element.bounds.y),
                int(element.bounds.width),
                int(element.bounds.height),
            )
            
            if key not in seen:
                seen.add(key)
                unique.append(element)
        
        return unique
    
    def clear_template_cache(self) -> None:
        """Clear template cache."""
        self.template_cache.clear()
        _LOG.info("template_cache_cleared")
