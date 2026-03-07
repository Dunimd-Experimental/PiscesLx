#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to Dunimd Team.
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
CMU Gesture Controller - Complex Gesture Recognition

This module provides complex gesture recognition and execution
for touch-enabled platforms.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUAction,
    POPSSCMUActionResult,
    POPSSCMUActionResultStatus,
    POPSSCMUScreenState,
    POPSSCMUCoordinate,
)
from .base import POPSSCMUActionExecutor, POPSSCMUActionConfig

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Action.Gesture", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Action.Gesture"), enable_file=True)


@dataclass
class POPSSCMUGesture:
    """Gesture definition."""
    gesture_id: str
    name: str
    description: str
    points: List[Tuple[float, float]]
    duration: float = 0.5
    confidence_threshold: float = 0.8


@dataclass
class POPSSCMUGestureConfig:
    """Gesture controller configuration."""
    gesture_library_path: str = "./cmu_gestures"
    auto_learn: bool = True
    min_gesture_points: int = 5
    max_gesture_points: int = 100


class POPSSCMUGestureController(POPSSCMUActionExecutor):
    """
    Gesture controller for complex touch gestures.
    
    Provides:
        - Gesture recording
        - Gesture recognition
        - Gesture playback
        - Custom gesture library
        - Natural gesture simulation
    
    Attributes:
        gesture_config: Gesture-specific configuration
        _gesture_library: Loaded gestures
        _recorded_points: Currently recorded points
    """
    
    def __init__(
        self,
        config: Optional[POPSSCMUActionConfig] = None,
        gesture_config: Optional[POPSSCMUGestureConfig] = None,
        platform_adapter: Optional[Any] = None,
    ):
        super().__init__(config, platform_adapter)
        
        self.gesture_config = gesture_config or POPSSCMUGestureConfig()
        self._gesture_library: Dict[str, POPSSCMUGesture] = {}
        self._recorded_points: List[Tuple[float, float]] = []
        self._is_recording = False
        
        self._load_gesture_library()
        
        _LOG.info("gesture_controller_initialized")
    
    def _load_gesture_library(self) -> None:
        """Load gesture library from file."""
        library_path = Path(self.gesture_config.gesture_library_path)
        
        if library_path.exists():
            try:
                with open(library_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._gesture_library = {
                        g_id: POPSSCMUGesture(**g_data)
                        for g_id, g_data in data.items()
                    }
                _LOG.info("gesture_library_loaded", count=len(self._gesture_library))
            except Exception as e:
                _LOG.error("gesture_library_load_failed", error=str(e))
    
    def _save_gesture_library(self) -> None:
        """Save gesture library to file."""
        library_path = Path(self.gesture_config.gesture_library_path)
        library_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                g_id: {
                    "gesture_id": g.gesture_id,
                    "name": g.name,
                    "description": g.description,
                    "points": g.points,
                    "duration": g.duration,
                    "confidence_threshold": g.confidence_threshold,
                }
                for g_id, g in self._gesture_library.items()
            }
            
            with open(library_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            _LOG.info("gesture_library_saved", count=len(self._gesture_library))
        except Exception as e:
            _LOG.error("gesture_library_save_failed", error=str(e))
    
    async def execute_action_internal(
        self,
        action: POPSSCMUAction,
    ) -> POPSSCMUActionResult:
        """Execute gesture action."""
        try:
            gesture_id = action.params.get("gesture_id", "")
            
            if gesture_id and gesture_id in self._gesture_library:
                return await self._play_gesture(gesture_id)
            
            points = action.params.get("points", [])
            if points:
                return await self._execute_gesture_points(points)
            
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="No gesture specified",
            )
        except Exception as e:
            _LOG.error("gesture_action_failed", error=str(e))
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message=str(e),
            )
    
    async def verify_action_result(
        self,
        action: POPSSCMUAction,
        result: POPSSCMUActionResult,
    ) -> bool:
        """Verify gesture action result."""
        return result.status == POPSSCMUActionResultStatus.SUCCESS
    
    async def start_recording(self) -> None:
        """Start gesture recording."""
        self._is_recording = True
        self._recorded_points = []
        _LOG.info("gesture_recording_started")
    
    async def stop_recording(self, name: str, description: str = "") -> POPSSCMUGesture:
        """Stop recording and save gesture."""
        if not self._is_recording:
            raise RuntimeError("Not currently recording")
        
        self._is_recording = False
        
        if len(self._recorded_points) < self.gesture_config.min_gesture_points:
            _LOG.warning("gesture_too_short", points=len(self._recorded_points))
        
        gesture = POPSSCMUGesture(
            gesture_id=f"gesture_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            description=description,
            points=self._recorded_points,
        )
        
        self._gesture_library[gesture.gesture_id] = gesture
        self._save_gesture_library()
        
        _LOG.info("gesture_recorded", gesture_id=gesture.gesture_id, name=name)
        return gesture
    
    async def record_point(self, x: float, y: float) -> None:
        """Record a point in gesture."""
        if not self._is_recording:
            return
        
        self._recorded_points.append((x, y))
    
    async def _play_gesture(self, gesture_id: str) -> POPSSCMUActionResult:
        """Play a recorded gesture."""
        if gesture_id not in self._gesture_library:
            return POPSSCMUActionResult(
                action_id="",
                status=POPSSCMUActionResultStatus.FAILED,
                error_message=f"Gesture not found: {gesture_id}",
            )
        
        gesture = self._gesture_library[gesture_id]
        return await self._execute_gesture_points(gesture.points, gesture.duration)
    
    async def _execute_gesture_points(
        self,
        points: List[Tuple[float, float]],
        duration: float = 0.5,
    ) -> POPSSCMUActionResult:
        """Execute gesture from points."""
        if not points:
            return POPSSCMUActionResult(
                action_id="",
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="No points provided",
            )
        
        if len(points) < 2:
            return POPSSCMUActionResult(
                action_id="",
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Gesture requires at least 2 points",
            )
        
        point_duration = duration / (len(points) - 1)
        
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            
            if self._platform_adapter and hasattr(self._platform_adapter, "drag"):
                await self._platform_adapter.drag(
                    start[0], start[1],
                    end[0], end[1],
                    point_duration
                )
            
            await asyncio.sleep(point_duration)
        
        return POPSSCMUActionResult(
            action_id="",
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"points_count": len(points), "duration": duration},
        )
    
    async def recognize_gesture(
        self,
        points: List[Tuple[float, float]],
    ) -> Optional[Tuple[str, float]]:
        """Recognize gesture from points."""
        if not points or len(points) < 2:
            return None
        
        best_match = None
        best_score = 0.0
        
        for gesture_id, gesture in self._gesture_library.items():
            score = self._calculate_similarity(points, gesture.points)
            
            if score > best_score and score >= gesture.confidence_threshold:
                best_score = score
                best_match = gesture_id
        
        if best_match:
            return (best_match, best_score)
        
        return None
    
    def _calculate_similarity(
        self,
        points1: List[Tuple[float, float]],
        points2: List[Tuple[float, float]],
    ) -> float:
        """Calculate similarity between two point sequences."""
        if len(points1) != len(points2):
            return 0.0
        
        if not points1:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for p1, p2 in zip(points1, points2):
            distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            total_distance += distance
            count += 1
        
        if count == 0:
            return 0.0
        
        avg_distance = total_distance / count
        similarity = max(0.0, 1.0 - avg_distance / 100.0)
        
        return similarity
    
    def get_gesture_library(self) -> Dict[str, POPSSCMUGesture]:
        """Get gesture library."""
        return self._gesture_library.copy()
    
    def add_gesture(
        self,
        gesture: POPSSCMUGesture,
    ) -> None:
        """Add gesture to library."""
        self._gesture_library[gesture.gesture_id] = gesture
        self._save_gesture_library()
    
    def remove_gesture(self, gesture_id: str) -> bool:
        """Remove gesture from library."""
        if gesture_id in self._gesture_library:
            del self._gesture_library[gesture_id]
            self._save_gesture_library()
            return True
        return False
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
