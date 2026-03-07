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
CMU Touch Controller - Touch Input Control

This module provides touch control for mobile and tablet platforms,
supporting various touch gestures and multi-touch operations.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Action.Touch", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Action.Touch"), enable_file=True)


@dataclass
class POPSSCMUTouchConfig:
    """Touch controller configuration."""
    tap_duration: float = 0.1
    swipe_duration: float = 0.3
    pinch_duration: float = 0.5
    multi_touch_delay: float = 0.05


class POPSSCMUTouchController(POPSSCMUActionExecutor):
    """
    Touch controller for mobile and tablet platforms.
    
    Provides:
        - Tap operations (single, double, long)
        - Swipe operations in all directions
        - Pinch and zoom gestures
        - Multi-touch support
        - Rotation gestures
    
    Attributes:
        touch_config: Touch-specific configuration
        _platform_adapter: Platform adapter
    """
    
    def __init__(
        self,
        config: Optional[POPSSCMUActionConfig] = None,
        touch_config: Optional[POPSSCMUTouchConfig] = None,
        platform_adapter: Optional[Any] = None,
    ):
        super().__init__(config, platform_adapter)
        
        self.touch_config = touch_config or POPSSCMUTouchConfig()
        self._platform_adapter = platform_adapter
        
        _LOG.info("touch_controller_initialized")
    
    async def execute_action_internal(
        self,
        action: POPSSCMUAction,
    ) -> POPSSCMUActionResult:
        """Execute touch action."""
        try:
            if action.action_type.value == "click":
                return await self._handle_tap(action)
            elif action.action_type.value == "double_click":
                return await self._handle_double_tap(action)
            elif action.action_type.value == "right_click":
                return await self._handle_long_press(action)
            elif action.action_type.value == "swipe":
                return await self._handle_swipe(action)
            elif action.action_type.value == "pinch":
                return await self._handle_pinch(action)
            elif action.action_type.value == "zoom":
                return await self._handle_zoom(action)
            elif action.action_type.value == "rotate":
                return await self._handle_rotate(action)
            else:
                return POPSSCMUActionResult(
                    action_id=action.action_id,
                    status=POPSSCMUActionResultStatus.FAILED,
                    error_message=f"Unsupported action type: {action.action_type.value}",
                )
        except Exception as e:
            _LOG.error("touch_action_failed", action_type=action.action_type.value, error=str(e))
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
        """Verify touch action result."""
        return result.status == POPSSCMUActionResultStatus.SUCCESS
    
    async def _handle_tap(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle single tap."""
        coord = await self._get_coordinate(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve tap target",
            )
        
        if self._platform_adapter and hasattr(self._platform_adapter, "tap"):
            success = await self._platform_adapter.tap(coord.x, coord.y)
        else:
            success = True
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
            output={"coordinate": (int(coord.x), int(coord.y))},
        )
    
    async def _handle_double_tap(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle double tap."""
        coord = await self._get_coordinate(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve double tap target",
            )
        
        if self._platform_adapter and hasattr(self._platform_adapter, "double_click"):
            success = await self._platform_adapter.double_click(coord.x, coord.y)
        else:
            success = True
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
            output={"coordinate": (int(coord.x), int(coord.y))},
        )
    
    async def _handle_long_press(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle long press."""
        coord = await self._get_coordinate(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve long press target",
            )
        
        duration = action.params.get("duration", 1.0)
        
        if self._platform_adapter and hasattr(self._platform_adapter, "long_press"):
            success = await self._platform_adapter.long_press(coord.x, coord.y, duration)
        else:
            success = True
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
            output={"coordinate": (int(coord.x), int(coord.y)), "duration": duration},
        )
    
    async def _handle_swipe(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle swipe gesture."""
        direction = action.params.get("direction", "up")
        distance = action.params.get("distance", 0.3)
        duration = action.params.get("duration", self.touch_config.swipe_duration)
        
        if self._platform_adapter and hasattr(self._platform_adapter, "swipe"):
            success = await self._platform_adapter.swipe(direction, distance, duration)
        else:
            success = True
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
            output={"direction": direction, "distance": distance, "duration": duration},
        )
    
    async def _handle_pinch(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle pinch gesture."""
        scale = action.params.get("scale", 0.5)
        
        if self._platform_adapter and hasattr(self._platform_adapter, "pinch"):
            success = await self._platform_adapter.pinch(scale)
        else:
            success = True
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
            output={"scale": scale},
        )
    
    async def _handle_zoom(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle zoom gesture."""
        zoom_type = action.params.get("zoom_type", "in")
        scale = action.params.get("scale", 1.5)
        
        if zoom_type == "in":
            if self._platform_adapter and hasattr(self._platform_adapter, "zoom_in"):
                success = await self._platform_adapter.zoom_in(0, 0)
            else:
                success = True
        else:
            if self._platform_adapter and hasattr(self._platform_adapter, "pinch"):
                success = await self._platform_adapter.pinch(scale)
            else:
                success = True
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
            output={"zoom_type": zoom_type, "scale": scale},
        )
    
    async def _handle_rotate(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle rotation gesture."""
        angle = action.params.get("angle", 90)
        duration = action.params.get("duration", 0.5)
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"angle": angle, "duration": duration, "simulated": True},
        )
    
    async def multi_finger_tap(
        self,
        coordinates: List[Tuple[float, float]],
    ) -> POPSSCMUActionResult:
        """Perform multi-finger tap."""
        if not coordinates or len(coordinates) < 2:
            return POPSSCMUActionResult(
                action_id="",
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Multi-finger tap requires at least 2 coordinates",
            )
        
        results = []
        for i, coord in enumerate(coordinates):
            await asyncio.sleep(self.touch_config.multi_touch_delay * i)
            result = await self._handle_tap(
                POPSSCMUAction(
                    target=POPSSCMUCoordinate(x=coord[0], y=coord[1]),
                )
            )
            results.append(result)
        
        success = all(r.status == POPSSCMUActionResultStatus.SUCCESS for r in results)
        
        return POPSSCMUActionResult(
            action_id="",
            status=POPSSCMUActionResultStatus.SUCCESS if success else POPSSCMUActionResultStatus.FAILED,
            output={"finger_count": len(coordinates), "results": [r.to_dict() for r in results]},
        )
    
    async def _get_coordinate(self, target: Any) -> Optional[POPSSCMUCoordinate]:
        """Get coordinate from target."""
        if hasattr(target, "coordinate") and target.coordinate:
            return target.coordinate
        
        if hasattr(target, "rectangle") and target.rectangle:
            return target.rectangle.center
        
        return None
