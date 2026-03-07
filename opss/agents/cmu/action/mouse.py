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
CMU Mouse Controller - Mouse Input Control

This module provides comprehensive mouse control for desktop platforms,
supporting various mouse operations with smooth movement.
"""

from __future__ import annotations

import asyncio
import math
import time
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

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Action.Mouse", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Action.Mouse"), enable_file=True)

_HAS_PYAUTOGUI = False

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
    _HAS_PYAUTOGUI = True
except ImportError:
    _LOG.warning("pyautogui_not_available")


@dataclass
class POPSSCMUMouseConfig:
    """Mouse controller configuration."""
    move_duration: float = 0.3
    click_interval: float = 0.1
    scroll_amount: int = 3
    smooth_movement: bool = True
    movement_curve: str = "ease_in_out"


class POPSSCMUMouseController(POPSSCMUActionExecutor):
    """
    Mouse controller for desktop platforms.
    
    Provides:
        - Click operations (single, double, right, middle)
        - Mouse movement with smooth curves
        - Drag operations
        - Scroll operations
        - Hover operations
    
    Attributes:
        mouse_config: Mouse-specific configuration
        _current_position: Current mouse position
    """
    
    def __init__(
        self,
        config: Optional[POPSSCMUActionConfig] = None,
        mouse_config: Optional[POPSSCMUMouseConfig] = None,
        platform_adapter: Optional[Any] = None,
    ):
        super().__init__(config, platform_adapter)
        
        self.mouse_config = mouse_config or POPSSCMUMouseConfig()
        self._current_position: Tuple[int, int] = (0, 0)
        
        if _HAS_PYAUTOGUI:
            self._current_position = pyautogui.position()
    
    async def execute_action_internal(
        self,
        action: POPSSCMUAction,
    ) -> POPSSCMUActionResult:
        """Execute mouse action."""
        if not _HAS_PYAUTOGUI:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="PyAutoGUI not available",
            )
        
        try:
            if action.action_type.value == "click":
                return await self._handle_click(action)
            elif action.action_type.value == "double_click":
                return await self._handle_double_click(action)
            elif action.action_type.value == "right_click":
                return await self._handle_right_click(action)
            elif action.action_type.value == "middle_click":
                return await self._handle_middle_click(action)
            elif action.action_type.value == "hover":
                return await self._handle_hover(action)
            elif action.action_type.value == "drag":
                return await self._handle_drag(action)
            elif action.action_type.value == "scroll":
                return await self._handle_scroll(action)
            else:
                return POPSSCMUActionResult(
                    action_id=action.action_id,
                    status=POPSSCMUActionResultStatus.FAILED,
                    error_message=f"Unsupported action type: {action.action_type.value}",
                )
        except Exception as e:
            _LOG.error("mouse_action_failed", action_type=action.action_type.value, error=str(e))
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
        """Verify mouse action result."""
        if result.status != POPSSCMUActionResultStatus.SUCCESS:
            return False
        
        if action.action_type.value in ["click", "double_click", "right_click", "middle_click"]:
            return True
        
        if action.action_type.value == "hover":
            return True
        
        return True
    
    async def _handle_click(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle single click."""
        button = action.params.get("button", "left")
        clicks = action.params.get("clicks", 1)
        
        coord = await self._get_coordinate(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve click target",
            )
        
        pyautogui.click(
            x=int(coord.x),
            y=int(coord.y),
            clicks=clicks,
            button=button,
            interval=self.mouse_config.click_interval,
        )
        
        self._current_position = (int(coord.x), int(coord.y))
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"button": button, "clicks": clicks, "coordinate": (int(coord.x), int(coord.y))},
        )
    
    async def _handle_double_click(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle double click."""
        coord = await self._get_coordinate(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve double click target",
            )
        
        pyautogui.doubleClick(x=int(coord.x), y=int(coord.y))
        self._current_position = (int(coord.x), int(coord.y))
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"coordinate": (int(coord.x), int(coord.y))},
        )
    
    async def _handle_right_click(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle right click."""
        coord = await self._get_coordinate(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve right click target",
            )
        
        pyautogui.rightClick(x=int(coord.x), y=int(coord.y))
        self._current_position = (int(coord.x), int(coord.y))
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"coordinate": (int(coord.x), int(coord.y))},
        )
    
    async def _handle_middle_click(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle middle click."""
        coord = await self._get_coordinate(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve middle click target",
            )
        
        pyautogui.middleClick(x=int(coord.x), y=int(coord.y))
        self._current_position = (int(coord.x), int(coord.y))
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"coordinate": (int(coord.x), int(coord.y))},
        )
    
    async def _handle_hover(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle hover operation."""
        coord = await self._get_coordinate(action.target)
        if coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve hover target",
            )
        
        duration = action.params.get("duration", 1.0)
        
        await self.move_to(coord.x, coord.y, duration=duration)
        await asyncio.sleep(duration)
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"coordinate": (int(coord.x), int(coord.y)), "hover_duration": duration},
        )
    
    async def _handle_drag(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle drag operation."""
        start_coord = await self._get_coordinate(action.target)
        if start_coord is None:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Could not resolve drag start target",
            )
        
        end_coords = action.params.get("end")
        if not end_coords:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Missing drag end coordinates",
            )
        
        if isinstance(end_coords, tuple):
            end_coord = POPSSCMUCoordinate(x=end_coords[0], y=end_coords[1])
        elif isinstance(end_coords, POPSSCMUCoordinate):
            end_coord = end_coords
        else:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message="Invalid drag end coordinates",
            )
        
        duration = action.params.get("duration", self.mouse_config.move_duration)
        
        await self.move_to(start_coord.x, start_coord.y, duration=duration / 2)
        pyautogui.drag(
            end_coord.x - start_coord.x,
            end_coord.y - start_coord.y,
            duration=duration / 2,
            button="left",
        )
        
        self._current_position = (int(end_coord.x), int(end_coord.y))
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={
                "start": (int(start_coord.x), int(start_coord.y)),
                "end": (int(end_coord.x), int(end_coord.y)),
                "duration": duration,
            },
        )
    
    async def _handle_scroll(self, action: POPSSCMUAction) -> POPSSCMUActionResult:
        """Handle scroll operation."""
        direction = action.params.get("direction", "down")
        amount = action.params.get("amount", self.mouse_config.scroll_amount)
        
        if direction == "up":
            pyautogui.scroll(amount)
        elif direction == "down":
            pyautogui.scroll(-amount)
        elif direction == "left":
            pyautogui.hscroll(amount)
        elif direction == "right":
            pyautogui.hscroll(-amount)
        else:
            return POPSSCMUActionResult(
                action_id=action.action_id,
                status=POPSSCMUActionResultStatus.FAILED,
                error_message=f"Invalid scroll direction: {direction}",
            )
        
        return POPSSCMUActionResult(
            action_id=action.action_id,
            status=POPSSCMUActionResultStatus.SUCCESS,
            output={"direction": direction, "amount": amount},
        )
    
    async def move_to(
        self,
        x: float,
        y: float,
        duration: float = 0.0,
    ) -> None:
        """Move mouse to coordinates with optional smooth movement."""
        if not self.mouse_config.smooth_movement or duration <= 0:
            pyautogui.moveTo(int(x), int(y))
            self._current_position = (int(x), int(y))
            return
        
        start_x, start_y = pyautogui.position()
        steps = int(duration * 60)
        
        for i in range(steps + 1):
            t = i / steps
            if self.mouse_config.movement_curve == "ease_in_out":
                t = self._ease_in_out(t)
            elif self.mouse_config.movement_curve == "ease_in":
                t = self._ease_in(t)
            elif self.mouse_config.movement_curve == "ease_out":
                t = self._ease_out(t)
            
            new_x = start_x + (x - start_x) * t
            new_y = start_y + (y - start_y) * t
            
            pyautogui.moveTo(int(new_x), int(new_y))
            await asyncio.sleep(duration / steps)
        
        self._current_position = (int(x), int(y))
    
    def _ease_in_out(self, t: float) -> float:
        """Ease in-out easing function."""
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2
    
    def _ease_in(self, t: float) -> float:
        """Ease in easing function."""
        return t * t
    
    def _ease_out(self, t: float) -> float:
        """Ease out easing function."""
        return t * (2 - t)
    
    async def _get_coordinate(self, target: Any) -> Optional[POPSSCMUCoordinate]:
        """Get coordinate from target."""
        if hasattr(target, "coordinate") and target.coordinate:
            return target.coordinate
        
        if hasattr(target, "rectangle") and target.rectangle:
            return target.rectangle.center
        
        return None
    
    def get_current_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        if _HAS_PYAUTOGUI:
            return pyautogui.position()
        return self._current_position
    
    async def get_position(self) -> Tuple[int, int]:
        """Get current mouse position (async)."""
        return self.get_current_position()
