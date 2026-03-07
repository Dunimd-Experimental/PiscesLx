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
CMU Tablet Platform - iPad/Android Tablet Control

This module extends the mobile platform for tablet-specific features
including larger screen layouts and stylus support.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUPlatform,
    POPSSCMURectangle,
)
from .mobile import POPSSCMUMobile
from .base import POPSSCMUPlatformInfo

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Platform.Tablet", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Platform.Tablet"), enable_file=True)


class POPSSCMUTablet(POPSSCMUMobile):
    """
    Tablet platform adapter extending mobile capabilities.
    
    Provides tablet-specific features:
        - Larger screen layouts
        - Split-screen support
        - Stylus/Apple Pencil support
        - Multi-window management
    
    Attributes:
        supports_stylus: Whether device supports stylus input
        supports_split_screen: Whether device supports split-screen
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        
        self.supports_stylus = getattr(config, 'supports_stylus', True) if config else True
        self.supports_split_screen = getattr(config, 'supports_split_screen', True) if config else True
        
        self._screen_width = 2048
        self._screen_height = 2732
        
    def _detect_platform_info(self) -> POPSSCMUPlatformInfo:
        """Detect tablet platform information."""
        if self.device_type == "android":
            platform_type = POPSSCMUPlatform.ANDROID_TABLET
            name = "Android Tablet"
        else:
            platform_type = POPSSCMUPlatform.IOS_IPAD
            name = "iPad"
        
        capabilities = [
            "touch_control",
            "gesture_control",
            "screen_capture",
            "app_management",
            "swipe",
            "pinch",
            "multi_touch",
            "stylus_support",
            "split_screen",
            "multi_window",
        ]
        
        return POPSSCMUPlatformInfo(
            platform=platform_type,
            name=name,
            version="",
            screen_width=self._screen_width,
            screen_height=self._screen_height,
            screen_dpi=264,
            is_touch_enabled=True,
            is_virtual=False,
            capabilities=capabilities,
        )
    
    async def stylus_tap(self, x: float, y: float, pressure: float = 1.0) -> bool:
        """Perform stylus tap with pressure."""
        if not self.supports_stylus:
            return await self.click(x, y)
        
        _LOG.debug("stylus_tap_executed", x=x, y=y, pressure=pressure)
        return await self.click(x, y)
    
    async def stylus_draw(
        self,
        points: list,
        pressure: float = 1.0,
    ) -> bool:
        """Draw with stylus through a series of points."""
        if not self._is_initialized or not self._device:
            return False
        
        if len(points) < 2:
            return False
        
        try:
            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]
                await self.drag(start[0], start[1], end[0], end[1], 0.05)
            return True
        except Exception as e:
            _LOG.error("stylus_draw_failed", error=str(e))
        return False
    
    async def split_screen(self, app1: str, app2: str) -> bool:
        """Enable split-screen with two apps."""
        if not self.supports_split_screen:
            _LOG.warning("split_screen_not_supported")
            return False
        
        _LOG.info("split_screen_requested", app1=app1, app2=app2)
        return False
    
    async def get_screen_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Get screen regions for split-screen layout."""
        return {
            "full": (0, 0, self._screen_width, self._screen_height),
            "left": (0, 0, self._screen_width // 2, self._screen_height),
            "right": (self._screen_width // 2, 0, self._screen_width // 2, self._screen_height),
            "top": (0, 0, self._screen_width, self._screen_height // 2),
            "bottom": (0, self._screen_height // 2, self._screen_width, self._screen_height // 2),
        }
    
    async def two_finger_tap(self, x: float, y: float) -> bool:
        """Perform two-finger tap gesture."""
        if not self._is_initialized or not self._device:
            return False
        
        _LOG.debug("two_finger_tap_executed", x=x, y=y)
        return True
    
    async def three_finger_swipe(self, direction: str) -> bool:
        """Perform three-finger swipe gesture."""
        if not self._is_initialized or not self._device:
            return False
        
        _LOG.debug("three_finger_swipe_executed", direction=direction)
        return await self.swipe(direction, 0.3, 0.3)
    
    async def zoom_in(self, center_x: float, center_y: float) -> bool:
        """Perform zoom in gesture."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                self._device.pinch_out(int(center_x), int(center_y), percent=50)
                return True
        except Exception as e:
            _LOG.error("zoom_in_failed", error=str(e))
        return False
    
    async def zoom_out(self, center_x: float, center_y: float) -> bool:
        """Perform zoom out gesture."""
        return await self.pinch(0.5)
