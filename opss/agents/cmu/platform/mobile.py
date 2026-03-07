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
CMU Mobile Platform - Android/iOS Mobile Device Control

This module provides mobile device control using ADB for Android
and WebDriverAgent/IDB for iOS devices.
"""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUPlatform,
    POPSSCMURectangle,
)
from .base import POPSSCMUPlatformAdapter, POPSSCMUPlatformInfo

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Platform.Mobile", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Platform.Mobile"), enable_file=True)

_HAS_ADB = False
_HAS_UIAUTOMATOR2 = False

try:
    import adb_shell
    from adb_shell.adb_device import AdbDeviceTcp
    _HAS_ADB = True
except ImportError:
    _LOG.warning("adb_shell_not_available")

try:
    import uiautomator2 as u2
    _HAS_UIAUTOMATOR2 = True
except ImportError:
    _LOG.warning("uiautomator2_not_available")


class POPSSCMUMobile(POPSSCMUPlatformAdapter):
    """
    Mobile platform adapter for Android and iOS devices.
    
    Provides comprehensive mobile device control including:
        - Touch gestures (tap, swipe, pinch)
        - App management
        - Screen capture
        - Device information
    
    Attributes:
        device_id: Device identifier
        device_type: "android" or "ios"
        screen_width: Screen width
        screen_height: Screen height
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        
        self.device_id = getattr(config, 'device_id', None) if config else None
        self.device_type = getattr(config, 'device_type', 'android') if config else 'android'
        
        self._device: Optional[Any] = None
        self._screen_width = 1080
        self._screen_height = 1920
        
    def _detect_platform_info(self) -> POPSSCMUPlatformInfo:
        """Detect mobile platform information."""
        if self.device_type == "android":
            platform_type = POPSSCMUPlatform.ANDROID_PHONE
            name = "Android Phone"
        else:
            platform_type = POPSSCMUPlatform.IOS_IPHONE
            name = "iOS iPhone"
        
        capabilities = [
            "touch_control",
            "gesture_control",
            "screen_capture",
            "app_management",
            "swipe",
            "pinch",
            "multi_touch",
        ]
        
        return POPSSCMUPlatformInfo(
            platform=platform_type,
            name=name,
            version="",
            screen_width=self._screen_width,
            screen_height=self._screen_height,
            screen_dpi=320,
            is_touch_enabled=True,
            is_virtual=False,
            capabilities=capabilities,
        )
    
    async def initialize(self) -> bool:
        """Initialize the mobile device connection."""
        if self.device_type == "android":
            return await self._initialize_android()
        else:
            return await self._initialize_ios()
    
    async def _initialize_android(self) -> bool:
        """Initialize Android device via uiautomator2."""
        if not _HAS_UIAUTOMATOR2:
            _LOG.error("uiautomator2_required_for_android")
            return False
        
        try:
            if self.device_id:
                self._device = u2.connect(self.device_id)
            else:
                self._device = u2.connect()
            
            info = self._device.info
            self._screen_width = info['displayWidth']
            self._screen_height = info['displayHeight']
            
            self._is_initialized = True
            _LOG.info(
                "android_device_initialized",
                screen_size=(self._screen_width, self._screen_height),
            )
            return True
            
        except Exception as e:
            _LOG.error("android_init_failed", error=str(e))
            return False
    
    async def _initialize_ios(self) -> bool:
        """Initialize iOS device."""
        _LOG.warning("ios_support_not_implemented")
        return False
    
    async def shutdown(self) -> None:
        """Shutdown the mobile device connection."""
        self._is_initialized = False
        _LOG.info("mobile_adapter_shutdown")
    
    async def click(self, x: float, y: float, button: str = "left", clicks: int = 1) -> bool:
        """Tap at coordinates."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                self._device.click(int(x), int(y))
                _LOG.debug("android_tap_executed", x=x, y=y)
                return True
        except Exception as e:
            _LOG.error("mobile_click_failed", error=str(e))
        return False
    
    async def double_click(self, x: float, y: float) -> bool:
        """Double tap at coordinates."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                self._device.double_click(int(x), int(y))
                return True
        except Exception as e:
            _LOG.error("mobile_double_click_failed", error=str(e))
        return False
    
    async def right_click(self, x: float, y: float) -> bool:
        """Long press (right click equivalent)."""
        return await self.long_press(x, y, 1.0)
    
    async def move_to(self, x: float, y: float, duration: float = 0.0) -> bool:
        """Move touch (not applicable for mobile)."""
        return True
    
    async def drag(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        duration: float = 0.5,
    ) -> bool:
        """Perform drag gesture."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                self._device.drag(int(start_x), int(start_y), int(end_x), int(end_y), duration)
                return True
        except Exception as e:
            _LOG.error("mobile_drag_failed", error=str(e))
        return False
    
    async def scroll(self, direction: str, amount: int = 3) -> bool:
        """Scroll by swiping."""
        if not self._is_initialized:
            return False
        
        center_x = self._screen_width // 2
        center_y = self._screen_height // 2
        swipe_distance = self._screen_height // 3
        
        if direction == "up":
            start_y = center_y + swipe_distance // 2
            end_y = center_y - swipe_distance // 2
        elif direction == "down":
            start_y = center_y - swipe_distance // 2
            end_y = center_y + swipe_distance // 2
        elif direction == "left":
            return await self.swipe("right", 0.5, 0.3)
        else:
            return await self.swipe("left", 0.5, 0.3)
        
        return await self.drag(center_x, start_y, center_x, end_y, 0.3)
    
    async def type_text(self, text: str, interval: float = 0.05) -> bool:
        """Type text using keyboard."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                self._device.send_keys(text)
                return True
        except Exception as e:
            _LOG.error("mobile_type_failed", error=str(e))
        return False
    
    async def key_press(self, key: str) -> bool:
        """Press a key."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                key_map = {
                    "enter": "enter",
                    "back": "back",
                    "home": "home",
                    "menu": "menu",
                    "search": "search",
                }
                android_key = key_map.get(key.lower(), key)
                self._device.press(android_key)
                return True
        except Exception as e:
            _LOG.error("mobile_key_press_failed", error=str(e))
        return False
    
    async def key_down(self, key: str) -> bool:
        """Press and hold a key."""
        return await self.key_press(key)
    
    async def key_up(self, key: str) -> bool:
        """Release a key."""
        return True
    
    async def hotkey(self, *keys: str) -> bool:
        """Press hotkey combination."""
        for key in keys:
            await self.key_press(key)
        return True
    
    async def capture_screen(self, region: Optional[POPSSCMURectangle] = None) -> Optional[Any]:
        """Capture screenshot."""
        if not self._is_initialized or not self._device:
            return None
        
        try:
            if self.device_type == "android":
                img = self._device.screenshot()
                if region:
                    img = img.crop((
                        int(region.x),
                        int(region.y),
                        int(region.x + region.width),
                        int(region.y + region.height),
                    ))
                return img
        except Exception as e:
            _LOG.error("mobile_screenshot_failed", error=str(e))
        return None
    
    async def get_cursor_position(self) -> Tuple[int, int]:
        """Get cursor position (not applicable)."""
        return (0, 0)
    
    async def get_active_window(self) -> Dict[str, Any]:
        """Get active app info."""
        if not self._is_initialized or not self._device:
            return {}
        
        try:
            if self.device_type == "android":
                current_app = self._device.app_current()
                return {
                    "package": current_app.get("package", ""),
                    "activity": current_app.get("activity", ""),
                }
        except Exception as e:
            _LOG.warning("get_active_window_failed", error=str(e))
        return {}
    
    async def set_clipboard(self, text: str) -> bool:
        """Set clipboard text."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                self._device.set_clipboard(text)
                return True
        except Exception as e:
            _LOG.warning("set_clipboard_failed", error=str(e))
        return False
    
    async def get_clipboard(self) -> str:
        """Get clipboard text."""
        if not self._is_initialized or not self._device:
            return ""
        
        try:
            if self.device_type == "android":
                return self._device.get_clipboard()
        except Exception as e:
            _LOG.warning("get_clipboard_failed", error=str(e))
        return ""
    
    async def swipe(
        self,
        direction: str,
        distance: float = 0.3,
        duration: float = 0.3,
    ) -> bool:
        """Perform swipe gesture."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            center_x = self._screen_width // 2
            center_y = self._screen_height // 2
            swipe_distance = int(self._screen_height * distance)
            
            direction_coords = {
                "up": (center_x, center_y + swipe_distance // 2, center_x, center_y - swipe_distance // 2),
                "down": (center_x, center_y - swipe_distance // 2, center_x, center_y + swipe_distance // 2),
                "left": (center_x + swipe_distance // 2, center_y, center_x - swipe_distance // 2, center_y),
                "right": (center_x - swipe_distance // 2, center_y, center_x + swipe_distance // 2, center_y),
            }
            
            if direction in direction_coords:
                sx, sy, ex, ey = direction_coords[direction]
                self._device.swipe(sx, sy, ex, ey, duration)
                return True
        except Exception as e:
            _LOG.error("mobile_swipe_failed", error=str(e))
        return False
    
    async def pinch(self, scale: float = 1.0) -> bool:
        """Perform pinch gesture."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                center_x = self._screen_width // 2
                center_y = self._screen_height // 2
                self._device.pinch_in(center_x, center_y, percent=50)
                return True
        except Exception as e:
            _LOG.error("mobile_pinch_failed", error=str(e))
        return False
    
    async def long_press(self, x: float, y: float, duration: float = 1.0) -> bool:
        """Perform long press."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                self._device.long_click(int(x), int(y), duration)
                return True
        except Exception as e:
            _LOG.error("mobile_long_press_failed", error=str(e))
        return False
    
    async def launch_app(self, package_name: str) -> bool:
        """Launch an app by package name."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                self._device.app_start(package_name)
                return True
        except Exception as e:
            _LOG.error("launch_app_failed", package=package_name, error=str(e))
        return False
    
    async def stop_app(self, package_name: str) -> bool:
        """Stop an app by package name."""
        if not self._is_initialized or not self._device:
            return False
        
        try:
            if self.device_type == "android":
                self._device.app_stop(package_name)
                return True
        except Exception as e:
            _LOG.error("stop_app_failed", package=package_name, error=str(e))
        return False
    
    async def get_installed_apps(self) -> List[str]:
        """Get list of installed apps."""
        if not self._is_initialized or not self._device:
            return []
        
        try:
            if self.device_type == "android":
                return self._device.app_list()
        except Exception as e:
            _LOG.warning("get_installed_apps_failed", error=str(e))
        return []
