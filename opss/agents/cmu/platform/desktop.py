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
CMU Desktop Platform - Windows/macOS/Linux Desktop Control

This module provides desktop platform control using PyAutoGUI and
platform-specific APIs for comprehensive mouse and keyboard control.
"""

from __future__ import annotations

import asyncio
import platform as sys_platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUPlatform,
    POPSSCMURectangle,
)
from .base import POPSSCMUPlatformAdapter, POPSSCMUPlatformInfo

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Platform.Desktop", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Platform.Desktop"), enable_file=True)

_HAS_PYAUTOGUI = False
_HAS_PIL = False
_HAS_PYWIN32 = False

try:
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
    _HAS_PYAUTOGUI = True
except ImportError:
    _LOG.warning("pyautogui_not_available")

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _LOG.warning("pillow_not_available")

if sys_platform.system().lower() == "windows":
    try:
        import win32api
        import win32con
        import win32gui
        _HAS_PYWIN32 = True
    except ImportError:
        _LOG.warning("pywin32_not_available")


class POPSSCMUDesktop(POPSSCMUPlatformAdapter):
    """
    Desktop platform adapter for Windows, macOS, and Linux.
    
    Provides comprehensive mouse and keyboard control using PyAutoGUI
    with platform-specific enhancements.
    
    Features:
        - Mouse control (click, move, drag, scroll)
        - Keyboard control (type, hotkey, special keys)
        - Screen capture
        - Window management
        - Clipboard operations
    
    Attributes:
        mouse_speed: Speed factor for mouse movements
        typing_speed: Interval between keystrokes
        scroll_amount: Default scroll amount
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        
        self.mouse_speed = getattr(config, 'mouse_speed', 0.3) if config else 0.3
        self.typing_speed = getattr(config, 'typing_speed', 0.05) if config else 0.05
        self.scroll_amount = getattr(config, 'scroll_amount', 3) if config else 3
        
        self._screen_width = 0
        self._screen_height = 0
        
    def _detect_platform_info(self) -> POPSSCMUPlatformInfo:
        """Detect desktop platform information."""
        system = sys_platform.system().lower()
        
        if system == "windows":
            platform_type = POPSSCMUPlatform.DESKTOP_WINDOWS
            name = "Windows Desktop"
            version = sys_platform.version()
        elif system == "darwin":
            platform_type = POPSSCMUPlatform.DESKTOP_MACOS
            name = "macOS Desktop"
            version = sys_platform.mac_ver()[0]
        elif system == "linux":
            platform_type = POPSSCMUPlatform.DESKTOP_LINUX
            name = "Linux Desktop"
            version = sys_platform.freedesktop_os_release().get('VERSION_ID', 'Unknown')
        else:
            platform_type = POPSSCMUPlatform.UNKNOWN
            name = "Unknown Desktop"
            version = "Unknown"
        
        screen_width, screen_height = 1920, 1080
        if _HAS_PYAUTOGUI:
            screen_width, screen_height = pyautogui.size()
        
        self._screen_width = screen_width
        self._screen_height = screen_height
        
        capabilities = [
            "mouse_control",
            "keyboard_control",
            "screen_capture",
            "clipboard",
            "window_management",
        ]
        
        if _HAS_PYWIN32 and system == "windows":
            capabilities.extend(["win32_api", "window_hooks"])
        
        return POPSSCMUPlatformInfo(
            platform=platform_type,
            name=name,
            version=version,
            screen_width=screen_width,
            screen_height=screen_height,
            screen_dpi=self._get_dpi(),
            is_touch_enabled=False,
            is_virtual=False,
            capabilities=capabilities,
        )
    
    def _get_dpi(self) -> int:
        """Get screen DPI."""
        if _HAS_PYWIN32 and sys_platform.system().lower() == "windows":
            try:
                dc = win32gui.GetDC(0)
                dpi = win32api.GetDeviceCaps(dc, win32con.LOGPIXELSX)
                win32gui.ReleaseDC(0, dc)
                return dpi
            except Exception as e:
                _LOG.warning("dpi_detection_failed", error=str(e), fallback=96)
        return 96
    
    async def initialize(self) -> bool:
        """Initialize the desktop platform adapter."""
        if not _HAS_PYAUTOGUI:
            _LOG.error("pyautogui_required_for_desktop")
            return False
        
        try:
            self._screen_width, self._screen_height = pyautogui.size()
            self._is_initialized = True
            _LOG.info(
                "desktop_adapter_initialized",
                screen_size=(self._screen_width, self._screen_height),
            )
            return True
        except Exception as e:
            _LOG.error("desktop_adapter_init_failed", error=str(e))
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the desktop platform adapter."""
        self._is_initialized = False
        _LOG.info("desktop_adapter_shutdown")
    
    async def click(self, x: float, y: float, button: str = "left", clicks: int = 1) -> bool:
        """Perform a click at the specified coordinates."""
        if not self._is_initialized:
            return False
        
        try:
            if _HAS_PYAUTOGUI:
                pyautogui.click(x=int(x), y=int(y), clicks=clicks, button=button)
                self._mouse_position = (int(x), int(y))
                _LOG.debug("click_executed", x=x, y=y, button=button, clicks=clicks)
                return True
        except Exception as e:
            _LOG.error("click_failed", error=str(e))
        return False
    
    async def double_click(self, x: float, y: float) -> bool:
        """Perform a double click."""
        return await self.click(x, y, button="left", clicks=2)
    
    async def right_click(self, x: float, y: float) -> bool:
        """Perform a right click."""
        return await self.click(x, y, button="right", clicks=1)
    
    async def move_to(self, x: float, y: float, duration: float = 0.0) -> bool:
        """Move cursor to the specified coordinates."""
        if not self._is_initialized:
            return False
        
        try:
            if _HAS_PYAUTOGUI:
                actual_duration = duration if duration > 0 else self.mouse_speed
                pyautogui.moveTo(x=int(x), y=int(y), duration=actual_duration)
                self._mouse_position = (int(x), int(y))
                _LOG.debug("move_to_executed", x=x, y=y)
                return True
        except Exception as e:
            _LOG.error("move_to_failed", error=str(e))
        return False
    
    async def drag(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        duration: float = 0.5,
    ) -> bool:
        """Perform a drag operation."""
        if not self._is_initialized:
            return False
        
        try:
            if _HAS_PYAUTOGUI:
                pyautogui.moveTo(int(start_x), int(start_y))
                pyautogui.drag(
                    end_x - start_x,
                    end_y - start_y,
                    duration=duration,
                    button="left",
                )
                self._mouse_position = (int(end_x), int(end_y))
                _LOG.debug("drag_executed", start=(start_x, start_y), end=(end_x, end_y))
                return True
        except Exception as e:
            _LOG.error("drag_failed", error=str(e))
        return False
    
    async def scroll(self, direction: str, amount: int = 3) -> bool:
        """Perform a scroll operation."""
        if not self._is_initialized:
            return False
        
        try:
            if _HAS_PYAUTOGUI:
                scroll_amount = amount if direction in ("up", "right") else -amount
                if direction in ("left", "right"):
                    pyautogui.hscroll(scroll_amount)
                else:
                    pyautogui.scroll(scroll_amount)
                _LOG.debug("scroll_executed", direction=direction, amount=amount)
                return True
        except Exception as e:
            _LOG.error("scroll_failed", error=str(e))
        return False
    
    async def type_text(self, text: str, interval: float = 0.05) -> bool:
        """Type text at the current cursor position."""
        if not self._is_initialized:
            return False
        
        try:
            if _HAS_PYAUTOGUI:
                actual_interval = interval if interval > 0 else self.typing_speed
                pyautogui.typewrite(text, interval=actual_interval)
                _LOG.debug("type_text_executed", text_length=len(text))
                return True
        except Exception as e:
            _LOG.error("type_text_failed", error=str(e))
        return False
    
    async def key_press(self, key: str) -> bool:
        """Press and release a single key."""
        if not self._is_initialized:
            return False
        
        try:
            if _HAS_PYAUTOGUI:
                pyautogui.press(key)
                _LOG.debug("key_press_executed", key=key)
                return True
        except Exception as e:
            _LOG.error("key_press_failed", error=str(e), key=key)
        return False
    
    async def key_down(self, key: str) -> bool:
        """Press and hold a key."""
        if not self._is_initialized:
            return False
        
        try:
            if _HAS_PYAUTOGUI:
                pyautogui.keyDown(key)
                self._keyboard_state[key] = True
                _LOG.debug("key_down_executed", key=key)
                return True
        except Exception as e:
            _LOG.error("key_down_failed", error=str(e), key=key)
        return False
    
    async def key_up(self, key: str) -> bool:
        """Release a held key."""
        if not self._is_initialized:
            return False
        
        try:
            if _HAS_PYAUTOGUI:
                pyautogui.keyUp(key)
                self._keyboard_state[key] = False
                _LOG.debug("key_up_executed", key=key)
                return True
        except Exception as e:
            _LOG.error("key_up_failed", error=str(e), key=key)
        return False
    
    async def hotkey(self, *keys: str) -> bool:
        """Press a keyboard hotkey combination."""
        if not self._is_initialized:
            return False
        
        try:
            if _HAS_PYAUTOGUI:
                pyautogui.hotkey(*keys)
                _LOG.debug("hotkey_executed", keys=keys)
                return True
        except Exception as e:
            _LOG.error("hotkey_failed", error=str(e), keys=keys)
        return False
    
    async def capture_screen(self, region: Optional[POPSSCMURectangle] = None) -> Optional[Any]:
        """Capture the screen or a region."""
        if not self._is_initialized:
            return None
        
        try:
            if _HAS_PYAUTOGUI and _HAS_PIL:
                if region:
                    screenshot = pyautogui.screenshot(region=(
                        int(region.x),
                        int(region.y),
                        int(region.width),
                        int(region.height),
                    ))
                else:
                    screenshot = pyautogui.screenshot()
                _LOG.debug("screen_captured", region=region is not None)
                return screenshot
        except Exception as e:
            _LOG.error("screen_capture_failed", error=str(e))
        return None
    
    async def get_cursor_position(self) -> Tuple[int, int]:
        """Get current cursor position."""
        if _HAS_PYAUTOGUI:
            return pyautogui.position()
        return self._mouse_position
    
    async def get_active_window(self) -> Dict[str, Any]:
        """Get information about the active window."""
        result = {
            "title": "",
            "bounds": None,
            "process_id": None,
        }
        
        if _HAS_PYWIN32 and sys_platform.system().lower() == "windows":
            try:
                hwnd = win32gui.GetForegroundWindow()
                result["title"] = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                result["bounds"] = {
                    "x": rect[0],
                    "y": rect[1],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1],
                }
                _, process_id = win32process.GetWindowThreadProcessId(hwnd)
                result["process_id"] = process_id
            except Exception as e:
                _LOG.error("get_active_window_failed", error=str(e))
        
        return result
    
    async def set_clipboard(self, text: str) -> bool:
        """Set clipboard text content."""
        try:
            if _HAS_PYAUTOGUI:
                pyautogui.copy(text)
                _LOG.debug("clipboard_set", text_length=len(text))
                return True
        except Exception as e:
            _LOG.error("set_clipboard_failed", error=str(e))
        
        if _HAS_PYWIN32 and sys_platform.system().lower() == "windows":
            try:
                import win32clipboard
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text)
                win32clipboard.CloseClipboard()
                return True
            except Exception as e:
                _LOG.warning("win32_clipboard_set_failed", error=str(e))
        
        return False
    
    async def get_clipboard(self) -> str:
        """Get clipboard text content."""
        try:
            if _HAS_PYAUTOGUI:
                return pyautogui.paste()
        except Exception as e:
            _LOG.error("get_clipboard_failed", error=str(e))
        
        if _HAS_PYWIN32 and sys_platform.system().lower() == "windows":
            try:
                import win32clipboard
                win32clipboard.OpenClipboard()
                text = win32clipboard.GetClipboardData()
                win32clipboard.CloseClipboard()
                return text
            except Exception as e:
                _LOG.warning("win32_clipboard_get_failed", error=str(e))
        
        return ""
    
    async def launch_application(self, app_name: str) -> bool:
        """Launch an application by name."""
        system = sys_platform.system().lower()
        
        try:
            if system == "windows":
                subprocess.Popen(["start", app_name], shell=True)
            elif system == "darwin":
                subprocess.Popen(["open", "-a", app_name])
            elif system == "linux":
                subprocess.Popen([app_name])
            
            _LOG.info("application_launched", app_name=app_name)
            return True
        except Exception as e:
            _LOG.error("launch_application_failed", app_name=app_name, error=str(e))
            return False
    
    async def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size."""
        if _HAS_PYAUTOGUI:
            return pyautogui.size()
        return (self._screen_width, self._screen_height)
    
    async def move_to_relative(self, x_ratio: float, y_ratio: float, duration: float = 0.0) -> bool:
        """Move cursor to relative position (0-1)."""
        x = int(x_ratio * self._screen_width)
        y = int(y_ratio * self._screen_height)
        return await self.move_to(x, y, duration)
