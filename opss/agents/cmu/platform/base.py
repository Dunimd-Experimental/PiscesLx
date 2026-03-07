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
CMU Platform Base - Abstract Platform Adapter

This module provides the abstract base class for all platform adapters,
defining the unified interface for cross-platform device control.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUAction,
    POPSSCMUActionResult,
    POPSSCMUPlatform,
    POPSSCMUScreenState,
    POPSSCMURectangle,
    POPSSCMUCoordinate,
)

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Platform.Base", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Platform.Base"), enable_file=True)


@dataclass
class POPSSCMUPlatformInfo:
    """Platform information container."""
    platform: POPSSCMUPlatform = POPSSCMUPlatform.UNKNOWN
    name: str = "Unknown"
    version: str = ""
    screen_width: int = 1920
    screen_height: int = 1080
    screen_dpi: int = 96
    is_touch_enabled: bool = False
    is_virtual: bool = False
    capabilities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value,
            "name": self.name,
            "version": self.version,
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "screen_dpi": self.screen_dpi,
            "is_touch_enabled": self.is_touch_enabled,
            "is_virtual": self.is_virtual,
            "capabilities": self.capabilities,
        }


class POPSSCMUPlatformAdapter(ABC):
    """
    Abstract base class for platform adapters.
    
    Defines the unified interface for cross-platform device control,
    including mouse, keyboard, touch, and screen operations.
    
    Subclasses must implement all abstract methods for their specific platform.
    """
    
    def __init__(self, config: Any = None):
        self.config = config
        self._platform_info: Optional[POPSSCMUPlatformInfo] = None
        self._is_initialized = False
        self._mouse_position: Tuple[int, int] = (0, 0)
        self._keyboard_state: Dict[str, bool] = {}
        
    @property
    def platform_info(self) -> POPSSCMUPlatformInfo:
        """Get platform information."""
        if self._platform_info is None:
            self._platform_info = self._detect_platform_info()
        return self._platform_info
    
    @abstractmethod
    def _detect_platform_info(self) -> POPSSCMUPlatformInfo:
        """Detect and return platform information."""
        raise NotImplementedError("Subclasses must implement _detect_platform_info()")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the platform adapter."""
        raise NotImplementedError("Subclasses must implement initialize()")
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the platform adapter."""
        raise NotImplementedError("Subclasses must implement shutdown()")
    
    @abstractmethod
    async def click(self, x: float, y: float, button: str = "left", clicks: int = 1) -> bool:
        """
        Perform a click at the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button ("left", "right", "middle")
            clicks: Number of clicks
        
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement click()")
    
    @abstractmethod
    async def double_click(self, x: float, y: float) -> bool:
        """Perform a double click at the specified coordinates."""
        raise NotImplementedError("Subclasses must implement double_click()")
    
    @abstractmethod
    async def right_click(self, x: float, y: float) -> bool:
        """Perform a right click at the specified coordinates."""
        raise NotImplementedError("Subclasses must implement right_click()")
    
    @abstractmethod
    async def move_to(self, x: float, y: float, duration: float = 0.0) -> bool:
        """
        Move cursor to the specified coordinates.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Duration of movement in seconds (0 for instant)
        
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement move_to()")
    
    @abstractmethod
    async def drag(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        duration: float = 0.5,
    ) -> bool:
        """
        Perform a drag operation from start to end coordinates.
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration: Duration of drag in seconds
        
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement drag()")
    
    @abstractmethod
    async def scroll(self, direction: str, amount: int = 3) -> bool:
        """
        Perform a scroll operation.
        
        Args:
            direction: Scroll direction ("up", "down", "left", "right")
            amount: Number of scroll clicks
        
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement scroll()")
    
    @abstractmethod
    async def type_text(self, text: str, interval: float = 0.05) -> bool:
        """
        Type text at the current cursor position.
        
        Args:
            text: Text to type
            interval: Interval between keystrokes in seconds
        
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement type_text()")
    
    @abstractmethod
    async def key_press(self, key: str) -> bool:
        """
        Press and release a single key.
        
        Args:
            key: Key to press (e.g., "enter", "tab", "escape")
        
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement key_press()")
    
    @abstractmethod
    async def key_down(self, key: str) -> bool:
        """Press and hold a key."""
        raise NotImplementedError("Subclasses must implement key_down()")
    
    @abstractmethod
    async def key_up(self, key: str) -> bool:
        """Release a held key."""
        raise NotImplementedError("Subclasses must implement key_up()")
    
    @abstractmethod
    async def hotkey(self, *keys: str) -> bool:
        """
        Press a keyboard hotkey combination.
        
        Args:
            *keys: Keys to press together (e.g., "ctrl", "c")
        
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclasses must implement hotkey()")
    
    @abstractmethod
    async def capture_screen(self, region: Optional[POPSSCMURectangle] = None) -> Optional[Any]:
        """
        Capture the screen or a region.
        
        Args:
            region: Optional region to capture
        
        Returns:
            Captured image data
        """
        raise NotImplementedError("Subclasses must implement capture_screen()")
    
    @abstractmethod
    async def get_cursor_position(self) -> Tuple[int, int]:
        """Get current cursor position."""
        raise NotImplementedError("Subclasses must implement get_cursor_position()")
    
    @abstractmethod
    async def get_active_window(self) -> Dict[str, Any]:
        """Get information about the active window."""
        raise NotImplementedError("Subclasses must implement get_active_window()")
    
    @abstractmethod
    async def set_clipboard(self, text: str) -> bool:
        """Set clipboard text content."""
        raise NotImplementedError("Subclasses must implement set_clipboard()")
    
    @abstractmethod
    async def get_clipboard(self) -> str:
        """Get clipboard text content."""
        raise NotImplementedError("Subclasses must implement get_clipboard()")
    
    async def swipe(
        self,
        direction: str,
        distance: float = 0.3,
        duration: float = 0.3,
    ) -> bool:
        """
        Perform a swipe gesture (for touch-enabled platforms).
        
        Args:
            direction: Swipe direction
            distance: Distance as fraction of screen (0-1)
            duration: Duration of swipe in seconds
        
        Returns:
            bool: True if successful
        """
        _LOG.warning("swipe_not_supported_on_platform", platform=self.platform_info.platform.value)
        return False
    
    async def pinch(self, scale: float = 1.0) -> bool:
        """Perform a pinch gesture (for touch-enabled platforms)."""
        _LOG.warning("pinch_not_supported_on_platform", platform=self.platform_info.platform.value)
        return False
    
    async def tap(self, x: float, y: float) -> bool:
        """Perform a tap gesture (for touch-enabled platforms)."""
        return await self.click(x, y)
    
    async def long_press(self, x: float, y: float, duration: float = 1.0) -> bool:
        """Perform a long press gesture."""
        await self.move_to(x, y)
        await asyncio.sleep(duration)
        return True
    
    def is_initialized(self) -> bool:
        """Check if adapter is initialized."""
        return self._is_initialized
    
    def get_capabilities(self) -> List[str]:
        """Get list of platform capabilities."""
        return self.platform_info.capabilities
    
    def has_capability(self, capability: str) -> bool:
        """Check if platform has a specific capability."""
        return capability in self.platform_info.capabilities
