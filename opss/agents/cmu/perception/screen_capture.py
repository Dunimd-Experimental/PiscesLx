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
CMU Screen Capture - Cross-Platform Screen Capture Module

This module provides screen capture capabilities for the Computer Use Agent,
supporting multiple platforms and capture modes.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUPlatform,
    POPSSCMURectangle,
    POPSSCMUScreenState,
    POPSSCMUElement,
)

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Perception.ScreenCapture", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Perception.ScreenCapture"), enable_file=True)

_HAS_MSS = False
_HAS_PIL = False

try:
    import mss
    import mss.tools
    _HAS_MSS = True
except ImportError:
    _LOG.warning("mss_not_available")

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _LOG.warning("pillow_not_available")


@dataclass
class POPSSCMUCaptureConfig:
    """Screen capture configuration."""
    screenshot_dir: str = "./cmu_screenshots"
    save_screenshots: bool = True
    compress_quality: int = 85
    max_cache_size: int = 100
    capture_delay: float = 0.1


class POPSSCMUScreenCapture:
    """
    Cross-platform screen capture module.
    
    Provides efficient screen capture with caching and multi-monitor support.
    
    Features:
        - Multi-platform support (Windows, macOS, Linux)
        - Region capture
        - Multi-monitor support
        - Screenshot caching
        - Automatic file management
    
    Attributes:
        config: Capture configuration
        _mss: MSS instance for capture
        _cache: Screenshot cache
        _monitor_info: Monitor information
    """
    
    def __init__(self, config: Any = None):
        self.config = config or POPSSCMUCaptureConfig()
        
        self._mss: Optional[Any] = None
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._monitor_info: List[Dict[str, Any]] = []
        self._screen_width = 1920
        self._screen_height = 1080
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize screen capture."""
        if _HAS_MSS:
            try:
                self._mss = mss.mss()
                monitors = self._mss.monitors
                if len(monitors) > 1:
                    primary = monitors[1]
                    self._screen_width = primary["width"]
                    self._screen_height = primary["height"]
                    self._monitor_info = [
                        {
                            "index": i,
                            "left": m["left"],
                            "top": m["top"],
                            "width": m["width"],
                            "height": m["height"],
                        }
                        for i, m in enumerate(monitors)
                    ]
                
                _LOG.info(
                    "screen_capture_initialized",
                    screen_size=(self._screen_width, self._screen_height),
                    monitor_count=len(self._monitor_info),
                )
            except Exception as e:
                _LOG.error("screen_capture_init_failed", error=str(e))
        
        if self.config.save_screenshots:
            Path(self.config.screenshot_dir).mkdir(parents=True, exist_ok=True)
    
    async def capture(
        self,
        region: Optional[POPSSCMURectangle] = None,
        monitor: int = 1,
    ) -> POPSSCMUScreenState:
        """
        Capture screen or region.
        
        Args:
            region: Optional region to capture
            monitor: Monitor index (1 for primary)
        
        Returns:
            POPSSCMUScreenState: Captured screen state
        """
        image_data = None
        screenshot_path = None
        
        if _HAS_MSS and self._mss:
            try:
                if region:
                    monitor_dict = {
                        "left": int(region.x),
                        "top": int(region.y),
                        "width": int(region.width),
                        "height": int(region.height),
                    }
                else:
                    monitor_dict = monitor if isinstance(monitor, dict) else monitor
                
                screenshot = self._mss.grab(monitor_dict)
                
                if _HAS_PIL:
                    image_data = Image.frombytes(
                        "RGB",
                        screenshot.size,
                        screenshot.bgra,
                        "raw",
                        "BGRX",
                    )
                
                if self.config.save_screenshots and image_data:
                    screenshot_path = await self._save_screenshot(image_data)
                
            except Exception as e:
                _LOG.error("capture_failed", error=str(e))
        
        width = int(region.width) if region else self._screen_width
        height = int(region.height) if region else self._screen_height
        
        return POPSSCMUScreenState(
            screenshot_path=screenshot_path,
            image_data=image_data,
            width=width,
            height=height,
            timestamp=datetime.now(),
        )
    
    async def capture_base64(
        self,
        region: Optional[POPSSCMURectangle] = None,
    ) -> str:
        """Capture screen and return as base64 string."""
        state = await self.capture(region)
        
        if state.image_data and _HAS_PIL:
            buffer = io.BytesIO()
            state.image_data.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return ""
    
    async def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> POPSSCMUScreenState:
        """Capture a specific screen region."""
        region = POPSSCMURectangle(x=x, y=y, width=width, height=height)
        return await self.capture(region)
    
    async def capture_window(
        self,
        window_title: str,
    ) -> Optional[POPSSCMUScreenState]:
        """Capture a specific window by title."""
        _LOG.warning("window_capture_not_implemented", window_title=window_title)
        return None
    
    async def _save_screenshot(self, image: Any) -> str:
        """Save screenshot to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"cmu_screenshot_{timestamp}.png"
        filepath = Path(self.config.screenshot_dir) / filename
        
        try:
            image.save(filepath, "PNG", quality=self.config.compress_quality)
            _LOG.debug("screenshot_saved", path=str(filepath))
            return str(filepath)
        except Exception as e:
            _LOG.error("save_screenshot_failed", error=str(e))
            return ""
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get primary screen size."""
        return (self._screen_width, self._screen_height)
    
    def get_monitor_count(self) -> int:
        """Get number of monitors."""
        return len(self._monitor_info)
    
    def get_monitor_info(self, index: int = 0) -> Optional[Dict[str, Any]]:
        """Get monitor information by index."""
        if 0 <= index < len(self._monitor_info):
            return self._monitor_info[index]
        return None
    
    def clear_cache(self) -> None:
        """Clear screenshot cache."""
        self._cache.clear()
    
    def cleanup_old_screenshots(self, max_age_hours: int = 24) -> int:
        """Remove screenshots older than specified age."""
        count = 0
        cutoff = time.time() - (max_age_hours * 3600)
        
        screenshot_dir = Path(self.config.screenshot_dir)
        if not screenshot_dir.exists():
            return 0
        
        for filepath in screenshot_dir.glob("cmu_screenshot_*.png"):
            if filepath.stat().st_mtime < cutoff:
                try:
                    filepath.unlink()
                    count += 1
                except Exception as e:
                    _LOG.error("cleanup_failed", path=str(filepath), error=str(e))
        
        if count > 0:
            _LOG.info("screenshots_cleaned", count=count)
        
        return count
    
    def shutdown(self) -> None:
        """Shutdown screen capture."""
        if self._mss:
            self._mss.close()
        self.clear_cache()
        _LOG.info("screen_capture_shutdown")
