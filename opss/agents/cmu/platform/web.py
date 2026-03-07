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
CMU Web Platform - Browser Automation Control

This module provides web browser control using Playwright for
comprehensive DOM interaction and browser automation.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from ..types import (
    POPSSCMUPlatform,
    POPSSCMURectangle,
)
from .base import POPSSCMUPlatformAdapter, POPSSCMUPlatformInfo

_LOG = PiscesLxLogger("PiscesLx.Opss.Agents.CMU.Platform.Web", file_path=get_log_file("PiscesLx.Opss.Agents.CMU.Platform.Web"), enable_file=True)

_HAS_PLAYWRIGHT = False

try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    _HAS_PLAYWRIGHT = True
except ImportError:
    _LOG.warning("playwright_not_available")


class POPSSCMUWeb(POPSSCMUPlatformAdapter):
    """
    Web browser platform adapter using Playwright.
    
    Provides comprehensive browser automation including:
        - Page navigation
        - DOM element interaction
        - JavaScript execution
        - Network interception
        - Multi-tab management
    
    Attributes:
        browser_type: Browser type ("chromium", "firefox", "webkit")
        headless: Whether to run in headless mode
        browser: Browser instance
        context: Browser context
        page: Current page
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        
        self.browser_type = getattr(config, 'browser_type', 'chromium') if config else 'chromium'
        self.headless = getattr(config, 'headless', False) if config else False
        
        self._playwright = None
        self._browser: Optional[Any] = None
        self._context: Optional[Any] = None
        self._page: Optional[Any] = None
        self._viewport_width = 1920
        self._viewport_height = 1080
        
    def _detect_platform_info(self) -> POPSSCMUPlatformInfo:
        """Detect web platform information."""
        browser_map = {
            "chromium": POPSSCMUPlatform.WEB_CHROME,
            "firefox": POPSSCMUPlatform.WEB_FIREFOX,
            "webkit": POPSSCMUPlatform.WEB_SAFARI,
        }
        
        platform_type = browser_map.get(self.browser_type, POPSSCMUPlatform.WEB_CHROME)
        
        capabilities = [
            "mouse_control",
            "keyboard_control",
            "dom_interaction",
            "javascript",
            "network_interception",
            "multi_tab",
            "cookies",
            "screenshots",
        ]
        
        return POPSSCMUPlatformInfo(
            platform=platform_type,
            name=f"{self.browser_type.capitalize()} Browser",
            version="latest",
            screen_width=self._viewport_width,
            screen_height=self._viewport_height,
            screen_dpi=96,
            is_touch_enabled=False,
            is_virtual=True,
            capabilities=capabilities,
        )
    
    async def initialize(self) -> bool:
        """Initialize the web browser."""
        if not _HAS_PLAYWRIGHT:
            _LOG.error("playwright_required_for_web")
            return False
        
        try:
            self._playwright = await async_playwright().start()
            
            if self.browser_type == "chromium":
                browser_launcher = self._playwright.chromium
            elif self.browser_type == "firefox":
                browser_launcher = self._playwright.firefox
            else:
                browser_launcher = self._playwright.webkit
            
            self._browser = await browser_launcher.launch(headless=self.headless)
            self._context = await self._browser.new_context(
                viewport={"width": self._viewport_width, "height": self._viewport_height}
            )
            self._page = await self._context.new_page()
            
            self._is_initialized = True
            _LOG.info("web_adapter_initialized", browser_type=self.browser_type)
            return True
            
        except Exception as e:
            _LOG.error("web_adapter_init_failed", error=str(e))
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the web browser."""
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            _LOG.error("web_adapter_shutdown_failed", error=str(e))
        
        self._is_initialized = False
        _LOG.info("web_adapter_shutdown")
    
    async def navigate(self, url: str) -> bool:
        """Navigate to a URL."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.goto(url)
            _LOG.info("navigated_to_url", url=url)
            return True
        except Exception as e:
            _LOG.error("navigation_failed", url=url, error=str(e))
            return False
    
    async def click(self, x: float, y: float, button: str = "left", clicks: int = 1) -> bool:
        """Click at coordinates."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            if clicks == 2:
                await self._page.mouse.dblclick(x, y)
            else:
                await self._page.mouse.click(x, y, button=button)
            _LOG.debug("web_click_executed", x=x, y=y)
            return True
        except Exception as e:
            _LOG.error("web_click_failed", error=str(e))
        return False
    
    async def double_click(self, x: float, y: float) -> bool:
        """Double click at coordinates."""
        return await self.click(x, y, clicks=2)
    
    async def right_click(self, x: float, y: float) -> bool:
        """Right click at coordinates."""
        return await self.click(x, y, button="right")
    
    async def move_to(self, x: float, y: float, duration: float = 0.0) -> bool:
        """Move mouse to coordinates."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.mouse.move(x, y)
            self._mouse_position = (int(x), int(y))
            return True
        except Exception as e:
            _LOG.error("web_move_failed", error=str(e))
        return False
    
    async def drag(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        duration: float = 0.5,
    ) -> bool:
        """Perform drag operation."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.mouse.move(start_x, start_y)
            await self._page.mouse.down()
            await asyncio.sleep(duration / 2)
            await self._page.mouse.move(end_x, end_y)
            await asyncio.sleep(duration / 2)
            await self._page.mouse.up()
            return True
        except Exception as e:
            _LOG.error("web_drag_failed", error=str(e))
        return False
    
    async def scroll(self, direction: str, amount: int = 3) -> bool:
        """Scroll the page."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            delta_y = amount * 100 if direction == "down" else -amount * 100
            delta_x = amount * 100 if direction == "right" else -amount * 100
            
            if direction in ("left", "right"):
                await self._page.mouse.wheel(delta_x, 0)
            else:
                await self._page.mouse.wheel(0, delta_y)
            return True
        except Exception as e:
            _LOG.error("web_scroll_failed", error=str(e))
        return False
    
    async def type_text(self, text: str, interval: float = 0.05) -> bool:
        """Type text."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.keyboard.type(text, delay=int(interval * 1000))
            return True
        except Exception as e:
            _LOG.error("web_type_failed", error=str(e))
        return False
    
    async def key_press(self, key: str) -> bool:
        """Press a key."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.keyboard.press(key)
            return True
        except Exception as e:
            _LOG.error("web_key_press_failed", error=str(e))
        return False
    
    async def key_down(self, key: str) -> bool:
        """Press and hold a key."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.keyboard.down(key)
            self._keyboard_state[key] = True
            return True
        except Exception as e:
            _LOG.error("web_key_down_failed", error=str(e))
        return False
    
    async def key_up(self, key: str) -> bool:
        """Release a key."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.keyboard.up(key)
            self._keyboard_state[key] = False
            return True
        except Exception as e:
            _LOG.error("web_key_up_failed", error=str(e))
        return False
    
    async def hotkey(self, *keys: str) -> bool:
        """Press hotkey combination."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            for key in keys:
                await self._page.keyboard.down(key)
            await asyncio.sleep(0.05)
            for key in reversed(keys):
                await self._page.keyboard.up(key)
            return True
        except Exception as e:
            _LOG.error("web_hotkey_failed", error=str(e))
        return False
    
    async def capture_screen(self, region: Optional[POPSSCMURectangle] = None) -> Optional[Any]:
        """Capture screenshot."""
        if not self._is_initialized or not self._page:
            return None
        
        try:
            if region:
                screenshot = await self._page.screenshot(
                    clip={
                        "x": region.x,
                        "y": region.y,
                        "width": region.width,
                        "height": region.height,
                    }
                )
            else:
                screenshot = await self._page.screenshot(full_page=True)
            return screenshot
        except Exception as e:
            _LOG.error("web_screenshot_failed", error=str(e))
        return None
    
    async def get_cursor_position(self) -> Tuple[int, int]:
        """Get cursor position."""
        return self._mouse_position
    
    async def get_active_window(self) -> Dict[str, Any]:
        """Get active page info."""
        if not self._is_initialized or not self._page:
            return {}
        
        try:
            return {
                "title": await self._page.title(),
                "url": self._page.url,
            }
        except Exception:
            return {}
    
    async def set_clipboard(self, text: str) -> bool:
        """Set clipboard (via JavaScript)."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.evaluate(f"navigator.clipboard.writeText('{text}')")
            return True
        except Exception:
            return False
    
    async def get_clipboard(self) -> str:
        """Get clipboard (via JavaScript)."""
        if not self._is_initialized or not self._page:
            return ""
        
        try:
            return await self._page.evaluate("navigator.clipboard.readText()")
        except Exception:
            return ""
    
    async def click_selector(self, selector: str) -> bool:
        """Click element by CSS selector."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.click(selector)
            return True
        except Exception as e:
            _LOG.error("click_selector_failed", selector=selector, error=str(e))
            return False
    
    async def fill_selector(self, selector: str, text: str) -> bool:
        """Fill input by CSS selector."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.fill(selector, text)
            return True
        except Exception as e:
            _LOG.error("fill_selector_failed", selector=selector, error=str(e))
            return False
    
    async def get_text(self, selector: str) -> str:
        """Get text content by selector."""
        if not self._is_initialized or not self._page:
            return ""
        
        try:
            return await self._page.text_content(selector) or ""
        except Exception:
            return ""
    
    async def execute_script(self, script: str) -> Any:
        """Execute JavaScript."""
        if not self._is_initialized or not self._page:
            return None
        
        try:
            return await self._page.evaluate(script)
        except Exception as e:
            _LOG.error("execute_script_failed", error=str(e))
            return None
    
    async def wait_for_selector(self, selector: str, timeout: float = 30.0) -> bool:
        """Wait for element to appear."""
        if not self._is_initialized or not self._page:
            return False
        
        try:
            await self._page.wait_for_selector(selector, timeout=timeout * 1000)
            return True
        except Exception:
            return False
    
    async def get_page_content(self) -> str:
        """Get page HTML content."""
        if not self._is_initialized or not self._page:
            return ""
        
        try:
            return await self._page.content()
        except Exception:
            return ""
