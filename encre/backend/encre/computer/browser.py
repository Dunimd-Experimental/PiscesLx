#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

from __future__ import annotations
import base64
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class BrowserState:
    url: str = ""
    title: str = ""
    html: str = ""
    text: str = ""


@dataclass
class BrowserViewport:
    width: int = 0
    height: int = 0
    scroll_x: int = 0
    scroll_y: int = 0
    device_pixel_ratio: float = 1.0


class EncreBrowserSession:
    def __init__(
        self,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        timeout: int = 30000,
    ):
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout = timeout
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None
        self._state = BrowserState()
        self._last_used = time.time()

    def _check_playwright(self) -> bool:
        try:
            import playwright  # noqa: F401
            return True
        except ImportError:
            return False

    async def _ensure_browser(self):
        if not self._check_playwright():
            raise RuntimeError(
                "Playwright not installed. "
                "Run: pip install playwright && playwright install chromium"
            )
        if self._browser is None:
            from playwright.async_api import async_playwright

            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height}
            )
            self._context.set_default_timeout(self.timeout)
            self._page = await self._context.new_page()
        self._last_used = time.time()

    async def navigate(self, url: str) -> BrowserState:
        await self._ensure_browser()
        await self._page.goto(url, wait_until="domcontentloaded")
        self._state.url = self._page.url
        self._state.title = await self._page.title()
        self._state.html = await self._page.content()
        try:
            self._state.text = await self._page.inner_text("body")
        except Exception:
            self._state.text = ""
        return self._state

    async def click(self, selector: str) -> bool:
        await self._ensure_browser()
        try:
            await self._page.click(selector)
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def type_text(self, selector: str, text: str) -> bool:
        await self._ensure_browser()
        try:
            await self._page.fill(selector, text)
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def screenshot(
        self, full_page: bool = False, selector: str | None = None
    ) -> str:
        await self._ensure_browser()
        if selector:
            element = await self._page.query_selector(selector)
            if element is None:
                raise ValueError(f"Element not found: {selector}")
            data = await element.screenshot(type="png")
        else:
            data = await self._page.screenshot(type="png", full_page=full_page)
        self._last_used = time.time()
        return base64.b64encode(data).decode("utf-8")

    async def get_html(self) -> str:
        await self._ensure_browser()
        self._state.html = await self._page.content()
        self._last_used = time.time()
        return self._state.html

    async def get_text(self, selector: str | None = None) -> str:
        await self._ensure_browser()
        if selector:
            element = await self._page.query_selector(selector)
            if element is None:
                raise ValueError(f"Element not found: {selector}")
            text = await element.inner_text()
        else:
            text = await self._page.inner_text("body")
        self._state.text = text
        self._last_used = time.time()
        return text

    async def execute_js(self, code: str) -> Any:
        await self._ensure_browser()
        result = await self._page.evaluate(code)
        self._last_used = time.time()
        return result

    async def get_state(self) -> BrowserState:
        if self._page is not None:
            try:
                self._state.url = self._page.url
                self._state.title = await self._page.title()
            except Exception:
                pass
        return self._state

    async def wait_for_selector(
        self, selector: str, timeout: int | None = None
    ) -> bool:
        await self._ensure_browser()
        try:
            await self._page.wait_for_selector(
                selector, timeout=timeout or self.timeout
            )
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def scroll_to(self, x: int = 0, y: int = 0) -> None:
        await self._ensure_browser()
        await self._page.evaluate(f"window.scrollTo({x}, {y})")
        self._last_used = time.time()

    async def fill_form(self, fields: dict[str, str]) -> bool:
        await self._ensure_browser()
        try:
            for selector, value in fields.items():
                await self._page.fill(selector, value)
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def press_key(self, key: str) -> None:
        await self._ensure_browser()
        await self._page.keyboard.press(key)
        self._last_used = time.time()

    async def get_viewport(self) -> BrowserViewport:
        await self._ensure_browser()
        vp = await self._page.evaluate("""() => ({
            width: window.innerWidth,
            height: window.innerHeight,
            scrollX: window.scrollX,
            scrollY: window.scrollY,
            devicePixelRatio: window.devicePixelRatio,
        })""")
        self._last_used = time.time()
        return BrowserViewport(
            width=int(vp["width"]),
            height=int(vp["height"]),
            scroll_x=int(vp["scrollX"]),
            scroll_y=int(vp["scrollY"]),
            device_pixel_ratio=float(vp["devicePixelRatio"]),
        )

    async def click_at(self, x: int, y: int) -> bool:
        await self._ensure_browser()
        try:
            await self._page.mouse.click(x, y)
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def double_click_at(self, x: int, y: int) -> bool:
        await self._ensure_browser()
        try:
            await self._page.mouse.dblclick(x, y)
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def right_click_at(self, x: int, y: int) -> bool:
        await self._ensure_browser()
        try:
            await self._page.mouse.click(x, y, button="right")
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def move_mouse(self, x: int, y: int) -> bool:
        await self._ensure_browser()
        try:
            await self._page.mouse.move(x, y)
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def drag(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        await self._ensure_browser()
        try:
            await self._page.mouse.move(x1, y1)
            await self._page.mouse.down()
            await self._page.mouse.move(x2, y2, steps=10)
            await self._page.mouse.up()
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def type_at(self, x: int, y: int, text: str) -> bool:
        await self._ensure_browser()
        try:
            await self._page.mouse.click(x, y)
            await self._page.keyboard.type(text)
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def hotkey(self, keys: list[str]) -> bool:
        await self._ensure_browser()
        try:
            main = keys[0]
            for k in keys[1:]:
                await self._page.keyboard.press(f"{main}+{k}")
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def screenshot_viewport(self) -> dict[str, Any]:
        await self._ensure_browser()
        vp = await self.get_viewport()
        data = await self._page.screenshot(type="png")
        b64 = base64.b64encode(data).decode("utf-8")
        self._last_used = time.time()
        return {
            "width": vp.width,
            "height": vp.height,
            "scroll_x": vp.scroll_x,
            "scroll_y": vp.scroll_y,
            "device_pixel_ratio": vp.device_pixel_ratio,
            "url": self._page.url,
            "title": await self._page.title(),
            "screenshot_base64": b64,
        }

    async def hover(self, selector: str) -> bool:
        await self._ensure_browser()
        try:
            await self._page.hover(selector)
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def hover_at(self, x: int, y: int) -> bool:
        await self._ensure_browser()
        try:
            await self._page.mouse.move(x, y)
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def save_cookies(self) -> list[dict]:
        if self._context is None:
            return []
        cookies = await self._context.cookies()
        return cookies

    async def load_cookies(self, cookies: list[dict]) -> None:
        await self._ensure_browser()
        await self._context.add_cookies(cookies)

    async def close(self) -> None:
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._pw:
            await self._pw.stop()
            self._pw = None
        self._page = None

    # ------------------------------------------------------------------
    # Accessibility
    # ------------------------------------------------------------------

    async def a11y_snapshot(self, interesting_only: bool = True,
                            root_selector: str | None = None) -> dict[str, Any]:
        """Return Playwright's accessibility snapshot of the current page.

        Args:
            interesting_only: When True (default), prune nodes that have no
                role/name (matches Playwright's default). Set False to get
                the full ARIA tree.
            root_selector: Limit the snapshot to a sub-tree.

        Returns:
            A tree of nodes with ``role``, ``name``, ``value``, ``children``,
            and any other ARIA attributes Playwright exposes. Stable enough
            for the model to pick targets by name/role without depending on
            visual coordinates.
        """
        await self._ensure_browser()
        root = None
        if root_selector:
            handle = await self._page.query_selector(root_selector)
            if handle is None:
                raise ValueError(f"a11y_snapshot root not found: {root_selector}")
            root = handle
        snapshot = await self._page.accessibility.snapshot(
            interesting_only=interesting_only, root=root,
        )
        self._last_used = time.time()
        return snapshot if isinstance(snapshot, dict) else {"role": "none", "children": []}

    async def click_by_role(self, role: str, name: str,
                            exact: bool = False) -> bool:
        """Click a DOM element resolved through Playwright's ARIA locator."""
        await self._ensure_browser()
        try:
            loc = self._page.get_by_role(role, name=name, exact=exact)
            await loc.click()
            self._last_used = time.time()
            return True
        except Exception:
            return False

    async def get_by_text_count(self, text: str, exact: bool = False) -> int:
        """Count matches for a given text. Useful for verifying a snapshot."""
        await self._ensure_browser()
        try:
            return await self._page.get_by_text(text, exact=exact).count()
        except Exception:
            return 0

    def is_idle(self, max_idle_seconds: int = 600) -> bool:
        return (time.time() - self._last_used) > max_idle_seconds

    # ------------------------------------------------------------------
    # Page structure
    # ------------------------------------------------------------------

    async def get_page_structure(self) -> list[dict[str, Any]]:
        """Extract all interactive elements with bounding boxes via the DOM.

        Returns a list of elements, each with tag, type, role, text,
        position (x, y, width, height, center_x, center_y), and href.

        Much faster and more accurate than OCR for browser content.
        """
        await self._ensure_browser()
        js = """() => {
            const ELEMENTS = [];
            const SEEN = new Set();

            // Collect all potentially interactive elements
            const RAW_SELECTORS = [
                'a[href]', 'button', 'input', 'select', 'textarea',
                '[contenteditable="true"]',
                '[role="button"]', '[role="link"]', '[role="textbox"]',
                '[role="combobox"]', '[role="checkbox"]', '[role="radio"]',
                '[role="switch"]', '[role="tab"]', '[role="menuitem"]',
                '[role="option"]', '[role="searchbox"]', '[role="slider"]',
                '[onclick]', '[tabindex]:not([tabindex="-1"])',
            ];

            // Visible text blocks for reading context
            const TEXT_SELECTORS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'p', 'label', 'span', 'strong', 'em', 'li'];

            function add(el, type) {
                const rect = el.getBoundingClientRect();
                if (rect.width < 4 || rect.height < 4) return;
                if (rect.bottom < -10 || rect.right < -10) return;
                if (rect.top > window.innerHeight + 10) return;

                const tag = el.tagName.toLowerCase();
                const role = el.getAttribute('role') || '';
                const typeAttr = el.getAttribute('type') || '';
                const text = (el.textContent || '').trim().slice(0, 200);
                const placeholder = el.getAttribute('placeholder') || '';
                const ariaLabel = el.getAttribute('aria-label') || '';
                const alt = el.getAttribute('alt') || '';
                const name = ariaLabel || alt || placeholder || text;

                const key = Math.round(rect.left/5) + ',' + Math.round(rect.top/5);
                const prev = ELEMENTS.find(e =>
                    Math.abs(e.x - rect.left) < 3 &&
                    Math.abs(e.y - rect.top) < 3 &&
                    e.name === name
                );
                if (prev) return;

                ELEMENTS.push({
                    tag: tag,
                    type: typeAttr || undefined,
                    role: role || undefined,
                    name: name.slice(0, 120),
                    x: Math.round(rect.left),
                    y: Math.round(rect.top),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    center_x: Math.round(rect.left + rect.width/2),
                    center_y: Math.round(rect.top + rect.height/2),
                    href: el.getAttribute('href') || undefined,
                    checked: (el.checked === true) || undefined,
                    selected: (el.tagName === 'OPTION' && el.selected) || undefined,
                });
            }

            // Interactive elements first
            const elements = document.querySelectorAll(RAW_SELECTORS.join(','));
            elements.forEach(el => add(el, 'interactive'));

            // Visible text headings and labels for reading context
            const texts = document.querySelectorAll(TEXT_SELECTORS.join(','));
            texts.forEach(el => {
                const rect = el.getBoundingClientRect();
                if (rect.width < 4 || rect.height < 4) return;
                if (rect.bottom < -10 || rect.right < -10) return;
                if (rect.top > window.innerHeight + 10) return;
                const text = (el.textContent || '').trim();
                if (!text) return;
                const tag = el.tagName.toLowerCase();
                const key = Math.round(rect.left/5) + ',' + Math.round(rect.top/5);
                if (ELEMENTS.some(e =>
                    Math.abs(e.x - rect.left) < 3 &&
                    Math.abs(e.y - rect.top) < 3)) return;
                ELEMENTS.push({
                    tag: tag,
                    role: 'heading',
                    name: text.slice(0, 200),
                    x: Math.round(rect.left),
                    y: Math.round(rect.top),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    center_x: Math.round(rect.left + rect.width/2),
                    center_y: Math.round(rect.top + rect.height/2),
                });
            });

            return ELEMENTS;
        }"""
        return await self._page.evaluate(js)
