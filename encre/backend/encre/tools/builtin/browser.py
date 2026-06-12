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

import json
from typing import TYPE_CHECKING, Any

from encre.tools.base import build_tool

if TYPE_CHECKING:
    from encre.computer.browser import EncreBrowserSession

_session: "EncreBrowserSession | None" = None


def _get_session():
    global _session
    if _session is None:
        from encre.computer.browser import EncreBrowserSession
        _session = EncreBrowserSession()
    return _session


async def _browser_execute(**kwargs: Any) -> str:
    action = kwargs.get("action", "")
    session = _get_session()

    try:
        if action == "navigate":
            url = kwargs.get("url", "")
            if not url:
                return "Error: url parameter required for navigate action"
            state = await session.navigate(url)
            return f"Navigated to {state.url}\nTitle: {state.title}"

        elif action == "click":
            selector = kwargs.get("selector", "")
            if not selector:
                return "Error: selector parameter required for click action"
            ok = await session.click(selector)
            return f"Clicked {selector}" if ok else f"Error: failed to click {selector}"

        elif action == "click_at":
            x = kwargs.get("x")
            y = kwargs.get("y")
            if x is None or y is None:
                return "Error: x and y coordinates required for click_at"
            ok = await session.click_at(int(x), int(y))
            return f"Clicked at ({x}, {y})" if ok else f"Error: failed to click at ({x}, {y})"

        elif action == "double_click_at":
            x = kwargs.get("x")
            y = kwargs.get("y")
            if x is None or y is None:
                return "Error: x and y coordinates required for double_click_at"
            ok = await session.double_click_at(int(x), int(y))
            return f"Double-clicked at ({x}, {y})" if ok else "Error: failed"

        elif action == "right_click_at":
            x = kwargs.get("x")
            y = kwargs.get("y")
            if x is None or y is None:
                return "Error: x and y coordinates required for right_click_at"
            ok = await session.right_click_at(int(x), int(y))
            return f"Right-clicked at ({x}, {y})" if ok else "Error: failed"

        elif action == "type":
            selector = kwargs.get("selector", "")
            text = kwargs.get("text", "")
            if not selector:
                return "Error: selector parameter required for type action"
            ok = await session.type_text(selector, text)
            return f"Typed into {selector}" if ok else f"Error: failed to type into {selector}"

        elif action == "type_at":
            x = kwargs.get("x")
            y = kwargs.get("y")
            text = kwargs.get("text", "")
            if x is None or y is None:
                return "Error: x and y coordinates required for type_at"
            if not text:
                return "Error: text parameter required for type_at"
            ok = await session.type_at(int(x), int(y), text)
            return f"Typed at ({x}, {y})" if ok else f"Error: failed to type at ({x}, {y})"

        elif action == "screenshot":
            full_page = kwargs.get("full_page", False)
            selector = kwargs.get("selector")
            return await session.screenshot(full_page=full_page, selector=selector)

        elif action == "screenshot_viewport":
            info = await session.screenshot_viewport()
            return json.dumps(info, ensure_ascii=False)

        elif action == "get_html":
            return await session.get_html()

        elif action == "get_text":
            selector = kwargs.get("selector")
            return await session.get_text(selector=selector)

        elif action == "execute_js":
            code = kwargs.get("code", "")
            if not code:
                return "Error: code parameter required for execute_js action"
            result = await session.execute_js(code)
            return str(result)

        elif action == "wait_for_selector":
            selector = kwargs.get("selector", "")
            if not selector:
                return "Error: selector parameter required for wait_for_selector action"
            timeout = kwargs.get("timeout")
            ok = await session.wait_for_selector(selector, timeout=timeout)
            return (
                f"Element found: {selector}"
                if ok
                else f"Timeout: element not found: {selector}"
            )

        elif action == "scroll_to":
            x = kwargs.get("x", 0)
            y = kwargs.get("y", 0)
            await session.scroll_to(x=x, y=y)
            return f"Scrolled to ({x}, {y})"

        elif action == "fill_form":
            fields = kwargs.get("fields", {})
            if not fields:
                return "Error: fields parameter required for fill_form action"
            ok = await session.fill_form(fields)
            return "Form filled successfully" if ok else "Error: failed to fill form"

        elif action == "press_key":
            key = kwargs.get("key", "")
            if not key:
                return "Error: key parameter required for press_key action"
            await session.press_key(key)
            return f"Pressed key: {key}"

        elif action == "hotkey":
            keys = kwargs.get("keys", [])
            if not keys:
                return "Error: keys array required for hotkey"
            ok = await session.hotkey([str(k) for k in keys])
            return f"Pressed hotkey: {'+'.join(keys)}" if ok else "Error: hotkey failed"

        elif action == "hover":
            selector = kwargs.get("selector", "")
            if not selector:
                return "Error: selector parameter required for hover action"
            ok = await session.hover(selector)
            return f"Hovered {selector}" if ok else f"Error: failed to hover {selector}"

        elif action == "hover_at":
            x = kwargs.get("x")
            y = kwargs.get("y")
            if x is None or y is None:
                return "Error: x and y coordinates required for hover_at"
            ok = await session.hover_at(int(x), int(y))
            return f"Hovered at ({x}, {y})" if ok else "Error: failed"

        elif action == "drag":
            x1 = kwargs.get("x")
            y1 = kwargs.get("y")
            x2 = kwargs.get("x2")
            y2 = kwargs.get("y2")
            if x1 is None or y1 is None or x2 is None or y2 is None:
                return "Error: x, y (start) and x2, y2 (end) required for drag"
            ok = await session.drag(int(x1), int(y1), int(x2), int(y2))
            return f"Dragged from ({x1},{y1}) to ({x2},{y2})" if ok else "Error: drag failed"

        elif action == "get_viewport":
            vp = await session.get_viewport()
            return json.dumps({
                "width": vp.width,
                "height": vp.height,
                "scroll_x": vp.scroll_x,
                "scroll_y": vp.scroll_y,
                "device_pixel_ratio": vp.device_pixel_ratio,
            })

        elif action == "get_state":
            state = await session.get_state()
            return f"URL: {state.url}\nTitle: {state.title}"

        elif action == "save_cookies":
            cookies = await session.save_cookies()
            return json.dumps(cookies)

        elif action == "load_cookies":
            cookies = kwargs.get("cookies", [])
            if not cookies:
                return "Error: cookies parameter required for load_cookies action"
            await session.load_cookies(cookies)
            return f"Loaded {len(cookies)} cookies"

        elif action == "close_session":
            await session.close()
            global _session
            _session = None
            return "Browser session closed"

        elif action == "a11y_snapshot":
            interesting_only = bool(kwargs.get("interesting_only", True))
            root_selector = kwargs.get("root_selector") or None
            snap = await session.a11y_snapshot(
                interesting_only=interesting_only,
                root_selector=root_selector,
            )
            return json.dumps(snap, ensure_ascii=False)

        elif action == "click_by_role":
            role = kwargs.get("role", "")
            name = kwargs.get("name", "")
            if not role or not name:
                return "Error: 'role' and 'name' are required for click_by_role"
            exact = bool(kwargs.get("exact", False))
            ok = await session.click_by_role(role, name, exact=exact)
            return (
                f"Clicked role={role} name={name!r}"
                if ok else f"Error: failed to click role={role} name={name!r}"
            )

        elif action == "get_by_text_count":
            text = kwargs.get("name") or kwargs.get("text") or ""
            if not text:
                return "Error: 'name' (text to match) required for get_by_text_count"
            exact = bool(kwargs.get("exact", False))
            count = await session.get_by_text_count(text, exact=exact)
            return json.dumps({"text": text, "count": count, "exact": exact})

        elif action == "get_page_structure":
            elements = await session.get_page_structure()
            return json.dumps({
                "url": session._state.url if session._page else "",
                "viewport_width": session.viewport_width,
                "viewport_height": session.viewport_height,
                "elements_count": len(elements),
                "elements": elements,
            }, ensure_ascii=False)

        else:
            return f"Error: unknown action '{action}'"

    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Browser action '{action}' failed: {e}"


EncreBrowserTool = build_tool(
    name="browser",
    description=(
        "Browser automation: navigate, click (by selector or coordinates), type, "
        "screenshot (with viewport info for visual models), get_html, get_text, "
        "execute_js, wait_for_selector, scroll, fill_form, press_key, hotkey, "
        "hover, drag, save/load cookies, and close_session on a Chromium browser. "
        "Supports both DOM selector-driven and coordinate-driven (visual model) actions. "
        "Use 'get_page_structure' to extract all interactive elements (buttons, links, "
        "inputs, headings, etc.) with bounding boxes so the model can see and interact "
        "with the page without requiring vision."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate", "click", "click_at", "double_click_at",
                    "right_click_at", "type", "type_at", "screenshot",
                    "screenshot_viewport", "get_html", "get_text",
                    "execute_js", "wait_for_selector", "scroll_to",
                    "fill_form", "press_key", "hotkey", "hover",
                    "hover_at", "drag", "get_viewport", "get_state",
                    "save_cookies", "load_cookies", "close_session",
                    "a11y_snapshot", "click_by_role", "get_by_text_count",
                    "get_page_structure",
                ],
                "description": "Browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL to navigate to (for navigate action)",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for DOM-based actions",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate in viewport pixels "
                "(for click_at, type_at, hover_at, move_mouse, drag start)",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate in viewport pixels "
                "(for click_at, type_at, hover_at, move_mouse, drag start)",
            },
            "x2": {
                "type": "integer",
                "description": "Target X coordinate (for drag end)",
            },
            "y2": {
                "type": "integer",
                "description": "Target Y coordinate (for drag end)",
            },
            "text": {
                "type": "string",
                "description": "Text to type (for type, type_at actions)",
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture full page screenshot",
            },
            "code": {
                "type": "string",
                "description": "JavaScript code to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds (for wait_for_selector)",
            },
            "fields": {
                "type": "object",
                "description": "Dict of selector:value pairs (for fill_form)",
            },
            "key": {
                "type": "string",
                "description": "Key name to press (for press_key)",
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key combination (e.g. [\"Control\", \"c\"] for copy)",
            },
            "cookies": {
                "type": "array",
                "description": "List of cookie dicts (for load_cookies)",
            },
            "interesting_only": {
                "type": "boolean",
                "description": "a11y_snapshot: prune uninteresting nodes (default true)",
            },
            "role": {
                "type": "string",
                "description": "ARIA role for click_by_role (button, link, textbox, ...)",
            },
            "name": {
                "type": "string",
                "description": "Accessible name for click_by_role / get_by_text_count",
            },
            "exact": {
                "type": "boolean",
                "description": "Use exact name matching (click_by_role / get_by_text_count)",
            },
            "root_selector": {
                "type": "string",
                "description": "Optional CSS root for a11y_snapshot to limit the tree",
            },
        },
        "required": ["action"],
    },
    execute=_browser_execute,
    intents=["coding", "system"],
)
