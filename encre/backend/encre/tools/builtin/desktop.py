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
import json
from typing import Any

from encre.tools.base import build_tool

_session: Any = None


def _get_session():
    global _session
    if _session is None:
        from encre.computer.desktop import EncreDesktopSession
        _session = EncreDesktopSession()
    return _session


async def _desktop_execute(**kwargs: Any) -> str:
    action = kwargs.get("action", "")
    session = _get_session()
    coord_space = str(kwargs.get("coord_space", "auto"))
    if coord_space not in ("auto", "physical", "logical"):
        return f"Error: invalid coord_space '{coord_space}'"

    try:
        if action == "screenshot":
            state = session.screenshot_with_cursor()
            return json.dumps({
                "width": state.width,
                "height": state.height,
                "logical_width": state.logical_width,
                "logical_height": state.logical_height,
                "dpi_scale_x": state.dpi_scale_x,
                "dpi_scale_y": state.dpi_scale_y,
                "cursor_x": state.cursor_x,
                "cursor_y": state.cursor_y,
                "platform": state.platform,
                "screenshot_base64": state.screenshot_b64,
            }, ensure_ascii=False)

        elif action == "get_screen_size":
            size = session.get_screen_size()
            return json.dumps(size)

        elif action == "get_cursor_position":
            pos = session.get_cursor_position()
            return json.dumps(pos)

        elif action == "move_mouse":
            x = kwargs.get("x")
            y = kwargs.get("y")
            if x is None or y is None:
                return "Error: x and y coordinates required for move_mouse"
            result = session.move_mouse(int(x), int(y), coord_space=coord_space)
            return json.dumps(result)

        elif action == "click":
            x = kwargs.get("x")
            y = kwargs.get("y")
            result = session.click(
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
                coord_space=coord_space,
            )
            return json.dumps(result)

        elif action == "double_click":
            x = kwargs.get("x")
            y = kwargs.get("y")
            result = session.double_click(
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
                coord_space=coord_space,
            )
            return json.dumps(result)

        elif action == "right_click":
            x = kwargs.get("x")
            y = kwargs.get("y")
            result = session.right_click(
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
                coord_space=coord_space,
            )
            return json.dumps(result)

        elif action == "drag":
            x1 = kwargs.get("x")
            y1 = kwargs.get("y")
            x2 = kwargs.get("x2")
            y2 = kwargs.get("y2")
            if x1 is None or y1 is None or x2 is None or y2 is None:
                return "Error: x, y (start) and x2, y2 (end) required for drag"
            result = session.drag(int(x1), int(y1), int(x2), int(y2),
                                  coord_space=coord_space)
            return json.dumps(result)

        elif action == "type_text":
            text = kwargs.get("text", "")
            if not text:
                return "Error: text parameter required for type_text"
            result = session.type_text(str(text))
            return json.dumps(result)

        elif action == "press_key":
            key = kwargs.get("key", "")
            if not key:
                return "Error: key parameter required for press_key"
            result = session.press_key(str(key))
            return json.dumps(result)

        elif action == "hotkey":
            keys = kwargs.get("keys", [])
            if not keys:
                return "Error: keys array required for hotkey"
            result = session.hotkey([str(k) for k in keys])
            return json.dumps(result)

        elif action == "scroll":
            clicks = kwargs.get("clicks")
            if clicks is None:
                return "Error: clicks parameter required for scroll"
            x = kwargs.get("x")
            y = kwargs.get("y")
            result = session.scroll(
                int(clicks),
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
            )
            return json.dumps(result)

        elif action == "locate_on_screen":
            template = kwargs.get("template", "")
            if not template:
                return "Error: template (base64 PNG) required for locate_on_screen"
            confidence = float(kwargs.get("confidence", 0.9))
            result = session.locate_on_screen(template, confidence=confidence)
            if result.found:
                return json.dumps({
                    "found": True,
                    "x": result.x,
                    "y": result.y,
                    "width": result.width,
                    "height": result.height,
                    "confidence": result.confidence,
                })
            return json.dumps({"found": False})

        elif action == "accessibility_tree":
            max_depth = int(kwargs.get("max_depth", 6))
            max_nodes = int(kwargs.get("max_nodes", 500))
            tree = session.accessibility_tree(max_depth=max_depth, max_nodes=max_nodes)
            return json.dumps(tree, ensure_ascii=False)

        elif action == "find_element_by_name":
            name = kwargs.get("name", "")
            if not name:
                return "Error: name parameter required for find_element_by_name"
            control_type = kwargs.get("control_type") or None
            result = session.find_element_by_name(name, control_type=control_type)
            return json.dumps(result, ensure_ascii=False)

        elif action == "get_elements":
            min_text_len = int(kwargs.get("min_text_length", 2))
            state = session.screenshot_with_cursor()
            from encre.computer.ocr import ocr_image
            img_bytes = __import__("base64").b64decode(state.screenshot_b64)
            elements = ocr_image(img_bytes)
            # Filter very short fragments
            elements = [e for e in elements if len(e["text"]) >= min_text_len]
            return json.dumps({
                "screen_width": state.width,
                "screen_height": state.height,
                "logical_width": state.logical_width,
                "logical_height": state.logical_height,
                "dpi_scale_x": state.dpi_scale_x,
                "dpi_scale_y": state.dpi_scale_y,
                "elements_count": len(elements),
                "elements": elements,
            }, ensure_ascii=False)

        else:
            return f"Error: unknown action '{action}'"

    except RuntimeError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Desktop action '{action}' failed: {e}"


EncreDesktopTool = build_tool(
    name="desktop",
    description=(
        "Cross-platform desktop automation: screenshot (with DPI scale info), "
        "click, type, scroll, drag, hotkey, locate image on screen, and on "
        "Windows walk the UI Automation accessibility tree. Mouse events "
        "auto-translate between physical and logical coordinate systems on "
        "HiDPI displays — pass coord_space='physical' (default 'auto') to "
        "click directly with coordinates read off a screenshot. "
        "Use 'get_elements' to extract visible text + bounding boxes via "
        "OCR so the model can see what is on screen without requiring vision."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "screenshot",
                    "click",
                    "double_click",
                    "right_click",
                    "move_mouse",
                    "drag",
                    "type_text",
                    "press_key",
                    "hotkey",
                    "scroll",
                    "locate_on_screen",
                    "get_screen_size",
                    "get_cursor_position",
                    "accessibility_tree",
                    "find_element_by_name",
                    "get_elements",
                ],
                "description": "Desktop action to perform",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate (for click, move, drag start, scroll)",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate (for click, move, drag start, scroll)",
            },
            "x2": {
                "type": "integer",
                "description": "Target X coordinate (for drag end)",
            },
            "y2": {
                "type": "integer",
                "description": "Target Y coordinate (for drag end)",
            },
            "coord_space": {
                "type": "string",
                "enum": ["auto", "physical", "logical"],
                "description": (
                    "Coordinate system of (x, y). 'physical' = pixels of the "
                    "screenshot, 'logical' = pyautogui's scaled coords, "
                    "'auto' (default) detects from value magnitude."
                ),
            },
            "text": {
                "type": "string",
                "description": "Text to type (for type_text action)",
            },
            "key": {
                "type": "string",
                "description": "Key name to press (enter, tab, escape, f1, etc.)",
            },
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key combination (e.g. [\"ctrl\", \"c\"] for copy)",
            },
            "clicks": {
                "type": "integer",
                "description": "Scroll amount — positive=up, negative=down",
            },
            "template": {
                "type": "string",
                "description": "Base64-encoded PNG image to locate on screen",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence threshold for locate_on_screen (0.0-1.0, default 0.9)",
            },
            "max_depth": {
                "type": "integer",
                "description": "accessibility_tree: max recursion depth (default 6)",
            },
            "max_nodes": {
                "type": "integer",
                "description": "accessibility_tree: max nodes returned (default 500)",
            },
            "name": {
                "type": "string",
                "description": "find_element_by_name: substring match against accessible name",
            },
            "control_type": {
                "type": "string",
                "description": "find_element_by_name: filter by UIA control type (e.g. ButtonControl)",
            },
            "min_text_length": {
                "type": "integer",
                "description": "get_elements: ignore text shorter than this (default 2, filter noise)",
            },
        },
        "required": ["action"],
    },
    execute=_desktop_execute,
    intents=["coding", "system"],
)
