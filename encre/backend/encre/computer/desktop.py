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

import base64
import io
import os
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

_PLATFORM = sys.platform


def _enable_windows_dpi_awareness() -> None:
    """Make the current process per-monitor DPI aware on Windows so
    screenshot pixels and mouse coordinates use the same coordinate system.

    Without this, on a HiDPI display mss returns physical pixels while
    pyautogui returns logical (scaled) pixels, and clicks miss their
    targets by the scale factor.
    """
    if _PLATFORM != "win32":
        return
    try:
        import ctypes
        # PROCESS_PER_MONITOR_DPI_AWARE = 2 — Win 8.1+. Fall back to
        # SetProcessDPIAware for older systems.
        shcore = ctypes.windll.shcore
        try:
            shcore.SetProcessDpiAwareness(2)
            return
        except Exception:
            pass
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


_enable_windows_dpi_awareness()


@dataclass
class DesktopScreenState:
    width: int = 0
    height: int = 0
    screenshot_b64: str = ""
    cursor_x: int = 0
    cursor_y: int = 0
    platform: str = _PLATFORM
    dpi_scale_x: float = 1.0
    dpi_scale_y: float = 1.0
    logical_width: int = 0
    logical_height: int = 0


@dataclass
class DesktopLocateResult:
    found: bool = False
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 0.0


class EncreDesktopSession:
    def __init__(self):
        self._last_used = time.time()
        self._state = DesktopScreenState()
        self._dpi_x: float | None = None
        self._dpi_y: float | None = None

    def _check_mss(self) -> bool:
        try:
            import mss  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_pyautogui(self) -> bool:
        try:
            import pyautogui  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_pillow(self) -> bool:
        try:
            import PIL  # noqa: F401
            return True
        except ImportError:
            return False

    def screenshot(self) -> DesktopScreenState:
        if not self._check_mss():
            raise RuntimeError(
                "mss not installed. Run: pip install mss pillow"
            )
        if not self._check_pillow():
                raise RuntimeError(
                    "Pillow not installed. Run: pip install pillow"
                ) from None
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)

        buf = io.BytesIO()
        self._ensure_pillow()
        from PIL import Image
        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        physical_w = monitor["width"]
        physical_h = monitor["height"]
        logical_w, logical_h = self._logical_size_fallback(physical_w, physical_h)
        scale_x, scale_y = self._compute_scale(physical_w, physical_h, logical_w, logical_h)

        self._state.width = physical_w
        self._state.height = physical_h
        self._state.logical_width = logical_w
        self._state.logical_height = logical_h
        self._state.dpi_scale_x = scale_x
        self._state.dpi_scale_y = scale_y
        self._state.screenshot_b64 = b64
        self._last_used = time.time()
        return self._state

    def _logical_size_fallback(self, physical_w: int, physical_h: int) -> tuple[int, int]:
        """Return logical (pyautogui-visible) screen size."""
        if self._check_pyautogui():
            try:
                import pyautogui
                w, h = pyautogui.size()
                return int(w), int(h)
            except Exception:
                pass
        return physical_w, physical_h

    def _compute_scale(self, phys_w: int, phys_h: int,
                       log_w: int, log_h: int) -> tuple[float, float]:
        sx = phys_w / log_w if log_w else 1.0
        sy = phys_h / log_h if log_h else 1.0
        self._dpi_x, self._dpi_y = sx, sy
        return sx, sy

    def _to_logical(self, x: int, y: int, coord_space: str) -> tuple[int, int]:
        """Convert a coordinate into pyautogui's logical coordinate system."""
        if coord_space == "logical":
            return x, y
        if coord_space == "physical":
            sx = self._dpi_x or 1.0
            sy = self._dpi_y or 1.0
            return int(round(x / sx)), int(round(y / sy))
        # auto: if dpi scale != 1 and the value looks "too big" for the
        # logical screen, assume it's physical.
        sx = self._dpi_x
        sy = self._dpi_y
        if sx is None or sy is None:
            return x, y
        if abs(sx - 1.0) < 1e-3 and abs(sy - 1.0) < 1e-3:
            return x, y
        try:
            import pyautogui
            lw, lh = pyautogui.size()
            if x > lw or y > lh:
                return int(round(x / sx)), int(round(y / sy))
        except Exception:
            pass
        return x, y

    def get_screen_size(self) -> dict[str, int]:
        if self._check_pyautogui():
            import pyautogui
            w, h = pyautogui.size()
            self._state.width = w
            self._state.height = h
        elif self._check_mss():
            import mss
            with mss.mss() as sct:
                m = sct.monitors[1]
                self._state.width = m["width"]
                self._state.height = m["height"]
        return {"width": self._state.width, "height": self._state.height}

    def get_cursor_position(self) -> dict[str, int]:
        if not self._check_pyautogui():
            return {"x": 0, "y": 0}
        import pyautogui
        x, y = pyautogui.position()
        self._state.cursor_x = x
        self._state.cursor_y = y
        self._last_used = time.time()
        return {"x": x, "y": y}

    def move_mouse(self, x: int, y: int, coord_space: str = "auto") -> dict[str, int]:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        import pyautogui
        lx, ly = self._to_logical(int(x), int(y), coord_space)
        pyautogui.moveTo(lx, ly)
        self._state.cursor_x = lx
        self._state.cursor_y = ly
        self._last_used = time.time()
        return {"x": lx, "y": ly}

    def click(
        self, x: int | None = None, y: int | None = None, button: str = "left",
        coord_space: str = "auto",
    ) -> dict[str, Any]:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        import pyautogui
        if x is not None and y is not None:
            lx, ly = self._to_logical(int(x), int(y), coord_space)
            pyautogui.click(lx, ly, button=button)
        else:
            pyautogui.click(button=button)
        pos = pyautogui.position()
        self._state.cursor_x = pos[0]
        self._state.cursor_y = pos[1]
        self._last_used = time.time()
        return {"action": "click", "button": button, "x": pos[0], "y": pos[1]}

    def double_click(self, x: int | None = None, y: int | None = None,
                     coord_space: str = "auto") -> dict[str, Any]:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        import pyautogui
        if x is not None and y is not None:
            lx, ly = self._to_logical(int(x), int(y), coord_space)
            pyautogui.doubleClick(lx, ly)
        else:
            pyautogui.doubleClick()
        pos = pyautogui.position()
        self._state.cursor_x = pos[0]
        self._state.cursor_y = pos[1]
        self._last_used = time.time()
        return {"action": "double_click", "x": pos[0], "y": pos[1]}

    def right_click(self, x: int | None = None, y: int | None = None,
                    coord_space: str = "auto") -> dict[str, Any]:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        import pyautogui
        if x is not None and y is not None:
            lx, ly = self._to_logical(int(x), int(y), coord_space)
            pyautogui.rightClick(lx, ly)
        else:
            pyautogui.rightClick()
        pos = pyautogui.position()
        self._state.cursor_x = pos[0]
        self._state.cursor_y = pos[1]
        self._last_used = time.time()
        return {"action": "right_click", "x": pos[0], "y": pos[1]}

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5,
             coord_space: str = "auto") -> dict[str, Any]:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        import pyautogui
        lx1, ly1 = self._to_logical(int(x1), int(y1), coord_space)
        lx2, ly2 = self._to_logical(int(x2), int(y2), coord_space)
        pyautogui.moveTo(lx1, ly1)
        pyautogui.drag(lx2 - lx1, ly2 - ly1, duration=duration)
        pos = pyautogui.position()
        self._state.cursor_x = pos[0]
        self._state.cursor_y = pos[1]
        self._last_used = time.time()
        return {"action": "drag", "from": {"x": lx1, "y": ly1}, "to": {"x": lx2, "y": ly2}}

    def type_text(self, text: str, interval: float = 0.02) -> dict[str, Any]:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        import pyautogui
        pyautogui.typewrite(text, interval=interval)
        self._last_used = time.time()
        return {"action": "type", "text": text[:200]}

    def press_key(self, key: str) -> dict[str, Any]:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        import pyautogui
        pyautogui.press(key)
        self._last_used = time.time()
        return {"action": "press_key", "key": key}

    def hotkey(self, keys: list[str]) -> dict[str, Any]:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        import pyautogui
        pyautogui.hotkey(*keys)
        self._last_used = time.time()
        return {"action": "hotkey", "keys": "+".join(keys)}

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        import pyautogui
        if x is not None and y is not None:
            pyautogui.scroll(clicks, x, y)
        else:
            pyautogui.scroll(clicks)
        self._last_used = time.time()
        return {"action": "scroll", "clicks": clicks}

    def locate_on_screen(self, image_b64: str, confidence: float = 0.9) -> DesktopLocateResult:
        if not self._check_pyautogui():
            raise RuntimeError(
                "pyautogui not installed. Run: pip install pyautogui"
            )
        if not self._check_pillow():
            raise RuntimeError(
                "Pillow not installed. Run: pip install pillow"
            ) from None
        import pyautogui
        from PIL import Image

        try:
            img_data = base64.b64decode(image_b64)
            needle = Image.open(io.BytesIO(img_data))
        except Exception:
            return DesktopLocateResult(found=False)

        needle_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                needle.save(f, format="PNG")
                needle_path = f.name

            location = pyautogui.locateOnScreen(needle_path, confidence=confidence)
            if location is not None:
                x, y = pyautogui.center(location)
                return DesktopLocateResult(
                    found=True,
                    x=int(x),
                    y=int(y),
                    width=int(location.width),
                    height=int(location.height),
                    confidence=confidence,
                )
            return DesktopLocateResult(found=False)
        except Exception:
            return DesktopLocateResult(found=False)
        finally:
            if needle_path:
                with suppress(OSError):
                    os.unlink(needle_path)

    def screenshot_with_cursor(self) -> DesktopScreenState:
        state = self.screenshot()
        if self._check_pyautogui():
            import pyautogui
            x, y = pyautogui.position()
            state.cursor_x = int(x)
            state.cursor_y = int(y)
        return state

    def is_idle(self, max_idle_seconds: int = 600) -> bool:
        return (time.time() - self._last_used) > max_idle_seconds

    # ------------------------------------------------------------------
    # Accessibility tree (Windows: UIAutomation)
    # ------------------------------------------------------------------

    def accessibility_tree(self, max_depth: int = 6,
                           max_nodes: int = 500) -> list[dict[str, Any]]:
        """Walk the active window's UI automation tree.

        Returns a flat list of nodes, each with ``name``, ``control_type``,
        ``automation_id``, ``class_name``, ``rect`` (physical pixel screen
        coordinates), ``depth``, and ``focusable`` keys. Returns an empty
        list (and surfaces a single error item) on unsupported platforms.
        """
        if _PLATFORM != "win32":
            return [{"error": f"accessibility_tree only supported on Windows (got {_PLATFORM})"}]
        try:
            import uiautomation as uia  # type: ignore
        except ImportError:
            return [{
                "error": (
                    "uiautomation package required. "
                    "Install with: pip install uiautomation"
                )
            }]

        try:
            root = uia.GetForegroundControl()
        except Exception as exc:
            return [{"error": f"Failed to acquire foreground control: {exc}"}]
        if root is None:
            return [{"error": "No foreground window detected"}]

        nodes: list[dict[str, Any]] = []
        self._walk_uia(root, 0, max_depth, max_nodes, nodes, uia)
        return nodes

    @staticmethod
    def _walk_uia(node, depth: int, max_depth: int, max_nodes: int,
                  out: list[dict[str, Any]], uia) -> None:
        if len(out) >= max_nodes or node is None or depth > max_depth:
            return
        try:
            rect = node.BoundingRectangle
            entry = {
                "name": node.Name or "",
                "control_type": getattr(node, "ControlTypeName", "") or "",
                "automation_id": getattr(node, "AutomationId", "") or "",
                "class_name": getattr(node, "ClassName", "") or "",
                "rect": {
                    "left": int(rect.left),
                    "top": int(rect.top),
                    "right": int(rect.right),
                    "bottom": int(rect.bottom),
                },
                "depth": depth,
                "focusable": bool(getattr(node, "IsKeyboardFocusable", False)),
            }
            out.append(entry)
        except Exception:
            return
        try:
            children = list(node.GetChildren())
        except Exception:
            return
        for c in children:
            if len(out) >= max_nodes:
                return
            EncreDesktopSession._walk_uia(c, depth + 1, max_depth, max_nodes, out, uia)

    def find_element_by_name(self, name: str, control_type: str | None = None,
                             max_depth: int = 8, max_nodes: int = 2000) -> dict[str, Any]:
        """Find a UIA element by accessible name. Returns center coords or error."""
        tree = self.accessibility_tree(max_depth=max_depth, max_nodes=max_nodes)
        if tree and "error" in tree[0]:
            return tree[0]
        target = name.strip().lower()
        for n in tree:
            if not n.get("name"):
                continue
            if target not in n["name"].lower():
                continue
            if control_type and control_type.lower() != n.get("control_type", "").lower():
                continue
            r = n["rect"]
            cx = (r["left"] + r["right"]) // 2
            cy = (r["top"] + r["bottom"]) // 2
            return {
                "found": True,
                "name": n["name"],
                "control_type": n["control_type"],
                "center_x": cx,
                "center_y": cy,
                "rect": r,
            }
        return {"found": False, "queried": name}

    @staticmethod
    def _ensure_pillow() -> None:
        try:
            import PIL  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Pillow not installed. Run: pip install pillow"
            ) from None
