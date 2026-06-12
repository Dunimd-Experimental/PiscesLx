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

"""Screen OCR — extract visible text with bounding boxes from the desktop.

Uses Windows.Media.Ocr on Windows 10+ (no external dependencies).
Falls back to pytesseract (optional) for cross-platform support.
"""

from __future__ import annotations

import io
import logging
from typing import Any

logger = logging.getLogger("encre.computer.ocr")


def _ocr_winrt(image_bytes: bytes) -> list[dict[str, Any]] | None:
    """Windows OCR via winrt-Windows.Media.Ocr (Win 10+ built-in)."""
    try:
        from winrt.windows.media.ocr import OcrEngine
        from winrt.windows.graphics.imaging import BitmapDecoder, BitmapPixelFormat
        from winrt.windows.storage.streams import (
            InMemoryRandomAccessStream,
        )
    except ImportError:
        return None

    import asyncio

    async def _run() -> list[dict[str, Any]]:
        engine = OcrEngine.try_create_from_user_profile_languages()
        if engine is None:
            return []

        stream = InMemoryRandomAccessStream()
        await stream.write_async(image_bytes)
        stream.seek(0)

        decoder = await BitmapDecoder.create_async(stream)
        software_bitmap = await decoder.get_software_bitmap_async(
            BitmapPixelFormat.bgra8
        )

        result = await engine.recognize_async(software_bitmap)

        lines = []
        for line in result.lines:
            r = line.bounding_rect
            lines.append({
                "text": line.text,
                "x": int(r.x),
                "y": int(r.y),
                "width": int(r.width),
                "height": int(r.height),
                "center_x": int(r.x + r.width / 2),
                "center_y": int(r.y + r.height / 2),
            })
        return lines

    try:
        loop = asyncio.get_running_loop()
        future = asyncio.run_coroutine_threadsafe(_run(), loop)
        return future.result(timeout=30)
    except RuntimeError:
        return asyncio.run(_run())
    except Exception as e:
        logger.warning("Windows OCR failed: %s", e)
        return None


def _ocr_pytesseract(image_bytes: bytes) -> list[dict[str, Any]] | None:
    """Fallback OCR via pytesseract (requires Tesseract installed)."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return None

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        lines = []
        seen = set()
        for i in range(len(data["text"])):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            # Deduplicate near-identical boxes (pytesseract often emits word fragments)
            key = (x // 5, y // 5, text.lower())
            if key in seen:
                continue
            seen.add(key)
            lines.append({
                "text": text,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "center_x": x + w // 2,
                "center_y": y + h // 2,
            })
        return lines
    except Exception as e:
        logger.warning("pytesseract OCR failed: %s", e)
        return None


def ocr_image(image_bytes: bytes) -> list[dict[str, Any]]:
    """Extract visible text + bounding boxes from a PNG screenshot.

    Tries Windows OCR first (fast, built-in), then pytesseract.
    Returns a list of dicts::

        [
            {
                "text": "Login",
                "x": 450, "y": 300,
                "width": 100, "height": 40,
                "center_x": 500, "center_y": 320,
            },
            ...
        ]

    Returns empty list if no OCR backend is available.
    """
    result = _ocr_winrt(image_bytes)
    if result is not None:
        return result

    result = _ocr_pytesseract(image_bytes)
    if result is not None:
        return result

    logger.warning(
        "No OCR backend available. "
        "On Windows, install: pip install winrt-Windows.Media.Ocr. "
        "Cross-platform: pip install pytesseract (requires Tesseract binary)"
    )
    return []


__all__ = ["ocr_image"]
