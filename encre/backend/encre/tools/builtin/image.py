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
import os
from typing import Any

from encre.tools.base import build_tool


async def _image_execute(**kwargs: Any) -> str:
    action = kwargs.get("action", "info")
    file_path = kwargs.get("file_path", "")
    options = kwargs.get("options", {}) or {}

    if not os.path.isfile(file_path):
        return f"Error: File not found: {file_path}"

    try:
        try:
            from PIL import Image, ExifTags
        except ImportError:
            return "Error: Pillow not installed. Install with: pip install Pillow"

        with Image.open(file_path) as img:
            if action == "info":
                info = {
                    "file": file_path,
                    "format": img.format,
                    "mode": img.mode,
                    "size": {"width": img.width, "height": img.height},
                    "file_size_bytes": os.path.getsize(file_path),
                }
                exif_data = img.getexif()
                if exif_data:
                    exif = {}
                    for tag_id, value in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        exif[tag_name] = str(value)
                    info["exif"] = exif
                return json.dumps(info, indent=2, default=str)

            elif action == "ocr":
                try:
                    import pytesseract
                except ImportError:
                    return "Error: pytesseract not installed. Install with: pip install pytesseract"
                text = pytesseract.image_to_string(img)
                return text if text.strip() else "(no text detected)"

            elif action == "convert":
                target_format = options.get("format", "PNG")
                output_path = options.get("output", "")
                if not output_path:
                    base = os.path.splitext(file_path)[0]
                    ext = target_format.lower()
                    output_path = f"{base}_converted.{ext}"
                converted = img.convert("RGB")
                converted.save(output_path, format=target_format)
                return f"Converted {file_path} to {output_path} (format: {target_format})"

    except Exception as e:
        return f"Error processing image: {e}"


EncreImageTool = build_tool(
    name="image",
    description="Read and analyze image files (format, dimensions, EXIF, OCR text)",
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["info", "ocr", "convert"],
                "description": "Action to perform on the image",
            },
            "file_path": {
                "type": "string",
                "description": "Path to the image file",
            },
            "options": {
                "type": "object",
                "description": "Additional options (e.g. convert format target)",
            },
        },
        "required": ["action", "file_path"],
    },
    execute=_image_execute,
    intents=["data", "research"],
    is_concurrency_safe=lambda _: True,
)
