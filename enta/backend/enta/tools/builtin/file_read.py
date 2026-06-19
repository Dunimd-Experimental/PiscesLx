#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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
import json
import os
from typing import Any

from enta.native import read_file as _native_read
from enta.tools.base import build_tool

_IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # WEBP starts with RIFF and "WEBP" at offset 8
    (b"BM", "image/bmp"),
]


def _detect_image_mime(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        return _IMAGE_EXTS[ext]
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    for magic, mime in _IMAGE_MAGIC:
        if mime == "image/webp":
            if head.startswith(b"RIFF") and len(head) >= 12 and head[8:
                12] == b"WEBP":
                return mime
        elif head.startswith(magic):
            return mime
    return None


def _is_pdf(path: str) -> bool:
    if os.path.splitext(path)[1].lower() == ".pdf":
        return True
    try:
        with open(path, "rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def _read_pdf_text(path: str, max_pages: int | None) -> str:
    """Extract text from a PDF. Uses pypdf if installed; otherwise raises."""
    try:
        import pypdf  # type: ignore
    except ImportError:
        try:
            import PyPDF2 as pypdf  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "PDF reading requires 'pypdf' (preferred) or 'PyPDF2'. "
                "Install with: pip install pypdf"
            ) from exc

    reader = pypdf.PdfReader(path)
    out: list[str] = []
    total = len(reader.pages)
    limit = total if not max_pages else min(total, max_pages)
    for i in range(limit):
        try:
            txt = reader.pages[i].extract_text() or ""
        except Exception:
            txt = ""
        out.append(f"--- page {i + 1} ---\n{txt}")
    if limit < total:
        out.append(f"\n... ({total - limit} more page(s) not shown)")
    return "\n\n".join(out)


def _looks_like_image(path: str) -> bool:
    return _detect_image_mime(path) is not None


async def _file_read_execute(**kwargs: Any) -> str:
    file_path = kwargs.get("file_path", "")
    if not file_path:
        return "Error: file_path is required"

    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    if os.path.isdir(file_path):
        return f"Error: Path is a directory, not a file: {file_path}"

    as_image = bool(kwargs.get("as_image", False))
    mime = _detect_image_mime(file_path) if (as_image or _looks_like_image(file_path)) else None
    if mime is not None:
        try:
            with open(file_path, "rb") as fh:
                data = fh.read()
        except PermissionError:
            return f"Error: Permission denied: {file_path}"
        except Exception as e:
            return f"Error reading image: {e}"
        return json.dumps({
            "type": "image",
            "path": file_path,
            "mime": mime,
            "size_bytes": len(data),
            "base64": base64.b64encode(data).decode("ascii"),
        }, ensure_ascii=False)

    if _is_pdf(file_path):
        try:
            text = _read_pdf_text(file_path, kwargs.get("max_pages"))
        except RuntimeError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error reading PDF: {exc}"
        return text

    # Text fallback (uses native paginated read)
    limit = int(kwargs.get("limit", 0) or 0)
    offset = int(kwargs.get("offset", 1) or 1)
    try:
        return _native_read(file_path, offset, limit)
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied: {file_path}"
    except UnicodeDecodeError:
        # Binary file the native layer couldn't decode -- surface size.
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = -1
        return (
            f"Error: file appears to be binary (no UTF-8 text). "
            f"Size: {size} bytes. Use as_image=true for images."
        )
    except Exception as e:
        return f"Error reading file: {e}"


EncreFileReadTool = build_tool(
    name="file_read",
    description=(
        "Read the contents of a file. Text files return paginated lines "
        "(offset/limit). Images (png/jpg/gif/webp/bmp) return a base64 + mime "
        "envelope so multimodal models can consume them directly. PDFs return "
        "extracted text (use max_pages to cap).\n\n"
        "DO NOT use this tool for reading persistent memory -- use memory_read "
        "instead. DO NOT use this tool for reading the user profile -- use "
        "memory_profile instead."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to read",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read (text mode)",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from, 1-indexed (text mode)",
            },
            "as_image": {
                "type": "boolean",
                "description": (
                    "Force the file to be returned as a base64 image envelope. "
                    "Auto-detected for known image extensions/magic numbers."
                ),
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum number of PDF pages to extract (default: all)",
            },
        },
        "required": ["file_path"],
    },
    execute=_file_read_execute,
    intents=["general", "coding", "data"],
    is_concurrency_safe=lambda _: True,
)
