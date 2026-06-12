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

from typing import Any

from encre.tools.base import build_tool


async def _pdf_execute(**kwargs: Any) -> str:
    action = kwargs.get("action", "read")
    file_path = kwargs.get("file_path", "")
    pages = kwargs.get("pages", "")

    try:
        pdfplumber = None
        PdfReader = None
        try:
            import pdfplumber  # noqa: F811
        except ImportError:
            pass

        if pdfplumber is None:
            try:
                from PyPDF2 import PdfReader  # noqa: F811
            except ImportError:
                pass

        if pdfplumber is None and PdfReader is None:
            return (
                "Error: No PDF library available. "
                "Install one with: pip install pdfplumber or pip install PyPDF2"
            )

        if pdfplumber is not None:
            with pdfplumber.open(file_path) as pdf:
                if action == "metadata":
                    return str({"metadata": pdf.metadata, "total_pages": len(pdf.pages)})
                page_nums = _parse_pages(pages, len(pdf.pages))
                texts = []
                for pn in page_nums:
                    page = pdf.pages[pn - 1]
                    text = page.extract_text()
                    if text:
                        texts.append(f"--- Page {pn} ---\n{text}")
                return "\n\n".join(texts) if texts else "(no text extracted)"

        else:
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)

            if action == "metadata":
                info = reader.metadata
                meta = {k: str(v) for k, v in info.items()} if info else {}
                meta["total_pages"] = num_pages
                return str(meta)

            page_nums = _parse_pages(pages, num_pages)
            texts = []
            for pn in page_nums:
                page = reader.pages[pn - 1]
                text = page.extract_text()
                if text:
                    texts.append(f"--- Page {pn} ---\n{text}")
            return "\n\n".join(texts) if texts else "(no text extracted)"

    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error processing PDF: {e}"


def _parse_pages(pages: str, num_pages: int) -> list[int]:
    if not pages:
        end = min(num_pages, 50)
        return list(range(1, end + 1))
    result = []
    for part in pages.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


EncrePDFTool = build_tool(
    name="pdf",
    description="Read, extract text, and parse PDF documents",
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "extract_text", "metadata"],
                "description": "Action to perform on the PDF",
            },
            "file_path": {
                "type": "string",
                "description": "Path to the PDF file",
            },
            "pages": {
                "type": "string",
                "description": "Page range to read (e.g. '1-5' or '1,3,5')",
            },
        },
        "required": ["action", "file_path"],
    },
    execute=_pdf_execute,
    intents=["data", "research"],
    is_concurrency_safe=lambda _: True,
)
