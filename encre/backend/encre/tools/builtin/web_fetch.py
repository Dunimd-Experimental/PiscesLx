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

import httpx

from encre.tools.base import build_tool


async def _web_fetch_execute(**kwargs: Any) -> str:
    url = kwargs.get("url", "")

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "text/html" in content_type or "html" in content_type:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "lxml")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                return "\n".join(lines[:500])
            else:
                text = resp.text
                if len(text) > 50000:
                    text = text[:50000] + "\n... (truncated)"
                return text

    except httpx.TimeoutException:
        return f"Error: Request timed out fetching {url}"
    except httpx.HTTPStatusError as e:
        return f"Error HTTP {e.response.status_code}: {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"


EncreWebFetchTool = build_tool(
    name="web_fetch",
    description="Fetch a URL and return its content as markdown",
    input_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch content from",
            },
        },
        "required": ["url"],
    },
    execute=_web_fetch_execute,
    intents=["general", "research"],
    is_concurrency_safe=lambda _: True,
)
