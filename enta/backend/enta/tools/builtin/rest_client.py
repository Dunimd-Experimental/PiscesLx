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



import json
from typing import Any

import httpx

from enta.tools.base import build_tool


async def _rest_client_execute(**kwargs: Any) -> str:
    method = kwargs.get("method", "GET")
    url = kwargs.get("url", "")
    headers = kwargs.get("headers", {}) or {}
    body = kwargs.get("body", "")
    timeout = kwargs.get("timeout", 30)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(float(timeout)),
            follow_redirects=True,
        ) as client:
            resp = await client.request(
                method=method,
                url=url,
                headers={k: str(v) for k, v in headers.items()},
                content=body if body else None,
            )

            status = resp.status_code
            content_type = resp.headers.get("content-type", "")
            response_body = resp.text

            if len(response_body) > 50000:
                response_body = response_body[:50000] + "\n... (truncated to 50K chars)"

            if "application/json" in content_type or response_body.strip().startswith(("{", "[")):
                try:
                    parsed = json.loads(response_body)
                    return json.dumps({"status": status, "headers": dict(resp.headers), "body": parsed}, indent=2, default=str)  # noqa: E501
                except (json.JSONDecodeError, ValueError):
                    pass

            return f"HTTP {status}\n{response_body}"

    except httpx.TimeoutException:
        return f"Error: Request timed out after {timeout}s for {url}"
    except httpx.InvalidURL:
        return f"Error: Invalid URL: {url}"
    except Exception as e:
        return f"Error making {method} request to {url}: {e}"


EncreRESTTool = build_tool(
    name="rest_client",
    description="Make HTTP requests to REST and GraphQL APIs",
    input_schema={
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "description": "HTTP method to use",
            },
            "url": {
                "type": "string",
                "description": "The URL to send the request to",
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers as key-value pairs",
            },
            "body": {
                "type": "string",
                "description": "Request body (JSON string, form data, etc.)",
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds (default: 30)",
            },
        },
        "required": ["method", "url"],
    },
    execute=_rest_client_execute,
    intents=["coding", "system"],
    is_concurrency_safe=lambda _: True,
)
