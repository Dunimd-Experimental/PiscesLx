#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

"""
Fetch Tool - HTTP requests and web content extraction
"""

import json
from typing import Any, Dict, Optional

from .base import POPSSMCPToolBase, POPSSMCPToolResult


class FetchTool(POPSSMCPToolBase):
    name = "fetch"
    description = "Fetch content from a URL with optional content extraction"
    category = "web"
    tags = ["http", "web", "fetch", "request"]
    
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch"
            },
            "method": {
                "type": "string",
                "description": "HTTP method: GET, POST, PUT, DELETE",
                "default": "GET"
            },
            "headers": {
                "type": "object",
                "description": "HTTP headers as key-value pairs",
                "default": {}
            },
            "body": {
                "type": "string",
                "description": "Request body for POST/PUT requests"
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds",
                "default": 30
            },
            "extract_text": {
                "type": "boolean",
                "description": "Extract text content from HTML",
                "default": True
            },
            "follow_redirects": {
                "type": "boolean",
                "description": "Follow HTTP redirects",
                "default": True
            }
        },
        "required": ["url"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        url = arguments.get("url", "")
        method = arguments.get("method", "GET").upper()
        headers = arguments.get("headers", {})
        body = arguments.get("body")
        timeout = arguments.get("timeout", 30)
        extract_text = arguments.get("extract_text", True)
        follow_redirects = arguments.get("follow_redirects", True)
        
        if not url:
            return self._create_error_result("URL is required", "ValidationError")
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            session.headers.update(headers)
            
            response = session.request(
                method=method,
                url=url,
                data=body,
                timeout=timeout,
                allow_redirects=follow_redirects
            )
            
            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "url": str(response.url),
            }
            
            content_type = response.headers.get("Content-Type", "")
            
            if "application/json" in content_type:
                try:
                    result["content"] = response.json()
                    result["content_type"] = "json"
                except json.JSONDecodeError:
                    result["content"] = response.text
                    result["content_type"] = "text"
            
            elif "text/html" in content_type and extract_text:
                soup = BeautifulSoup(response.text, "html.parser")
                
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                text = soup.get_text(separator="\n", strip=True)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                result["content"] = "\n".join(lines[:100])
                result["content_type"] = "html_text"
                result["title"] = soup.title.string if soup.title else None
            
            else:
                result["content"] = response.text[:10000] if len(response.text) > 10000 else response.text
                result["content_type"] = "text"
            
            return self._create_success_result(result)
            
        except ImportError:
            return self._create_error_result(
                "requests and beautifulsoup4 required. Install with: pip install requests beautifulsoup4",
                "DependencyError"
            )
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)


class URLInfoTool(POPSSMCPToolBase):
    name = "url_info"
    description = "Get information about a URL without fetching full content"
    category = "web"
    tags = ["url", "http", "info"]
    
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to check"
            }
        },
        "required": ["url"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        url = arguments.get("url", "")
        
        if not url:
            return self._create_error_result("URL is required", "ValidationError")
        
        try:
            import requests
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            
            response = requests.head(url, timeout=10, allow_redirects=True)
            
            return self._create_success_result({
                "url": url,
                "domain": parsed.netloc,
                "scheme": parsed.scheme,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
                "last_modified": response.headers.get("Last-Modified"),
                "final_url": str(response.url),
            })
            
        except ImportError:
            return self._create_error_result(
                "requests required. Install with: pip install requests",
                "DependencyError"
            )
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)
