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
Web Search Tool - DuckDuckGo search integration
"""

from typing import Any, Dict, List

from .base import POPSSMCPToolBase, POPSSMCPToolResult


class WebSearchTool(POPSSMCPToolBase):
    name = "web_search"
    description = "Search the web using DuckDuckGo search engine"
    category = "search"
    tags = ["web", "search", "internet"]
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            },
            "region": {
                "type": "string",
                "description": "Geographic region for search results (e.g., 'wt-wt', 'us-en', 'cn-zh')",
                "default": "wt-wt"
            },
            "safesearch": {
                "type": "string",
                "description": "Safe search setting: 'on', 'moderate', 'off'",
                "default": "moderate"
            },
            "timelimit": {
                "type": "string",
                "description": "Time limit: 'd' (day), 'w' (week), 'm' (month), 'y' (year)",
                "default": None
            }
        },
        "required": ["query"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        region = arguments.get("region", "wt-wt")
        safesearch = arguments.get("safesearch", "moderate")
        timelimit = arguments.get("timelimit")
        
        if not query:
            return self._create_error_result("Query is required", "ValidationError")
        
        try:
            from duckduckgo_search import DDGS
            
            ddgs = DDGS()
            results: List[Dict[str, Any]] = []
            
            search_results = ddgs.text(
                query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit
            )
            
            if search_results is None:
                search_results = []
            
            for i, item in enumerate(search_results):
                if i >= max_results:
                    break
                results.append({
                    "title": item.get("title"),
                    "url": item.get("href"),
                    "snippet": item.get("body"),
                })
            
            return self._create_success_result({
                "query": query,
                "count": len(results),
                "results": results,
            })
            
        except ImportError:
            return self._create_error_result(
                "duckduckgo_search not installed. Install with: pip install duckduckgo-search",
                "DependencyError"
            )
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)


class ImageSearchTool(POPSSMCPToolBase):
    name = "image_search"
    description = "Search for images using DuckDuckGo"
    category = "search"
    tags = ["image", "search", "web"]
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Image search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results",
                "default": 5
            }
        },
        "required": ["query"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        
        if not query:
            return self._create_error_result("Query is required", "ValidationError")
        
        try:
            from duckduckgo_search import DDGS
            
            ddgs = DDGS()
            results = []
            
            search_results = ddgs.images(query)
            
            if search_results is None:
                search_results = []
            
            for i, item in enumerate(search_results):
                if i >= max_results:
                    break
                results.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "thumbnail": item.get("thumbnail"),
                    "width": item.get("width"),
                    "height": item.get("height"),
                })
            
            return self._create_success_result({
                "query": query,
                "count": len(results),
                "results": results,
            })
            
        except ImportError:
            return self._create_error_result(
                "duckduckgo_search not installed",
                "DependencyError"
            )
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)
