#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from MCP import mcp
from typing import Dict, Any, List
from duckduckgo_search import DDGS

@mcp.tool()
def web_search(query: str, max_results: int = 5, region: str = "wt-wt", safesearch: str = "moderate", time: str = None) -> Dict[str, Any]:
    """Perform a web search (DuckDuckGo) and return top results.
    
    Args:
        query: Search query string.
        max_results: Maximum number of results to return (1-25 typical).
        region: Region code (e.g., 'wt-wt', 'us-en', 'cn-zh').
        safesearch: 'on' | 'moderate' | 'off'.
        time: Optional time range filter: 'd' (day), 'w' (week), 'm' (month), 'y' (year).
    """
    try:
        ddgs = DDGS()
        results: List[Dict[str, Any]] = []
        for i, item in enumerate(ddgs.text(query, region=region, safesearch=safesearch, timelimit=time) or []):
            if i >= max_results:
                break
            results.append({
                "title": item.get("title"),
                "href": item.get("href"),
                "body": item.get("body")
            })
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
