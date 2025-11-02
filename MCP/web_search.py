#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import sys
from pathlib import Path
from typing import Dict, Any, List
from duckduckgo_search import DDGS

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.mcp import PiscesLxCoreMCPPlaza

# Instantiate MCP for tool registration
mcp = PiscesLxCoreMCPPlaza()

@mcp.tool()
def web_search(
    query: str, 
    max_results: int = 5, 
    region: str = "wt-wt", 
    safesearch: str = "moderate", 
    time: str = None
) -> Dict[str, Any]:
    """Execute a DuckDuckGo web search and return structured results.

    Args:
        query (str): Search keyword or phrase.
        max_results (int): Number of results to retrieve, default is 5.
        region (str): Geographic region for search results, e.g., 'wt-wt', 'us-en'.
        safesearch (str): Safe search setting ('on', 'moderate', 'off').
        time (str, optional): Time-based filtering ('d', 'w', 'm', 'y').

    Returns:
        Dict[str, Any]: Contains status, query, result count, and list of results.
                        On failure, includes error details.
    """
    try:
        ddgs = DDGS()
        results: List[Dict[str, Any]] = []
        search_results = ddgs.text(
            query, 
            region=region, 
            safesearch=safesearch, 
            timelimit=time
        )
        
        # Normalize empty generator to empty list
        if search_results is None:
            search_results = []
            
        for i, item in enumerate(search_results):
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
