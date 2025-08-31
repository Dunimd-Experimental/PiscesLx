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

import re
import asyncio
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from .simple_mcp import register_tool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FetchTool:
    """Web content fetching tool"""
    
    def __init__(self):
        self.name = "fetch"
        self.description = "Fetch and extract content from web URLs"
        self.user_agent = "PiscesL1-MCP/1.0 (+https://github.com/piscesl1)"
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch content from"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length to return",
                    "default": 5000,
                    "minimum": 100,
                    "maximum": 50000
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Extract clean text from HTML",
                    "default": True
                },
                "include_metadata": {
                    "type": "boolean", 
                    "description": "Include page metadata",
                    "default": False
                }
            },
            "required": ["url"]
        }
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract clean text from HTML content"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _get_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract page metadata"""
        metadata = {}
        
        # Title
        title = soup.find('title')
        if title:
            metadata['title'] = title.get_text().strip()
        
        # Meta description
        desc = soup.find('meta', attrs={'name': 'description'})
        if desc:
            metadata['description'] = desc.get('content', '').strip()
        
        # Meta keywords
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        if keywords:
            metadata['keywords'] = keywords.get('content', '').strip()
        
        return metadata
    
    def execute(self, url: str, max_length: int = 5000, extract_text: bool = True, 
                include_metadata: bool = False) -> Dict[str, Any]:
        """Execute web fetch operation"""
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return {
                    "success": False,
                    "error": "Invalid URL format"
                }
            
            # Make request
            response = requests.get(
                url, 
                headers={'User-Agent': self.user_agent},
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            
            content = response.text
            content_type = response.headers.get('content-type', '')
            
            result = {
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "encoding": response.encoding
            }
            
            # Handle HTML content
            if 'text/html' in content_type.lower():
                soup = BeautifulSoup(content, 'html.parser')
                
                if include_metadata:
                    result['metadata'] = self._get_metadata(soup)
                
                if extract_text:
                    content = self._extract_text_from_html(content)
                
            # Truncate content if needed
            if len(content) > max_length:
                content = content[:max_length] + "... [truncated]"
            
            result['content'] = content
            result['length'] = len(content)
            
            return {
                "success": True,
                "data": result
            }
            
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Connection error"
            }
        except requests.exceptions.HTTPError as e:
            return {
                "success": False,
                "error": f"HTTP error {e.response.status_code}: {e.response.reason}"
            }
        except Exception as e:
            logger.error(f"Fetch error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

# Pisces L1 MCP广场集成
from . import register_custom_tool

# 注册获取工具到MCP广场
register_custom_tool(
    name="fetch",
    description="获取网页内容，支持HTML文本提取和元数据",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要获取的网页URL"
            },
            "max_length": {
                "type": "integer",
                "description": "返回内容的最大长度",
                "default": 5000,
                "minimum": 100,
                "maximum": 50000
            },
            "extract_text": {
                "type": "boolean",
                "description": "是否从HTML提取纯文本",
                "default": True
            },
            "include_metadata": {
                "type": "boolean",
                "description": "是否包含页面元数据",
                "default": False
            }
        },
        "required": ["url"]
    },
    function=FetchTool().execute,
    category="Web"
)