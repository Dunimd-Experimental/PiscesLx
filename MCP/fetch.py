#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import re
import sys
import asyncio
import requests
from pathlib import Path
from typing import Dict, Any
from bs4 import BeautifulSoup

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.mcp import PiscesLxCoreMCPPlaza

# Create mcp instance for tool registration
mcp = PiscesLxCoreMCPPlaza()

@mcp.tool()
def fetch_url(url: str, max_length: int = 5000, extract_text: bool = True) -> Dict[str, Any]:
    """
    Fetch and extract content from a web URL.
    
    Args:
        url (str): The URL to fetch content from.
        max_length (int, optional): Maximum length of extracted text. Defaults to 5000.
        extract_text (bool, optional): Whether to extract text content. Defaults to True.
        
    Returns:
        Dict[str, Any]: A dictionary containing the fetch result with keys:
            - success (bool): Indicates if the fetch was successful.
            - url (str): The fetched URL.
            - title (str): The page title.
            - description (str): The meta description.
            - text_content (str): Extracted text content.
            - links (list): List of extracted links.
            - status_code (int): HTTP status code.
            - error (str): Error message if any.
            - error_type (str): Type of error if any.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract metadata
        title = soup.find('title')
        title = title.get_text().strip() if title else ""
        
        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description.get('content', '').strip() if meta_description else ""
        
        # Extract text content
        if extract_text:
            text_content = _extract_text_from_html(str(soup))
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            if len(text_content) > max_length:
                text_content = text_content[:max_length] + "..."
        else:
            text_content = ""
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True)[:10]:
            links.append({
                'text': link.get_text().strip(),
                'url': link['href']
            })
        
        return {
            "success": True,
            "url": url,
            "title": title,
            "description": description,
            "text_content": text_content,
            "links": links,
            "status_code": response.status_code
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Network error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def _extract_text_from_html(html: str) -> str:
    """
    Extract clean text from HTML content.
    
    Args:
        html (str): HTML content to extract text from.
        
    Returns:
        str: Cleaned text content.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted tags
    for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
        tag.decompose()
    
    # Get text from main content areas
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main'))
    if main_content:
        text = main_content.get_text(separator=' ', strip=True)
    else:
        text = soup.get_text(separator=' ', strip=True)
    
    return text

def _get_metadata(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extract metadata from HTML.
    
    Args:
        soup (BeautifulSoup): BeautifulSoup object containing parsed HTML.
        
    Returns:
        Dict[str, str]: Dictionary containing extracted metadata.
    """
    metadata = {}
    
    # Title
    title = soup.find('title')
    if title:
        metadata['title'] = title.get_text().strip()
    
    # Meta description
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc:
        metadata['description'] = meta_desc.get('content', '').strip()
    
    # Meta keywords
    meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
    if meta_keywords:
        metadata['keywords'] = meta_keywords.get('content', '').strip()
    
    # Open Graph data
    og_title = soup.find('meta', attrs={'property': 'og:title'})
    if og_title:
        metadata['og_title'] = og_title.get('content', '').strip()
    
    og_description = soup.find('meta', attrs={'property': 'og:description'})
    if og_description:
        metadata['og_description'] = og_description.get('content', '').strip()
    
    return metadata
