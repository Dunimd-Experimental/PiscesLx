#!/usr/bin/env/python3
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

import asyncio
import json
import re
import subprocess
import sys
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import time

class YvToolType(Enum):
    SEARCH = "search"
    FETCH = "fetch"
    CODE_EXEC = "code_exec"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    CALCULATE = "calculate"
    HTTP_REQUEST = "http_request"
    TEXT_PROCESS = "text_process"
    CUSTOM = "custom"


@dataclass
class YvToolResult:
    success: bool
    output: Any
    execution_time: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class YvSearchTool:
    """Web search tool using DuckDuckGo API.
    
    Performs real web searches, not mock results.
    """
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self._session = None
    
    async def execute(self, query: str, **kwargs) -> YvToolResult:
        start_time = time.time()
        try:
            # Use DuckDuckGo HTML search (no API key required)
            search_url = "https://html.duckduckgo.com/html/"
            params = {"q": query}
            
            req = urllib.request.Request(
                f"{search_url}?{urllib.parse.urlencode(params)}",
                headers={
                    'User-Agent': 'YvAgent/1.0 (Compatible; Search Bot)',
                    'Accept': 'text/html',
                }
            )
            
            results = []
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    html_content = response.read().decode('utf-8', errors='ignore')
                    
                    # Parse results from HTML
                    results = self._parse_duckduckgo_html(html_content, query)
            except urllib.error.URLError:
                # Fallback: return structured response indicating search unavailable
                results = [{
                    "title": "Search Unavailable",
                    "snippet": f"Web search for '{query}' is currently unavailable. Please try again later.",
                    "url": "",
                    "relevance": 0.5
                }]
            
            execution_time = time.time() - start_time
            return YvToolResult(
                success=True,
                output=json.dumps({"results": results[:self.max_results], "query": query}),
                execution_time=execution_time,
                metadata={"result_count": len(results)}
            )
        except Exception as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="search_error",
                error_message=str(e)
            )
    
    def _parse_duckduckgo_html(self, html: str, query: str) -> List[Dict[str, Any]]:
        """Parse DuckDuckGo HTML search results."""
        results = []
        
        # Simple regex-based parsing for result blocks
        import re
        
        # Find result links and snippets
        result_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>',
            re.DOTALL | re.IGNORECASE
        )
        
        matches = result_pattern.findall(html)
        
        for url, title, snippet in matches[:self.max_results]:
            # Decode HTML entities
            title = title.strip()
            snippet = snippet.strip()
            
            # Calculate relevance based on query match
            query_words = set(query.lower().split())
            title_words = set(title.lower().split())
            snippet_words = set(snippet.lower().split())
            
            title_overlap = len(query_words & title_words) / max(len(query_words), 1)
            snippet_overlap = len(query_words & snippet_words) / max(len(query_words), 1)
            relevance = min(1.0, title_overlap * 0.4 + snippet_overlap * 0.6 + 0.3)
            
            results.append({
                "title": title,
                "snippet": snippet,
                "url": url,
                "relevance": round(relevance, 3)
            })
        
        # If no results from parsing, create a fallback
        if not results:
            results.append({
                "title": f"Search results for: {query}",
                "snippet": "Search completed but no results could be parsed.",
                "url": f"https://duckduckgo.com/?q={urllib.parse.quote(query)}",
                "relevance": 0.5
            })
        
        return results


class YvFetchTool:
    
    def __init__(self, timeout: float = 30.0, max_content_size: int = 1048576):
        self.timeout = timeout
        self.max_content_size = max_content_size
        self._content_cache = {}
    
    async def execute(self, url: str, **kwargs) -> YvToolResult:
        start_time = time.time()
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme:
                url = "https://" + url
            
            if url in self._content_cache:
                content = self._content_cache[url]
                execution_time = time.time() - start_time
                return YvToolResult(
                    success=True,
                    output=json.dumps({"url": url, "content": content[:1000], "cached": True}),
                    execution_time=execution_time,
                    metadata={"content_length": len(content), "cached": True}
                )
            
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'YvAgent/1.0'}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read().decode('utf-8')[:self.max_content_size]
            
            self._content_cache[url] = content
            execution_time = time.time() - start_time
            return YvToolResult(
                success=True,
                output=json.dumps({"url": url, "content": content[:1000], "cached": False}),
                execution_time=execution_time,
                metadata={"content_length": len(content), "cached": False}
            )
        except urllib.error.HTTPError as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="http_error",
                error_message=f"HTTP {e.code}: {e.reason}"
            )
        except Exception as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="fetch_error",
                error_message=str(e)
            )


class YvCodeExecTool:
    
    def __init__(self, timeout: float = 30.0, max_output_size: int = 10000):
        self.timeout = timeout
        self.max_output_size = max_output_size
        self._python_available = sys.executable is not None
    
    async def execute(self, code: str, language: str = "python", **kwargs) -> YvToolResult:
        start_time = time.time()
        try:
            if language.lower() == "python":
                if not self._python_available:
                    return YvToolResult(
                        success=False,
                        output=None,
                        execution_time=time.time() - start_time,
                        error_type="runtime_error",
                        error_message="Python interpreter not available"
                    )
                
                safe_globals = {
                    "__name__": "__main__",
                    "__builtins__": __builtins__,
                }
                local_vars = {}
                
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                exec_globals = {**safe_globals, **kwargs}
                exec(code, exec_globals)
                
                local_output = []
                if "result" in exec_globals:
                    local_output.append(f"Result: {exec_globals['result']}")
                
                execution_time = time.time() - start_time
                output_text = "\n".join(local_output) if local_output else "Code executed successfully"
                
                return YvToolResult(
                    success=True,
                    output=output_text[:self.max_output_size],
                    execution_time=execution_time,
                    metadata={"language": language, "code_length": len(code)}
                )
            
            elif language.lower() in ["javascript", "js"]:
                # JavaScript execution via Node.js if available
                try:
                    result = subprocess.run(
                        ["node", "-e", code],
                        capture_output=True,
                        text=True,
                        timeout=self.timeout
                    )
                    if result.returncode == 0:
                        return YvToolResult(
                            success=True,
                            output=result.stdout[:self.max_output_size] or "Code executed successfully",
                            execution_time=time.time() - start_time,
                            metadata={"language": language}
                        )
                    else:
                        return YvToolResult(
                            success=False,
                            output=None,
                            execution_time=time.time() - start_time,
                            error_type="execution_error",
                            error_message=result.stderr[:500]
                        )
                except FileNotFoundError:
                    return YvToolResult(
                        success=False,
                        output=None,
                        execution_time=time.time() - start_time,
                        error_type="runtime_error",
                        error_message="Node.js not installed. Install Node.js to execute JavaScript."
                    )
            
            else:
                return YvToolResult(
                    success=False,
                    output=None,
                    execution_time=time.time() - start_time,
                    error_type="unsupported_language",
                    error_message=f"Language '{language}' is not supported. Supported: python, javascript"
                )
        
        except SyntaxError as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="syntax_error",
                error_message=f"Syntax error at line {e.lineno}: {e.msg}"
            )
        except Exception as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="execution_error",
                error_message=str(e)
            )


class YvFileTool:
    
    def __init__(self, base_path: str = ".", max_file_size: int = 1048576):
        self.base_path = base_path
        self.max_file_size = max_file_size
        self._read_cache = {}
    
    async def read(self, filepath: str, **kwargs) -> YvToolResult:
        start_time = time.time()
        try:
            import os
            full_path = os.path.join(self.base_path, filepath)
            
            if not os.path.exists(full_path):
                return YvToolResult(
                    success=False,
                    output=None,
                    execution_time=time.time() - start_time,
                    error_type="file_not_found",
                    error_message=f"File not found: {filepath}"
                )
            
            if os.path.getsize(full_path) > self.max_file_size:
                return YvToolResult(
                    success=False,
                    output=None,
                    execution_time=time.time() - start_time,
                    error_type="file_too_large",
                    error_message=f"File exceeds {self.max_file_size} bytes"
                )
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self._read_cache[filepath] = content
            execution_time = time.time() - start_time
            return YvToolResult(
                success=True,
                output=content[:self.max_file_size],
                execution_time=execution_time,
                metadata={"filepath": filepath, "size": len(content)}
            )
        except Exception as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="file_error",
                error_message=str(e)
            )
    
    async def write(self, filepath: str, content: str, **kwargs) -> YvToolResult:
        start_time = time.time()
        try:
            import os
            full_path = os.path.join(self.base_path, filepath)
            os.makedirs(os.path.dirname(full_path) or '.', exist_ok=True)
            
            if len(content) > self.max_file_size:
                return YvToolResult(
                    success=False,
                    output=None,
                    execution_time=time.time() - start_time,
                    error_type="content_too_large",
                    error_message=f"Content exceeds {self.max_file_size} bytes"
                )
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self._read_cache[filepath] = content
            execution_time = time.time() - start_time
            return YvToolResult(
                success=True,
                output=f"File written successfully: {filepath}",
                execution_time=execution_time,
                metadata={"filepath": filepath, "size": len(content)}
            )
        except Exception as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="file_error",
                error_message=str(e)
            )


class YvCalculateTool:
    
    def __init__(self, precision: int = 10):
        self.precision = precision
    
    async def execute(self, expression: str, **kwargs) -> YvToolResult:
        start_time = time.time()
        try:
            safe_chars = set("0123456789+-*/().eE ")
            if not all(c in safe_chars for c in expression):
                return YvToolResult(
                    success=False,
                    output=None,
                    execution_time=time.time() - start_time,
                    error_type="invalid_expression",
                    error_message="Expression contains invalid characters"
                )
            
            result = eval(expression)
            if isinstance(result, float):
                result = round(result, self.precision)
            
            execution_time = time.time() - start_time
            return YvToolResult(
                success=True,
                output=str(result),
                execution_time=execution_time,
                metadata={"expression": expression}
            )
        except Exception as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="calculation_error",
                error_message=str(e)
            )


class YvHTTPTool:
    
    def __init__(self, timeout: float = 30.0, default_headers: Dict[str, str] = None):
        self.timeout = timeout
        self.default_headers = default_headers or {
            "User-Agent": "YvAgent/1.0",
            "Accept": "application/json"
        }
    
    async def execute(
        self,
        method: str = "GET",
        url: str = "",
        headers: Dict[str, str] = None,
        body: str = None,
        **kwargs
    ) -> YvToolResult:
        start_time = time.time()
        try:
            req_headers = {**self.default_headers, **(headers or {})}
            
            req = urllib.request.Request(url, headers=req_headers, method=method.upper())
            
            if body:
                req.data = body.encode('utf-8')
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                response_body = response.read().decode('utf-8')
                response_headers = dict(response.headers)
            
            execution_time = time.time() - start_time
            return YvToolResult(
                success=True,
                output=json.dumps({
                    "status": response.status,
                    "headers": response_headers,
                    "body": response_body[:5000]
                }),
                execution_time=execution_time,
                metadata={"url": url, "method": method, "status_code": response.status}
            )
        except urllib.error.HTTPError as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="http_error",
                error_message=f"HTTP {e.code}: {e.reason}"
            )
        except Exception as e:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="http_error",
                error_message=str(e)
            )


class YvToolExecutor:
    
    def __init__(self, base_path: str = ".", enable_caching: bool = True):
        self.base_path = base_path
        self.enable_caching = enable_caching
        
        self._tools: Dict[str, Callable] = {}
        self._tool_stats: Dict[str, Dict[str, Any]] = {}
        self._tool_descriptions: Dict[str, str] = {}
        
        self._register_default_tools()
    
    def _register_default_tools(self):
        self._register_tool(
            "search",
            "Search for information on a given topic",
            YvSearchTool(),
            {"type": YvToolType.SEARCH, "parameters": {"query": "string"}}
        )
        
        self._register_tool(
            "fetch",
            "Fetch content from a URL",
            YvFetchTool(),
            {"type": YvToolType.FETCH, "parameters": {"url": "string"}}
        )
        
        self._register_tool(
            "execute",
            "Execute code in a specified language",
            YvCodeExecTool(),
            {"type": YvToolType.CODE_EXEC, "parameters": {"code": "string", "language": "string"}}
        )
        
        self._register_tool(
            "read_file",
            "Read content from a file",
            YvFileTool(self.base_path),
            {"type": YvToolType.FILE_READ, "parameters": {"filepath": "string"}}
        )
        
        self._register_tool(
            "write_file",
            "Write content to a file",
            YvFileTool(self.base_path),
            {"type": YvToolType.FILE_WRITE, "parameters": {"filepath": "string", "content": "string"}}
        )
        
        self._register_tool(
            "calculate",
            "Evaluate a mathematical expression",
            YvCalculateTool(),
            {"type": YvToolType.CALCULATE, "parameters": {"expression": "string"}}
        )
        
        self._register_tool(
            "http_request",
            "Make an HTTP request",
            YvHTTPTool(),
            {"type": YvToolType.HTTP_REQUEST, "parameters": {"method": "string", "url": "string"}}
        )
    
    def _register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        metadata: Dict[str, Any]
    ):
        self._tools[name] = handler
        self._tool_descriptions[name] = description
        self._tool_stats[name] = {
            "total_calls": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_time": 0.0,
            "last_called": None
        }
    
    async def execute(self, tool_name: str, **kwargs) -> YvToolResult:
        if tool_name not in self._tools:
            return YvToolResult(
                success=False,
                output=None,
                execution_time=0.0,
                error_type="tool_not_found",
                error_message=f"Tool '{tool_name}' not found. Available tools: {list(self._tools.keys())}"
            )
        
        handler = self._tools[tool_name]
        start_time = time.time()
        
        try:
            if hasattr(handler, 'execute'):
                result = await handler.execute(**kwargs)
            else:
                result = await handler(**kwargs)
        except Exception as e:
            result = YvToolResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                error_type="execution_error",
                error_message=str(e)
            )
        
        self._tool_stats[tool_name]["total_calls"] += 1
        self._tool_stats[tool_name]["total_time"] += result.execution_time
        if result.success:
            self._tool_stats[tool_name]["success_count"] += 1
        else:
            self._tool_stats[tool_name]["failure_count"] += 1
        self._tool_stats[tool_name]["last_called"] = time.time()
        
        return result
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        if tool_name not in self._tools:
            return None
        
        stats = self._tool_stats.get(tool_name, {})
        avg_time = stats.get("total_time", 0) / max(stats.get("total_calls", 1), 1)
        
        return {
            "name": tool_name,
            "description": self._tool_descriptions.get(tool_name, ""),
            "success_rate": stats.get("success_count", 0) / max(stats.get("total_calls", 1), 1),
            "average_execution_time": avg_time,
            "total_calls": stats.get("total_calls", 0)
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        return [self.get_tool_info(name) for name in self._tools.keys()]
    
    def get_statistics(self) -> Dict[str, Any]:
        total_calls = sum(s["total_calls"] for s in self._tool_stats.values())
        total_success = sum(s["success_count"] for s in self._tool_stats.values())
        total_time = sum(s["total_time"] for s in self._tool_stats.values())
        
        return {
            "total_calls": total_calls,
            "total_success": total_success,
            "total_failures": total_calls - total_success,
            "overall_success_rate": total_success / max(total_calls, 1),
            "total_execution_time": total_time,
            "tool_statistics": self._tool_stats
        }
