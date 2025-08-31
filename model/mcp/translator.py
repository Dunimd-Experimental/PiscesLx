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

import os
import sys
import re
import json
import uuid
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.log import RIGHT, DEBUG, ERROR

try:
    from .server import mcp_server
    MCP_SDK_AVAILABLE = True
except ImportError:
    ERROR("Official MCP SDK not available. Please install: pip install 'mcp[cli]'")
    MCP_SDK_AVAILABLE = False

@dataclass
class AgentCall:
    """Represents a parsed agent call from model output"""
    tool_name: str
    parameters: Dict[str, Any]
    raw_match: str
    start_pos: int
    end_pos: int

class MCPTranslationLayer:
    """
    MCP Translation Layer with Official SDK Integration
    
    Core component for parsing model output containing <agent> tags and translating them
    to official MCP SDK calls. Bridges between model XML output and official MCP protocol.
    Located in model/mcp/ as part of the model's core interaction capability.
    """
    
    def __init__(self):
        self.mcp_server = mcp_server if MCP_SDK_AVAILABLE else None
        
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass
    
    def extract_agent_calls(self, text: str) -> List[AgentCall]:
        """
        Extract all <agent> tags from model output
        
        Args:
            text (str): Model output text containing agent tags
            
        Returns:
            List[AgentCall]: List of parsed agent calls
        """
        agent_calls = []
        
        # Regex pattern to match <agent><an>tool_name</an><ap1>param1</ap1>...</agent>
        agent_pattern = r'<agent><an>(.+?)</an>(.*?)</agent>'
        
        for match in re.finditer(agent_pattern, text, re.DOTALL):
            tool_name = match.group(1).strip()
            params_text = match.group(2).strip()
            
            # Extract parameters ap1, ap2, ap3, etc.
            parameters = {}
            param_pattern = r'<ap(\d+)>(.+?)</ap\1>'
            
            for param_match in re.finditer(param_pattern, params_text, re.DOTALL):
                param_index = param_match.group(1)
                param_value = param_match.group(2).strip()
                parameters[f"ap{param_index}"] = param_value
            
            agent_call = AgentCall(
                tool_name=tool_name,
                parameters=parameters,
                raw_match=match.group(0),
                start_pos=match.start(),
                end_pos=match.end()
            )
            
            agent_calls.append(agent_call)
            
        return agent_calls
    
    def remove_agent_tags(self, text: str, placeholder: str = "") -> str:
        """
        Remove all <agent> tags from text, optionally replacing with placeholder
        
        Args:
            text (str): Text containing agent tags
            placeholder (str): Text to replace agent tags with
            
        Returns:
            str: Text with agent tags removed/replaced
        """
        agent_pattern = r'<agent>.*?</agent>'
        return re.sub(agent_pattern, placeholder, text, flags=re.DOTALL)
    
    def replace_agent_tags_with_status(self, text: str, results: List[Dict[str, Any]]) -> str:
        """
        Replace agent tags with execution status and results
        
        Args:
            text (str): Original text with agent tags
            results (List[Dict]): Results from tool executions
            
        Returns:
            str: Text with agent tags replaced by results
        """
        agent_calls = self.extract_agent_calls(text)
        
        # Sort by position in reverse order to maintain correct positions
        agent_calls.sort(key=lambda x: x.start_pos, reverse=True)
        
        modified_text = text
        
        for i, call in enumerate(agent_calls):
            if i < len(results):
                result = results[i]
                
                if result.get('success', False):
                    # Format successful result
                    tool_result = result.get('result', {})
                    if isinstance(tool_result, dict):
                        # Create a user-friendly result display
                        result_text = self._format_tool_result(call.tool_name, tool_result)
                    else:
                        result_text = f"✅ {call.tool_name} 执行成功: {tool_result}"
                else:
                    # Format error result
                    error_msg = result.get('error_message', 'Unknown error')
                    result_text = f"❌ {call.tool_name} 执行失败: {error_msg}"
            else:
                result_text = f"⏳ {call.tool_name} 正在执行..."
            
            # Replace the agent tag with the result
            modified_text = (
                modified_text[:call.start_pos] + 
                result_text + 
                modified_text[call.end_pos:]
            )
        
        return modified_text
    
    def _format_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format tool execution result for user display"""
        
        if tool_name == "web_search":
            query = result.get('query', '')
            results = result.get('results', [])
            count = result.get('count', 0)
            
            formatted = f"🔍 网络搜索 \"{query}\" 完成，找到 {count} 个结果：\n"
            for i, item in enumerate(results[:3], 1):  # Show top 3 results
                title = item.get('title', '无标题')
                snippet = item.get('snippet', '无描述')
                formatted += f"{i}. {title}\n   {snippet}\n"
            
            if count > 3:
                formatted += f"   ... 还有 {count - 3} 个结果"
            
            return formatted
            
        elif tool_name == "calculator":
            expression = result.get('expression', '')
            calc_result = result.get('result', '')
            
            return f"🧮 计算结果: {expression} = {calc_result}"
            
        elif tool_name == "file_operations":
            operation = result.get('operation', '')
            filepath = result.get('filepath', '')
            
            if operation == "read":
                size = result.get('size', 0)
                return f"📖 读取文件 {filepath} 完成，大小: {size} 字符"
            elif operation == "write":
                bytes_written = result.get('bytes_written', 0)
                return f"📝 写入文件 {filepath} 完成，大小: {bytes_written} 字节"
            elif operation == "list":
                count = result.get('count', 0)
                return f"📁 列出目录 {filepath} 完成，找到 {count} 个文件/文件夹"
            
        elif tool_name == "image_analysis":
            analysis_type = result.get('analysis_type', '')
            image_path = result.get('image_path', '')
            
            if analysis_type == "description":
                description = result.get('description', '无描述')
                return f"🖼️ 图像分析完成: {description}"
            elif analysis_type == "objects":
                total_objects = result.get('total_objects', 0)
                return f"🎯 对象检测完成，发现 {total_objects} 个对象"
            elif analysis_type == "text":
                extracted_text = result.get('extracted_text', '')
                return f"📝 文字识别完成: {extracted_text}"
                
        elif tool_name == "text_processing":
            operation = result.get('operation', '')
            
            if operation == "summary":
                summary = result.get('summary', '')
                word_count = result.get('word_count', 0)
                return f"📝 文本摘要完成 ({word_count} 词): {summary}"
            elif operation == "translate":
                translated_text = result.get('translated_text', '')
                source_lang = result.get('source_language', '')
                target_lang = result.get('target_language', '')
                return f"🌐 翻译完成 ({source_lang} → {target_lang}): {translated_text}"
            elif operation == "sentiment":
                sentiment = result.get('sentiment', '')
                confidence = result.get('confidence', 0)
                return f"💭 情感分析完成: {sentiment} (置信度: {confidence:.2f})"
        
        # Default formatting
        return f"✅ {tool_name} 执行完成"
    
    async def execute_agent_calls(self, agent_calls: List[AgentCall], 
                                session_id: str = "default", 
                                agent_id: str = "pisces_model") -> List[Dict[str, Any]]:
        """
        Execute multiple agent calls via MCP server
        
        Args:
            agent_calls (List[AgentCall]): List of agent calls to execute
            session_id (str): Session identifier
            agent_id (str): Agent identifier
            
        Returns:
            List[Dict]: List of execution results
        """
        if not self.session:
            raise RuntimeError("MCPTranslationLayer must be used as async context manager")
        
        results = []
        
        for call in agent_calls:
            try:
                # Prepare MCP request
                mcp_request = {
                    "tool_name": call.tool_name,
                    "parameters": call.parameters,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "correlation_id": str(uuid.uuid4())
                }
                
                # Send request to MCP server
                async with self.session.post(
                    f"{self.mcp_server_url}/mcp/execute",
                    json=mcp_request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        results.append(result)
                        DEBUG(f"Tool {call.tool_name} executed successfully")
                    else:
                        error_text = await response.text()
                        ERROR(f"MCP server error {response.status}: {error_text}")
                        results.append({
                            "success": False,
                            "error_code": f"HTTP_{response.status}",
                            "error_message": f"Server returned {response.status}: {error_text}",
                            "tool_name": call.tool_name
                        })
                        
            except asyncio.TimeoutError:
                ERROR(f"Timeout executing tool: {call.tool_name}")
                results.append({
                    "success": False,
                    "error_code": "TIMEOUT",
                    "error_message": "Tool execution timed out",
                    "tool_name": call.tool_name
                })
                
            except Exception as e:
                ERROR(f"Error executing tool {call.tool_name}: {e}")
                results.append({
                    "success": False,
                    "error_code": "EXECUTION_ERROR",
                    "error_message": str(e),
                    "tool_name": call.tool_name
                })
        
        return results
    
    async def process_model_output(self, model_output: str, 
                                 session_id: str = "default",
                                 agent_id: str = "pisces_model") -> str:
        """
        Process model output with agent calls using official MCP protocol
        
        Args:
            model_output (str): Raw model output with agent tags
            session_id (str): Session identifier
            agent_id (str): Agent identifier
            
        Returns:
            str: Processed output with agent calls replaced by results
        """
        # Extract agent calls from XML tags
        agent_calls = self.extract_agent_calls(model_output)
        
        if not agent_calls:
            return model_output
        
        DEBUG(f"Found {len(agent_calls)} agent calls in model output")
        
        # Convert to official MCP requests and execute
        results = await self.execute_mcp_calls(agent_calls, session_id, agent_id)
        
        # Replace agent tags with results
        processed_output = self.replace_agent_tags_with_status(model_output, results)
        
        return processed_output
    
    async def execute_mcp_calls(self, agent_calls: List[AgentCall], 
                               session_id: str = "default", 
                               agent_id: str = "pisces_model") -> List[Dict[str, Any]]:
        """
        Execute agent calls using official MCP SDK
        """
        if not MCP_SDK_AVAILABLE or not self.mcp_server:
            return [{
                "success": False,
                "error_code": "SDK_UNAVAILABLE",
                "error_message": "Official MCP SDK not available",
                "tool_name": call.tool_name
            } for call in agent_calls]
        
        results = []
        
        for call in agent_calls:
            try:
                # Convert parameters from ap1/ap2 format to standard names
                converted_params = self._convert_parameters(call.tool_name, call.parameters)
                
                # Find the tool in the official SDK server
                tool_func = None
                for tool_name, tool_info in self.mcp_server.tools.items():
                    if tool_name == call.tool_name:
                        tool_func = tool_info.get('func')
                        break
                
                if not tool_func:
                    results.append({
                        "success": False,
                        "error_code": "TOOL_NOT_FOUND",
                        "error_message": f"Tool '{call.tool_name}' not found",
                        "tool_name": call.tool_name
                    })
                    continue
                
                # Execute tool directly using official SDK
                try:
                    if asyncio.iscoroutinefunction(tool_func):
                        tool_result = await tool_func(**converted_params)
                    else:
                        tool_result = tool_func(**converted_params)
                    
                    results.append({
                        "success": True,
                        "result": tool_result,
                        "tool_name": call.tool_name
                    })
                    
                except Exception as e:
                    results.append({
                        "success": False,
                        "error_code": "EXECUTION_ERROR",
                        "error_message": str(e),
                        "tool_name": call.tool_name
                    })
                
            except Exception as e:
                ERROR(f"Error executing tool {call.tool_name}: {e}")
                results.append({
                    "success": False,
                    "error_code": "EXECUTION_ERROR",
                    "error_message": str(e),
                    "tool_name": call.tool_name
                })
        
        return results
    
    def _convert_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert ap1, ap2... parameters to official MCP SDK format
        
        XML: <ap1>value1</ap1><ap2>value2</ap2>
        SDK: {"param1": "value1", "param2": "value2"}
        """
        converted = {}
        
        # Tool-specific parameter mappings for official SDK
        param_mappings = {
            "calculator": {
                "ap1": "expression"
            },
            "text_processing": {
                "ap1": "text",
                "ap2": "operation", 
                "ap3": "language",
                "ap4": "target_language"
            },
            "web_search": {
                "ap1": "query",
                "ap2": "limit",
                "ap3": "language"
            },
            "file_operations": {
                "ap1": "operation",
                "ap2": "filepath",
                "ap3": "content",
                "ap4": "encoding"
            },
            "image_analysis": {
                "ap1": "image_path",
                "ap2": "analysis_type",
                "ap3": "language"
            }
        }
        
        # Get mappings for this specific tool
        tool_mappings = param_mappings.get(tool_name, {})
        
        # Apply mapping
        for ap_key, value in parameters.items():
            if ap_key in tool_mappings:
                converted[tool_mappings[ap_key]] = value
            elif not ap_key.startswith("ap"):
                # Keep non-ap parameters as is
                converted[ap_key] = value
        
        return converted
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools from official MCP SDK"""
        if not MCP_SDK_AVAILABLE or not self.mcp_server:
            return []
        
        try:
            tools = []
            for tool_name, tool_info in self.mcp_server.tools.items():
                tools.append({
                    "name": tool_name,
                    "description": tool_info.get('description', 'No description'),
                    "category": "mcp_sdk_tool"
                })
            return tools
        except Exception as e:
            ERROR(f"Error getting available tools: {e}")
            return []

# Utility functions for standalone usage
async def process_text_with_mcp(text: str) -> str:
    """
    Standalone function to process text containing agent calls
    
    Args:
        text (str): Text with agent tags
        
    Returns:
        str: Processed text with results
    """
    async with MCPTranslationLayer() as translator:
        return await translator.process_model_output(text)

def extract_agent_calls_sync(text: str) -> List[AgentCall]:
    """
    Synchronous version of agent call extraction
    
    Args:
        text (str): Text containing agent tags
        
    Returns:
        List[AgentCall]: Extracted agent calls
    """
    translator = MCPTranslationLayer()
    return translator.extract_agent_calls(text)

def execute_tool_call_sync(xml_tag: str) -> Dict[str, Any]:
    """
    Synchronous execution of a single XML tool call
    
    Args:
        xml_tag (str): XML tag containing tool call
        
    Returns:
        Dict: Tool execution result
    """
    translator = MCPTranslationLayer()
    
    # Extract agent calls
    agent_calls = translator.extract_agent_calls(xml_tag)
    
    if not agent_calls:
        return {
            "success": False,
            "error_code": "NO_AGENT_CALLS",
            "error_message": "No valid agent calls found in XML"
        }
    
    call = agent_calls[0]  # Take first call
    
    try:
        # Convert parameters
        converted_params = translator._convert_parameters(call.tool_name, call.parameters)
        
        # Find the tool
        tool_func = None
        if translator.mcp_server:
            for tool_name, tool_info in translator.mcp_server.tools.items():
                if tool_name == call.tool_name:
                    tool_func = tool_info.get('func')
                    break
        
        if not tool_func:
            return {
                "success": False,
                "error_code": "TOOL_NOT_FOUND",
                "error_message": f"Tool '{call.tool_name}' not found",
                "tool_name": call.tool_name
            }
        
        # Execute tool (synchronously only)
        if asyncio.iscoroutinefunction(tool_func):
            return {
                "success": False,
                "error_code": "ASYNC_NOT_SUPPORTED",
                "error_message": f"Tool '{call.tool_name}' is async, use async execution",
                "tool_name": call.tool_name
            }
        
        tool_result = tool_func(**converted_params)
        
        return {
            "success": True,
            "result": tool_result,
            "tool_name": call.tool_name
        }
        
    except Exception as e:
        return {
            "success": False,
            "error_code": "EXECUTION_ERROR",
            "error_message": str(e),
            "tool_name": call.tool_name
        }