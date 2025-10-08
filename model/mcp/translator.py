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
import re
import sys
import json
import uuid
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils import RIGHT, DEBUG, ERROR

# Config-driven MCP: load tools from configuration, no import-time discovery
from tools import read_config
import importlib

_mcp_available = False
_tools_cache: Dict[str, Any] = {}

def _load_tools_from_config() -> Dict[str, Any]:
    """
    Load enabled tools from the configuration.

    Retrieves the configuration using `read_config()` and filters out tools 
    that are not enabled. Only tools that are represented as dictionaries 
    and have the `enabled` key set to `True` (default) are included.

    Returns:
        Dict[str, Any]: A dictionary containing the names and configurations of enabled tools.
    """
    cfg = read_config()
    tools = cfg.get("tools", {})
    return {k: v for k, v in tools.items() if isinstance(v, dict) and v.get("enabled", True)}

@dataclass
class AgentCall:
    """
    Represents a parsed agent call from model output.

    Attributes:
        tool_name (str): Name of the tool to be called.
        parameters (Dict[str, Any]): Parameters for the tool call.
        raw_match (str): Raw matched text from the model output.
        start_pos (int): Start position of the match in the original text.
        end_pos (int): End position of the match in the original text.
    """
    tool_name: str
    parameters: Dict[str, Any]
    raw_match: str
    start_pos: int
    end_pos: int

class MCPTranslationLayer:
    """
    MCP Translation Layer with Official SDK Integration.

    Core component for parsing model output containing <agent> tags and translating them
    to official MCP SDK calls. Bridges between model XML output and official MCP protocol.
    Located in model/mcp/ as part of the model's core interaction capability.
    """
    
    def __init__(self):
        """
        Initialize the MCPTranslationLayer instance.

        Loads the tools from the configuration once without blocking and sets the availability status.
        If an error occurs during loading, logs the error and marks MCP as unavailable.
        """
        global _mcp_available, _tools_cache
        self._ready = False
        try:
            _tools_cache = _load_tools_from_config()
            _mcp_available = len(_tools_cache) > 0
        except Exception as e:
            ERROR(f"Failed to load MCP tools config: {e}")
            _mcp_available = False
        
    def _wait_for_ready(self, timeout: float = 0.0) -> bool:
        """
        Perform a non-blocking readiness check.

        Checks if the instance is ready by verifying that MCP is available and the tools cache is non-empty.
        Updates the readiness status if it hasn't been set yet.

        Args:
            timeout (float, optional): Timeout value (currently unused). Defaults to 0.0.

        Returns:
            bool: True if the instance is ready, False otherwise.
        """
        if not self._ready:
            self._ready = _mcp_available and bool(_tools_cache)
        return self._ready
        
    async def __aenter__(self):
        """
        Enter the asynchronous context manager.

        Returns:
            MCPTranslationLayer: The current instance of MCPTranslationLayer.
        """
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the asynchronous context manager.

        Args:
            exc_type: Type of the exception.
            exc_val: Value of the exception.
            exc_tb: Traceback of the exception.
        """
        pass
    
    def extract_agent_calls(self, text: str) -> List[AgentCall]:
        """
        Extract all <agent> tags from model output.

        Uses regular expressions to find all <agent> tags in the input text, 
        parses the tool name and parameters, and creates AgentCall objects.

        Args:
            text (str): Model output text containing agent tags.

        Returns:
            List[AgentCall]: List of parsed agent calls.
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
        Remove all <agent> tags from text, optionally replacing with placeholder.

        Args:
            text (str): Text containing agent tags.
            placeholder (str, optional): Text to replace agent tags with. Defaults to "".

        Returns:
            str: Text with agent tags removed/replaced.
        """
        agent_pattern = r'<agent>.*?</agent>'
        return re.sub(agent_pattern, placeholder, text, flags=re.DOTALL)
    
    def replace_agent_tags_with_status(self, text: str, results: List[Dict[str, Any]]) -> str:
        """
        Replace agent tags with execution status and results.

        Extracts agent calls from the input text, sorts them in reverse order of their positions,
        and replaces each agent tag with the corresponding execution result or status.

        Args:
            text (str): Original text with agent tags.
            results (List[Dict]): Results from tool executions.

        Returns:
            str: Text with agent tags replaced by results.
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
                        result_text = f"✅\t{call.tool_name} executed successfully: {tool_result}"
                else:
                    # Format error result
                    error_msg = result.get('error_message', 'Unknown error')
                    result_text = f"❌\t{call.tool_name} execution failed: {error_msg}"
            else:
                result_text = f"🟧\t{call.tool_name} is being executed..."
            
            # Replace the agent tag with the result
            modified_text = (
                modified_text[:call.start_pos] + 
                result_text + 
                modified_text[call.end_pos:]
            )
        
        return modified_text
    
    def _format_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """
        Format tool execution result for user display.

        Generates a user-friendly string representation of the tool execution result 
        based on the tool name and result data.

        Args:
            tool_name (str): Name of the tool.
            result (Dict[str, Any]): Tool execution result.

        Returns:
            str: Formatted result string.
        """
        if tool_name == "web_search":
            query = result.get('query', '')
            results = result.get('results', [])
            count = result.get('count', 0)
            
            formatted = f"🟧\tWeb search for \"{query}\" completed, found {count} results:\n"
            for i, item in enumerate(results[:3], 1):  # Show top 3 results
                title = item.get('title', 'No title')
                snippet = item.get('snippet', 'No description')
                formatted += f"{i}. {title}\n   {snippet}\n"
            
            if count > 3:
                formatted += f"   ... {count - 3} more results"
            
            return formatted
            
        elif tool_name == "calculator":
            expression = result.get('expression', '')
            calc_result = result.get('result', '')
            
            return f"🟧\tCalculation result: {expression} = {calc_result}"
            
        elif tool_name == "file_operations":
            operation = result.get('operation', '')
            filepath = result.get('filepath', '')
            
            if operation == "read":
                size = result.get('size', 0)
                return f"🟧\tFile read completed: {filepath}, size: {size} characters"
            elif operation == "write":
                bytes_written = result.get('bytes_written', 0)
                return f"🟧\tFile write completed: {filepath}, size: {bytes_written} bytes"
            elif operation == "list":
                count = result.get('count', 0)
                return f"🟧\tDirectory listing completed: {filepath}, found {count} files/folders"
            
        elif tool_name == "image_analysis":
            analysis_type = result.get('analysis_type', '')
            image_path = result.get('image_path', '')
            
            if analysis_type == "description":
                description = result.get('description', 'No description')
                return f"🟧\tImage analysis completed: {description}"
            elif analysis_type == "objects":
                total_objects = result.get('total_objects', 0)
                return f"🟧\tObject detection completed, found {total_objects} objects"
            elif analysis_type == "text":
                extracted_text = result.get('extracted_text', '')
                return f"🟧\tText recognition completed: {extracted_text}"
                
        elif tool_name == "text_processing":
            operation = result.get('operation', '')
            
            if operation == "summary":
                summary = result.get('summary', '')
                word_count = result.get('word_count', 0)
                return f"🟧\tText summarization completed ({word_count} words): {summary}"
            elif operation == "translate":
                translated_text = result.get('translated_text', '')
                source_lang = result.get('source_language', '')
                target_lang = result.get('target_language', '')
                return f"🟧\tTranslation completed ({source_lang} → {target_lang}): {translated_text}"
            elif operation == "sentiment":
                sentiment = result.get('sentiment', '')
                confidence = result.get('confidence', 0)
                return f"🟧\tSentiment analysis completed: {sentiment} (confidence: {confidence:.2f})"
        
        # Default formatting
        return f"✅\t{tool_name} execution completed"
    
    async def execute_agent_calls(self, agent_calls: List[AgentCall], 
                                session_id: str = "default", 
                                agent_id: str = "pisces_model") -> List[Dict[str, Any]]:
        """
        Execute multiple agent calls using config-registered local tools.

        Iterates through the list of agent calls, converts parameters, loads the corresponding tool function,
        and executes it. Collects and returns the execution results.

        Args:
            agent_calls (List[AgentCall]): List of agent calls to execute.
            session_id (str, optional): Session identifier. Defaults to "default".
            agent_id (str, optional): Agent identifier. Defaults to "pisces_model".

        Returns:
            List[Dict[str, Any]]: List of execution results for each agent call.
        """
        results = []
        
        for call in agent_calls:
            try:
                converted_params = self._convert_parameters(call.tool_name, call.parameters)
                
                tool_meta = _tools_cache.get(call.tool_name)
                if not tool_meta:
                    results.append({
                        "success": False,
                        "error_code": "TOOL_NOT_FOUND",
                        "error_message": f"Tool '{call.tool_name}' not found in config",
                        "tool_name": call.tool_name
                    })
                    continue
                
                module_name = tool_meta.get("module")
                func_name = tool_meta.get("func")
                if not module_name or not func_name:
                    results.append({
                        "success": False,
                        "error_code": "INVALID_CONFIG",
                        "error_message": f"Tool '{call.tool_name}' missing module/func in config",
                        "tool_name": call.tool_name
                    })
                    continue
                
                mod = importlib.import_module(module_name)
                tool_func = getattr(mod, func_name, None)
                if tool_func is None:
                    results.append({
                        "success": False,
                        "error_code": "FUNC_NOT_FOUND",
                        "error_message": f"Function '{func_name}' not found in module '{module_name}'",
                        "tool_name": call.tool_name
                    })
                    continue
                
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
        
        return results
    
    async def process_model_output(self, model_output: str, 
                                 session_id: str = "default",
                                 agent_id: str = "pisces_model") -> str:
        """
        Process model output with agent calls using official MCP protocol.

        Extracts agent calls from the model output, executes them, and replaces the agent tags 
        with the execution results.

        Args:
            model_output (str): Raw model output with agent tags.
            session_id (str, optional): Session identifier. Defaults to "default".
            agent_id (str, optional): Agent identifier. Defaults to "pisces_model".

        Returns:
            str: Processed output with agent calls replaced by results.
        """
        # Extract agent calls from XML tags
        agent_calls = self.extract_agent_calls(model_output)
        
        if not agent_calls:
            return model_output
        
        DEBUG(f"Found {len(agent_calls)} agent calls in model output")
        
        # Convert to official MCP requests and execute
        results = await self.execute_agent_calls(agent_calls, session_id, agent_id)
        
        # Replace agent tags with results
        processed_output = self.replace_agent_tags_with_status(model_output, results)
        
        return processed_output
    
    async def execute_mcp_calls(self, agent_calls: List[AgentCall], 
                               session_id: str = "default", 
                               agent_id: str = "pisces_model") -> List[Dict[str, Any]]:
        """
        Execute agent calls using config-registered local tools (non-blocking).

        Checks if the instance is ready before executing the agent calls. If not ready, 
        returns a list of error results for each call.

        Args:
            agent_calls (List[AgentCall]): List of agent calls to execute.
            session_id (str, optional): Session identifier. Defaults to "default".
            agent_id (str, optional): Agent identifier. Defaults to "pisces_model".

        Returns:
            List[Dict[str, Any]]: List of execution results for each agent call.
        """
        if not self._wait_for_ready(timeout=0.0):
            return [{
                "success": False,
                "error_code": "MCP_NOT_READY",
                "error_message": "No MCP tools configured",
                "tool_name": call.tool_name
            } for call in agent_calls]
        return await self.execute_agent_calls(agent_calls, session_id, agent_id)
    
    def _convert_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert ap1, ap2... parameters to official MCP SDK format.

        Maps XML-style parameters (ap1, ap2, etc.) to the official MCP SDK format 
        based on tool-specific mappings. Non-ap parameters are kept as-is.

        Args:
            tool_name (str): Name of the tool.
            parameters (Dict[str, Any]): Original parameters in XML format.

        Returns:
            Dict[str, Any]: Converted parameters in official MCP SDK format.
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
        """
        Get list of available tools from config.

        Retrieves the list of available tools from the tools cache and formats them 
        into a list of dictionaries containing tool names, descriptions, and categories.

        Returns:
            List[Dict[str, Any]]: List of available tools with their metadata.
        """
        try:
            return [{
                "name": name,
                "description": meta.get('description', 'No description'),
                "category": "config_tool"
            } for name, meta in _tools_cache.items()]
        except Exception as e:
            ERROR(f"Error getting available tools: {e}")
            return []

# Utility functions for standalone usage
async def process_text_with_mcp(text: str) -> str:
    """
    Standalone function to process text containing agent calls.

    Creates an instance of MCPTranslationLayer using an async context manager,
    and processes the input text containing agent calls.

    Args:
        text (str): Text with agent tags.

    Returns:
        str: Processed text with results.
    """
    async with MCPTranslationLayer() as translator:
        return await translator.process_model_output(text)

def extract_agent_calls_sync(text: str) -> List[AgentCall]:
    """
    Synchronous version of agent call extraction.

    Creates an instance of MCPTranslationLayer and extracts agent calls from the input text.

    Args:
        text (str): Text containing agent tags.

    Returns:
        List[AgentCall]: Extracted agent calls.
    """
    translator = MCPTranslationLayer()
    return translator.extract_agent_calls(text)

def execute_tool_call_sync(xml_tag: str) -> Dict[str, Any]:
    """
    Synchronous execution of a single XML tool call.

    Extracts the first agent call from the input XML tag, converts its parameters,
    finds the corresponding tool function, and executes it synchronously.

    Args:
        xml_tag (str): XML tag containing tool call.

    Returns:
        Dict: Tool execution result.
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