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

from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Your Tool Name")

def get_tool_definition() -> 'MCPTool':
    """
    Generate the official MCP tool definition for auto-discovery.

    Returns:
        MCPTool: An instance of MCPTool containing the tool's metadata and input schema.
    """
    from MCP import MCPTool
    return MCPTool(
        name="your_tool_name",  # Tool name used by the model to call
        description="Brief description of what this tool does",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Description of parameter 1"
                },
                "param2": {
                    "type": "number", 
                    "description": "Description of parameter 2"
                }
            },
            "required": ["param1"]  # List of required parameters
        }
    )

def execute_tool(arguments: Dict[str, Any]) -> Any:
    """
    Execute the tool with the provided arguments.

    Args:
        arguments (Dict[str, Any]): A dictionary containing the input parameters for the tool.

    Returns:
        Any: A dictionary representing the execution result, including success status, data, or error information.
    """
    try:
        param1 = arguments.get("param1")
        param2 = arguments.get("param2", 0)
        
        # Implement your tool logic here
        result = {
            "success": True,
            "data": f"Processed: {param1} with {param2}",
            "timestamp": "2024-..."
        }
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def helper_function(arg1: str, arg2: int = 42) -> Dict[str, Any]:
    """
    An optional helper tool that is also auto-discovered.

    Args:
        arg1 (str): The first argument for the helper function.
        arg2 (int, optional): The second argument for the helper function. Defaults to 42.

    Returns:
        Dict[str, Any]: A dictionary containing the helper result and success status.
    """
    return {
        "helper_result": f"Helper processed {arg1} and {arg2}",
        "success": True
    }

if __name__ == "__main__":
    # Test the tool's basic functionality
    print("Testing tool...")
    definition = get_tool_definition()
    print(f"Tool: {definition.name}")
    print(f"Description: {definition.description}")
    
    # Test the tool's execution
    test_result = execute_tool({"param1": "test", "param2": 123})
    print(f"Result: {test_result}")