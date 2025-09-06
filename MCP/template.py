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
from typing import Dict, Any

@mcp.tool()
def your_tool_name(param1: str, param2: int = 0) -> Dict[str, Any]:
    """
    Brief description of what this tool does.
    
    Args:
        param1 (str): Description of parameter 1
        param2 (int, optional): Description of parameter 2. Defaults to 0.
        
    Returns:
        Dict[str, Any]: A dictionary containing the execution result, including success status and data.
    """
    try:
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