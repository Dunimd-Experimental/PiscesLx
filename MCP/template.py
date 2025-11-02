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
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.mcp import PiscesLxCoreMCPPlaza

# Create mcp instance for tool registration
mcp = PiscesLxCoreMCPPlaza()

@mcp.tool()
def your_tool_name(param1: str, param2: int = 0) -> Dict[str, Any]:
    """
    Process input parameters and return a structured result.
    
    Args:
        param1 (str): The primary input string to process.
        param2 (int, optional): An integer value for additional processing. Defaults to 0.
        
    Returns:
        Dict[str, Any]: A dictionary containing the processing result with success status and data.
    """
    try:
        # Process the input parameters
        result = {
            "success": True,
            "data": f"Processed: {param1} with {param2}",
            "timestamp": "2024-..."
        }
        
        return result
        
    except Exception as e:
        # Handle any exceptions that occur during processing
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def helper_function(arg1: str, arg2: int = 42) -> Dict[str, Any]:
    """
    Process helper function arguments and return a structured result.

    Args:
        arg1 (str): The first argument for processing.
        arg2 (int, optional): The second argument for processing. Defaults to 42.

    Returns:
        Dict[str, Any]: A dictionary containing the processed result and success status.
    """
    # Return the processed result
    return {
        "helper_result": f"Helper processed {arg1} and {arg2}",
        "success": True
    }
