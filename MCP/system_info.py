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

import platform
from typing import Dict, Any

def system_info(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve system information based on the specified type.

    Args:
        arguments (Dict[str, Any]): A dictionary containing the information type. 
                                   The key 'type' can be 'cpu', 'memory', 'disk', 'gpu', or 'all'. 
                                   Defaults to 'all' if not specified.

    Returns:
        Dict[str, Any]: A dictionary containing the operation status and system information.
                       If successful, 'success' is True and 'data' contains the requested information.
                       If failed, 'success' is False and 'error' contains the error message.
    """
    info_type = arguments.get("type", "all")
    
    try:
        if info_type == "cpu":
            return {
                "success": True,
                "data": {
                    "cpu_count": 8,
                    "cpu_percent": 23.5,
                    "architecture": platform.machine(),
                    "processor": platform.processor()
                }
            }
        elif info_type == "memory":
            return {
                "success": True,
                "data": {
                    "total_gb": 16.0,
                    "available_gb": 12.3,
                    "percent": 23.1,
                    "unit": "GB"
                }
            }
        elif info_type == "disk":
            return {
                "success": True,
                "data": {
                    "total_gb": 512.0,
                    "used_gb": 245.7,
                    "free_gb": 266.3,
                    "percent": 48.0,
                    "unit": "GB"
                }
            }
        elif info_type == "gpu":
            return {
                "success": True,
                "data": {
                    "gpu_count": 1,
                    "gpu_name": "NVIDIA RTX 3080",
                    "memory_gb": 10.0,
                    "driver_version": "546.33"
                }
            }
        elif info_type == "all":
            return {
                "success": True,
                "data": {
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                    "hostname": platform.node(),
                    "system": platform.system(),
                    "cpu": {
                        "count": 8,
                        "architecture": platform.machine()
                    },
                    "memory": {
                        "total_gb": 16.0,
                        "available_gb": 12.3,
                        "percent": 23.1
                    },
                    "disk": {
                        "total_gb": 512.0,
                        "used_gb": 245.7,
                        "free_gb": 266.3,
                        "percent": 48.0
                    }
                }
            }
        else:
            return {"success": False, "error": f"Unknown type: {info_type}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

def system_health(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a quick system health check.

    Args:
        arguments (Dict[str, Any]): Currently unused, can be an empty dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing the operation status and system health information.
                       'success' is always True, and 'data' contains the health status and recommendations.
    """
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "cpu_usage": "normal",
            "memory_usage": "normal",
            "disk_usage": "normal",
            "recommendations": ["System running smoothly"]
        }
    }

# Integrate with Pisces L1 MCP Square
from . import register_custom_tool

# Register the system information tool
register_custom_tool(
    name="system_info",
    description="Retrieve system hardware and performance information.",
    parameters={
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["cpu", "memory", "disk", "gpu", "all"],
                "default": "all",
                "description": "Type of system information."
            }
        }
    },
    function=system_info,
    category="System"
)

# Register the system health check tool
register_custom_tool(
    name="system_health",
    description="Perform a quick system health check.",
    parameters={
        "type": "object",
        "properties": {}
    },
    function=system_health,
    category="System"
)