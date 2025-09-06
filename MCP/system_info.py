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

import socket
import platform
from MCP import mcp
from typing import Dict, Any

@mcp.tool()
def system_info() -> Dict[str, Any]:
    """Get comprehensive system information including CPU, memory, disk, and network."""
    try:
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory information
        memory = psutil.virtual_memory()
        
        # Disk information
        disk_usage = psutil.disk_usage('/')
        
        # Network information
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        # Platform information
        system = platform.system()
        release = platform.release()
        version = platform.version()
        machine = platform.machine()
        processor = platform.processor()
        
        return {
            "success": True,
            "cpu": {
                "usage_percent": cpu_percent,
                "cores": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None,
                "max_frequency_mhz": cpu_freq.max if cpu_freq else None
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "usage_percent": memory.percent
            },
            "disk": {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "used_gb": round(disk_usage.used / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "usage_percent": round((disk_usage.used / disk_usage.total) * 100, 2)
            },
            "network": {
                "hostname": hostname,
                "ip_address": ip_address
            },
            "platform": {
                "system": system,
                "release": release,
                "version": version,
                "machine": machine,
                "processor": processor
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def system_health() -> Dict[str, Any]:
    """Get system health status with warnings and recommendations."""
    try:
        # Get system info
        info = system_info()
        if not info["success"]:
            return info
        
        data = info
        warnings = []
        recommendations = []
        
        # CPU health check
        cpu_usage = data["cpu"]["usage_percent"]
        if cpu_usage > 80:
            warnings.append(f"High CPU usage: {cpu_usage}%")
            recommendations.append("Consider closing unnecessary applications")
        elif cpu_usage > 90:
            warnings.append(f"Critical CPU usage: {cpu_usage}%")
            recommendations.append("Immediate action required: check running processes")
        
        # Memory health check
        memory_usage = data["memory"]["usage_percent"]
        if memory_usage > 80:
            warnings.append(f"High memory usage: {memory_usage}%")
            recommendations.append("Consider restarting applications or system")
        elif memory_usage > 90:
            warnings.append(f"Critical memory usage: {memory_usage}%")
            recommendations.append("Immediate action: free up memory")
        
        # Disk health check
        disk_usage = data["disk"]["usage_percent"]
        if disk_usage > 85:
            warnings.append(f"High disk usage: {disk_usage}%")
            recommendations.append("Consider cleaning up disk space")
        elif disk_usage > 95:
            warnings.append(f"Critical disk usage: {disk_usage}%")
            recommendations.append("Urgent: free up disk space immediately")
        
        # Overall health score
        health_score = 100
        if warnings:
            health_score -= len(warnings) * 10
        if cpu_usage > 70:
            health_score -= 10
        if memory_usage > 70:
            health_score -= 10
        if disk_usage > 70:
            health_score -= 10
        
        health_score = max(0, min(100, health_score))
        
        status = "healthy"
        if health_score < 50:
            status = "critical"
        elif health_score < 70:
            status = "warning"
        elif health_score < 90:
            status = "good"
        
        return {
            "success": True,
            "health": {
                "score": health_score,
                "status": status,
                "warnings": warnings,
                "recommendations": recommendations
            },
            "system": {
                "cpu_usage": data["cpu"]["usage_percent"],
                "memory_usage": data["memory"]["usage_percent"],
                "disk_usage": data["disk"]["usage_percent"]
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }