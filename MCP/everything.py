#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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
import json
import time
import random
import asyncio
import subprocess
from MCP import mcp
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Everything search tool for Windows file system
class EverythingSearch:
    """Everything search tool for Windows file system."""
    
    def __init__(self):
        self.es_path = None
        self._find_es_path()
    
    def _find_es_path(self):
        """Find Everything command line tool path."""
        possible_paths = [
            "C:\\Program Files\\Everything\\es.exe",
            "C:\\Program Files (x86)\\Everything\\es.exe",
            "C:\\Windows\\es.exe",
            "es.exe"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.es_path = path
                break
    
    def is_available(self) -> bool:
        """Check if Everything is available."""
        return self.es_path is not None
    
    def search_files(self, query: str, max_results: int = 100, sort: str = "name") -> List[str]:
        """Search files using Everything."""
        if not self.is_available():
            return []
        
        try:
            cmd = [self.es_path, query, f"-n{max_results}", f"-sort-{sort}"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                return [f for f in files if f.strip()]
        except Exception:
            pass
        
        return []
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed file information."""
        try:
            stat = os.stat(file_path)
            return {
                "path": file_path,
                "size": stat.st_size,
                "modified": os.path.getmtime(file_path),
                "is_directory": os.path.isdir(file_path),
                "exists": os.path.exists(file_path)
            }
        except Exception as e:
            return {"error": str(e)}

# Global instance
everything_search = EverythingSearch()

@mcp.tool()
def search_everything(query: str, max_results: int = 50, sort_by: str = "name") -> Dict[str, Any]:
    """Search files using Everything search engine on Windows."""
    try:
        if not everything_search.is_available():
            return {
                "success": False,
                "error": "Everything search tool is not available. Please install Everything with es.exe",
                "available_paths": [
                    "C:\\Program Files\\Everything\\es.exe",
                    "C:\\Program Files (x86)\\Everything\\es.exe"
                ]
            }
        
        files = everything_search.search_files(query, max_results, sort_by)
        
        return {
            "success": True,
            "query": query,
            "results": files,
            "count": len(files),
            "sort_by": sort_by,
            "max_results": max_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def get_file_details(file_path: str) -> Dict[str, Any]:
    """Get detailed information about a specific file or directory."""
    try:
        info = everything_search.get_file_info(file_path)
        
        if "error" in info:
            return {
                "success": False,
                "error": info["error"]
            }
        
        return {
            "success": True,
            "file_info": info
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def list_drives() -> Dict[str, Any]:
    """List all available drive letters on Windows."""
    try:
        import string
        import win32api
        
        drives = win32api.GetLogicalDriveStrings().split('\x00')[:-1]
        
        drive_info = []
        for drive in drives:
            drive_letter = drive[0]
            try:
                total, used, free = win32api.GetDiskFreeSpaceEx(drive)
                drive_info.append({
                    "drive": drive_letter,
                    "path": drive,
                    "total_bytes": total,
                    "used_bytes": used,
                    "free_bytes": free,
                    "total_gb": round(total / (1024**3), 2),
                    "used_gb": round(used / (1024**3), 2),
                    "free_gb": round(free / (1024**3), 2)
                })
            except:
                drive_info.append({
                    "drive": drive_letter,
                    "path": drive,
                    "error": "Unable to access drive"
                })
        
        return {
            "success": True,
            "drives": drive_info,
            "count": len(drive_info)
        }
        
    except ImportError:
        return {
            "success": False,
            "error": "win32api not available. This function requires pywin32."
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def get_recent_files(directory: str = None, limit: int = 20) -> Dict[str, Any]:
    """Get recently modified files in a directory."""
    try:
        if directory is None:
            directory = os.getcwd()
        
        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}"
            }
        
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                try:
                    stat = os.stat(filepath)
                    files.append({
                        "path": filepath,
                        "modified": stat.st_mtime,
                        "size": stat.st_size,
                        "extension": os.path.splitext(filename)[1],
                        "directory": os.path.dirname(filepath)
                    })
                except (OSError, PermissionError):
                    continue
        
        # Sort by modification time, most recent first
        files.sort(key=lambda x: x["modified"], reverse=True)
        recent_files = files[:limit]
        
        # Convert timestamps to readable format
        for file in recent_files:
            file["modified_readable"] = datetime.fromtimestamp(file["modified"]).strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "success": True,
            "directory": directory,
            "files": recent_files,
            "count": len(recent_files)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def find_large_files(directory: str = None, min_size_mb: int = 100, limit: int = 50) -> Dict[str, Any]:
    """Find large files in a directory."""
    try:
        if directory is None:
            directory = os.getcwd()
        
        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}"
            }
        
        min_size_bytes = min_size_mb * 1024 * 1024
        large_files = []
        
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                try:
                    stat = os.stat(filepath)
                    if stat.st_size >= min_size_bytes:
                        large_files.append({
                            "path": filepath,
                            "size_bytes": stat.st_size,
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "modified": stat.st_mtime,
                            "extension": os.path.splitext(filename)[1]
                        })
                except (OSError, PermissionError):
                    continue
        
        # Sort by size, largest first
        large_files.sort(key=lambda x: x["size_bytes"], reverse=True)
        large_files = large_files[:limit]
        
        for file in large_files:
            file["modified_readable"] = datetime.fromtimestamp(file["modified"]).strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "success": True,
            "directory": directory,
            "files": large_files,
            "count": len(large_files),
            "min_size_mb": min_size_mb
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def search_by_extension(extension: str, directory: str = None, limit: int = 100) -> Dict[str, Any]:
    """Search files by extension in a directory."""
    try:
        if directory is None:
            directory = os.getcwd()
        
        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}"
            }
        
        # Ensure extension starts with a dot
        if not extension.startswith('.'):
            extension = '.' + extension
        
        files = []
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                if filename.lower().endswith(extension.lower()):
                    filepath = os.path.join(root, filename)
                    try:
                        stat = os.stat(filepath)
                        files.append({
                            "path": filepath,
                            "size": stat.st_size,
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "modified": stat.st_mtime,
                            "directory": os.path.dirname(filepath)
                        })
                    except (OSError, PermissionError):
                        continue
        
        # Sort by modification time, most recent first
        files.sort(key=lambda x: x["modified"], reverse=True)
        files = files[:limit]
        
        for file in files:
            file["modified_readable"] = datetime.fromtimestamp(file["modified"]).strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            "success": True,
            "directory": directory,
            "extension": extension,
            "files": files,
            "count": len(files)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }