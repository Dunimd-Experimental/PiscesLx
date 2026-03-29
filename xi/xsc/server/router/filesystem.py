#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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

"""
Filesystem routes for file browsing and management.
"""

import os
import platform
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException

from ...core.dc import XiLogger


def setup_filesystem_routes(app: FastAPI, root_dir: Path, logger: XiLogger, request_count: Dict[str, int]) -> None:
    """
    Setup filesystem routes.
    
    Args:
        app: FastAPI application
        root_dir: Working directory
        logger: XiLogger instance
        request_count: Mutable request count reference
    """
    is_windows = platform.system() == "Windows"
    
    @app.get("/v1/fs/drives")
    async def get_drives():
        request_count["value"] = request_count.get("value", 0) + 1
        drives = []
        
        if is_windows:
            import string
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    try:
                        usage = shutil.disk_usage(drive)
                        drives.append({
                            "name": f"Local Disk ({letter}:)",
                            "path": drive,
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free,
                        })
                    except Exception:
                        drives.append({
                            "name": f"Local Disk ({letter}:)",
                            "path": drive,
                            "total": 0,
                            "used": 0,
                            "free": 0,
                        })
        else:
            common_mounts = ["/", "/home", "/mnt", "/media", "/opt", "/var"]
            for mount in common_mounts:
                if os.path.exists(mount):
                    try:
                        usage = shutil.disk_usage(mount)
                        drives.append({
                            "name": mount,
                            "path": mount,
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free,
                        })
                    except Exception:
                        pass
        
        return {
            "is_windows": is_windows,
            "drives": drives,
        }
    
    @app.get("/v1/fs/directory")
    async def get_directory(path: str = "/"):
        request_count["value"] = request_count.get("value", 0) + 1
        
        try:
            if is_windows:
                if path == "/" or path == "":
                    target_path = str(root_dir)
                elif len(path) == 2 and path[1] == ":":
                    target_path = path + "\\"
                elif path.startswith("/") and len(path) > 2 and path[2] == "/":
                    target_path = path[1:2] + ":" + path[2:]
                else:
                    target_path = path
            else:
                if path == "/" or path == "":
                    target_path = "/"
                else:
                    target_path = path
            
            target_path = os.path.normpath(target_path)
            
            if not os.path.exists(target_path):
                raise HTTPException(status_code=404, detail="Directory not found")
            
            if not os.path.isdir(target_path):
                raise HTTPException(status_code=400, detail="Not a directory")
            
            def scan_directory(dir_path: str):
                items = []
                try:
                    entries = os.listdir(dir_path)
                except PermissionError:
                    return items
                except Exception:
                    return items
                
                for item in entries:
                    try:
                        item_path = os.path.join(dir_path, item)
                        stat = os.stat(item_path)
                        items.append({
                            "name": item,
                            "path": item_path.replace("\\", "/"),
                            "is_dir": os.path.isdir(item_path),
                            "size": stat.st_size if os.path.isfile(item_path) else 0,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        })
                    except PermissionError:
                        continue
                    except Exception:
                        continue
                
                items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
                return items
            
            items = await asyncio.get_event_loop().run_in_executor(
                None, scan_directory, target_path
            )
            
            def get_disk_usage(dir_path: str):
                try:
                    usage = shutil.disk_usage(dir_path)
                    return {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                    }
                except Exception:
                    return None
            
            disk_info = await asyncio.get_event_loop().run_in_executor(
                None, get_disk_usage, target_path
            )
            
            return {
                "path": target_path.replace("\\", "/"),
                "items": items,
                "disk": disk_info,
                "is_windows": is_windows,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/v1/fs/folder")
    async def create_folder(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        path = request.get("path", "")
        if not path:
            raise HTTPException(status_code=400, detail="Path is required")
        
        def do_create():
            try:
                os.makedirs(path, exist_ok=True)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return await asyncio.get_event_loop().run_in_executor(None, do_create)
    
    @app.post("/v1/fs/file")
    async def create_file(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        path = request.get("path", "")
        content = request.get("content", "")
        
        if not path:
            raise HTTPException(status_code=400, detail="Path is required")
        
        def do_create():
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return await asyncio.get_event_loop().run_in_executor(None, do_create)
    
    @app.delete("/v1/fs/item")
    async def delete_item(path: str = ""):
        request_count["value"] = request_count.get("value", 0) + 1
        if not path:
            raise HTTPException(status_code=400, detail="Path is required")
        
        def do_delete():
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return await asyncio.get_event_loop().run_in_executor(None, do_delete)
    
    @app.post("/v1/fs/rename")
    async def rename_item(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        old_path = request.get("old_path", "")
        new_path = request.get("new_path", "")
        
        if not old_path or not new_path:
            raise HTTPException(status_code=400, detail="Both old_path and new_path are required")
        
        def do_rename():
            try:
                os.rename(old_path, new_path)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return await asyncio.get_event_loop().run_in_executor(None, do_rename)
    
    @app.post("/v1/fs/copy")
    async def copy_item(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        source = request.get("source", "")
        destination = request.get("destination", "")
        
        if not source or not destination:
            raise HTTPException(status_code=400, detail="Both source and destination are required")
        
        def do_copy():
            try:
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return await asyncio.get_event_loop().run_in_executor(None, do_copy)
    
    @app.post("/v1/fs/move")
    async def move_item(request: dict):
        request_count["value"] = request_count.get("value", 0) + 1
        source = request.get("source", "")
        destination = request.get("destination", "")
        
        if not source or not destination:
            raise HTTPException(status_code=400, detail="Both source and destination are required")
        
        def do_move():
            try:
                shutil.move(source, destination)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return await asyncio.get_event_loop().run_in_executor(None, do_move)
