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
Explorer WebSocket handler for file system operations.
"""

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Any, Optional, List
from fastapi import WebSocket, WebSocketDisconnect

from ...core.dc import XiLogger


class PiscesL1ExplorerWebSocket:
    """
    WebSocket handler for file system explorer with real-time operations.
    """

    def __init__(self, root_dir: str, logger: XiLogger):
        self.root_dir = Path(root_dir)
        self.logger = logger
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(
            f"Explorer WebSocket connected. Total clients: {len(self.active_connections)}",
            event="xi.explorer_ws.connect"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        self.logger.info(
            f"Explorer WebSocket disconnected. Total clients: {len(self.active_connections)}",
            event="xi.explorer_ws.disconnect"
        )

    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        msg_type = data.get("type")

        handlers = {
            "browse": self._handle_browse,
            "create_folder": self._handle_create_folder,
            "create_file": self._handle_create_file,
            "delete": self._handle_delete,
            "rename": self._handle_rename,
            "copy": self._handle_copy,
            "move": self._handle_move,
        }

        handler = handlers.get(msg_type)
        if handler:
            await handler(websocket, data)
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to root directory."""
        if path.startswith("/"):
            path = path[1:]
        return self.root_dir / path

    def _get_file_info(self, path: Path) -> Dict[str, Any]:
        """Get file/directory information."""
        try:
            stat = path.stat()
            return {
                "name": path.name,
                "path": str(path.relative_to(self.root_dir)).replace("\\", "/"),
                "type": "directory" if path.is_dir() else "file",
                "size": stat.st_size if path.is_file() else None,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            }
        except Exception as e:
            return {
                "name": path.name,
                "path": str(path.relative_to(self.root_dir)).replace("\\", "/"),
                "type": "unknown",
                "error": str(e),
            }

    async def _handle_browse(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        path_str = data.get("path", ".")
        
        try:
            target_path = self._resolve_path(path_str)
            
            # Security check: ensure path is within root_dir
            try:
                target_path.relative_to(self.root_dir)
            except ValueError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Access denied: path outside root directory"
                })
                return

            if not target_path.exists():
                await websocket.send_json({
                    "type": "error",
                    "message": f"Path not found: {path_str}"
                })
                return

            items: List[Dict[str, Any]] = []
            
            if target_path.is_dir():
                for item in sorted(target_path.iterdir()):
                    items.append(self._get_file_info(item))
            
            # Get disk info
            try:
                disk_usage = shutil.disk_usage(self.root_dir)
                disk_info = {
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "is_windows": os.name == "nt",
                }
            except Exception:
                disk_info = None

            await websocket.send_json({
                "type": "directory",
                "path": path_str,
                "items": items,
                "disk": disk_info,
            })

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    async def _handle_create_folder(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        path_str = data.get("path")
        
        if not path_str:
            await websocket.send_json({
                "type": "error",
                "message": "path is required"
            })
            return

        try:
            target_path = self._resolve_path(path_str)
            
            # Security check
            try:
                target_path.relative_to(self.root_dir)
            except ValueError:
                await websocket.send_json({
                    "type": "operation_result",
                    "success": False,
                    "error": "Access denied: path outside root directory"
                })
                return

            target_path.mkdir(parents=True, exist_ok=True)
            
            await websocket.send_json({
                "type": "operation_result",
                "success": True,
                "operation": "create_folder",
                "path": path_str,
            })

        except Exception as e:
            await websocket.send_json({
                "type": "operation_result",
                "success": False,
                "error": str(e)
            })

    async def _handle_create_file(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        path_str = data.get("path")
        content = data.get("content", "")
        
        if not path_str:
            await websocket.send_json({
                "type": "error",
                "message": "path is required"
            })
            return

        try:
            target_path = self._resolve_path(path_str)
            
            # Security check
            try:
                target_path.relative_to(self.root_dir)
            except ValueError:
                await websocket.send_json({
                    "type": "operation_result",
                    "success": False,
                    "error": "Access denied: path outside root directory"
                })
                return

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
            
            await websocket.send_json({
                "type": "operation_result",
                "success": True,
                "operation": "create_file",
                "path": path_str,
            })

        except Exception as e:
            await websocket.send_json({
                "type": "operation_result",
                "success": False,
                "error": str(e)
            })

    async def _handle_delete(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        path_str = data.get("path")
        
        if not path_str:
            await websocket.send_json({
                "type": "error",
                "message": "path is required"
            })
            return

        try:
            target_path = self._resolve_path(path_str)
            
            # Security check
            try:
                target_path.relative_to(self.root_dir)
            except ValueError:
                await websocket.send_json({
                    "type": "operation_result",
                    "success": False,
                    "error": "Access denied: path outside root directory"
                })
                return

            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
            
            await websocket.send_json({
                "type": "operation_result",
                "success": True,
                "operation": "delete",
                "path": path_str,
            })

        except Exception as e:
            await websocket.send_json({
                "type": "operation_result",
                "success": False,
                "error": str(e)
            })

    async def _handle_rename(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        old_path_str = data.get("old_path")
        new_path_str = data.get("new_path")
        
        if not old_path_str or not new_path_str:
            await websocket.send_json({
                "type": "error",
                "message": "old_path and new_path are required"
            })
            return

        try:
            old_path = self._resolve_path(old_path_str)
            new_path = self._resolve_path(new_path_str)
            
            # Security check
            try:
                old_path.relative_to(self.root_dir)
                new_path.relative_to(self.root_dir)
            except ValueError:
                await websocket.send_json({
                    "type": "operation_result",
                    "success": False,
                    "error": "Access denied: path outside root directory"
                })
                return

            old_path.rename(new_path)
            
            await websocket.send_json({
                "type": "operation_result",
                "success": True,
                "operation": "rename",
                "old_path": old_path_str,
                "new_path": new_path_str,
            })

        except Exception as e:
            await websocket.send_json({
                "type": "operation_result",
                "success": False,
                "error": str(e)
            })

    async def _handle_copy(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        source_str = data.get("source")
        dest_str = data.get("destination")
        
        if not source_str or not dest_str:
            await websocket.send_json({
                "type": "error",
                "message": "source and destination are required"
            })
            return

        try:
            source_path = self._resolve_path(source_str)
            dest_path = self._resolve_path(dest_str)
            
            # Security check
            try:
                source_path.relative_to(self.root_dir)
                dest_path.relative_to(self.root_dir)
            except ValueError:
                await websocket.send_json({
                    "type": "operation_result",
                    "success": False,
                    "error": "Access denied: path outside root directory"
                })
                return

            if source_path.is_dir():
                shutil.copytree(source_path, dest_path)
            else:
                shutil.copy2(source_path, dest_path)
            
            await websocket.send_json({
                "type": "operation_result",
                "success": True,
                "operation": "copy",
                "source": source_str,
                "destination": dest_str,
            })

        except Exception as e:
            await websocket.send_json({
                "type": "operation_result",
                "success": False,
                "error": str(e)
            })

    async def _handle_move(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        source_str = data.get("source")
        dest_str = data.get("destination")
        
        if not source_str or not dest_str:
            await websocket.send_json({
                "type": "error",
                "message": "source and destination are required"
            })
            return

        try:
            source_path = self._resolve_path(source_str)
            dest_path = self._resolve_path(dest_str)
            
            # Security check
            try:
                source_path.relative_to(self.root_dir)
                dest_path.relative_to(self.root_dir)
            except ValueError:
                await websocket.send_json({
                    "type": "operation_result",
                    "success": False,
                    "error": "Access denied: path outside root directory"
                })
                return

            shutil.move(str(source_path), str(dest_path))
            
            await websocket.send_json({
                "type": "operation_result",
                "success": True,
                "operation": "move",
                "source": source_str,
                "destination": dest_str,
            })

        except Exception as e:
            await websocket.send_json({
                "type": "operation_result",
                "success": False,
                "error": str(e)
            })


_explorer_ws_handler: Optional[PiscesL1ExplorerWebSocket] = None


def get_explorer_ws_handler(root_dir: str, logger: XiLogger) -> PiscesL1ExplorerWebSocket:
    """Get or create the global explorer WebSocket handler instance."""
    global _explorer_ws_handler
    if _explorer_ws_handler is None:
        _explorer_ws_handler = PiscesL1ExplorerWebSocket(root_dir, logger)
    return _explorer_ws_handler
