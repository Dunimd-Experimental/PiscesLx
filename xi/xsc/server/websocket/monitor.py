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
Monitor WebSocket handler for real-time system stats.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

from ...core.dc import XiLogger
from ..hardware import collect_system_stats


class PiscesL1MonitorWebSocket:
    """
    WebSocket handler for system monitoring with real-time stats.
    """
    
    def __init__(self, logger: XiLogger, start_time: datetime = None, request_count: Dict[str, int] = None):
        self.logger = logger
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None
        self.start_time = start_time or datetime.now()
        self.request_count = request_count or {"value": 0}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(
            f"Monitor WebSocket connected. Total clients: {len(self.active_connections)}",
            event="xi.monitor_ws.connect"
        )
        
        # Send initial stats immediately with start_time
        try:
            stats = await collect_system_stats(
                start_time=self.start_time,
                request_count=self.request_count.get("value", 0)
            )
            from dataclasses import asdict
            await websocket.send_json({
                "type": "stats",
                "data": asdict(stats),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            self.logger.error(f"Failed to send initial stats: {e}", event="xi.monitor_ws.error")
        
        # Start broadcast task if not running
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        self.logger.info(
            f"Monitor WebSocket disconnected. Total clients: {len(self.active_connections)}",
            event="xi.monitor_ws.disconnect"
        )
        
        # Stop broadcast task if no clients
        if not self.active_connections and self._broadcast_task:
            self._broadcast_task.cancel()
            self._broadcast_task = None

    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        msg_type = data.get("type")

        if msg_type == "get_stats":
            await self._handle_get_stats(websocket)
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def _handle_get_stats(self, websocket: WebSocket) -> None:
        try:
            stats = await collect_system_stats(
                start_time=self.start_time,
                request_count=self.request_count.get("value", 0)
            )
            from dataclasses import asdict
            await websocket.send_json({
                "type": "stats",
                "data": asdict(stats),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    async def _broadcast_loop(self) -> None:
        """Broadcast system stats to all connected clients every 2 seconds."""
        try:
            while True:
                if self.active_connections:
                    try:
                        stats = await collect_system_stats(
                            start_time=self.start_time,
                            request_count=self.request_count.get("value", 0)
                        )
                        from dataclasses import asdict
                        message = {
                            "type": "stats",
                            "data": asdict(stats),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Send to all connected clients
                        for websocket in list(self.active_connections):
                            try:
                                await websocket.send_json(message)
                            except Exception:
                                # Client disconnected, will be cleaned up
                                pass
                    except Exception as e:
                        self.logger.error(f"Failed to collect stats: {e}", event="xi.monitor_ws.error")
                
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            pass


_monitor_ws_handler: Optional[PiscesL1MonitorWebSocket] = None


def get_monitor_ws_handler(logger: XiLogger, start_time: datetime = None, request_count: Dict[str, int] = None) -> PiscesL1MonitorWebSocket:
    """Get or create the global monitor WebSocket handler instance."""
    global _monitor_ws_handler
    if _monitor_ws_handler is None:
        _monitor_ws_handler = PiscesL1MonitorWebSocket(logger, start_time, request_count)
    return _monitor_ws_handler
