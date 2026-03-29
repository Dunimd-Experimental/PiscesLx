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
Runs WebSocket handler for real-time run status updates.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

from ...core.dc import XiLogger
from ...core.types import XiRunStatus
from ...executor import XiExecutor


class PiscesL1RunsWebSocket:
    """
    WebSocket handler for runs management with real-time updates.
    """

    def __init__(self, executor: XiExecutor, logger: XiLogger):
        self.executor = executor
        self.logger = logger
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(
            f"Runs WebSocket connected. Total clients: {len(self.active_connections)}",
            event="xi.runs_ws.connect"
        )
        
        # Start broadcast task if not running
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        self.logger.info(
            f"Runs WebSocket disconnected. Total clients: {len(self.active_connections)}",
            event="xi.runs_ws.disconnect"
        )
        
        # Stop broadcast task if no clients
        if not self.active_connections and self._broadcast_task:
            self._broadcast_task.cancel()
            self._broadcast_task = None

    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        msg_type = data.get("type")

        if msg_type == "get_runs":
            await self._handle_get_runs(websocket)
        elif msg_type == "control":
            await self._handle_control(websocket, data)
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def _handle_get_runs(self, websocket: WebSocket) -> None:
        try:
            runs = []
            for run_id, status in self.executor.list_active_runs().items():
                runs.append({
                    "run_id": run_id,
                    "status": status.value,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                })
            
            await websocket.send_json({
                "type": "runs_list",
                "runs": runs,
                "total": len(runs)
            })
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    async def _handle_control(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        run_id = data.get("run_id")
        action = data.get("action")

        if not run_id or not action:
            await websocket.send_json({
                "type": "error",
                "message": "run_id and action are required"
            })
            return

        process = self.executor.active_processes.get(run_id)
        if not process:
            await websocket.send_json({
                "type": "error",
                "run_id": run_id,
                "message": f"No active process found for run_id: {run_id}"
            })
            return

        try:
            import signal as sig_module
            import sys
            
            if action == "pause":
                if sys.platform == "win32":
                    process.send_signal(sig_module.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(sig_module.SIGSTOP)
                self.executor._process_status[run_id] = XiRunStatus.PAUSED

            elif action == "resume":
                if sys.platform == "win32":
                    process.send_signal(sig_module.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(sig_module.SIGCONT)
                self.executor._process_status[run_id] = XiRunStatus.RUNNING

            elif action == "cancel":
                process.terminate()
                self.executor._process_status[run_id] = XiRunStatus.CANCELLED

            elif action == "kill":
                process.kill()
                self.executor._process_status[run_id] = XiRunStatus.CANCELLED

            else:
                await websocket.send_json({
                    "type": "error",
                    "run_id": run_id,
                    "message": f"Unknown action: {action}"
                })
                return

            await websocket.send_json({
                "type": "control_result",
                "run_id": run_id,
                "action": action,
                "success": True
            })

            # Broadcast status update to all clients
            await self._broadcast_status(run_id, self.executor._process_status[run_id].value)

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "run_id": run_id,
                "message": str(e)
            })

    async def _broadcast_loop(self) -> None:
        """Broadcast runs updates to all connected clients every 2 seconds."""
        try:
            while True:
                if self.active_connections:
                    runs = []
                    for run_id, status in self.executor.list_active_runs().items():
                        runs.append({
                            "run_id": run_id,
                            "status": status.value,
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                        })
                    
                    message = {
                        "type": "runs_update",
                        "runs": runs,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Send to all connected clients
                    for websocket in list(self.active_connections):
                        try:
                            await websocket.send_json(message)
                        except Exception:
                            # Client disconnected, will be cleaned up
                            pass
                
                await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            pass

    async def _broadcast_status(self, run_id: str, status: str) -> None:
        """Broadcast a single run status update."""
        message = {
            "type": "run_update",
            "run": {
                "run_id": run_id,
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
        }
        
        for websocket in list(self.active_connections):
            try:
                await websocket.send_json(message)
            except Exception:
                pass


_runs_ws_handler: Optional[PiscesL1RunsWebSocket] = None


def get_runs_ws_handler(executor: XiExecutor, logger: XiLogger) -> PiscesL1RunsWebSocket:
    """Get or create the global runs WebSocket handler instance."""
    global _runs_ws_handler
    if _runs_ws_handler is None:
        _runs_ws_handler = PiscesL1RunsWebSocket(executor, logger)
    return _runs_ws_handler
