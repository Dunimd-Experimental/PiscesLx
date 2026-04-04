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
from pathlib import Path
from typing import Dict, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect

from ...core.dc import XiLogger
from ...core.types import XiRunStatus, XiLogEntry
from ...executor import XiExecutor
from ...config.loader import load_run_types, get_xi_config


class PiscesL1RunsWebSocket:
    """
    WebSocket handler for runs management with real-time updates.
    """

    def __init__(self, executor: XiExecutor, logger: XiLogger, root_dir: str = "."):
        self.executor = executor
        self.logger = logger
        self.root_dir = root_dir
        self.active_connections: Set[WebSocket] = set()
        self.client_subscriptions: Dict[WebSocket, Set[str]] = {}
        self._broadcast_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)
        self.client_subscriptions[websocket] = set()
        self.logger.info(
            f"Runs WebSocket connected. Total clients: {len(self.active_connections)}",
            event="xi.runs_ws.connect"
        )
        
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        self.client_subscriptions.pop(websocket, None)
        self.logger.info(
            f"Runs WebSocket disconnected. Total clients: {len(self.active_connections)}",
            event="xi.runs_ws.disconnect"
        )
        
        if not self.active_connections and self._broadcast_task:
            self._broadcast_task.cancel()
            self._broadcast_task = None

    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        msg_type = data.get("type")

        if msg_type == "get_runs":
            await self._handle_get_runs(websocket)
        elif msg_type == "get_run_types":
            await self._handle_get_run_types(websocket)
        elif msg_type == "get_schema":
            await self._handle_get_schema(websocket, data)
        elif msg_type == "create_run":
            await self._handle_create_run(websocket, data)
        elif msg_type == "control":
            await self._handle_control(websocket, data)
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def _handle_get_run_types(self, websocket: WebSocket) -> None:
        try:
            run_types = load_run_types()
            await websocket.send_json({
                "type": "run_types",
                "run_types": [rt.to_dict() for rt in run_types],
                "total": len(run_types)
            })
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    async def _handle_get_schema(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        run_type = data.get("run_type", "train")

        try:
            config = get_xi_config()
            cmd_config = config.commands.get(run_type)

            if not cmd_config or not cmd_config.schema:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Schema not found for run type: {run_type}"
                })
                return

            schema = cmd_config.schema

            tabs = []
            for tab in schema.tabs:
                tabs.append({
                    "name": tab.name,
                    "label": tab.label or tab.name,
                    "available": tab.available,
                    "unavailable_reason": tab.unavailable_reason or "",
                })

            parameters = []
            for param in schema.parameters:
                param_dict = {
                    "name": param.name,
                    "type": param.type if isinstance(param.type, str) else str(param.type),
                    "description": param.description or "",
                    "required": param.required,
                    "default": param.default,
                    "options": param.options,
                    "min": param.min,
                    "max": param.max,
                    "source": param.source,
                    "source_type": param.source_type if isinstance(param.source_type, str) else str(param.source_type) if param.source_type else None,
                    "filter": param.filter,
                    "available": param.available,
                    "unavailable_reason": param.unavailable_reason or "",
                    "tab": param.tab or "basic",
                    "widget": None,
                }
                if param.widget:
                    param_dict["widget"] = {
                        "type": param.widget.type,
                        "style": {
                            "width": param.widget.style.width if param.widget.style else "full",
                            "placeholder": param.widget.style.placeholder if param.widget.style else "",
                        },
                        "props": param.widget.props or {},
                    }
                parameters.append(param_dict)

            await websocket.send_json({
                "type": "schema",
                "run_type": run_type,
                "description": schema.description or "",
                "available": schema.available,
                "unavailable_reason": schema.unavailable_reason or "",
                "tabs": tabs,
                "parameters": parameters
            })

        except Exception as e:
            self.logger.error(
                f"Failed to get schema: {e}",
                event="xi.runs_ws.schema_error"
            )
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to get schema: {str(e)}"
            })

    async def _handle_create_run(self, websocket: WebSocket, data: Dict[str, Any]) -> None:
        run_type = data.get("run_type", "train")
        run_name = data.get("name", "")
        run_id_prefix = data.get("run_id")
        args = data.get("config", {})

        try:
            config = get_xi_config()
            cmd_config = config.commands.get(run_type)

            if not cmd_config:
                await websocket.send_json({
                    "type": "error",
                    "run_type": run_type,
                    "message": f"Command '{run_type}' not found in configuration"
                })
                return

            schema_params = cmd_config.schema.parameters if cmd_config.schema else []

            argv = self.executor.build_argv_from_schema(
                run_type,
                args,
                schema_params
            )

            if run_id_prefix:
                argv.extend(["--run_id", run_id_prefix])
            if run_name:
                argv.extend(["--run_name", run_name])

            import sys
            import shutil
            
            venv_python = None
            if config.environment.virtualenv and config.environment.virtualenv.enabled:
                venv_path = config.environment.virtualenv.path
                if not Path(venv_path).is_absolute():
                    venv_path = str(Path(self.root_dir) / venv_path)
                
                if sys.platform == "win32":
                    venv_python = str(Path(venv_path) / "Scripts" / "python.exe")
                else:
                    venv_python = str(Path(venv_path) / "bin" / "python")
                
                if not Path(venv_python).exists():
                    self.logger.warning(
                        f"Virtual environment Python not found: {venv_python}, falling back to system Python",
                        event="xi.runs_ws.venv_not_found"
                    )
                    venv_python = None
            
            if venv_python:
                if argv and argv[0] in ("python", "python3", sys.executable):
                    argv[0] = venv_python
                elif argv and not argv[0].endswith("python"):
                    argv.insert(0, venv_python)
            
            self.logger.info(
                f"WS Creating run: type={run_type}, name={run_name}, venv={venv_python is not None}",
                event="xi.runs_ws.create"
            )

            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.root_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"}
            )

            from ...executor.process import generate_run_id
            from ...core.types import XiCommand
            
            type_to_command = {
                "train": XiCommand.TRAIN,
                "inference": XiCommand.INFERENCE,
                "benchmark": XiCommand.BENCHMARK,
                "serve": XiCommand.SERVE,
                "download": XiCommand.DOWNLOAD,
                "monitor": XiCommand.MONITOR,
            }
            command = type_to_command.get(run_type, XiCommand.TRAIN)
            generated_run_id = run_id_prefix or generate_run_id(command)
            run_id = generated_run_id

            self.executor.active_processes[run_id] = process
            self.executor._process_status[run_id] = XiRunStatus.RUNNING
            self.executor.output_queues[run_id] = asyncio.Queue()

            self.client_subscriptions[websocket].add(run_id)

            await websocket.send_json({
                "type": "run_created",
                "run_id": run_id,
                "run_type": run_type,
                "name": run_name,
                "status": "running",
                "message": f"Run {run_type} created successfully"
            })

            asyncio.create_task(
                self._stream_run_output(run_id, process, websocket)
            )

        except Exception as e:
            self.logger.error(
                f"Failed to create run: {e}",
                event="xi.runs_ws.create_error"
            )
            await websocket.send_json({
                "type": "error",
                "run_type": run_type,
                "message": str(e)
            })

    async def _stream_run_output(
        self,
        run_id: str,
        process: asyncio.subprocess.Process,
        websocket: WebSocket
    ) -> None:
        queue = self.executor.output_queues.get(run_id)

        async def read_stream(stream, source: str):
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break

                    text = line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        entry = XiLogEntry(
                            timestamp=datetime.now().isoformat(),
                            level="info",
                            message=text,
                            source=source,
                            run_id=run_id
                        )
                        if queue:
                            await queue.put(entry)

                        message = {
                            "type": "output",
                            "run_id": run_id,
                            "line": text,
                            "source": source,
                            "timestamp": entry.timestamp
                        }
                        try:
                            await websocket.send_json(message)
                        except Exception:
                            pass

            except Exception as e:
                self.logger.error(
                    f"Error streaming {source}: {e}",
                    event="xi.runs_ws.stream_error"
                )

        stdout_task = asyncio.create_task(read_stream(process.stdout, "stdout"))
        stderr_task = asyncio.create_task(read_stream(process.stderr, "stderr"))

        await process.wait()

        try:
            await stdout_task
        except asyncio.CancelledError:
            pass
        try:
            await stderr_task
        except asyncio.CancelledError:
            pass

        exit_code = process.returncode
        final_status = XiRunStatus.COMPLETED if exit_code == 0 else XiRunStatus.FAILED

        self.executor._process_status[run_id] = final_status

        try:
            await websocket.send_json({
                "type": "run_completed",
                "run_id": run_id,
                "exit_code": exit_code,
                "status": final_status.value,
                "timestamp": datetime.now().isoformat()
            })
        except Exception:
            pass

        await self._broadcast_status(run_id, final_status.value)

        self._cleanup_run(run_id)

    def _cleanup_run(self, run_id: str) -> None:
        if run_id in self.executor.active_processes:
            del self.executor.active_processes[run_id]
        if run_id in self.executor.output_queues:
            del self.executor.output_queues[run_id]

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


def get_runs_ws_handler(executor: XiExecutor, logger: XiLogger, root_dir: str = ".") -> PiscesL1RunsWebSocket:
    """Get or create the global runs WebSocket handler instance."""
    global _runs_ws_handler
    if _runs_ws_handler is None:
        _runs_ws_handler = PiscesL1RunsWebSocket(executor, logger, root_dir)
    return _runs_ws_handler
