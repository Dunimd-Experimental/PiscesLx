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
WebSocket configuration for Xi Server.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Set, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ...core.dc import XiLogger
from ...core.types import XiRunStatus, XiLogEntry
from ...executor import XiExecutor
from ...config import get_xi_config

from .runs import get_runs_ws_handler
from .monitor import get_monitor_ws_handler
from .explorer import get_explorer_ws_handler
from .welcome import get_welcome_ws_handler


class PiscesL1TrainingWebSocket:
    """
    WebSocket handler for training commands with bidirectional communication.
    """

    def __init__(self, executor: XiExecutor, root_dir: str, logger: XiLogger):
        self.executor = executor
        self.root_dir = root_dir
        self.logger = logger
        self.active_connections: Set[WebSocket] = set()
        self.client_subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        client_host = websocket.client.host
        client_port = websocket.client.port
        self.logger.info(
            f"Training WebSocket connection attempt from {client_host}:{client_port}",
            event="xi.training_ws.connect_attempt"
        )
        try:
            await websocket.accept()
            self.active_connections.add(websocket)
            self.client_subscriptions[websocket] = set()
            self.logger.info(
                f"Training WebSocket connected successfully from {client_host}:{client_port}. Total clients: {len(self.active_connections)}",
                event="xi.training_ws.connect"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to accept WebSocket connection: {e}",
                event="xi.training_ws.accept_error"
            )
            raise

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)
        self.client_subscriptions.pop(websocket, None)
        self.logger.info(
            f"Training WebSocket disconnected. Total clients: {len(self.active_connections)}",
            event="xi.training_ws.disconnect"
        )

    async def broadcast_to_subscribed(
        self,
        run_id: str,
        message: Dict[str, Any]
    ) -> None:
        for websocket in self.active_connections:
            if run_id in self.client_subscriptions.get(websocket, set()):
                try:
                    await websocket.send_json(message)
                except Exception:
                    pass

    async def broadcast_all(self, message: Dict[str, Any]) -> None:
        for websocket in list(self.active_connections):
            try:
                await websocket.send_json(message)
            except Exception:
                pass

    async def handle_message(
        self,
        websocket: WebSocket,
        data: Dict[str, Any]
    ) -> None:
        msg_type = data.get("type")

        if msg_type == "start_training":
            await self._handle_start_training(websocket, data)
        elif msg_type == "control":
            await self._handle_control(websocket, data)
        elif msg_type == "get_runs":
            await self._handle_get_runs(websocket)
        elif msg_type == "get_schema":
            await self._handle_get_schema(websocket, data)
        elif msg_type == "subscribe":
            await self._handle_subscribe(websocket, data)
        elif msg_type == "unsubscribe":
            await self._handle_unsubscribe(websocket, data)
        else:
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}"
            })

    async def _handle_start_training(
        self,
        websocket: WebSocket,
        data: Dict[str, Any]
    ) -> None:
        command = data.get("command", "train")
        args = data.get("args", {})
        run_name = data.get("run_name")
        run_id_prefix = data.get("run_id")

        try:
            config = get_xi_config()
            cmd_config = config.commands.get(command)

            if not cmd_config:
                await websocket.send_json({
                    "type": "error",
                    "command": command,
                    "message": f"Command '{command}' not found in configuration"
                })
                return

            schema_params = cmd_config.schema.parameters if cmd_config.schema else []

            argv = self.executor.build_argv_from_schema(
                command,
                args,
                schema_params
            )

            if run_id_prefix:
                argv.extend(["--run_id", run_id_prefix])
            if run_name:
                argv.extend(["--run_name", run_name])

            self.logger.info(
                f"WS Starting training: command={command}, args={args}",
                event="xi.training_ws.start"
            )

            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=self.root_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"}
            )

            generated_run_id = run_id_prefix or self._generate_run_id(command)
            run_id = generated_run_id

            self.executor.active_processes[run_id] = process
            self.executor._process_status[run_id] = XiRunStatus.RUNNING
            self.executor.output_queues[run_id] = asyncio.Queue()

            self.client_subscriptions[websocket].add(run_id)

            await websocket.send_json({
                "type": "training_started",
                "run_id": run_id,
                "command": command,
                "argv": argv,
                "message": f"Training {command} started"
            })

            asyncio.create_task(
                self._stream_training_output(run_id, process, websocket)
            )

        except Exception as e:
            self.logger.error(
                f"Failed to start training: {e}",
                event="xi.training_ws.start_error"
            )
            await websocket.send_json({
                "type": "error",
                "command": command,
                "message": str(e)
            })

    async def _stream_training_output(
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
                        await websocket.send_json(message)

                        await self.broadcast_all({
                            "type": "metrics",
                            "run_id": run_id,
                            "line": text
                        })

            except Exception as e:
                self.logger.error(
                    f"Error streaming {source}: {e}",
                    event="xi.training_ws.stream_error"
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

        await websocket.send_json({
            "type": "completed",
            "run_id": run_id,
            "exit_code": exit_code,
            "status": final_status.value,
            "timestamp": datetime.now().isoformat()
        })

        await self.broadcast_all({
            "type": "status",
            "run_id": run_id,
            "status": final_status.value
        })

        self._cleanup_run(run_id)

    def _cleanup_run(self, run_id: str) -> None:
        if run_id in self.executor.active_processes:
            del self.executor.active_processes[run_id]
        if run_id in self.executor.output_queues:
            del self.executor.output_queues[run_id]

    async def _handle_control(
        self,
        websocket: WebSocket,
        data: Dict[str, Any]
    ) -> None:
        run_id = data.get("run_id")
        action = data.get("action")

        if not run_id:
            await websocket.send_json({
                "type": "error",
                "message": "run_id is required"
            })
            return

        if not action:
            await websocket.send_json({
                "type": "error",
                "message": "action is required"
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
            if action == "pause":
                if __import__("sys").platform == "win32":
                    process.send_signal(sig_module.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(sig_module.SIGSTOP)
                self.executor._process_status[run_id] = XiRunStatus.PAUSED

            elif action == "resume":
                if __import__("sys").platform == "win32":
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
                "type": "control_acknowledged",
                "run_id": run_id,
                "action": action,
                "status": self.executor._process_status[run_id].value
            })

            await self.broadcast_all({
                "type": "status",
                "run_id": run_id,
                "status": self.executor._process_status[run_id].value
            })

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "run_id": run_id,
                "message": str(e)
            })

    async def _handle_get_runs(
        self,
        websocket: WebSocket
    ) -> None:
        runs = []
        for run_id, status in self.executor.list_active_runs().items():
            runs.append({
                "run_id": run_id,
                "status": status.value,
                "timestamp": datetime.now().isoformat()
            })

        await websocket.send_json({
            "type": "runs_list",
            "runs": runs,
            "total": len(runs)
        })

    async def _handle_get_schema(
        self,
        websocket: WebSocket,
        data: Dict[str, Any]
    ) -> None:
        command = data.get("command", "train")

        try:
            config = get_xi_config()
            cmd_config = config.commands.get(command)

            if not cmd_config or not cmd_config.schema:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Schema not found for command: {command}"
                })
                return

            schema = cmd_config.schema

            # Build tabs with availability info
            tabs = []
            for tab in schema.tabs:
                tabs.append({
                    "name": tab.name,
                    "label": tab.label or tab.name,
                    "available": tab.available,
                    "unavailable_reason": tab.unavailable_reason or "",
                    "parameters": tab.parameters or []
                })

            # Build parameters with full info
            parameters = []
            for param in schema.parameters:
                param_dict = {
                    "name": param.name,
                    "type": param.type.value if hasattr(param.type, 'value') else str(param.type),
                    "description": param.description or "",
                    "required": param.required,
                    "default": param.default,
                    "options": param.options,
                    "min": param.min,
                    "max": param.max,
                    "source": param.source,
                    "source_type": param.source_type.value if hasattr(param.source_type, 'value') else str(param.source_type) if param.source_type else None,
                    "filter": param.filter,
                    "available": param.available,
                    "unavailable_reason": param.unavailable_reason or "",
                    "tab": param.tab or "basic"
                }
                parameters.append(param_dict)

            await websocket.send_json({
                "type": "schema",
                "command": command,
                "description": schema.description or "",
                "available": schema.available,
                "unavailable_reason": schema.unavailable_reason or "",
                "tabs": tabs,
                "parameters": parameters
            })

        except Exception as e:
            self.logger.error(
                f"Failed to get schema: {e}",
                event="xi.training_ws.schema_error"
            )
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to get schema: {str(e)}"
            })

    async def _handle_subscribe(
        self,
        websocket: WebSocket,
        data: Dict[str, Any]
    ) -> None:
        run_id = data.get("run_id")
        if run_id:
            self.client_subscriptions[websocket].add(run_id)
            await websocket.send_json({
                "type": "subscribed",
                "run_id": run_id
            })

    async def _handle_unsubscribe(
        self,
        websocket: WebSocket,
        data: Dict[str, Any]
    ) -> None:
        run_id = data.get("run_id")
        if run_id:
            self.client_subscriptions[websocket].discard(run_id)
            await websocket.send_json({
                "type": "unsubscribed",
                "run_id": run_id
            })

    def _generate_run_id(self, command: str) -> str:
        from ...executor.process import generate_run_id
        from ...core.types import XiCommand
        return generate_run_id(XiCommand.TRAIN)


_training_ws_handler: Optional[PiscesL1TrainingWebSocket] = None


def setup_websockets(
    app: FastAPI,
    executor: XiExecutor,
    logger: XiLogger,
    root_dir: str = ".",
    start_time: datetime = None,
    request_count: Dict[str, int] = None
) -> None:
    """
    Setup WebSocket endpoints for the FastAPI application.

    Args:
        app: FastAPI application
        executor: XiExecutor instance
        logger: XiLogger instance
        root_dir: Working directory for command execution
        start_time: Server start time for uptime calculation
        request_count: Request count dictionary for QPS calculation
    """
    global _training_ws_handler
    _training_ws_handler = PiscesL1TrainingWebSocket(executor, root_dir, logger)

    @app.websocket("/ws/training")
    async def training_websocket(websocket: WebSocket):
        await _training_ws_handler.connect(websocket)

        try:
            async for raw_message in websocket.iter_text():
                try:
                    data = json.loads(raw_message)
                    await _training_ws_handler.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Training WebSocket error: {e}", event="xi.training_ws.error")
        finally:
            _training_ws_handler.disconnect(websocket)

    @app.websocket("/ws/logs/{run_id}")
    async def stream_logs(websocket: WebSocket, run_id: str):
        await websocket.accept()
        logger.info(f"WebSocket connected for run: {run_id}", event="xi.ws.connect")

        try:
            async for entry in executor.get_output_stream(run_id):
                await websocket.send_json({
                    "timestamp": entry.timestamp,
                    "level": entry.level,
                    "message": entry.message,
                    "source": entry.source,
                    "run_id": entry.run_id
                })
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for run: {run_id}", event="xi.ws.disconnect")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", event="xi.ws.error")
            await websocket.close()

    @app.websocket("/ws/stats")
    async def stream_stats(websocket: WebSocket):
        await websocket.accept()
        logger.info("Stats WebSocket connected", event="xi.ws.stats.connect")

        try:
            from ..hardware import collect_system_stats
            while True:
                stats = await collect_system_stats()
                from dataclasses import asdict
                await websocket.send_json(asdict(stats))
                await asyncio.sleep(2.0)
        except WebSocketDisconnect:
            logger.info("Stats WebSocket disconnected", event="xi.ws.stats.disconnect")
        except Exception as e:
            logger.error(f"Stats WebSocket error: {e}", event="xi.ws.stats.error")

    # Runs WebSocket endpoint
    runs_handler = get_runs_ws_handler(executor, logger, root_dir)
    
    @app.websocket("/ws/runs")
    async def runs_websocket(websocket: WebSocket):
        await runs_handler.connect(websocket)

        try:
            async for raw_message in websocket.iter_text():
                try:
                    data = json.loads(raw_message)
                    await runs_handler.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Runs WebSocket error: {e}", event="xi.runs_ws.error")
        finally:
            runs_handler.disconnect(websocket)

    # Monitor WebSocket endpoint
    monitor_handler = get_monitor_ws_handler(logger, start_time, request_count)
    
    @app.websocket("/ws/monitor")
    async def monitor_websocket(websocket: WebSocket):
        await monitor_handler.connect(websocket)

        try:
            async for raw_message in websocket.iter_text():
                try:
                    data = json.loads(raw_message)
                    await monitor_handler.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Monitor WebSocket error: {e}", event="xi.monitor_ws.error")
        finally:
            monitor_handler.disconnect(websocket)

    # Explorer WebSocket endpoint
    explorer_handler = get_explorer_ws_handler(root_dir, logger)
    
    @app.websocket("/ws/fs")
    async def explorer_websocket(websocket: WebSocket):
        await explorer_handler.connect(websocket)

        try:
            async for raw_message in websocket.iter_text():
                try:
                    data = json.loads(raw_message)
                    await explorer_handler.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Explorer WebSocket error: {e}", event="xi.explorer_ws.error")
        finally:
            explorer_handler.disconnect(websocket)

    # Welcome WebSocket endpoint
    welcome_handler = get_welcome_ws_handler(logger)
    
    @app.websocket("/ws/welcome")
    async def welcome_websocket(websocket: WebSocket):
        await welcome_handler.connect(websocket)
        last_activity = asyncio.get_event_loop().time()
        CONNECTION_TIMEOUT = 300  # 5 minutes timeout

        async def ping_task():
            nonlocal last_activity
            try:
                while True:
                    await asyncio.sleep(30)
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        break
            except asyncio.CancelledError:
                pass

        async def timeout_monitor():
            nonlocal last_activity
            try:
                while True:
                    await asyncio.sleep(60)
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_activity > CONNECTION_TIMEOUT:
                        logger.info("Welcome WebSocket connection timed out", event="xi.welcome_ws.timeout")
                        try:
                            await websocket.close(code=1001, reason="Connection timeout")
                        except Exception:
                            pass
                        break
            except asyncio.CancelledError:
                pass

        ping = asyncio.create_task(ping_task())
        timeout_task = asyncio.create_task(timeout_monitor())

        try:
            async for raw_message in websocket.iter_text():
                last_activity = asyncio.get_event_loop().time()
                try:
                    data = json.loads(raw_message)
                    if data.get("type") == "pong":
                        continue
                    await welcome_handler.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"Welcome WebSocket error: {e}", event="xi.welcome_ws.error")
        finally:
            ping.cancel()
            timeout_task.cancel()
            try:
                await ping
            except asyncio.CancelledError:
                pass
            try:
                await timeout_task
            except asyncio.CancelledError:
                pass
            await welcome_handler.disconnect(websocket)
