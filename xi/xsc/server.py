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
FastAPI Server for Xi Studio Backend.

This module provides the main FastAPI server that exposes REST and WebSocket
endpoints for the Xi Studio frontend. It runs on port 3140 and serves as
the bridge between the Next.js frontend and manage.py commands.

Endpoints:
    REST API:
        - GET  /healthz: Health check endpoint
        - GET  /stats: System resource statistics
        - POST /v1/runs: Start a new training/inference run
        - GET  /v1/runs: List all runs
        - GET  /v1/runs/{run_id}: Get run details
        - POST /v1/runs/{run_id}/control: Control a run (pause/resume/cancel)
        - GET  /v1/models: List available models
        - POST /v1/chat/completions: Chat completion endpoint
        - POST /v1/embeddings: Embedding generation endpoint

    WebSocket:
        - WS   /ws/logs/{run_id}: Stream logs for a run
        - WS   /ws/stats: Stream system statistics
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .dc import XiLogger
from .executor import XiExecutor
from .session import XmcSessionManager, XmcNotificationManager
from .types import (
    XiCommand,
    XiRequest,
    XiResponse,
    XiRunStatus,
    XiRunInfo,
    XiSystemStats,
    XiLogEntry,
    XiControlRequest,
    XiControlResponse,
)


class XiServer:
    def __init__(self, port: int = 3140, root_dir: Optional[str] = None):
        self.port = port
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.logger = XiLogger(
            "Xi.Server",
            enable_file=True
        )
        self.session_manager = XmcSessionManager()
        self.notification_manager = XmcNotificationManager()
        self.executor = XiExecutor(str(self.root_dir))
        self._start_time = datetime.now()
        self._request_count = 0

        self.app = FastAPI(
            title="Xi Studio API",
            description="Backend API for Xi Studio",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        self._setup_middleware()
        self._setup_routes()
        self._setup_websockets()

    def _setup_middleware(self) -> None:
        @self.app.middleware("http")
        async def check_session(request, call_next):
            public_paths = {"/docs", "/redoc", "/openapi.json", "/healthz", "/handshake", "/v1/xi/validate-config", "/v1/xi/setup-environment", "/v1/xi/first-launch"}
            
            if request.url.path in public_paths or request.url.path.startswith("/docs") or request.url.path.startswith("/redoc"):
                return await call_next(request)

            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                session = self.session_manager.validate_token(token)
                if session:
                    return await call_next(request)

            accept_header = request.headers.get("accept", "")
            is_browser = "text/html" in accept_header and "application/json" not in accept_header
            if is_browser:
                from starlette.responses import HTMLResponse
                return HTMLResponse(
                    content=self._get_access_denied_html(),
                    status_code=403
                )

            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied"}
            )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _get_access_denied_html(self) -> str:
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Access Denied</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #ffffff;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;
        }
        .card {
            max-width: 400px;
            width: 100%;
            background: #ffffff;
            border: 1px solid #e5e5e5;
            border-radius: 12px;
            padding: 32px;
            text-align: center;
        }
        .icon-wrap {
            width: 56px;
            height: 56px;
            margin: 0 auto 20px;
            border-radius: 50%;
            background: #fef2f2;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .icon-wrap svg { width: 28px; height: 28px; color: #ef4444; }
        .title {
            font-size: 18px;
            font-weight: 600;
            color: #171717;
            margin-bottom: 8px;
        }
        .desc {
            font-size: 14px;
            color: #737373;
            line-height: 1.5;
            margin-bottom: 20px;
        }
        .info {
            background: #f5f5f5;
            border-radius: 8px;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .info svg { width: 18px; height: 18px; color: #a3a3a3; flex-shrink: 0; }
        .info-text { text-align: left; }
        .info-label { font-size: 12px; color: #737373; }
        .info-value { font-family: monospace; font-size: 12px; color: #171717; }
    </style>
</head>
<body>
    <div class="card">
        <div class="icon-wrap">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
        </div>
        <h1 class="title">Access Denied</h1>
        <p class="desc">Direct browser access is not permitted.<br>Please use Xi Studio frontend.</p>
        <div class="info">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
            </svg>
            <div class="info-text">
                <div class="info-label">Server</div>
                <div class="info-value">127.0.0.1:3140</div>
            </div>
        </div>
    </div>
</body>
</html>'''

    def _setup_routes(self) -> None:

        @self.app.post("/handshake")
        async def handshake(request: dict):
            client = request.get("client", "unknown")
            version = request.get("version", "1.0.0")
            timestamp = request.get("timestamp")
            auth = request.get("auth", {})

            self.logger.info(
                f"Handshake request from {client} v{version}",
                event="xsc.handshake.request"
            )

            if version != "1.0.0":
                return {
                    "type": "handshake_error",
                    "error": "Unsupported version",
                    "supported_versions": ["1.0.0"]
                }, 400

            session = self.session_manager.create_session(
                client=client,
                version=version,
                auth=auth
            )

            self.logger.info(
                f"Handshake success: {session.session_id}",
                event="xsc.handshake.success"
            )

            return {
                "type": "handshake_ack",
                "session_id": session.session_id,
                "token": session.token,
                "capabilities": session.capabilities,
                "endpoints": {
                    "ws": f"ws://127.0.0.1:{self.port}/ws",
                    "api": f"http://127.0.0.1:{self.port}/api"
                },
                "server_info": {
                    "version": "1.0.0",
                    "uptime": (datetime.now() - self._start_time).total_seconds()
                }
            }

        @self.app.get("/healthz")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.get("/stats")
        async def get_stats():
            self._request_count += 1
            stats = await self._collect_system_stats()
            from dataclasses import asdict
            return asdict(stats)

        @self.app.get("/v1/models")
        async def list_models():
            self._request_count += 1
            models = await self._list_available_models()
            return {"data": models, "object": "list"}

        @self.app.get("/v1/runs")
        async def list_runs():
            self._request_count += 1
            runs = self.executor.list_active_runs()
            run_list = []
            for run_id, status in runs.items():
                run_list.append({
                    "run_id": run_id,
                    "status": status.value,
                    "created_at": datetime.now().isoformat()
                })
            return {"runs": run_list, "total": len(run_list)}

        @self.app.post("/v1/runs")
        async def create_run(request: dict):
            self._request_count += 1

            command_name = request.get("command", "train")
            args = request.get("args", {})
            run_id = request.get("run_id")
            run_name = request.get("run_name")
            background = request.get("background", True)

            try:
                from .loader import get_xi_config
                config = get_xi_config()
                cmd_config = config.commands.get(command_name)
                
                if not cmd_config:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Command '{command_name}' not found in configuration"
                    )
                
                schema_params = cmd_config.schema.parameters if cmd_config.schema else []
                
                argv = self.executor.build_argv_from_schema(
                    command_name,
                    args,
                    schema_params
                )
                
                if run_id:
                    argv.extend(["--run_id", run_id])
                if run_name:
                    argv.extend(["--run_name", run_name])
                
                self.logger.info(
                    f"Executing command: {' '.join(argv)}",
                    event="xi.run.create"
                )

                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=str(self.root_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}
                )

                generated_run_id = run_id or self.executor._generate_run_id(
                    XiCommand(command_name)
                )
                
                self.executor.active_processes[generated_run_id] = process
                self.executor._process_status[generated_run_id] = XiRunStatus.RUNNING
                self.executor.output_queues[generated_run_id] = asyncio.Queue()

                asyncio.create_task(
                    self.executor._stream_output(generated_run_id, process)
                )

                return {
                    "success": True,
                    "run_id": generated_run_id,
                    "message": f"Command {command_name} started",
                    "command": command_name,
                    "argv": argv,
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to create run: {e}", event="xi.run.error")
                return {
                    "success": False,
                    "error": str(e),
                }

        @self.app.get("/v1/runs/{run_id}")
        async def get_run(run_id: str):
            self._request_count += 1
            status = self.executor.get_status(run_id)
            if not status:
                raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

            return {
                "run_id": run_id,
                "status": status.value,
                "created_at": datetime.now().isoformat()
            }

        @self.app.post("/v1/runs/{run_id}/control")
        async def control_run(run_id: str, request: dict):
            self._request_count += 1
            action = request.get("action")
            if not action:
                raise HTTPException(status_code=400, detail="Action is required")

            response = await self.executor.control(run_id, action)
            return response.__dict__

        @self.app.get("/v1/fs/drives")
        async def get_drives():
            self._request_count += 1
            import platform
            import shutil
            
            drives = []
            is_windows = platform.system() == "Windows"
            
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

        @self.app.get("/v1/fs/directory")
        async def get_directory(path: str = "/"):
            self._request_count += 1
            import platform
            import shutil
            import asyncio
            
            is_windows = platform.system() == "Windows"
            
            try:
                if is_windows:
                    if path == "/" or path == "":
                        target_path = str(self.root_dir)
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

        @self.app.post("/v1/fs/folder")
        async def create_folder(request: dict):
            self._request_count += 1
            import asyncio
            
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

        @self.app.post("/v1/fs/file")
        async def create_file(request: dict):
            self._request_count += 1
            import asyncio
            
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

        @self.app.delete("/v1/fs/item")
        async def delete_item(path: str = ""):
            self._request_count += 1
            import asyncio
            import shutil
            
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

        @self.app.post("/v1/fs/rename")
        async def rename_item(request: dict):
            self._request_count += 1
            import asyncio
            
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

        @self.app.post("/v1/fs/copy")
        async def copy_item(request: dict):
            self._request_count += 1
            import asyncio
            import shutil
            
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

        @self.app.post("/v1/fs/move")
        async def move_item(request: dict):
            self._request_count += 1
            import asyncio
            import shutil
            
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

        @self.app.get("/v1/xi/first-launch")
        async def check_first_launch():
            self._request_count += 1
            import tomli
            
            xi_toml_path = self.root_dir / ".xi" / "xi.toml"
            
            if not xi_toml_path.exists():
                return {"is_first_launch": True}
            
            try:
                with open(xi_toml_path, "rb") as f:
                    config = tomli.load(f)
                
                first_launch = config.get("project", {}).get("first_launch", True)
                return {"is_first_launch": bool(first_launch)}
            except Exception:
                return {"is_first_launch": True}

        @self.app.post("/v1/xi/complete-first-launch")
        async def complete_first_launch():
            self._request_count += 1
            import re

            xi_toml_path = self.root_dir / ".xi" / "xi.toml"

            if not xi_toml_path.exists():
                return {"success": False, "error": "xi.toml not found"}

            try:
                with open(xi_toml_path, "r", encoding="utf-8") as f:
                    content = f.read()

                content = re.sub(r'first_launch\s*=\s*true', 'first_launch = false', content, flags=re.IGNORECASE)

                with open(xi_toml_path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(content)

                return {"success": True}
            except Exception as e:
                return {"success": False, "error": str(e)}

        @self.app.get("/v1/xi/validate-config")
        async def validate_xi_config():
            from fastapi.responses import StreamingResponse
            import tomli
            import shutil
            import subprocess
            import json
            import asyncio
            import httpx
            
            async def generate_validation_events():
                all_valid = True
                xi_dir = self.root_dir / ".xi"
                xi_toml_path = xi_dir / "xi.toml"
                
                yield f"data: {json.dumps({'event': 'checking', 'step': 'xi_toml_syntax', 'message': 'Checking main configuration syntax'})}\n\n"
                await asyncio.sleep(0.1)
                
                if not xi_toml_path.exists():
                    all_valid = False
                    yield f"data: {json.dumps({'event': 'result', 'step': 'xi_toml_syntax', 'valid': False, 'error': 'xi.toml not found'})}\n\n"
                    yield f"data: {json.dumps({'event': 'done', 'valid': False})}\n\n"
                    return
                
                try:
                    with open(xi_toml_path, "rb") as f:
                        tomli.load(f)
                    yield f"data: {json.dumps({'event': 'result', 'step': 'xi_toml_syntax', 'valid': True, 'error': None})}\n\n"
                except tomli.TOMLDecodeError as e:
                    all_valid = False
                    error_msg = f"Syntax error at line {e.lineno}: {e.msg}" if hasattr(e, 'lineno') else str(e)
                    yield f"data: {json.dumps({'event': 'result', 'step': 'xi_toml_syntax', 'valid': False, 'error': error_msg})}\n\n"
                    yield f"data: {json.dumps({'event': 'done', 'valid': False})}\n\n"
                    return
                
                yield f"data: {json.dumps({'event': 'checking', 'step': 'project_info', 'message': 'Checking project information'})}\n\n"
                await asyncio.sleep(0.1)
                
                try:
                    from .loader import get_xi_config
                    config = get_xi_config()
                    
                    if not config.project.name:
                        all_valid = False
                        yield f"data: {json.dumps({'event': 'result', 'step': 'project_info', 'valid': False, 'error': 'Project name not configured'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': 'result', 'step': 'project_info', 'valid': True, 'error': None, 'data': {'name': config.project.name, 'version': config.project.version, 'backend': config.project.backend}})}\n\n"
                except Exception as e:
                    all_valid = False
                    yield f"data: {json.dumps({'event': 'result', 'step': 'project_info', 'valid': False, 'error': str(e)})}\n\n"
                
                yield f"data: {json.dumps({'event': 'checking', 'step': 'subcommands', 'message': 'Checking subcommand configurations'})}\n\n"
                await asyncio.sleep(0.1)
                
                try:
                    if config.project.commands and config.project.commands.enabled:
                        commands_dir = xi_dir / "commands"
                        missing_commands = []
                        
                        for cmd_name in config.project.commands.enabled:
                            cmd_file = commands_dir / f"{cmd_name}.toml"
                            if not cmd_file.exists():
                                missing_commands.append(cmd_name)
                            else:
                                try:
                                    with open(cmd_file, "rb") as f:
                                        tomli.load(f)
                                except tomli.TOMLDecodeError as e:
                                    all_valid = False
                                    yield f"data: {json.dumps({'event': 'result', 'step': 'subcommands', 'valid': False, 'error': f'{cmd_name}.toml: Syntax error at line {e.lineno}' if hasattr(e, 'lineno') else f'{cmd_name}.toml: {str(e)}'})}\n\n"
                                    continue
                        
                        if missing_commands:
                            all_valid = False
                            yield f"data: {json.dumps({'event': 'result', 'step': 'subcommands', 'valid': False, 'error': f'Missing command files: {missing_commands}'})}\n\n"
                        else:
                            yield f"data: {json.dumps({'event': 'result', 'step': 'subcommands', 'valid': True, 'error': None, 'data': {'enabled': config.project.commands.enabled}})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': 'result', 'step': 'subcommands', 'valid': True, 'error': 'No subcommands configured'})}\n\n"
                except Exception as e:
                    all_valid = False
                    yield f"data: {json.dumps({'event': 'result', 'step': 'subcommands', 'valid': False, 'error': str(e)})}\n\n"
                
                yield f"data: {json.dumps({'event': 'checking', 'step': 'python_env', 'message': 'Checking Python environment'})}\n\n"
                await asyncio.sleep(0.1)
                
                try:
                    python_path = config.environment.python_path
                    python_resolved = shutil.which(python_path)
                    
                    if not python_resolved:
                        all_valid = False
                        yield f"data: {json.dumps({'event': 'result', 'step': 'python_env', 'valid': False, 'error': f'Python not found: {python_path}'})}\n\n"
                    else:
                        venv_info = None
                        if config.environment.virtualenv and config.environment.virtualenv.enabled:
                            venv_path = self.root_dir / config.environment.virtualenv.path
                            if venv_path.exists():
                                venv_info = {"path": str(venv_path), "exists": True}
                            else:
                                if config.environment.virtualenv.create_if_missing:
                                    venv_info = {"path": str(venv_path), "exists": False, "will_create": True}
                                else:
                                    all_valid = False
                                    yield f"data: {json.dumps({'event': 'result', 'step': 'python_env', 'valid': False, 'error': f'Virtual environment not found: {venv_path}'})}\n\n"
                                    return
                        
                        result = subprocess.run(
                            [python_path, "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        version = result.stdout.strip() or result.stderr.strip()
                        
                        yield f"data: {json.dumps({'event': 'result', 'step': 'python_env', 'valid': True, 'error': None, 'data': {'path': python_resolved, 'version': version, 'venv': venv_info}})}\n\n"
                except Exception as e:
                    all_valid = False
                    yield f"data: {json.dumps({'event': 'result', 'step': 'python_env', 'valid': False, 'error': str(e)})}\n\n"
                
                yield f"data: {json.dumps({'event': 'checking', 'step': 'ui_config', 'message': 'Checking UI configuration'})}\n\n"
                await asyncio.sleep(0.1)
                
                try:
                    ui_data = {
                        "theme": config.ui.theme,
                        "language": config.ui.language,
                        "sidebar_collapsed": config.ui.sidebar_collapsed
                    }
                    valid_themes = ["light", "dark", "system"]
                    if config.ui.theme not in valid_themes:
                        all_valid = False
                        yield f"data: {json.dumps({'event': 'result', 'step': 'ui_config', 'valid': False, 'error': f'Invalid theme: {config.ui.theme}. Must be one of {valid_themes}'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': 'result', 'step': 'ui_config', 'valid': True, 'error': None, 'data': ui_data})}\n\n"
                except Exception as e:
                    all_valid = False
                    yield f"data: {json.dumps({'event': 'result', 'step': 'ui_config', 'valid': False, 'error': str(e)})}\n\n"
                
                yield f"data: {json.dumps({'event': 'done', 'valid': all_valid})}\n\n"
            
            return StreamingResponse(
                generate_validation_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        @self.app.get("/v1/xi/setup-environment")
        async def setup_environment():
            from fastapi.responses import StreamingResponse
            import subprocess
            import shutil
            import json
            import asyncio

            async def generate_setup_events():
                all_valid = True

                yield f"data: {json.dumps({'event': 'checking', 'step': 'venv_create', 'message': 'Checking virtual environment'})}\n\n"
                await asyncio.sleep(0.1)

                try:
                    from .loader import get_xi_config
                    config = get_xi_config()

                    venv_enabled = config.environment.virtualenv and config.environment.virtualenv.enabled
                    venv_path = self.root_dir / config.environment.virtualenv.path if venv_enabled else None

                    if venv_enabled and venv_path:
                        if not venv_path.exists():
                            yield f"data: {json.dumps({'event': 'checking', 'step': 'venv_create', 'message': 'Creating virtual environment...'})}\n\n"
                            await asyncio.sleep(0.1)

                            create_result = subprocess.run(
                                ["python", "-m", "venv", str(venv_path)],
                                capture_output=True,
                                text=True,
                                timeout=300
                            )

                            if create_result.returncode != 0:
                                all_valid = False
                                yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': False, 'error': f'Failed to create venv: {create_result.stderr}'})}\n\n"
                            else:
                                yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': True, 'error': None, 'data': {'path': str(venv_path)}})}\n\n"
                        else:
                            yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': True, 'error': None, 'data': {'path': str(venv_path), 'exists': True}})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': True, 'error': None, 'data': {'message': 'Virtual environment not configured'}})}\n\n"

                except Exception as e:
                    all_valid = False
                    yield f"data: {json.dumps({'event': 'result', 'step': 'venv_create', 'valid': False, 'error': str(e)})}\n\n"

                yield f"data: {json.dumps({'event': 'checking', 'step': 'install_deps', 'message': 'Installing dependencies...'})}\n\n"
                await asyncio.sleep(0.1)

                try:
                    from .loader import get_xi_config
                    config = get_xi_config()

                    if config.environment.requirements:
                        for req in config.environment.requirements:
                            if req.required:
                                req_path = self.root_dir / req.path
                                if req_path.exists():
                                    yield f"data: {json.dumps({'event': 'checking', 'step': 'install_deps', 'message': f'Installing {req.name}...'})}\n\n"

                                    python_path = config.environment.python_path
                                    if config.environment.virtualenv and config.environment.virtualenv.enabled:
                                        venv_python = self.root_dir / config.environment.virtualenv.path / "Scripts" / "python.exe"
                                        if venv_python.exists():
                                            python_path = str(venv_python)

                                    install_result = subprocess.run(
                                        [python_path, "-m", "pip", "install", "-r", str(req_path)],
                                        capture_output=True,
                                        text=True,
                                        timeout=600
                                    )

                                    if install_result.returncode != 0:
                                        yield f"data: {json.dumps({'event': 'result', 'step': 'install_deps', 'valid': True, 'error': f'{req.name}: {install_result.stderr[:200]} (ignored)'})}\n\n"
                                    else:
                                        yield f"data: {json.dumps({'event': 'result', 'step': 'install_deps', 'valid': True, 'error': None, 'data': {'package': req.name, 'path': req.path}})}\n\n"
                                    await asyncio.sleep(0.1)
                                else:
                                    yield f"data: {json.dumps({'event': 'result', 'step': 'install_deps', 'valid': True, 'error': f'{req.name}: file not found at {req.path} (ignored)'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': 'result', 'step': 'install_deps', 'valid': True, 'error': None, 'data': {'message': 'No requirements configured'}})}\n\n"

                except Exception as e:
                    yield f"data: {json.dumps({'event': 'result', 'step': 'install_deps', 'valid': True, 'error': f'{str(e)} (ignored)'})}\n\n"

                yield f"data: {json.dumps({'event': 'checking', 'step': 'verify_setup', 'message': 'Verifying installation...'})}\n\n"
                await asyncio.sleep(0.1)

                try:
                    from .loader import get_xi_config
                    config = get_xi_config()

                    python_path = config.environment.python_path
                    if config.environment.virtualenv and config.environment.virtualenv.enabled:
                        venv_python = self.root_dir / config.environment.virtualenv.path / "Scripts" / "python.exe"
                        if venv_python.exists():
                            python_path = str(venv_python)

                    version_result = subprocess.run(
                        [python_path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    version = version_result.stdout.strip() or version_result.stderr.strip()

                    yield f"data: {json.dumps({'event': 'result', 'step': 'verify_setup', 'valid': True, 'error': None, 'data': {'version': version}})}\n\n"

                except Exception as e:
                    all_valid = False
                    yield f"data: {json.dumps({'event': 'result', 'step': 'verify_setup', 'valid': False, 'error': str(e)})}\n\n"

                yield f"data: {json.dumps({'event': 'done', 'valid': all_valid})}\n\n"

            return StreamingResponse(
                generate_setup_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: dict):
            self._request_count += 1
            return await self._proxy_inference_request("/v1/chat/completions", request)

        @self.app.post("/v1/embeddings")
        async def create_embeddings(request: dict):
            self._request_count += 1
            return await self._proxy_inference_request("/v1/embeddings", request)

        @self.app.post("/v1/images/generations")
        async def generate_images(request: dict):
            self._request_count += 1
            return await self._proxy_inference_request("/v1/images/generations", request)

        @self.app.get("/v1/tools/list")
        async def list_tools(category: Optional[str] = None):
            self._request_count += 1
            tools = await self._list_mcp_tools(category)
            return {"tools": tools, "total": len(tools)}

        @self.app.post("/v1/tools/execute")
        async def execute_tool(request: dict):
            self._request_count += 1
            result = await self._execute_mcp_tool(request)
            return result

        @self.app.get("/v1/xi/config")
        async def get_xi_config():
            self._request_count += 1
            return await self._get_xi_config()

        @self.app.get("/v1/xi/paths")
        async def get_xi_paths():
            self._request_count += 1
            return await self._get_xi_paths()

        @self.app.get("/v1/xi/commands")
        async def get_xi_commands():
            self._request_count += 1
            return await self._get_xi_commands()

        @self.app.get("/v1/xi/commands/{command_name}")
        async def get_xi_command(command_name: str):
            self._request_count += 1
            return await self._get_xi_command(command_name)

        @self.app.get("/v1/xi/commands/{command_name}/schema")
        async def get_command_schema(command_name: str):
            self._request_count += 1
            return await self._get_command_schema(command_name)

        @self.app.get("/v1/xi/commands/{command_name}/options/{parameter_name}")
        async def get_command_options(command_name: str, parameter_name: str):
            self._request_count += 1
            return await self._get_command_options(command_name, parameter_name)

        @self.app.get("/v1/notifications")
        async def list_notifications():
            self._request_count += 1
            return await self._list_notifications()

        @self.app.post("/v1/notifications")
        async def create_notification(request: dict):
            self._request_count += 1
            return await self._create_notification(request)

        @self.app.post("/v1/notifications/{notification_id}/read")
        async def mark_notification_read(notification_id: str):
            self._request_count += 1
            return await self._mark_notification_read(notification_id)

        @self.app.delete("/v1/notifications/{notification_id}")
        async def delete_notification(notification_id: str):
            self._request_count += 1
            return await self._delete_notification(notification_id)

    def _setup_websockets(self) -> None:

        @self.app.websocket("/ws/logs/{run_id}")
        async def stream_logs(websocket: WebSocket, run_id: str):
            await websocket.accept()
            self.logger.info(f"WebSocket connected for run: {run_id}", event="xi.ws.connect")

            try:
                async for entry in self.executor.get_output_stream(run_id):
                    await websocket.send_json({
                        "timestamp": entry.timestamp,
                        "level": entry.level,
                        "message": entry.message,
                        "source": entry.source,
                        "run_id": entry.run_id
                    })
            except WebSocketDisconnect:
                self.logger.info(f"WebSocket disconnected for run: {run_id}", event="xi.ws.disconnect")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}", event="xi.ws.error")
                await websocket.close()

        @self.app.websocket("/ws/stats")
        async def stream_stats(websocket: WebSocket):
            await websocket.accept()
            self.logger.info("Stats WebSocket connected", event="xi.ws.stats.connect")

            try:
                while True:
                    stats = await self._collect_system_stats()
                    from dataclasses import asdict
                    await websocket.send_json(asdict(stats))
                    await asyncio.sleep(2.0)
            except WebSocketDisconnect:
                self.logger.info("Stats WebSocket disconnected", event="xi.ws.stats.disconnect")
            except Exception as e:
                self.logger.error(f"Stats WebSocket error: {e}", event="xi.ws.stats.error")

    def _get_nvidia_gpus(self) -> List[Dict[str, Any]]:
        gpus = []
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()

            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)

                temp = 0.0
                power = 0.0
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    pass

                gpus.append({
                    "vendor": "nvidia",
                    "name": name if isinstance(name, str) else name.decode(),
                    "utilization": util.gpu,
                    "memory_used_gb": mem.used / (1024**3),
                    "memory_total_gb": mem.total / (1024**3),
                    "temperature": temp,
                    "power_draw": power,
                })

            pynvml.nvmlShutdown()
        except Exception:
            pass

        return gpus

    def _get_amd_gpus(self) -> List[Dict[str, Any]]:
        gpus = []

        try:
            import pyamdgpuinfo
            gpu_count = pyamdgpuinfo.detect_gpus()

            for i in range(gpu_count):
                gpu = pyamdgpuinfo.get_gpu(i)
                gpus.append({
                    "vendor": "amd",
                    "name": gpu.name if hasattr(gpu, 'name') else f"AMD GPU {i}",
                    "utilization": gpu.load * 100 if hasattr(gpu, 'load') else 0.0,
                    "memory_used_gb": gpu.memory_used / (1024**3) if hasattr(gpu, 'memory_used') else 0.0,
                    "memory_total_gb": gpu.memory_total / (1024**3) if hasattr(gpu, 'memory_total') else 0.0,
                    "temperature": gpu.temperature if hasattr(gpu, 'temperature') else 0.0,
                    "power_draw": gpu.power if hasattr(gpu, 'power') else 0.0,
                })
        except ImportError:
            pass

        if not gpus:
            try:
                import subprocess
                result = subprocess.run(
                    ["rocm-smi", "--showuse", "--showmeminfo", "--showtemp", "--showpower", "--json"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    if "card" in data:
                        for card_id, card_data in data["card"].items():
                            gpus.append({
                                "vendor": "amd",
                                "name": card_data.get("Card series", f"AMD GPU {card_id}"),
                                "utilization": float(card_data.get("GPU use (%)", 0)),
                                "memory_used_gb": float(card_data.get("GPU memory used (MB)", 0)) / 1024,
                                "memory_total_gb": float(card_data.get("GPU memory total (MB)", 0)) / 1024,
                                "temperature": float(card_data.get("Temperature (Sensor edge) (C)", 0)),
                                "power_draw": float(card_data.get("Average Graphics Package Power (W)", 0)),
                            })
            except Exception:
                pass

        return gpus

    def _get_intel_gpus(self) -> List[Dict[str, Any]]:
        gpus = []

        try:
            import subprocess
            result = subprocess.run(
                ["xpu-smi", "discovery", "-l"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                import re
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if 'Device' in line or 'GPU' in line:
                        device_id = i
                        stats_result = subprocess.run(
                            ["xpu-smi", "stats", "-d", str(device_id), "-m", "0,1,2,3"],
                            capture_output=True, text=True, timeout=5
                        )
                        if stats_result.returncode == 0:
                            util = 0.0
                            mem_used = 0.0
                            mem_total = 0.0
                            temp = 0.0
                            power = 0.0

                            for stat_line in stats_result.stdout.split('\n'):
                                if 'GPU Utilization' in stat_line:
                                    match = re.search(r'(\d+\.?\d*)', stat_line)
                                    if match:
                                        util = float(match.group(1))
                                elif 'Memory Used' in stat_line:
                                    match = re.search(r'(\d+\.?\d*)', stat_line)
                                    if match:
                                        mem_used = float(match.group(1)) / 1024
                                elif 'Memory Total' in stat_line:
                                    match = re.search(r'(\d+\.?\d*)', stat_line)
                                    if match:
                                        mem_total = float(match.group(1)) / 1024
                                elif 'Temperature' in stat_line:
                                    match = re.search(r'(\d+\.?\d*)', stat_line)
                                    if match:
                                        temp = float(match.group(1))
                                elif 'Power' in stat_line:
                                    match = re.search(r'(\d+\.?\d*)', stat_line)
                                    if match:
                                        power = float(match.group(1))

                            gpus.append({
                                "vendor": "intel",
                                "name": f"Intel GPU {device_id}",
                                "utilization": util,
                                "memory_used_gb": mem_used,
                                "memory_total_gb": mem_total,
                                "temperature": temp,
                                "power_draw": power,
                            })
        except Exception:
            pass

        if not gpus:
            try:
                import subprocess
                result = subprocess.run(
                    ["cat", "/sys/class/drm/card*/device/gpu_busy_percent"],
                    capture_output=True, text=True, timeout=2, shell=True
                )
                if result.returncode == 0:
                    for i, line in enumerate(result.stdout.strip().split('\n')):
                        if line.strip():
                            gpus.append({
                                "vendor": "intel",
                                "name": f"Intel GPU {i}",
                                "utilization": float(line.strip()),
                                "memory_used_gb": 0.0,
                                "memory_total_gb": 0.0,
                                "temperature": 0.0,
                                "power_draw": 0.0,
                            })
            except Exception:
                pass

        return gpus

    async def _collect_system_stats(self) -> XiSystemStats:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        all_gpus = []
        all_gpus.extend(self._get_nvidia_gpus())
        all_gpus.extend(self._get_amd_gpus())
        all_gpus.extend(self._get_intel_gpus())

        gpu_utilization = [g["utilization"] for g in all_gpus]
        gpu_memory_used = [g["memory_used_gb"] for g in all_gpus]
        gpu_memory_total = [g["memory_total_gb"] for g in all_gpus]
        gpu_vendors = [g["vendor"] for g in all_gpus]
        gpu_names = [g["name"] for g in all_gpus]
        gpu_temperatures = [g["temperature"] for g in all_gpus]
        gpu_power_draw = [g["power_draw"] for g in all_gpus]

        uptime = (datetime.now() - self._start_time).total_seconds()

        return XiSystemStats(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            gpu_count=len(all_gpus),
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            gpu_vendors=gpu_vendors,
            gpu_names=gpu_names,
            gpu_temperatures=gpu_temperatures,
            gpu_power_draw=gpu_power_draw,
            uptime_seconds=uptime,
            request_count=self._request_count,
            qps=self._request_count / max(uptime, 1.0)
        )

    async def _list_available_models(self) -> List[Dict[str, Any]]:
        models = []

        config_dir = self.root_dir / "configs"
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                model_id = config_file.stem
                models.append({
                    "id": f"piscesl1-{model_id.lower()}",
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "piscesl1"
                })

        if not models:
            default_sizes = ["0.5B", "1B", "7B", "14B", "72B", "671B", "1T"]
            for size in default_sizes:
                models.append({
                    "id": f"piscesl1-{size.lower()}",
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "piscesl1"
                })

        return models

    async def _proxy_inference_request(self, endpoint: str, request: dict) -> dict:
        import httpx

        inference_url = os.environ.get("PISCESLX_INFERENCE_URL", "http://127.0.0.1:8000")

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{inference_url}{endpoint}",
                    json=request
                )
                return response.json()
            except Exception as e:
                self.logger.error(f"Inference proxy error: {e}", event="xi.proxy.error")
                return {"error": str(e)}

    async def _list_mcp_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            from opss.mcp.mcps import POPSSToolRegistry
            registry = POPSSToolRegistry.get_instance()
            tools = registry.list_tools()

            if category:
                tools = [t for t in tools if t.get("category") == category]

            return tools
        except Exception as e:
            self.logger.error(f"MCP tools list error: {e}", event="xi.mcp.error")
            return []

    async def _execute_mcp_tool(self, request: dict) -> Dict[str, Any]:
        try:
            from opss.mcp.mcps import POPSSToolRegistry
            registry = POPSSToolRegistry.get_instance()

            tool_name = request.get("tool")
            arguments = request.get("arguments", {})

            result = await registry.execute_tool(tool_name, arguments)
            return {"success": True, "result": result}
        except Exception as e:
            self.logger.error(f"MCP tool execute error: {e}", event="xi.mcp.exec_error")
            return {"success": False, "error": str(e)}

    async def _get_xi_config(self) -> Dict[str, Any]:
        try:
            from .loader import get_xi_config
            config = get_xi_config()
            return config.to_dict()
        except Exception as e:
            self.logger.error(f"Failed to get Xi config: {e}", event="xi.config.error")
            return {"error": str(e)}

    async def _get_xi_paths(self) -> Dict[str, Any]:
        try:
            from .loader import get_xi_config
            config = get_xi_config()
            resolved = config.get_resolved_paths()
            return {
                "root": str(config.project_root),
                "models": str(resolved.get("models", config.paths.models)),
                "checkpoints": str(resolved.get("checkpoints", config.paths.checkpoints)),
                "data": str(resolved.get("data", config.paths.data)),
                "outputs": str(resolved.get("outputs", config.paths.outputs)),
                "logs": str(resolved.get("logs", config.paths.logs)),
                "cache": str(resolved.get("cache", config.paths.cache)),
                "temp": str(resolved.get("temp", config.paths.temp)),
                "configs": str(resolved.get("configs", config.paths.configs)),
            }
        except Exception as e:
            self.logger.error(f"Failed to get Xi paths: {e}", event="xi.paths.error")
            return {"error": str(e)}

    async def _get_xi_commands(self) -> Dict[str, Any]:
        try:
            from .loader import get_xi_config
            config = get_xi_config()
            commands = {}
            for name, cmd in config.commands.items():
                commands[name] = {
                    "executable": cmd.executable,
                    "script": cmd.script,
                    "args": cmd.args,
                    "env": cmd.env,
                    "cwd": cmd.cwd,
                    "timeout": cmd.timeout,
                    "background": cmd.background,
                    "defaults": cmd.defaults,
                }
            return {"commands": commands, "total": len(commands)}
        except Exception as e:
            self.logger.error(f"Failed to get Xi commands: {e}", event="xi.commands.error")
            return {"error": str(e)}

    async def _get_xi_command(self, command_name: str) -> Dict[str, Any]:
        try:
            from .loader import get_xi_config
            config = get_xi_config()
            cmd = config.commands.get(command_name)
            if not cmd:
                return {"error": f"Command '{command_name}' not found"}
            
            result = {
                "name": command_name,
                "executable": cmd.executable,
                "script": cmd.script,
                "args": cmd.args,
                "env": cmd.env,
                "cwd": cmd.cwd,
                "timeout": cmd.timeout,
                "background": cmd.background,
                "defaults": cmd.defaults,
            }
            
            if cmd.schema:
                result["schema"] = {
                    "description": cmd.schema.description,
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "description": p.description,
                            "required": p.required,
                            "default": p.default,
                            "options": p.options,
                            "min": p.min,
                            "max": p.max,
                            "source": p.source,
                            "source_type": p.source_type,
                            "filter": p.filter,
                        }
                        for p in cmd.schema.parameters
                    ],
                }
            
            return result
        except Exception as e:
            self.logger.error(f"Failed to get Xi command: {e}", event="xi.command.error")
            return {"error": str(e)}

    async def _get_command_schema(self, command_name: str) -> Dict[str, Any]:
        try:
            from .loader import get_xi_config
            from .config import XiTabSchema
            config = get_xi_config()
            cmd = config.commands.get(command_name)
            if not cmd:
                return {
                    "command": command_name,
                    "available": False,
                    "unavailable_reason": f"Command '{command_name}' not found",
                }
            
            if not cmd.schema:
                return {
                    "command": command_name,
                    "available": False,
                    "unavailable_reason": f"No schema defined for command '{command_name}'",
                }
            
            tabs = []
            for t in cmd.schema.tabs:
                tabs.append({
                    "name": t.name,
                    "label": t.label,
                    "available": t.available,
                    "unavailable_reason": t.unavailable_reason,
                })
            
            if not tabs:
                tabs = [
                    {"name": "basic", "label": "Basic", "available": True, "unavailable_reason": ""},
                ]
            
            parameters = []
            command_unavailable_reason = ""
            
            for p in cmd.schema.parameters:
                available = p.available
                unavailable_reason = p.unavailable_reason
                
                if p.source and p.source_type == "directory":
                    resolved_source = self._resolve_source_path(p.source, config)
                    if not resolved_source.exists():
                        available = False
                        unavailable_reason = f"Source directory not found: {resolved_source}"
                        if p.required:
                            command_unavailable_reason = f"Required configuration directory not found: {resolved_source}"
                    else:
                        options = self._list_directory_options(resolved_source, p.filter)
                        if not options and p.required:
                            available = False
                            unavailable_reason = f"No options available in: {resolved_source}"
                            command_unavailable_reason = f"Required configuration directory is empty: {resolved_source}"
                
                param_dict = {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    "available": available,
                    "unavailable_reason": unavailable_reason,
                    "tab": p.tab,
                }
                if p.options:
                    param_dict["options"] = p.options
                if p.min is not None:
                    param_dict["min"] = p.min
                if p.max is not None:
                    param_dict["max"] = p.max
                if p.source:
                    param_dict["source"] = p.source
                if p.source_type:
                    param_dict["source_type"] = p.source_type
                if p.filter:
                    param_dict["filter"] = p.filter
                parameters.append(param_dict)
            
            tab_availability = {}
            for tab in tabs:
                tab_params = [p for p in parameters if p.get("tab", "basic") == tab["name"]]
                required_unavailable = [p for p in tab_params if p["required"] and not p["available"]]
                if required_unavailable:
                    tab_availability[tab["name"]] = {
                        "available": False,
                        "unavailable_reason": f"Required parameters unavailable: {', '.join(p['name'] for p in required_unavailable)}",
                    }
                else:
                    tab_availability[tab["name"]] = {
                        "available": True,
                        "unavailable_reason": "",
                    }
            
            tabs = [
                {
                    **tab,
                    "available": tab_availability.get(tab["name"], {}).get("available", True),
                    "unavailable_reason": tab_availability.get(tab["name"], {}).get("unavailable_reason", ""),
                }
                for tab in tabs
            ]
            
            all_required_available = all(
                p["available"] for p in parameters if p["required"]
            )
            
            is_available = all_required_available and not command_unavailable_reason
            unavailable_reason = command_unavailable_reason if command_unavailable_reason else (
                "" if all_required_available else "Some required parameters are unavailable"
            )
            
            return {
                "command": command_name,
                "description": cmd.schema.description,
                "available": is_available,
                "unavailable_reason": unavailable_reason,
                "tabs": tabs if is_available else [],
                "parameters": parameters if is_available else [],
            }
        except Exception as e:
            self.logger.error(f"Failed to get command schema: {e}", event="xi.schema.error")
            return {
                "command": command_name,
                "available": False,
                "unavailable_reason": str(e),
            }

    async def _get_command_options(self, command_name: str, parameter_name: str) -> Dict[str, Any]:
        try:
            from .loader import get_xi_config
            config = get_xi_config()
            cmd = config.commands.get(command_name)
            if not cmd:
                return {"error": f"Command '{command_name}' not found"}
            
            if not cmd.schema:
                return {"error": f"No schema defined for command '{command_name}'"}
            
            parameter = None
            for p in cmd.schema.parameters:
                if p.name == parameter_name:
                    parameter = p
                    break
            
            if not parameter:
                return {"error": f"Parameter '{parameter_name}' not found in command '{command_name}'"}
            
            if parameter.options:
                return {
                    "parameter": parameter_name,
                    "options": [{"value": opt, "label": opt} for opt in parameter.options],
                }
            
            if parameter.source and parameter.source_type == "directory":
                resolved_source = self._resolve_source_path(parameter.source, config)
                options = self._list_directory_options(
                    resolved_source, 
                    parameter.filter
                )
                return {
                    "parameter": parameter_name,
                    "options": options,
                    "source": str(resolved_source),
                }
            
            return {
                "parameter": parameter_name,
                "options": [],
                "message": "No dynamic options configured for this parameter",
            }
        except Exception as e:
            self.logger.error(f"Failed to get command options: {e}", event="xi.options.error")
            return {"error": str(e)}

    def _resolve_source_path(self, source: str, config) -> Path:
        """Resolve source path with variable substitution."""
        import re
        
        resolved = source
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_path = match.group(1)
            parts = var_path.split(".")
            
            if len(parts) >= 2:
                section = parts[0]
                key = parts[1]
                
                if section == "paths":
                    value = getattr(config.paths, key, None)
                    if value:
                        path = Path(value)
                        if not path.is_absolute():
                            path = config.project_root / path
                        return str(path)
                elif section == "env":
                    env_key = ".".join(parts[1:])
                    return os.environ.get(env_key, "")
            
            return match.group(0)
        
        resolved = re.sub(pattern, replace_var, source)
        path = Path(resolved)
        
        if not path.is_absolute():
            path = config.project_root / path
        
        return path

    def _list_directory_options(self, directory: Path, filter_pattern: str = "") -> List[Dict[str, str]]:
        """List options from a directory."""
        options = []
        
        if not directory.exists():
            return options
        
        if filter_pattern:
            import fnmatch
            for item in directory.iterdir():
                if item.is_file() and fnmatch.fnmatch(item.name, filter_pattern):
                    options.append({
                        "value": item.stem,
                        "label": item.name,
                    })
                elif item.is_dir():
                    sub_options = self._list_directory_options(item, filter_pattern)
                    options.extend(sub_options)
        else:
            for item in directory.iterdir():
                if item.is_dir():
                    options.append({
                        "value": item.name,
                        "label": item.name,
                    })
                else:
                    options.append({
                        "value": item.stem,
                        "label": item.name,
                    })
        
        return sorted(options, key=lambda x: x["label"])

    async def _list_notifications(self) -> Dict[str, Any]:
        try:
            notifications = self.notification_manager.list_notifications()
            result = [
                {
                    "id": n.id,
                    "type": n.type,
                    "title": n.title,
                    "message": n.message,
                    "time": n.time.isoformat(),
                    "read": n.read,
                }
                for n in notifications
            ]
            return {"notifications": result, "total": len(result)}
        except Exception as e:
            self.logger.error(f"Failed to list notifications: {e}", event="xi.notifications.error")
            return {"error": str(e)}

    async def _create_notification(self, request: dict) -> Dict[str, Any]:
        try:
            notification = self.notification_manager.create_notification(
                notification_type=request.get("type", "info"),
                title=request.get("title", ""),
                message=request.get("message", ""),
                metadata=request.get("metadata"),
            )
            return {
                "id": notification.id,
                "type": notification.type,
                "title": notification.title,
                "message": notification.message,
                "time": notification.time.isoformat(),
                "read": notification.read,
            }
        except Exception as e:
            self.logger.error(f"Failed to create notification: {e}", event="xi.notification.create_error")
            return {"error": str(e)}

    async def _mark_notification_read(self, notification_id: str) -> Dict[str, Any]:
        try:
            success = self.notification_manager.mark_read(notification_id)
            return {"success": success}
        except Exception as e:
            self.logger.error(f"Failed to mark notification read: {e}", event="xi.notification.read_error")
            return {"error": str(e)}

    async def _delete_notification(self, notification_id: str) -> Dict[str, Any]:
        try:
            success = self.notification_manager.delete_notification(notification_id)
            return {"success": success}
        except Exception as e:
            self.logger.error(f"Failed to delete notification: {e}", event="xi.notification.delete_error")
            return {"error": str(e)}

    def run(self, host: str = "127.0.0.1") -> None:
        self.logger.info(
            f"Starting Xi server on {host}:{self.port}",
            event="xi.server.start"
        )

        uvicorn.run(
            self.app,
            host=host,
            port=self.port,
            log_level="info"
        )


_server_instance: Optional[XiServer] = None


def get_app() -> FastAPI:
    global _server_instance
    if _server_instance is None:
        _server_instance = XiServer()
    return _server_instance.app


app = get_app()
