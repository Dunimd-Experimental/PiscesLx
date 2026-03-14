#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
FastAPI Server for PLxS (PLx Studio) Backend.

This module provides the main FastAPI server that exposes REST and WebSocket
endpoints for the PLx Studio frontend. It runs on port 3140 and serves as
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
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from utils.plxs.executor import PiscesLxPlxsExecutor
from utils.plxs.types import (
    PiscesLxPlxsCommand,
    PiscesLxPlxsCommandRequest,
    PiscesLxPlxsCommandResponse,
    PiscesLxPlxsRunStatus,
    PiscesLxPlxsRunInfo,
    PiscesLxPlxsSystemStats,
    PiscesLxPlxsLogLevel,
    PiscesLxPlxsLogEntry,
    PiscesLxPlxsControlRequest,
    PiscesLxPlxsControlResponse,
)


class PiscesLxPlxsServer:
    """
    FastAPI server for PLx Studio backend.
    
    This class encapsulates the FastAPI application and provides all
    endpoints for the PLx Studio frontend to interact with manage.py
    commands and system resources.
    
    Attributes:
        app (FastAPI): The FastAPI application instance.
        executor (PiscesLxPlxsExecutor): Command executor for manage.py.
        logger (PiscesLxLogger): Logger instance for server operations.
        port (int): The port number for the server (default: 3140).
        _start_time (datetime): Server start time for uptime calculation.
    """

    def __init__(self, port: int = 3140, root_dir: Optional[str] = None):
        """
        Initialize the PLxS server.
        
        Args:
            port: The port number to run the server on.
            root_dir: Optional root directory for the project.
        """
        self.port = port
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.logger = PiscesLxLogger(
            "PiscesLx.Plxs.Server",
            file_path=get_log_file("PiscesLx.Plxs.Server"),
            enable_file=True
        )
        self.executor = PiscesLxPlxsExecutor(str(self.root_dir))
        self._start_time = datetime.now()
        self._request_count = 0
        
        self.app = FastAPI(
            title="PLx Studio API",
            description="Backend API for PLx Studio - PiscesL1 Workstation",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self._setup_middleware()
        self._setup_routes()
        self._setup_websockets()

    def _setup_middleware(self) -> None:
        """Configure CORS middleware for frontend access."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Set up all REST API routes."""
        
        @self.app.get("/healthz")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/stats")
        async def get_stats():
            """Get system resource statistics."""
            self._request_count += 1
            stats = await self._collect_system_stats()
            return stats
        
        @self.app.get("/v1/models")
        async def list_models():
            """List available models."""
            self._request_count += 1
            models = await self._list_available_models()
            return {"data": models, "object": "list"}
        
        @self.app.get("/v1/runs")
        async def list_runs():
            """List all runs."""
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
            """Start a new run."""
            self._request_count += 1
            
            try:
                cmd_request = PiscesLxPlxsCommandRequest(
                    command=PiscesLxPlxsCommand(request.get("command", "train")),
                    args=request.get("args", {}),
                    run_id=request.get("run_id"),
                    run_name=request.get("run_name"),
                    background=request.get("background", True)
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
            
            response = await self.executor.execute(cmd_request)
            return response.__dict__
        
        @self.app.get("/v1/runs/{run_id}")
        async def get_run(run_id: str):
            """Get run details."""
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
            """Control a run (pause/resume/cancel/kill)."""
            self._request_count += 1
            action = request.get("action")
            if not action:
                raise HTTPException(status_code=400, detail="Action is required")
            
            response = await self.executor.control(run_id, action)
            return response.__dict__
        
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: dict):
            """Chat completion endpoint (proxy to inference server)."""
            self._request_count += 1
            return await self._proxy_inference_request("/v1/chat/completions", request)
        
        @self.app.post("/v1/embeddings")
        async def create_embeddings(request: dict):
            """Embedding generation endpoint (proxy to inference server)."""
            self._request_count += 1
            return await self._proxy_inference_request("/v1/embeddings", request)
        
        @self.app.post("/v1/images/generations")
        async def generate_images(request: dict):
            """Image generation endpoint (proxy to inference server)."""
            self._request_count += 1
            return await self._proxy_inference_request("/v1/images/generations", request)
        
        @self.app.get("/v1/tools/list")
        async def list_tools(category: Optional[str] = None):
            """List available MCP tools."""
            self._request_count += 1
            tools = await self._list_mcp_tools(category)
            return {"tools": tools, "total": len(tools)}
        
        @self.app.post("/v1/tools/execute")
        async def execute_tool(request: dict):
            """Execute an MCP tool."""
            self._request_count += 1
            result = await self._execute_mcp_tool(request)
            return result

    def _setup_websockets(self) -> None:
        """Set up WebSocket endpoints for real-time streaming."""
        
        @self.app.websocket("/ws/logs/{run_id}")
        async def stream_logs(websocket: WebSocket, run_id: str):
            """Stream logs for a specific run."""
            await websocket.accept()
            self.logger.info(f"WebSocket connected for run: {run_id}", event="plxs.ws.connect")
            
            try:
                async for entry in self.executor.get_output_stream(run_id):
                    await websocket.send_json({
                        "timestamp": entry.timestamp,
                        "level": entry.level.value,
                        "message": entry.message,
                        "source": entry.source,
                        "run_id": entry.run_id
                    })
            except WebSocketDisconnect:
                self.logger.info(f"WebSocket disconnected for run: {run_id}", event="plxs.ws.disconnect")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}", event="plxs.ws.error")
                await websocket.close()
        
        @self.app.websocket("/ws/stats")
        async def stream_stats(websocket: WebSocket):
            """Stream system statistics in real-time."""
            await websocket.accept()
            self.logger.info("Stats WebSocket connected", event="plxs.ws.stats.connect")
            
            try:
                while True:
                    stats = await self._collect_system_stats()
                    await websocket.send_json(stats.__dict__)
                    await asyncio.sleep(2.0)
            except WebSocketDisconnect:
                self.logger.info("Stats WebSocket disconnected", event="plxs.ws.stats.disconnect")
            except Exception as e:
                self.logger.error(f"Stats WebSocket error: {e}", event="plxs.ws.stats.error")

    def _get_nvidia_gpus(self) -> List[Dict[str, Any]]:
        """Get NVIDIA GPU information using pynvml."""
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
        """Get AMD GPU information using pyamdgpuinfo or subprocess."""
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
        """Get Intel GPU information using xpu-smi or subprocess."""
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

    async def _collect_system_stats(self) -> PiscesLxPlxsSystemStats:
        """
        Collect system resource statistics including multi-vendor GPU support.
        
        Returns:
            PiscesLxPlxsSystemStats: Current system statistics.
        """
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
        
        return PiscesLxPlxsSystemStats(
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
        """
        List available models from the model directory.
        
        Returns:
            List[Dict[str, Any]]: List of model information.
        """
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
        """
        Proxy request to the inference server.
        
        Args:
            endpoint: The API endpoint to call.
            request: The request body.
        
        Returns:
            dict: Response from the inference server.
        """
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
                self.logger.error(f"Inference proxy error: {e}", event="plxs.proxy.error")
                return {"error": str(e)}

    async def _list_mcp_tools(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available MCP tools.
        
        Args:
            category: Optional category filter.
        
        Returns:
            List[Dict[str, Any]]: List of tool information.
        """
        try:
            from opss.mcp.mcps import POPSSToolRegistry
            registry = POPSSToolRegistry.get_instance()
            tools = registry.list_tools()
            
            if category:
                tools = [t for t in tools if t.get("category") == category]
            
            return tools
        except Exception as e:
            self.logger.error(f"MCP tools list error: {e}", event="plxs.mcp.error")
            return []

    async def _execute_mcp_tool(self, request: dict) -> Dict[str, Any]:
        """
        Execute an MCP tool.
        
        Args:
            request: The tool execution request.
        
        Returns:
            Dict[str, Any]: Tool execution result.
        """
        try:
            from opss.mcp.mcps import POPSSToolRegistry
            registry = POPSSToolRegistry.get_instance()
            
            tool_name = request.get("tool")
            arguments = request.get("arguments", {})
            
            result = await registry.execute_tool(tool_name, arguments)
            return {"success": True, "result": result}
        except Exception as e:
            self.logger.error(f"MCP tool execute error: {e}", event="plxs.mcp.exec_error")
            return {"success": False, "error": str(e)}

    def run(self, host: str = "127.0.0.1") -> None:
        """
        Start the FastAPI server.
        
        Args:
            host: The host address to bind to.
        """
        self.logger.info(
            f"Starting PLxS server on {host}:{self.port}",
            event="plxs.server.start"
        )
        
        uvicorn.run(
            self.app,
            host=host,
            port=self.port,
            log_level="info"
        )


_server_instance: Optional[PiscesLxPlxsServer] = None


def get_app() -> FastAPI:
    """Get or create the FastAPI application instance."""
    global _server_instance
    if _server_instance is None:
        _server_instance = PiscesLxPlxsServer()
    return _server_instance.app


app = get_app()
