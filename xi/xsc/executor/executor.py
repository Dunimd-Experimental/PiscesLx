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
Command Executor for Xi Studio Backend Server.

This module provides the command execution layer that bridges the Xi
frontend to manage.py commands. It handles subprocess management,
output streaming, and process lifecycle control.

The executor uses asyncio subprocess for non-blocking command execution
and supports real-time output streaming via queues.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, List

from ..core.dc import XiLogger
from ..core.types import (
    XiCommand,
    XiRequest,
    XiResponse,
    XiRunStatus,
    XiLogEntry,
)
from .argv import build_argv, build_argv_from_schema
from .stream import stream_output, get_output_stream
from .process import control_process, generate_run_id, update_status_on_exit


class XiExecutor:
    """
    Command executor for Xi Studio.
    
    This class handles:
    - Command execution via subprocess
    - Output streaming
    - Process lifecycle control
    """
    
    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize the executor.
        
        Args:
            root_dir: Working directory for command execution
        """
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.logger = XiLogger(
            "Xi.Executor",
            enable_file=True
        )
        self.active_processes: Dict[str, asyncio.subprocess.Process] = {}
        self.output_queues: Dict[str, asyncio.Queue] = {}
        self._process_status: Dict[str, XiRunStatus] = {}
    
    async def execute(
        self,
        request: XiRequest
    ) -> XiResponse:
        """
        Execute a command.
        
        Args:
            request: XiRequest with command and arguments
            
        Returns:
            XiResponse with execution result
        """
        run_id = request.run_id or generate_run_id(request.command)
        
        try:
            argv = build_argv(request, self.root_dir)
            
            import sys
            from ..config import get_xi_config
            
            config = get_xi_config()
            venv_python = None
            
            if config.environment.virtualenv and config.environment.virtualenv.enabled:
                venv_path = config.environment.virtualenv.path
                if not Path(venv_path).is_absolute():
                    venv_path = str(self.root_dir / venv_path)
                
                if sys.platform == "win32":
                    venv_python = str(Path(venv_path) / "Scripts" / "python.exe")
                else:
                    venv_python = str(Path(venv_path) / "bin" / "python")
                
                if not Path(venv_python).exists():
                    self.logger.warning(
                        f"Virtual environment Python not found: {venv_python}, falling back to system Python",
                        event="xi.executor.venv_not_found"
                    )
                    venv_python = None
            
            if venv_python:
                if argv and argv[0] in ("python", "python3", sys.executable):
                    argv[0] = venv_python
                elif argv and not argv[0].endswith("python"):
                    argv.insert(0, venv_python)
            
            self.logger.info(
                f"Executing command: {' '.join(argv)}, venv={venv_python is not None}",
                event="xi.executor.execute"
            )
            
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.root_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            self.active_processes[run_id] = process
            self._process_status[run_id] = XiRunStatus.RUNNING
            self.output_queues[run_id] = asyncio.Queue()
            
            asyncio.create_task(self._stream_and_wait(run_id, process))
            
            if not request.background:
                await process.wait()
                self._cleanup_process(run_id)
                
                if process.returncode == 0:
                    return XiResponse(
                        success=True,
                        run_id=run_id,
                        message=f"Command {request.command.value} completed successfully"
                    )
                else:
                    return XiResponse(
                        success=False,
                        run_id=run_id,
                        error=f"Command failed with exit code {process.returncode}"
                    )
            
            return XiResponse(
                success=True,
                run_id=run_id,
                message=f"Command {request.command.value} started in background"
            )
        
        except Exception as e:
            self.logger.error(
                f"Failed to execute command: {e}",
                event="xi.executor.error"
            )
            return XiResponse(
                success=False,
                error=str(e)
            )
    
    async def _stream_and_wait(
        self,
        run_id: str,
        process: asyncio.subprocess.Process
    ) -> None:
        """
        Stream output and wait for process to complete.
        
        Args:
            run_id: Run identifier
            process: asyncio subprocess
        """
        queue = self.output_queues.get(run_id)
        
        await stream_output(run_id, process, queue, self.logger)
        
        await process.wait()
        
        update_status_on_exit(run_id, process, self._process_status)
        
        self._cleanup_process(run_id)
    
    def _cleanup_process(self, run_id: str) -> None:
        """
        Clean up process resources.
        
        Args:
            run_id: Run identifier
        """
        if run_id in self.active_processes:
            del self.active_processes[run_id]
    
    async def control(
        self,
        run_id: str,
        action: str
    ) -> XiResponse:
        """
        Control a running process.
        
        Args:
            run_id: Run identifier
            action: Control action (pause, resume, cancel, kill)
            
        Returns:
            XiResponse with result
        """
        process = self.active_processes.get(run_id)
        if not process:
            return XiResponse(
                success=False,
                run_id=run_id,
                error=f"No active process found for run_id: {run_id}"
            )
        
        return control_process(run_id, process, action, self._process_status)
    
    async def get_output_stream(
        self,
        run_id: str
    ) -> AsyncGenerator[XiLogEntry, None]:
        """
        Get output stream for a run.
        
        Args:
            run_id: Run identifier
            
        Yields:
            XiLogEntry objects
        """
        queue = self.output_queues.get(run_id)
        if not queue:
            return
        
        async for entry in get_output_stream(run_id, queue, self.active_processes):
            yield entry
    
    def get_status(self, run_id: str) -> Optional[XiRunStatus]:
        """
        Get status of a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            XiRunStatus or None
        """
        return self._process_status.get(run_id)
    
    def list_active_runs(self) -> Dict[str, XiRunStatus]:
        """
        List all active runs.
        
        Returns:
            Dictionary of run_id to status
        """
        return dict(self._process_status)
    
    def build_argv_from_schema(
        self,
        command_name: str,
        parameters: Dict[str, Any],
        schema_params: List[Any]
    ) -> List[str]:
        """
        Build command line arguments from schema and parameter values.
        
        Args:
            command_name: Name of the command
            parameters: Dictionary of parameter values
            schema_params: List of parameter schema objects
            
        Returns:
            List of command line argument strings
        """
        from ..config import get_config_loader
        
        config_loader = get_config_loader()
        return build_argv_from_schema(
            command_name,
            parameters,
            schema_params,
            config_loader
        )
