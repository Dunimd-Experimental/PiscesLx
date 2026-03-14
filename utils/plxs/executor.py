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
Command Executor for PLxS (PLx Studio) Backend Server.

This module provides the command execution layer that bridges the PLxS
frontend to manage.py commands. It handles subprocess management,
output streaming, and process lifecycle control.

The executor uses asyncio subprocess for non-blocking command execution
and supports real-time output streaming via queues.
"""

import os
import sys
import asyncio
import signal
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, Callable
from datetime import datetime
import json

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from utils.plxs.types import (
    PiscesLxPlxsCommand,
    PiscesLxPlxsCommandRequest,
    PiscesLxPlxsCommandResponse,
    PiscesLxPlxsRunStatus,
    PiscesLxPlxsLogLevel,
    PiscesLxPlxsLogEntry,
)


class PiscesLxPlxsExecutor:
    """
    Command executor for manage.py commands.
    
    This class handles the execution of manage.py commands as subprocesses,
    providing lifecycle management, output streaming, and process control.
    
    Attributes:
        root_dir (Path): Project root directory for manage.py location.
        logger (PiscesLxLogger): Logger instance for executor operations.
        active_processes (Dict[str, asyncio.subprocess.Process]): Map of
            run_id to active subprocess instances.
        output_queues (Dict[str, asyncio.Queue]): Map of run_id to output
            queues for streaming logs to WebSocket clients.
    """

    def __init__(self, root_dir: Optional[str] = None):
        """
        Initialize the command executor.
        
        Args:
            root_dir: Optional root directory path. Defaults to current
                working directory.
        """
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.logger = PiscesLxLogger(
            "PiscesLx.Plxs.Executor",
            file_path=get_log_file("PiscesLx.Plxs.Executor"),
            enable_file=True
        )
        self.active_processes: Dict[str, asyncio.subprocess.Process] = {}
        self.output_queues: Dict[str, asyncio.Queue] = {}
        self._process_status: Dict[str, PiscesLxPlxsRunStatus] = {}

    def _build_argv(self, request: PiscesLxPlxsCommandRequest) -> list:
        """
        Build command-line arguments for manage.py.
        
        Args:
            request: The command request containing command and arguments.
        
        Returns:
            list: List of command-line arguments for subprocess.
        """
        argv = [sys.executable, "manage.py", request.command.value]
        
        for key, value in request.args.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    argv.append(f"--{key}")
            elif isinstance(value, list):
                for item in value:
                    argv.extend([f"--{key}", str(item)])
            else:
                argv.extend([f"--{key}", str(value)])
        
        if request.run_id:
            argv.extend(["--run_id", request.run_id])
        if request.run_name:
            argv.extend(["--run_name", request.run_name])
        
        return argv

    async def execute(
        self,
        request: PiscesLxPlxsCommandRequest
    ) -> PiscesLxPlxsCommandResponse:
        """
        Execute a manage.py command.
        
        Args:
            request: The command request to execute.
        
        Returns:
            PiscesLxPlxsCommandResponse: Response indicating success/failure.
        """
        run_id = request.run_id or self._generate_run_id(request.command)
        
        try:
            argv = self._build_argv(request)
            
            self.logger.info(
                f"Executing command: {' '.join(argv)}",
                event="plxs.executor.execute"
            )
            
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.root_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            self.active_processes[run_id] = process
            self._process_status[run_id] = PiscesLxPlxsRunStatus.RUNNING
            self.output_queues[run_id] = asyncio.Queue()
            
            asyncio.create_task(self._stream_output(run_id, process))
            
            if not request.background:
                await process.wait()
                self._cleanup_process(run_id)
                
                if process.returncode == 0:
                    return PiscesLxPlxsCommandResponse(
                        success=True,
                        run_id=run_id,
                        message=f"Command {request.command.value} completed successfully"
                    )
                else:
                    return PiscesLxPlxsCommandResponse(
                        success=False,
                        run_id=run_id,
                        error=f"Command failed with exit code {process.returncode}"
                    )
            
            return PiscesLxPlxsCommandResponse(
                success=True,
                run_id=run_id,
                message=f"Command {request.command.value} started in background"
            )
            
        except Exception as e:
            self.logger.error(
                f"Failed to execute command: {e}",
                event="plxs.executor.error"
            )
            return PiscesLxPlxsCommandResponse(
                success=False,
                error=str(e)
            )

    async def _stream_output(
        self,
        run_id: str,
        process: asyncio.subprocess.Process
    ) -> None:
        """
        Stream stdout and stderr from a subprocess.
        
        Args:
            run_id: The run identifier for the process.
            process: The subprocess instance to stream from.
        """
        queue = self.output_queues.get(run_id)
        
        async def read_stream(stream, source: str):
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    
                    text = line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        entry = PiscesLxPlxsLogEntry(
                            timestamp=datetime.now().isoformat(),
                            level=PiscesLxPlxsLogLevel.INFO,
                            message=text,
                            source=source,
                            run_id=run_id
                        )
                        if queue:
                            await queue.put(entry)
            except Exception as e:
                self.logger.error(
                    f"Error streaming {source}: {e}",
                    event="plxs.executor.stream_error"
                )
        
        await asyncio.gather(
            read_stream(process.stdout, "stdout"),
            read_stream(process.stderr, "stderr")
        )
        
        await process.wait()
        
        if process.returncode == 0:
            self._process_status[run_id] = PiscesLxPlxsRunStatus.COMPLETED
        else:
            self._process_status[run_id] = PiscesLxPlxsRunStatus.FAILED
        
        self._cleanup_process(run_id)

    def _generate_run_id(self, command: PiscesLxPlxsCommand) -> str:
        """
        Generate a unique run ID for a command.
        
        Args:
            command: The command being executed.
        
        Returns:
            str: A unique run identifier.
        """
        from opss.run import POPSSRunIdFactory
        factory = POPSSRunIdFactory(prefix=command.value)
        return factory.new_id()

    def _cleanup_process(self, run_id: str) -> None:
        """
        Clean up process resources after completion.
        
        Args:
            run_id: The run identifier to clean up.
        """
        if run_id in self.active_processes:
            del self.active_processes[run_id]

    async def control(
        self,
        run_id: str,
        action: str
    ) -> PiscesLxPlxsCommandResponse:
        """
        Control a running process (pause, resume, cancel, kill).
        
        Args:
            run_id: The run identifier to control.
            action: The action to perform (pause, resume, cancel, kill).
        
        Returns:
            PiscesLxPlxsCommandResponse: Response indicating success/failure.
        """
        process = self.active_processes.get(run_id)
        if not process:
            return PiscesLxPlxsCommandResponse(
                success=False,
                run_id=run_id,
                error=f"No active process found for run_id: {run_id}"
            )
        
        previous_status = self._process_status.get(run_id)
        
        try:
            if action == "pause":
                if sys.platform == "win32":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGSTOP)
                self._process_status[run_id] = PiscesLxPlxsRunStatus.PAUSED
                return PiscesLxPlxsCommandResponse(
                    success=True,
                    run_id=run_id,
                    message=f"Process {run_id} paused",
                    data={"previous_status": previous_status.value if previous_status else None}
                )
            
            elif action == "resume":
                if sys.platform == "win32":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGCONT)
                self._process_status[run_id] = PiscesLxPlxsRunStatus.RUNNING
                return PiscesLxPlxsCommandResponse(
                    success=True,
                    run_id=run_id,
                    message=f"Process {run_id} resumed",
                    data={"previous_status": previous_status.value if previous_status else None}
                )
            
            elif action == "cancel":
                process.terminate()
                self._process_status[run_id] = PiscesLxPlxsRunStatus.CANCELLED
                return PiscesLxPlxsCommandResponse(
                    success=True,
                    run_id=run_id,
                    message=f"Process {run_id} cancelled",
                    data={"previous_status": previous_status.value if previous_status else None}
                )
            
            elif action == "kill":
                process.kill()
                self._process_status[run_id] = PiscesLxPlxsRunStatus.CANCELLED
                return PiscesLxPlxsCommandResponse(
                    success=True,
                    run_id=run_id,
                    message=f"Process {run_id} killed",
                    data={"previous_status": previous_status.value if previous_status else None}
                )
            
            else:
                return PiscesLxPlxsCommandResponse(
                    success=False,
                    run_id=run_id,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            self.logger.error(
                f"Failed to control process {run_id}: {e}",
                event="plxs.executor.control_error"
            )
            return PiscesLxPlxsCommandResponse(
                success=False,
                run_id=run_id,
                error=str(e)
            )

    async def get_output_stream(
        self,
        run_id: str
    ) -> AsyncGenerator[PiscesLxPlxsLogEntry, None]:
        """
        Get an async generator for streaming output from a run.
        
        Args:
            run_id: The run identifier to stream output from.
        
        Yields:
            PiscesLxPlxsLogEntry: Log entries from the process output.
        """
        queue = self.output_queues.get(run_id)
        if not queue:
            return
        
        while True:
            try:
                entry = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield entry
            except asyncio.TimeoutError:
                if run_id not in self.active_processes:
                    break
                continue

    def get_status(self, run_id: str) -> Optional[PiscesLxPlxsRunStatus]:
        """
        Get the current status of a run.
        
        Args:
            run_id: The run identifier to check.
        
        Returns:
            Optional[PiscesLxPlxsRunStatus]: Current status or None if not found.
        """
        return self._process_status.get(run_id)

    def list_active_runs(self) -> Dict[str, PiscesLxPlxsRunStatus]:
        """
        List all active runs and their statuses.
        
        Returns:
            Dict[str, PiscesLxPlxsRunStatus]: Map of run_id to status.
        """
        return dict(self._process_status)
