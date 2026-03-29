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
Execute command implementation for Xi Executor.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict

from ..core.types import XiRequest, XiResponse, XiRunStatus
from ..core.dc import XiLogger
from .argv import build_argv
from .stream import stream_output
from .process import generate_run_id, update_status_on_exit


from ..config.loader import get_xi_config


async def execute_command(
    request: XiRequest,
    root_dir: Optional[Path] = None,
    active_processes: Optional[Dict] = None,
    output_queues: Optional[Dict] = None,
    process_status: Optional[Dict] = None,
    logger: Optional[XiLogger] = None
) -> XiResponse:
    """
    Execute a command asynchronously.
    
    Args:
        request: XiRequest object
        root_dir: Working directory path
        active_processes: Dictionary of active processes
        output_queues: Dictionary of output queues
        process_status: Dictionary of process statuses
        logger: XiLogger instance
        
    Returns:
        XiResponse with result
    """
    _root_dir = root_dir or Path.cwd()
    _logger = logger or XiLogger("Xi.Executor", enable_file=True)
    _active_processes = active_processes or {}
    _output_queues = output_queues or {}
    _process_status = process_status or {}
    
    run_id = request.run_id or generate_run_id(request.command)
    
    try:
        argv = build_argv(request, _root_dir)
        
        _logger.info(
            f"Executing command: {' '.join(argv)}",
            event="xi.executor.execute"
        )
        
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(_root_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        
        _active_processes[run_id] = process
        _process_status[run_id] = XiRunStatus.RUNNING
        _output_queues[run_id] = asyncio.Queue()
        
        asyncio.create_task(
            stream_output(run_id, process, _output_queues, _active_processes, _logger)
        )
        
        if not request.background:
            await process.wait()
            update_status_on_exit(run_id, process, _process_status)
            
            if run_id in _active_processes:
                del _active_processes[run_id]
            
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
        _logger.error(
            f"Failed to execute command: {e}",
            event="xi.executor.error"
        )
        return XiResponse(
            success=False,
            error=str(e)
        )
