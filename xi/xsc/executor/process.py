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
Process management utilities for Xi Executor.

This module provides process control and lifecycle management.
"""

import sys
import signal
from typing import Dict, Optional

from ..core.types import XiRunStatus, XiResponse


def control_process(
    run_id: str,
    process,
    action: str,
    process_status: Dict[str, XiRunStatus]
) -> XiResponse:
    """
    Control a running process.
    
    Args:
        run_id: Run identifier
        process: asyncio subprocess
        action: Control action (pause, resume, cancel, kill)
        process_status: Dictionary of process statuses
        
    Returns:
        XiResponse with result
    """
    previous_status = process_status.get(run_id)
    
    try:
        if action == "pause":
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGSTOP)
            process_status[run_id] = XiRunStatus.PAUSED
            return XiResponse(
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
            process_status[run_id] = XiRunStatus.RUNNING
            return XiResponse(
                success=True,
                run_id=run_id,
                message=f"Process {run_id} resumed",
                data={"previous_status": previous_status.value if previous_status else None}
            )
        
        elif action == "cancel":
            process.terminate()
            process_status[run_id] = XiRunStatus.CANCELLED
            return XiResponse(
                success=True,
                run_id=run_id,
                message=f"Process {run_id} cancelled",
                data={"previous_status": previous_status.value if previous_status else None}
            )
        
        elif action == "kill":
            process.kill()
            process_status[run_id] = XiRunStatus.CANCELLED
            return XiResponse(
                success=True,
                run_id=run_id,
                message=f"Process {run_id} killed",
                data={"previous_status": previous_status.value if previous_status else None}
            )
        
        else:
            return XiResponse(
                success=False,
                run_id=run_id,
                error=f"Unknown action: {action}"
            )
    
    except Exception as e:
        return XiResponse(
            success=False,
            run_id=run_id,
            error=str(e)
        )


def generate_run_id(command) -> str:
    """
    Generate a unique run ID.
    
    Args:
        command: XiCommand enum value
        
    Returns:
        Unique run ID string
    """
    from opss.run import POPSSRunIdFactory
    factory = POPSSRunIdFactory(prefix=command.value)
    return factory.new_id()


def update_status_on_exit(
    run_id: str,
    process,
    process_status: Dict[str, XiRunStatus]
) -> None:
    """
    Update process status when it exits.
    
    Args:
        run_id: Run identifier
        process: asyncio subprocess
        process_status: Dictionary of process statuses
    """
    if process.returncode == 0:
        process_status[run_id] = XiRunStatus.COMPLETED
    else:
        process_status[run_id] = XiRunStatus.FAILED
