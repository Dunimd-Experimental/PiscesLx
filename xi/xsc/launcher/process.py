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
Process management module for Xi Launcher.
"""

import sys
import signal
import platform
import subprocess
from typing import List

from ..core.dc import XiLogger


def kill_port_process(port: int, logger: Optional[XiLogger] = None) -> bool:
    """
    Kill any process listening on the specified port.
    
    Args:
        port: Port number
        logger: Optional logger instance
        
    Returns:
        True if a process was killed
    """
    system = platform.system().lower()
    
    try:
        if system == "windows":
            result = subprocess.run(
                f'netstat -ano | findstr ":{port}"',
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            for line in result.stdout.strip().split('\n'):
                if 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        subprocess.run(
                            ['taskkill', '/F', '/PID', pid],
                            capture_output=True,
                            timeout=5
                        )
                        if logger:
                            logger.info(
                                f"Killed process {pid} on port {port}",
                                event="xi.launcher.kill_port"
                            )
                        return True
        else:
            result = subprocess.run(
                f'lsof -ti:{port}',
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.stdout.strip():
                pid = result.stdout.strip().split('\n')[0]
                subprocess.run(
                    ['kill', '-9', pid],
                    capture_output=True,
                    timeout=5
                )
                if logger:
                    logger.info(
                        f"Killed process {pid} on port {port}",
                        event="xi.launcher.kill_port"
                    )
                return True
    except Exception as e:
        if logger:
            logger.error(
                f"Error killing port {port}: {e}",
                event="xi.launcher.kill_port_error"
            )
    
    return False


def cleanup_processes(
    processes: List[subprocess.Popen],
    xi_port: int,
    frontend_port: int,
    logger: XiLogger
) -> None:
    """
    Clean up all running processes.
    
    Args:
        processes: List of subprocess.Popen instances
        xi_port: Backend port
        frontend_port: Frontend port
        logger: XiLogger instance
    """
    for proc in processes:
        try:
            if proc.poll() is None:
                logger.info(
                    f"Terminating process PID={proc.pid}",
                    event="xi.launcher.terminate"
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"Force killing process PID={proc.pid}",
                        event="xi.launcher.kill"
                    )
                    proc.kill()
        except Exception as e:
            logger.error(
                f"Error cleaning up process: {e}",
                event="xi.launcher.cleanup_error"
            )
    
    processes.clear()
    
    kill_port_process(xi_port, logger)
    kill_port_process(frontend_port, logger)


def setup_signal_handlers(handler) -> None:
    """
    Setup signal handlers for graceful shutdown.
    
    Args:
        handler: Signal handler function
    """
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
