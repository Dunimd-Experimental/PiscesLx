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
Backend startup module for Xi Launcher.
"""

import os
import sys
import time
import socket
import subprocess
from pathlib import Path
from typing import Optional

from ..core.dc import XiLogger


def check_port_available(port: int) -> bool:
    """
    Check if a port is available for binding.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is available
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', port))
            return True
    except OSError:
        return False


def wait_for_backend(port: int, timeout: float = 30.0, logger: Optional[XiLogger] = None) -> bool:
    """
    Wait for backend server to be ready.
    
    Args:
        port: Backend server port
        timeout: Maximum wait time in seconds
        logger: Optional logger instance
        
    Returns:
        True if backend is ready
    """
    import httpx
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = httpx.get(
                f"http://127.0.0.1:{port}/healthz",
                timeout=1.0
            )
            if response.status_code == 200:
                if logger:
                    logger.info(
                        "Backend server is ready",
                        event="xi.launcher.backend_ready"
                    )
                return True
        except Exception:
            pass
        time.sleep(0.5)
    
    if logger:
        logger.error(
            f"Backend server failed to start within {timeout}s",
            event="xi.launcher.backend_timeout"
        )
    return False


def start_backend(
    port: int,
    root_dir: Path,
    logger: XiLogger,
    processes: list
) -> Optional[subprocess.Popen]:
    """
    Start the backend server.
    
    Args:
        port: Backend server port
        root_dir: Working directory
        logger: XiLogger instance
        processes: List to append process to
        
    Returns:
        subprocess.Popen instance or None on failure
    """
    from .process import kill_port_process
    
    logger.info(
        f"Starting backend server on port {port}",
        event="xi.launcher.backend_start"
    )
    
    kill_port_process(port, logger)
    
    cmd = [
        sys.executable,
        "-m", "uvicorn",
        "xi.xsc.server:app",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--log-level", "info"
    ]
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        processes.append(proc)
        
        logger.info(
            f"Backend process started with PID={proc.pid}",
            event="xi.launcher.backend_pid"
        )
        
        return proc
    except Exception as e:
        logger.error(
            f"Failed to start backend: {e}",
            event="xi.launcher.backend_error"
        )
        return None
