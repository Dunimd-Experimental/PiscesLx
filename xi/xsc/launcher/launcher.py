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
Xi Launcher - Orchestrates backend and frontend startup.

This module provides the launcher that starts both the Xi backend API
server (FastAPI on port 3140) and the frontend (Next.js on port 3000)
as coordinated subprocesses.

The launcher ensures:
    1. Backend starts first and is ready
    2. Frontend starts after backend confirmation
    3. Both processes share the same lifecycle
    4. Graceful shutdown on SIGINT/SIGTERM
"""

import os
import sys
import signal
import subprocess
import time
import platform
import shutil
import socket
from pathlib import Path
from typing import Optional, List

from ..core.dc import XiLogger
from ..config.loader import find_project_root
from .backend import start_backend, wait_for_backend
from .frontend import start_frontend
from .process import cleanup_processes, kill_port_process


class XiLauncher:
    """
    Main launcher for Xi Studio.
    
    Orchestrates the startup of backend and frontend servers.
    """
    
    def __init__(
        self,
        xi_port: int = 3140,
        frontend_port: int = 3000,
        root_dir: Optional[str] = None
    ):
        """
        Initialize the launcher.
        
        Args:
            xi_port: Backend API port (default: 3140)
            frontend_port: Frontend port (default: 3000)
            root_dir: Project root directory
        """
        self.xi_port = xi_port
        self.frontend_port = frontend_port
        if root_dir:
            self.root_dir = Path(root_dir)
        else:
            self.root_dir = find_project_root()
        self.logger = XiLogger("Xi.Launcher", enable_file=True)
        self.processes: List[subprocess.Popen] = []
        self._shutdown_requested = False
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        self.logger.info(
            f"Received signal {signum}, initiating shutdown...",
            event="xi.launcher.signal"
        )
        self._shutdown_requested = True
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self) -> None:
        """Clean up all processes."""
        cleanup_processes(
            self.processes,
            self.xi_port,
            self.frontend_port,
            self.logger
        )
    
    def run(self) -> int:
        """
        Run the launcher.
        
        Starts backend and frontend servers and monitors them.
        
        Returns:
            Exit code (0 for success)
        """
        self._setup_signal_handlers()
        
        print(f"\n{'='*60}")
        print(f"  Xi Studio - PiscesL1 Workstation")
        print(f"{'='*60}")
        print(f"  Backend API:  http://127.0.0.1:{self.xi_port}")
        print(f"  Frontend:     http://127.0.0.1:{self.frontend_port}")
        print(f"  API Docs:     http://127.0.0.1:{self.xi_port}/docs")
        print(f"{'='*60}\n")
        
        backend_proc = start_backend(
            self.xi_port,
            self.root_dir,
            self.logger,
            self.processes
        )
        if not backend_proc:
            self._cleanup()
            return 1
        
        if not wait_for_backend(self.xi_port, logger=self.logger):
            self._cleanup()
            return 1
        
        frontend_proc = start_frontend(
            self.frontend_port,
            self.root_dir,
            self.logger,
            self.processes
        )
        if not frontend_proc:
            self.logger.warning(
                "Frontend failed to start, running backend only",
                event="xi.launcher.frontend_fallback"
            )
            print("\n[WARNING] Frontend failed to start. Running backend only.")
            print(f"[INFO] Access API at http://127.0.0.1:{self.xi_port}/docs\n")
        
        print("\n[INFO] Press Ctrl+C to stop Xi Studio\n")
        
        try:
            while not self._shutdown_requested:
                for proc in self.processes:
                    if proc.poll() is not None:
                        self.logger.info(
                            f"Process PID={proc.pid} exited with code {proc.returncode}",
                            event="xi.launcher.process_exit"
                        )
                        self._cleanup()
                        return proc.returncode or 0
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down Xi Studio...")
        
        self._cleanup()
        return 0
