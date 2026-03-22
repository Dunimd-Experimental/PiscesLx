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

from .dc import XiLogger


class XiLauncher:
    def __init__(
        self,
        xi_port: int = 3140,
        frontend_port: int = 3000,
        root_dir: Optional[str] = None
    ):
        self.xi_port = xi_port
        self.frontend_port = frontend_port
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.logger = XiLogger(
            "Xi.Launcher",
            enable_file=True
        )
        self.processes: List[subprocess.Popen] = []
        self._shutdown_requested = False

    def _setup_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        self.logger.info(
            f"Received signal {signum}, initiating shutdown...",
            event="xi.launcher.signal"
        )
        self._shutdown_requested = True
        self._cleanup()
        sys.exit(0)

    def _kill_port_process(self, port: int) -> bool:
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
                            self.logger.info(
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
                    self.logger.info(
                        f"Killed process {pid} on port {port}",
                        event="xi.launcher.kill_port"
                    )
                    return True
        except Exception as e:
            self.logger.error(
                f"Error killing port {port}: {e}",
                event="xi.launcher.kill_port_error"
            )

        return False

    def _cleanup(self) -> None:
        for proc in self.processes:
            try:
                if proc.poll() is None:
                    self.logger.info(
                        f"Terminating process PID={proc.pid}",
                        event="xi.launcher.terminate"
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(
                            f"Force killing process PID={proc.pid}",
                            event="xi.launcher.kill"
                        )
                        proc.kill()
            except Exception as e:
                self.logger.error(
                    f"Error cleaning up process: {e}",
                    event="xi.launcher.cleanup_error"
                )

        self.processes.clear()

        self._kill_port_process(self.xi_port)
        self._kill_port_process(self.frontend_port)

    def _check_port_available(self, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False

    def _wait_for_backend(self, timeout: float = 30.0) -> bool:
        import httpx

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{self.xi_port}/healthz",
                    timeout=1.0
                )
                if response.status_code == 200:
                    self.logger.info(
                        "Backend server is ready",
                        event="xi.launcher.backend_ready"
                    )
                    return True
            except Exception:
                pass
            time.sleep(0.5)

        self.logger.error(
            f"Backend server failed to start within {timeout}s",
            event="xi.launcher.backend_timeout"
        )
        return False

    def _start_backend(self) -> Optional[subprocess.Popen]:
        self.logger.info(
            f"Starting backend server on port {self.xi_port}",
            event="xi.launcher.backend_start"
        )

        self._kill_port_process(self.xi_port)

        cmd = [
            sys.executable,
            "-m", "uvicorn",
            "xi.xsc.server:app",
            "--host", "127.0.0.1",
            "--port", str(self.xi_port),
            "--log-level", "info"
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            self.processes.append(proc)

            self.logger.info(
                f"Backend process started with PID={proc.pid}",
                event="xi.launcher.backend_pid"
            )

            return proc
        except Exception as e:
            self.logger.error(
                f"Failed to start backend: {e}",
                event="xi.launcher.backend_error"
            )
            return None

    def _ensure_frontend_build(self) -> Optional[Path]:
        is_windows = platform.system().lower().startswith("win")
        npm_cmd = "npm.cmd" if is_windows else "npm"

        source_dir = self.root_dir / "xi" / "xis"
        prod_dir = self.root_dir / ".pisceslx" / "plxs"

        if not source_dir.exists():
            self.logger.error(
                f"Source frontend directory not found: {source_dir}",
                event="xi.launcher.source_missing"
            )
            return None

        needs_build = False

        if not prod_dir.exists():
            self.logger.info(
                "Production directory does not exist, creating...",
                event="xi.launcher.create_prod_dir"
            )
            needs_build = True
        elif not (prod_dir / "node_modules").exists():
            self.logger.info(
                "node_modules not found, need to install dependencies",
                event="xi.launcher.need_install"
            )
            needs_build = True
        elif not (prod_dir / ".next").exists():
            self.logger.info(
                ".next build not found, need to build",
                event="xi.launcher.need_build"
            )
            needs_build = True
        else:
            source_pkg = source_dir / "package.json"
            prod_pkg = prod_dir / "package.json"

            if source_pkg.exists() and prod_pkg.exists():
                import json
                with open(source_pkg) as f:
                    src_data = json.load(f)
                with open(prod_pkg) as f:
                    prod_data = json.load(f)

                if src_data.get("version") != prod_data.get("version"):
                    self.logger.info(
                        "Version mismatch, need to rebuild",
                        event="xi.launcher.version_mismatch"
                    )
                    needs_build = True

        if needs_build:
            print("\n[INFO] Copying source to production directory...")

            if prod_dir.exists():
                self.logger.info(
                    f"Removing existing production directory: {prod_dir}",
                    event="xi.launcher.remove_prod_dir"
                )
                shutil.rmtree(prod_dir)

            self.logger.info(
                f"Copying source to production directory: {prod_dir}",
                event="xi.launcher.copy_source"
            )
            shutil.copytree(source_dir, prod_dir)

            node_modules = prod_dir / "node_modules"
            lock_file = prod_dir / "package-lock.json"

            if node_modules.exists():
                self.logger.info(
                    "Removing existing node_modules for clean install",
                    event="xi.launcher.remove_node_modules"
                )
                shutil.rmtree(node_modules)

            if lock_file.exists():
                lock_file.unlink()

            print("[INFO] Installing dependencies...")
            self.logger.info(
                "Running npm install",
                event="xi.launcher.npm_install"
            )

            install_result = subprocess.run(
                [npm_cmd, "install"],
                cwd=str(prod_dir),
                capture_output=True,
                text=True
            )

            if install_result.returncode != 0:
                self.logger.error(
                    f"npm install failed: {install_result.stderr}",
                    event="xi.launcher.npm_install_error"
                )
                print(f"[ERROR] npm install failed: {install_result.stderr}")
                return None

            print("[INFO] Building production bundle...")
            self.logger.info(
                "Running npm run build",
                event="xi.launcher.npm_build"
            )

            build_result = subprocess.run(
                [npm_cmd, "run", "build"],
                cwd=str(prod_dir),
                capture_output=True,
                text=True
            )

            if build_result.returncode != 0:
                self.logger.error(
                    f"npm run build failed: {build_result.stderr}",
                    event="xi.launcher.npm_build_error"
                )
                print(f"[ERROR] npm run build failed: {build_result.stderr}")
                return None

            print("[INFO] Frontend build complete!\n")
            self.logger.info(
                "Frontend build complete",
                event="xi.launcher.build_complete"
            )

        return prod_dir

    def _start_frontend(self) -> Optional[subprocess.Popen]:
        is_windows = platform.system().lower().startswith("win")
        npm_cmd = "npm.cmd" if is_windows else "npm"

        frontend_dir = self._ensure_frontend_build()
        if not frontend_dir:
            return None

        self.logger.info(
            f"Starting frontend server on port {self.frontend_port}",
            event="xi.launcher.frontend_start"
        )

        self._kill_port_process(self.frontend_port)

        cmd = [npm_cmd, "run", "start"]

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(frontend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PORT": str(self.frontend_port)}
            )
            self.processes.append(proc)

            time.sleep(2)

            if proc.poll() is not None:
                output = ""
                try:
                    output = proc.stdout.read().decode('utf-8', errors='ignore')
                except Exception:
                    pass

                self.logger.error(
                    f"Frontend process exited immediately with code {proc.returncode}: {output}",
                    event="xi.launcher.frontend_exit"
                )
                print(f"[ERROR] Frontend failed to start: {output}")
                return None

            self.logger.info(
                f"Frontend process started with PID={proc.pid}",
                event="xi.launcher.frontend_pid"
            )

            return proc
        except Exception as e:
            self.logger.error(
                f"Failed to start frontend: {e}",
                event="xi.launcher.frontend_error"
            )
            return None

    def run(self) -> int:
        self._setup_signal_handlers()

        print(f"\n{'='*60}")
        print(f"  Xi Studio - PiscesL1 Workstation")
        print(f"{'='*60}")
        print(f"  Backend API:  http://127.0.0.1:{self.xi_port}")
        print(f"  Frontend:     http://127.0.0.1:{self.frontend_port}")
        print(f"  API Docs:     http://127.0.0.1:{self.xi_port}/docs")
        print(f"{'='*60}\n")

        backend_proc = self._start_backend()
        if not backend_proc:
            self._cleanup()
            return 1

        if not self._wait_for_backend():
            self._cleanup()
            return 1

        frontend_proc = self._start_frontend()
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
