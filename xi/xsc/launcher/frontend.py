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
Frontend startup module for Xi Launcher.
"""

import os
import sys
import json
import time
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from ..core.dc import XiLogger


def ensure_frontend_build(root_dir: Path, logger: XiLogger) -> Optional[Path]:
    """
    Ensure frontend is built and ready.
    
    Args:
        root_dir: Project root directory
        logger: XiLogger instance
        
    Returns:
        Path to frontend directory or None on failure
    """
    is_windows = platform.system().lower().startswith("win")
    npm_cmd = "npm.cmd" if is_windows else "npm"
    
    source_dir = root_dir / "xi" / "xis"
    prod_dir = root_dir / ".pisceslx" / "plxs"
    
    if not source_dir.exists():
        logger.error(
            f"Source frontend directory not found: {source_dir}",
            event="xi.launcher.source_missing"
        )
        return None
    
    needs_build = False
    
    if not prod_dir.exists():
        logger.info(
            "Production directory does not exist, creating...",
            event="xi.launcher.create_prod_dir"
        )
        needs_build = True
    elif not (prod_dir / "node_modules").exists():
        logger.info(
            "node_modules not found, need to install dependencies",
            event="xi.launcher.need_install"
        )
        needs_build = True
    elif not (prod_dir / ".next").exists():
        logger.info(
            ".next build not found, need to build",
            event="xi.launcher.need_build"
        )
        needs_build = True
    else:
        source_pkg = source_dir / "package.json"
        prod_pkg = prod_dir / "package.json"
        
        if source_pkg.exists() and prod_pkg.exists():
            with open(source_pkg) as f:
                src_data = json.load(f)
            with open(prod_pkg) as f:
                prod_data = json.load(f)
            
            if src_data.get("version") != prod_data.get("version"):
                logger.info(
                    "Version mismatch, need to rebuild",
                    event="xi.launcher.version_mismatch"
                )
                needs_build = True
    
    if needs_build:
        print("\n[INFO] Copying source to production directory...")
        
        if prod_dir.exists():
            logger.info(
                f"Removing existing production directory: {prod_dir}",
                event="xi.launcher.remove_prod_dir"
            )
            shutil.rmtree(prod_dir)
        
        logger.info(
            f"Copying source to production directory: {prod_dir}",
            event="xi.launcher.copy_source"
        )
        shutil.copytree(source_dir, prod_dir)
        
        node_modules = prod_dir / "node_modules"
        lock_file = prod_dir / "package-lock.json"
        
        if node_modules.exists():
            logger.info(
                "Removing existing node_modules for clean install",
                event="xi.launcher.remove_node_modules"
            )
            shutil.rmtree(node_modules)
        
        if lock_file.exists():
            lock_file.unlink()
        
        print("[INFO] Installing dependencies...")
        logger.info(
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
            logger.error(
                f"npm install failed: {install_result.stderr}",
                event="xi.launcher.npm_install_error"
            )
            print(f"[ERROR] npm install failed: {install_result.stderr}")
            return None
        
        print("[INFO] Building production bundle...")
        logger.info(
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
            logger.error(
                f"npm run build failed: {build_result.stderr}",
                event="xi.launcher.npm_build_error"
            )
            print(f"[ERROR] npm run build failed: {build_result.stderr}")
            return None
        
        print("[INFO] Frontend build complete!\n")
        logger.info(
            "Frontend build complete",
            event="xi.launcher.build_complete"
        )
    
    return prod_dir


def start_frontend(
    port: int,
    root_dir: Path,
    logger: XiLogger,
    processes: list
) -> Optional[subprocess.Popen]:
    """
    Start the frontend server.
    
    Args:
        port: Frontend server port
        root_dir: Project root directory
        logger: XiLogger instance
        processes: List to append process to
        
    Returns:
        subprocess.Popen instance or None on failure
    """
    from .process import kill_port_process
    
    is_windows = platform.system().lower().startswith("win")
    npm_cmd = "npm.cmd" if is_windows else "npm"
    
    frontend_dir = ensure_frontend_build(root_dir, logger)
    if not frontend_dir:
        return None
    
    logger.info(
        f"Starting frontend server on port {port}",
        event="xi.launcher.frontend_start"
    )
    
    kill_port_process(port, logger)
    
    cmd = [npm_cmd, "run", "start"]
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(frontend_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PORT": str(port)}
        )
        processes.append(proc)
        
        time.sleep(2)
        
        if proc.poll() is not None:
            output = ""
            try:
                output = proc.stdout.read().decode('utf-8', errors='ignore')
            except Exception:
                pass
            
            logger.error(
                f"Frontend process exited immediately with code {proc.returncode}: {output}",
                event="xi.launcher.frontend_exit"
            )
            print(f"[ERROR] Frontend failed to start: {output}")
            return None
        
        logger.info(
            f"Frontend process started with PID={proc.pid}",
            event="xi.launcher.frontend_pid"
        )
        
        return proc
    except Exception as e:
        logger.error(
            f"Failed to start frontend: {e}",
            event="xi.launcher.frontend_error"
        )
        return None
