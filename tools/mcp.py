#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import time
import queue
import signal
import asyncio
import requests
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.log import RIGHT, DEBUG, ERROR

@dataclass
class ServerMetrics:
    """
    Represents server performance metrics.

    Attributes:
        cpu_percent (float): CPU usage percentage.
        memory_mb (float): Memory usage in megabytes.
        uptime_seconds (float): Server uptime in seconds.
        request_count (int): Number of requests received.
        error_count (int): Number of errors occurred.
        avg_response_time (float): Average response time.
    """
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    uptime_seconds: float = 0.0
    request_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0

# Try to import psutil for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    ERROR("psutil not available - some features will be disabled")

def check_mcp_server_status(host: str = "localhost", port: int = 8080) -> Dict[str, Any]:
    """
    Check if the MCP server is running and retrieve its status.

    Args:
        host (str): Server host address. Defaults to "localhost".
        port (int): Server port number. Defaults to 8080.

    Returns:
        Dict[str, Any]: A dictionary containing server running status, health status, message, version, and available tools.
    """
    try:
        response = requests.get(f"http://{host}:{port}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "running": True,
                "status": "healthy",
                "message": data.get("message", "Unknown"),
                "version": data.get("version", "Unknown"),
                "tools": data.get("available_tools", 0)
            }
        else:
            return {
                "running": False,
                "status": "error",
                "message": f"Server returned status {response.status_code}"
            }
    except requests.exceptions.ConnectionError:
        return {
            "running": False,
            "status": "not_running",
            "message": "MCP server is not running"
        }
    except Exception as e:
        return {
            "running": False,
            "status": "error",
            "message": f"Error checking server: {e}"
        }

def start_mcp_server(host: str = "localhost", port: int = 8080, background: bool = True):
    """
    Start the MCP server.

    Args:
        host (str): Server host address. Defaults to "localhost".
        port (int): Server port number. Defaults to 8080.
        background (bool): Whether to start the server in the background. Defaults to True.

    Returns:
        bool: True if the server started successfully, False otherwise.
    """
    # Check if the server is already running
    status = check_mcp_server_status(host, port)
    if status["running"]:
        RIGHT(f"MCP server is already running on {host}:{port}")
        return True
    
    try:
        # Directly import and run the server instead of using subprocess
        from model.mcp.server import mcp_server
        import uvicorn
        
        def run_server():
            """Run the MCP server using Uvicorn."""
            config = uvicorn.Config(mcp_server.app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            server.run()
        
        if background:
            # Start the server in a background thread
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Wait for the server to start
            time.sleep(3)
            
            # Check if the server started successfully
            status = check_mcp_server_status(host, port)
            if status["running"]:
                RIGHT(f"MCP server started successfully on {host}:{port}")
                RIGHT(f"Server status: {status['status']}")
                RIGHT(f"Available tools: {status['tools']}")
                return True
            else:
                ERROR(f"Failed to start MCP server: {status['message']}")
                return False
        else:
            # Start the server in the foreground
            RIGHT(f"Starting MCP server on {host}:{port} (foreground mode)")
            run_server()
            return True
            
    except Exception as e:
        ERROR(f"Error starting MCP server: {e}")
        return False

def stop_mcp_server(host: str = "localhost", port: int = 8080):
    """
    Stop the MCP server.

    Args:
        host (str): Server host address. Defaults to "localhost".
        port (int): Server port number. Defaults to 8080.

    Returns:
        bool: True if the server stopped successfully, False otherwise.
    """
    # Check if the server is running
    status = check_mcp_server_status(host, port)
    if not status["running"]:
        RIGHT("MCP server is not running")
        return True
    
    try:
        if sys.platform == "win32":
            # Windows - Use taskkill to terminate the process
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
        else:
            # Unix-like systems - Find the process by port and kill it
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
            
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    subprocess.run(["kill", "-9", pid])
        
        # Wait and check if the server stopped
        time.sleep(2)
        status = check_mcp_server_status(host, port)
        if not status["running"]:
            RIGHT("MCP server stopped successfully")
            return True
        else:
            ERROR("Failed to stop MCP server")
            return False
            
    except Exception as e:
        ERROR(f"Error stopping MCP server: {e}")
        return False

async def test_mcp_server(host: str = "localhost", port: int = 8080):
    """
    Test the functionality of the MCP server.

    Args:
        host (str): Server host address. Defaults to "localhost".
        port (int): Server port number. Defaults to 8080.

    Returns:
        bool: True if all tests passed, False otherwise.
    """
    from model.mcp.translator import MCPTranslationLayer
    
    RIGHT(f"Testing MCP server at {host}:{port}")
    
    # Check server status
    status = check_mcp_server_status(host, port)
    if not status["running"]:
        ERROR("MCP server is not running. Please start it first.")
        return False
    
    RIGHT(f"✅ Server is running - {status['message']}")
    RIGHT(f"✅ Available tools: {status['tools']}")
    
    try:
        async with MCPTranslationLayer(f"http://{host}:{port}") as translator:
            # Test 1: Get available tools
            RIGHT("Testing tool capabilities...")
            tools = await translator.get_available_tools()
            RIGHT(f"✅ Found {len(tools)} available tools:")
            for tool in tools[:5]:  # Show first 5 tools
                tool_info = tool if isinstance(tool, dict) else tool.dict()
                RIGHT(f"   - {tool_info.get('function_name', 'Unknown')}: {tool_info.get('description', 'No description')}")
            
            # Test 2: Test simple agent call
            RIGHT("\nTesting simple tool execution...")
            test_text = """
            Let me test the calculation function:
            
            <agent><an>calculator</an><ap1>2 + 3 * 4</ap1></agent>
            
            Calculation completed!
            """
            
            processed_text = await translator.process_model_output(test_text.strip())
            RIGHT("✅ Tool execution test completed:")
            print(processed_text)
            
            # Test 3: Test complex agent calls
            RIGHT("\nTesting multiple tool execution...")
            complex_test = """
            I'll help you with multiple operations:
            
            1. Calculation: <agent><an>calculator</an><ap1>sqrt(144) + log(10)</ap1></agent>
            
            2. Web search: <agent><an>web_search</an><ap1>Python FastAPI tutorial</ap1><ap2>3</ap2></agent>
            
            Operations completed!
            """
            
            processed_complex = await translator.process_model_output(complex_test.strip())
            RIGHT("✅ Complex tool execution test completed:")
            print(processed_complex)
            
            RIGHT("\n✅ All MCP server tests passed!")
            return True
            
    except Exception as e:
        ERROR(f"Error testing MCP server: {e}")
        return False

class AdvancedMCPServerManager:
    """
    Advanced MCP server manager with monitoring and optimization features.
    """
    
    def __init__(self):
        self.pid_file = Path("mcp_server.pid")
        self.log_file = Path("mcp_server.log")
        self.metrics_file = Path("mcp_metrics.json")
        self._monitor_thread = None
        self._stop_monitor = threading.Event()
        self.metrics = ServerMetrics()
        
    def start(self, host="localhost", port=8080, workers=1):
        """
        Start the MCP server with optimization.

        Args:
            host (str): Server host address. Defaults to "localhost".
            port (int): Server port number. Defaults to 8080.
            workers (int): Number of worker processes. Defaults to 1.

        Returns:
            bool: True if the server started successfully, False otherwise.
        """
        if self.is_running():
            ERROR("MCP server is already running")
            return False
            
        # Clean up the previous instance
        self.pid_file.unlink(missing_ok=True)
        
        try:
            # Directly import and run the server
            from model.mcp.server import mcp_server
            import uvicorn
            
            def run_server():
                """Run the MCP server using Uvicorn and save the PID."""
                config = uvicorn.Config(
                    mcp_server.app, 
                    host=host, 
                    port=port, 
                    log_level="info",
                    access_log=True
                )
                server = uvicorn.Server(config)
                
                # Save the process ID
                with open(self.pid_file, 'w') as f:
                    f.write(str(os.getpid()))
                    
                server.run()
            
            # Start the server in a background thread
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Wait for the server to start
            time.sleep(3)
            
            if self.is_running():
                RIGHT(f"🚀 Optimized MCP server started on {host}:{port}")
                self._start_monitoring()
                return True
            else:
                ERROR("Failed to start MCP server")
                return False
                
        except Exception as e:
            ERROR(f"Error starting MCP server: {e}")
            return False
    
    def _start_monitoring(self):
        """
        Start the background monitoring thread if psutil is available and the thread is not running.
        """
        if HAS_PSUTIL and (self._monitor_thread is None or not self._monitor_thread.is_alive()):
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
    
    def _monitor_loop(self):
        """
        Background monitoring loop to collect server metrics.
        """
        if not HAS_PSUTIL:
            return
            
        while not self._stop_monitor.is_set():
            try:
                if self.is_running():
                    pid = self._get_pid()
                    if pid:
                        process = psutil.Process(pid)
                        self.metrics.cpu_percent = process.cpu_percent()
                        self.metrics.memory_mb = process.memory_info().rss / 1024 / 1024
                        self.metrics.uptime_seconds = time.time() - process.create_time()
                        
                        # Save metrics to a file
                        with open(self.metrics_file, 'w') as f:
                            json.dump(asdict(self.metrics), f, indent=2)
                            
                time.sleep(5)
            except Exception as e:
                DEBUG(f"Monitor error: {e}")
    
    def stop(self):
        """
        Stop the MCP server and perform cleanup.

        Returns:
            bool: True if the server stopped successfully, False otherwise.
        """
        if not self.is_running():
            ERROR("MCP server is not running")
            return False
            
        self._stop_monitor.set()
        
        try:
            pid = self._get_pid()
            if pid:
                if HAS_PSUTIL:
                    process = psutil.Process(pid)
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        process.kill()
                else:
                    # Fallback method
                    os.kill(pid, signal.SIGTERM)
                    
            # Clean up files
            self.pid_file.unlink(missing_ok=True)
            self.metrics_file.unlink(missing_ok=True)
            
            RIGHT("✅ MCP server stopped")
            return True
            
        except Exception as e:
            ERROR(f"Error stopping MCP server: {e}")
            return False
    
    def restart(self, host="localhost", port=8080):
        """
        Restart the MCP server with optimization.

        Args:
            host (str): Server host address. Defaults to "localhost".
            port (int): Server port number. Defaults to 8080.

        Returns:
            bool: True if the server restarted successfully, False otherwise.
        """
        self.stop()
        time.sleep(1)
        return self.start(host, port)
    
    def status(self):
        """
        Get the detailed status of the MCP server.

        Returns:
            Dict[str, Any]: A dictionary containing server status, PID, uptime, memory usage, CPU usage, etc.
        """
        if not self.is_running():
            return {"status": "stopped", "message": "Server is not running"}
            
        try:
            pid = self._get_pid()
            if not pid:
                return {"status": "stopped", "message": "PID file not found"}
                
            if HAS_PSUTIL:
                process = psutil.Process(pid)
                
                # Load metrics if available
                metrics_data = {}
                if self.metrics_file.exists():
                    try:
                        with open(self.metrics_file, 'r') as f:
                            metrics_data = json.load(f)
                    except:
                        pass
                
                status = {
                    "status": "running",
                    "pid": pid,
                    "uptime_seconds": time.time() - process.create_time(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads(),
                    "metrics": metrics_data,
                    "started_at": datetime.fromtimestamp(process.create_time()).isoformat()
                }
                
                return status
            else:
                return {
                    "status": "running", 
                    "pid": pid,
                    "message": "Basic status (psutil not available)"
                }
                
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _get_pid(self):
        """
        Safely get the process ID of the MCP server from the PID file.

        Returns:
            Optional[int]: The process ID if available, None otherwise.
        """
        if not self.pid_file.exists():
            return None
            
        try:
            with open(self.pid_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, FileNotFoundError):
            self.pid_file.unlink(missing_ok=True)
            return None
    
    def is_running(self):
        """
        Check if the MCP server is running.

        Returns:
            bool: True if the server is running, False otherwise.
        """
        pid = self._get_pid()
        if pid is None:
            return False
            
        if HAS_PSUTIL:
            return psutil.pid_exists(pid)
        else:
            # Fallback check
            try:
                os.kill(pid, 0)
                return True
            except (ProcessLookupError, OSError):
                return False
    
    def logs(self, lines=50, follow=False):
        """
        View the MCP server logs.

        Args:
            lines (int): Number of log lines to show. Defaults to 50.
            follow (bool): Whether to follow the log output. Defaults to False.
        """
        if not self.log_file.exists():
            ERROR("No log file found")
            return
            
        try:
            if follow:
                # Tail-like behavior
                with open(self.log_file, 'r') as f:
                    f.seek(0, 2)
                    while True:
                        line = f.readline()
                        if line:
                            print(line.rstrip())
                        else:
                            time.sleep(0.1)
            else:
                with open(self.log_file, 'r') as f:
                    content = f.readlines()
                    for line in content[-lines:]:
                        print(line.rstrip())
        except KeyboardInterrupt:
            pass
        except Exception as e:
            ERROR(f"Error reading logs: {e}")
    
    def reload_tools(self):
        """
        Hot-reload MCP tools.

        Returns:
            bool: True if tools were reloaded successfully, False otherwise.
        """
        if not self.is_running():
            ERROR("Server must be running to reload tools")
            return False
            
        try:
            from model.mcp.server import mcp_server
            if hasattr(mcp_server, 'auto_discover_tools'):
                mcp_server.auto_discover_tools(force_reload=True)
                RIGHT("🔄 Tools reloaded successfully")
                return True
            else:
                RIGHT("🔧 Using restart for tool reload")
                return self.restart()
        except Exception as e:
            ERROR(f"Error reloading tools: {e}")
            return False

def list_mcp_tools(host: str = "localhost", port: int = 8080):
    """
    List all available MCP tools.

    Args:
        host (str): Server host address. Defaults to "localhost".
        port (int): Server port number. Defaults to 8080.
    """
    status = check_mcp_server_status(host, port)
    if not status["running"]:
        ERROR("MCP server is not running. Please start it first.")
        return
    
    try:
        response = requests.get(f"http://{host}:{port}/mcp/capabilities", timeout=10)
        if response.status_code == 200:
            data = response.json()
            capabilities = data.get("capabilities", [])
            
            RIGHT(f"📋 Available MCP Tools ({len(capabilities)} total):")
            RIGHT("=" * 60)
            
            for i, tool in enumerate(capabilities, 1):
                tool_info = tool if isinstance(tool, dict) else tool.dict()
                name = tool_info.get('function_name', 'Unknown')
                desc = tool_info.get('description', 'No description')
                category = tool_info.get('category', 'general')
                
                RIGHT(f"{i:2d}. {name}")
                RIGHT(f"    📝 {desc}")
                RIGHT(f"    🏷️  Category: {category}")
                RIGHT("")
        else:
            ERROR(f"Failed to get tools list: HTTP {response.status_code}")
    except Exception as e:
        ERROR(f"Error listing tools: {e}")

def mcp_command(args):
    """
    Main MCP command handler with enhanced features.

    Args:
        args: Parsed command-line arguments.
    """
    action = args.mcp_action
    host = args.mcp_host
    port = args.mcp_port
    
    RIGHT(f"🔧 MCP Command: {action}")
    
    # Use the advanced manager for enhanced features
    manager = AdvancedMCPServerManager()
    
    if action == "start":
        RIGHT("Starting MCP server...")
        success = manager.start(host, port)
        if success:
            RIGHT("✅ MCP server started successfully")
        else:
            ERROR("❌ Failed to start MCP server")
    
    elif action == "stop":
        RIGHT("Stopping MCP server...")
        success = manager.stop()
        if success:
            RIGHT("✅ MCP server stopped successfully")
        else:
            ERROR("❌ Failed to stop MCP server")
    
    elif action == "restart":
        RIGHT("Restarting MCP server...")
        success = manager.restart(host, port)
        if success:
            RIGHT("✅ MCP server restarted successfully")
        else:
            ERROR("❌ Failed to restart MCP server")
    
    elif action == "status":
        RIGHT("Checking MCP server status...")
        status = manager.status()
        
        if status["status"] == "running":
            RIGHT(f"✅ MCP Server Status: {status['status']}")
            RIGHT(f"📍 Location: {host}:{port}")
            RIGHT(f"🔧 PID: {status.get('pid', 'Unknown')}")
            RIGHT(f"⏱️  Uptime: {status.get('uptime_seconds', 0):.1f}s")
            RIGHT(f"💾 Memory: {status.get('memory_mb', 0):.1f} MB")
            RIGHT(f"⚡ CPU: {status.get('cpu_percent', 0):.1f}%")
            
            if status.get('metrics'):
                RIGHT(f"📊 Metrics: {json.dumps(status['metrics'], indent=2)}")
        else:
            RIGHT(f"❌ MCP Server Status: {status['status']}")
            RIGHT(f"📝 Message: {status['message']}")
        
        # Also check basic server status
        basic_status = check_mcp_server_status(host, port)
        if basic_status["running"]:
            RIGHT("\n" + "="*60)
            list_mcp_tools(host, port)
    
    elif action == "test":
        RIGHT("Testing MCP server functionality...")
        
        # Run async test
        async def run_test():
            success = await test_mcp_server(host, port)
            if success:
                RIGHT("✅ All MCP tests passed!")
            else:
                ERROR("❌ MCP tests failed!")
        
        asyncio.run(run_test())
    
    elif action == "logs":
        RIGHT("Viewing MCP server logs...")
        manager.logs(lines=getattr(args, 'lines', 50), follow=getattr(args, 'follow', False))
    
    elif action == "reload":
        RIGHT("Reloading MCP tools...")
        success = manager.reload_tools()
        if success:
            RIGHT("✅ Tools reloaded successfully")
        else:
            ERROR("❌ Failed to reload tools")
    
    else:
        ERROR(f"Unknown MCP action: {action}")
        RIGHT("Available actions: start, stop, restart, status, test, logs, reload")

if __name__ == "__main__":
    # Enhanced CLI for direct usage
    import argparse
    parser = argparse.ArgumentParser(description="Advanced MCP Server Management")
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status', 'test', 'logs', 'reload'], 
                       help='Action to perform')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--lines', type=int, default=50, help='Lines of log to show')
    parser.add_argument('--follow', action='store_true', help='Follow log output')
    
    args = parser.parse_args()
    args.mcp_action = args.action
    args.mcp_host = args.host
    args.mcp_port = args.port
    
    mcp_command(args)