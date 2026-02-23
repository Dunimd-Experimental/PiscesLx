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

import os
import sys
import json
import time
import signal
import threading
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from contextlib import contextmanager
import tempfile
import hashlib
import resource

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir

_LOG = PiscesLxLogger("PiscesLx.Opss.MCP",file_path=get_log_file("PiscesLx.Opss.MCP"), enable_file=True)

@dataclass
class POPSSSandboxConfig:
    enabled: bool = True
    
    timeout_seconds: float = 30.0
    max_memory_mb: int = 512
    max_cpu_time_seconds: float = 60.0
    max_output_size_bytes: int = 1024 * 1024
    
    allow_network: bool = False
    allowed_hosts: List[str] = field(default_factory=list)
    
    allowed_directories: List[str] = field(default_factory=lambda: ["./workspace", "/tmp"])
    blocked_paths: List[str] = field(default_factory=lambda: ["/etc", "/usr", "/bin", "/sbin"])
    
    allow_syscalls: List[str] = field(default_factory=list)
    blocked_syscalls: List[str] = field(default_factory=lambda: [
        "execve", "clone", "fork", "vfork", "kill", "ptrace",
        "mount", "umount", "chmod", "chown", "unlink",
    ])
    
    max_file_size_bytes: int = 10 * 1024 * 1024
    max_files_open: int = 64
    
    enable_logging: bool = True
    log_level: str = "INFO"
    
    capture_output: bool = True
    sanitize_output: bool = True
    
    isolation_level: str = "container"
    
    def __post_init__(self):
        if self.isolation_level == "container":
            self.timeout_seconds = min(self.timeout_seconds, 60.0)
            self.max_memory_mb = min(self.max_memory_mb, 1024)

class POPSSExecutionResult:
    def __init__(
        self,
        success: bool,
        output: str = "",
        error: Optional[str] = None,
        return_code: int = 0,
        execution_time: float = 0.0,
        memory_used: int = 0,
        cpu_time: float = 0.0,
        killed: bool = False,
        killed_reason: Optional[str] = None,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.return_code = return_code
        self.execution_time = execution_time
        self.memory_used = memory_used
        self.cpu_time = cpu_time
        self.killed = killed
        self.killed_reason = killed_reason
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "return_code": self.return_code,
            "execution_time": self.execution_time,
            "memory_used": self.memory_used,
            "cpu_time": self.cpu_time,
            "killed": self.killed,
            "killed_reason": self.killed_reason,
        }
    
    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"POPSSExecutionResult({status}, time={self.execution_time:.3f}s)"

class POPSSToolSandbox:
    def __init__(self, config: Optional[POPSSSandboxConfig] = None):
        self.config = config or POPSSSandboxConfig()
        
        self._execution_count = 0
        self._failed_count = 0
        self._total_time = 0.0
        
        self._workspace_dir = Path(get_work_dir("workspace"))
        self._workspace_dir.mkdir(parents=True, exist_ok=True)
        
        _LOG.info("POPSSToolSandbox initialized")
        _LOG.info(f"  - Timeout: {self.config.timeout_seconds}s")
        _LOG.info(f"  - Memory limit: {self.config.max_memory_mb}MB")
        _LOG.info(f"  - Network allowed: {self.config.allow_network}")
        _LOG.info(f"  - Isolation level: {self.config.isolation_level}")
    
    @contextmanager
    def run(
        self,
        tool_name: str,
        code: str,
        language: str = "python",
        **kwargs
    ):
        if not self.config.enabled:
            yield self._run_unsafe(code, language, **kwargs)
            return
        
        start_time = time.time()
        
        if self.config.isolation_level == "container":
            result = self._run_in_container(tool_name, code, language, **kwargs)
        elif self.config.isolation_level == "process":
            result = self._run_in_process(tool_name, code, language, **kwargs)
        else:
            result = self._run_in_process(tool_name, code, language, **kwargs)
        
        result.execution_time = time.time() - start_time
        
        self._execution_count += 1
        self._total_time += result.execution_time
        
        if not result.success:
            self._failed_count += 1
        
        if self.config.enable_logging:
            _LOG.info(
                f"Tool execution: {tool_name} - "
                f"{'SUCCESS' if result.success else 'FAILED'} "
                f"({result.execution_time:.3f}s)"
            )
        
        yield result
    
    def _run_in_container(
        self,
        tool_name: str,
        code: str,
        language: str,
        **kwargs
    ) -> POPSSExecutionResult:
        try:
            import docker
            from docker.types import ResourceLimits, HostConfig
            
            client = docker.from_env()
            
            image = f"pisceslx-tool-{language}:latest"
            
            memory_limit = self.config.max_memory_mb * 1024 * 1024
            
            host_config = HostConfig(
                memory=memory_limit,
                cpu_period=100000,
                cpu_quota=50000,
                network_mode="none" if not self.config.allow_network else "bridge",
                read_only_rootfs=False,
                tmpfs={"/tmp": "size=10M,mode=1777"},
            )
            
            command = self._get_container_command(code, language)
            
            container = client.containers.run(
                image=image,
                command=command,
                detach=True,
                host_config=host_config,
                stderr=True,
                stdout=True,
                remove=True,
            )
            
            try:
                container.wait(timeout=self.config.timeout_seconds)
                
                output = container.logs(stdout=True, stderr=True).decode("utf-8")
                
                container.reload()
                exit_code = container.exit_code
                
                return POPSSExecutionResult(
                    success=(exit_code == 0),
                    output=output,
                    return_code=exit_code,
                )
                
            finally:
                container.stop(timeout=5)
                container.remove(force=True)
                
        except ImportError:
            _LOG.warning("Docker not available, falling back to process isolation")
            return self._run_in_process(tool_name, code, language, **kwargs)
        except Exception as e:
            _LOG.error(f"Container execution failed: {e}")
            return POPSSExecutionResult(
                success=False,
                error=str(e),
                killed=True,
                killed_reason="container_error",
            )
    
    def _run_in_process(
        self,
        tool_name: str,
        code: str,
        language: str,
        **kwargs
    ) -> POPSSExecutionResult:
        start_time = time.time()
        
        result = None
        killed = False
        killed_reason = None
        
        try:
            with self._resource_limits():
                result = self._execute_code(code, language, **kwargs)
                
        except MemoryError:
            killed = True
            killed_reason = "memory_limit"
            result = POPSSExecutionResult(
                success=False,
                error="Memory limit exceeded",
                killed=True,
                killed_reason="memory_limit",
            )
        except TimeoutError:
            killed = True
            killed_reason = "timeout"
            result = POPSSExecutionResult(
                success=False,
                error="Execution timeout",
                killed=True,
                killed_reason="timeout",
            )
        except Exception as e:
            result = POPSSExecutionResult(
                success=False,
                error=str(e),
                killed=killed,
                killed_reason=killed_reason,
            )
        
        if result is None:
            result = POPSSExecutionResult(
                success=False,
                error="Unknown execution error",
            )
        
        return result
    
    @contextmanager
    def _resource_limits(self):
        def set_limits():
            try:
                max_memory = self.config.max_memory_mb * 1024 * 1024
                
                resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
                resource.setrlimit(resource.RLIMIT_DATA, (max_memory, max_memory))
                resource.setrlimit(resource.RLIMIT_STACK, (max_memory, max_memory))
                
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (int(self.config.max_cpu_time_seconds), int(self.config.max_cpu_time_seconds) + 1)
                )
                
                resource.setrlimit(resource.RLIMIT_NOFILE, (self.config.max_files_open, self.config.max_files_open))
                
            except (ValueError, resource.error) as e:
                _LOG.warning(f"Failed to set resource limits: {e}")
        
        old_limits = []
        try:
            set_limits()
            yield
        finally:
            pass
    
    def _execute_code(
        self,
        code: str,
        language: str,
        **kwargs
    ) -> POPSSExecutionResult:
        handlers = {
            "python": self._execute_python,
            "javascript": self._execute_javascript,
            "bash": self._execute_bash,
            "shell": self._execute_bash,
        }
        
        handler = handlers.get(language.lower(), self._execute_python)
        
        return handler(code, **kwargs)
    
    def _execute_python(
        self,
        code: str,
        **kwargs
    ) -> POPSSExecutionResult:
        try:
            from io import StringIO
            import sys
            
            output_capture = StringIO()
            error_capture = StringIO()
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = output_capture
            sys.stderr = error_capture
            
            try:
                exec(compile(code, "<sandbox>", "exec"), {"__builtins__": self._get_restricted_builtins()})
                
                stdout_output = output_capture.getvalue()
                stderr_output = error_capture.getvalue()
                
                return POPSSExecutionResult(
                    success=True,
                    output=stdout_output,
                    error=stderr_output if stderr_output else None,
                    return_code=0,
                )
                
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except SyntaxError as e:
            return POPSSExecutionResult(
                success=False,
                error=f"SyntaxError: {e.msg} (line {e.lineno})",
                return_code=1,
            )
        except Exception as e:
            return POPSSExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                return_code=1,
            )
    
    def _get_restricted_builtins(self) -> Dict[str, Any]:
        safe_builtins = {
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "pow": pow,
            "divmod": divmod,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "delattr": delattr,
            "property": property,
            "classmethod": classmethod,
            "staticmethod": staticmethod,
            "super": super,
            "object": object,
            "type": type,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "RuntimeError": RuntimeError,
        }
        
        return safe_builtins
    
    def _execute_javascript(
        self,
        code: str,
        **kwargs
    ) -> POPSSExecutionResult:
        try:
            import subprocess
            
            result = subprocess.run(
                ["node", "-e", code],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            
            return POPSSExecutionResult(
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                return_code=result.returncode,
            )
            
        except FileNotFoundError:
            return POPSSExecutionResult(
                success=False,
                error="Node.js not available",
                return_code=-1,
            )
        except subprocess.TimeoutExpired:
            return POPSSExecutionResult(
                success=False,
                error="Execution timeout",
                killed=True,
                killed_reason="timeout",
            )
        except Exception as e:
            return POPSSExecutionResult(
                success=False,
                error=str(e),
                return_code=-1,
            )
    
    def _execute_bash(
        self,
        code: str,
        **kwargs
    ) -> POPSSExecutionResult:
        try:
            import subprocess
            
            safe_code = self._sanitize_bash(code)
            
            result = subprocess.run(
                ["/bin/bash", "-c", safe_code],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                cwd=str(self._workspace_dir),
            )
            
            return POPSSExecutionResult(
                success=(result.returncode == 0),
                output=result.stdout,
                error=result.stderr if result.stderr else None,
                return_code=result.returncode,
            )
            
        except subprocess.TimeoutExpired:
            return POPSSExecutionResult(
                success=False,
                error="Execution timeout",
                killed=True,
                killed_reason="timeout",
            )
        except Exception as e:
            return POPSSExecutionResult(
                success=False,
                error=str(e),
                return_code=-1,
            )
    
    def _sanitize_bash(self, code: str) -> str:
        dangerous = [
            "rm -rf", "mkfs", "dd if=", "cat /dev/", "tail -f",
            ":(){:|:&};:", "chmod 777", "chown", "wget", "curl",
        ]
        
        sanitized = code
        for pattern in dangerous:
            if pattern in code.lower():
                _LOG.warning(f"Potentially dangerous pattern detected: {pattern}")
        
        return sanitized
    
    def _run_unsafe(
        self,
        code: str,
        language: str,
        **kwargs
    ) -> POPSSExecutionResult:
        _LOG.warning("Running code WITHOUT sandbox - FOR DEBUGGING ONLY")
        
        return self._execute_code(code, language, **kwargs)
    
    def _get_container_command(self, code: str, language: str) -> str:
        import base64
        
        encoded_code = base64.b64encode(code.encode()).decode()
        
        if language == "python":
            return f"python3 - <<'PY'\nimport base64\ncode = base64.b64decode('{encoded_code}').decode()\nexec(compile(code, '<sandbox>', 'exec'))\nPY"
        else:
            return f"echo '{encoded_code}' | base64 -d | {language}"
    
    def validate_code(self, code: str, language: str = "python") -> Tuple[bool, Optional[str]]:
        dangerous_patterns = [
            (r"import\s+os\s*;", "os module import"),
            (r"subprocess", "subprocess module"),
            (r"eval\s*\(", "eval function"),
            (r"exec\s*\(", "exec function"),
            (r"__import__", "dynamic imports"),
            (r"open\s*\(", "file operations"),
            (r"sys\.exit", "system exit"),
            (r"os\.chmod", "permission changes"),
            (r"os\.remove", "file deletion"),
        ]
        
        import re
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Potentially dangerous pattern: {description}"
        
        if len(code) > 10000:
            return False, "Code too long"
        
        return True, None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_executions": self._execution_count,
            "failed_executions": self._failed_count,
            "success_rate": (
                (self._execution_count - self._failed_count) / 
                max(1, self._execution_count)
            ),
            "total_time": self._total_time,
            "avg_time": self._total_time / max(1, self._execution_count),
            "config": {
                "timeout": self.config.timeout_seconds,
                "memory_limit_mb": self.config.max_memory_mb,
                "network_allowed": self.config.allow_network,
                "isolation_level": self.config.isolation_level,
            }
        }

def create_tool_sandbox(
    config: Optional[POPSSSandboxConfig] = None,
) -> POPSSToolSandbox:
    return POPSSToolSandbox(config)
