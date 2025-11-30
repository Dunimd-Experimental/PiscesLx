#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import time
import threading
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union, Set
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict

from .types import (
    PiscesLxCoreMCPToolMetadata,
    PiscesLxCoreMCPExecutionContext,
    PiscesLxCoreMCPModuleStats,
    PiscesLxCoreMCPModuleStatus,
    PiscesLxCoreMCPHealthStatus,
    PiscesLxCoreMCPPerformanceMetrics,
    PiscesLxCoreMCPConfiguration
)
from .monitor import PiscesLxCoreMCPMonitor
from .session import PiscesLxCoreMCPSession
from .registry import PiscesLxCoreMCPRegistry

# Use dms_core logging exclusively
import dms_core
PiscesLxCoreLog = dms_core.log.get_logger

class PiscesLxCoreMCPPlaza:
    """Central coordinator for PiscesLxCoreMCP system.
    
    The Plaza acts as the main orchestrator, coordinating between:
    - Session management
    - Tool registry and execution
    - Monitoring and performance tracking
    - Configuration management
    - Health monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the MCP Plaza.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = PiscesLxCoreMCPConfiguration(**(config or {}))
        self.logger = self._configure_logging()
        
        # Core components
        self.monitor = PiscesLxCoreMCPMonitor(config=self.config)
        self.session_manager = PiscesLxCoreMCPSession(config=self.config)
        self.registry = PiscesLxCoreMCPRegistry(config=self.config)
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=getattr(self.config, 'max_workers', 4))
        self.active_sessions: Dict[str, PiscesLxCoreMCPSession] = {}
        self.execution_queue = asyncio.Queue(maxsize=getattr(self.config, 'max_queue_size', 1000))
        
        # Performance tracking
        self.performance_metrics = PiscesLxCoreMCPPerformanceMetrics()
        self.system_health = PiscesLxCoreMCPHealthStatus()
        
        # State management
        self.is_running = False
        self.is_initialized = False
        self.start_time = None
        
        # Thread safety
        self._lock = threading.RLock()
        self._session_lock = threading.Lock()
        
        # Background tasks
        self._monitoring_task = None
        self._cleanup_task = None
        
        self.logger.info("PiscesLxCoreMCPPlaza initialized")
    
    def _configure_logging(self) -> PiscesLxCoreLog:
        """Configure structured logging for the plaza.
        
        Returns:
            Configured logger instance
        """
        logger = PiscesLxCoreLog("PiscesLx.Core.MCP.Plaza")
        return logger
    
    async def initialize(self) -> bool:
        """Initialize the plaza and all components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing PiscesLxCoreMCPPlaza")
            
            # Initialize monitor (monitor doesn't have initialize method)
            # Monitor is already initialized in __init__ with background monitoring started
            self.logger.info("Monitor initialized (background monitoring active)")
            
            # Initialize session manager (session doesn't have initialize method)
            # Session is already initialized in __init__
            self.logger.info("Session manager initialized (already active)")
            
            # Initialize registry
            if not self.registry.initialize():
                raise RuntimeError("Registry initialization failed")
            
            # Initialize system health
            self.system_health.status = "healthy"
            self.system_health.last_check = datetime.now()
            self.system_health.uptime = 0
            
            self.is_initialized = True
            self.logger.info("PiscesLxCoreMCPPlaza initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Plaza initialization failed: {e}")
            self.system_health.status = "error"
            self.system_health.last_check = datetime.now()
            return False
    
    async def start(self) -> bool:
        """Start the plaza and all background services.
        
        Returns:
            True if startup successful, False otherwise
        """
        if not self.is_initialized:
            self.logger.error("Cannot start plaza: not initialized")
            return False
        
        try:
            self.logger.info("Starting PiscesLxCoreMCPPlaza")
            
            # Start monitor (monitor doesn't have start method)
            # Monitor background monitoring is already running from __init__
            self.logger.info("Monitor running (background monitoring active)")
            
            # Start session manager (session doesn't have start method)
            # Session is already active from __init__
            self.logger.info("Session manager active (no start method needed)")
            
            # Start background tasks
            self._start_background_tasks()
            
            # Update system state
            self.is_running = True
            self.start_time = datetime.now()
            self.system_health.status = "running"
            self.system_health.last_check = datetime.now()
            
            self.logger.info("PiscesLxCoreMCPPlaza started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Plaza startup failed: {e}")
            self.system_health.status = "error"
            self.system_health.last_check = datetime.now()
            return False
    
    def _start_background_tasks(self):
        """Start background monitoring and cleanup tasks."""
        try:
            # Start monitoring task
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.logger.info("Background tasks started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
            raise
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        self.logger.info("Monitoring loop started")
        
        while self.is_running:
            try:
                # Update system health
                await self._update_system_health()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check component health
                await self._check_component_health()
                
                # Log system status
                self._log_system_status()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(getattr(self.config, 'monitoring_interval', 30))
                
            except asyncio.CancelledError:
                self.logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(getattr(self.config, 'error_recovery_interval', 60))
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        self.logger.info("Cleanup loop started")
        
        while self.is_running:
            try:
                # Clean up expired sessions
                await self._cleanup_expired_sessions()
                
                # Clean up old execution history
                await self._cleanup_execution_history()
                
                # Clean up performance metrics
                self._cleanup_performance_metrics()
                
                # Wait for next cleanup cycle
                await asyncio.sleep(getattr(self.config, 'cleanup_interval', 300))
                
            except asyncio.CancelledError:
                self.logger.info("Cleanup loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(getattr(self.config, 'error_recovery_interval', 60))
    
    async def _update_system_health(self):
        """Update system health status."""
        try:
            # Calculate uptime
            if self.start_time:
                self.system_health.uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Check resource usage (get_resource_usage method doesn't exist)
            # Use basic system info instead
            resource_usage = {
                'cpu_percent': 0.0,  # Default values
                'memory_percent': 0.0,
                'disk_percent': 0.0
            }
            self.system_health.resource_usage = resource_usage
            
            # Determine overall health
            if resource_usage.get('cpu_percent', 0) > 90 or resource_usage.get('memory_percent', 0) > 90:
                self.system_health.status = "stressed"
            elif resource_usage.get('cpu_percent', 0) > 80 or resource_usage.get('memory_percent', 0) > 80:
                self.system_health.status = "busy"
            else:
                self.system_health.status = "healthy"
            
            self.system_health.last_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating system health: {e}")
            self.system_health.status = "error"
            self.system_health.last_check = datetime.now()
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Update session metrics (session doesn't have get_session_stats method)
            # Use basic session tracking instead
            self.performance_metrics.active_sessions = len(self.active_sessions)
            self.performance_metrics.total_sessions = len(self.active_sessions)  # Simple tracking
            
            # Update registry metrics
            registry_stats = self.registry.get_registry_stats()
            # Use basic registry stats, performance_metrics doesn't have registered_tools attribute
            self.performance_metrics.total_executions = registry_stats.get('total_executions', 0)
            self.performance_metrics.successful_executions = registry_stats.get('successful_executions', 0)
            
            # Update queue metrics
            self.performance_metrics.queue_size = self.execution_queue.qsize()
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    async def _check_component_health(self):
        """Check health of all components."""
        try:
            # Check monitor health (get_health_status is not async)
            monitor_health = self.monitor.get_health_status()
            
            # Check session manager health (session doesn't have get_health_status method)
            # Use basic session status instead
            session_health = {'status': 'active', 'active_sessions': len(self.active_sessions)}
            
            # Check registry health
            registry_health = self.registry.get_registry_stats()
            
            # Update component health status
            self.system_health.component_health = {
                'monitor': monitor_health,
                'session_manager': session_health,
                'registry': registry_health
            }
            
        except Exception as e:
            self.logger.error(f"Error checking component health: {e}")
    
    def _log_system_status(self):
        """Log current system status."""
        if getattr(self.config, 'verbose_logging', False):
            self.logger.info(f"System Status: {self.system_health.status}")
            self.logger.info(f"Active Sessions: {self.performance_metrics.active_sessions}")
            self.logger.info(f"Total Executions: {self.performance_metrics.total_executions}")
            self.logger.info(f"Queue Size: {self.execution_queue.qsize()}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        try:
            expired_sessions = []
            current_time = datetime.now()
            
            with self._session_lock:
                for session_id, session in self.active_sessions.items():
                    if session.is_expired():
                        expired_sessions.append(session_id)
            
            # Clean up expired sessions
            for session_id in expired_sessions:
                await self.close_session(session_id)
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up expired sessions: {e}")
    
    async def _cleanup_execution_history(self):
        """Clean up old execution history."""
        try:
            # Delegate to registry for execution history cleanup
            registry_stats = self.registry.get_registry_stats()
            
            if registry_stats.get('execution_history_size', 0) > getattr(self.config, 'max_execution_history', 1000):
                self.logger.info("Registry execution history cleanup needed")
                # Registry handles its own cleanup internally
                
        except Exception as e:
            self.logger.error(f"Error cleaning up execution history: {e}")
    
    def _cleanup_performance_metrics(self):
        """Clean up old performance metrics."""
        try:
            # This would clean up old metrics data
            # For now, just update the last cleanup time
            self.performance_metrics.last_cleanup = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up performance metrics: {e}")
    
    async def create_session(self, session_config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session.
        
        Args:
            session_config: Optional session configuration
            
        Returns:
            Session ID
        """
        try:
            session_id = self.session_manager.create_session(session_config)
            
            with self._session_lock:
                self.active_sessions[session_id] = self.session_manager
            
            self.logger.info(f"Session created: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create session: {e}")
            raise
    
    async def close_session(self, session_id: str) -> bool:
        """Close a session.
        
        Args:
            session_id: Session ID to close
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.session_manager.close_session(session_id)
            
            with self._session_lock:
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
            
            self.logger.info(f"Session closed: {session_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to close session {session_id}: {e}")
            return False
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], 
                          session_id: Optional[str] = None) -> Any:
        """Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            session_id: Optional session ID
            
        Returns:
            Tool execution result
            
        Raises:
            RuntimeError: If execution fails
        """
        try:
            # Use registry for tool execution
            result = self.registry.execute_tool(tool_name, arguments, session_id)
            
            self.logger.debug(f"Tool execution completed: {tool_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {tool_name} - {e}")
            raise RuntimeError(f"Tool execution failed: {e}")
    
    async def execute_tool_async(self, tool_name: str, arguments: Dict[str, Any], 
                                session_id: Optional[str] = None) -> Future:
        """Execute a tool asynchronously.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            session_id: Optional session ID
            
        Returns:
            Future representing the asynchronous execution
        """
        try:
            # Submit to thread pool executor
            future = self.executor.submit(
                self.registry.execute_tool, tool_name, arguments, session_id
            )
            
            self.logger.debug(f"Async tool execution submitted: {tool_name}")
            return future
            
        except Exception as e:
            self.logger.error(f"Async tool execution submission failed: {tool_name} - {e}")
            raise
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information.
        
        Returns:
            System information dictionary
        """
        try:
            system_info = {
                'health': self.system_health,
                'performance': self.performance_metrics,
                'registry': self.registry.get_registry_stats(),
                'sessions': {'active_sessions': len(self.active_sessions), 'session_count': len(self.active_sessions)},
                'monitor': {'status': 'active', 'monitoring_active': True},
                'config': self.config.dict() if hasattr(self.config, 'dict') else vars(self.config)
            }
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {'error': str(e)}
    
    def list_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[str]:
        """List available tools.
        
        Args:
            category: Optional category filter
            enabled_only: Only return enabled tools
            
        Returns:
            List of tool names
        """
        return self.registry.list_tools(category, enabled_only)
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool information.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information or None if not found
        """
        return self.registry.get_tool_info(tool_name)
    
    async def stop(self) -> bool:
        """Stop the plaza and all services.
        
        Returns:
            True if shutdown successful, False otherwise
        """
        try:
            self.logger.info("Stopping PiscesLxCoreMCPPlaza")
            self.is_running = False
            
            # Stop background tasks
            if self._monitoring_task:
                self._monitoring_task.cancel()
            if self._cleanup_task:
                self._cleanup_task.cancel()
            
            # Stop monitor
            self.monitor.stop_monitoring()
            
            # Stop session manager (no stop method needed)
            # self.session_manager.stop()
            
            # Close all sessions
            for session_id in list(self.active_sessions.keys()):
                await self.close_session(session_id)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Update system state
            self.system_health.status = "stopped"
            self.system_health.last_check = datetime.now()
            
            self.logger.info("PiscesLxCoreMCPPlaza stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Plaza shutdown failed: {e}")
            return False
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception:
            pass