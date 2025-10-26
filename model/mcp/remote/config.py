#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

"""
Configuration management for Remote MCP Client.

This module provides configuration loading, validation, and management
for the ArcticRemoteMCPClient system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict

from .types import RemoteClientConfig, RemoteExecutionMode
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("Arctic.Remote.MCP.Config")

@dataclass
class RemoteMCPConfig:
    """Main configuration for Remote MCP system."""
    
    # Global settings
    enabled: bool = True
    default_timeout: float = 30.0
    max_retry_attempts: int = 3
    connection_pool_size: int = 10
    
    # Routing settings
    default_execution_mode: RemoteExecutionMode = RemoteExecutionMode.AUTO
    routing_enabled: bool = True
    enable_load_balancing: bool = True
    enable_failover: bool = True
    
    # Client settings
    clients: List[RemoteClientConfig] = None
    
    # Security settings
    require_encryption: bool = True
    allowed_client_ids: List[str] = None
    blocked_tool_patterns: List[str] = None
    
    # Performance settings
    heartbeat_interval: float = 30.0
    health_check_interval: float = 60.0
    connection_timeout: float = 10.0
    
    # Logging settings
    log_level: str = "INFO"
    log_requests: bool = True
    log_responses: bool = True
    log_errors: bool = True
    
    def __post_init__(self):
        """Initialize default values."""
        if self.clients is None:
            self.clients = []
        if self.allowed_client_ids is None:
            self.allowed_client_ids = []
        if self.blocked_tool_patterns is None:
            self.blocked_tool_patterns = []

@dataclass
class ToolRegistryConfig:
    """Configuration for tool registration and management."""
    
    # Tool discovery settings
    auto_discovery: bool = True
    discovery_paths: List[str] = None
    discovery_patterns: List[str] = None
    
    # Tool filtering settings
    enable_whitelist: bool = False
    whitelisted_tools: List[str] = None
    enable_blacklist: bool = True
    blacklisted_tools: List[str] = None
    
    # Tool metadata settings
    require_metadata: bool = True
    validate_parameters: bool = True
    
    def __post_init__(self):
        """Initialize default values."""
        if self.discovery_paths is None:
            self.discovery_paths = ["tools", "mcp_tools", "user_tools"]
        if self.discovery_patterns is None:
            self.discovery_patterns = ["*.json", "mcp.json"]
        if self.whitelisted_tools is None:
            self.whitelisted_tools = []
        if self.blacklisted_tools is None:
            self.blacklisted_tools = [
                "rm", "del", "format", "fdisk", "rm -rf",
                "sudo", "su", "passwd", "useradd", "userdel"
            ]

class RemoteMCPConfigManager:
    """
    Configuration manager for Remote MCP system.
    
    Handles loading, validation, and management of configuration files.
    """
    
    DEFAULT_CONFIG_FILES = [
        "mcp_remote.json",
        "mcp_remote_config.json",
        "~/.mcp/remote_config.json",
        "/etc/mcp/remote_config.json"
    ]
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = self._resolve_config_path(config_path)
        self.config = RemoteMCPConfig()
        self.tool_registry_config = ToolRegistryConfig()
        
        logger.info(f"Initialized config manager with path: {self.config_path}")
    
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolve configuration file path."""
        if config_path:
            path = Path(config_path).expanduser().resolve()
            if path.exists():
                return path
            else:
                logger.warning(f"Specified config file not found: {path}")
                return None
        
        # Search for default config files
        for config_file in self.DEFAULT_CONFIG_FILES:
            path = Path(config_file).expanduser().resolve()
            if path.exists():
                logger.info(f"Found config file: {path}")
                return path
        
        return None
    
    async def load_config(self) -> RemoteMCPConfig:
        """
        Load configuration from file.
        
        Returns:
            Loaded configuration
        """
        if not self.config_path or not self.config_path.exists():
            logger.info("No config file found, using defaults")
            return self.config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Parse main configuration
            if 'remote_mcp' in config_data:
                main_config = config_data['remote_mcp']
                self._parse_main_config(main_config)
            
            # Parse tool registry configuration
            if 'tool_registry' in config_data:
                registry_config = config_data['tool_registry']
                self._parse_tool_registry_config(registry_config)
            
            # Parse client configurations
            if 'clients' in config_data:
                clients_config = config_data['clients']
                await self._parse_clients_config(clients_config)
            
            logger.info(f"Successfully loaded configuration from {self.config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration")
            return self.config
    
    def _parse_main_config(self, config_data: Dict[str, Any]) -> None:
        """Parse main configuration section."""
        # Global settings
        if 'enabled' in config_data:
            self.config.enabled = config_data['enabled']
        
        if 'default_timeout' in config_data:
            self.config.default_timeout = float(config_data['default_timeout'])
        
        if 'max_retry_attempts' in config_data:
            self.config.max_retry_attempts = int(config_data['max_retry_attempts'])
        
        if 'connection_pool_size' in config_data:
            self.config.connection_pool_size = int(config_data['connection_pool_size'])
        
        # Routing settings
        if 'routing' in config_data:
            routing_config = config_data['routing']
            
            if 'default_execution_mode' in routing_config:
                mode_str = routing_config['default_execution_mode'].upper()
                self.config.default_execution_mode = RemoteExecutionMode(mode_str)
            
            if 'enabled' in routing_config:
                self.config.routing_enabled = routing_config['enabled']
            
            if 'enable_load_balancing' in routing_config:
                self.config.enable_load_balancing = routing_config['enable_load_balancing']
            
            if 'enable_failover' in routing_config:
                self.config.enable_failover = routing_config['enable_failover']
        
        # Security settings
        if 'security' in config_data:
            security_config = config_data['security']
            
            if 'require_encryption' in security_config:
                self.config.require_encryption = security_config['require_encryption']
            
            if 'allowed_client_ids' in security_config:
                self.config.allowed_client_ids = security_config['allowed_client_ids']
            
            if 'blocked_tool_patterns' in security_config:
                self.config.blocked_tool_patterns = security_config['blocked_tool_patterns']
        
        # Performance settings
        if 'performance' in config_data:
            perf_config = config_data['performance']
            
            if 'heartbeat_interval' in perf_config:
                self.config.heartbeat_interval = float(perf_config['heartbeat_interval'])
            
            if 'health_check_interval' in perf_config:
                self.config.health_check_interval = float(perf_config['health_check_interval'])
            
            if 'connection_timeout' in perf_config:
                self.config.connection_timeout = float(perf_config['connection_timeout'])
        
        # Logging settings
        if 'logging' in config_data:
            logging_config = config_data['logging']
            
            if 'log_level' in logging_config:
                self.config.log_level = logging_config['log_level'].upper()
            
            if 'log_requests' in logging_config:
                self.config.log_requests = logging_config['log_requests']
            
            if 'log_responses' in logging_config:
                self.config.log_responses = logging_config['log_responses']
            
            if 'log_errors' in logging_config:
                self.config.log_errors = logging_config['log_errors']
    
    def _parse_tool_registry_config(self, config_data: Dict[str, Any]) -> None:
        """Parse tool registry configuration section."""
        # Tool discovery settings
        if 'auto_discovery' in config_data:
            self.tool_registry_config.auto_discovery = config_data['auto_discovery']
        
        if 'discovery_paths' in config_data:
            self.tool_registry_config.discovery_paths = config_data['discovery_paths']
        
        if 'discovery_patterns' in config_data:
            self.tool_registry_config.discovery_patterns = config_data['discovery_patterns']
        
        # Tool filtering settings
        if 'filtering' in config_data:
            filtering_config = config_data['filtering']
            
            if 'enable_whitelist' in filtering_config:
                self.tool_registry_config.enable_whitelist = filtering_config['enable_whitelist']
            
            if 'whitelisted_tools' in filtering_config:
                self.tool_registry_config.whitelisted_tools = filtering_config['whitelisted_tools']
            
            if 'enable_blacklist' in filtering_config:
                self.tool_registry_config.enable_blacklist = filtering_config['enable_blacklist']
            
            if 'blacklisted_tools' in filtering_config:
                self.tool_registry_config.blacklisted_tools = filtering_config['blacklisted_tools']
        
        # Tool metadata settings
        if 'metadata' in config_data:
            metadata_config = config_data['metadata']
            
            if 'require_metadata' in metadata_config:
                self.tool_registry_config.require_metadata = metadata_config['require_metadata']
            
            if 'validate_parameters' in metadata_config:
                self.tool_registry_config.validate_parameters = metadata_config['validate_parameters']
    
    async def _parse_clients_config(self, clients_data: List[Dict[str, Any]]) -> None:
        """Parse client configurations."""
        self.config.clients = []
        
        for client_data in clients_data:
            try:
                client_config = self._create_client_config(client_data)
                self.config.clients.append(client_config)
                logger.debug(f"Added client config: {client_config.client_id}")
            except Exception as e:
                logger.error(f"Failed to parse client config: {e}")
    
    def _create_client_config(self, client_data: Dict[str, Any]) -> RemoteClientConfig:
        """Create RemoteClientConfig from data."""
        return RemoteClientConfig(
            client_id=client_data['client_id'],
            server_url=client_data['server_url'],
            capabilities=client_data.get('capabilities', []),
            metadata=client_data.get('metadata', {}),
            connection_timeout=client_data.get('connection_timeout', self.config.connection_timeout),
            heartbeat_interval=client_data.get('heartbeat_interval', self.config.heartbeat_interval),
            retry_attempts=client_data.get('retry_attempts', self.config.max_retry_attempts)
        )
    
    async def save_config(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Optional output path
        """
        if output_path:
            save_path = Path(output_path).expanduser().resolve()
        else:
            save_path = self.config_path or Path("mcp_remote_config.json")
        
        try:
            # Convert to serializable format
            config_data = {
                "remote_mcp": self._serialize_main_config(),
                "tool_registry": self._serialize_tool_registry_config(),
                "clients": self._serialize_clients_config()
            }
            
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _serialize_main_config(self) -> Dict[str, Any]:
        """Serialize main configuration."""
        return {
            "enabled": self.config.enabled,
            "default_timeout": self.config.default_timeout,
            "max_retry_attempts": self.config.max_retry_attempts,
            "connection_pool_size": self.config.connection_pool_size,
            "routing": {
                "default_execution_mode": self.config.default_execution_mode.value,
                "enabled": self.config.routing_enabled,
                "enable_load_balancing": self.config.enable_load_balancing,
                "enable_failover": self.config.enable_failover
            },
            "security": {
                "require_encryption": self.config.require_encryption,
                "allowed_client_ids": self.config.allowed_client_ids,
                "blocked_tool_patterns": self.config.blocked_tool_patterns
            },
            "performance": {
                "heartbeat_interval": self.config.heartbeat_interval,
                "health_check_interval": self.config.health_check_interval,
                "connection_timeout": self.config.connection_timeout
            },
            "logging": {
                "log_level": self.config.log_level,
                "log_requests": self.config.log_requests,
                "log_responses": self.config.log_responses,
                "log_errors": self.config.log_errors
            }
        }
    
    def _serialize_tool_registry_config(self) -> Dict[str, Any]:
        """Serialize tool registry configuration."""
        return {
            "auto_discovery": self.tool_registry_config.auto_discovery,
            "discovery_paths": self.tool_registry_config.discovery_paths,
            "discovery_patterns": self.tool_registry_config.discovery_patterns,
            "filtering": {
                "enable_whitelist": self.tool_registry_config.enable_whitelist,
                "whitelisted_tools": self.tool_registry_config.whitelisted_tools,
                "enable_blacklist": self.tool_registry_config.enable_blacklist,
                "blacklisted_tools": self.tool_registry_config.blacklisted_tools
            },
            "metadata": {
                "require_metadata": self.tool_registry_config.require_metadata,
                "validate_parameters": self.tool_registry_config.validate_parameters
            }
        }
    
    def _serialize_clients_config(self) -> List[Dict[str, Any]]:
        """Serialize client configurations."""
        clients_data = []
        
        for client_config in self.config.clients:
            client_data = {
                "client_id": client_config.client_id,
                "server_url": client_config.server_url,
                "capabilities": client_config.capabilities,
                "metadata": client_config.metadata
            }
            
            # Only include optional fields if they differ from defaults
            if client_config.connection_timeout != self.config.connection_timeout:
                client_data["connection_timeout"] = client_config.connection_timeout
            
            if client_config.heartbeat_interval != self.config.heartbeat_interval:
                client_data["heartbeat_interval"] = client_config.heartbeat_interval
            
            if client_config.retry_attempts != self.config.max_retry_attempts:
                client_data["retry_attempts"] = client_config.retry_attempts
            
            clients_data.append(client_data)
        
        return clients_data
    
    def get_config(self) -> RemoteMCPConfig:
        """Get current configuration."""
        return self.config
    
    def get_tool_registry_config(self) -> ToolRegistryConfig:
        """Get tool registry configuration."""
        return self.tool_registry_config
    
    def validate_config(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate main configuration
        if self.config.default_timeout <= 0:
            errors.append("default_timeout must be positive")
        
        if self.config.max_retry_attempts < 0:
            errors.append("max_retry_attempts cannot be negative")
        
        if self.config.connection_pool_size <= 0:
            errors.append("connection_pool_size must be positive")
        
        # Validate client configurations
        for i, client_config in enumerate(self.config.clients):
            if not client_config.client_id:
                errors.append(f"Client {i}: client_id cannot be empty")
            
            if not client_config.server_url:
                errors.append(f"Client {i}: server_url cannot be empty")
            
            if client_config.connection_timeout <= 0:
                errors.append(f"Client {i}: connection_timeout must be positive")
            
            if client_config.heartbeat_interval <= 0:
                errors.append(f"Client {i}: heartbeat_interval must be positive")
            
            if client_config.retry_attempts < 0:
                errors.append(f"Client {i}: retry_attempts cannot be negative")
        
        return errors
    
    async def create_default_config(self, output_path: str = "mcp_remote_config.json") -> None:
        """
        Create a default configuration file.
        
        Args:
            output_path: Output file path
        """
        # Reset to defaults
        self.config = RemoteMCPConfig()
        self.tool_registry_config = ToolRegistryConfig()
        
        # Add some example clients
        self.config.clients = [
            RemoteClientConfig(
                client_id="user_local_client_1",
                server_url="http://localhost:8080",
                capabilities=["file_operations", "shell_commands", "calculator"],
                metadata={
                    "user_id": "example_user",
                    "platform": "windows",
                    "location": "user_local"
                }
            ),
            RemoteClientConfig(
                client_id="user_local_client_2",
                server_url="http://localhost:8081",
                capabilities=["text_processing", "data_analysis", "web_tools"],
                metadata={
                    "user_id": "example_user",
                    "platform": "linux",
                    "location": "user_local"
                }
            )
        ]
        
        # Save the configuration
        await self.save_config(output_path)
        
        logger.info(f"Default configuration created at {output_path}")

# Global configuration manager instance
_config_manager: Optional[RemoteMCPConfigManager] = None

async def get_config_manager(config_path: Optional[Union[str, Path]] = None) -> RemoteMCPConfigManager:
    """
    Get the global configuration manager instance.
    
    Args:
        config_path: Optional configuration file path
        
    Returns:
        Configuration manager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = RemoteMCPConfigManager(config_path)
        await _config_manager.load_config()
    
    return _config_manager

def reset_config_manager() -> None:
    """Reset the global configuration manager instance."""
    global _config_manager
    _config_manager = None