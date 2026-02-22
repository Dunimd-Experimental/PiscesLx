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
import json
import time
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set, Union, TypeVar, Tuple
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

from .types import (
    POPSSMCPToolMetadata,
    POPSSMCPModuleStats,
    POPSSMCPModuleStatus,
    POPSSMCPToolCategory
)

from .execution import POPSSMCPUnifiedToolExecutor

T = TypeVar('T')

_global_unified_executor = None

def get_unified_tool_executor() -> 'POPSSMCPUnifiedToolExecutor':
    global _global_unified_executor
    if _global_unified_executor is None:
        _global_unified_executor = POPSSMCPUnifiedToolExecutor()
    return _global_unified_executor

@dataclass
class POPSSMCPToolAlias:
    alias: str
    canonical_name: str
    created_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0

@dataclass
class POPSSMCPToolExecution:
    tool_name: str
    execution_id: str
    arguments: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    status: str = "pending"

class POPSSMCPToolRegistry:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._LOG = self._configure_logging()
        
        self.tools: Dict[str, POPSSMCPToolMetadata] = {}
        self.tool_functions: Dict[str, Callable] = {}
        self.tool_modules: Dict[str, Any] = {}
        self.tool_aliases: Dict[str, POPSSMCPToolAlias] = {}
        self.tool_tags: Dict[str, Set[str]] = defaultdict(set)
        
        self.category_tools: Dict[str, Set[str]] = defaultdict(set)
        self.tag_tools: Dict[str, Set[str]] = defaultdict(set)
        
        self.execution_history: List[POPSSMCPToolExecution] = []
        self.active_executions: Dict[str, POPSSMCPToolExecution] = {}
        
        self._lock = threading.RLock()
        
        self.auto_register = getattr(self.config, 'auto_register', True)
        self.strict_validation = getattr(self.config, 'strict_validation', True)
        self.enable_aliases = getattr(self.config, 'enable_aliases', True)
        
        self.registry_stats = {
            'total_registrations': 0,
            'total_aliases': 0,
            'total_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'last_registry_update': None
        }
        
        self._LOG.info("POPSSMCPToolRegistry initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        logger = PiscesLxLogger("PiscesLx.Opss.MCP",file_path=get_log_file("PiscesLx.Opss.MCP"), enable_file=True)
        return logger
    
    def register_tool(
        self,
        tool_name: str,
        description: str,
        category: Union[str, POPSSMCPToolCategory],
        function: Optional[Callable] = None,
        metadata: Optional[POPSSMCPToolMetadata] = None,
        tags: Optional[Set[str]] = None,
        **kwargs
    ) -> bool:
        with self._lock:
            try:
                if tool_name in self.tools:
                    self._LOG.debug(f"Tool already registered: {tool_name}")
                    return self._update_existing_tool(tool_name, description, category, function, metadata, tags, **kwargs)
                
                if isinstance(category, POPSSMCPToolCategory):
                    category_str = category.value
                else:
                    category_str = category
                
                if metadata is None:
                    metadata = POPSSMCPToolMetadata(
                        name=tool_name,
                        description=description,
                        category=category_str,
                        version=kwargs.get('version', '1.0.0'),
                        author=kwargs.get('author', 'PiscesL1 Team'),
                        dependencies=kwargs.get('dependencies', []),
                        performance_score=kwargs.get('performance_score', 1.0),
                        usage_count=0,
                        error_rate=0.0,
                        memory_usage=kwargs.get('memory_usage', None),
                        last_used=None
                    )
                
                self.tools[tool_name] = metadata
                self.tool_functions[tool_name] = function if function else self._default_tool_function
                self.tool_tags[tool_name] = tags if tags else set()
                
                if self.enable_aliases and 'aliases' in kwargs:
                    for alias in kwargs['aliases']:
                        self._register_alias(alias, tool_name)
                
                self.category_tools[category_str].add(tool_name)
                for tag in self.tool_tags[tool_name]:
                    self.tag_tools[tag].add(tool_name)
                
                self.registry_stats['total_registrations'] += 1
                self.registry_stats['last_registry_update'] = datetime.now()
                
                self._LOG.info(f"Tool registered successfully: {tool_name} (category: {category_str})")
                return True
                
            except Exception as e:
                self._LOG.error(f"Failed to register tool {tool_name}: {e}")
                return False
    
    def _update_existing_tool(
        self,
        tool_name: str,
        description: str,
        category: Union[str, POPSSMCPToolCategory],
        function: Optional[Callable] = None,
        metadata: Optional[POPSSMCPToolMetadata] = None,
        tags: Optional[Set[str]] = None,
        **kwargs
    ) -> bool:
        try:
            existing_metadata = self.tools[tool_name]
            
            existing_metadata.description = description
            if isinstance(category, POPSSMCPToolCategory):
                existing_metadata.category = category.value
            else:
                existing_metadata.category = category
            if 'version' in kwargs:
                existing_metadata.version = kwargs['version']
            if 'dependencies' in kwargs:
                existing_metadata.dependencies = kwargs['dependencies']
            if 'performance_score' in kwargs:
                existing_metadata.performance_score = kwargs['performance_score']
            if 'memory_usage' in kwargs:
                existing_metadata.memory_usage = kwargs['memory_usage']
            
            if function is not None:
                self.tool_functions[tool_name] = function
            
            if tags is not None:
                old_tags = self.tool_tags.get(tool_name, set())
                for tag in old_tags:
                    if tag in self.tag_tools and tool_name in self.tag_tools[tag]:
                        self.tag_tools[tag].discard(tool_name)
                self.tool_tags[tool_name] = tags
                for tag in tags:
                    self.tag_tools[tag].add(tool_name)
            
            self.registry_stats['last_registry_update'] = datetime.now()
            
            self._LOG.info(f"Tool updated successfully: {tool_name}")
            return True
            
        except Exception as e:
            self._LOG.error(f"Failed to update tool {tool_name}: {e}")
            return False
    
    def _register_alias(self, alias: str, canonical_name: str) -> bool:
        with self._lock:
            try:
                if alias in self.tool_aliases:
                    existing_alias = self.tool_aliases[alias]
                    if existing_alias.canonical_name != canonical_name:
                        self._LOG.warning(f"Alias '{alias}' already registered for '{existing_alias.canonical_name}', not overwriting")
                        return False
                    return True
                
                if alias in self.tools:
                    self._LOG.warning(f"Alias '{alias}' conflicts with existing tool name")
                    return False
                
                self.tool_aliases[alias] = POPSSMCPToolAlias(
                    alias=alias,
                    canonical_name=canonical_name
                )
                self.registry_stats['total_aliases'] += 1
                
                self._LOG.debug(f"Alias registered: {alias} -> {canonical_name}")
                return True
                
            except Exception as e:
                self._LOG.error(f"Failed to register alias '{alias}': {e}")
                return False
    
    def _default_tool_function(self, arguments: Dict[str, Any]) -> Any:
        return {"status": "executed", "tool": "default"}
    
    def unregister_tool(self, tool_name: str) -> bool:
        with self._lock:
            if tool_name not in self.tools:
                self._LOG.warning(f"Tool not found for unregistration: {tool_name}")
                return False
            
            try:
                metadata = self.tools.pop(tool_name)
                self.tool_functions.pop(tool_name, None)
                self.tool_modules.pop(tool_name, None)
                self.tool_tags.pop(tool_name, None)
                
                category = metadata.category
                if category in self.category_tools:
                    self.category_tools[category].discard(tool_name)
                    if not self.category_tools[category]:
                        del self.category_tools[category]
                
                for tag, tools in self.tag_tools.items():
                    tools.discard(tool_name)
                    if not tools:
                        del self.tag_tools[tag]
                
                aliases_to_remove = [alias for alias, tool_alias in self.tool_aliases.items() 
                                   if tool_alias.canonical_name == tool_name]
                for alias in aliases_to_remove:
                    del self.tool_aliases[alias]
                
                self._LOG.info(f"Tool unregistered: {tool_name}")
                return True
                
            except Exception as e:
                self._LOG.error(f"Failed to unregister tool {tool_name}: {e}")
                return False
    
    def get_tool(self, tool_name: str) -> Optional[POPSSMCPToolMetadata]:
        with self._lock:
            return self.tools.get(tool_name)
    
    def resolve_alias(self, alias: str) -> Optional[str]:
        with self._lock:
            if alias in self.tool_aliases:
                return self.tool_aliases[alias].canonical_name
            return None
    
    def get_tool_by_alias(self, alias: str) -> Optional[POPSSMCPToolMetadata]:
        with self._lock:
            canonical_name = self.resolve_alias(alias)
            if canonical_name:
                return self.tools.get(canonical_name)
            return None
    
    def list_tools(self, category: Optional[str] = None, tags: Optional[Set[str]] = None) -> List[str]:
        with self._lock:
            if category:
                category_tools = self.category_tools.get(category, set())
                if tags:
                    return sorted(self.tools.keys() & category_tools & 
                                set(tool for tag in tags for tool in self.tag_tools.get(tag, set())))
                return sorted(category_tools)
            if tags:
                tag_tools = set()
                for tag in tags:
                    tag_tools |= self.tag_tools.get(tag, set())
                return sorted(tag_tools)
            return sorted(self.tools.keys())
    
    def list_categories(self) -> List[str]:
        with self._lock:
            return sorted(self.category_tools.keys())
    
    def list_tags(self) -> List[str]:
        with self._lock:
            return sorted(self.tag_tools.keys())
    
    def list_aliases(self) -> Dict[str, str]:
        with self._lock:
            return {alias: tool_alias.canonical_name for alias, tool_alias in self.tool_aliases.items()}
    
    def search_tools(self, query: str, search_tags: bool = True, search_description: bool = True) -> List[str]:
        with self._lock:
            query_lower = query.lower()
            matching_tools = []
            
            for tool_name, metadata in self.tools.items():
                if search_description and query_lower in metadata.description.lower():
                    matching_tools.append(tool_name)
                    continue
                
                if search_tags and query_lower in tool_name.lower():
                    matching_tools.append(tool_name)
                    continue
                
                if search_tags:
                    for tag in self.tool_tags.get(tool_name, set()):
                        if query_lower in tag.lower():
                            matching_tools.append(tool_name)
                            break
            
            return sorted(set(matching_tools))
    
    def get_tool_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if tool_name:
                if tool_name not in self.tools:
                    return {}
                metadata = self.tools[tool_name]
                return {
                    'name': tool_name,
                    'category': metadata.category,
                    'version': metadata.version,
                    'usage_count': metadata.usage_count,
                    'error_rate': metadata.error_rate,
                    'performance_score': metadata.performance_score,
                    'has_alias': tool_name in [alias.canonical_name for alias in self.tool_aliases.values()],
                    'tags': list(self.tool_tags.get(tool_name, set())),
                    'last_used': metadata.last_used.isoformat() if metadata.last_used else None
                }
            
            return {
                'total_tools': len(self.tools),
                'total_categories': len(self.category_tools),
                'total_tags': len(self.tag_tools),
                'total_aliases': len(self.tool_aliases),
                'total_registrations': self.registry_stats['total_registrations'],
                'category_distribution': {
                    category: len(tools) for category, tools in self.category_tools.items()
                },
                'tag_distribution': {
                    tag: len(tools) for tag, tools in self.tag_tools.items()
                },
                'average_tools_per_category': len(self.tools) / max(len(self.category_tools), 1)
            }
    
    def record_tool_usage(self, tool_name: str, success: bool = True, execution_time: Optional[float] = None):
        with self._lock:
            if tool_name not in self.tools:
                return
            
            metadata = self.tools[tool_name]
            metadata.usage_count += 1
            metadata.last_used = datetime.now()
            
            if not success:
                metadata.error_rate = min(metadata.error_rate + 0.01, 1.0)
            else:
                metadata.error_rate = max(metadata.error_rate - 0.001, 0.0)
            
            self.registry_stats['total_executions'] += 1
            if not success:
                self.registry_stats['failed_executions'] += 1
            
            if execution_time is not None:
                if self.registry_stats['average_execution_time'] == 0:
                    self.registry_stats['average_execution_time'] = execution_time
                else:
                    self.registry_stats['average_execution_time'] = (
                        self.registry_stats['average_execution_time'] * 0.9 + execution_time * 0.1
                    )
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        with self._lock:
            canonical_name = self.resolve_alias(tool_name) or tool_name
            
            if canonical_name not in self.tools:
                raise ValueError(f"Tool not found: {tool_name}")
            
            function = self.tool_functions.get(canonical_name, self._default_tool_function)
            
            execution_id = f"exec_{uuid.uuid4().hex[:12]}"
            execution = POPSSMCPToolExecution(
                tool_name=canonical_name,
                execution_id=execution_id,
                arguments=arguments,
                start_time=datetime.now(),
                status="running"
            )
            self.active_executions[execution_id] = execution
            
            start_time = time.time()
            result = None
            error = None
            
            try:
                result = function(arguments)
                execution.status = "completed"
                execution.result = result
                self.record_tool_usage(canonical_name, success=True, execution_time=time.time() - start_time)
                
            except Exception as e:
                execution.status = "failed"
                execution.error = str(e)
                execution.result = None
                self.record_tool_usage(canonical_name, success=False, execution_time=time.time() - start_time)
                raise
            
            finally:
                execution.end_time = datetime.now()
                self.execution_history.append(execution)
                if execution_id in self.active_executions:
                    del self.active_executions[execution_id]
            
            return result
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    'execution_id': exec.execution_id,
                    'tool_name': exec.tool_name,
                    'status': exec.status,
                    'start_time': exec.start_time.isoformat(),
                    'duration': (datetime.now() - exec.start_time).total_seconds()
                }
                for exec in self.active_executions.values()
            ]
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            recent = self.execution_history[-limit:]
            return [
                {
                    'execution_id': exec.execution_id,
                    'tool_name': exec.tool_name,
                    'status': exec.status,
                    'start_time': exec.start_time.isoformat(),
                    'end_time': exec.end_time.isoformat() if exec.end_time else None,
                    'duration': (exec.end_time - exec.start_time).total_seconds() if exec.end_time else None,
                    'error': exec.error
                }
                for exec in recent
            ]
    
    def export_registry(self, path: str) -> bool:
        with self._lock:
            try:
                registry_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'version': '1.0',
                    'tools': {
                        name: {
                            'name': metadata.name,
                            'description': metadata.description,
                            'category': metadata.category,
                            'version': metadata.version,
                            'author': metadata.author,
                            'dependencies': metadata.dependencies,
                            'performance_score': metadata.performance_score,
                            'tags': list(self.tool_tags.get(name, set()))
                        }
                        for name, metadata in self.tools.items()
                    },
                    'aliases': {
                        alias: info.canonical_name
                        for alias, info in self.tool_aliases.items()
                    },
                    'categories': {
                        category: list(tools)
                        for category, tools in self.category_tools.items()
                    }
                }
                
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(registry_data, f, indent=2, ensure_ascii=False)
                
                self._LOG.info(f"Registry exported to: {path}")
                return True
                
            except Exception as e:
                self._LOG.error(f"Failed to export registry: {e}")
                return False
    
    def import_registry(self, path: str, overwrite: bool = False) -> int:
        with self._lock:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                
                tools_imported = 0
                
                for tool_name, tool_data in registry_data.get('tools', {}).items():
                    if tool_name in self.tools and not overwrite:
                        self._LOG.debug(f"Tool already registered, skipping: {tool_name}")
                        continue
                    
                    self.register_tool(
                        tool_name=tool_name,
                        description=tool_data['description'],
                        category=tool_data['category'],
                        version=tool_data.get('version', '1.0.0'),
                        author=tool_data.get('author', 'Unknown'),
                        dependencies=tool_data.get('dependencies', []),
                        performance_score=tool_data.get('performance_score', 1.0),
                        tags=set(tool_data.get('tags', []))
                    )
                    tools_imported += 1
                
                for alias, canonical_name in registry_data.get('aliases', {}).items():
                    self._register_alias(alias, canonical_name)
                
                self._LOG.info(f"Registry imported from: {path} ({tools_imported} tools)")
                return tools_imported
                
            except Exception as e:
                self._LOG.error(f"Failed to import registry: {e}")
                return 0
    
    def validate_tool(self, tool_name: str) -> Tuple[bool, Optional[str]]:
        with self._lock:
            canonical_name = self.resolve_alias(tool_name) or tool_name
            
            if canonical_name not in self.tools:
                return False, f"Tool not found: {tool_name}"
            
            metadata = self.tools[canonical_name]
            
            if not metadata.description:
                return False, "Tool description is empty"
            
            if self.strict_validation:
                if not self.tool_functions.get(canonical_name):
                    return False, "Tool function not registered"
            
            return True, None
    
    def batch_register(self, tools: List[Dict[str, Any]]) -> Dict[str, bool]:
        results = {}
        for tool_config in tools:
            tool_name = tool_config.get('name')
            if tool_name:
                results[tool_name] = self.register_tool(**tool_config)
        return results
    
    def get_registry_summary(self) -> Dict[str, Any]:
        with self._lock:
            total_usage = sum(metadata.usage_count for metadata in self.tools.values())
            avg_error_rate = sum(metadata.error_rate for metadata in self.tools.values()) / max(len(self.tools), 1)
            avg_performance = sum(metadata.performance_score for metadata in self.tools.values()) / max(len(self.tools), 1)
            
            return {
                'total_tools': len(self.tools),
                'total_categories': len(self.category_tools),
                'total_tags': len(self.tag_tools),
                'total_aliases': len(self.tool_aliases),
                'total_executions': self.registry_stats['total_executions'],
                'failed_executions': self.registry_stats['failed_executions'],
                'active_executions': len(self.active_executions),
                'execution_history_size': len(self.execution_history),
                'total_usage_count': total_usage,
                'average_error_rate': avg_error_rate,
                'average_performance_score': avg_performance,
                'registry_update': self.registry_stats['last_registry_update'].isoformat() if self.registry_stats['last_registry_update'] else None,
                'top_tools_by_usage': sorted(
                    [(name, metadata.usage_count) for name, metadata in self.tools.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }
    
    def cleanup(self):
        with self._lock:
            self.execution_history = self.execution_history[-1000:]
            
            for tool_name in list(self.tools.keys()):
                if self.tools[tool_name].usage_count == 0:
                    self._LOG.info(f"Removing unused tool: {tool_name}")
                    self.unregister_tool(tool_name)
            
            self._LOG.info("Registry cleanup completed")
