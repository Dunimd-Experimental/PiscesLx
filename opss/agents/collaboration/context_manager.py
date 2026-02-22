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

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

from utils.dc import PiscesLxLogger

class POPSSContextScope(Enum):
    GLOBAL = "global"
    SESSION = "session"
    TASK = "task"
    AGENT = "agent"
    CONVERSATION = "conversation"

class POPSSContextPriority(Enum):
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20

@dataclass
class POPSSContextEntry:
    entry_id: str
    key: str
    value: Any
    
    scope: POPSSContextScope
    priority: POPSSContextPriority = POPSSContextPriority.NORMAL
    
    owner_id: Optional[str] = None
    parent_entry_id: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    access_count: int = 0
    is_persistent: bool = False
    is_locked: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class POPSSContextSnapshot:
    snapshot_id: str
    context_id: str
    
    entries: Dict[str, POPSSContextEntry] = field(default_factory=dict)
    
    scope: POPSSContextScope
    owner_id: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class POPSSContextManagerConfig:
    max_entries: int = 10000
    max_memory_usage_mb: int = 512
    
    default_ttl_seconds: int = 3600
    cleanup_interval_seconds: int = 300
    
    enable_compression: bool = True
    enable_deduplication: bool = True
    
    max_snapshot_count: int = 100
    snapshot_retention_days: int = 7
    
    scope_hierarchy: Dict[POPSSContextScope, int] = field(default_factory=lambda: {
        POPSSContextScope.GLOBAL: 100,
        POPSSContextScope.SESSION: 80,
        POPSSContextScope.TASK: 60,
        POPSSContextScope.AGENT: 40,
        POPSSContextScope.CONVERSATION: 20,
    })

class POPSSContextManager:
    def __init__(self, config: Optional[POPSSContextManagerConfig] = None):
        self.config = config or POPSSContextManagerConfig()
        self._LOG = self._configure_logging()
        
        self._contexts: Dict[str, Dict[str, POPSSContextEntry]] = {}
        self._context_locks: Dict[str, any] = {}
        
        self._snapshots: Dict[str, POPSSContextSnapshot] = {}
        
        self._access_history: List[Dict[str, Any]] = []
        
        self._metrics: Dict[str, Any] = {
            'total_entries': 0,
            'total_accesses': 0,
            'total_snapshots': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        self._async_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="piscesl1_context_manager"
        )
        
        self._LOG.info("POPSSContextManager initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        logger = get_logger("PiscesLx.Core.Agents.Collaboration.ContextManager")
        return logger
    
    def create_context(self, context_id: str, scope: POPSSContextScope, 
                     owner_id: Optional[str] = None) -> bool:
        if context_id in self._contexts:
            return False
        
        self._contexts[context_id] = {}
        self._context_locks[context_id] = __import__('threading').RLock()
        
        self._LOG.info(f"Created context: {context_id} (scope: {scope.value})")
        return True
    
    def delete_context(self, context_id: str) -> bool:
        if context_id not in self._contexts:
            return False
        
        del self._contexts[context_id]
        if context_id in self._context_locks:
            del self._context_locks[context_id]
        
        self._LOG.info(f"Deleted context: {context_id}")
        return True
    
    def set(
        self,
        context_id: str,
        key: str,
        value: Any,
        scope: POPSSContextScope,
        priority: POPSSContextPriority = POPSSContextPriority.NORMAL,
        owner_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        is_persistent: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        if context_id not in self._contexts:
            return False
        
        entry = POPSSContextEntry(
            entry_id=f"entry_{uuid.uuid4().hex[:12]}",
            key=key,
            value=value,
            scope=scope,
            priority=priority,
            owner_id=owner_id,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=ttl_seconds or self.config.default_ttl_seconds) 
                       if ttl_seconds is not None else None,
            is_persistent=is_persistent,
            metadata=metadata or {},
        )
        
        self._contexts[context_id][key] = entry
        self._metrics['total_entries'] += 1
        
        self._LOG.debug(f"Set context entry: {context_id}.{key}")
        return True
    
    def get(
        self,
        context_id: str,
        key: str,
        default: Any = None,
        update_access: bool = True
    ) -> Any:
        if context_id not in self._contexts:
            self._metrics['cache_misses'] += 1
            return default
        
        if key not in self._contexts[context_id]:
            self._metrics['cache_misses'] += 1
            return default
        
        entry = self._contexts[context_id][key]
        
        if entry.expires_at and datetime.now() > entry.expires_at:
            del self._contexts[context_id][key]
            self._metrics['cache_misses'] += 1
            return default
        
        if update_access:
            entry.accessed_at = datetime.now()
            entry.access_count += 1
        
        self._metrics['cache_hits'] += 1
        self._metrics['total_accesses'] += 1
        
        self._access_history.append({
            'context_id': context_id,
            'key': key,
            'access_type': 'get',
            'timestamp': datetime.now().isoformat(),
        })
        
        if len(self._access_history) > 10000:
            self._access_history = self._access_history[-5000:]
        
        return entry.value
    
    def get_all(
        self,
        context_id: str,
        scope: Optional[POPSSContextScope] = None,
        include_expired: bool = False
    ) -> Dict[str, Any]:
        if context_id not in self._contexts:
            return {}
        
        result = {}
        now = datetime.now()
        
        for key, entry in self._contexts[context_id].items():
            if scope and entry.scope != scope:
                continue
            
            if not include_expired and entry.expires_at and now > entry.expires_at:
                continue
            
            result[key] = entry.value
        
        return result
    
    def delete(self, context_id: str, key: str) -> bool:
        if context_id not in self._contexts:
            return False
        
        if key in self._contexts[context_id]:
            del self._contexts[context_id][key]
            self._LOG.debug(f"Deleted context entry: {context_id}.{key}")
            return True
        
        return False
    
    def clear(self, context_id: str, scope: Optional[POPSSContextScope] = None) -> int:
        if context_id not in self._contexts:
            return 0
        
        count = 0
        keys_to_delete = []
        
        for key, entry in self._contexts[context_id].items():
            if scope is None or entry.scope == scope:
                if not entry.is_locked:
                    keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self._contexts[context_id][key]
            count += 1
        
        self._LOG.info(f"Cleared {count} entries from context: {context_id}")
        return count
    
    def has(self, context_id: str, key: str) -> bool:
        if context_id not in self._contexts:
            return False
        
        if key not in self._contexts[context_id]:
            return False
        
        entry = self._contexts[context_id][key]
        
        if entry.expires_at and datetime.now() > entry.expires_at:
            del self._contexts[context_id][key]
            return False
        
        return True
    
    def lock(self, context_id: str, key: str) -> bool:
        if context_id not in self._context_ids[context_id]:
            return False
        
        if key not in self._contexts[context_id]:
            return False
        
        self._contexts[context_id][key].is_locked = True
        return True
    
    def unlock(self, context_id: str, key: str) -> bool:
        if context_id not in self._contexts:
            return False
        
        if key not in self._contexts[context_id]:
            return False
        
        self._contexts[context_id][key].is_locked = False
        return True
    
    def create_snapshot(
        self,
        context_id: str,
        snapshot_id: Optional[str] = None,
        description: str = ""
    ) -> Optional[str]:
        if context_id not in self._contexts:
            return None
        
        snapshot_id = snapshot_id or f"snapshot_{uuid.uuid4().hex[:12]}"
        
        entries = {}
        for key, entry in self._contexts[context_id].items():
            entries[key] = entry
        
        scope = POPSSContextScope.GLOBAL
        owner_id = None
        for entry in entries.values():
            scope = entry.scope
            owner_id = entry.owner_id
            break
        
        snapshot = POPSSContextSnapshot(
            snapshot_id=snapshot_id,
            context_id=context_id,
            entries=entries,
            scope=scope,
            owner_id=owner_id,
            description=description,
        )
        
        self._snapshots[snapshot_id] = snapshot
        self._metrics['total_snapshots'] += 1
        
        self._LOG.info(f"Created context snapshot: {snapshot_id}")
        return snapshot_id
    
    def restore_snapshot(self, context_id: str, snapshot_id: str) -> bool:
        if snapshot_id not in self._snapshots:
            return False
        
        snapshot = self._snapshots[snapshot_id]
        
        for key, entry in snapshot.entries.items():
            self._contexts[context_id][key] = entry
        
        self._LOG.info(f"Restored context snapshot: {snapshot_id}")
        return True
    
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        if snapshot_id not in self._snapshots:
            return None
        
        snapshot = self._snapshots[snapshot_id]
        return {
            'snapshot_id': snapshot.snapshot_id,
            'context_id': snapshot.context_id,
            'scope': snapshot.scope.value,
            'owner_id': snapshot.owner_id,
            'entry_count': len(snapshot.entries),
            'description': snapshot.description,
            'created_at': snapshot.created_at.isoformat(),
        }
    
    def list_snapshots(self, context_id: Optional[str] = None) -> List[Dict[str, Any]]:
        snapshots = []
        for snapshot_id, snapshot in self._snapshots.items():
            if context_id is None or snapshot.context_id == context_id:
                snapshots.append({
                    'snapshot_id': snapshot.snapshot_id,
                    'context_id': snapshot.context_id,
                    'scope': snapshot.scope.value,
                    'entry_count': len(snapshot.entries),
                    'description': snapshot.description,
                    'created_at': snapshot.created_at.isoformat(),
                })
        return snapshots
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        if snapshot_id in self._snapshots:
            del self._snapshots[snapshot_id]
            self._LOG.info(f"Deleted context snapshot: {snapshot_id}")
            return True
        return False
    
    def share_context(
        self,
        source_context_id: str,
        target_context_id: str,
        keys: Optional[List[str]] = None,
        scope: Optional[POPSSContextScope] = None
    ) -> int:
        if source_context_id not in self._contexts:
            return 0
        
        if target_context_id not in self._contexts:
            self.create_context(target_context_id, scope or POPSSContextScope.SESSION)
        
        shared_count = 0
        
        for key, entry in self._contexts[source_context_id].items():
            if keys and key not in keys:
                continue
            
            if scope and entry.scope != scope:
                continue
            
            if entry.is_locked:
                continue
            
            new_entry = POPSSContextEntry(
                entry_id=f"entry_{uuid.uuid4().hex[:12]}",
                key=key,
                value=entry.value,
                scope=entry.scope,
                priority=entry.priority,
                owner_id=entry.owner_id,
                parent_entry_id=entry.entry_id,
                is_persistent=entry.is_persistent,
            )
            
            self._contexts[target_context_id][key] = new_entry
            shared_count += 1
        
        self._LOG.info(f"Shared {shared_count} entries from {source_context_id} to {target_context_id}")
        return shared_count
    
    def merge_contexts(
        self,
        context_ids: List[str],
        target_context_id: str,
        conflict_resolution: str = "priority"
    ) -> bool:
        for context_id in context_ids:
            if context_id not in self._contexts:
                return False
        
        for context_id in context_ids:
            for key, entry in self._contexts[context_id].items():
                if key in self._contexts[target_context_id]:
                    existing = self._contexts[target_context_id][key]
                    
                    if conflict_resolution == "priority":
                        if entry.priority.value > existing.priority.value:
                            self._contexts[target_context_id][key] = entry
                    elif conflict_resolution == "latest":
                        if entry.accessed_at > existing.accessed_at:
                            self._contexts[target_context_id][key] = entry
                    elif conflict_resolution == "first":
                        pass
                else:
                    self._contexts[target_context_id][key] = entry
        
        self._LOG.info(f"Merged {len(context_ids)} contexts into {target_context_id}")
        return True
    
    def get_context_info(self, context_id: str) -> Optional[Dict[str, Any]]:
        if context_id not in self._contexts:
            return None
        
        entries = self._contexts[context_id]
        
        scope_counts = {}
        total_access = 0
        
        for entry in entries.values():
            scope = entry.scope.value
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
            total_access += entry.access_count
        
        locked_count = sum(1 for e in entries.values() if e.is_locked)
        persistent_count = sum(1 for e in entries.values() if e.is_persistent)
        
        return {
            'context_id': context_id,
            'entry_count': len(entries),
            'scope_distribution': scope_counts,
            'total_accesses': total_access,
            'locked_entries': locked_count,
            'persistent_entries': persistent_count,
        }
    
    def get_all_contexts_info(self) -> List[Dict[str, Any]]:
        return [
            self.get_context_info(ctx_id)
            for ctx_id in self._contexts
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        total_entries = sum(len(ctx) for ctx in self._contexts.values())
        
        access_history = self._access_history[-1000:]
        
        return {
            'context_count': len(self._contexts),
            'total_entries': total_entries,
            'total_accesses': self._metrics['total_accesses'],
            'cache_hits': self._metrics['cache_hits'],
            'cache_misses': self._metrics['cache_misses'],
            'cache_hit_rate': (
                self._metrics['cache_hits'] / 
                max(self._metrics['cache_hits'] + self._metrics['cache_misses'], 1)
            ),
            'total_snapshots': self._metrics['total_snapshots'],
        }
    
    def cleanup_expired(self) -> int:
        now = datetime.now()
        cleaned = 0
        
        for context_id in self._contexts:
            expired_keys = []
            
            for key, entry in self._contexts[context_id].items():
                if entry.expires_at and now > entry.expires_at:
                    if not entry.is_locked and not entry.is_persistent:
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self._contexts[context_id][key]
                cleaned += 1
        
        self._LOG.info(f"Cleaned up {cleaned} expired entries")
        return cleaned
    
    def shutdown(self):
        self._async_executor.shutdown(wait=True)
        self._LOG.info("POPSSContextManager shutdown")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
