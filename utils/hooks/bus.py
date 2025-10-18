#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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
from utils.log.core import PiscesLxCoreLog
from .types import PiscesLxCoreEventMetrics
from typing import Any, Dict, Optional, List
from .executor import PiscesLxCoreHookExecutor
from .registry import PiscesLxCoreListenerRegistry

logger = PiscesLxCoreLog("PiscesLx.Utils.Hooks.Bus")

class PiscesLxCoreHookBus:
    """HookBus - Unified event bus implementation."""
    
    def __init__(self) -> None:
        """Initialize the PiscesLxCoreHookBus instance.
        
        Initializes the listener registry, hook executor, event metrics, 
        reentrant lock, and logger.
        """
        self.registry = PiscesLxCoreListenerRegistry()
        self.executor = PiscesLxCoreHookExecutor()
        self._metrics: Dict[str, PiscesLxCoreEventMetrics] = {}
        self._lock = threading.RLock()

    
    def emit(self, event_type: str, **kwargs: Any) -> None:
        """Emit an event: execute the listeners and merge execution statistics into metrics.
        
        Args:
            event_type (str): The type of the event to emit.
            **kwargs (Any): Additional keyword arguments to pass to the listeners.
        """
        # Read and execute listeners first to avoid holding the lock for too long
        listeners = self.registry.get_listeners(event_type)
        summary = None
        if listeners:
            summary = self.executor.execute(event_type, listeners, **kwargs)

        # Then update the metrics
        with self._lock:
            m = self._metrics.get(event_type)
            if m is None:
                m = PiscesLxCoreEventMetrics(event_type=event_type)
                self._metrics[event_type] = m
            m.count += 1
            m.last_executed = kwargs.get('timestamp', time.time())
            if summary:
                try:
                    m.total_time += float(summary.get('total_time', 0.0))
                    m.errors += int(summary.get('errors', 0))
                except Exception as e:
                    logger.debug("METRICS_UPDATE_FAILED", {"event_type": event_type, "error": str(e), "summary": summary})
    
    def get_listeners(self, event_type: str) -> List:
        """Get the listeners for a specified event.
        
        Args:
            event_type (str): The type of the event to get listeners for.
            
        Returns:
            List: A list of listeners for the specified event.
        """
        return self.registry.get_listeners(event_type)
    
    def get_metrics(self, event_type: Optional[str] = None) -> Dict[str, PiscesLxCoreEventMetrics]:
        """Get the event metrics.
        
        Args:
            event_type (Optional[str]): The type of the event to get metrics for. 
                If None, return metrics for all events.
                
        Returns:
            Dict[str, PiscesLxCoreEventMetrics]: A dictionary containing the event metrics.
        """
        with self._lock:
            if event_type:
                return {event_type: self._metrics.get(event_type, PiscesLxCoreEventMetrics(event_type=event_type))}
            return self._metrics.copy()


# Global HookBus instance
_global_hook_bus: Optional[PiscesLxCoreHookBus] = None
_global_lock = threading.Lock()


class PiscesLxCoreGlobalHookBusFacade:
    """
    Facade class for global HookBus operations.
    Provides a unified interface for accessing the global HookBus instance.
    """
    
    def __init__(self) -> None:
        """Initialize the facade with the global HookBus instance."""
        self._hook_bus = get_global_hook_bus()
    
    def get_hook_bus(self) -> PiscesLxCoreHookBus:
        """
        Get the global HookBus instance.
        
        Returns:
            PiscesLxCoreHookBus: The global HookBus instance.
        """
        return self._hook_bus
    
    def emit(self, event_type: str, **kwargs: Any) -> None:
        """
        Emit an event through the global HookBus.
        
        Args:
            event_type (str): The type of the event to emit.
            **kwargs (Any): Additional keyword arguments to pass to the listeners.
        """
        self._hook_bus.emit(event_type, **kwargs)
    
    def get_listeners(self, event_type: str) -> List:
        """
        Get listeners for a specific event type.
        
        Args:
            event_type (str): The type of the event to get listeners for.
            
        Returns:
            List: A list of listeners for the specified event.
        """
        return self._hook_bus.get_listeners(event_type)
    
    def get_metrics(self, event_type: Optional[str] = None) -> Dict[str, PiscesLxCoreEventMetrics]:
        """
        Get event metrics from the global HookBus.
        
        Args:
            event_type (Optional[str]): The type of the event to get metrics for.
                If None, return metrics for all events.
                
        Returns:
            Dict[str, PiscesLxCoreEventMetrics]: A dictionary containing the event metrics.
        """
        return self._hook_bus.get_metrics(event_type)


def get_global_hook_bus() -> PiscesLxCoreHookBus:
    """Get the global HookBus instance.
    
    Returns:
        PiscesLxCoreHookBus: The global HookBus instance.
    """
    global _global_hook_bus
    if _global_hook_bus is None:
        with _global_lock:
            if _global_hook_bus is None:
                _global_hook_bus = PiscesLxCoreHookBus()
    return _global_hook_bus