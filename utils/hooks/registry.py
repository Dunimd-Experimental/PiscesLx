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

import threading
from dataclasses import dataclass, field
from .types import PiscesLxCoreAlgorithmicListener
from typing import Any, Callable, Dict, List, Optional, Set, Type

@dataclass
class PiscesLxCoreRegistryEntry:
    """Represents an entry in the registry.

    Attributes:
        listener (PiscesLxCoreAlgorithmicListener): The listener instance.
        priority (int): The priority of the listener.
        metadata (Dict[str, Any]): Additional metadata associated with the listener.
    """
    listener: PiscesLxCoreAlgorithmicListener
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class PiscesLxCoreListenerRegistry:
    """A registry for managing listeners in an object-oriented manner."""
    
    def __init__(self) -> None:
        """Initialize the listener registry."""
        self._listeners: Dict[str, List[PiscesLxCoreRegistryEntry]] = {}
        self._listener_types: Dict[str, Type[PiscesLxCoreAlgorithmicListener]] = {}
        self._enabled = True
        self._lock = threading.RLock()
    
    def register_listener(
        self,
        listener: PiscesLxCoreAlgorithmicListener,
        event_types: Optional[List[str]] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a listener for specified event types.

        Args:
            listener (PiscesLxCoreAlgorithmicListener): The listener to register.
            event_types (Optional[List[str]]): List of event types to register the listener for. 
                If None, uses the listener's event_types attribute. Defaults to None.
            priority (int): The priority of the listener. Higher values have higher priority. Defaults to 0.
            metadata (Optional[Dict[str, Any]]): Additional metadata for the listener. Defaults to None.
        """
        if not self._enabled:
            return

        if event_types is None:
            event_types = listener.event_types or []

        if metadata is None:
            metadata = {}

        entry = PiscesLxCoreRegistryEntry(
            listener=listener,
            priority=priority,
            metadata=metadata
        )

        with self._lock:
            for event_type in event_types:
                if event_type not in self._listeners:
                    self._listeners[event_type] = []
                self._listeners[event_type].append(entry)
                # Sort listeners by priority in descending order
                self._listeners[event_type].sort(key=lambda x: x.priority, reverse=True)
    
    def unregister_listener(
        self,
        listener: PiscesLxCoreAlgorithmicListener,
        event_types: Optional[List[str]] = None
    ) -> None:
        """Unregister a listener from specified event types.

        Args:
            listener (PiscesLxCoreAlgorithmicListener): The listener to unregister.
            event_types (Optional[List[str]]): List of event types to unregister the listener from. 
                If None, unregisters from all event types. Defaults to None.
        """
        with self._lock:
            if event_types is None:
                event_types = list(self._listeners.keys())

            for event_type in event_types:
                if event_type in self._listeners:
                    self._listeners[event_type] = [
                        entry for entry in self._listeners[event_type]
                        if entry.listener != listener
                    ]
    
    def get_listeners(self, event_type: str) -> List[PiscesLxCoreAlgorithmicListener]:
        """Get a list of listeners for a specified event type.

        Args:
            event_type (str): The event type to get listeners for.

        Returns:
            List[PiscesLxCoreAlgorithmicListener]: List of listeners for the specified event type.
                Returns an empty list if no listeners are registered for the event type.
        """
        with self._lock:
            if event_type not in self._listeners:
                return []
            # Return a shallow copy to avoid external modification and concurrent iteration risks
            return [entry.listener for entry in list(self._listeners[event_type])]
    
    def get_all_listeners(self) -> List[PiscesLxCoreAlgorithmicListener]:
        """Get all unique listeners in the registry.

        Returns:
            List[PiscesLxCoreAlgorithmicListener]: List of all unique listeners in the registry.
        """
        with self._lock:
            seen = set()
            listeners: List[PiscesLxCoreAlgorithmicListener] = []
            for entries in self._listeners.values():
                for entry in entries:
                    if entry.listener not in seen:
                        seen.add(entry.listener)
                        listeners.append(entry.listener)
            return list(listeners)
    
    def clear_listeners(self, event_type: Optional[str] = None) -> None:
        """Clear listeners from the registry.

        Args:
            event_type (Optional[str]): The event type to clear listeners from. 
                If None, clears all listeners from the registry. Defaults to None.
        """
        with self._lock:
            if event_type is None:
                self._listeners.clear()
            else:
                self._listeners.pop(event_type, None)
    
    def has_listeners(self, event_type: str) -> bool:
        """Check if there are listeners registered for a specified event type.

        Args:
            event_type (str): The event type to check.

        Returns:
            bool: True if there are listeners registered for the event type, False otherwise.
        """
        with self._lock:
            return event_type in self._listeners and len(self._listeners[event_type]) > 0
    
    def get_event_types(self) -> List[str]:
        """Get all event types with registered listeners.

        Returns:
            List[str]: List of all event types with registered listeners.
        """
        with self._lock:
            return list(self._listeners.keys())
    
    def enable(self) -> None:
        """Enable the registry to allow listener registration and unregistration."""
        with self._lock:
            self._enabled = True
    
    def disable(self) -> None:
        """Disable the registry to prevent listener registration and unregistration."""
        with self._lock:
            self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if the registry is enabled.

        Returns:
            bool: True if the registry is enabled, False otherwise.
        """
        with self._lock:
            return self._enabled
    
    def register_listener_type(
        self,
        listener_type: Type[PiscesLxCoreAlgorithmicListener],
        name: Optional[str] = None
    ) -> None:
        """Register a listener type with an optional name.

        Args:
            listener_type (Type[PiscesLxCoreAlgorithmicListener]): The listener type to register.
            name (Optional[str]): The name to associate with the listener type. 
                If None, uses the listener type's class name. Defaults to None.
        """
        if name is None:
            name = listener_type.__name__
        with self._lock:
            self._listener_types[name] = listener_type
    
    def get_listener_type(self, name: str) -> Optional[Type[PiscesLxCoreAlgorithmicListener]]:
        """Get a listener type by its name.

        Args:
            name (str): The name of the listener type.

        Returns:
            Optional[Type[PiscesLxCoreAlgorithmicListener]]: The listener type associated with the name,
                or None if not found.
        """
        with self._lock:
            return self._listener_types.get(name)
    
    def get_listener_types(self) -> Dict[str, Type[PiscesLxCoreAlgorithmicListener]]:
        """Get all registered listener types.

        Returns:
            Dict[str, Type[PiscesLxCoreAlgorithmicListener]]: A copy of the dictionary mapping names to listener types.
        """
        with self._lock:
            return self._listener_types.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the registry.

        Returns:
            Dict[str, Any]: A dictionary containing registry statistics, including:
                - total_event_types: The total number of event types with registered listeners.
                - total_listeners: The total number of unique listeners.
                - enabled: Whether the registry is enabled.
                - event_types: List of all event types with registered listeners.
        """
        with self._lock:
            return {
                "total_event_types": len(self._listeners),
                "total_listeners": len(self.get_all_listeners()),
                "enabled": self._enabled,
                "event_types": list(self._listeners.keys())
            }
