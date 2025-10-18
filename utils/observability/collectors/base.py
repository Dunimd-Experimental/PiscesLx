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

from typing import Any, Dict, Optional, Protocol

class Collector(Protocol):
    """Protocol defining the interface for metric collectors."""
    def name(self) -> str:
        """Retrieve the name of the collector.

        Returns:
            str: The class name of the collector instance.
        """
        return self.__class__.__name__

    def configure(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Configure the collector with the provided configuration.

        If the collector already has a '_config' attribute, it updates the existing configuration.
        Otherwise, it creates a new copy of the provided configuration.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary. Defaults to None.
        """
        if config and hasattr(self, '_config'):
            self._config.update(config)
        elif config:
            self._config = config.copy()

    def enabled(self) -> bool:
        """Check if the collector is enabled.

        Returns:
            bool: True if the collector is enabled, False otherwise. 
                  Returns True by default if '_enabled' attribute doesn't exist.
        """
        return getattr(self, '_enabled', True)

    def collect(self, registry: Any) -> None:
        """Collect metrics and add them to the specified registry.

        If the registry has a 'counter' method, it increments the collection counter.

        Args:
            registry (Any): The metrics registry to which metrics will be added.
        """
        if hasattr(registry, 'counter'):
            registry.counter(f"{self.name()}_collections_total").inc()

    def health(self) -> Dict[str, Any]:
        """Get the current health status of the collector.

        Returns:
            Dict[str, Any]: A dictionary containing the health status, collector name, 
                            enabled state, current timestamp, and total collections.
        """
        return {
            "status": "healthy",
            "collector_name": self.name(),
            "enabled": self.enabled(),
            "timestamp": __import__('time').time(),
            "collections_total": getattr(self, '_collections_total', 0)
        }

class BaseCollector:
    """A convenience base class that provides safe default implementations."""
    _name: str = "collector"

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize the base collector.

        Args:
            name (Optional[str]): The name of the collector. If None, uses the default name.
        """
        self._name = name or self._name
        self._enabled = True
        self._failures = 0
        self._last_ok: float = 0.0
        self._last_err: float = 0.0

    def name(self) -> str:
        """Retrieve the name of the collector.

        Returns:
            str: The class name of the collector instance.
        """
        return self.__class__.__name__

    def configure(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Configure the collector with the provided configuration.

        If the configuration is provided, it initializes the '_config' attribute if it doesn't exist,
        then updates it with the new configuration.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary. Defaults to None.
        """
        if config:
            if not hasattr(self, '_config'):
                self._config = {}
            self._config.update(config)

    def enabled(self) -> bool:
        """Check if the collector is enabled.

        Returns:
            bool: True if the collector is enabled, False otherwise.
        """
        return self._enabled

    def set_enabled(self, flag: bool) -> None:
        """Set the enabled state of the collector.

        Args:
            flag (bool): True to enable the collector, False to disable it.
        """
        self._enabled = bool(flag)

    def register_metrics(self, registry: Any) -> None:
        """Register metrics with the provided registry.

        If the registry has a 'counter' method, it increments the registration counter.
        If the registry has a 'gauge' method, it sets the health status gauge.

        Args:
            registry (Any): The metrics registry.
        """
        if hasattr(registry, 'counter'):
            registry.counter(f"{self.name()}_registrations_total").inc()
        if hasattr(registry, 'gauge'):
            registry.gauge(f"{self.name()}_health_status").set(1.0 if self.enabled() else 0.0)

    def health(self) -> Dict[str, Any]:
        """Get the current health status of the collector.

        Returns:
            Dict[str, Any]: A dictionary containing the health status, collector name, 
                            enabled state, and current timestamp.
        """
        return {
            "status": "healthy",
            "collector_name": self.name(),
            "enabled": self.enabled(),
            "timestamp": __import__('time').time()
        }

    def _mark_success(self, now: float) -> None:
        """Mark the collector as having completed a successful operation.

        Args:
            now (float): The current timestamp.
        """
        self._last_ok = float(now)
        self._failures = 0

    def _mark_error(self, now: float) -> None:
        """Mark the collector as having encountered an error.

        Args:
            now (float): The current timestamp.
        """
        self._last_err = float(now)
        self._failures += 1
