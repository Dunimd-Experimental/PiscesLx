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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

import threading
from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar
)

from .interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorConfig
)

T = TypeVar('T', bound=PiscesLxOperatorInterface)


class PiscesLxOperatorRegistry:
    """
    Central registry for operator management.

    This class provides a thread-safe registry for managing operator classes
    and instances. It handles registration, discovery, versioning, and
    dependency resolution for all operators in the system.

    Attributes:
        _operators: Dictionary mapping operator names to their classes
        _instances: Dictionary mapping operator names to instances
        _versions: Dictionary mapping names to version lists
        _dependencies: Dictionary mapping names to dependency sets
        _categories: Dictionary mapping categories to operator sets

    Thread Safety:
        All public methods use internal locking for thread-safe access.
        Multiple threads can safely register and query operators.

    Usage:
        registry = PiscesLxOperatorRegistry()
        registry.register(DataTransformOperator)
        registry.register(FilterOperator)

        # Create instance with configuration
        config = PiscesLxOperatorConfig(name="transformer", timeout=60.0)
        instance = registry.create_instance("data_transformer", config=config)

        # List all operators
        for op_info in registry.list_operators():
            print(f"{op_info['name']} v{op_info['latest_version']}")

    Error Handling:
        - TypeError: If operator doesn't implement the interface
        - ValueError: If duplicate registration detected
        - KeyError: If operator not found in lookups
    """

    def __init__(self):
        """
        Initialize an empty operator registry.

        Creates internal data structures for storing operators, instances,
        versions, dependencies, and categories. Thread-safe initialization.
        """
        self._operators: Dict[str, Dict[str, Type[PiscesLxOperatorInterface]]] = defaultdict(dict)
        self._instances: Dict[str, PiscesLxOperatorInterface] = {}
        self._versions: Dict[str, List[str]] = defaultdict(list)
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._categories: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        from ..dc import PiscesLxLogger, PiscesLxMetrics
        from utils.paths import get_log_file
        self.logger = PiscesLxLogger(f"PiscesLx.Core.OPSC.{self.__class__.__name__}", file_path=get_log_file(f"PiscesLx.Core.OPSC.{self.__class__.__name__}"), enable_file=True)
        self._metrics = PiscesLxMetrics()

    @staticmethod
    def _version_key(v: str) -> tuple:
        parts = []
        for p in str(v).split("."):
            try:
                parts.append(int(p))
            except ValueError:
                parts.append(p)
        return tuple(parts)

    @staticmethod
    def _class_metadata(operator_class: Type[PiscesLxOperatorInterface]) -> Dict[str, Any]:
        name = getattr(operator_class, "OPERATOR_NAME", None) or getattr(operator_class, "operator_name", None)
        version = getattr(operator_class, "OPERATOR_VERSION", None) or getattr(operator_class, "operator_version", None)
        description = getattr(operator_class, "OPERATOR_DESCRIPTION", None) or getattr(operator_class, "operator_description", None)
        if not description:
            doc = (operator_class.__doc__ or "").strip()
            description = doc.splitlines()[0].strip() if doc else operator_class.__name__
        return {
            "name": name,
            "version": version,
            "description": description
        }

    def register(self, operator_class: Type[T]) -> None:
        """
        Register an operator class with the registry.

        This method registers a new operator class, making it available
        for creation and execution. The operator class must implement
        the PiscesLxOperatorInterface.

        Args:
            operator_class: The operator class to register

        Raises:
            TypeError: If the class doesn't inherit from PiscesLxOperatorInterface
            ValueError: If an operator with the same name and version exists

        Process:
            1. Validate the operator class implements the interface
            2. Create temporary instance to get name and version
            3. Check for duplicate registration
            4. Store the operator class
            5. Update version list (sorted descending)
            6. Log registration

        Usage:
            class MyOperator(PiscesLxOperatorInterface):
                @property
                def name(self) -> str:
                    return "my_operator"
                # ... other required methods

            registry.register(MyOperator)
        """
        if not isinstance(operator_class, type):
            raise TypeError(
                f"Expected a class, got {type(operator_class).__name__}"
            )
        
        if not issubclass(operator_class, PiscesLxOperatorInterface):
            raise TypeError(
                f"Operator must inherit from PiscesLxOperatorInterface, "
                f"got {operator_class.__name__}"
            )

        meta = self._class_metadata(operator_class)
        name = meta["name"]
        version = meta["version"]
        if not name:
            name = operator_class.__name__
        if not version:
            from configs.version import VERSION
            version = VERSION

        with self._lock:
            if version in self._operators[name]:
                raise ValueError(f"Operator '{name}' version '{version}' already registered")

            self._operators[name][version] = operator_class
            self._versions[name].append(version)
            self._versions[name].sort(key=self._version_key, reverse=True)
            self._metrics.counter("registry_operator_registrations")
            self._metrics.gauge("registry_total_operators", len(self._operators))
            self.logger.info("operator_registered", operator=name, version=version)

    def unregister(self, name: str, version: Optional[str] = None) -> bool:
        """
        Unregister an operator from the registry.

        Removes an operator from the registry. If no version is specified,
        all versions of the operator are removed.

        Args:
            name: The operator name to unregister
            version: Optional specific version to unregister

        Returns:
            True if operator was found and unregistered, False otherwise

        Side Effects:
            - Removes operator class from _operators
            - Removes version from _versions
            - Removes instance from _instances
            - Cleans up any associated dependencies

        Usage:
            # Unregister specific version
            registry.unregister("my_operator", "1.0.0")

            # Unregister all versions
            registry.unregister("my_operator")
        """
        with self._lock:
            if name not in self._operators:
                return False

            if version is None:
                del self._operators[name]
                del self._versions[name]
                to_remove = [k for k in self._instances.keys() if k.startswith(f"{name}:")]
                for k in to_remove:
                    self._instances.pop(k, None)
                return True
            else:
                if version not in self._operators[name]:
                    return False
                self._operators[name].pop(version, None)
                if version in self._versions[name]:
                    self._versions[name].remove(version)
                self._instances.pop(f"{name}:{version}", None)
                if not self._operators[name]:
                    del self._operators[name]
                    del self._versions[name]
                return True

    def get_operator_class(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[Type[PiscesLxOperatorInterface]]:
        """
        Retrieve an operator class by name and optional version.

        Args:
            name: The operator name to look up
            version: Optional specific version (defaults to latest)

        Returns:
            The operator class or None if not found

        Usage:
            # Get latest version
            operator_class = registry.get_operator_class("my_operator")

            # Get specific version
            operator_class = registry.get_operator_class("my_operator", "1.0.0")
        """
        with self._lock:
            if name not in self._operators:
                return None

            if version is None:
                if not self._versions.get(name):
                    return None
                latest = self._versions[name][0]
                return self._operators[name].get(latest)
            return self._operators[name].get(version)

    def create_instance(
        self,
        name: str,
        version: Optional[str] = None,
        config: Optional['PiscesLxOperatorConfig'] = None
    ) -> Optional[PiscesLxOperatorInterface]:
        """
        Create and return an operator instance.

        Creates a new instance of the specified operator with optional
        configuration. The instance is cached for subsequent lookups.

        Args:
            name: The operator name
            version: Optional specific version
            config: Optional configuration parameters

        Returns:
            Configured operator instance or None if not found

        Lifecycle:
            1. Get operator class (latest or specific version)
            2. Create instance with configuration
            3. Call setup() to initialize resources
            4. Cache instance for future lookups
            5. Return instance

        Usage:
            config = PiscesLxOperatorConfig(
                name="processor",
                timeout=120.0,
                parameters={"mode": "fast"}
            )
            instance = registry.create_instance("data_processor", config=config)
        """
        operator_class = self.get_operator_class(name, version)
        if not operator_class:
            return None

        meta = self._class_metadata(operator_class)
        resolved_version = version or meta.get("version")
        if not resolved_version:
            from configs.version import VERSION
            resolved_version = VERSION

        key = f"{name}:{resolved_version}"
        if key in self._instances:
            return self._instances[key]

        try:
            if config is None:
                config = PiscesLxOperatorConfig(name=name, version=resolved_version)
            instance = operator_class(config)
            instance.setup()
            self._instances[key] = instance
            return instance
        except Exception as e:
            self.logger.error("operator_create_failed", operator=name, version=resolved_version, error=str(e))
            return None

    def get_instance(self, name: str) -> Optional[PiscesLxOperatorInterface]:
        """
        Get a previously created operator instance.

        Returns the cached instance if it exists, otherwise returns None.
        Use create_instance() to create a new instance.

        Args:
            name: The operator name

        Returns:
            Cached instance or None

        Usage:
            instance = registry.get_instance("my_operator")
            if instance:
                instance.execute(inputs)
        """
        with self._lock:
            versions = self._versions.get(name, [])
            if not versions:
                return None
            key = f"{name}:{versions[0]}"
            return self._instances.get(key)

    def list_operators(self) -> List[Dict[str, str]]:
        """
        List all registered operators with their metadata.

        Returns a sorted list of operator information including name,
        version details, description, and schema information.

        Returns:
            List of dictionaries containing operator information

        Information Returned:
            - name: Operator unique identifier
            - latest_version: Highest version number
            - all_versions: List of all registered versions
            - description: Human-readable description
            - input_schema: String representation of input schema
            - output_schema: String representation of output schema

        Usage:
            for op in registry.list_operators():
                print(f"{op['name']} v{op['latest_version']}")
        """
        result = []
        with self._lock:
            for name, versions in self._versions.items():
                latest_version = versions[0] if versions else ""
                operator_class = self._operators[name].get(latest_version)
                if not operator_class:
                    continue
                meta = self._class_metadata(operator_class)

                result.append({
                    "name": name,
                    "latest_version": latest_version,
                    "all_versions": versions,
                    "description": meta.get("description", ""),
                    "input_schema": str(getattr(operator_class, "INPUT_SCHEMA", {"type": "object"})),
                    "output_schema": str(getattr(operator_class, "OUTPUT_SCHEMA", {"type": "any"}))
                })

        return sorted(result, key=lambda x: x["name"])

    def get_dependencies(self, name: str) -> Set[str]:
        """
        Get the set of dependencies for an operator.

        Args:
            name: The operator name

        Returns:
            Set of operator names that this operator depends on

        Usage:
            deps = registry.get_dependencies("complex_operator")
            for dep in deps:
                print(f"Depends on: {dep}")
        """
        with self._lock:
            return self._dependencies.get(name, set()).copy()

    def add_dependency(self, name: str, dependency: str) -> None:
        """
        Add a dependency relationship between operators.

        Records that one operator depends on another. This affects
        the execution order in pipelines.

        Args:
            name: The operator that has the dependency
            dependency: The operator that is depended upon

        Usage:
            # Filter depends on transformer
            registry.add_dependency("filter_operator", "transform_operator")
        """
        with self._lock:
            self._dependencies[name].add(dependency)

    def remove_dependency(self, name: str, dependency: str) -> bool:
        """
        Remove a dependency relationship.

        Args:
            name: The operator with the dependency
            dependency: The operator to remove dependency on

        Returns:
            True if dependency was found and removed, False otherwise

        Usage:
            removed = registry.remove_dependency("filter", "transform")
        """
        with self._lock:
            if name in self._dependencies and dependency in self._dependencies[name]:
                self._dependencies[name].remove(dependency)
                return True
            return False

    def resolve_dependencies(self, name: str) -> List[str]:
        """
        Resolve dependency tree and return execution order.

        Performs topological sort to determine the correct order for
        executing an operator and its dependencies.

        Args:
            name: The operator name to resolve dependencies for

        Returns:
            List of operator names in dependency-sorted order

        Algorithm:
            Uses depth-first search (DFS) with post-order traversal
            to build the dependency graph and produce sorted results.

        Usage:
            order = registry.resolve_dependencies("complex_operator")
            # Returns: ["dep1", "dep2", "main_operator"]
        """
        visited: Set[str] = set()
        visiting: Set[str] = set()
        order: List[str] = []

        with self._lock:
            def dfs(current: str) -> None:
                if current in visited:
                    return
                if current in visiting:
                    raise ValueError(f"Dependency cycle detected at '{current}'")
                visiting.add(current)
                for dep in self._dependencies.get(current, set()):
                    if dep in self._operators:
                        dfs(dep)
                visiting.remove(current)
                visited.add(current)
                order.append(current)

            dfs(name)

        return order

    def clear(self) -> None:
        """
        Clear all registered operators and clean up resources.

        This method removes all operators from the registry and calls
        teardown() on any created instances to release resources.

        Side Effects:
            - Calls teardown() on all cached instances
            - Clears all internal data structures
            - Logs the clearing operation

        Usage:
            registry.clear()  # Cleanup before shutdown
        """
        with self._lock:
            for instance in self._instances.values():
                try:
                    instance.teardown()
                except Exception as e:
                    self.logger.error("operator_teardown_failed", error=str(e))

            self._operators.clear()
            self._instances.clear()
            self._versions.clear()
            self._dependencies.clear()
            self._categories.clear()


class PiscesLxOperatorRegistryHub:
    _lock = threading.RLock()
    _registry: Optional[PiscesLxOperatorRegistry] = None

    @classmethod
    def get_registry(cls) -> PiscesLxOperatorRegistry:
        with cls._lock:
            if cls._registry is None:
                cls._registry = PiscesLxOperatorRegistry()
            return cls._registry

    @classmethod
    def reset_registry(cls) -> PiscesLxOperatorRegistry:
        with cls._lock:
            cls._registry = PiscesLxOperatorRegistry()
            return cls._registry


class PiscesLxOperatorRegistrar:
    def __init__(self, registry: Optional[PiscesLxOperatorRegistry] = None):
        self.registry = registry or PiscesLxOperatorRegistryHub.get_registry()

    def __call__(self, operator_class: Any):
        self.registry.register(operator_class)
        return operator_class
