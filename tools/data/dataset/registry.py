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

from typing import Callable, Dict, Optional, Any

class DatasetRegistry:
    """Manages a registry of dataset builders.

    This class implements a simple key-value store that allows users to register
    dataset builders and retrieve them by name.
    """

    def __init__(self):
        """Initialize the DatasetRegistry instance.

        Creates an empty dictionary to store dataset builders, 
        where keys are builder names and values are callable builders.
        """
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, builder: Callable[..., Any]):
        """Register a dataset builder with a specified name.

        Args:
            name (str): The unique identifier for the dataset builder.
            builder (Callable[..., Any]): A callable object that builds the dataset.
        """
        self._builders[name] = builder

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        """Retrieve a dataset builder by its name.

        Args:
            name (str): The name of the dataset builder to retrieve.

        Returns:
            Optional[Callable[..., Any]]: The dataset builder if found, None otherwise.
        """
        return self._builders.get(name)

    def build(self, name: str, **kwargs):
        """Build a dataset using the registered builder with the given name.

        Args:
            name (str): The name of the dataset builder to use.
            **kwargs: Additional keyword arguments to pass to the builder.

        Returns:
            Any: The result of calling the dataset builder.

        Raises:
            KeyError: If no dataset builder is found for the specified name.
        """
        builder = self.get(name)
        if not builder:
            raise KeyError(f"Dataset builder not found: {name}")
        return builder(**kwargs)

REGISTRY = DatasetRegistry()
