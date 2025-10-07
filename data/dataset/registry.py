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

from typing import Callable, Dict, Optional, Any

class DatasetRegistry:
    """A registry class for managing dataset builders.
    
    This class provides a simple key-value store to register and retrieve dataset builders.
    """

    def __init__(self):
        """Initialize the DatasetRegistry instance.
        
        Creates an empty dictionary to store dataset builders.
        """
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, builder: Callable[..., Any]):
        """Register a dataset builder with the given name.
        
        Args:
            name (str): The name under which the builder will be registered.
            builder (Callable[..., Any]): The dataset builder callable.
        """
        self._builders[name] = builder

    def get(self, name: str) -> Optional[Callable[..., Any]]:
        """Retrieve the dataset builder for the given name.
        
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
            KeyError: If no dataset builder is found for the given name.
        """
        b = self.get(name)
        if not b:
            raise KeyError(f"Dataset builder not found: {name}")
        return b(**kwargs)

# Default singleton instance of DatasetRegistry
REGISTRY = DatasetRegistry()