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

import os
from typing import Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class PiscesLxCoreClusterEnv:
    """Class representing the environment of the Pisces Lx core cluster.

    Attributes:
        master_addr (str | None): Address of the master node.
        master_port (int | None): Port of the master node.
        node_rank (int): Rank of the current node.
        nnodes (int): Total number of nodes in the cluster.
        local_rank (int): Local rank of the current process within the node.
        rank (int): Global rank of the current process across all nodes.
        world_size (int): Total number of processes across all nodes.
    """
    master_addr: str | None
    master_port: int | None
    node_rank: int
    nnodes: int
    local_rank: int
    rank: int
    world_size: int

    @classmethod
    def detect(cls) -> "PiscesLxCoreClusterEnv":
        """Detect and create a PiscesLxCoreClusterEnv instance based on environment variables.

        Returns:
            PiscesLxCoreClusterEnv: An instance populated with values from environment variables.
        """
        def _env_i(name: str, default: int) -> int:
            """Helper function to retrieve an integer value from environment variables.

            Args:
                name (str): Name of the environment variable.
                default (int): Default value to return if the variable is not set or invalid.

            Returns:
                int: The integer value of the environment variable, or the default value if parsing fails.
            """
            try:
                return int(os.environ.get(name, default))
            except (ValueError, TypeError):
                return default

        # Get the master node address from environment variable
        master_addr = os.environ.get("MASTER_ADDR")
        # Get the master node port from environment variable, set to None if 0
        master_port = _env_i("MASTER_PORT", 0) or None
        # Get the current node rank from environment variable
        node_rank = _env_i("NODE_RANK", 0)
        # Get the total number of nodes from environment variable
        nnodes = _env_i("NNODES", 1)
        # Get the local rank of the current process from environment variable
        local_rank = _env_i("LOCAL_RANK", -1)
        # Get the global rank of the current process from environment variable
        rank = _env_i("RANK", 0)
        # Get the total number of processes from environment variable
        world_size = _env_i("WORLD_SIZE", 1)
        return cls(master_addr, master_port, node_rank, nnodes, local_rank, rank, world_size)

    def is_distributed(self) -> bool:
        """Check if the current environment is a distributed environment.

        Returns:
            bool: True if the environment is distributed, False otherwise.
        """
        return self.world_size > 1 and self.local_rank >= 0

    def is_multi_node(self) -> bool:
        """Check if the current environment spans multiple nodes.

        Returns:
            bool: True if the environment spans multiple nodes, False otherwise.
        """
        return self.nnodes > 1 or (self.master_addr is not None and self.master_addr not in ("127.0.0.1", "localhost"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the instance's attributes.
        """
        return asdict(self)
