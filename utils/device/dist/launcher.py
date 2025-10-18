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

import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class PiscesLxCoreLaunchSpec:
    """
    Data class for storing launch specifications for the PiscesLx Core.

    Attributes:
        nnodes (int): Number of nodes for distributed training. Defaults to 1.
        nproc_per_node (int): Number of processes per node. Defaults to 1.
        node_rank (int): Rank of the current node. Defaults to 0.
        master_addr (Optional[str]): Address of the master node. Defaults to None.
        master_port (Optional[int]): Port of the master node. Defaults to None.
        env (Optional[Dict[str, str]]): Extra environment variables to export before launch. Defaults to None.
    """
    nnodes: int = 1
    nproc_per_node: int = 1
    node_rank: int = 0
    master_addr: Optional[str] = None
    master_port: Optional[int] = None
    env: Optional[Dict[str, str]] = None

    def to_torchrun_cmd(self, entry: str, entry_args: Optional[List[str]] = None) -> List[str]:
        """
        Generate a torchrun command based on the launch specifications.

        Args:
            entry (str): The entry point script to run.
            entry_args (Optional[List[str]]): Additional arguments for the entry point script. Defaults to None.

        Returns:
            List[str]: The generated torchrun command as a list of strings.
        """
        cmd = [
            "torchrun",
            "--nnodes", str(self.nnodes),
            "--nproc_per_node", str(self.nproc_per_node),
            "--node_rank", str(self.node_rank),
        ]
        if self.master_addr:
            cmd += ["--master_addr", self.master_addr]
        if self.master_port:
            cmd += ["--master_port", str(self.master_port)]
        cmd += [entry]
        if entry_args:
            cmd += entry_args
        return cmd

    def apply_env(self) -> None:
        """
        Apply the extra environment variables to the current process.

        If the `env` attribute is None, this method does nothing.
        """
        if self.env is None:
            return
        for k, v in self.env.items():
            os.environ[str(k)] = str(v)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the dataclass instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the dataclass instance.
        """
        return asdict(self)
