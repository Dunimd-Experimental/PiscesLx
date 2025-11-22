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

from typing import Any
from .facade import PiscesLxCoreDeviceFacade
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("PiscesLx.Core.Device.Runner")

class PiscesLxCoreDeviceRunner:
    """Device runner class that offers static methods for device management.

    This class provides a unified interface for device setup and management,
    encapsulating the functionality previously implemented in standalone functions.
    """
    
    @staticmethod
    def setup_devices(args: Any = None, mode: str = "auto") -> dict:
        """Sets up devices for training or inference.

        Args:
            args (Any, optional): Command line arguments or configuration object. Defaults to None.
            mode (str, optional): Device setup mode. Supported modes: 'auto', 'manual', 'distributed', 'cluster'. Defaults to "auto".

        Returns:
            dict: Dictionary containing device configuration and recommendations.
        """
        facade = PiscesLxCoreDeviceFacade(args)
        return facade.setup_devices(mode)

    @staticmethod
    def setup_cluster(world_size: int = None, rank: int = None, local_rank: int = None,
                      master_addr: str = None, master_port: int = None) -> dict:
        """Configures a multi-node GPU cluster for distributed training or inference.

        Args:
            world_size (int, optional): Total number of processes across all nodes. Defaults to None.
            rank (int, optional): Global rank of the current process (0 to world_size - 1). Defaults to None.
            local_rank (int, optional): Local rank of the process on the current node. Defaults to None.
            master_addr (str, optional): IP address of the master node. Defaults to None.
            master_port (int, optional): Port number for communication on the master node. Defaults to None.

        Returns:
            dict: Dictionary containing cluster configuration details.
        """
        facade = PiscesLxCoreDeviceFacade()
        return facade.setup_cluster(world_size, rank, local_rank, master_addr, master_port)

    @staticmethod
    def get_cluster_status() -> dict:
        """Retrieves the current status and configuration of the cluster.

        Returns:
            dict: Dictionary with cluster status information.
        """
        facade = PiscesLxCoreDeviceFacade()
        return facade.get_cluster_status()

