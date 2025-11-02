#!/usr/bin/env python3

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

import torch
from typing import Any, Optional
from torch.utils.data import DistributedSampler

class PiscesLxCoreDistributedSamplerBuilder:
    """Builder class for creating distributed samplers for distributed training setups."""
    
    def build_distributed_sampler(self, dataset: Any, world_size: int, rank: int, *, shuffle: bool = True, drop_last: bool = True) -> Optional[Any]:
        """Create and return a DistributedSampler for distributed training, or None if not applicable.

        This method abstracts the creation of a DistributedSampler to simplify its integration 
        with DataLoader in distributed settings.

        Args:
            dataset (Any): Dataset to be used for sampling.
            world_size (int): Total number of processes in the distributed training setup.
            rank (int): Unique identifier of the current process within the world_size.
            shuffle (bool, optional): Flag to enable shuffling of data at every epoch. Defaults to True.
            drop_last (bool, optional): Flag to drop the last batch if it is incomplete. Defaults to True.

        Returns:
            Optional[Any]: Instance of DistributedSampler if in distributed mode and initialization succeeds,
                           otherwise None.
        """
        # Early return if torch is not available
        if torch is None:
            return None
        
        # Early return if not running in distributed mode
        if world_size <= 1:
            return None

        try:
            # Initialize and return the DistributedSampler
            return DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        except Exception:
            # Silently handle exceptions and return None
            return None

# Legacy function for backward compatibility
def build_distributed_sampler(dataset: Any, world_size: int, rank: int, *, shuffle: bool = True, drop_last: bool = True) -> Optional[Any]:
    """Create and return a DistributedSampler for distributed training, or None if not applicable.

    This function abstracts the creation of a DistributedSampler to simplify its integration 
    with DataLoader in distributed settings.

    Args:
        dataset (Any): Dataset to be used for sampling.
        world_size (int): Total number of processes in the distributed training setup.
        rank (int): Unique identifier of the current process within the world_size.
        shuffle (bool, optional): Flag to enable shuffling of data at every epoch. Defaults to True.
        drop_last (bool, optional): Flag to drop the last batch if it is incomplete. Defaults to True.

    Returns:
        Optional[Any]: Instance of DistributedSampler if in distributed mode and initialization succeeds,
                       otherwise None.
    """
    builder = PiscesLxCoreDistributedSamplerBuilder()
    return builder.build_distributed_sampler(dataset, world_size, rank, shuffle=shuffle, drop_last=drop_last)
