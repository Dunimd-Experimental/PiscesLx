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

from typing import Optional
from torch.utils.data import DataLoader

class BatchConfig:
    """Stores configuration parameters for batch processing in data loading.
    
    These parameters are used to configure the PyTorch DataLoader.
    """
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2
    ):
        """Initialize a BatchConfig object with batch processing parameters.

        Args:
            batch_size (int, optional): Number of samples per batch to load. Defaults to 32.
            num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.
            pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to True.
            drop_last (bool, optional): If True, the last incomplete batch will be dropped. Defaults to False.
            prefetch_factor (int, optional): Number of samples loaded in advance by each worker. Defaults to 2.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor

class OptimizedDataLoader:
    """A wrapper class for PyTorch's DataLoader with optimized configuration.
    
    Provides different data loading mechanisms based on the dataset type.
    """
    def __init__(self, dataset, batch_config: Optional[BatchConfig] = None):
        """Initialize an OptimizedDataLoader object.

        Args:
            dataset: Dataset from which to load the data.
            batch_config (Optional[BatchConfig], optional): Batch configuration object. 
                If None, a default BatchConfig will be used. Defaults to None.
        """
        self.dataset = dataset
        self.cfg = batch_config or BatchConfig()

    def get(self) -> DataLoader:
        """Get a configured PyTorch DataLoader instance based on the dataset type.

        Returns:
            DataLoader: A PyTorch DataLoader instance configured according to the dataset type.
        """
        # If the dataset is iterable but does not have a length, treat it as an IterableDataset
        if hasattr(self.dataset, "__iter__") and not hasattr(self.dataset, "__len__"):
            return DataLoader(
                self.dataset,
                batch_size=None,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                prefetch_factor=self.cfg.prefetch_factor,
                persistent_workers=True
            )
        # For regular datasets, use the configured batch size and enable shuffling
        else:
            return DataLoader(
                self.dataset,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                drop_last=self.cfg.drop_last,
                prefetch_factor=self.cfg.prefetch_factor,
                persistent_workers=True
            )
