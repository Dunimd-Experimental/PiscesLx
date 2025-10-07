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

from typing import Optional
from utils import PiscesLxCoreLog
from torch.utils.data import DataLoader

class BatchConfig:
    """Configuration class for batch processing.

    Stores parameters related to batch processing in data loading.
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
        """Initialize the BatchConfig object.

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

    Provides an optimized data loading mechanism based on the dataset type.
    """
    def __init__(self, dataset, batch_config: Optional[BatchConfig] = None):
        """Initialize the OptimizedDataLoader object.

        Args:
            dataset: Dataset from which to load the data.
            batch_config (Optional[BatchConfig], optional): Batch configuration object. 
                If None, default BatchConfig will be used. Defaults to None.
        """
        self.dataset = dataset
        self.cfg = batch_config or BatchConfig()
        self.logger = PiscesLxCoreLog("pisceslx.data.dataset.loader")

    def get(self) -> DataLoader:
        """Get a PyTorch DataLoader instance based on the dataset type.

        Simplified to use a fixed batch strategy. Complex dynamic batching can be extended based on MemoryMonitor later.

        Returns:
            DataLoader: A PyTorch DataLoader instance configured according to the dataset type.
        """
        # For IterableDataset: do not pass batch_size
        if hasattr(self.dataset, "__iter__") and not hasattr(self.dataset, "__len__"):
            return DataLoader(
                self.dataset,
                batch_size=None,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                prefetch_factor=self.cfg.prefetch_factor,
                persistent_workers=True
            )
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
