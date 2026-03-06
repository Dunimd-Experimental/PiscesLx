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

from .core import Dataset
from .registry import REGISTRY as DATASETS
from typing import List, Optional, Dict, Any
from .streaming import PiscesLxToolsDataLargeScaleStreamingDataset
from .loader import PiscesLxToolsDataOptimizedDataLoader, PiscesLxToolsDataBatchConfig

class PiscesLxToolsDataDatasetManager:
    """
    A manager class for Pisces datasets that provides a unified interface for dataset operations.
    """
    def __init__(self):
        """
        Initialize the dataset manager.
        """
        pass

    def load(self, name: str = None, subset: str = "tiny", split: str = "train", config: Optional[Dict[str, Any]] = None, max_samples: Optional[int] = None) -> Dataset:
        """
        Load a dataset based on the specified subset and split.

        This method first checks if there is a registered builder for the given subset.
        If a builder is found, it uses the builder to create the dataset.
        Otherwise, it creates a new Dataset instance directly.

        Args:
            name (str, optional): The name of the dataset. Defaults to None.
            subset (str, optional): The name of the dataset subset. Defaults to "tiny".
            split (str, optional): The split of the dataset, e.g., "train", "val", "test". Defaults to "train".
            config (Optional[Dict[str, Any]], optional): Configuration dictionary for the dataset. Defaults to None.
            max_samples (Optional[int], optional): The maximum number of samples to load. Defaults to None.

        Returns:
            Dataset: A loaded Dataset instance.
        """
        # Retrieve the registered builder for the subset
        builder = DATASETS.get(subset)
        if builder:
            ds = builder(name=name, subset=subset, split=split, config=config, max_samples=max_samples)
            return ds
        ds = Dataset(name=name, subset=subset, split=split, config=config, max_samples=max_samples)
        return ds

    def dataloader(self, dataset, batch_config: Optional[PiscesLxToolsDataBatchConfig] = None):
        """
        Create a data loader for the given dataset.

        Args:
            dataset: The dataset for which to create a data loader.
            batch_config (Optional[PiscesLxToolsDataBatchConfig], optional): Configuration for batch processing. Defaults to None.

        Returns:
            The data loader instance created by PiscesLxToolsDataOptimizedDataLoader.
        """
        return PiscesLxToolsDataOptimizedDataLoader(dataset, batch_config).get()

    def streaming_dataloader(self, data_sources: List[str], config: Optional[Dict[str, Any]] = None, batch_config: Optional[PiscesLxToolsDataBatchConfig] = None):
        """
        Create a data loader for large-scale streaming datasets.

        Args:
            data_sources (List[str]): List of data source paths.
            config (Optional[Dict[str, Any]], optional): Configuration dictionary for the streaming dataset. Defaults to None.
            batch_config (Optional[PiscesLxToolsDataBatchConfig], optional): Configuration for batch processing. Defaults to None.

        Returns:
            The data loader instance created by PiscesLxToolsDataOptimizedDataLoader for the streaming dataset.
        """
        ds = PiscesLxToolsDataLargeScaleStreamingDataset(data_sources=data_sources, config=config)
        return PiscesLxToolsDataOptimizedDataLoader(ds, batch_config).get()

    def register(self, name: str, builder):
        """
        Register a custom dataset builder.

        Args:
            name (str): The name of the dataset to register.
            builder: The builder function or class for the dataset.
        """
        DATASETS.register(name, builder)
