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

"""
Dynamic Batch Sampler for efficient training with variable-length sequences.

This module implements a dynamic batching strategy that organizes batches
based on total token count rather than fixed sample count, significantly
reducing padding waste and improving GPU utilization.

Key Features:
    - Token-based batching: Groups samples to maximize GPU utilization
    - Length prediction: Pre-computes or predicts sequence lengths
    - Padding optimization: Minimizes padding tokens per batch
    - Distributed support: Compatible with multi-GPU training

Usage:
    >>> from tools.data.sampler import PiscesLxDynamicBatchSampler
    >>> sampler = PiscesLxDynamicBatchSampler(
    ...     data_source=dataset,
    ...     max_tokens=8192,
    ...     length_fn=lambda idx: len(dataset[idx]['input_ids'])
    ... )
    >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
"""

from typing import Callable, Iterator, List, Optional, Sequence
from torch.utils.data import Sampler
import numpy as np


class PiscesLxDynamicBatchSampler(Sampler[List[int]]):
    """
    Dynamic batch sampler that organizes batches by total token count.
    
    This sampler groups samples into batches such that each batch contains
    approximately the same number of tokens, rather than the same number
    of samples. This approach significantly reduces padding waste when
    dealing with variable-length sequences.
    
    Attributes:
        data_source: The dataset to sample from.
        max_tokens: Maximum number of tokens per batch.
        max_samples: Maximum number of samples per batch (safety limit).
        length_fn: Function to get the length of a sample by index.
        drop_last: Whether to drop the last incomplete batch.
        shuffle: Whether to shuffle indices before batching.
        seed: Random seed for reproducibility.
        rank: Rank for distributed training.
        world_size: Total number of processes for distributed training.
    
    Example:
        >>> dataset = MyDataset()
        >>> sampler = PiscesLxDynamicBatchSampler(
        ...     data_source=dataset,
        ...     max_tokens=8192,
        ...     length_fn=lambda i: len(dataset[i]['input_ids'])
        ... )
        >>> for batch_indices in sampler:
        ...     batch = [dataset[i] for i in batch_indices]
        ...     # Process batch
    """
    
    def __init__(
        self,
        data_source: Sequence,
        max_tokens: int = 8192,
        max_samples: int = 128,
        length_fn: Optional[Callable[[int], int]] = None,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        rank: Optional[int] = None,
        world_size: Optional[int] = None
    ) -> None:
        """
        Initialize the dynamic batch sampler.
        
        Args:
            data_source: Dataset to sample from.
            max_tokens: Maximum tokens per batch. Defaults to 8192.
            max_samples: Maximum samples per batch as safety limit. Defaults to 128.
            length_fn: Function returning sample length by index. If None,
                tries to access 'input_ids' or 'length' attribute.
            drop_last: Drop incomplete last batch. Defaults to False.
            shuffle: Shuffle indices each epoch. Defaults to True.
            seed: Random seed. Defaults to 42.
            rank: Process rank for distributed training.
            world_size: Total processes for distributed training.
        """
        self.data_source = data_source
        self.max_tokens = max_tokens
        self.max_samples = max_samples
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size or 1
        
        if length_fn is not None:
            self.length_fn = length_fn
        else:
            self.length_fn = self._default_length_fn
        
        self._lengths: Optional[List[int]] = None
        self._rng = np.random.default_rng(seed)
    
    def _default_length_fn(self, idx: int) -> int:
        """
        Default function to get sample length.
        
        Tries to access 'input_ids' or 'length' attribute from the sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            int: Length of the sample.
        """
        try:
            sample = self.data_source[idx]
            if isinstance(sample, dict):
                if 'input_ids' in sample:
                    ids = sample['input_ids']
                    return len(ids) if hasattr(ids, '__len__') else 1
                elif 'length' in sample:
                    return int(sample['length'])
            return 1
        except Exception:
            return 1
    
    def _compute_lengths(self) -> List[int]:
        """
        Compute lengths for all samples in the dataset.
        
        Returns:
            List[int]: List of sample lengths.
        """
        if self._lengths is None:
            self._lengths = [self.length_fn(i) for i in range(len(self.data_source))]
        return self._lengths
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch for deterministic shuffling.
        
        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch
        self._rng = np.random.default_rng(self.seed + epoch)
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches of indices.
        
        Yields:
            List[int]: List of indices for each batch.
        """
        lengths = self._compute_lengths()
        indices = list(range(len(self.data_source)))
        
        if self.shuffle:
            self._rng.shuffle(indices)
        
        if self.rank is not None and self.world_size > 1:
            indices = indices[self.rank::self.world_size]
        
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_tokens = 0
        max_len_in_batch = 0
        
        for idx in indices:
            sample_len = lengths[idx]
            
            new_tokens = (len(current_batch) + 1) * max(max_len_in_batch, sample_len)
            
            if len(current_batch) == 0:
                current_batch.append(idx)
                current_tokens = sample_len
                max_len_in_batch = sample_len
            elif (new_tokens <= self.max_tokens and 
                  len(current_batch) < self.max_samples):
                current_batch.append(idx)
                max_len_in_batch = max(max_len_in_batch, sample_len)
                current_tokens = len(current_batch) * max_len_in_batch
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_tokens = sample_len
                max_len_in_batch = sample_len
        
        if current_batch and not self.drop_last:
            batches.append(current_batch)
        
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """
        Estimate the number of batches.
        
        Returns:
            int: Estimated number of batches.
        """
        lengths = self._compute_lengths()
        if not lengths:
            return 0
        
        avg_len = sum(lengths) / len(lengths)
        avg_batch_size = min(self.max_tokens // max(int(avg_len), 1), self.max_samples)
        avg_batch_size = max(avg_batch_size, 1)
        
        total = len(self.data_source)
        if self.rank is not None and self.world_size > 1:
            total = (total + self.world_size - 1) // self.world_size
        
        return (total + avg_batch_size - 1) // avg_batch_size
