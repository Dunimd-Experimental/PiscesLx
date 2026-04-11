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
Curriculum Learning Sampler for progressive training difficulty.

This module implements a curriculum learning strategy that organizes
training samples from easy to hard, progressively increasing difficulty
as training progresses. This approach has been shown to accelerate
convergence and improve final model performance.

Key Features:
    - Difficulty-based ordering: Samples sorted by difficulty scores
    - Progressive exposure: Gradually increases training difficulty
    - Epoch-based progression: Difficulty range expands each epoch
    - Configurable pace: Control how quickly difficulty increases

Usage:
    >>> from tools.data.sampler import PiscesLxCurriculumSampler
    >>> difficulty_scores = [0.2, 0.8, 0.5, 0.3, 0.9]  # Per-sample difficulty
    >>> sampler = PiscesLxCurriculumSampler(
    ...     data_source=dataset,
    ...     difficulty_scores=difficulty_scores,
    ...     total_epochs=10
    ... )
    >>> dataloader = DataLoader(dataset, sampler=sampler)
"""

from typing import Callable, Iterator, List, Optional, Sequence
from torch.utils.data import Sampler
import numpy as np


class PiscesLxCurriculumSampler(Sampler[int]):
    """
    Curriculum learning sampler that organizes samples by difficulty.
    
    This sampler sorts samples by difficulty and progressively exposes
    harder samples as training progresses. At epoch 0, only the easiest
    samples are used. As epochs progress, harder samples are gradually
    included until all samples are used.
    
    Attributes:
        data_source: The dataset to sample from.
        difficulty_scores: List of difficulty scores for each sample.
        total_epochs: Total training epochs for curriculum planning.
        start_ratio: Initial ratio of samples to use (easiest). Defaults to 0.1.
        pace: How quickly to add samples ('linear', 'exp', or 'log'). Defaults to 'linear'.
        shuffle_within: Shuffle samples within the current difficulty range.
        seed: Random seed for reproducibility.
    
    Example:
        >>> dataset = MyDataset()
        >>> scores = compute_difficulty_scores(dataset)
        >>> sampler = PiscesLxCurriculumSampler(
        ...     data_source=dataset,
        ...     difficulty_scores=scores,
        ...     total_epochs=10
        ... )
        >>> for epoch in range(10):
        ...     sampler.set_epoch(epoch)
        ...     for idx in sampler:
        ...         sample = dataset[idx]
    """
    
    def __init__(
        self,
        data_source: Sequence,
        difficulty_scores: Optional[List[float]] = None,
        total_epochs: int = 10,
        start_ratio: float = 0.1,
        pace: str = 'linear',
        shuffle_within: bool = True,
        seed: int = 42
    ) -> None:
        """
        Initialize the curriculum sampler.
        
        Args:
            data_source: Dataset to sample from.
            difficulty_scores: Difficulty score for each sample (0=easy, 1=hard).
                If None, will try to compute from dataset.
            total_epochs: Total epochs for curriculum planning. Defaults to 10.
            start_ratio: Initial ratio of samples to use. Defaults to 0.1 (10%).
            pace: Progression pace - 'linear', 'exp', or 'log'. Defaults to 'linear'.
            shuffle_within: Shuffle within current difficulty range. Defaults to True.
            seed: Random seed. Defaults to 42.
        """
        self.data_source = data_source
        self.total_epochs = total_epochs
        self.start_ratio = start_ratio
        self.pace = pace
        self.shuffle_within = shuffle_within
        self.seed = seed
        self.epoch = 0
        
        if difficulty_scores is not None:
            self.difficulty_scores = list(difficulty_scores)
        else:
            self.difficulty_scores = self._compute_default_scores()
        
        if len(self.difficulty_scores) != len(self.data_source):
            raise ValueError(
                f"Length mismatch: {len(self.difficulty_scores)} scores "
                f"for {len(self.data_source)} samples"
            )
        
        self._sorted_indices = np.argsort(self.difficulty_scores).tolist()
        self._rng = np.random.default_rng(seed)
    
    def _compute_default_scores(self) -> List[float]:
        """
        Compute default difficulty scores from dataset.
        
        Uses sample length as a proxy for difficulty.
        
        Returns:
            List[float]: Difficulty scores for each sample.
        """
        scores = []
        for i in range(len(self.data_source)):
            try:
                sample = self.data_source[i]
                if isinstance(sample, dict):
                    if 'input_ids' in sample:
                        length = len(sample['input_ids'])
                    elif 'text' in sample:
                        length = len(sample['text'].split())
                    else:
                        length = 1
                else:
                    length = 1
                scores.append(min(length / 1000.0, 1.0))
            except Exception:
                scores.append(0.5)
        return scores
    
    def _get_current_ratio(self) -> float:
        """
        Calculate the current ratio of samples to use based on epoch.
        
        Returns:
            float: Current ratio (0 to 1).
        """
        progress = min(self.epoch / max(self.total_epochs - 1, 1), 1.0)
        
        if self.pace == 'linear':
            ratio = self.start_ratio + (1.0 - self.start_ratio) * progress
        elif self.pace == 'exp':
            ratio = 1.0 - (1.0 - self.start_ratio) * (1.0 - progress) ** 2
        elif self.pace == 'log':
            import math
            ratio = self.start_ratio + (1.0 - self.start_ratio) * math.log(1 + 9 * progress) / math.log(10)
        else:
            ratio = self.start_ratio + (1.0 - self.start_ratio) * progress
        
        return min(ratio, 1.0)
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch for curriculum progression.
        
        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch
        self._rng = np.random.default_rng(self.seed + epoch)
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate indices for the current epoch.
        
        Yields:
            int: Sample index.
        """
        ratio = self._get_current_ratio()
        num_samples = max(1, int(len(self.data_source) * ratio))
        
        current_indices = self._sorted_indices[:num_samples]
        
        if self.shuffle_within:
            current_indices = list(current_indices)
            self._rng.shuffle(current_indices)
        
        for idx in current_indices:
            yield idx
    
    def __len__(self) -> int:
        """
        Get the number of samples for the current epoch.
        
        Returns:
            int: Number of samples.
        """
        ratio = self._get_current_ratio()
        return max(1, int(len(self.data_source) * ratio))
    
    def get_difficulty_stats(self) -> dict:
        """
        Get statistics about the current difficulty distribution.
        
        Returns:
            dict: Statistics including min, max, mean difficulty.
        """
        ratio = self._get_current_ratio()
        num_samples = max(1, int(len(self.data_source) * ratio))
        current_indices = self._sorted_indices[:num_samples]
        current_scores = [self.difficulty_scores[i] for i in current_indices]
        
        return {
            'epoch': self.epoch,
            'ratio': ratio,
            'num_samples': num_samples,
            'min_difficulty': min(current_scores),
            'max_difficulty': max(current_scores),
            'mean_difficulty': sum(current_scores) / len(current_scores)
        }
