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

from __future__ import annotations

from typing import Optional, Dict, Any, List, Callable, Iterator
from torch.utils.data import DataLoader, Sampler, DistributedSampler as TorchDistributedSampler
from torch.utils.data import get_worker_info
import torch
import math
import random
from dataclasses import dataclass
from collections import defaultdict


class PiscesLxToolsDataBatchConfig:
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        max_token_length: int = 4096,
        dynamic_batching: bool = False,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.max_token_length = max_token_length
        self.dynamic_batching = dynamic_batching
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank


class PiscesLxToolsDataDynamicBatchSampler(Sampler):
    def __init__(
        self,
        data_source,
        max_tokens: int = 4096,
        max_batch_size: int = 64,
        min_batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = True,
        length_fn: Optional[Callable[[int], int]] = None
    ):
        self.data_source = data_source
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.length_fn = length_fn or self._default_length_fn

        self._lengths: Optional[List[int]] = None
        self._batches: Optional[List[List[int]]] = None

    def _default_length_fn(self, idx: int) -> int:
        try:
            if hasattr(self.data_source, 'ds'):
                item = self.data_source.ds[idx]
                if 'input_ids' in item:
                    return len(item['input_ids'])
                text = item.get('text', '') or item.get('content', '') or item.get('input', '')
                return len(str(text).split())
        except Exception:
            pass
        return 128

    def _compute_lengths(self):
        if self._lengths is not None:
            return self._lengths

        self._lengths = []
        for i in range(len(self.data_source)):
            self._lengths.append(self.length_fn(i))

        return self._lengths

    def _create_batches(self):
        if self._batches is not None:
            return self._batches

        lengths = self._compute_lengths()
        indices = list(range(len(self.data_source)))

        if self.shuffle:
            paired = list(zip(indices, lengths))
            random.shuffle(paired)
            indices, lengths = zip(*paired)
            indices, lengths = list(indices), list(lengths)

        sorted_paired = sorted(zip(indices, lengths), key=lambda x: x[1])

        self._batches = []
        current_batch: List[int] = []
        current_tokens = 0

        for idx, length in sorted_paired:
            if current_tokens + length <= self.max_tokens and len(current_batch) < self.max_batch_size:
                current_batch.append(idx)
                current_tokens += length
            else:
                if len(current_batch) >= self.min_batch_size:
                    self._batches.append(current_batch)
                current_batch = [idx]
                current_tokens = length

        if current_batch and len(current_batch) >= self.min_batch_size:
            if self.drop_last or len(current_batch) >= self.min_batch_size:
                self._batches.append(current_batch)

        if self.shuffle:
            random.shuffle(self._batches)

        return self._batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._create_batches()
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return len(self._create_batches())

    def reset(self):
        self._batches = None


class PiscesLxToolsDataDistributedSampler(TorchDistributedSampler):
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class PiscesLxToolsDataCurriculumSampler(Sampler):
    def __init__(
        self,
        data_source,
        difficulty_fn: Optional[Callable[[int], float]] = None,
        initial_difficulty: float = 0.3,
        final_difficulty: float = 1.0,
        total_steps: int = 10000,
        current_step: int = 0,
        batch_size: int = 32,
        shuffle_within_batch: bool = True
    ):
        self.data_source = data_source
        self.difficulty_fn = difficulty_fn or self._default_difficulty_fn
        self.initial_difficulty = initial_difficulty
        self.final_difficulty = final_difficulty
        self.total_steps = total_steps
        self.current_step = current_step
        self.batch_size = batch_size
        self.shuffle_within_batch = shuffle_within_batch

        self._difficulties: Optional[List[float]] = None

    def _default_difficulty_fn(self, idx: int) -> float:
        try:
            if hasattr(self.data_source, 'ds'):
                item = self.data_source.ds[idx]
                text = item.get('text', '') or item.get('content', '') or item.get('input', '')
                length = len(str(text).split())

                return min(length / 1000, 1.0)
        except Exception:
            pass
        return 0.5

    def _compute_difficulties(self):
        if self._difficulties is not None:
            return self._difficulties

        self._difficulties = []
        for i in range(len(self.data_source)):
            self._difficulties.append(self.difficulty_fn(i))

        return self._difficulties

    def _get_current_threshold(self) -> float:
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.initial_difficulty + (self.final_difficulty - self.initial_difficulty) * progress

    def __iter__(self) -> Iterator[int]:
        difficulties = self._compute_difficulties()
        threshold = self._get_current_threshold()

        eligible_indices = [
            i for i, d in enumerate(difficulties)
            if d <= threshold
        ]

        if not eligible_indices:
            eligible_indices = list(range(len(self.data_source)))

        random.shuffle(eligible_indices)

        for i in range(0, len(eligible_indices), self.batch_size):
            batch = eligible_indices[i:i + self.batch_size]
            if self.shuffle_within_batch:
                random.shuffle(batch)
            yield from batch

    def __len__(self) -> int:
        return len(self.data_source)

    def step(self):
        self.current_step += 1

    def set_step(self, step: int):
        self.current_step = step


class PiscesLxToolsDataPrefetchDataLoader:
    def __init__(
        self,
        dataloader: DataLoader,
        prefetch_batches: int = 2,
        device: Optional[torch.device] = None
    ):
        self.dataloader = dataloader
        self.prefetch_batches = prefetch_batches
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        self._prefetch_queue: List[Any] = []
        self._iterator: Optional[Iterator] = None

    def _prefetch_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        prefetched = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prefetched[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                prefetched[key] = self._prefetch_batch(value)
            elif isinstance(value, list):
                prefetched[key] = [
                    v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                prefetched[key] = value
        return prefetched

    def __iter__(self):
        self._iterator = iter(self.dataloader)
        self._prefetch_queue = []

        for _ in range(self.prefetch_batches):
            try:
                batch = next(self._iterator)
                self._prefetch_queue.append(self._prefetch_batch(batch))
            except StopIteration:
                break

        while self._prefetch_queue:
            yield self._prefetch_queue.pop(0)

            try:
                batch = next(self._iterator)
                self._prefetch_queue.append(self._prefetch_batch(batch))
            except StopIteration:
                pass

    def __len__(self):
        return len(self.dataloader)


class PiscesLxToolsDataOptimizedDataLoader:
    def __init__(
        self,
        dataset,
        batch_config: Optional[PiscesLxToolsDataBatchConfig] = None,
        collate_fn: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.cfg = batch_config or PiscesLxToolsDataBatchConfig()
        self.collate_fn = collate_fn

    def get(self) -> DataLoader:
        if hasattr(self.dataset, "__iter__") and not hasattr(self.dataset, "__len__"):
            return self._get_iterable_dataloader()

        if self.cfg.dynamic_batching:
            return self._get_dynamic_dataloader()

        if self.cfg.distributed:
            return self._get_distributed_dataloader()

        return self._get_standard_dataloader()

    def _get_standard_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
            collate_fn=self.collate_fn
        )

    def _get_iterable_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False
        )

    def _get_dynamic_dataloader(self) -> DataLoader:
        sampler = PiscesLxToolsDataDynamicBatchSampler(
            self.dataset,
            max_tokens=self.cfg.max_token_length,
            max_batch_size=self.cfg.batch_size,
            drop_last=self.cfg.drop_last,
            shuffle=True
        )

        return DataLoader(
            self.dataset,
            batch_sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
            collate_fn=self.collate_fn
        )

    def _get_distributed_dataloader(self) -> DataLoader:
        sampler = PiscesLxToolsDataDistributedSampler(
            self.dataset,
            num_replicas=self.cfg.world_size,
            rank=self.cfg.rank,
            shuffle=True,
            drop_last=self.cfg.drop_last
        )

        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
            collate_fn=self.collate_fn
        )

    def get_prefetched(self, prefetch_batches: int = 2) -> PiscesLxToolsDataPrefetchDataLoader:
        return PiscesLxToolsDataPrefetchDataLoader(
            self.get(),
            prefetch_batches=prefetch_batches
        )

    def get_curriculum(
        self,
        difficulty_fn: Optional[Callable] = None,
        total_steps: int = 10000
    ) -> DataLoader:
        sampler = PiscesLxToolsDataCurriculumSampler(
            self.dataset,
            difficulty_fn=difficulty_fn,
            total_steps=total_steps,
            batch_size=self.cfg.batch_size
        )

        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None,
            persistent_workers=self.cfg.persistent_workers if self.cfg.num_workers > 0 else False,
            collate_fn=self.collate_fn
        )


def create_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    dynamic_batching: bool = False,
    max_token_length: int = 4096,
    collate_fn: Optional[Callable] = None
) -> DataLoader:
    config = PiscesLxToolsDataBatchConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        max_token_length=max_token_length,
        dynamic_batching=dynamic_batching,
        distributed=distributed,
        world_size=world_size,
        rank=rank
    )

    loader = PiscesLxToolsDataOptimizedDataLoader(dataset, config, collate_fn)
    return loader.get()
