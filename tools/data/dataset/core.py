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

import os
import torch
import threading
import queue
import time
from datasets import load_from_disk
from torch.utils.data import Dataset as TorchDataset
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from model.tokenizer import YvTokenizer
from utils.paths import get_cache_dir
from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from model.multimodal import (
    YvVisionEncoder as VisionEncoder,
    YvAudioEncoder as AudioEncoder,
    YvDocEncoder as DocEncoder,
    YvVideoEncoder as VideoEncoder
)

IMAGE_KEYS = [
    "image", "img_path", "image_path", "picture", "pic",
    "img", "images", "img_file", "image_file", "photo",
    "screenshot", "frame", "frames", "visual", "visual_input"
]
AUDIO_KEYS = [
    "audio", "audio_path", "wav", "sound",
    "audio_file", "audio_input", "speech", "voice",
    "waveform", "spectrogram", "mel", "audio_data"
]
DOC_KEYS = [
    "doc", "document", "doc_path", "pdf",
    "document_path", "doc_file", "text_file", "txt",
    "markdown", "md", "html", "document_input", "file_path"
]
VIDEO_KEYS = [
    "video", "video_path", "mp4", "avi", "mov", "mkv",
    "video_file", "video_input", "clip", "movie", "footage",
    "video_data", "frames_path", "video_frames_path"
]

_LOG = PiscesLxLogger("PiscesLx.Data", file_path=get_log_file("PiscesLx.Data"), enable_file=True)


@dataclass
class PiscesLxDatasetConfig:
    cache_enabled: bool = True
    cache_size: int = 10000
    prefetch_enabled: bool = True
    prefetch_workers: int = 2
    prefetch_queue_size: int = 100
    augmentation_enabled: bool = False
    text_augmentation_prob: float = 0.1
    image_augmentation_prob: float = 0.1
    max_text_length: int = 8192
    truncate_mode: str = "tail"
    return_quality_score: bool = False


class PiscesLxDataPrefetcher:
    def __init__(
        self,
        dataset: TorchDataset,
        queue_size: int = 100,
        num_workers: int = 2
    ):
        self.dataset = dataset
        self.queue_size = queue_size
        self.num_workers = num_workers
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._workers: List[threading.Thread] = []
        self._current_idx = 0
        self._lock = threading.Lock()

    def start(self):
        self._stop_event.clear()
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self._workers.append(worker)

    def stop(self):
        self._stop_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._workers.clear()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    if self._current_idx >= len(self.dataset):
                        break
                    idx = self._current_idx
                    self._current_idx += 1

                item = self.dataset[idx]

                if not self._stop_event.is_set():
                    self._queue.put(item, block=True, timeout=1.0)

            except Exception:
                time.sleep(0.01)

    def get(self, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        try:
            return self._queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def __iter__(self):
        self.start()
        try:
            while True:
                item = self.get()
                if item is None:
                    break
                yield item
        finally:
            self.stop()


class PiscesLxDataAugmentationPipeline:
    def __init__(
        self,
        text_aug_prob: float = 0.1,
        image_aug_prob: float = 0.1
    ):
        self.text_aug_prob = text_aug_prob
        self.image_aug_prob = image_aug_prob
        self._text_augmenter = None
        self._image_augmenter = None

    def _lazy_init_text_augmenter(self):
        if self._text_augmenter is not None:
            return

        try:
            from ..augment.text_aug import PiscesLxDataTextAugmenter
            self._text_augmenter = PiscesLxDataTextAugmenter()
        except ImportError:
            self._text_augmenter = None

    def _lazy_init_image_augmenter(self):
        if self._image_augmenter is not None:
            return

        try:
            from ..augment.image_aug import PiscesLxDataImageAugmenter
            self._image_augmenter = PiscesLxDataImageAugmenter()
        except ImportError:
            self._image_augmenter = None

    def augment_text(self, text: str) -> str:
        import random
        if random.random() > self.text_aug_prob:
            return text

        self._lazy_init_text_augmenter()

        if self._text_augmenter is None:
            return text

        try:
            return self._text_augmenter.augment(text)
        except Exception:
            return text

    def augment_image(self, image_path: str) -> str:
        import random
        if random.random() > self.image_aug_prob:
            return image_path

        self._lazy_init_image_augmenter()

        if self._image_augmenter is None:
            return image_path

        try:
            return self._image_augmenter.augment(image_path)
        except Exception:
            return image_path


class Dataset(TorchDataset):
    _global_cache: Dict[str, Any] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        name: str,
        subset: Optional[str] = None,
        split: str = "train",
        config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
        dataset_config: Optional[PiscesLxDatasetConfig] = None
    ):
        self.subset = subset
        self.split = split
        self.config = config or {}
        self.dataset_config = dataset_config or PiscesLxDatasetConfig()

        try:
            from types import SimpleNamespace
            if isinstance(self.config, dict):
                self.config = SimpleNamespace(**self.config)
        except Exception:
            pass

        self.max_samples = max_samples

        self._image_keys = getattr(self.config, 'custom_image_keys', None) or IMAGE_KEYS
        self._audio_keys = getattr(self.config, 'custom_audio_keys', None) or AUDIO_KEYS
        self._doc_keys = getattr(self.config, 'custom_doc_keys', None) or DOC_KEYS
        self._video_keys = getattr(self.config, 'custom_video_keys', None) or VIDEO_KEYS

        self._force_vision = getattr(self.config, 'force_enable_vision', False)
        self._force_audio = getattr(self.config, 'force_enable_audio', False)
        self._force_doc = getattr(self.config, 'force_enable_doc', False)
        self._force_video = getattr(self.config, 'force_enable_video', False)

        data_cache = cache_dir or get_cache_dir("data_cache")
        cache_path = os.path.join(str(data_cache), self.subset)

        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Dataset cache not found at {cache_path}. Please run downloader to prepare local cache.")

        ds = load_from_disk(cache_path)
        if hasattr(ds, 'keys'):
            if self.split in ds:
                ds = ds[self.split]
            else:
                available_splits = list(ds.keys()) if hasattr(ds, 'keys') else []
                if available_splits:
                    default_split = available_splits[0]
                    _LOG.warning(f"Dataset split '{self.split}' not found. Using first available split: '{default_split}'")
                    ds = ds[default_split]
                    self.split = default_split
        self.ds = ds

        if self.max_samples is not None and len(self.ds) > self.max_samples:
            self.ds = self.ds.select(range(self.max_samples))

        self.tokenizer = YvTokenizer()
        self._vision_encoder = None
        self._audio_encoder = None
        self._doc_encoder = None
        self._video_encoder = None

        self._augmentation_pipeline: Optional[PiscesLxDataAugmentationPipeline] = None
        if self.dataset_config.augmentation_enabled:
            self._augmentation_pipeline = PiscesLxDataAugmentationPipeline(
                text_aug_prob=self.dataset_config.text_augmentation_prob,
                image_aug_prob=self.dataset_config.image_augmentation_prob
            )

        self._prefetcher: Optional[PiscesLxDataPrefetcher] = None
        self._sample_cache: Dict[int, Dict[str, Any]] = {}
        self._cache_size = self.dataset_config.cache_size

    @property
    def vision_encoder(self):
        if self._vision_encoder is None and self.config:
            self._vision_encoder = VisionEncoder(self.config)
        return self._vision_encoder

    @property
    def audio_encoder(self):
        if self._audio_encoder is None and self.config:
            self._audio_encoder = AudioEncoder(self.config)
        return self._audio_encoder

    @property
    def doc_encoder(self):
        if self._doc_encoder is None and self.config:
            self._doc_encoder = DocEncoder(self.config)
        return self._doc_encoder

    @property
    def video_encoder(self):
        if self._video_encoder is None and self.config:
            self._video_encoder = VideoEncoder(self.config)
        return self._video_encoder

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.dataset_config.cache_enabled and idx in self._sample_cache:
            return self._sample_cache[idx].copy()

        item = self.ds[idx]
        text = self._extract_text(item)

        if not text:
            text = "<empty>"

        if self._augmentation_pipeline and self.dataset_config.augmentation_enabled:
            text = self._augmentation_pipeline.augment_text(text)

        try:
            ids = self.tokenizer.encode(text, return_tensors="pt")[0]
            vocab_size = len(self.tokenizer)
            ids = torch.clamp(ids, 0, vocab_size - 1)

            if len(ids) > self.dataset_config.max_text_length:
                if self.dataset_config.truncate_mode == "head":
                    ids = ids[-self.dataset_config.max_text_length:]
                elif self.dataset_config.truncate_mode == "middle":
                    half = self.dataset_config.max_text_length // 2
                    ids = torch.cat([ids[:half], ids[-half:]])
                else:
                    ids = ids[:self.dataset_config.max_text_length]
        except Exception:
            ids = torch.tensor([0], dtype=torch.long)

        pixel_values = self._process_mm_lazy(item, self._image_keys, "vision_encoder", "image", self._force_vision)
        audio_input = self._process_mm_lazy(item, self._audio_keys, "audio_encoder", "audio", self._force_audio)
        doc_input = self._process_mm_lazy(item, self._doc_keys, "doc_encoder", "document", self._force_doc)
        video_frames = self._process_mm_lazy(item, self._video_keys, "video_encoder", "video", self._force_video)

        result = {
            "input_ids": ids,
            "labels": ids.clone(),
            "pixel_values": pixel_values,
            "audio_input": audio_input if audio_input is not None else {"input_values": None},
            "doc_input": doc_input,
            "video_frames": video_frames,
        }

        if self.dataset_config.return_quality_score:
            try:
                from ..clean.quality import PiscesLxToolsDataQualityController
                result["quality_score"] = PiscesLxToolsDataQualityController.calculate_text_quality_score(text)
            except Exception:
                result["quality_score"] = 1.0

        if self.dataset_config.cache_enabled and len(self._sample_cache) < self._cache_size:
            self._sample_cache[idx] = result.copy()

        return result

    def get_prefetcher(self, queue_size: int = 100, num_workers: int = 2) -> PiscesLxDataPrefetcher:
        if self._prefetcher is None:
            self._prefetcher = PiscesLxDataPrefetcher(
                self, queue_size=queue_size, num_workers=num_workers
            )
        return self._prefetcher

    def clear_cache(self):
        self._sample_cache.clear()

    def _process_mm_lazy(
        self,
        item: Dict[str, Any],
        keys: list,
        encoder_attr: str,
        kind: str,
        force_enable: bool = False
    ) -> Optional[Any]:
        path = None
        for key in keys:
            value = item.get(key) if isinstance(item, dict) else None
            if isinstance(value, str) and value.strip():
                path = value.strip()
                break

        if not path and not force_enable:
            return None

        encoder = getattr(self, encoder_attr)
        if not encoder or not getattr(encoder, "enabled", False):
            return None

        if not path:
            return None

        if self._augmentation_pipeline and kind == "image" and self.dataset_config.augmentation_enabled:
            path = self._augmentation_pipeline.augment_image(path)

        try:
            if kind == "image":
                return encoder.process_image(path)
            if kind == "audio":
                return encoder.process_audio(path)
            if kind == "document":
                return encoder.process_doc(path)
            if kind == "video":
                return encoder.process_video(path)
        except Exception:
            pass
        return None

    def _extract_text(self, item: Dict[str, Any]) -> str:
        from tools.data import TEXT_FIELD_KEYS
        if isinstance(item, dict):
            for key in TEXT_FIELD_KEYS:
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            conversations = item.get("conversations")
            if isinstance(conversations, list) and conversations:
                text_parts = []
                for turn in conversations:
                    if isinstance(turn, dict):
                        content = turn.get("value") or turn.get("content") or turn.get("text")
                        if content and str(content).strip():
                            role = turn.get("from", turn.get("role", ""))
                            text_parts.append(f"{role}: {content}" if role else str(content))
                if text_parts:
                    return "\n".join(text_parts)

            for value in item.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    @classmethod
    def get_global_cache(cls, key: str) -> Optional[Any]:
        with cls._cache_lock:
            return cls._global_cache.get(key)

    @classmethod
    def set_global_cache(cls, key: str, value: Any):
        with cls._cache_lock:
            cls._global_cache[key] = value

    @classmethod
    def clear_global_cache(cls):
        with cls._cache_lock:
            cls._global_cache.clear()


class PiscesLxDataIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        buffer_size: int = 1000
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self):
        prefetcher = self.dataset.get_prefetcher()

        if self.shuffle:
            import random
            indices = list(range(len(self.dataset)))
            random.shuffle(indices)

            for idx in indices:
                yield self.dataset[idx]
        else:
            for item in prefetcher:
                yield item

    def __len__(self):
        return len(self.dataset)
