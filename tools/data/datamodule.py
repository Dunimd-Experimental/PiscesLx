#!/usr/bin/env/python3
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

"""
Unified DataModule for PiscesL1 training.

This module provides a unified data loading interface supporting:
- Text data for language modeling
- Image-text pairs for vision-language training
- Audio-text pairs for speech-language training
- Video-text pairs for video-language training
- Multi-turn conversations for chat fine-tuning
- Preference pairs for DPO training
- Streaming data processing for large datasets
"""

import os
import sys
import json
import io
import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.dataset import T_co
import numpy as np

from utils.dc import PiscesLxLogger

_LOG = PiscesLxLogger(__name__)


@dataclass
class PiscesL1DataConfig:
    """Data module configuration."""
    
    train_data: Dict[str, Any] = field(default_factory=dict)
    val_data: Dict[str, Any] = field(default_factory=dict)
    test_data: Dict[str, Any] = field(default_factory=dict)
    
    batch_size: int = 4
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = True
    
    max_seq_length: int = 4096
    tokenizer_path: str = "./checkpoints/ruchbah"
    
    shuffle_buffer_size: int = 10000
    streaming_buffer_size: int = 1000
    
    use_async_loading: bool = True
    cache_data: bool = True
    cache_dir: str = "./data/cache"
    
    multimodal_fusion: bool = True
    
    def add_text_data(
        self,
        path: str,
        split: str = "train",
        weight: float = 1.0,
    ) -> None:
        """Add text data source."""
        data_dict = self.train_data if split == "train" else self.val_data
        if "text" not in data_dict:
            data_dict["text"] = []
        data_dict["text"].append({
            "path": path,
            "weight": weight,
        })
    
    def add_image_text_data(
        self,
        image_path: str,
        caption_path: str,
        split: str = "train",
        weight: float = 1.0,
    ) -> None:
        """Add image-text data source."""
        data_dict = self.train_data if split == "train" else self.val_data
        if "image_text" not in data_dict:
            data_dict["image_text"] = []
        data_dict["image_text"].append({
            "image_path": image_path,
            "caption_path": caption_path,
            "weight": weight,
        })
    
    def add_conversation_data(
        self,
        path: str,
        split: str = "train",
        weight: float = 1.0,
    ) -> None:
        """Add conversation data source."""
        data_dict = self.train_data if split == "train" else self.val_data
        if "conversation" not in data_dict:
            data_dict["conversation"] = []
        data_dict["conversation"].append({
            "path": path,
            "weight": weight,
        })
    
    def add_preference_data(
        self,
        path: str,
        split: str = "train",
        weight: float = 1.0,
    ) -> None:
        """Add preference data source for DPO."""
        data_dict = self.train_data if split == "train" else self.val_data
        if "preference" not in data_dict:
            data_dict["preference"] = []
        data_dict["preference"].append({
            "path": path,
            "weight": weight,
        })


class PiscesL1BaseDataset(Dataset):
    """Base dataset class for PiscesL1."""
    
    def __init__(
        self,
        config: PiscesL1DataConfig,
        tokenizer: Any,
        split: str = "train",
    ):
        """Initialize base dataset.
        
        Args:
            config: Data configuration.
            tokenizer: Tokenizer for encoding text.
            split: Dataset split ('train', 'val', 'test').
        """
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        self.data_sources = []
        self.weights = []
        
        self._load_data_sources()
        
        _LOG.info(
            f"Loaded {len(self)} samples for {split} from {len(self.data_sources)} sources"
        )
    
    def _load_data_sources(self) -> None:
        """Load data sources based on configuration."""
        data_dict = getattr(self.config, f"{self.split}_data", {})
        
        for data_type, sources in data_dict.items():
            if not isinstance(sources, list):
                continue
            
            for source in sources:
                if data_type == "text":
                    self._load_text_source(source)
                elif data_type == "image_text":
                    self._load_image_text_source(source)
                elif data_type == "conversation":
                    self._load_conversation_source(source)
                elif data_type == "preference":
                    self._load_preference_source(source)
    
    def _load_text_source(self, source: Dict[str, Any]) -> None:
        """Load text data source."""
        path = source.get("path", "")
        weight = source.get("weight", 1.0)
        
        if not os.path.exists(path):
            _LOG.warning(f"Text data not found: {path}")
            return
        
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        samples.append({
                            "type": "text",
                            "text": sample.get("text", sample.get("content", "")),
                        })
                    except json.JSONDecodeError:
                        samples.append({
                            "type": "text",
                            "text": line,
                        })
        
        self.data_sources.append(samples)
        self.weights.append(weight * len(samples))
    
    def _load_image_text_source(self, source: Dict[str, Any]) -> None:
        """Load image-text data source."""
        image_path = source.get("image_path", "")
        caption_path = source.get("caption_path", "")
        weight = source.get("weight", 1.0)
        
        if not os.path.exists(caption_path):
            _LOG.warning(f"Caption data not found: {caption_path}")
            return
        
        samples = []
        with open(caption_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        caption_data = json.loads(line)
                        samples.append({
                            "type": "image_text",
                            "image_path": os.path.join(
                                image_path,
                                caption_data.get("image_id", "") + ".jpg"
                            ),
                            "caption": caption_data.get("caption", ""),
                        })
                    except json.JSONDecodeError:
                        continue
        
        self.data_sources.append(samples)
        self.weights.append(weight * len(samples))
    
    def _load_conversation_source(self, source: Dict[str, Any]) -> None:
        """Load conversation data source."""
        path = source.get("path", "")
        weight = source.get("weight", 1.0)
        
        if not os.path.exists(path):
            _LOG.warning(f"Conversation data not found: {path}")
            return
        
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        conversation = json.loads(line)
                        samples.append({
                            "type": "conversation",
                            "messages": conversation.get("messages", []),
                        })
                    except json.JSONDecodeError:
                        continue
        
        self.data_sources.append(samples)
        self.weights.append(weight * len(samples))
    
    def _load_preference_source(self, source: Dict[str, Any]) -> None:
        """Load preference data source for DPO."""
        path = source.get("path", "")
        weight = source.get("weight", 1.0)
        
        if not os.path.exists(path):
            _LOG.warning(f"Preference data not found: {path}")
            return
        
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        preference = json.loads(line)
                        if "chosen" in preference and "rejected" in preference:
                            samples.append({
                                "type": "preference",
                                "prompt": preference.get("prompt", ""),
                                "chosen": preference.get("chosen", ""),
                                "rejected": preference.get("rejected", ""),
                            })
                    except json.JSONDecodeError:
                        continue
        
        self.data_sources.append(samples)
        self.weights.append(weight * len(samples))
    
    def __len__(self) -> int:
        if not self.weights:
            return 0
        return int(sum(self.weights))
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        raise NotImplementedError


class PiscesL1TextDataset(PiscesL1BaseDataset):
    """Text dataset for language modeling."""
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a text sample."""
        source_idx = 0
        sample_idx = idx
        
        for i, weight in enumerate(self.weights):
            if sample_idx < weight:
                source_idx = i
                break
            sample_idx -= weight
        
        sample = self.data_sources[source_idx][sample_idx % len(self.data_sources[source_idx])]
        
        text = sample.get("text", "")
        
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PiscesL1MultimodalDataset(PiscesL1BaseDataset):
    """Multimodal dataset supporting text, image, audio, and video."""
    
    def __init__(
        self,
        config: PiscesL1DataConfig,
        tokenizer: Any,
        split: str = "train",
        vision_encoder: Any = None,
        audio_encoder: Any = None,
    ):
        """Initialize multimodal dataset.
        
        Args:
            config: Data configuration.
            tokenizer: Text tokenizer.
            split: Dataset split.
            vision_encoder: Optional vision encoder for image processing.
            audio_encoder: Optional audio encoder for audio processing.
        """
        super().__init__(config, tokenizer, split)
        
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        
        self.image_processor = None
        if vision_encoder is not None and hasattr(vision_encoder, 'image_processor'):
            self.image_processor = vision_encoder.image_processor
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a multimodal sample."""
        source_idx = 0
        sample_idx = idx
        
        for i, weight in enumerate(self.weights):
            if sample_idx < weight:
                source_idx = i
                break
            sample_idx -= weight
        
        sample = self.data_sources[source_idx][sample_idx % len(self.data_sources[source_idx])]
        
        sample_type = sample.get("type", "text")
        
        if sample_type == "text":
            return self._get_text_sample(sample)
        elif sample_type == "image_text":
            return self._get_image_text_sample(sample)
        elif sample_type == "conversation":
            return self._get_conversation_sample(sample)
        elif sample_type == "preference":
            return self._get_preference_sample(sample)
        else:
            return self._get_text_sample(sample)
    
    def _get_text_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get text sample."""
        text = sample.get("text", "")
        
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
            "modality": "text",
        }
    
    def _get_image_text_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get image-text sample."""
        image_path = sample.get("image_path", "")
        caption = sample.get("caption", "")
        
        encoding = self.tokenizer(
            caption,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        image_features = None
        if self.image_processor is not None and os.path.exists(image_path):
            try:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                image_inputs = self.image_processor(
                    images=image,
                    return_tensors="pt",
                )
                image_features = image_inputs.get("pixel_values", None)
            except Exception as e:
                _LOG.warning(f"Failed to load image: {image_path}, error: {e}")
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
            "image_features": image_features,
            "modality": "image_text",
        }
    
    def _get_conversation_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get conversation sample."""
        messages = sample.get("messages", [])
        
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"{role}: {content}\n"
        
        encoding = self.tokenizer(
            text.strip(),
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
            "messages": messages,
            "modality": "conversation",
        }
    
    def _get_preference_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Get preference sample for DPO."""
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        rejected = sample.get("rejected", "")
        
        chosen_text = f"User: {prompt}\nAssistant: {chosen}"
        rejected_text = f"User: {prompt}\nAssistant: {rejected}"
        
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
            "modality": "preference",
        }


class PiscesL1StreamingDataset(IterableDataset):
    """Streaming dataset for large-scale data processing."""
    
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: Any,
        max_seq_length: int = 4096,
        shuffle: bool = True,
        buffer_size: int = 10000,
    ):
        """Initialize streaming dataset.
        
        Args:
            file_paths: List of data file paths.
            tokenizer: Tokenizer for encoding text.
            max_seq_length: Maximum sequence length.
            shuffle: Whether to shuffle data.
            buffer_size: Buffer size for shuffling.
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        
        self.buffer = []
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over streaming data."""
        buffer = []
        
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        buffer.append(sample)
                        
                        if len(buffer) >= self.buffer_size:
                            if self.shuffle:
                                np.random.shuffle(buffer)
                            
                            for item in buffer:
                                processed = self._process_sample(item)
                                if processed:
                                    yield processed
                            
                            buffer = []
                    
                    except json.JSONDecodeError:
                        continue
        
        if buffer:
            if self.shuffle:
                np.random.shuffle(buffer)
            
            for item in buffer:
                processed = self._process_sample(item)
                if processed:
                    yield processed
    
    def _process_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single sample."""
        text = sample.get("text", sample.get("content", ""))
        
        if not text:
            return None
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0),
        }


class PiscesL1DataModule:
    """Unified DataModule for PiscesL1 training."""
    
    def __init__(
        self,
        config: PiscesL1DataConfig,
        tokenizer: Any,
        vision_encoder: Any = None,
        audio_encoder: Any = None,
    ):
        """Initialize DataModule.
        
        Args:
            config: Data configuration.
            tokenizer: Text tokenizer.
            vision_encoder: Optional vision encoder.
            audio_encoder: Optional audio encoder.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        
        logger.info("PiscesL1DataModule initialized")
    
    def setup(self, stage: str = "fit") -> None:
        """Setup datasets for different stages."""
        if stage in ("fit", "train"):
            self._train_dataset = self._create_dataset("train")
        
        if stage in ("fit", "validate"):
            self._val_dataset = self._create_dataset("val")
        
        if stage in ("test", "predict"):
            self._test_dataset = self._create_dataset("test")
    
    def _create_dataset(self, split: str) -> Dataset:
        """Create dataset for given split."""
        if self.config.multimodal_fusion:
            return PiscesL1MultimodalDataset(
                config=self.config,
                tokenizer=self.tokenizer,
                split=split,
                vision_encoder=self.vision_encoder,
                audio_encoder=self.audio_encoder,
            )
        else:
            return PiscesL1TextDataset(
                config=self.config,
                tokenizer=self.tokenizer,
                split=split,
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self._train_dataset is None:
            self.setup("fit")
        
        return DataLoader(
            self._train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self._val_dataset is None:
            self.setup("fit")
        
        return DataLoader(
            self._val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self._test_dataset is None:
            self.setup("test")
        
        return DataLoader(
            self._test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )
    
    def teardown(self, stage: str = "fit") -> None:
        """Cleanup resources."""
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        
        logger.info("PiscesL1DataModule teardown complete")


def create_datamodule(
    config: PiscesL1DataConfig,
    tokenizer: Any,
    vision_encoder: Any = None,
    audio_encoder: Any = None,
) -> PiscesL1DataModule:
    """Factory function to create DataModule.
    
    Args:
        config: Data configuration.
        tokenizer: Text tokenizer.
        vision_encoder: Optional vision encoder.
        audio_encoder: Optional audio encoder.
        
    Returns:
        Initialized DataModule.
    """
    return PiscesL1DataModule(
        config=config,
        tokenizer=tokenizer,
        vision_encoder=vision_encoder,
        audio_encoder=audio_encoder,
    )


def datamodule_main(args):
    """Main entry point for data module testing."""
    from transformers import AutoTokenizer
    
    config = PiscesL1DataConfig(
        batch_size=4,
        max_seq_length=2048,
    )
    
    config.add_text_data("./data/text_train.jsonl", split="train", weight=1.0)
    config.add_conversation_data("./data/conversation_train.jsonl", split="train", weight=0.5)
    
    tokenizer = AutoTokenizer.from_pretrained("./checkpoints/ruchbah")
    
    datamodule = create_datamodule(config, tokenizer)
    
    datamodule.setup("fit")
    
    train_loader = datamodule.train_dataloader()
    
    _LOG.info(f"Training batches: {len(train_loader)}")
    
    for batch in train_loader:
        logger.info(f"Batch keys: {batch.keys()}")
        break
    
    datamodule.teardown()
    
    _LOG.info("DataModule test completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DataModule for PiscesL1")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--test", action="store_true", default=False)
    
    args = parser.parse_args()
    
    datamodule_main(args)
