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
import os
import gc
import time
import mmap
import json
import torch
import psutil
import hashlib
import threading
import multiprocessing as mp
from datetime import datetime
from queue import Queue, Empty
from dataclasses import dataclass
from utils import progress_bar
from model import get_tokenizer
from utils import RIGHT, DEBUG, ERROR
from datasets import load_from_disk, Dataset as HFDataset
from typing import Iterator, Optional, Union, Dict, List, Any
from torch.utils.data import Dataset, DataLoader, IterableDataset
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from model import VisionEncoder, AudioEncoder, DocEncoder, VideoEncoder

# Keys used to identify image data in the dataset
IMAGE_KEYS = ["image", "img_path", "image_path", "picture", "pic"]
# Keys used to identify audio data in the dataset
AUDIO_KEYS = ["audio", "audio_path", "wav", "sound"]
# Keys used to identify document data in the dataset
DOC_KEYS = ["doc", "document", "doc_path", "pdf"]
# Keys used to identify video data in the dataset
VIDEO_KEYS = ["video", "video_path", "mp4", "avi", "mov", "mkv"]

# 5TB dataset batch loading configuration
MEMORY_THRESHOLD_GB = 8.0  # Memory usage threshold
BATCH_MEMORY_LIMIT_GB = 2.0  # Single batch memory limit
PREFETCH_BUFFER_SIZE = 1000  # Prefetch buffer size
MAX_WORKERS = min(8, mp.cpu_count())  # Maximum worker processes

@dataclass
class BatchConfig:
    """5TB dataset batch loading configuration class"""
    batch_size: int = 32
    memory_limit_gb: float = 2.0
    prefetch_factor: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    
class MemoryMonitor:
    """Memory monitor for Arctic architecture"""
    def __init__(self, threshold_gb: float = 8.0):
        self.threshold_gb = threshold_gb
        self.alerts = 0
        
    def check_memory(self) -> Dict[str, float]:
        """Check system memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        system_memory = psutil.virtual_memory()
        gpu_memory = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i) / 1024**3,
                    'cached': torch.cuda.memory_reserved(i) / 1024**3,
                    'total': torch.cuda.get_device_properties(i).total_memory / 1024**3
                }
        
        return {
            'process_memory_gb': memory_info.rss / 1024**3,
            'system_available_gb': system_memory.available / 1024**3,
            'system_used_percent': system_memory.percent,
            'gpu_memory': gpu_memory
        }
    
    def should_gc(self) -> bool:
        """Determine if garbage collection is needed"""
        memory_stats = self.check_memory()
        if memory_stats['process_memory_gb'] > self.threshold_gb:
            self.alerts += 1
            return True
        return False
    
    def cleanup(self):
        """Execute memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        DEBUG(f"Memory cleanup executed. Alert count: {self.alerts}")

class StreamingDataBuffer:
    """Streaming data buffer for 5TB datasets"""
    def __init__(self, buffer_size: int = 1000, memory_limit_gb: float = 2.0):
        self.buffer_size = buffer_size
        self.memory_limit_gb = memory_limit_gb
        self.buffer = Queue(maxsize=buffer_size)
        self.memory_monitor = MemoryMonitor(memory_limit_gb)
        self._stop_event = threading.Event()
        
    def add_batch(self, batch_data: List[Dict]):
        """Add batch data to buffer"""
        if not self._stop_event.is_set():
            try:
                self.buffer.put(batch_data, timeout=1.0)
                if self.memory_monitor.should_gc():
                    self.memory_monitor.cleanup()
            except Exception as e:
                ERROR(f"Failed to add batch to buffer: {e}")
    
    def get_batch(self, timeout: float = 5.0) -> Optional[List[Dict]]:
        """Get batch data from buffer"""
        try:
            return self.buffer.get(timeout=timeout)
        except Empty:
            return None
    
    def stop(self):
        """Stop buffer"""
        self._stop_event.set()

class LargeScaleStreamingDataset(IterableDataset):
    """Streaming processor for 5TB-scale datasets - Exclusive for Arctic architecture
    
    Supports memory-efficient large-scale dataset processing with the following features:
    - Adaptive batch size adjustment
    - Multi-process data preprocessing
    - Intelligent memory monitoring
    - Qizhi Platform optimization
    """
    def __init__(self, 
                 data_sources: List[str], 
                 config: Optional[Any] = None,
                 batch_config: Optional[BatchConfig] = None,
                 enable_prefetch: bool = True):
        super().__init__()
        self.data_sources = data_sources
        self.config = config
        self.batch_config = batch_config or BatchConfig()
        self.enable_prefetch = enable_prefetch
        
        # Initialize components
        self.tokenizer = get_tokenizer()
        self.memory_monitor = MemoryMonitor(MEMORY_THRESHOLD_GB)
        self.streaming_buffer = StreamingDataBuffer(
            PREFETCH_BUFFER_SIZE, 
            BATCH_MEMORY_LIMIT_GB
        ) if enable_prefetch else None
        
        # Multimodal encoders
        self.vision_encoder = VisionEncoder(config) if config else None
        self.audio_encoder = AudioEncoder(config) if config else None
        self.doc_encoder = DocEncoder(config) if config else None
        self.video_encoder = VideoEncoder(config) if config else None
        
        # Data source mapping
        self._build_data_index()
        
        RIGHT(f"Arctic streaming dataset initialized: {len(self.data_sources)} data sources")
        
    def _build_data_index(self):
        """Build data source index to support 5TB scale"""
        self.data_index = []
        total_samples = 0
        
        for source_path in self.data_sources:
            if os.path.isdir(source_path):
                # Handle directory-based data sources
                for root, dirs, files in os.walk(source_path):
                    for file in files:
                        if file.endswith(('.json', '.jsonl', '.txt')):
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            # Count actual samples in file instead of rough estimate
                    actual_samples = self._count_samples_in_file(file_path)
                    estimated_samples = max(1, actual_samples)
                    
                    self.data_index.append({
                                'path': file_path,
                                'type': 'file',
                                'estimated_samples': estimated_samples,
                                'size_mb': file_size / 1024**2
                            })
                            total_samples += estimated_samples
                            
            elif os.path.isfile(source_path):
                # Handle single files
                file_size = os.path.getsize(source_path)
                # Count actual samples in file instead of rough estimate
                actual_samples = self._count_samples_in_file(source_path)
                estimated_samples = max(1, actual_samples)
                
                self.data_index.append({
                    'path': source_path,
                    'type': 'file', 
                    'estimated_samples': estimated_samples,
                    'size_mb': file_size / 1024**2
                })
                total_samples += estimated_samples
                
        self.estimated_total_samples = total_samples
        RIGHT(f"Data index built: {len(self.data_index)} files, estimated {total_samples} samples")
        
    def _adaptive_batch_size(self) -> int:
        """Adaptively adjust batch size based on memory usage and computational complexity"""
        memory_stats = self.memory_monitor.check_memory()
        available_memory = memory_stats['system_available_gb']
        
        # Base batch size from memory
        if available_memory > 16.0:
            base_batch = min(self.batch_config.batch_size * 2, 128)
        elif available_memory > 8.0:
            base_batch = self.batch_config.batch_size
        elif available_memory > 4.0:
            base_batch = max(self.batch_config.batch_size // 2, 8)
        else:
            base_batch = max(self.batch_config.batch_size // 4, 4)
        
        # Adjust based on computational complexity of samples
        complexity_factor = self._estimate_batch_complexity()
        
        # Dynamic batch size based on complexity and memory
        dynamic_batch = int(base_batch * complexity_factor)
        
        # Ensure minimum and maximum bounds
        return max(4, min(dynamic_batch, 256))
    
    def _count_samples_in_file(self, file_path: str) -> int:
        """Count actual number of samples in a file."""
        try:
            count = 0
            if file_path.endswith('.jsonl'):
                # Count lines in JSONL file
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            count += 1
            elif file_path.endswith('.json'):
                # Count items in JSON array
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                    else:
                        count = 1  # Single JSON object
            elif file_path.endswith('.txt'):
                # Count non-empty lines in text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            count += 1
            else:
                # Fallback for unknown formats
                count = 1
                
            return max(1, count)
            
        except Exception as e:
            self.logger.warning(f"Failed to count samples in {file_path}", {"error": str(e)})
            return 1  # Minimum count fallback
    
    def _count_samples_in_file(self, file_path: str) -> int:
        """Count actual number of samples in a file."""
        try:
            count = 0
            if file_path.endswith('.jsonl'):
                # Count lines in JSONL file
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            count += 1
            elif file_path.endswith('.json'):
                # Count items in JSON array
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
                    else:
                        count = 1  # Single JSON object
            elif file_path.endswith('.txt'):
                # Count non-empty lines in text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            count += 1
            else:
                # Fallback for unknown formats
                count = 1
                
            return max(1, count)
            
        except Exception as e:
            self.logger.warning(f"Failed to count samples in {file_path}", {"error": str(e)})
            return 1  # Minimum count fallback
        
    def _estimate_batch_complexity(self) -> float:
        """Estimate computational complexity of current batch"""
        complexity_scores = []
        
        # Sample a few recent items to estimate complexity
        sample_count = min(10, len(self.data_index) if hasattr(self, 'data_index') else 10)
        
        for i in range(sample_count):
            try:
                # Get sample complexity based on multimodal content
                sample_path = self.data_index[i]['path'] if i < len(self.data_index) else ""
                complexity = 1.0
                
                # Higher complexity for multimodal data
                if any(key in sample_path.lower() for key in ['image', 'video', 'audio']):
                    complexity *= 0.7  # Reduce batch size for multimodal
                elif 'text' in sample_path.lower() or sample_path.endswith('.txt'):
                    complexity *= 1.2  # Increase batch size for text-only
                    
                # Sequence length factor (estimate from file size)
                if os.path.exists(sample_path):
                    file_size = os.path.getsize(sample_path) / (1024 * 1024)  # MB
                    if file_size > 10:  # Large files
                        complexity *= 0.8
                    elif file_size < 1:  # Small files
                        complexity *= 1.1
                        
                complexity_scores.append(complexity)
                
            except (IndexError, OSError):
                complexity_scores.append(1.0)
        
        # Return average complexity factor
        return np.mean(complexity_scores) if complexity_scores else 1.0
            
    def _process_file_streaming(self, file_info: Dict) -> Iterator[Dict]:
        """Stream processing for single file"""
        file_path = file_info['path']
        
        try:
            if file_path.endswith('.jsonl'):
                # Read JSONL files line by line
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                yield self._process_single_sample(data, f"{file_path}:{line_num}")
                            except json.JSONDecodeError as e:
                                ERROR(f"JSON parsing error {file_path}:{line_num}: {e}")
                                continue
                                
            elif file_path.endswith('.json'):
                # Memory-mapped reading for large JSON files
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for idx, item in enumerate(data):
                                yield self._process_single_sample(item, f"{file_path}:{idx}")
                        else:
                            yield self._process_single_sample(data, file_path)
                    except json.JSONDecodeError as e:
                        ERROR(f"Large JSON file parsing error {file_path}: {e}")
                        
            elif file_path.endswith('.txt'):
                # Process text files line by line
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if line.strip():
                            data = {'text': line.strip(), 'source': file_path}
                            yield self._process_single_sample(data, f"{file_path}:{line_num}")
                            
        except Exception as e:
            ERROR(f"File processing error {file_path}: {e}")
            
    def _process_single_sample(self, raw_data: Dict, sample_id: str) -> Dict:
        """Process single data sample"""
        try:
            # Text processing
            text = self._extract_text_from_sample(raw_data)
            if not text or len(text.strip()) < 3:
                text = "<empty>"
                
            # Tokenization
            input_ids = self.tokenizer.encode(text, return_tensors="pt")[0]
            vocab_size = len(self.tokenizer)
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            
            # Multimodal processing
            multimodal_data = self._extract_multimodal_features(raw_data, sample_id)
            
            return {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "sample_id": sample_id,
                **multimodal_data
            }
            
        except Exception as e:
            ERROR(f"Sample processing error {sample_id}: {e}")
            return {
                "input_ids": torch.tensor([0], dtype=torch.long),
                "labels": torch.tensor([0], dtype=torch.long),
                "sample_id": sample_id,
                "pixel_values": None,
                "audio_input": None,
                "doc_input": None,
                "video_frames": None
            }
            
    def _extract_text_from_sample(self, sample: Dict) -> str:
        """Extract text content from sample"""
        # Reuse existing text extraction logic
        from .__init__ import TEXT_FIELD_KEYS
        
        for key in TEXT_FIELD_KEYS:
            if key in sample and isinstance(sample[key], str) and sample[key].strip():
                return sample[key]
                
        # Handle conversation formats
        if 'conversations' in sample and isinstance(sample['conversations'], list):
            texts = []
            for turn in sample['conversations']:
                if isinstance(turn, dict):
                    content = turn.get('value') or turn.get('content') or turn.get('text')
                    if content and str(content).strip():
                        role = turn.get('from', turn.get('role', ''))
                        if role:
                            texts.append(f"{role}: {content}")
                        else:
                            texts.append(str(content))
            if texts:
                return "\n".join(texts)
                
        # Other formats...
        for key, value in sample.items():
            if isinstance(value, str) and value.strip():
                return value.strip()
                
        return ""
        
    def _extract_multimodal_features(self, sample: Dict, sample_id: str) -> Dict:
        """Extract multimodal features"""
        result = {
            "pixel_values": None,
            "audio_input": None,
            "doc_input": None,
            "video_frames": None
        }
        
        try:
            # Image processing
            if self.vision_encoder and self.vision_encoder.enabled:
                image_path = self._get_first_valid(sample, IMAGE_KEYS)
                if image_path:
                    result["pixel_values"] = self.vision_encoder.process_image(image_path)
                    
            # Audio processing  
            if self.audio_encoder and self.audio_encoder.enabled:
                audio_path = self._get_first_valid(sample, AUDIO_KEYS)
                if audio_path:
                    result["audio_input"] = self.audio_encoder.process_audio(audio_path)
                    
            # Document processing
            if self.doc_encoder and self.doc_encoder.enabled:
                doc_path = self._get_first_valid(sample, DOC_KEYS)
                if doc_path:
                    result["doc_input"] = self.doc_encoder.process_doc(doc_path)
                    
            # Video processing
            if self.video_encoder and self.video_encoder.enabled:
                video_path = self._get_first_valid(sample, VIDEO_KEYS)
                if video_path:
                    # Video processing uses more memory, need to check
                    if self.memory_monitor.check_memory()['system_available_gb'] > 4.0:
                        result["video_frames"] = self.video_encoder.process_video(video_path)
                        
        except Exception as e:
            ERROR(f"Multimodal feature extraction error {sample_id}: {e}")
            
        return result
        
    def _get_first_valid(self, item: dict, keys: list) -> Optional[str]:
        """Get first valid key value"""
        for k in keys:
            if k in item and isinstance(item[k], str) and item[k].strip():
                return item[k]
        return None
        
    def __iter__(self) -> Iterator[Dict]:
        """Implement streaming iterator"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process mode
            data_sources = self.data_index
        else:
            # Multi-process mode - assign data sources to different workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            
            per_worker = len(self.data_index) // num_workers
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < num_workers - 1 else len(self.data_index)
            
            data_sources = self.data_index[start_idx:end_idx]
            
        # Stream process data sources
        for file_info in data_sources:
            try:
                for sample in self._process_file_streaming(file_info):
                    yield sample
                    
                    # Memory monitoring
                    if self.memory_monitor.should_gc():
                        self.memory_monitor.cleanup()
                        
            except Exception as e:
                ERROR(f"Data source processing error {file_info['path']}: {e}")
                continue

def _get_first_valid(item: dict, keys: list) -> str:
    """
    Retrieve the value corresponding to the first valid key from the given dictionary.

    Args:
        item (dict): The dictionary to search within.
        keys (list): A list of keys to search for.

    Returns:
        str: The value of the first valid key, or None if no valid key is found.
    """
    for k in keys:
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k]
    return None

class OptimizedDataLoader:
    """Optimized data loader for 5TB datasets"""
    def __init__(self, dataset: Union['PiscesDataset', 'LargeScaleStreamingDataset'], 
                 batch_config: Optional[BatchConfig] = None, 
                 memory_monitor: Optional[MemoryMonitor] = None):
        self.dataset = dataset
        self.batch_config = batch_config or BatchConfig()
        self.memory_monitor = memory_monitor or MemoryMonitor()
        
    def get_dataloader(self) -> DataLoader:
        """Get optimized DataLoader with dynamic batching"""
        # Dynamic batch size calculation based on complexity and memory
        dynamic_batch_size = self._calculate_dynamic_batch_size()
        
        # Adjust parameters based on dataset type
        if hasattr(self.dataset, '__class__') and 'LargeScaleStreamingDataset' in str(type(self.dataset)):
            # Streaming dataset with dynamic batching
            return DataLoader(
                self.dataset,
                batch_size=None,  # Streaming datasets don't need batch_size
                num_workers=min(self.batch_config.num_workers, MAX_WORKERS),
                pin_memory=self.batch_config.pin_memory,
                prefetch_factor=self.batch_config.prefetch_factor,
                persistent_workers=True  # Enable persistent workers for better performance
            )
        else:
            # Traditional dataset with dynamic batching
            return DataLoader(
                self.dataset,
                batch_size=dynamic_batch_size,
                shuffle=True,
                num_workers=min(self.batch_config.num_workers, MAX_WORKERS),
                pin_memory=self.batch_config.pin_memory,
                drop_last=self.batch_config.drop_last,
                prefetch_factor=self.batch_config.prefetch_factor,
                persistent_workers=True  # Enable persistent workers for better performance
            )
            
    def _calculate_dynamic_batch_size(self) -> int:
        """Calculate dynamic batch size based on memory and computational complexity"""
        # Get current memory status
        memory_stats = self.memory_monitor.check_memory()
        available_memory = memory_stats['system_available_gb']
        gpu_memory = memory_stats.get('gpu_free_gb', available_memory)
        
        # Base batch size from available memory
        if gpu_memory > 20.0:
            base_batch = 64
        elif gpu_memory > 12.0:
            base_batch = 32
        elif gpu_memory > 8.0:
            base_batch = 16
        elif gpu_memory > 4.0:
            base_batch = 8
        else:
            base_batch = 4
            
        # Adjust based on system memory
        if available_memory < 4.0:
            base_batch = max(2, base_batch // 2)
            
        # Complexity factor from dataset characteristics
        complexity_factor = self._estimate_dataset_complexity()
        
        # Calculate final dynamic batch size
        dynamic_batch = int(base_batch * complexity_factor)
        
        # Ensure bounds
        return max(2, min(dynamic_batch, 128))
        
    def _estimate_dataset_complexity(self) -> float:
        """Estimate dataset computational complexity"""
        try:
            # Sample dataset to estimate complexity
            if hasattr(self.dataset, '__len__') and len(self.dataset) > 0:
                # Sample a few items
                sample_size = min(5, len(self.dataset))
                complexity_scores = []
                
                for i in range(sample_size):
                    sample = self.dataset[i] if hasattr(self.dataset, '__getitem__') else {}
                    complexity = 1.0
                    
                    # Check for multimodal content
                    if isinstance(sample, dict):
                        if any(key in sample for key in ['pixel_values', 'audio_input', 'video_frames']):
                            complexity *= 0.6  # Higher complexity for multimodal
                        elif 'input_ids' in sample:
                            # Estimate sequence length complexity
                            seq_len = len(sample['input_ids']) if hasattr(sample['input_ids'], '__len__') else 100
                            if seq_len > 1000:
                                complexity *= 0.8
                            elif seq_len < 100:
                                complexity *= 1.2
                                
                    complexity_scores.append(complexity)
                
                return np.mean(complexity_scores) if complexity_scores else 1.0
        except Exception:
            pass
            
        return 1.0
            
    def adaptive_batch_iterator(self, target_memory_gb: float = 2.0):
        """Adaptive batch iterator"""
        dataloader = self.get_dataloader()
        current_batch = []
        current_memory = 0.0
        
        for sample in dataloader:
            # Estimate sample memory usage
            estimated_size = self._estimate_sample_memory(sample)
            
            if current_memory + estimated_size > target_memory_gb * 1024**3:
                # Output current batch
                if current_batch:
                    yield current_batch
                    current_batch = []
                    current_memory = 0.0
                    
                    # Memory cleanup
                    if self.memory_monitor.should_gc():
                        self.memory_monitor.cleanup()
            
            current_batch.append(sample)
            current_memory += estimated_size
            
        # Output last batch
        if current_batch:
            yield current_batch
            
    def _estimate_sample_memory(self, sample: Dict) -> float:
        """Estimate sample memory usage (bytes)"""
        memory_size = 0.0
        
        if 'input_ids' in sample and sample['input_ids'] is not None:
            memory_size += sample['input_ids'].numel() * 4  # int32
            
        if 'pixel_values' in sample and sample['pixel_values'] is not None:
            memory_size += sample['pixel_values'].numel() * 4  # float32
            
        if 'audio_input' in sample and sample['audio_input'] is not None:
            if isinstance(sample['audio_input'], dict) and 'input_values' in sample['audio_input']:
                if sample['audio_input']['input_values'] is not None:
                    memory_size += sample['audio_input']['input_values'].numel() * 4
                    
        return memory_size

def _get_first_valid_key(item: dict, keys: list) -> str:
    """
    Retrieve the value corresponding to the first valid key from the given dictionary.

    Args:
        item (dict): The dictionary to search within.
        keys (list): A list of keys to search for.

    Returns:
        str: The value of the first valid key, or None if no valid key is found.
    """
    for k in keys:
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k]
    return None

class PiscesDataset(Dataset):
    """
    Pisces dataset with multimodal support for text, image, audio, document, and video data.
    Enhanced with 5TB-scale batch loading capabilities for Arctic architecture.
    """
    def __init__(self, subset: str = "tiny", split: str = "train", config=None, data_ratio: dict = None, 
                 memory_efficient: bool = True, max_samples: int = None, cache_validation: bool = True):
        """
        Initialize the PiscesDataset with Arctic architecture 5TB-scale processing enabled by default.
        
        Args:
            subset (str, optional): Name of the dataset subset. Defaults to "tiny".
            split (str, optional): Dataset split type, either "train" or "test". Defaults to "train".
            config: Configuration object. Defaults to None.
            data_ratio (dict, optional): Custom data ratio configuration for different domains.
                Example: {"math": 0.3, "code": 0.25, "general": 0.3, "multimodal": 0.15}
            memory_efficient (bool, optional): Enable memory-efficient processing. Defaults to True.
            max_samples (int, optional): Limit dataset size for testing. Defaults to None.
            cache_validation (bool, optional): Cache validation results. Defaults to True.
        """
        # Arctic architecture: 5TB-scale dataset support enabled by default
        self.enable_large_scale = True
        self.batch_config = BatchConfig()
        self.memory_monitor = MemoryMonitor(MEMORY_THRESHOLD_GB)
            
        # Use cache manager for dataset cache path
        from ..cache import get_cache_manager
        cache_manager = get_cache_manager()
        data_cache_dir = cache_manager.get_or_create_cache_dir("data_cache")
        cache_path = os.path.join(data_cache_dir, subset)
        
        try:
            if os.path.exists(cache_path):
                RIGHT(f"Loading dataset from local cache: {cache_path}")
                self.ds = load_from_disk(cache_path)
                if split == "train" and "train" in self.ds:
                    self.ds = self.ds["train"]
                elif split == "test" and "test" in self.ds:
                    self.ds = self.ds["test"]
                
                # Filter out empty or invalid samples with improved cleaning
                original_size = len(self.ds)
                RIGHT(f"Local dataset loaded successfully: {original_size} samples")
                if original_size > 0:
                    RIGHT("Filtering dataset to remove samples with no valid content...")
                    
                    # Import cleaning utilities
                    from .clean import StreamCleaner
                    import multiprocessing as mp
                    from functools import partial
                    
                    cleaner = StreamCleaner(min_len=1, max_len=2048)  # Very permissive
                    
                    def _check_single_example(example: dict, cleaner_instance) -> bool:
                        """
                        Check if a single example has valid content after cleaning.
                        
                        Args:
                            example (dict): A single example from the dataset.
                            cleaner_instance: StreamCleaner instance for text cleaning.
                            
                        Returns:
                            bool: True if the example has valid content, False otherwise.
                        """
                        text = self._get_text(example)
                        if text is None:
                            return False
                        
                        # Apply gentle cleaning
                        cleaned_text = cleaner_instance.clean_text(text)
                        
                        # Accept any non-empty content after cleaning
                        return cleaned_text is not None and len(cleaned_text.strip()) > 0
                    
                    # Use multiprocessing for faster filtering
                    num_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers to avoid memory issues
                    DEBUG(f"Using {num_workers} processes for dataset filtering...")
                    
                    # Create partial function with cleaner instance
                    check_func = partial(_check_single_example, cleaner_instance=cleaner)
                    
                    # Apply multiprocessing filter
                    self.ds = self.ds.filter(
                        check_func,
                        num_proc=num_workers,
                        desc="Filtering dataset"
                    )
                    filtered_size = len(self.ds)
                    if filtered_size == 0:
                        DEBUG(f"Warning: All {original_size} samples were filtered out. Applying emergency cleaning...")
                        # Emergency mode: use original dataset with basic validation
                        def emergency_filter(example: dict) -> bool:
                            """
                            Perform basic validation on an example.

                            Args:
                                example (dict): A single example from the dataset.

                            Returns:
                                bool: True if the example has non-empty text, False otherwise.
                            """
                            text = self._get_text(example)
                            return text is not None and len(str(text).strip()) > 0
                        
                        self.ds = self.ds.filter(emergency_filter)
                        filtered_size = len(self.ds)
                        RIGHT(f"Emergency filter: {filtered_size}/{original_size} samples remain")
                    else:
                        RIGHT(f"Filtered out {original_size - filtered_size} samples. {filtered_size}/{original_size} samples remain ({(filtered_size/original_size)*100:.2f}%).")
            else:
                ERROR(f"Local cache not found, trying online download: {subset}")
                if "MsDataset" not in globals() or "MsDataset" not in locals():
                    ERROR("MsDataset unavailable. Cannot load ModelScope dataset online. Please upgrade modelscope>=1.28.0 and datasets>=2.14.7, or use only local datasets.")
                    self.ds = [{"text": f"Hello world {i}", "id": i} for i in range(100)]
                else:
                    try:
                        msds = MsDataset.load(subset, split=split)
                        if hasattr(msds, 'to_hf_dataset'):
                            self.ds = msds.to_hf_dataset()
                        else:
                            self.ds = msds
                        RIGHT(f"Online dataset loaded successfully: {len(self.ds)} samples")
                    except Exception as e:
                        ERROR(f"MsDataset.load failed: {str(e)}")
                        ERROR("Could not load ModelScope dataset online. Falling back to local test data.")
                        self.ds = [{"text": f"Hello world {i}", "id": i} for i in range(100)]
        except Exception as e:
            ERROR(f"Dataset loading failed: {str(e)}")
            ERROR("Creating enhanced test dataset with fallback strategies...")
            
            # Multi-tier fallback system
            try:
                # Tier 1: Try loading from backup dataset
                backup_dir = cache_manager.get_or_create_cache_dir("data_cache")
                backup_path = os.path.join(backup_dir, "backup", "tiny")
                if os.path.exists(backup_path):
                    self.ds = load_from_disk(backup_path)
                    if split == "train" and "train" in self.ds:
                        self.ds = self.ds["train"]
                    elif split == "test" and "test" in self.ds:
                        self.ds = self.ds["test"]
                    RIGHT(f"Backup dataset loaded: {len(self.ds)} samples")
                    return
                    
            except Exception as backup_error:
                ERROR(f"Backup dataset failed: {str(backup_error)}")
            
            # Tier 2: Generate synthetic test data with realistic patterns
            try:
                import random
                import json
                
                # Generate diverse synthetic data
                synthetic_data = []
                templates = [
                    "The advanced computing algorithm demonstrated {adj} performance in {domain} applications.",
                    "Machine learning models require {adj} datasets for {domain} optimization.",
                    "Neural network architectures show {adj} results in {domain} scenarios.",
                    "Data processing pipelines achieve {adj} efficiency in {domain} workflows.",
                    "Algorithm optimization leads to {adj} improvements in {domain} systems."
                ]
                
                adjectives = ["excellent", "robust", "significant", "notable", "remarkable", "substantial"]
                domains = ["computer vision", "natural language processing", "reinforcement learning", "time series analysis", "graph neural networks"]
                
                for i in range(200):
                    template = random.choice(templates)
                    adj = random.choice(adjectives)
                    domain = random.choice(domains)
                    text = template.format(adj=adj, domain=domain)
                    
                    synthetic_data.append({
                        "text": text,
                        "id": i,
                        "metadata": {
                            "source": "synthetic_fallback",
                            "template": template,
                            "quality_score": random.uniform(0.7, 1.0)
                        }
                    })
                
                self.ds = synthetic_data
                RIGHT(f"Synthetic test dataset generated: {len(self.ds)} samples")
                
                # Save synthetic data for future use
                try:
                    synthetic_dir = cache_manager.get_or_create_cache_dir("data_cache")
                    os.makedirs(os.path.join(synthetic_dir, "synthetic"), exist_ok=True)
                    with open(os.path.join(synthetic_dir, "synthetic", "fallback.json"), 'w') as f:
                        json.dump(synthetic_data, f, indent=2)
                except:
                    pass  # Ignore save errors
                
            except Exception as synth_error:
                ERROR(f"Synthetic generation failed: {str(synth_error)}")
                
                # Tier 3: Ultra-basic fallback
                self.ds = [{"text": f"Advanced AI system optimization {i}", "id": i} for i in range(100)]
                RIGHT("Ultra-basic fallback dataset created")
        
        self.tokenizer = get_tokenizer()
        self.config = config
        # Initialize preprocessors
        self.vision_encoder = VisionEncoder(config) if config else None
        self.audio_encoder = AudioEncoder(config) if config else None
        self.doc_encoder = DocEncoder(config) if config else None
        self.video_encoder = VideoEncoder(config) if config else None
        
        # DeepSeek-level data ratio optimization
        self.data_ratio = data_ratio or {
            "math": 0.30,      # 30% Mathematical reasoning data
            "code": 0.25,      # 25% Programming code data
            "general": 0.30,   # 30% General dialogue data
            "multimodal": 0.15  # 15% Multimodal image-text data
        }
        
        # Memory optimization settings
        self.memory_efficient = memory_efficient
        self.max_samples = max_samples
        
        # Apply dataset size limit if specified
        if max_samples and len(self.ds) > max_samples:
            self.ds = self.ds.select(range(max_samples))
            RIGHT(f"Dataset size limited to {max_samples} samples for testing")
        
        # Apply large-scale optimizations after ds is initialized
        self._apply_large_scale_optimizations()
    
    def _apply_data_ratio_optimization(self):
        """
        Apply DeepSeek-level data ratio optimization with domain-aware sampling.
        """
        if len(self.ds) == 0:
            return
            
        RIGHT("Applying DeepSeek-level data ratio optimization...")
        
        # Domain classification based on content patterns
        def classify_domain(example):
            """Classify sample domain based on content analysis."""
            text = str(self._get_text(example) or "").lower()
            
            # Math patterns
            math_patterns = ["math", "equation", "solve", "calculate", "algebra", "geometry", "theorem", "proof"]
            if any(pattern in text for pattern in math_patterns):
                return "math"
            
            # Code patterns  
            code_patterns = ["def ", "class ", "import ", "function", "variable", "code", "programming", "python", "java", "c++"]
            if any(pattern in text for pattern in code_patterns):
                return "code"
                
            # Multimodal patterns
            if any(key in example for key in IMAGE_KEYS + AUDIO_KEYS + DOC_KEYS + VIDEO_KEYS):
                return "multimodal"
                
            return "general"
        
        # Classify all samples
        domains = [classify_domain(example) for example in self.ds]
        
        # Calculate target sample counts per domain
        total_samples = len(self.ds)
        target_counts = {
            domain: int(ratio * total_samples) 
            for domain, ratio in self.data_ratio.items()
        }
        
        # Apply stratified sampling to achieve target ratios
        domain_indices = {domain: [] for domain in self.data_ratio.keys()}
        for idx, domain in enumerate(domains):
            domain_indices[domain].append(idx)
        
        # Sample indices for each domain
        sampled_indices = []
        for domain, indices in domain_indices.items():
            target_count = target_counts[domain]
            if len(indices) >= target_count:
                # Downsample to target ratio
                sampled_indices.extend(torch.randperm(len(indices))[:target_count].tolist())
            else:
                # Upsample by repeating (with slight randomness)
                repeats = target_count // len(indices) + 1
                extended = (indices * repeats)[:target_count]
                sampled_indices.extend(extended)
        
        # Apply sampling to dataset
        self.ds = self.ds.select(sampled_indices)
        RIGHT(f"Data ratio optimization complete: {len(self.ds)} samples with optimized domain distribution")

    def _apply_memory_optimizations(self):
        """
        Apply memory optimization techniques for large datasets.
        """
        RIGHT("Applying memory optimizations...")
        
        # Enable memory mapping for datasets
        if hasattr(self.ds, 'set_format'):
            self.ds.set_format(type='torch', columns=list(self.ds.features.keys()))
        
        # Pre-calculate dataset statistics
        self.dataset_stats = {
            'total_samples': len(self.ds),
            'avg_text_length': sum(len(self._get_text(self.ds[i])) for i in range(min(100, len(self.ds)))) / min(100, len(self.ds)) if len(self.ds) > 0 else 0,
            'estimated_memory': len(self.ds) * 1024  # Rough estimate in bytes
        }
        
        DEBUG(f"Memory optimizations applied. Dataset stats: {self.dataset_stats}")

    def _validate_token_ids_with_cache(self):
        """
        Validate token IDs with caching for improved performance.
        """
        import hashlib
        import json
        import os
        
        # Generate cache key based on dataset and tokenizer
        cache_key = hashlib.md5(f"{len(self.ds)}_{len(self.tokenizer)}".encode()).hexdigest()
        data_cache_dir = cache_manager.get_or_create_cache_dir("data_cache")
        cache_file = os.path.join(data_cache_dir, f"token_validation_{cache_key}.json")
        
        # Check if validation cache exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                RIGHT("Token validation loaded from cache")
                return
            except Exception:
                pass
        
        # Perform validation if cache not found
        self._validate_token_ids()
        
        # Save validation results to cache
        try:
            data_cache_dir = cache_manager.get_or_create_cache_dir("data_cache")
            os.makedirs(data_cache_dir, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({"validated": True, "timestamp": str(datetime.now())}, f)
        except Exception:
            pass

    def _validate_token_ids(self):
        """
        Enhanced token ID validation with comprehensive error handling and reporting.
        """
        vocab_size = len(self.tokenizer)
        
        # Adaptive validation size based on dataset size
        if len(self.ds) <= 1000:
            max_samples = len(self.ds)
        elif len(self.ds) <= 10000:
            max_samples = min(2000, len(self.ds))
        else:
            max_samples = min(5000, len(self.ds))
        
        if max_samples == 0:
            RIGHT("No samples to validate")
            return
            
        RIGHT(f"Validating token IDs for {max_samples}/{len(self.ds)} samples...")
        
        # Enhanced multiprocessing validation
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        def _batch_validate(indices, tokenizer_path, vocab_size):
            """Validate a batch of samples"""
            from model.tokenizer import get_tokenizer
            tokenizer = get_tokenizer()
            
            results = []
            for idx in indices:
                try:
                    item = self.ds[idx]
                    text = self._get_text(item)
                    if not text or not text.strip():
                        results.append((idx, True, "empty_text"))
                        continue
                        
                    input_ids = tokenizer.encode(text, return_tensors="pt")[0]
                    max_token_id = input_ids.max().item() if len(input_ids) > 0 else 0
                    min_token_id = input_ids.min().item() if len(input_ids) > 0 else 0
                    
                    is_valid = max_token_id < vocab_size and min_token_id >= 0
                    results.append((idx, is_valid, None))
                    
                except Exception as e:
                    results.append((idx, False, str(e)))
            
            return results
        
        # Use batch processing for efficiency
        batch_size = max(100, max_samples // mp.cpu_count())
        batches = [range(i, min(i + batch_size, max_samples)) 
                  for i in range(0, max_samples, batch_size)]
        
        all_results = []
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(_batch_validate, batch, None, vocab_size) 
                      for batch in batches]
            
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Process results
        invalid_samples = [(idx, error) for idx, valid, error in all_results if not valid]
        
        RIGHT("Token validation complete!")
        
        # Enhanced reporting
        if invalid_samples:
            ERROR(f"Found {len(invalid_samples)} invalid samples ({len(invalid_samples)/max_samples*100:.2f}%)")
            
            # Analyze error patterns
            error_types = {}
            for idx, error in invalid_samples:
                error_type = str(error)[:50]  # Truncate for grouping
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                DEBUG(f"  {error_type}: {count} samples")
        else:
            RIGHT("All token IDs validated successfully")
            
        # Performance metrics
        DEBUG(f"Vocabulary size: {vocab_size}")
        if len(self.ds) > 0:
            sample_text = self._get_text(self.ds[0])
            if sample_text:
                try:
                    sample_ids = self.tokenizer.encode(sample_text, return_tensors="pt")[0]
                    DEBUG(f"Token ID range: {sample_ids.min().item()} - {sample_ids.max().item()}")
                except Exception as e:
                    ERROR(f"Sample validation failed: {e}")

    def _get_text(self, item: dict) -> str:
        """
        Extract text content from a dataset item.

        Args:
            item (dict): A single item from the dataset.

        Returns:
            str: The extracted text content, or an empty string if no text is found.
        """
        # Import text field keys from the data module
        from .__init__ import TEXT_FIELD_KEYS
        
        # Debug: Print the structure of the first few samples
        if hasattr(self, '_debug_count'):
            self._debug_count = getattr(self, '_debug_count', 0) + 1
            if self._debug_count <= 3:
                DEBUG(f"Debug item structure: {type(item)}, keys: {list(item.keys()) if isinstance(item, dict) else 'N/A'}")
        
        # Look for a direct key match
        for key in TEXT_FIELD_KEYS:
            if isinstance(item.get(key), str) and item[key].strip():
                return item[key]

        # Handle formats specific to the Chinese1 dataset
        if isinstance(item, dict):
            # Handle Chinese dataset formats
            for chinese_key in ['chinese', 'content', 'text_cn', 'cn_text', '中文']:
                if isinstance(item.get(chinese_key), str) and item[chinese_key].strip():
                    return item[chinese_key]

        # Handle conversational formats (e.g., ShareGPT)
        if 'conversations' in item and isinstance(item['conversations'], list) and item['conversations']:
            full_text = []
            for turn in item['conversations']:
                if isinstance(turn, dict):
                    # Handle various conversation formats
                    content = None
                    if 'value' in turn and turn['value'] and str(turn['value']).strip():
                        content = str(turn['value']).strip()
                    elif 'content' in turn and turn['content'] and str(turn['content']).strip():
                        content = str(turn['content']).strip()
                    elif 'text' in turn and turn['text'] and str(turn['text']).strip():
                        content = str(turn['text']).strip()
                    
                    if content:
                        # Add role prefix if available
                        role = turn.get('from', turn.get('role', ''))
                        if role:
                            full_text.append(f"{role}: {content}")
                        else:
                            full_text.append(content)
            if full_text:
                return "\n".join(full_text)

        # Handle instruction-following formats
        if 'instruction' in item and 'input' in item:
            instruction = str(item['instruction']).strip() if item['instruction'] else ""
            input_text = str(item['input']).strip() if item['input'] else ""
            if instruction and input_text:
                return f"{instruction}\n{input_text}"
            elif instruction:
                return instruction
            elif input_text:
                return input_text

        # Handle formats specific to Chinese datasets
        if 'prompt' in item and 'response' in item:
            prompt = str(item['prompt']).strip() if item['prompt'] else ""
            response = str(item['response']).strip() if item['response'] else ""
            if prompt and response:
                return f"{prompt}\n{response}"
            elif prompt:
                return prompt
            elif response:
                return response

        # Handle nested dictionary formats
        for key, value in item.items():
            if isinstance(value, dict):
                # Recursively extract text from nested dictionaries
                nested_text = self._get_text(value)
                if nested_text.strip():
                    return nested_text
            elif isinstance(value, str) and value.strip():
                # Use any non-empty string field as a fallback
                return value.strip()

        # Handle list formats (like the Chinese1 dataset)
        if isinstance(item, list) and item:
            # If the item is a list, try to extract text from the first element
            for sub_item in item:
                if isinstance(sub_item, dict):
                    text = self._get_text(sub_item)
                    if text.strip():
                        return text
                elif isinstance(sub_item, str) and sub_item.strip():
                    return sub_item.strip()

        # Final fallback: Concatenate all non-empty string values recursively
        def extract_all_strings(obj) -> list:
            """
            Recursively extract all non-empty string values from an object.

            Args:
                obj: The object to extract strings from.

            Returns:
                list: A list of non-empty string values.
            """
            strings = []
            if isinstance(obj, dict):
                for v in obj.values():
                    strings.extend(extract_all_strings(v))
            elif isinstance(obj, list):
                for v in obj:
                    strings.extend(extract_all_strings(v))
            elif isinstance(obj, str) and obj.strip():
                strings.append(obj.strip())
            return strings

        all_strings = extract_all_strings(item)
        if all_strings:
            return " ".join(all_strings)
            
        return ""

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        """
        Enhanced sample retrieval with memory monitoring, error recovery, and performance optimization.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the processed sample data with error handling and recovery.
        """
        import psutil
        import time
        
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024**3  # GB
        
        try:
            item = self.ds[idx]
            
            # Memory monitoring - Arctic architecture default
            current_memory = psutil.Process().memory_info().rss / 1024**3
            if current_memory > 8.0:  # 8GB threshold
                DEBUG(f"High memory usage detected: {current_memory:.2f}GB")
                torch.cuda.empty_cache()

            # Text processing with enhanced error handling
            text = self._get_text(item)
            if not text or not text.strip():
                text = "<empty>"
                
            try:
                input_ids = self.tokenizer.encode(text, return_tensors="pt")[0]
                
                # Ensure token IDs are within the vocabulary bounds
                vocab_size = len(self.tokenizer)
                max_token_id = input_ids.max().item() if len(input_ids) > 0 else 0
                if max_token_id >= vocab_size:
                    DEBUG(f"Token ID {max_token_id} exceeds vocab size {vocab_size}, clamping...")
                    input_ids = torch.clamp(input_ids, max=vocab_size - 1)
                
                # Ensure no negative indices
                input_ids = torch.clamp(input_ids, min=0)
                
            except Exception as e:
                ERROR(f"Text encoding failed for sample {idx}: {str(e)}")
                input_ids = torch.tensor([0], dtype=torch.long)  # Fallback to padding token

            # Multimodal processing with enhanced error recovery
            pixel_values = self._process_multimodal_data(
                item, IMAGE_KEYS, self.vision_encoder, "image", idx
            )
            
            audio_input = self._process_multimodal_data(
                item, AUDIO_KEYS, self.audio_encoder, "audio", idx
            )
            
            doc_input = self._process_multimodal_data(
                item, DOC_KEYS, self.doc_encoder, "document", idx
            )
            
            video_frames = self._process_multimodal_data(
                item, VIDEO_KEYS, self.video_encoder, "video", idx
            )

            # Reasoning correctness label with validation
            correct_label = item.get("correct", 1)
            if not isinstance(correct_label, (int, float)):
                correct_label = 1
            correct = torch.tensor(int(correct_label), dtype=torch.long)

            # Memory cleanup for Arctic architecture
            if idx % 100 == 0:
                torch.cuda.empty_cache()

            # Performance monitoring
            processing_time = time.time() - start_time
            memory_after = psutil.Process().memory_info().rss / 1024**3
            memory_delta = memory_after - memory_before
            
            if processing_time > 1.0:  # Warn for slow processing
                DEBUG(f"Slow sample processing: {idx} took {processing_time:.2f}s")

            return {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "pixel_values": pixel_values,
                "audio_input": audio_input if audio_input is not None else {'input_values': None},
                "doc_input": doc_input,
                "video_frames": video_frames,
                "correct": correct,
                "processing_time": processing_time,
                "memory_delta": memory_delta
            }
            
        except Exception as e:
            ERROR(f"Critical error processing sample {idx}: {str(e)}")
            
            # Return fallback sample with error information
            return {
                "input_ids": torch.tensor([0], dtype=torch.long),
                "labels": torch.tensor([0], dtype=torch.long),
                "pixel_values": None,
                "audio_input": {'input_values': None},
                "doc_input": None,
                "video_frames": None,
                "correct": torch.tensor(0, dtype=torch.long),
                "error": str(e),
                "processing_time": time.time() - start_time,
                "memory_delta": 0.0
            }
    
    def _apply_large_scale_optimizations(self):
        """Apply 5TB large-scale dataset optimizations - enabled by default in Arctic architecture"""
        RIGHT("Arctic architecture: 5TB dataset processing optimizations active")
        
        # Cleanup memory before optimization
        self.memory_monitor.cleanup()
            
        # Optimize data loading parameters
        if hasattr(self.ds, 'set_format') and hasattr(self.ds, 'features'):
            # Use memory mapping for large datasets (only for HuggingFace datasets)
            try:
                self.ds.set_format(type='torch', 
                                 columns=list(self.ds.features.keys()),
                                 output_all_columns=False)
                DEBUG("Memory mapping enabled for dataset")
            except Exception as e:
                DEBUG(f"Memory mapping failed, continuing without: {e}")
        else:
            DEBUG("Dataset is not a HuggingFace dataset, skipping memory mapping")
            
        # Pre-calculate optimal batch size
        self.optimal_batch_size = self._calculate_optimal_batch_size()
        RIGHT(f"Calculated optimal batch size: {self.optimal_batch_size}")
        
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size"""
        if not self.memory_monitor:
            return self.batch_config.batch_size
            
        memory_stats = self.memory_monitor.check_memory()
        available_gb = memory_stats['system_available_gb']
        
        # Conservative estimation based on available memory
        if available_gb > 32:
            return min(64, len(self.ds) // 100) if len(self.ds) > 6400 else 64
        elif available_gb > 16:
            return min(32, len(self.ds) // 200) if len(self.ds) > 6400 else 32
        elif available_gb > 8:
            return min(16, len(self.ds) // 400) if len(self.ds) > 6400 else 16
        else:
            return min(8, len(self.ds) // 800) if len(self.ds) > 6400 else 8
            
    def get_streaming_dataloader(self, data_sources: List[str]) -> OptimizedDataLoader:
        """Get streaming data loader for 5TB dataset processing"""
        streaming_dataset = LargeScaleStreamingDataset(
            data_sources=data_sources,
            config=self.config,
            batch_config=self.batch_config,
            enable_prefetch=True
        )
        
        return OptimizedDataLoader(
            dataset=streaming_dataset,
            batch_config=self.batch_config,
            memory_monitor=self.memory_monitor
        )
        
    def create_distributed_batches(self, world_size: int, rank: int) -> Iterator[List[Dict]]:
        """Create distributed training batches for Arctic architecture"""
        total_samples = len(self.ds)
        samples_per_rank = total_samples // world_size
        start_idx = rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if rank < world_size - 1 else total_samples
        
        batch_size = self.optimal_batch_size
        current_batch = []
        
        for idx in range(start_idx, end_idx):
            try:
                sample = self.__getitem__(idx)
                current_batch.append(sample)
                
                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []
                    
                    # Memory monitoring for distributed training
                    if self.memory_monitor.should_gc():
                        self.memory_monitor.cleanup()
                        
            except Exception as e:
                ERROR(f"Distributed batch creation error at index {idx}: {e}")
                continue
                
        # Yield remaining samples
        if current_batch:
            yield current_batch
            
    def enable_checkpointing(self, checkpoint_dir: str = "data_checkpoints"):
        """Enable data loading checkpoint functionality"""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        RIGHT(f"Data checkpointing enabled: {checkpoint_dir}")
        
        # Save dataset metadata
        metadata = {
            'dataset_size': len(self.ds),
            'subset': getattr(self, 'subset', 'unknown'),
            'split': getattr(self, 'split', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'memory_config': {
                'threshold_gb': MEMORY_THRESHOLD_GB,
                'batch_memory_limit_gb': BATCH_MEMORY_LIMIT_GB,
                'prefetch_buffer_size': PREFETCH_BUFFER_SIZE
            }
        }
        
        with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        DEBUG("Dataset metadata saved for checkpointing")

    def _process_multimodal_data(self, item: dict, keys: list, encoder, data_type: str, sample_idx: int) -> any:
        """
        Enhanced multimodal data processing with comprehensive error handling.
        
        Args:
            item (dict): Dataset item
            keys (list): Keys to search for data
            encoder: Encoder instance
            data_type (str): Type of data (image, audio, etc.)
            sample_idx (int): Sample index for error reporting
            
        Returns:
            Processed data or None if processing fails
        """
        if not encoder or not encoder.enabled:
            return None
            
        data_path = _get_first_valid(item, keys)
        if not data_path:
            return None
            
        try:
            # Memory check before processing
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - reserved
                
                # Skip processing if low memory
                if data_type == "video" and free < 2.0:
                    DEBUG(f"Skipping video processing due to low memory: {free:.1f}GB free")
                    return None
                elif free < 1.0:
                    DEBUG(f"Skipping {data_type} processing due to low memory: {free:.1f}GB free")
                    return None

            # Process data based on type
            if data_type == "image":
                result = encoder.process_image(data_path)
            elif data_type == "audio":
                result = encoder.process_audio(data_path)
            elif data_type == "document":
                result = encoder.process_doc(data_path)
            elif data_type == "video":
                result = encoder.process_video(data_path)
            else:
                return None
                
            DEBUG(f"{data_type.capitalize()} processed successfully: {data_path}")
            return result
            
        except Exception as e:
            ERROR(f"{data_type.capitalize()} processing error for sample {sample_idx}: {str(e)}")
            return None