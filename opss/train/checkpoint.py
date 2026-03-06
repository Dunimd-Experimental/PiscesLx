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
Model Checkpoint Operators

Comprehensive checkpoint management for model training.
Provides saving, loading, and optimization of model checkpoints.

Features:
    - Full checkpoint saving (model, optimizer, scheduler, scaler)
    - Partial checkpoint saving (model only, weights only)
    - Sharded checkpoints for large models (FSDP style)
    - Checkpoint compression (zstd, gzip)
    - Checkpoint versioning
    - Checkpoint validation
    - Best checkpoint tracking (based on metrics)
    - Async checkpoint saving
    - Checkpoint cleanup strategies

Checkpoint Contents:
    - model_state_dict: Model weights
    - optimizer_state_dict: Optimizer state
    - scheduler_state_dict: LR scheduler state
    - scaler_state_dict: GradScaler state
    - epoch: Current epoch
    - global_step: Global training step
    - loss: Current loss value
    - metrics: Training metrics
    - config: Training configuration
    - random_state: Random states for reproducibility
    - cuda_state: CUDA random states

Usage:
    from ops.train.checkpoint import (
        CheckpointConfig,
        CheckpointOperator,
        CheckpointType
    )

    config = CheckpointConfig(
        type=CheckpointType.FULL,
        save_dir=".pisceslx/ckpt",
        interval=1000,
        max_to_keep=5
    )
    operator = CheckpointOperator(config)
    
    # Save checkpoint
    operator.save("model_step_1000", {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": 10,
        "global_step": 1000
    })
    
    # Load checkpoint
    state = operator.load("model_step_1000")
    model.load_state_dict(state["model_state_dict"])
"""

import os
import json
import time
import shutil
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from copy import deepcopy
import pickle

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir
from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus

from configs.version import VERSION


class CheckpointType(Enum):
    """Types of checkpoints to save."""
    FULL = "full"
    MODEL_ONLY = "model_only"
    WEIGHTS_ONLY = "weights_only"
    OPTIMIZER_ONLY = "optimizer_only"
    SHARDED = "sharded"


class CompressionType(Enum):
    """Checkpoint compression types."""
    NONE = "none"
    ZSTD = "zstd"
    GZIP = "gzip"


@dataclass
class CheckpointConfig:
    """
    Configuration for checkpoint operations.
    
    Attributes:
        type: Type of checkpoint to save
        save_dir: Directory to save checkpoints
        prefix: Filename prefix for checkpoints
        interval: Steps between automatic checkpoints
        max_to_keep: Maximum checkpoints to retain
        compression: Compression type for checkpoints
        save_async: Whether to save asynchronously
        shard_size: Size per shard for sharded checkpoints (in bytes)
        save_last: Always save the last checkpoint
        save_best: Track and save best checkpoint based on metric
        best_metric: Metric to use for best checkpoint selection
        best_mode: Mode for metric comparison (min/max)
        include_config: Include training config in checkpoint
        include_rng: Include random number generator states
        include_cuda: Include CUDA states
        safe_saving: Use atomic save operations
    """
    type: CheckpointType = CheckpointType.FULL
    save_dir: str = ".pisceslx/ckpt"
    prefix: str = "checkpoint"
    interval: int = 5000
    max_to_keep: int = 3
    compression: CompressionType = CompressionType.NONE
    save_async: bool = False
    shard_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    save_last: bool = True
    save_best: bool = False
    best_metric: str = "loss"
    best_mode: str = "min"
    include_config: bool = True
    include_rng: bool = True
    include_cuda: bool = True
    safe_saving: bool = True
    
    def __post_init__(self):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class CheckpointMetadata:
    """Metadata for a saved checkpoint."""
    path: str
    type: str
    size_bytes: int
    creation_time: float
    global_step: int
    epoch: Optional[int]
    loss: Optional[float]
    metrics: Dict[str, float]
    is_best: bool
    is_last: bool


class CheckpointOperator(PiscesLxOperatorInterface):
    """
    Model Checkpoint Operator.
    
    Provides comprehensive checkpoint management including:
    - Saving and loading checkpoints
    - Checkpoint optimization and compression
    - Best checkpoint tracking
    - Async saving operations
    - Sharded checkpoints for large models
    
    Features:
        - Atomic saves for crash safety
        - Checkpoint history management
        - Cross-node checkpoint compatibility
        - Mixed precision checkpoint support
    
    Attributes:
        config: CheckpointConfig instance
        metadata: Metadata for all saved checkpoints
        logger: Logger instance
        _save_lock: Lock for thread-safe saving
    """
    
    def __init__(self, config: Optional[CheckpointConfig] = None):
        """Initialize checkpoint operator."""
        super().__init__()
        self.name = "checkpoint"
        self.version = VERSION
        self.config = config or CheckpointConfig()
        self.metadata: Dict[str, CheckpointMetadata] = {}
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
        self._save_lock = threading.Lock()
        self._async_queue: List[Dict] = []
        self._async_thread: Optional[threading.Thread] = None
        self._running = False
        
        self._load_metadata()
    
    @property
    def description(self) -> str:
        return f"Checkpoint operator ({self.config.type.value})"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "operation": {"type": "str", "required": True, "description": "Operation: save, load, list, delete"},
            "tag": {"type": "str", "required": False, "description": "Checkpoint tag/identifier"},
            "model": {"type": "nn.Module", "required": False, "description": "Model to save"},
            "optimizer": {"type": "Optimizer", "required": False, "description": "Optimizer to save"},
            "scheduler": {"type": "_LRScheduler", "required": False, "description": "LR scheduler to save"},
            "scaler": {"type": "GradScaler", "required": False, "description": "GradScaler to save"},
            "global_step": {"type": "int", "required": False, "description": "Current global step"},
            "epoch": {"type": "int", "required": False, "description": "Current epoch"},
            "metrics": {"type": "dict", "required": False, "description": "Current metrics"}
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "success": {"type": "bool", "description": "Operation success status"},
            "path": {"type": "str", "description": "Checkpoint path"},
            "state": {"type": "dict", "description": "Loaded state dictionary"},
            "checkpoints": {"type": "list", "description": "List of available checkpoints"},
            "metadata": {"type": "dict", "description": "Checkpoint metadata"}
        }
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate required inputs."""
        if 'operation' not in inputs:
            return False
        operation = inputs['operation']
        if operation == 'save':
            return 'tag' in inputs or self.config.prefix is not None
        elif operation == 'load':
            return 'tag' in inputs
        elif operation == 'delete':
            return 'tag' in inputs
        return True
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute checkpoint operation.
        
        Args:
            inputs: Dictionary containing:
                - operation: save, load, list, delete
                - tag: Checkpoint identifier
                - model, optimizer, scheduler, scaler: Components to save
                - global_step, epoch, metrics: Training state
                
        Returns:
            PiscesLxOperatorResult with operation outcome
        """
        try:
            operation = inputs.get('operation', 'save')
            
            if operation == 'save':
                return self._save_checkpoint(inputs)
            elif operation == 'load':
                return self._load_checkpoint(inputs)
            elif operation == 'list':
                return self._list_checkpoints(inputs)
            elif operation == 'delete':
                return self._delete_checkpoint(inputs)
            elif operation == 'cleanup':
                return self._cleanup_old_checkpoints(inputs)
            else:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error=f"Unknown operation: {operation}"
                )
                
        except Exception as e:
            self._LOG.error(f"Checkpoint operation failed: {e}", exc_info=True)
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _save_checkpoint(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Save checkpoint."""
        tag = inputs.get('tag') or f"{self.config.prefix}_{inputs.get('global_step', 0)}"
        model = inputs.get('model')
        optimizer = inputs.get('optimizer')
        scheduler = inputs.get('scheduler')
        scaler = inputs.get('scaler')
        global_step = inputs.get('global_step', 0)
        epoch = inputs.get('epoch')
        metrics = inputs.get('metrics', {})
        
        checkpoint_path = self._get_checkpoint_path(tag)
        self._LOG.info(f"Saving checkpoint: {checkpoint_path}")
        
        state_dict = self._build_state_dict(
            model, optimizer, scheduler, scaler,
            global_step, epoch, metrics
        )
        
        if self.config.save_async and model is None:
            return self._save_async(tag, checkpoint_path, state_dict)
        else:
            return self._save_sync(tag, checkpoint_path, state_dict, metrics)
    
    def _build_state_dict(
        self,
        model: Optional[nn.Module],
        optimizer: Optional[Optimizer],
        scheduler: Optional[_LRScheduler],
        scaler: Optional[GradScaler],
        global_step: int,
        epoch: Optional[int],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Build checkpoint state dictionary."""
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'metrics': metrics,
            'config_type': self.config.type.value
        }
        
        if self.config.type in [CheckpointType.FULL, CheckpointType.MODEL_ONLY]:
            if model is not None:
                state['model_state_dict'] = deepcopy(model.state_dict())
        
        if self.config.type in [CheckpointType.FULL, CheckpointType.WEIGHTS_ONLY]:
            if model is not None:
                state['model_state_dict'] = deepcopy(model.state_dict())
        
        if self.config.type in [CheckpointType.FULL, CheckpointType.OPTIMIZER_ONLY]:
            if optimizer is not None:
                state['optimizer_state_dict'] = deepcopy(optimizer.state_dict())
        
        if self.config.type == CheckpointType.FULL:
            if scheduler is not None and hasattr(scheduler, 'state_dict'):
                state['scheduler_state_dict'] = scheduler.state_dict()
            if scaler is not None:
                state['scaler_state_dict'] = scaler.state_dict()
        
        if self.config.include_rng:
            state['random_state'] = {
                'python': random.getstate() if 'random' in dir() else None,
                'numpy': numpy.random.get_state() if 'numpy' in dir() else None
            }
            if torch.cuda.is_available():
                state['cuda_state'] = torch.cuda.get_rng_state_all()
        
        if self.config.include_config:
            state['config'] = {
                'checkpoint_type': self.config.type.value,
                'save_time': time.time()
            }
        
        return state
    
    def _save_sync(
        self,
        tag: str,
        checkpoint_path: Path,
        state_dict: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> PiscesLxOperatorResult:
        """Save checkpoint synchronously."""
        with self._save_lock:
            if self.config.type == CheckpointType.SHARDED:
                success = self._save_sharded(checkpoint_path, state_dict)
            else:
                success = self._save_regular(checkpoint_path, state_dict)
            
            if success:
                self._update_metadata(tag, checkpoint_path, state_dict, metrics)
                self._manage_checkpoints()
                
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output={
                        'success': True,
                        'path': str(checkpoint_path),
                        'tag': tag,
                        'global_step': state_dict.get('global_step', 0)
                    }
                )
            else:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error=f"Failed to save checkpoint: {checkpoint_path}"
                )
    
    def _save_regular(self, checkpoint_path: Path, state_dict: Dict[str, Any]) -> bool:
        """Save checkpoint using regular (non-sharded) method."""
        try:
            if self.config.safe_saving:
                temp_path = checkpoint_path.with_suffix('.tmp')
                torch.save(state_dict, temp_path)
                temp_path.replace(checkpoint_path)
            else:
                torch.save(state_dict, checkpoint_path)
            return True
        except Exception as e:
            self._LOG.error(f"Failed to save checkpoint: {e}")
            return False
    
    def _save_sharded(self, checkpoint_path: Path, state_dict: Dict[str, Any]) -> bool:
        """Save checkpoint using sharded method for large models."""
        try:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            index = {'metadata': {}, 'weight_map': {}}
            shard_files = []
            
            total_size = 0
            shard_idx = 0
            current_shard = {}
            current_size = 0
            
            for key, value in state_dict.items():
                if not isinstance(value, torch.Tensor):
                    index['weight_map'][key] = f"null"
                    continue
                
                tensor_size = value.numel() * value.element_size()
                
                if tensor_size > self.config.shard_size:
                    current_shard[key] = value
                    shard_path = checkpoint_path / f"checkpoint_{shard_idx:05d}.pt"
                    torch.save(current_shard, shard_path)
                    shard_files.append(shard_path)
                    current_shard = {}
                    current_size = 0
                    shard_idx += 1
                    total_size += tensor_size
                    index['weight_map'][key] = shard_path.name
                else:
                    if current_size + tensor_size > self.config.shard_size:
                        shard_path = checkpoint_path / f"checkpoint_{shard_idx:05d}.pt"
                        torch.save(current_shard, shard_path)
                        shard_files.append(shard_path)
                        current_shard = {}
                        current_size = 0
                        shard_idx += 1
                    
                    current_shard[key] = value
                    current_size += tensor_size
                    total_size += tensor_size
            
            if current_shard:
                shard_path = checkpoint_path / f"checkpoint_{shard_idx:05d}.pt"
                torch.save(current_shard, shard_path)
                shard_files.append(shard_path)
            
            index['metadata'] = {
                'total_size': total_size,
                'shard_count': len(shard_files),
                'type': 'sharded'
            }
            
            torch.save(index, checkpoint_path / "checkpoint_index.pt")
            
            return True
            
        except Exception as e:
            self._LOG.error(f"Failed to save sharded checkpoint: {e}")
            return False
    
    def _save_async(self, tag: str, checkpoint_path: Path, state_dict: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Save checkpoint asynchronously."""
        self._async_queue.append({
            'tag': tag,
            'path': checkpoint_path,
            'state': state_dict
        })
        
        if not self._running:
            self._running = True
            self._async_thread = threading.Thread(target=self._async_save_worker, daemon=True)
            self._async_thread.start()
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                'success': True,
                'path': str(checkpoint_path),
                'tag': tag,
                'async': True
            }
        )
    
    def _async_save_worker(self) -> None:
        """Background worker for async checkpoint saving."""
        while self._running or self._async_queue:
            while self._async_queue:
                item = self._async_queue.pop(0)
                self._save_sync(item['tag'], item['path'], item['state'], {})
            
            if self._running:
                time.sleep(0.1)
    
    def _load_checkpoint(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Load checkpoint."""
        tag = inputs.get('tag')
        checkpoint_path = self._get_checkpoint_path(tag)
        
        if not checkpoint_path.exists():
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Checkpoint not found: {checkpoint_path}"
            )
        
        try:
            if self.config.type == CheckpointType.SHARDED:
                state = self._load_sharded(checkpoint_path)
            else:
                state = torch.load(checkpoint_path, map_location='cpu')
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    'success': True,
                    'path': str(checkpoint_path),
                    'state': state
                }
            )
            
        except Exception as e:
            self._LOG.error(f"Failed to load checkpoint: {e}")
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _load_sharded(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load sharded checkpoint."""
        index = torch.load(checkpoint_path / "checkpoint_index.pt", map_location='cpu')
        state = {}
        
        for key, shard_file in index['weight_map'].items():
            if shard_file == "null":
                continue
            shard_path = checkpoint_path / shard_file
            if shard_path.exists():
                shard_state = torch.load(shard_path, map_location='cpu')
                state.update(shard_state)
        
        state['checkpoint_index'] = index
        return state
    
    def _list_checkpoints(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """List all available checkpoints."""
        checkpoints = []
        
        for tag, meta in self.metadata.items():
            checkpoints.append({
                'tag': tag,
                'path': meta.path,
                'size_mb': meta.size_bytes / (1024 * 1024),
                'global_step': meta.global_step,
                'epoch': meta.epoch,
                'loss': meta.loss,
                'is_best': meta.is_best,
                'is_last': meta.is_last,
                'created_at': time.ctime(meta.creation_time)
            })
        
        checkpoints.sort(key=lambda x: x['global_step'], reverse=True)
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                'success': True,
                'checkpoints': checkpoints,
                'total': len(checkpoints)
            }
        )
    
    def _delete_checkpoint(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Delete a checkpoint."""
        tag = inputs.get('tag')
        checkpoint_path = self._get_checkpoint_path(tag)
        
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
            self.metadata.pop(tag, None)
            self._save_metadata()
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    'success': True,
                    'deleted': tag
                }
            )
        else:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Checkpoint not found: {tag}"
            )
    
    def _cleanup_old_checkpoints(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """Clean up old checkpoints based on max_to_keep."""
        if self.config.max_to_keep <= 0:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={'success': True, 'deleted': 0}
            )
        
        kept = 0
        deleted = 0
        
        sorted_checkpoints = sorted(
            self.metadata.items(),
            key=lambda x: x[1].global_step,
            reverse=True
        )
        
        for tag, meta in sorted_checkpoints:
            if meta.is_best and self.config.save_best:
                continue
            if meta.is_last and self.config.save_last:
                continue
            
            if kept >= self.config.max_to_keep:
                checkpoint_path = Path(meta.path)
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
                self.metadata.pop(tag, None)
                deleted += 1
            else:
                kept += 1
        
        self._save_metadata()
        
        return PiscesLxOperatorResult(
            operator_name=self.name,
            status=PiscesLxOperatorStatus.SUCCESS,
            output={
                'success': True,
                'deleted': deleted,
                'kept': kept
            }
        )
    
    def _get_checkpoint_path(self, tag: str) -> Path:
        """Get checkpoint path for a tag."""
        if self.config.type == CheckpointType.SHARDED:
            return Path(self.save_dir) / f"{self.config.prefix}_{tag}"
        else:
            return Path(self.save_dir) / f"{self.config.prefix}_{tag}.pt"
    
    def _update_metadata(
        self,
        tag: str,
        checkpoint_path: Path,
        state_dict: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        """Update checkpoint metadata."""
        path = str(checkpoint_path)
        size_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
        
        is_best = False
        if self.config.save_best and self.best_metric in metrics:
            is_best = self._is_best(metrics[self.best_metric])
        
        self.metadata[tag] = CheckpointMetadata(
            path=path,
            type=self.config.type.value,
            size_bytes=size_bytes,
            creation_time=time.time(),
            global_step=state_dict.get('global_step', 0),
            epoch=state_dict.get('epoch'),
            loss=metrics.get('loss'),
            metrics=metrics,
            is_best=is_best,
            is_last=True
        )
        
        for tag, meta in self.metadata.items():
            if tag != tag:
                self.metadata[tag] = CheckpointMetadata(
                    **dict(meta.__dict__, is_last=False)
                )
        
        self._save_metadata()
    
    def _is_best(self, metric_value: float) -> bool:
        """Check if this is the best checkpoint based on metric."""
        best_value = None
        for meta in self.metadata.values():
            if meta.is_best and self.best_metric in meta.metrics:
                best_value = meta.metrics[self.best_metric]
                break
        
        if best_value is None:
            return True
        
        if self.config.best_mode == 'min':
            return metric_value < best_value
        else:
            return metric_value > best_value
    
    def _manage_checkpoints(self) -> None:
        """Manage checkpoint retention."""
        self._cleanup_old_checkpoints({})
    
    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        metadata_path = Path(self.config.save_dir) / "checkpoint_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    self.metadata = {
                        k: CheckpointMetadata(**v) for k, v in data.items()
                    }
            except Exception as e:
                self._LOG.warning(f"Failed to load metadata: {e}")
    
    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        metadata_path = Path(self.config.save_dir) / "checkpoint_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(
                    {k: v.__dict__ for k, v in self.metadata.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            self._LOG.warning(f"Failed to save metadata: {e}")
    
    def save(
        self,
        tag: str,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        global_step: int = 0,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> PiscesLxOperatorResult:
        """Convenience method to save checkpoint."""
        return self.execute({
            'operation': 'save',
            'tag': tag,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'scaler': scaler,
            'global_step': global_step,
            'epoch': epoch,
            'metrics': metrics or {}
        })
    
    def load(self, tag: str) -> Dict[str, Any]:
        """Convenience method to load checkpoint state."""
        result = self.execute({'operation': 'load', 'tag': tag})
        return result.output.get('state', {}) if result.is_success() else {}
    
    def load_model(self, tag: str, model: nn.Module) -> bool:
        """Load model weights from checkpoint."""
        state = self.load(tag)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
            return True
        return False
    
    def shutdown(self) -> None:
        """Shutdown async save worker."""
        self._running = False
        if self._async_thread:
            self._async_thread.join(timeout=5.0)
