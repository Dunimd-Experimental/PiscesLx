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

"""
Training Pipeline Operator

End-to-end training pipeline management for the PiscesLx framework. This module
provides a high-level interface for managing complete training workflows including
epoch management, evaluation loops, checkpointing, and callback systems.

Pipeline Stages:
    1. Initialization: Model, optimizer, scheduler setup
    2. Training Loop: Epoch-based training with progress tracking
    3. Evaluation Loop: Validation with metric computation
    4. Checkpointing: Automatic and manual model saving
    5. Logging: Structured logging with metrics history

Features:
    - Epoch-based training with progress bars
    - Automatic evaluation at specified intervals
    - Callback system for custom training logic
    - Early stopping with configurable patience
    - Learning rate scheduling integration
    - Distributed training support
    - Resume from checkpoint

Callback System:
    The pipeline supports callbacks at various stages:
    - on_training_start: Called before training begins
    - on_epoch_start: Called at the start of each epoch
    - on_step_end: Called after each training step
    - on_epoch_end: Called at the end of each epoch
    - on_evaluation_end: Called after evaluation
    - on_training_end: Called when training completes

Usage Examples:
    Basic Pipeline:
        >>> from tools.train import TrainingConfig, TrainingPipelineOperator
        >>> 
        >>> config = TrainingConfig(max_steps=10000)
        >>> pipeline = TrainingPipelineOperator(config)
        >>> 
        >>> pipeline.initialize(model, train_loader, val_loader)
        >>> pipeline.train(epochs=3)

    With Callbacks:
        >>> def log_callback(step, metrics):
        ...     print(f"Step {step}: loss={metrics['loss']:.4f}")
        >>> 
        >>> pipeline.add_callback(log_callback)
        >>> pipeline.train(epochs=3)

    Resume Training:
        >>> pipeline.load_checkpoint("checkpoint-5000.pt")
        >>> pipeline.train(epochs=3, resume=True)
"""

import torch
from typing import Dict, Any, Callable, Optional, List, Union
from pathlib import Path
import time
from datetime import datetime
import json
import yaml

# OPSC operator system integration
from utils.opsc.base import PiscesLxTransformOperator
from utils.opsc.interface import PiscesLxOperatorConfig
from utils.opsc.registry import PiscesLxOperatorRegistrar
from utils.dc import PiscesLxLogger

from .core import PiscesLxTrainingOperator
from .config import TrainingConfig
from configs.version import VERSION


from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Train", file_path=get_log_file("PiscesLx.Tools.Train"), enable_file=True)


@PiscesLxOperatorRegistrar()
class TrainingPipelineOperator(PiscesLxTransformOperator):
    """
    Training Pipeline Operator
    
    Manages complete end-to-end training workflows with support for callbacks,
    evaluation loops, and checkpoint management.
    
    Attributes:
        config: TrainingConfig instance
        trainer: PiscesLxTrainingOperator for core training logic
        callbacks: List of callback functions for custom logic
        metrics: Dictionary storing training and evaluation metrics
        
    Callback Interface:
        Callbacks receive (step, metrics) and can modify metrics in-place:
        >>> def callback(step, metrics):
        ...     metrics['custom_metric'] = compute_metric()
    
    Example:
        >>> config = TrainingConfig(max_steps=10000)
        >>> pipeline = TrainingPipelineOperator(config)
        >>> pipeline.initialize(model, train_loader, val_loader)
        >>> pipeline.train(epochs=3)
    """
    
    def __init__(self, config: Optional[Union[PiscesLxOperatorConfig, TrainingConfig, str, Dict[str, Any]]] = None):
        """
        Initialize training pipeline.
        
        Args:
            config: TrainingConfig with all training parameters
        """
        op_config = config if isinstance(config, PiscesLxOperatorConfig) else None
        super().__init__(op_config)
        self.train_config = self._normalize_train_config(config)
        self.trainer = PiscesLxTrainingOperator(self.train_config)
        self.callbacks = []
        self.metrics = {}
        self.stage = getattr(self.train_config, 'stage', None)

    def _normalize_train_config(self, config: Optional[Union[PiscesLxOperatorConfig, TrainingConfig, str, Dict[str, Any]]]) -> TrainingConfig:
        if isinstance(config, TrainingConfig):
            return config
        if isinstance(config, str):
            if config.lower().endswith(".json"):
                return TrainingConfig.load_from_json(config)
            if config.lower().endswith((".yaml", ".yml")):
                with open(config, "r", encoding="utf-8") as f:
                    return TrainingConfig.from_dict(yaml.safe_load(f) or {})
            raise ValueError(f"Unsupported config file format: {config}")
        if isinstance(config, dict):
            return TrainingConfig.from_dict(config)
        if isinstance(config, PiscesLxOperatorConfig):
            params = getattr(config, "parameters", {}) or {}
            cfg = params.get("training_config", None)
            if isinstance(cfg, TrainingConfig):
                return cfg
            if isinstance(cfg, str):
                if cfg.lower().endswith(".json"):
                    return TrainingConfig.load_from_json(cfg)
                if cfg.lower().endswith((".yaml", ".yml")):
                    with open(cfg, "r", encoding="utf-8") as f:
                        return TrainingConfig.from_dict(yaml.safe_load(f) or {})
                raise ValueError(f"Unsupported config file format: {cfg}")
            if isinstance(cfg, dict):
                return TrainingConfig.from_dict(cfg)
        return TrainingConfig()
    
    @property
    def name(self) -> str:
        return "training_pipeline_operator"
    
    @property
    def version(self) -> str:
        return VERSION
    
    @property
    def description(self) -> str:
        return "End-to-end training pipeline management with callbacks and checkpointing"
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute training pipeline transformation."""
        return data
        
    def add_callback(self, callback: Callable):
        """
        Add training callback function.
        
        Callbacks are called at each training step with current metrics.
        
        Args:
            callback: Function with signature callback(step, metrics)
        """
        self.callbacks.append(callback)
        _LOG.info(f"Callback added: {callback.__name__}")
    
    def remove_callback(self, callback: Callable):
        """Remove training callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            _LOG.info(f"Callback removed: {callback.__name__}")
    
    def execute_callbacks(self, stage: str, **kwargs):
        """Execute all callback functions with training stage context."""
        for callback in self.callbacks:
            try:
                callback(stage=stage, training_stage=self.stage.value if self.stage else None, **kwargs)
            except Exception as e:
                PiscesLxRunCancelled = None
                try:
                    from .run_reporter import PiscesLxRunCancelled as _PiscesLxRunCancelled
                    PiscesLxRunCancelled = _PiscesLxRunCancelled
                except Exception:
                    PiscesLxRunCancelled = None
                if PiscesLxRunCancelled is not None and isinstance(e, PiscesLxRunCancelled):
                    raise
                _LOG.error(f"Callback {callback.__name__} failed: {e}")
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        Train one epoch.
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Epoch statistics
        """
        self.trainer.model.train()
        epoch_stats = {
            'epoch': epoch,
            'total_loss': 0.0,
            'batch_count': 0,
            'avg_loss': 0.0,
            'avg_grad_norm': 0.0,
            'avg_throughput': 0.0
        }
        
        _LOG.info(f"Starting epoch {epoch}")
        self.execute_callbacks('epoch_start', epoch=epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            step_result = self.trainer.train_step(batch)
            
            epoch_stats['total_loss'] += step_result['loss']
            epoch_stats['batch_count'] += 1
            epoch_stats['avg_grad_norm'] += step_result['grad_norm']
            epoch_stats['avg_throughput'] += step_result['throughput']
            
            self.execute_callbacks('batch_end', 
                                 batch_idx=batch_idx,
                                 step_result=step_result)
            
            if batch_idx % self.train_config.log_steps == 0:
                _LOG.info(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Loss={step_result['loss']:.4f}, "
                    f"LR={step_result['learning_rate']:.2e}, "
                    f"Throughput={step_result['throughput']:.2f} samples/sec"
                )
            
            if self.trainer.global_step % self.train_config.save_steps == 0:
                checkpoint_path = Path(self.train_config.output_dir) / f"checkpoint_step_{self.trainer.global_step}.pt"
                self.trainer.save_checkpoint(str(checkpoint_path))
                _LOG.info(f"Checkpoint saved at step {self.trainer.global_step}")
                self.execute_callbacks(
                    'checkpoint_saved',
                    path=str(checkpoint_path),
                    global_step=self.trainer.global_step,
                )
            
            if self.trainer.global_step % self.train_config.eval_steps == 0:
                self.execute_callbacks('evaluation_start', global_step=self.trainer.global_step)
        
        if epoch_stats['batch_count'] > 0:
            epoch_stats['avg_loss'] = epoch_stats['total_loss'] / epoch_stats['batch_count']
            epoch_stats['avg_grad_norm'] /= epoch_stats['batch_count']
            epoch_stats['avg_throughput'] /= epoch_stats['batch_count']
        
        _LOG.info(f"Epoch {epoch} completed - Avg Loss: {epoch_stats['avg_loss']:.4f}")
        self.execute_callbacks('epoch_end', epoch_stats=epoch_stats)
        
        return epoch_stats
    
    def evaluate(self, dataloader, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            dataloader: Validation data loader
            metrics: List of evaluation metrics
            
        Returns:
            Evaluation results dictionary
        """
        self.trainer.model.eval()
        eval_stats = {'eval_loss': 0.0, 'batch_count': 0}
        
        if metrics is None:
            metrics = ['accuracy', 'perplexity']
        
        for metric in metrics:
            eval_stats[f'eval_{metric}'] = 0.0
        
        _LOG.info("Starting evaluation...")
        self.execute_callbacks('evaluation_start')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                outputs = self.trainer.forward_pass(batch)
                loss = outputs['loss']
                
                eval_stats['eval_loss'] += loss.item()
                eval_stats['batch_count'] += 1
                
                if 'accuracy' in metrics:
                    accuracy = self._compute_accuracy(outputs, batch)
                    eval_stats['eval_accuracy'] += accuracy
                
                if 'perplexity' in metrics:
                    perplexity = torch.exp(loss).item()
                    eval_stats['eval_perplexity'] += perplexity
                
                self.execute_callbacks('eval_batch_end', 
                                     batch_idx=batch_idx,
                                     batch_outputs=outputs)
        
        if eval_stats['batch_count'] > 0:
            eval_stats['eval_loss'] /= eval_stats['batch_count']
            if 'accuracy' in metrics:
                eval_stats['eval_accuracy'] /= eval_stats['batch_count']
            if 'perplexity' in metrics:
                eval_stats['eval_perplexity'] /= eval_stats['batch_count']
        
        _LOG.info(f"Evaluation completed - Loss: {eval_stats['eval_loss']:.4f}")
        self.execute_callbacks('evaluation_end', eval_stats=eval_stats)
        
        return eval_stats
    
    def _compute_accuracy(self, outputs: Dict[str, torch.Tensor], 
                         batch: Dict[str, torch.Tensor]) -> float:
        """Compute accuracy metric."""
        if 'logits' in outputs and 'labels' in batch:
            predictions = torch.argmax(outputs['logits'], dim=-1)
            labels = batch['labels']
            mask = labels != -100
            if mask.sum() > 0:
                correct = (predictions[mask] == labels[mask]).sum().item()
                return correct / mask.sum().item()
        return 0.0
    
    def fit(self, train_dataloader, val_dataloader=None, 
            epochs: int = 1, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Number of training epochs
            resume_from: Checkpoint path to resume training from
            
        Returns:
            Training history
        """
        if resume_from:
            self.trainer.load_checkpoint(resume_from)
            _LOG.info(f"Resumed training from {resume_from}")
        
        training_history = {
            'epochs': [],
            'evaluations': [],
            'best_checkpoint': None,
            'training_start_time': datetime.now().isoformat()
        }
        
        _LOG.info("Starting training pipeline...")
        self.execute_callbacks('training_start', 
                             total_epochs=epochs,
                             max_steps=self.train_config.max_steps)
        from .run_reporter import PiscesLxRunCancelled

        try:
            for epoch in range(epochs):
                epoch_stats = self.train_epoch(train_dataloader, epoch)
                training_history['epochs'].append(epoch_stats)
                
                if val_dataloader:
                    eval_stats = self.evaluate(val_dataloader)
                    training_history['evaluations'].append(eval_stats)
                    
                    if eval_stats['eval_loss'] < self.trainer.best_metric:
                        self.trainer.best_metric = eval_stats['eval_loss']
                        best_path = Path(self.train_config.output_dir) / "best_model.pt"
                        self.trainer.save_checkpoint(str(best_path), 
                                                   metadata={'is_best': True})
                        training_history['best_checkpoint'] = str(best_path)
                        _LOG.info(f"New best model saved with loss: {eval_stats['eval_loss']:.4f}")
                        self.execute_callbacks(
                            'checkpoint_saved',
                            path=str(best_path),
                            global_step=self.trainer.global_step,
                        )
                
                if self.trainer.global_step >= self.train_config.max_steps:
                    _LOG.info(f"Reached maximum steps ({self.train_config.max_steps}), stopping training")
                    break
                    
        except PiscesLxRunCancelled:
            training_history['cancelled'] = True
            _LOG.info("Training cancelled")
        except KeyboardInterrupt:
            _LOG.info("Training interrupted by user")
        except Exception as e:
            _LOG.error(f"Training failed with error: {e}")
            raise
        finally:
            final_path = Path(self.train_config.output_dir) / "final_model.pt"
            self.trainer.save_checkpoint(str(final_path), 
                                       metadata={'is_final': True})
            training_history['final_checkpoint'] = str(final_path)
            training_history['training_end_time'] = datetime.now().isoformat()
            
            _LOG.info("Training pipeline completed")
            self.execute_callbacks('training_end', training_history=training_history)
        
        return training_history


@PiscesLxOperatorRegistrar()
class CurriculumLearningOperator(PiscesLxTransformOperator):
    """
    Curriculum Learning Operator.
    
    Implements progressive difficulty training strategy.
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, difficulty_schedule: Optional[List[Dict[str, Any]]] = None):
        super().__init__(config)
        if config is not None and difficulty_schedule is None:
            params = getattr(config, "parameters", {}) or {}
            difficulty_schedule = params.get("difficulty_schedule")
        self.difficulty_schedule = difficulty_schedule or []
        self.current_stage = 0
        self.stage_metrics = []
    
    @property
    def name(self) -> str:
        return "curriculum_learning_operator"
    
    @property
    def version(self) -> str:
        return VERSION
    
    @property
    def description(self) -> str:
        return "Progressive difficulty training with curriculum learning strategy"
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute curriculum learning transformation."""
        return data
        
    def get_current_dataloader(self, base_dataloader_factory: Callable) -> torch.utils.data.DataLoader:
        """
        Get data loader for current stage.
        
        Args:
            base_dataloader_factory: Base data loader factory function
            
        Returns:
            Data loader for current stage
        """
        stage_config = self.difficulty_schedule[self.current_stage]
        return base_dataloader_factory(**stage_config.get('dataloader_args', {}))
    
    def should_advance_stage(self, current_metrics: Dict[str, float], 
                           patience: int = 3) -> bool:
        """
        Determine whether to advance to next stage.
        
        Args:
            current_metrics: Current stage metrics
            patience: Number of epochs to wait
            
        Returns:
            Whether to advance
        """
        self.stage_metrics.append(current_metrics)
        
        if len(self.stage_metrics) < patience:
            return False
        
        recent_metrics = self.stage_metrics[-patience:]
        loss_improved = all(
            recent_metrics[i]['loss'] <= recent_metrics[i-1]['loss'] * 0.98
            for i in range(1, len(recent_metrics))
        )
        
        return loss_improved and self.current_stage < len(self.difficulty_schedule) - 1
    
    def advance_stage(self):
        """Advance to next stage."""
        if self.current_stage < len(self.difficulty_schedule) - 1:
            self.current_stage += 1
            self.stage_metrics = []
            _LOG.info(f"Advanced to curriculum stage {self.current_stage}")


@PiscesLxOperatorRegistrar()
class PiscesLxMultiTaskTrainingOperator(PiscesLxTransformOperator):
    """
    Multi-Task Training Operator.
    
    Supports simultaneous training of multiple related tasks.
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None, task_weights: Optional[Dict[str, float]] = None):
        super().__init__(config)
        if config is not None and task_weights is None:
            params = getattr(config, "parameters", {}) or {}
            task_weights = params.get("task_weights")
        self.task_weights = task_weights or {}
        self.task_losses = {}
    
    @property
    def name(self) -> str:
        return "multi_task_training_operator"
    
    @property
    def version(self) -> str:
        return VERSION
    
    @property
    def description(self) -> str:
        return "Multi-task training with weighted loss computation"
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-task training transformation."""
        return data
        
    def compute_multi_task_loss(self, task_outputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Compute multi-task joint loss.
        
        Args:
            task_outputs: Output dictionary for each task
            
        Returns:
            Weighted total loss
        """
        total_loss = 0.0
        
        for task_name, outputs in task_outputs.items():
            if 'loss' in outputs:
                weight = self.task_weights.get(task_name, 1.0)
                task_loss = outputs['loss'] * weight
                total_loss += task_loss
                self.task_losses[task_name] = task_loss.item()
        
        return total_loss
    
    def get_task_metrics(self) -> Dict[str, float]:
        """Get metrics for each task."""
        return self.task_losses.copy()


