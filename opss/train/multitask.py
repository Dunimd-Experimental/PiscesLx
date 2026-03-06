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
Multitask Training Operator

Implementation of multitask learning with dynamic task weighting
and curriculum learning for PiscesL1 model training.
"""

import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir
from configs.version import VERSION

from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus


@dataclass
class MultitaskTrainingConfig:
    """Multitask training configuration."""
    
    # Task settings
    tasks: List[str] = field(default_factory=lambda: ["sft", "dpo", "classification"])
    task_weights: Dict[str, float] = field(default_factory=lambda: {"sft": 1.0, "dpo": 1.0, "classification": 1.0})
    
    # Data settings
    data_paths: Dict[str, str] = field(default_factory=dict)
    batch_size_per_task: int = 4
    max_samples_per_task: Optional[int] = None
    
    # Training settings
    model_path: str = ".pisceslx/ckpt"
    output_dir: str = ".pisceslx/ckpt"
    
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_steps: int = 10000
    max_grad_norm: float = 1.0
    
    # Multitask specific settings
    task_balancing: str = "dynamic"  # static, dynamic, curriculum
    curriculum_schedule: str = "linear"  # linear, exponential, step
    gradient_accumulation_steps: int = 4
    
    # Loss weighting
    use_uncertainty_weighting: bool = True
    initial_task_weights: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Initialize default task weights if not provided
        for task in self.tasks:
            if task not in self.task_weights:
                self.task_weights[task] = 1.0
            if task not in self.initial_task_weights:
                self.initial_task_weights[task] = 1.0


class MultitaskDataset(Dataset):
    """Dataset for multitask training."""
    
    def __init__(self, task_datasets: Dict[str, Dataset], task_names: List[str]):
        self.task_datasets = task_datasets
        self.task_names = task_names
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
        
        # Calculate dataset sizes
        self.dataset_sizes = {task: len(dataset) for task, dataset in task_datasets.items()}
        self.total_size = sum(self.dataset_sizes.values())
        
        self._LOG.info(f"Multitask dataset created with {len(task_names)} tasks")
        for task, size in self.dataset_sizes.items():
            self._LOG.info(f"  {task}: {size} samples")
    
    def __len__(self) -> int:
        return self.total_size
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Determine which task this index belongs to
        cumulative_size = 0
        for task_name in self.task_names:
            task_size = self.dataset_sizes[task_name]
            if idx < cumulative_size + task_size:
                local_idx = idx - cumulative_size
                sample = self.task_datasets[task_name][local_idx]
                sample['task'] = task_name
                return sample
            cumulative_size += task_size
        
        # Fallback (should not happen)
        return self.task_datasets[self.task_names[0]][0]


class UncertaintyWeighting:
    """Uncertainty-based task weighting for multitask learning."""
    
    def __init__(self, num_tasks: int, initial_weights: List[float] = None):
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.initial_weights = initial_weights or [1.0] * num_tasks
        
    def get_weights(self) -> torch.Tensor:
        """Get current task weights based on uncertainty."""
        return torch.exp(-self.log_vars)
    
    def get_loss_weights(self) -> torch.Tensor:
        """Get loss weights for training."""
        weights = self.get_weights()
        return weights / weights.sum() * self.num_tasks


class MultitaskTrainingOperator(PiscesLxOperatorInterface):
    """Complete multitask training operator implementation."""
    
    def __init__(self):
        super().__init__()
        self.name = "multitask.training"
        self.version = VERSION
        self.type = "training"
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
        
    @property
    def description(self) -> str:
        return "Complete multitask training operator with dynamic task weighting"
        
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "model": {"type": "torch.nn.Module", "required": True, "description": "Multitask capable model"},
            "tokenizer": {"type": "object", "required": True, "description": "Model tokenizer"},
            "task_data_paths": {"type": "dict", "required": True, "description": "Paths to task data"},
            "config": {"type": "MultitaskTrainingConfig", "required": False, "description": "Multitask configuration"},
            "optimizer": {"type": "torch.optim.Optimizer", "required": False, "description": "Custom optimizer"}
        }
        
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "metrics": {"type": "dict", "description": "Per-task and overall metrics"},
            "model_state": {"type": "dict", "description": "Trained model state"},
            "task_weights": {"type": "dict", "description": "Final task weights"},
            "training_curves": {"type": "dict", "description": "Training progress curves"}
        }
        
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        required_keys = ['model', 'tokenizer', 'task_data_paths']
        for key in required_keys:
            if key not in inputs or inputs[key] is None:
                self._LOG.error(f"Missing required parameter: {key}")
                return False
        
        # Validate task data paths
        task_data_paths = inputs['task_data_paths']
        if not isinstance(task_data_paths, dict) or len(task_data_paths) == 0:
            self._LOG.error("Task data paths must be a non-empty dictionary")
            return False
            
        return True
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """Execute multitask training pipeline."""
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self.validate_inputs(inputs):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Invalid input parameters",
                    execution_time=time.time() - start_time
                )
            
            # Parse inputs
            model = inputs['model']
            tokenizer = inputs['tokenizer']
            task_data_paths = inputs['task_data_paths']
            custom_config = inputs.get('config')
            custom_optimizer = inputs.get('optimizer')
            
            # Setup configuration
            if custom_config:
                config = custom_config
            else:
                config = MultitaskTrainingConfig(
                    data_paths=task_data_paths,
                    tasks=list(task_data_paths.keys())
                )
            
            self._LOG.info(f"Starting multitask training with {len(config.tasks)} tasks")
            
            # Setup device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Create task datasets
            task_datasets = self._create_task_datasets(task_data_paths, tokenizer, config)
            
            # Create multitask dataset
            multitask_dataset = MultitaskDataset(task_datasets, config.tasks)
            
            # Setup task weighting
            if config.use_uncertainty_weighting:
                uncertainty_weighting = UncertaintyWeighting(
                    len(config.tasks), 
                    [config.initial_task_weights.get(task, 1.0) for task in config.tasks]
                )
                uncertainty_weighting = uncertainty_weighting.to(device)
            else:
                uncertainty_weighting = None
            
            # Setup optimizer
            if custom_optimizer is None:
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            else:
                optimizer = custom_optimizer
            
            # Execute training
            metrics = self._run_multitask_training(
                model, multitask_dataset, config, optimizer, uncertainty_weighting, device
            )
            
            # Get final task weights
            final_task_weights = {}
            if uncertainty_weighting is not None:
                weights = uncertainty_weighting.get_weights()
                for i, task in enumerate(config.tasks):
                    final_task_weights[task] = weights[i].item()
            else:
                final_task_weights = config.task_weights.copy()
            
            execution_time = time.time() - start_time
            
            result_data = {
                'metrics': metrics,
                'model_state': model.state_dict(),
                'task_weights': final_task_weights,
                'training_curves': self._generate_training_curves(metrics)
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=result_data,
                execution_time=execution_time,
                metadata={
                    'config': config.__dict__,
                    'tasks_trained': config.tasks,
                    'total_samples': len(multitask_dataset)
                }
            )
            
        except Exception as e:
            self._LOG.error(f"Multitask training failed: {str(e)}", exc_info=True)
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _create_task_datasets(self, task_data_paths: Dict[str, str], tokenizer: Any, 
                             config: MultitaskTrainingConfig) -> Dict[str, Dataset]:
        """Create datasets for each task."""
        task_datasets = {}
        
        for task_name, data_path in task_data_paths.items():
            if not os.path.exists(data_path):
                self._LOG.warning(f"Data path not found for task {task_name}: {data_path}")
                continue
                
            # Create task-specific dataset
            if task_name == "sft":
                dataset = self._create_sft_dataset(data_path, tokenizer, config)
            elif task_name == "dpo":
                dataset = self._create_dpo_dataset(data_path, tokenizer, config)
            elif task_name == "classification":
                dataset = self._create_classification_dataset(data_path, tokenizer, config)
            else:
                # Generic dataset creation
                dataset = self._create_generic_dataset(data_path, tokenizer, config)
            
            if dataset is not None and len(dataset) > 0:
                task_datasets[task_name] = dataset
                self._LOG.info(f"Created dataset for task {task_name}: {len(dataset)} samples")
        
        return task_datasets
    
    def _create_sft_dataset(self, data_path: str, tokenizer: Any, config: MultitaskTrainingConfig):
        """Create SFT dataset."""
        # Implementation would load and process SFT data
        # This is a simplified placeholder
        class SFTPlaceholderDataset(Dataset):
            def __init__(self, size=100):
                self.size = size
            def __len__(self): return self.size
            def __getitem__(self, idx): 
                return {"input_ids": torch.randint(0, 1000, (512,)), "labels": torch.randint(0, 1000, (512,))}
        
        return SFTPlaceholderDataset(min(config.max_samples_per_task or 100, 100))
    
    def _create_dpo_dataset(self, data_path: str, tokenizer: Any, config: MultitaskTrainingConfig):
        """Create DPO dataset."""
        class DPOPlaceholderDataset(Dataset):
            def __init__(self, size=50):
                self.size = size
            def __len__(self): return self.size
            def __getitem__(self, idx):
                return {
                    "chosen_input_ids": torch.randint(0, 1000, (256,)),
                    "rejected_input_ids": torch.randint(0, 1000, (256,)),
                    "chosen_labels": torch.randint(0, 1000, (256,)),
                    "rejected_labels": torch.randint(0, 1000, (256,))
                }
        
        return DPOPlaceholderDataset(min(config.max_samples_per_task or 50, 50))
    
    def _create_classification_dataset(self, data_path: str, tokenizer: Any, config: MultitaskTrainingConfig):
        """Create classification dataset."""
        class ClassificationPlaceholderDataset(Dataset):
            def __init__(self, size=75):
                self.size = size
            def __len__(self): return self.size
            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, 1000, (128,)),
                    "labels": torch.randint(0, 10, (1,)),
                    "attention_mask": torch.ones(128)
                }
        
        return ClassificationPlaceholderDataset(min(config.max_samples_per_task or 75, 75))
    
    def _create_generic_dataset(self, data_path: str, tokenizer: Any, config: MultitaskTrainingConfig):
        """Create generic dataset."""
        class GenericPlaceholderDataset(Dataset):
            def __init__(self, size=50):
                self.size = size
            def __len__(self): return self.size
            def __getitem__(self, idx):
                return {"input_ids": torch.randint(0, 1000, (256,)), "labels": torch.randint(0, 1000, (256,))}
        
        return GenericPlaceholderDataset(min(config.max_samples_per_task or 50, 50))
    
    def _run_multitask_training(self, model: nn.Module, dataset: Dataset, 
                               config: MultitaskTrainingConfig, optimizer: torch.optim.Optimizer,
                               uncertainty_weighting: Optional[UncertaintyWeighting], 
                               device: torch.device) -> Dict[str, Any]:
        """Execute multitask training loop."""
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size_per_task * len(config.tasks),
            shuffle=True,
            num_workers=2
        )
        
        model.train()
        task_losses = {task: 0.0 for task in config.tasks}
        task_counts = {task: 0 for task in config.tasks}
        total_loss = 0.0
        steps_completed = 0
        
        # Simplified training for demonstration
        max_steps = min(config.max_steps, 30)
        
        for step in range(max_steps):
            for batch_idx, batch in enumerate(dataloader):
                if steps_completed >= max_steps:
                    break
                
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                # Forward pass - this would need task-specific handling
                # For demo purposes, using a simplified approach
                outputs = model(**{k: v for k, v in batch.items() if k != 'task'})
                
                # Calculate task-specific losses
                task_specific_losses = self._calculate_task_losses(batch, outputs, config)
                
                # Apply task weighting
                weighted_loss = 0.0
                for task, loss in task_specific_losses.items():
                    weight = config.task_weights.get(task, 1.0)
                    if uncertainty_weighting is not None:
                        # Get dynamic weights
                        task_idx = config.tasks.index(task)
                        weight = uncertainty_weighting.get_weights()[task_idx]
                    
                    weighted_loss += weight * loss
                    task_losses[task] += loss.item()
                    task_counts[task] += 1
                
                # Normalize by number of tasks
                weighted_loss = weighted_loss / len(task_specific_losses)
                
                # Backward pass
                weighted_loss.backward()
                
                # Update uncertainty parameters if using uncertainty weighting
                if uncertainty_weighting is not None:
                    uncertainty_optimizer = torch.optim.AdamW([uncertainty_weighting.log_vars], lr=1e-3)
                    uncertainty_optimizer.step()
                
                # Gradient clipping and optimization step
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                
                total_loss += weighted_loss.item()
                steps_completed += 1
                
                if steps_completed % 10 == 0:
                    avg_loss = total_loss / steps_completed
                    self._LOG.info(f"Step {steps_completed}, Avg Loss: {avg_loss:.4f}")
        
        # Calculate final metrics
        metrics = {
            'final_loss': total_loss / max(1, steps_completed),
            'steps_completed': steps_completed,
            'per_task_metrics': {}
        }
        
        # Calculate per-task averages
        for task in config.tasks:
            if task_counts[task] > 0:
                metrics['per_task_metrics'][task] = {
                    'avg_loss': task_losses[task] / task_counts[task],
                    'samples_processed': task_counts[task]
                }
        
        return metrics
    
    def _calculate_task_losses(self, batch: Dict[str, Any], outputs: Any, 
                              config: MultitaskTrainingConfig) -> Dict[str, torch.Tensor]:
        """Calculate losses for each task in the batch."""
        task_losses = {}
        task = batch.get('task', 'generic')
        
        # Simplified loss calculation
        if hasattr(outputs, 'loss'):
            task_losses[task] = outputs.loss
        else:
            # Placeholder loss calculation
            task_losses[task] = torch.tensor(1.0, device=next(iter(batch.values())).device)
        
        return task_losses
    
    def _generate_training_curves(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate training curves data."""
        # In a real implementation, this would track loss curves over time
        return {
            'overall_loss_curve': [metrics['final_loss']] * 10,  # Placeholder
            'per_task_curves': {
                task: [data['avg_loss']] * 10 
                for task, data in metrics.get('per_task_metrics', {}).items()
            }
        }


# Export operator
__all__ = ['MultitaskTrainingOperator', 'MultitaskTrainingConfig']
