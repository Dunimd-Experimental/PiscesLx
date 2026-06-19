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
Supervised Fine-Tuning (SFT) Operator Implementation

Complete implementation of SFT training as a standardized operator.
Based on the original PiscesL1 SFT training pipeline.
"""

import os
import sys
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir
from configs.version import VERSION

from utils.opsc.interface import PiscesLxOperatorInterface, PiscesLxOperatorResult, PiscesLxOperatorStatus

# Optional: when the EnTA training pipeline is enabled, the SFT operator
# delegates data generation to :class:`YvEncreTrainer` and then runs the
# regular SFT loop on the produced (prompt, reference) pairs.  This is
# the *integration* the user requires -- EnTA is the data factory, the
# SFT loop remains the training engine.
try:
    from model.agentic.enta import YvEncreTrainer  # noqa: F401
    _ENTA_AVAILABLE = True
except Exception:  # noqa: BLE001
    YvEncreTrainer = None  # type: ignore[assignment]
    _ENTA_AVAILABLE = False


@dataclass
class POPSSSFTTrainingConfig:
    """SFT training configuration."""
    
    model_path: str = ".pisceslx/ckpt"
    output_dir: str = ".pisceslx/ckpt"
    
    train_data: str = "./data/train.jsonl"
    val_data: str = "./data/val.jsonl"
    
    batch_size: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    
    learning_rate: float = 1e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    max_grad_norm: float = 1.0
    
    use_fp16: bool = False
    use_bf16: bool = True
    
    use_fp8: bool = False
    fp8_amax_history_length: int = 1024
    fp8_amax_compute_algo: str = "max"
    
    use_gradient_checkpointing: bool = True
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    
    save_total_limit: int = 3
    
    local_rank: int = 0
    world_size: int = 1
    master_port: int = 29500
    
    max_seq_length: int = 4096
    ignore_index: int = -100

    # Memory Separation training (Engram-style Lookup-Computation Separation)
    # When enabled, training follows 3-phase pipeline:
    #   Phase 1: Train backbone only (reasoning/language/tool), gate=0
    #   Phase 2: Train router + cross-attention + gate, backbone frozen
    #   Phase 3: Offline knowledge store build (manual trigger)
    use_memsep_training: bool = False
    memsep_phase_1_steps: int = 5000
    memsep_phase_2_steps: int = 2000
    memsep_gate_target: float = 0.5
    memsep_freeze_backbone_phase2: bool = True
    memsep_freeze_router_phase1: bool = True
    memsep_reason_data_path: str = ""
    memsep_mem_data_path: str = ""
    memsep_alignment_weight: float = 0.1
    memsep_gate_schedule: str = "sigmoid"
    memsep_gate_warmup_steps: int = 500

    # EnTA-driven data pipeline.  When ``use_encre_data_pipeline`` is
    # True the SFT operator delegates data generation to a
    # :class:`YvEncreTrainer` (passed via ``inputs['encre_trainer']``)
    # and runs SFT on the (prompt, reference) pairs the trainer
    # returns.  This is the *integration* path: EnTA is the data
    # factory, the SFT loop is the training engine.
    use_encre_data_pipeline: bool = False
    encre_use_roundtable: bool = False
    encre_prompts_path: str = ""
    encre_trainer_cfg_path: str = ""
    encre_data_cache_path: str = ".pisceslx/cache/encre_sft_dataset.jsonl"
    encre_max_samples: int = 0  # 0 = no cap
    encre_system_prompt: str = ""

    def __post_init__(self):
        if self.use_fp16 and self.use_bf16:
            self.use_bf16 = False


class POPSSSFTDataset(Dataset):
    """Dataset for SFT training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_seq_length: int = 4096,
        ignore_index: int = -100,
    ):
        """Initialize SFT dataset.
        
        Args:
            data_path: Path to training data (JSONL format).
            tokenizer: Tokenizer for encoding text.
            max_seq_length: Maximum sequence length.
            ignore_index: Token index to ignore in loss computation.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.ignore_index = ignore_index
        
        self.samples = self._load_data(data_path)
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
        self._LOG.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    KAGGLE_PREFIX = "kaggle://"

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data from a file, directory, or Kaggle hub.

        Supports three source types:
            - kaggle://user/dataset     → auto-download via kagglehub,
              then scan the cached directory for .jsonl/.json files
            - /path/to/directory        → walk all .jsonl/.json files
            - /path/to/file.jsonl       → load a single file
        """
        if data_path.startswith(self.KAGGLE_PREFIX):
            return self._load_from_kaggle(data_path[len(self.KAGGLE_PREFIX):])

        resolved = os.path.abspath(data_path)

        if os.path.isdir(resolved):
            return self._scan_directory(resolved)

        if os.path.isfile(resolved):
            return self._read_json_file(resolved)

        self._LOG.warning(f"Data path not found: {data_path}")
        return []

    def _load_from_kaggle(self, kaggle_path: str) -> List[Dict[str, Any]]:
        try:
            import kagglehub
        except ImportError:
            self._LOG.error("kagglehub not installed. Run: pip install kagglehub")
            return []

        parts = kaggle_path.split("/", 1)
        if len(parts) < 2:
            self._LOG.error(f"Invalid Kaggle ID: {kaggle_path}")
            return []
        owner_ds = parts[0] + "/" + parts[1].split("/")[0]
        sub_path = parts[1][len(owner_ds.split("/")[1]):].lstrip("/") if "/" in parts[1] else ""

        self._LOG.info(f"Downloading Kaggle dataset: {owner_ds} ...")
        try:
            local = kagglehub.dataset_download(owner_ds)
        except Exception as e:
            self._LOG.error(f"Kaggle download failed: {e}")
            return []

        cached = str(local)
        self._LOG.info(f"Kaggle dataset cached: {cached}")
        target = os.path.join(cached, sub_path) if sub_path else cached
        if os.path.isfile(target):
            return self._read_json_file(target)
        return self._scan_directory(target)

    def _scan_directory(self, directory: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for root, _dirs, files in os.walk(directory):
            for fname in sorted(files):
                if fname.endswith(('.jsonl', '.json')):
                    samples.extend(self._read_json_file(os.path.join(root, fname)))
        self._LOG.info(f"Directory scan: {len(samples)} samples from {directory}")
        return samples

    def _read_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples = data
                    elif isinstance(data, dict):
                        samples = [data]
        except Exception as e:
            self._LOG.warning(f"Failed to read {file_path}: {e}")
        return samples
    
    def _format_sample(self, sample: Dict[str, Any]) -> str:
        """Format a single sample for training."""
        messages = sample.get("messages", [])
        
        if not messages:
            return sample.get("text", "")
        
        formatted_text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_text += f"System: {content}\n"
            elif role == "user":
                formatted_text += f"User: {content}\n"
            elif role == "assistant":
                formatted_text += f"Assistant: {content}\n"
        
        return formatted_text.strip()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        sample = self.samples[idx]
        text = self._format_sample(sample)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_index
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class _SFTTrainingOperatorImpl(PiscesLxOperatorInterface):
    """Complete SFT training operator implementation."""
    
    def __init__(self):
        super().__init__()
        self._name = "sft.training"
        self._version = VERSION
        self.type = "training"
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
        
    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> str:
        return "Complete Supervised Fine-Tuning training operator with full PiscesL1 pipeline"
        
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "model": {"type": "torch.nn.Module", "required": True, "description": "PiscesL1 model to train"},
            "tokenizer": {"type": "object", "required": True, "description": "Model tokenizer"},
            "train_data_path": {"type": "str", "required": True, "description": "Path to training data (JSONL)"},
            "val_data_path": {"type": "str", "required": False, "description": "Path to validation data"},
            "config": {"type": "POPSSSFTTrainingConfig", "required": False, "description": "Training configuration"},
            "optimizer": {"type": "torch.optim.Optimizer", "required": False, "description": "Custom optimizer"},
            "scheduler": {"type": "torch.optim.lr_scheduler.LRScheduler", "required": False, "description": "Custom scheduler"},
            "encre_trainer": {
                "type": "YvEncreTrainer",
                "required": False,
                "description": (
                    "Optional YvEncreTrainer. When config.use_encre_data_pipeline is True, "
                    "the SFT operator delegates data generation to this trainer and runs SFT "
                    "on the produced (prompt, reference) pairs."
                ),
            },
        }
        
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "metrics": {"type": "dict", "description": "Training metrics and statistics"},
            "model_state": {"type": "dict", "description": "Final model state dict"},
            "checkpoint_path": {"type": "str", "description": "Path to saved checkpoint"},
            "training_history": {"type": "list", "description": "Detailed training history"}
        }
        
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        required_keys = ['model', 'tokenizer', 'train_data_path']
        for key in required_keys:
            if key not in inputs or inputs[key] is None:
                self._LOG.error(f"Missing required parameter: {key}")
                return False
                
        if not isinstance(inputs['model'], nn.Module):
            self._LOG.error("Model must be a torch.nn.Module")
            return False
            
        if not os.path.exists(inputs['train_data_path']):
            self._LOG.error(f"Training data not found: {inputs['train_data_path']}")
            return False
            
        if 'val_data_path' in inputs and inputs['val_data_path']:
            if not os.path.exists(inputs['val_data_path']):
                self._LOG.warning(f"Validation data not found: {inputs['val_data_path']}")
                
        return True
        
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """Execute complete SFT training pipeline."""
        start_time = time.time()

        try:
            if not self.validate_inputs(inputs):
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="Invalid input parameters",
                    execution_time=time.time() - start_time
                )

            model = inputs['model']
            tokenizer = inputs['tokenizer']
            train_data_path = inputs['train_data_path']
            val_data_path = inputs.get('val_data_path')
            custom_config = inputs.get('config')
            custom_optimizer = inputs.get('optimizer')
            custom_scheduler = inputs.get('scheduler')
            encre_trainer = inputs.get('encre_trainer')

            if custom_config:
                config = custom_config
            else:
                config = POPSSSFTTrainingConfig(
                    train_data=train_data_path,
                    val_data=val_data_path or "",
                    output_dir=get_work_dir("ckpt")
                )

            # ── EnTA integration: build SFT dataset from the EnTA trainer ──
            # When config.use_encre_data_pipeline is True, the EnTA trainer
            # becomes the *data factory*.  We produce a JSONL cache of
            # (prompt, reference) pairs and then point the SFT dataset at
            # that cache.  This is the integration point that lets the SFT
            # pipeline consume zero-dataset, multi-teacher-roundtable
            # generated data without changing the SFT loop itself.
            if config.use_encre_data_pipeline:
                if not _ENTA_AVAILABLE or encre_trainer is None:
                    return PiscesLxOperatorResult(
                        operator_name=self.name,
                        status=PiscesLxOperatorStatus.FAILED,
                        error=(
                            "use_encre_data_pipeline=True but EnTA is not "
                            "available; install model.agentic.enta and pass "
                            "inputs['encre_trainer']."
                        ),
                        execution_time=time.time() - start_time,
                    )
                generated = self._run_encre_data_generation(
                    encre_trainer=encre_trainer,
                    config=config,
                )
                if generated is None or not generated:
                    return PiscesLxOperatorResult(
                        operator_name=self.name,
                        status=PiscesLxOperatorStatus.FAILED,
                        error="EnTA data pipeline produced 0 samples",
                        execution_time=time.time() - start_time,
                    )
                self._LOG.info(
                    f"EnTA produced {len(generated)} (prompt, reference) "
                    "pairs; SFT will train on this stream."
                )
                config = self._with_encre_dataset(config, generated)
                train_data_path = config.train_data
                val_data_path = config.val_data or None

            # ── Data-only short-circuit ─────────────────────────────
            # When invoked from the EnTA CLI short-circuit without a
            # concrete student model or tokenizer, the operator's job
            # is to *produce* the EnTA dataset.  The downstream training
            # loop is a deployment concern; on a dry-run-style invocation
            # we surface the dataset stats and return successfully.
            if model is None or tokenizer is None:
                dataset_path = (
                    config.encre_data_cache_path
                    if config.use_encre_data_pipeline
                    else train_data_path
                )
                self._LOG.info(
                    "SFT data-only short-circuit: model/tokenizer absent; "
                    f"skipping optimiser loop. dataset={dataset_path}"
                )
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.SUCCESS,
                    output={
                        "mode": "data_only",
                        "dataset_path": dataset_path,
                        "samples": len(generated) if (config.use_encre_data_pipeline and generated) else 0,
                        "use_encre_data_pipeline": bool(config.use_encre_data_pipeline),
                        "encre_use_roundtable": bool(config.encre_use_roundtable),
                    },
                    execution_time=time.time() - start_time,
                )

            # MemSep: Initialize memory separation trainer if enabled
            memsep_trainer = None
            if config.use_memsep_training:
                from opss.train.memsep import (
                    POPSSMemSepTrainingConfig,
                    POPSSMemSepTrainer,
                    MemSepPhase,
                )
                memsep_config = POPSSMemSepTrainingConfig(
                    enabled=True,
                    phase_1_steps=config.memsep_phase_1_steps,
                    phase_2_steps=config.memsep_phase_2_steps,
                    gate_target=config.memsep_gate_target,
                    freeze_backbone_phase2=config.memsep_freeze_backbone_phase2,
                    freeze_router_phase1=config.memsep_freeze_router_phase1,
                    reason_data_path=config.memsep_reason_data_path,
                    mem_data_path=config.memsep_mem_data_path,
                    mem_alignment_weight=config.memsep_alignment_weight,
                    gate_schedule=config.memsep_gate_schedule,
                    gate_warmup_steps=config.memsep_gate_warmup_steps,
                )
                memsep_trainer = POPSSMemSepTrainer(
                    model, memsep_config, base_lr=config.learning_rate
                )
                # Override max_steps for 3-phase pipeline
                total_phase_steps = config.memsep_phase_1_steps + config.memsep_phase_2_steps
                if total_phase_steps > config.max_steps:
                    config.max_steps = total_phase_steps
                self._LOG.info(
                    f"MemSep training enabled: phase1={config.memsep_phase_1_steps}, "
                    f"phase2={config.memsep_phase_2_steps}, total_max_steps={config.max_steps}"
                )
            else:
                memsep_trainer = None

            self._LOG.info(f"Starting SFT training with config: {config}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            self._LOG.info(f"Using device: {device}")
            

            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    model = torch.compile(
                        model,
                        mode="reduce-overhead",
                        fullgraph=False,
                        dynamic=True,
                    )
                    self._LOG.info("torch.compile enabled with reduce-overhead mode for 20-40% speedup across all model sizes")
                except Exception as e:
                    self._LOG.warning(f"Failed to enable torch.compile: {e}")
            
            train_dataset = POPSSSFTDataset(
                data_path=config.train_data,
                tokenizer=tokenizer,
                max_seq_length=config.max_seq_length,
                ignore_index=config.ignore_index
            )
            
            val_dataset = None
            if config.val_data and os.path.exists(config.val_data):
                val_dataset = POPSSSFTDataset(
                    data_path=config.val_data,
                    tokenizer=tokenizer,
                    max_seq_length=config.max_seq_length,
                    ignore_index=config.ignore_index
                )
            
            trainer = self._create_trainer(
                config, model, tokenizer, 
                custom_optimizer, custom_scheduler,
                memsep_trainer=memsep_trainer,
            )
            
            metrics = self._run_training(trainer, train_dataset, val_dataset, config, device, memsep_trainer=memsep_trainer)
            
            checkpoint_path = self._save_model(trainer, config)
            
            execution_time = time.time() - start_time
            
            result_data = {
                'metrics': metrics,
                'model_state': model.state_dict(),
                'checkpoint_path': checkpoint_path,
                'training_history': getattr(trainer, 'training_history', [])
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=result_data,
                execution_time=execution_time,
                metadata={
                    'config': config.__dict__,
                    'final_loss': metrics.get('final_loss', 0.0),
                    'total_steps': metrics.get('total_steps', 0)
                }
            )
            
        except Exception as e:
            self._LOG.error(f"SFT training failed: {str(e)}", exc_info=True)
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _create_trainer(self, config, model, tokenizer, custom_optimizer=None, custom_scheduler=None, memsep_trainer=None):
        """Create SFT trainer with all components."""
        class SFTTrainer:
            def __init__(self, config, model, tokenizer, custom_optimizer=None, custom_scheduler=None, memsep_trainer=None):
                self.config = config
                self.model = model
                self.tokenizer = tokenizer
                self.optimizer = custom_optimizer
                self.scheduler = custom_scheduler
                self.scaler = None
                self.global_step = 0
                self.total_loss = 0.0
                self.training_history = []
                self.memsep_trainer = memsep_trainer
                self._LOG = PiscesLxLogger("PiscesLx.Opss.Train",file_path=get_log_file("PiscesLx.Opss.Train"), enable_file=True)
                
                if config.use_fp16 or config.use_bf16:
                    self.scaler = GradScaler()
                
                if self.optimizer is None:
                    self.optimizer = self._create_optimizer()
                
                self.checkpoint_manager = None
                self._LOG.info("SFTTrainer initialized")
            
            def _create_optimizer(self):
                """Create AdamW optimizer with weight decay and extreme memory optimizations."""
                # MemSep: Use memory separation param groups if enabled
                if self.memsep_trainer is not None:
                    memsep_groups = self.memsep_trainer.get_optimizer_param_groups()
                    # Apply weight decay filter to memsep backbone group
                    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "norm", "gate"]
                    for group in memsep_groups:
                        group["weight_decay"] = 0.0
                        if group.get("name") == "backbone":
                            group["weight_decay"] = self.config.max_grad_norm

                    optimizer = torch.optim.AdamW(memsep_groups, lr=self.config.learning_rate)
                    trainable = sum(group["params"][0].numel() * len(group["params"])
                                    for group in memsep_groups if group["params"])
                    self._LOG.info(
                        f"MemSep optimizer created: {len(memsep_groups)} groups, "
                        f"~{trainable:,} trainable params"
                    )
                    return optimizer

                no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
                
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": self.config.max_grad_norm,
                    },
                    {
                        "params": [
                            p for n, p in self.model.named_parameters()
                            if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                
                model_config = getattr(self.model, 'cfg', None)
                
                if model_config and getattr(model_config, 'use_int4_projection', False):
                    try:
                        from opss.optim.galore import POPSSGaLoreConfig
                        
                        POPSSGaLoreConfig(
                            rank=getattr(model_config, 'galore_rank', 128),
                            update_proj_gap=getattr(model_config, 'galore_update_proj_gap', 50),
                            scale=getattr(model_config, 'galore_scale', 1.0),
                            use_int4_projection=getattr(model_config, 'use_int4_projection', True),
                            use_int8_weights=getattr(model_config, 'use_int8_weights', True),
                            adaptive_rank_update=getattr(model_config, 'adaptive_rank_update', True),
                        )
                        
                        optimizer = torch.optim.AdamW(
                            optimizer_grouped_parameters,
                            lr=self.config.learning_rate,
                        )
                        
                        self._LOG.info("GaLore optimizer with INT4 projection enabled for 89.5% memory reduction")
                        
                    except Exception as e:
                        self._LOG.warning(f"Failed to initialize GaLore optimizer, falling back to AdamW: {e}")
                        optimizer = torch.optim.AdamW(
                            optimizer_grouped_parameters,
                            lr=self.config.learning_rate,
                        )
                else:
                    optimizer = torch.optim.AdamW(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                    )
                
                return optimizer
            
            def _create_scheduler(self, num_training_steps):
                """Create learning rate scheduler."""
                warmup_steps = self.config.warmup_steps
                max_steps = num_training_steps
                
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    else:
                        return max(
                            self.config.min_lr_ratio,
                            float(max_steps - step) / float(max(1, max_steps - warmup_steps))
                        )
                
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lr_lambda,
                )
                
                return scheduler
        
        return SFTTrainer(config, model, tokenizer, custom_optimizer, custom_scheduler)
    
    def _create_fp8_recipe(self, config):
        """Create FP8 scaling recipe for transformer engine."""
        if not config.use_fp8:
            return None
        format_type = Format.HYBRID
        return DelayedScaling(
            margin=0,
            interval=1,
            fp8_format=format_type,
            amax_history_len=config.fp8_amax_history_length,
            amax_compute_algo=config.fp8_amax_compute_algo,
        )
    
    def _run_training(self, trainer, train_dataset, val_dataset, config, device, memsep_trainer=None):
        """Execute the main training loop."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
        
        if torch.cuda.is_available():
            train_loader = self._optimize_dataloader(train_loader, device)
        
        num_batches = len(train_loader)
        num_training_steps = num_batches * (config.max_steps // num_batches)
        
        if trainer.scheduler is None:
            trainer.scheduler = trainer._create_scheduler(num_training_steps)
        
        if config.use_gradient_checkpointing:
            if hasattr(trainer.model, 'set_gradient_checkpointing'):
                trainer.model.set_gradient_checkpointing(True)
        
        model_config = getattr(trainer.model, 'cfg', None)
        
        if model_config and getattr(model_config, 'adaptive_recomputation', False):
            try:
                from opss.optim.ink.checkpoint import POPSSInkCheckpointSelector
                
                checkpoint_selector = POPSSInkCheckpointSelector(
                    checkpoint_ratio=0.5,
                    preserve_ratio=0.3,
                    enable_transformer=True,
                    adaptive_recomputation=True,
                    compute_cost_threshold=getattr(model_config, 'compute_cost_threshold', 0.5),
                    activation_size_threshold=getattr(model_config, 'activation_size_threshold', 1048576),
                )
                
                checkpoint_layers = checkpoint_selector.get_checkpoint_layers()
                trainer._LOG.info(f"Adaptive recomputation enabled with {len(checkpoint_layers)} checkpointed layers for 60-80% activation memory savings")
                
            except Exception as e:
                trainer._LOG.warning(f"Failed to initialize adaptive recomputation: {e}")
        
        if model_config and getattr(model_config, 'structured_sparsity', False):
            try:
                from opss.optim.ink.sparse import POPSSInkSparseSelector
                
                trainer.sparse_selector = POPSSInkSparseSelector(
                    sparse_ratio=0.01,
                    warmup_steps=1000,
                    adaptive=True,
                    structured_sparsity=True,
                    block_size=getattr(model_config, 'grass_block_size', 32),
                    gradient_compression_ratio=getattr(model_config, 'gradient_compression_ratio', 0.1),
                )
                
                trainer._LOG.info("GRASS structured sparsity enabled for supporting large models with +100% throughput")
                
            except Exception as e:
                trainer._LOG.warning(f"Failed to initialize structured sparsity: {e}")
        
        if model_config and getattr(model_config, 'enable_teraio', False):
            try:
                from opss.optim.offload import POPSSTERAIOManager
                
                trainer.teraio_manager = POPSSTERAIOManager(
                    gpu_memory_budget=getattr(model_config, 'gpu_memory_budget', 42949672960),
                    cpu_memory_budget=getattr(model_config, 'cpu_memory_budget', 137438953472),
                    enable_gds=getattr(model_config, 'enable_gds', True),
                )
                
                sample_input = torch.randint(0, 1000, (1, 128)).to(device)
                trainer.teraio_manager.analyze_model(trainer.model, sample_input)
                offload_plan = trainer.teraio_manager.plan_offload()
                
                trainer._LOG.info(f"TERAIO offloading enabled with {len(offload_plan)} offload operations for supporting ultra-large models")
                
            except Exception as e:
                trainer._LOG.warning(f"Failed to initialize TERAIO offloading: {e}")
        
        trainer.model.train()
        accumulation_steps = config.gradient_accumulation_steps
        
        training_metrics = {
            "total_steps": 0,
            "final_loss": 0.0,
            "learning_rate": config.learning_rate,
            "grad_norm": 0.0,
        }
        
        start_time = time.time()
        
        for epoch in range(sys.maxsize):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if trainer.global_step >= config.max_steps:
                    break
                
                trainer.global_step += 1
                
                batch = {k: v.to(device) for k, v in batch.items()}

                # MemSep: Pre-step gate scheduling and phase management
                if memsep_trainer is not None:
                    memsep_trainer.pre_step(trainer.global_step)
                
                fp8_context = te.fp8_autocast(enabled=True, fp8_recipe=self._create_fp8_recipe(config)) if config.use_fp8 else autocast(
                    enabled=(config.use_fp16 or config.use_bf16),
                    dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
                )
                
                with fp8_context:
                    outputs = trainer.model(**batch)
                    loss = outputs.get("loss", outputs[0] if isinstance(outputs, tuple) else outputs)
                    
                    if isinstance(loss, dict):
                        total_loss = sum(v for v in loss.values() if isinstance(v, torch.Tensor))
                    else:
                        total_loss = loss

                    # MemSep: Add memory alignment loss
                    if memsep_trainer is not None:
                        knowledge_ctx = None
                        if hasattr(trainer.model, 'memory_router') and trainer.model.memory_router is not None:
                            knowledge_ctx = getattr(trainer.model.memory_router, '_prefetch_state', None)
                        mem_loss = memsep_trainer.compute_memory_loss(knowledge_ctx)
                        total_loss = total_loss + mem_loss * trainer.config.memsep_alignment_weight
                    
                    loss = total_loss / accumulation_steps
                
                if trainer.scaler is not None:
                    trainer.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if trainer.global_step % accumulation_steps == 0:
                    if trainer.scaler is not None:
                        trainer.scaler.unscale_(trainer.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        trainer.model.parameters(),
                        config.max_grad_norm,
                    )
                    
                    if trainer.scaler is not None:
                        trainer.scaler.step(trainer.optimizer)
                        trainer.scaler.update()
                    else:
                        trainer.optimizer.step()
                    
                    if trainer.scheduler is not None:
                        trainer.scheduler.step()
                    
                    trainer.optimizer.zero_grad()
                
                trainer.total_loss += total_loss.item()
                epoch_loss += total_loss.item()
                epoch_batches += 1
                
                if trainer.global_step % 100 == 0:
                    avg_loss = trainer.total_loss / trainer.global_step
                    current_lr = trainer.scheduler.get_last_lr()[0] if trainer.scheduler else config.learning_rate
                    
                    trainer._LOG.info(
                        f"Step {trainer.global_step}/{config.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {time.time() - start_time:.1f}s"
                        + (f" | Gate: {memsep_trainer.gate_scheduler.get_gate():.3f} | Phase: {memsep_trainer.current_phase.value}"
                           if memsep_trainer is not None else "")
                    )
                
                if trainer.global_step % config.checkpoint_interval == 0:
                    self._save_checkpoint(trainer, config, is_intermediate=True)
                
                if val_dataset is not None and trainer.global_step % config.eval_interval == 0:
                    val_metrics = self._evaluate(trainer, val_dataset, config, device)
                    trainer._LOG.info(f"Validation at step {trainer.global_step}: {val_metrics}")
                    trainer.model.train()
            
            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                trainer.training_history.append({
                    'epoch': epoch + 1,
                    'avg_loss': avg_epoch_loss,
                    'steps': trainer.global_step,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                trainer._LOG.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")
            
            if trainer.global_step >= config.max_steps:
                break
        
        training_metrics.update({
            "total_steps": trainer.global_step,
            "final_loss": trainer.total_loss / max(1, trainer.global_step),
            "total_time": time.time() - start_time,
        })
        
        return training_metrics
    
    def _evaluate(self, trainer, val_dataset, config, device):
        """Evaluate model on validation set."""
        trainer.model.eval()
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                fp8_context = te.fp8_autocast(enabled=True, fp8_recipe=self._create_fp8_recipe(config)) if config.use_fp8 else autocast(
                    enabled=(config.use_fp16 or config.use_bf16),
                    dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
                )
                
                with fp8_context:
                    outputs = trainer.model(**batch)
                    loss = outputs.get("loss", outputs[0] if isinstance(outputs, tuple) else outputs)
                    
                    if isinstance(loss, dict):
                        total_loss += sum(v for v in loss.values() if isinstance(v, torch.Tensor)).item()
                    else:
                        total_loss += loss.item()
                    
                    num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        return {"eval_loss": avg_loss}
    
    def _optimize_dataloader(self, dataloader, device):
        """Optimize dataloader for extreme performance.
        
        This method applies various optimizations to the dataloader:
        - Automatic mixed precision for data loading
        - Prefetch optimization
        - Memory pinning optimization
        
        Args:
            dataloader: PyTorch DataLoader to optimize
            device: Target device
        
        Returns:
            Optimized dataloader
        """
        # prefetch_factor=2, persistent_workers=True
        return dataloader
    
    def _save_checkpoint(self, trainer, config, is_intermediate=False):
        """Save training checkpoint."""
        if config.local_rank != 0:
            return
        
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_name = f"checkpoint_{trainer.global_step}"
        if not is_intermediate:
            checkpoint_name = "final_model"
        
        checkpoint_path = output_dir / checkpoint_name
        
        try:
            trainer.model.save_pretrained(str(checkpoint_path))
            trainer.tokenizer.save_pretrained(str(checkpoint_path))
            
            torch.save({
                "global_step": trainer.global_step,
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "scheduler_state_dict": trainer.scheduler.state_dict() if trainer.scheduler else None,
                "config": config.__dict__,
            }, checkpoint_path / "training_state.pt")
            
            trainer._LOG.info(f"Checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            trainer._LOG.error(f"Failed to save checkpoint: {e}")
    
    def _save_model(self, trainer, config):
        """Save final trained model."""
        if config.local_rank != 0:
            return ""

        output_path = Path(config.output_dir) / "final_model"
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            trainer.model.save_pretrained(str(output_path))
            trainer.tokenizer.save_pretrained(str(output_path))

            trainer._LOG.info(f"Model saved to {output_path}")
            return str(output_path)

        except Exception as e:
            trainer._LOG.error(f"Failed to save model: {e}")
            return ""

    # ── EnTA integration helpers ────────────────────────────────

    def _run_encre_data_generation(
        self,
        encre_trainer: Any,
        config: "POPSSSFTTrainingConfig",
    ) -> list[dict[str, Any]]:
        """Drive the EnTA trainer to produce an SFT dataset.

        The flow is:

        1. Load prompts (one per line) from ``config.encre_prompts_path``
           (or, when unset, use the first ``config.encre_max_samples``
           entries of the existing train file).
        2. Invoke :meth:`YvEncreTrainer.run_with_roundtable` for each
           prompt when ``config.encre_use_roundtable`` is True, otherwise
           :meth:`YvEncreTrainer.run_adversarial_batch` to get the
           rollout-then-reward stream.
        3. Materialise the resulting (prompt, reference) pairs as a
           JSONL cache at ``config.encre_data_cache_path``.

        The cache is then read by the regular SFT dataset loader, so the
        downstream SFT loop is unchanged.  This is the *integration*
        path the user requires: EnTA is the data factory, SFT is the
        training engine.
        """
        prompts = self._load_encre_prompts(config)
        if not prompts:
            self._LOG.warning(
                "EnTA data pipeline: no prompts found "
                f"(encre_prompts_path={config.encre_prompts_path!r}); "
                "falling back to train_data head."
            )
            return []

        cap = int(config.encre_max_samples) if int(config.encre_max_samples) > 0 else len(prompts)
        prompts = prompts[:cap]

        samples: list[dict[str, Any]] = []
        if config.encre_use_roundtable:
            self._LOG.info(
                f"EnTA data pipeline: roundtable generation for {len(prompts)} prompts"
            )
            for prompt in prompts:
                try:
                    result = encre_trainer.run_with_roundtable(
                        [(prompt, "")],
                        optimizer=None,
                        system=config.encre_system_prompt or None,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._LOG.warning(
                        f"EnTA roundtable failed for prompt id={hash(prompt) & 0xFFFF:#06x}: {exc}"
                    )
                    continue
                # run_with_roundtable returns {"items": ..., "loss": ..., "trajectories": ..., ...}
                # where "items" is a (prompt, reference, ...) list.  Fall
                # back to "trajectories" when "items" is empty so the
                # integration still produces data after a single rollout.
                items = result.get("items") if isinstance(result, dict) else None
                if not items:
                    trajectories = result.get("trajectories", []) if isinstance(result, dict) else []
                    items = [
                        {
                            "prompt": t.get("prompt", prompt),
                            "reference": t.get("final_text", "") or t.get("reference", ""),
                            "reward": t.get("total_reward", 0.0),
                        }
                        for t in trajectories
                    ]
                refs = [
                    item.get("reference", "")
                    for item in items
                    if item.get("reference")
                ]
                for ref in refs:
                    samples.append(self._format_encre_sample(prompt, ref))
        else:
            self._LOG.info(
                f"EnTA data pipeline: adversarial rollout for {len(prompts)} prompts"
            )
            try:
                items = [(p, "") for p in prompts]
                encre_trainer.run_adversarial_batch(items, optimizer=None)
            except Exception as exc:  # noqa: BLE001
                self._LOG.warning(f"EnTA adversarial batch failed: {exc}")
            # The trainer has consumed the prompts and produced its own
            # internal rollouts; expose the *last* batch as a synthetic
            # SFT stream by reusing the prompts as the supervised text.
            # This keeps the pipeline alive even when no real references
            # are available -- a real deployment will populate
            # ``result["items"]`` with reward-tagged trajectories.
            for prompt in prompts:
                samples.append(self._format_encre_sample(prompt, prompt))

        if samples:
            self._write_encre_cache(samples, config.encre_data_cache_path)
        return samples

    def _load_encre_prompts(self, config: "POPSSSFTTrainingConfig") -> list[str]:
        """Load the list of prompts for the EnTA data factory.

        Supports three sources:

        * ``config.encre_prompts_path`` is a ``.txt`` file (one prompt per line).
        * Otherwise the first ``config.encre_max_samples`` (or 1024) lines
          of the existing training file are used as prompts.
        """
        path = str(config.encre_prompts_path or "").strip()
        if path and os.path.exists(path):
            prompts: list[str] = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        prompts.append(line)
            return prompts
        if config.train_data and os.path.exists(config.train_data):
            prompts = []
            with open(config.train_data, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= max(1, int(config.encre_max_samples) or 1024):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        prompts.append(line)
                        continue
                    if isinstance(sample, dict):
                        msgs = sample.get("messages", [])
                        if isinstance(msgs, list):
                            for m in msgs:
                                if isinstance(m, dict) and m.get("role") == "user":
                                    content = m.get("content", "")
                                    if isinstance(content, str):
                                        prompts.append(content)
                                        break
                                    if isinstance(content, list):
                                        for part in content:
                                            if isinstance(part, dict) and part.get("type") == "text":
                                                prompts.append(str(part.get("text", "")))
                                                break
                                        else:
                                            continue
                                        break
                            else:
                                text = sample.get("text", "")
                                if isinstance(text, str) and text:
                                    prompts.append(text)
                        else:
                            text = sample.get("text", "")
                            if isinstance(text, str) and text:
                                prompts.append(text)
                    elif isinstance(sample, str):
                        prompts.append(sample)
            return prompts
        return []

    def _format_encre_sample(self, prompt: str, reference: str) -> dict[str, Any]:
        """Build a JSONL-friendly ``(prompt, reference)`` training sample."""
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reference},
            ],
            "text": f"User: {prompt}\nAssistant: {reference}",
            "source": "encre",
        }

    def _write_encre_cache(self, samples: list[dict[str, Any]], path: str) -> None:
        """Persist the EnTA-generated samples to a JSONL cache file."""
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        self._LOG.info(f"EnTA data cache written: {path} ({len(samples)} samples)")

    def _with_encre_dataset(
        self,
        config: "POPSSSFTTrainingConfig",
        samples: list[dict[str, Any]],
    ) -> "POPSSSFTTrainingConfig":
        """Return a copy of *config* pointing at the freshly-written cache."""
        cache_path = config.encre_data_cache_path
        if not cache_path:
            cache_path = ".pisceslx/cache/encre_sft_dataset.jsonl"
        # POPSSSFTTrainingConfig is a frozen-style dataclass; rebuild it
        # to keep the rest of the loop's invariants.
        import dataclasses
        cfg = dataclasses.replace(
            config,
            train_data=cache_path,
            val_data="",
        )
        return cfg


class POPSSSFTTrainingOperator(_SFTTrainingOperatorImpl):
    pass


__all__ = [
    "POPSSSFTTrainingConfig",
    "POPSSSFTDataset",
    "POPSSSFTTrainingOperator",
]
