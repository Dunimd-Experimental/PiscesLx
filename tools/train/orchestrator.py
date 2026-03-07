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
Training Orchestrator

Central coordinator for managing all training components in the PiscesLx framework.
The orchestrator provides a unified interface for initializing, configuring, and
executing complex training workflows.

Responsibilities:
    1. Configuration Management: Parse and validate training configurations
    2. Component Initialization: Setup models, optimizers, schedulers, and data
    3. Training Execution: Coordinate training, evaluation, and checkpointing
    4. Resource Management: Handle device placement and distributed training
    5. Monitoring Integration: Connect monitoring and logging systems

Architecture:
    The orchestrator acts as a high-level controller that manages:
    - PiscesLxTrainingOperator: Core training logic
    - TrainingPipelineOperator: End-to-end pipeline management
    - Optimizer operators: GaLore, Lion, Sophia implementations
    - Quantization operators: QAT and post-training quantization
    - Multi-task operators: Task balancing and continual learning

Configuration Sources:
    Configurations can be provided as:
    - TrainingConfig object
    - Dictionary with configuration values
    - Path to JSON/YAML configuration file

Usage Examples:
    From Configuration File:
        >>> orchestrator = PiscesLxTrainOrchestrator("config.yaml")
        >>> orchestrator.initialize_training(
        ...     model_class=MyModel,
        ...     train_dataloader_factory=train_loader_fn,
        ...     val_dataloader_factory=val_loader_fn
        ... )
        >>> results = orchestrator.start_training(epochs=3)

    From Dictionary:
        >>> config = {"model_name": "gpt-7b", "max_steps": 100000}
        >>> orchestrator = PiscesLxTrainOrchestrator(config)

    Advanced Usage:
        >>> orchestrator = PiscesLxTrainOrchestrator(config)
        >>> orchestrator.initialize_training(...)
        >>> 
        >>> # Custom training loop
        >>> for epoch in range(epochs):
        ...     metrics = orchestrator.train_epoch()
        ...     if epoch % 10 == 0:
        ...         orchestrator.evaluate()
        ...         orchestrator.save_checkpoint()
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
import time
from datetime import datetime
import json
import yaml
import os
import importlib

# OPSC operator system integration
from utils.opsc.base import PiscesLxBaseOperator
from utils.opsc.interface import PiscesLxOperatorConfig
from utils.opsc.registry import PiscesLxOperatorRegistrar
from utils.dc import PiscesLxLogger

from .config import TrainingConfig
from .core import PiscesLxTrainingOperator
from .pipeline import TrainingPipelineOperator, CurriculumLearningOperator, PiscesLxMultiTaskTrainingOperator
from .watermark import TrainingWatermarkIntegrationOperator, TrainingPipelineWatermarkOperator

from utils.paths import get_cache_dir
from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Train", file_path=get_log_file("PiscesLx.Tools.Train"), enable_file=True)


@PiscesLxOperatorRegistrar()
class PiscesLxTrainOrchestrator(PiscesLxBaseOperator):
    """
    Training Orchestrator
    
    Central coordinator that manages the entire training lifecycle. Provides
    a unified interface for configuration, initialization, and execution of
    training workflows.
    
    Attributes:
        config: TrainingConfig instance with all parameters
        training_operator: Core training operator
        pipeline_operator: Pipeline management operator
        is_initialized: Whether training components are initialized
        
    Example:
        >>> orchestrator = PiscesLxTrainOrchestrator("config.yaml")
        >>> orchestrator.initialize_training(
        ...     model_class=MyModel,
        ...     train_dataloader_factory=train_fn,
        ...     val_dataloader_factory=val_fn
        ... )
        >>> results = orchestrator.start_training(epochs=3)
    """
    
    def __init__(self, config: Optional[Union[PiscesLxOperatorConfig, TrainingConfig, str, Dict[str, Any]]] = None):
        """
        Initialize training orchestrator.
        
        Args:
            config: TrainingConfig, config file path, or configuration dictionary
        """
        op_config = config if isinstance(config, PiscesLxOperatorConfig) else None
        super().__init__(op_config)

        self.train_config = self._normalize_train_config(config)
        
        self.trainer = None
        self.pipeline = None
        self.optimizers = {}
        self.quantizers = {}
        self.multitask_ops = {}
        
        self.is_initialized = False
        self.current_phase = "initialization"
        self.stage = getattr(self.train_config, 'stage', None)
        self.training_history = {
            'phases': [],
            'metrics': {},
            'timestamps': {},
            'stage_history': []
        }

        self._train_dataloader_factory: Optional[Callable] = None
        self._val_dataloader_factory: Optional[Callable] = None
        
        if self.stage:
            _LOG.info(f"PiscesLxTrainOrchestrator initialized with stage={self.stage.value}")
        else:
            _LOG.info("PiscesLxTrainOrchestrator initialized")

    def _normalize_train_config(self, config: Optional[Union[PiscesLxOperatorConfig, TrainingConfig, str, Dict[str, Any]]]) -> TrainingConfig:
        if isinstance(config, TrainingConfig):
            return config
        
        if isinstance(config, str):
            return self._load_config_from_file(config)
        
        if isinstance(config, dict):
            return TrainingConfig.from_dict(config)
        
        if isinstance(config, PiscesLxOperatorConfig):
            params = getattr(config, "parameters", {}) or {}
            cfg = params.get("training_config", None)
            if isinstance(cfg, TrainingConfig):
                return cfg
            if isinstance(cfg, str):
                return self._load_config_from_file(cfg)
            if isinstance(cfg, dict):
                return TrainingConfig.from_dict(cfg)
            cfg_path = params.get("config_path")
            if isinstance(cfg_path, str) and cfg_path:
                return self._load_config_from_file(cfg_path)
        
        return TrainingConfig()
    
    def _load_config_from_file(self, config_path: str) -> TrainingConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        if config_path.suffix.lower() == '.json':
            return TrainingConfig.load_from_json(str(config_path))
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            return TrainingConfig.load_from_yaml(str(config_path))
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def _apply_model_training_config(self, model_train_cfg: Dict[str, Any]):
        """
        Apply training configuration from model config file.
        
        This is called when no --stage is specified, using the model's
        built-in training_config as the default training parameters.
        
        Args:
            model_train_cfg: training_config dict from model config
        """
        if not model_train_cfg:
            return
        
        if "lr" in model_train_cfg:
            self.train_config.optimizer.learning_rate = model_train_cfg["lr"]
        if "learning_rate" in model_train_cfg:
            self.train_config.optimizer.learning_rate = model_train_cfg["learning_rate"]
        
        if "batch_size" in model_train_cfg:
            self.train_config.data.batch_size = model_train_cfg["batch_size"]
        
        if "seq_len" in model_train_cfg:
            self.train_config.data.sequence_length = model_train_cfg["seq_len"]
        if "sequence_length" in model_train_cfg:
            self.train_config.data.sequence_length = model_train_cfg["sequence_length"]
        
        if "accum" in model_train_cfg:
            self.train_config.gradient_accumulation_steps = model_train_cfg["accum"]
        if "gradient_accumulation_steps" in model_train_cfg:
            self.train_config.gradient_accumulation_steps = model_train_cfg["gradient_accumulation_steps"]
        
        if "max_steps" in model_train_cfg:
            self.train_config.max_steps = model_train_cfg["max_steps"]
        elif "epochs" in model_train_cfg and model_train_cfg["epochs"] > 0:
            self.train_config.max_steps = model_train_cfg["epochs"] * 1000
        
        if "weight_decay" in model_train_cfg:
            self.train_config.optimizer.weight_decay = model_train_cfg["weight_decay"]
        
        if "grad_clip" in model_train_cfg:
            gc = model_train_cfg["grad_clip"]
            if "initial_max_norm" in gc:
                self.train_config.optimizer.max_grad_norm = gc["initial_max_norm"]
            if "max_grad_norm" in gc:
                self.train_config.optimizer.max_grad_norm = gc["max_grad_norm"]
        
        if "warmup_ratio" in model_train_cfg:
            self.train_config.scheduler.warmup_ratio = model_train_cfg["warmup_ratio"]
        
        if "min_lr_ratio" in model_train_cfg:
            self.train_config.scheduler.min_lr_ratio = model_train_cfg["min_lr_ratio"]
        
        if "gradient_checkpointing" in model_train_cfg:
            self.train_config.gradient_checkpointing = model_train_cfg["gradient_checkpointing"]
        
        if "mixed_precision" in model_train_cfg:
            self.train_config.mixed_precision = model_train_cfg["mixed_precision"]
        
        if "flash_attention" in model_train_cfg:
            self.train_config.flash_attention = model_train_cfg["flash_attention"]
        
        if "loss_type" in model_train_cfg:
            self.train_config.loss_type = model_train_cfg["loss_type"]

        if "moe" in model_train_cfg and isinstance(model_train_cfg["moe"], dict):
            try:
                self.train_config.moe.update(model_train_cfg["moe"])
            except Exception:
                pass

        if "response_only_loss" in model_train_cfg:
            self.train_config.response_only_loss = model_train_cfg["response_only_loss"]
        
        if "num_workers" in model_train_cfg:
            self.train_config.data.num_workers = model_train_cfg["num_workers"]
        
        if "pin_memory" in model_train_cfg:
            self.train_config.data.pin_memory = model_train_cfg["pin_memory"]
        
        if "prefetch_factor" in model_train_cfg:
            self.train_config.data.prefetch_factor = model_train_cfg["prefetch_factor"]
        
        if "advanced_operators" in model_train_cfg:
            adv = model_train_cfg["advanced_operators"]
            if "moe_gradient" in adv:
                self.train_config.moe_gradient = adv["moe_gradient"]
            if "kfac" in adv:
                self.train_config.kfac = adv["kfac"]
            if "multitask" in adv:
                self.train_config.multitask = adv["multitask"]
            if "watermark" in adv:
                self.train_config.watermark = adv["watermark"]
            if "modality_scheduler" in adv:
                self.train_config.modality_scheduler = adv["modality_scheduler"]
        
        if "moe_gradient" in model_train_cfg:
            self.train_config.moe_gradient = model_train_cfg["moe_gradient"]
        if "kfac" in model_train_cfg:
            self.train_config.kfac = model_train_cfg["kfac"]
        if "multitask" in model_train_cfg:
            self.train_config.multitask = model_train_cfg["multitask"]
        
        # Apply GaLore configuration from model config
        if "galore_enabled" in model_train_cfg:
            self.train_config.optimizer.use_galore = model_train_cfg["galore_enabled"]
        if "galore_rank" in model_train_cfg:
            self.train_config.optimizer.galore_rank = model_train_cfg["galore_rank"]
        if "galore_update_interval" in model_train_cfg:
            self.train_config.optimizer.galore_update_proj_gap = model_train_cfg["galore_update_interval"]
        
        _LOG.info("Applied training config from model config file")

    def _apply_training_moe_overrides_to_model_config(self, model_cfg: Any) -> Any:
        moe_cfg = getattr(self.train_config, "moe", None)
        if not isinstance(moe_cfg, dict) or not moe_cfg:
            return model_cfg

        mapping = {
            "routing_temperature": "moe_routing_temperature",
            "temperature_min": "moe_temperature_min",
            "temperature_max": "moe_temperature_max",
            "expert_temperature_max": "expert_temperature_max",
            "expert_load_balance_threshold": "expert_load_balance_threshold",
        }
        for src_key, dst_key in mapping.items():
            if src_key in moe_cfg and moe_cfg[src_key] is not None:
                try:
                    setattr(model_cfg, dst_key, moe_cfg[src_key])
                except Exception:
                    pass

        return model_cfg

    def _seed_training_moe_defaults_from_model_config(self, model_cfg: Any) -> None:
        moe_cfg = getattr(self.train_config, "moe", None)
        if not isinstance(moe_cfg, dict):
            return

        mapping = {
            "moe_routing_temperature": "routing_temperature",
            "moe_temperature_min": "temperature_min",
            "moe_temperature_max": "temperature_max",
            "expert_temperature_max": "expert_temperature_max",
            "expert_load_balance_threshold": "expert_load_balance_threshold",
        }

        for src_key, dst_key in mapping.items():
            if hasattr(model_cfg, src_key):
                try:
                    val = getattr(model_cfg, src_key)
                    if val is not None:
                        moe_cfg[dst_key] = val
                except Exception:
                    pass

    def switch_stage(self, new_stage, **stage_config) -> 'PiscesLxTrainOrchestrator':
        """
        Switch to a new training stage.
        
        This method allows dynamic switching between training stages (e.g., from
        pretrain to SFT to DPO) while preserving the model weights.
        
        Args:
            new_stage: The target TrainingStage (can be string or TrainingStage enum)
            **stage_config: Additional configuration overrides for the new stage
            
        Returns:
            Self for method chaining
            
        Example:
            >>> orchestrator.switch_stage(TrainingStage.SFT, learning_rate=5e-5)
            >>> orchestrator.switch_stage("alignment_dpo", beta=0.1)
        """
        from .config import TrainingStage
        
        if isinstance(new_stage, str):
            new_stage = TrainingStage(new_stage)
        
        old_stage = self.stage
        _LOG.info(f"Switching training stage from {old_stage.value if old_stage else 'None'} to {new_stage.value}")
        
        self.stage = new_stage
        self.train_config.switch_stage(new_stage, **stage_config)
        
        if self.trainer is not None:
            self.trainer.stage = new_stage
            self.trainer.loss_type = getattr(self.train_config, 'loss_type', 'lm')
            self.trainer.response_only_loss = getattr(self.train_config, 'response_only_loss', False)
            
            if hasattr(self.train_config, 'optimizer'):
                self.trainer.initialize_optimizer()
        
        self.training_history['stage_history'].append({
            'from_stage': old_stage.value if old_stage else None,
            'to_stage': new_stage.value,
            'timestamp': datetime.now().isoformat(),
            'config': stage_config
        })
        
        _LOG.info(f"Stage switched to {new_stage.value}, config updated")
        return self

    def run(self, args) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        try:
            if hasattr(args, "__dict__"):
                params = dict(vars(args))
        except Exception:
            params = {}

        train_cfg_path = params.get("train_config")
        if isinstance(train_cfg_path, str) and train_cfg_path.strip():
            self.train_config = self._load_config_from_file(train_cfg_path.strip())

        train_mode = str(params.get("train_mode") or "standard").strip() or "standard"
        if train_mode == "standard" and params.get("rlhf"):
            train_mode = "preference"
        if train_mode == "standard" and (params.get("save") is not None or params.get("bits") is not None):
            train_mode = "quant_export"

        stage_param = params.get("stage")
        if stage_param:
            from .config import TrainingStage
            try:
                if isinstance(stage_param, str):
                    stage = TrainingStage(stage_param)
                else:
                    stage = stage_param
                self.stage = stage
                if self.stage == TrainingStage.ALIGNMENT_DPO or self.stage == TrainingStage.ALIGNMENT_PPO:
                    train_mode = "preference"
            except Exception as e:
                _LOG.warning(f"Failed to parse stage parameter: {e}")
        
        if params.get("dry_run"):
            run_id = str(params.get("run_id") or "").strip()
            if run_id:
                try:
                    from .run_reporter import PiscesLxTrainingRunReporter
                    rep = PiscesLxTrainingRunReporter(
                        run_id=run_id,
                        run_dir=params.get("run_dir"),
                        run_name=params.get("run_name"),
                        control_interval_s=float(params.get("control_interval") or 0.5),
                    )
                    rep.init({"type": "train", "mode": "dry_run", "train_mode": train_mode, "run_id": run_id})
                    rep.controller.update_state({"status": "completed", "phase": "dry_run"})
                except Exception:
                    pass
            out = {"status": "dry_run", "train_config": self.train_config.to_dict(), "train_mode": train_mode}
            try:
                print(json.dumps(out, ensure_ascii=False, indent=2))
            except Exception:
                pass
            return out

        if train_mode == "quant_export":
            return self._run_quant_export(params)
        if train_mode == "preference":
            return self._run_preference(params)

        dataset = str(params.get("dataset") or "").strip()
        dataset_list: List[str] = []
        if dataset:
            dataset_list = [dataset]
        else:
            model_txt_path = os.path.join(get_cache_dir("data_cache"), "model.txt")
            if not os.path.exists(model_txt_path):
                _LOG.error(
                    f"{model_txt_path} not found! Please create it with one dataset name per line, or use --dataset argument."
                )
                raise SystemExit(1)
            with open(model_txt_path, "r", encoding="utf-8") as f:
                dataset_list = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                ]
            if not dataset_list:
                _LOG.error(
                    f"No dataset names found in {model_txt_path}! Please use --dataset argument instead."
                )
                raise SystemExit(1)

        from model import YvConfig, YvModel
        from model.tokenizer import YvTokenizer
        from torch.utils.data import DataLoader
        from tools.data.dataset.manager import PiscesLxToolsDataDatasetManager

        if train_mode != "standard":
            _LOG.warning(f"Unsupported train_mode for CLI entrypoint: {train_mode}; falling back to standard")

        model_size = str(params.get("model_size") or "0.5B").strip()
        cand_paths = [
            Path("configs") / "model" / f"{model_size}.yaml",
            Path("configs") / f"{model_size}.yaml",
            Path(str(params.get("config") or "")) if params.get("config") else None,
        ]
        model_cfg_path = next((p for p in cand_paths if p is not None and p.exists()), None)
        if model_cfg_path is None:
            raise FileNotFoundError(f"Model config not found for model_size='{model_size}'")

        model_cfg = YvConfig.from_yaml(str(model_cfg_path))

        self._seed_training_moe_defaults_from_model_config(model_cfg)

        if hasattr(model_cfg, 'training_config') and model_cfg.training_config:
            self._apply_model_training_config(model_cfg.training_config)
        
        if self.stage is not None:
            self.train_config.load_stage_config(self.stage)
            from .config import TrainingStage
            if self.stage in [TrainingStage.ALIGNMENT_DPO, TrainingStage.ALIGNMENT_PPO, TrainingStage.ALIGNMENT_ORPO]:
                self.train_config.output_dir = f".pisceslx/ckpt"
        
        try:
            self.train_config.apply_cli_overrides(params)
        except Exception:
            pass

        model_cfg = self._apply_training_moe_overrides_to_model_config(model_cfg)
        
        seq_len = int(getattr(self.train_config.data, "sequence_length", 2048))
        if params.get("seq_len") is not None:
            try:
                seq_len = int(params["seq_len"])
                self.train_config.data.sequence_length = seq_len
            except Exception:
                pass

        tokenizer = YvTokenizer()
        pad_id = int(getattr(tokenizer, "pad_token_id", 0))

        def collate(batch):
            if not batch:
                return {"input_ids": torch.empty((0, 0), dtype=torch.long), "labels": torch.empty((0, 0), dtype=torch.long)}
            ids_list = []
            for item in batch:
                ids = item.get("input_ids")
                if ids is None:
                    ids = torch.tensor([pad_id], dtype=torch.long)
                ids_list.append(ids.long().view(-1))
            max_len = min(seq_len, max(int(x.numel()) for x in ids_list))
            input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
            labels = torch.full((len(ids_list), max_len), -100, dtype=torch.long)
            attention_mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
            for i, ids in enumerate(ids_list):
                ids = ids[:max_len]
                n = int(ids.numel())
                input_ids[i, :n] = ids
                labels[i, :n] = ids
                attention_mask[i, :n] = 1
            return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

        dm = PiscesLxToolsDataDatasetManager()

        dl_kwargs: Dict[str, Any] = {
            "batch_size": int(getattr(self.train_config.data, "batch_size", 1)),
            "shuffle": True,
            "num_workers": int(getattr(self.train_config.data, "num_workers", 0)),
            "pin_memory": bool(getattr(self.train_config.data, "pin_memory", False)),
            "drop_last": True,
            "collate_fn": collate,
        }
        if dl_kwargs["num_workers"] > 0:
            dl_kwargs["prefetch_factor"] = int(getattr(self.train_config.data, "prefetch_factor", 2))

        model_cfg = YvConfig.from_yaml(str(model_cfg_path))
        self._seed_training_moe_defaults_from_model_config(model_cfg)
        model_cfg = self._apply_training_moe_overrides_to_model_config(model_cfg)

        resume_from = str(params.get("resume_ckpt") or "").strip() or None
        epochs = 1
        if params.get("rlhf_epochs") is not None:
            try:
                epochs = int(params["rlhf_epochs"])
            except Exception:
                epochs = 1

        run_id = str(params.get("run_id") or "").strip() or None
        reporter = None
        if run_id:
            try:
                from .run_reporter import PiscesLxTrainingRunReporter
                reporter = PiscesLxTrainingRunReporter(
                    run_id=run_id,
                    run_dir=params.get("run_dir"),
                    run_name=params.get("run_name"),
                    control_interval_s=float(params.get("control_interval") or 0.5),
                )
                reporter.init(
                    {
                        "run_id": run_id,
                        "type": "train",
                        "train_mode": train_mode,
                        "model_size": model_size,
                        "datasets": list(dataset_list),
                    }
                )
            except Exception:
                reporter = None

        results: List[Dict[str, Any]] = []
        for idx, ds_name in enumerate(dataset_list):
            train_ds = dm.load(name=str(ds_name), subset=str(ds_name), split="train", max_samples=None, config=model_cfg)
            train_loader = DataLoader(train_ds, **dl_kwargs)

            if idx == 0:
                self.initialize_training(
                    model_class=YvModel,
                    train_dataloader_factory=lambda *_args, **_kwargs: train_loader,
                    val_dataloader_factory=None,
                    cfg=model_cfg,
                )
                if reporter is not None:
                    try:
                        reporter.bind(self.trainer, self.train_config)
                        self.add_training_callback(reporter.on_stage)
                    except Exception:
                        pass
            else:
                self._train_dataloader_factory = lambda *_args, **_kwargs: train_loader

            try:
                results.append(
                    {
                        "dataset": str(ds_name),
                        "result": self.start_training(epochs=epochs, resume_from=(resume_from if idx == 0 else None)),
                    }
                )
            except KeyboardInterrupt:
                _LOG.info("Training interrupted by user, stopping all datasets")
                results.append(
                    {
                        "dataset": str(ds_name),
                        "result": {"status": "interrupted", "mode": "full_stop"},
                    }
                )
                raise
            except Exception as e:
                from .pipeline import TrainingInterruption
                if isinstance(e, TrainingInterruption):
                    if e.mode == TrainingInterruption.SKIP_DATASET:
                        _LOG.info(f"Skipping dataset {ds_name}, continuing to next")
                        results.append(
                            {
                                "dataset": str(ds_name),
                                "result": {"status": "skipped", "reason": str(e.message)},
                            }
                        )
                        continue
                    else:
                        _LOG.info("Full stop requested, stopping all datasets")
                        results.append(
                            {
                                "dataset": str(ds_name),
                                "result": {"status": "interrupted", "mode": "full_stop"},
                            }
                        )
                        raise
                raise

        return {"status": "ok", "mode": "standard", "results": results}

    def _run_preference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from pathlib import Path

        ckpt_path = params.get("resume_ckpt") or params.get("ckpt") or params.get("model_path")
        ckpt_path = str(ckpt_path).strip() if ckpt_path is not None else ""
        if not ckpt_path:
            return {"status": "error", "reason": "missing_ckpt_for_preference"}

        rlhf_dataset = params.get("rlhf_dataset") or "dunimd/human_feedback"
        rlhf_epochs = int(params.get("rlhf_epochs") or 3)
        rlhf_max_samples = int(params.get("rlhf_max_samples") or 1000)
        rlhf_max_length = int(params.get("rlhf_max_length") or 512)

        model_size = str(params.get("model_size") or "0.5B").strip()
        cand_paths = [
            Path("configs") / "model" / f"{model_size}.yaml",
            Path("configs") / f"{model_size}.yaml",
            Path(str(params.get("config") or "")) if params.get("config") else None,
        ]
        model_cfg_path = next((p for p in cand_paths if p is not None and p.exists()), None)
        if model_cfg_path is None:
            return {"status": "error", "reason": f"model_config_not_found:{model_size}"}

        from model import YvConfig, YvModel
        from model.tokenizer import YvTokenizer
        from opss.train.pref_align import PPOConfig, POPSSPreferenceAlignmentOperator

        cfg = YvConfig.from_yaml(str(model_cfg_path))
        cfg = self._apply_training_moe_overrides_to_model_config(cfg)
        model = YvModel(cfg)
        try:
            raw = torch.load(ckpt_path, map_location="cpu")
            if isinstance(raw, dict):
                sd = raw.get("model_state_dict") or raw.get("model") or raw.get("state_dict") or raw
            else:
                sd = raw
            if isinstance(sd, dict):
                model.load_state_dict(sd, strict=False)
        except Exception as e:
            return {"status": "error", "reason": f"ckpt_load_failed:{e}"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        base_tokenizer = YvTokenizer()

        class _TorchTokenizerAdapter:
            def __init__(self, tok, max_length: int):
                self._tok = tok
                self._max_length = int(max_length)

            @property
            def eos_token_id(self):
                return int(getattr(self._tok, "eos_token_id", 2))

            def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 512):
                ids = self._tok.encode(text, return_tensors=None)
                if truncation:
                    ids = ids[: int(max_length)]
                attn = [1] * len(ids)
                if return_tensors == "pt":
                    return {
                        "input_ids": torch.tensor([ids], dtype=torch.long),
                        "attention_mask": torch.tensor([attn], dtype=torch.long),
                    }
                return {"input_ids": [ids], "attention_mask": [attn]}

        tokenizer = _TorchTokenizerAdapter(base_tokenizer, max_length=rlhf_max_length)

        try:
            from datasets import load_dataset
        except Exception as e:
            return {"status": "error", "reason": f"datasets_not_available:{e}"}

        dataset = load_dataset(rlhf_dataset, split="train")
        try:
            dataset = dataset.select(range(min(len(dataset), rlhf_max_samples)))
        except Exception:
            pass

        def _get_field(x: dict, keys: list, default=None):
            for k in keys:
                if k in x and x[k] is not None:
                    return x[k]
            return default

        prompts: List[str] = []
        for ex in dataset:
            prompt = _get_field(ex, ["prompt", "input", "query", "text"], default="")
            if prompt:
                prompts.append(str(prompt))
        if not prompts:
            return {"status": "error", "reason": "no_prompts_in_dataset"}

        op_cfg = PPOConfig(
            learning_rate=float(params.get("rlhf_lr") or 1e-5),
            ppo_epochs=int(rlhf_epochs),
            mini_batch_size=int(params.get("rlhf_mini_batch_size") or 1),
            max_grad_norm=float(params.get("max_grad_norm") or 1.0),
        )

        op = POPSSPreferenceAlignmentOperator()
        res = op.execute(
            {
                "method": "ppo",
                "model": model,
                "reference_model": None,
                "reward_model": None,
                "prompts": prompts,
                "tokenizer": tokenizer,
                "config": op_cfg,
            }
        )
        if not res.is_success():
            return {"status": "error", "reason": f"preference_align_failed:{res.error}"}

        out_dir = str(self.train_config.output_dir or ".pisceslx/ckpt")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "preference_ppo.pt")
        try:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "source_ckpt": ckpt_path,
                    "model_size": model_size,
                    "preference": res.output,
                },
                save_path,
            )
        except Exception as e:
            return {"status": "error", "reason": f"save_failed:{e}"}

        return {"status": "ok", "output": save_path, "mode": "preference", "metrics": res.output}

    def _run_quant_export(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ckpt_path = str(params.get("resume_ckpt") or "").strip()
        if not ckpt_path:
            return {"status": "error", "reason": "missing_resume_ckpt_for_quant_export"}

        save_path = params.get("save")
        save_path = str(save_path).strip() if save_path is not None else ""
        if not save_path:
            os.makedirs(".pisceslx", exist_ok=True)
            save_path = os.path.join(".pisceslx", "quant_export.pt")

        bits = params.get("bits")
        if bits is None:
            bits = params.get("quant_bits")
        bits = int(bits) if bits is not None else 8

        model_size = str(params.get("model_size") or "0.5B").strip()
        cand_paths = [
            Path("configs") / "model" / f"{model_size}.yaml",
            Path("configs") / f"{model_size}.yaml",
            Path(str(params.get("config") or "")) if params.get("config") else None,
        ]
        model_cfg_path = next((p for p in cand_paths if p is not None and p.exists()), None)
        if model_cfg_path is None:
            return {"status": "error", "reason": f"model_config_not_found:{model_size}"}

        from model import YvConfig, YvModel
        cfg = YvConfig.from_yaml(str(model_cfg_path))
        cfg = self._apply_training_moe_overrides_to_model_config(cfg)
        model = YvModel(cfg)

        try:
            raw = torch.load(ckpt_path, map_location="cpu")
            if isinstance(raw, dict):
                sd = raw.get("model_state_dict") or raw.get("model") or raw.get("state_dict") or raw
            else:
                sd = raw
            if isinstance(sd, dict):
                model.load_state_dict(sd, strict=False)
        except Exception as e:
            return {"status": "error", "reason": f"ckpt_load_failed:{e}"}

        model.eval()
        model.to("cpu")

        from opss.quantize.pipeline import POPSSQuantizationPipelineOperator, QuantizationPipelineConfig

        if bits not in (4, 8):
            return {"status": "error", "reason": f"unsupported_bits:{bits}"}

        method = "gptq" if bits == 4 else "smoothquant"
        qcfg = QuantizationPipelineConfig(
            method=method,
            bits=int(bits),
            group_size=int(params.get("group_size") or 128),
            calibration_dataset=str(params.get("calibration_dataset") or "wikitext"),
            calibration_samples=int(params.get("calibration_samples") or 128),
            sequence_length=int(params.get("seq_len") or 512),
            enable_sensitivity_analysis=bool(params.get("enable_sensitivity_analysis") or False),
            enable_adaptive_allocation=bool(params.get("enable_adaptive_allocation") or False),
            validate_after_quantization=False,
        )
        op = POPSSQuantizationPipelineOperator()
        res = op.execute({"model": model, "config": qcfg})
        if not res.is_success():
            return {"status": "error", "reason": f"quantize_failed:{res.error}"}

        out = res.output or {}
        qmodel = out.get("quantized_model")
        if qmodel is None:
            return {"status": "error", "reason": "quantize_failed:no_quantized_model"}

        try:
            torch.save(
                {
                    "model_state_dict": qmodel.state_dict(),
                    "quantization": out,
                    "source_ckpt": ckpt_path,
                    "model_size": model_size,
                },
                save_path,
            )
        except Exception as e:
            return {"status": "error", "reason": f"save_failed:{e}"}

        return {"status": "ok", "output": save_path, "quantization": out}
    
    def initialize_training(self, model_class: type, 
                          train_dataloader_factory: Callable,
                          val_dataloader_factory: Optional[Callable] = None,
                          **model_kwargs) -> 'PiscesLxTrainOrchestrator':
        """
        Initialize the complete training environment.
        
        Args:
            model_class: Model class to instantiate.
            train_dataloader_factory: Factory function for creating training dataloaders.
            val_dataloader_factory: Factory function for creating validation dataloaders.
            **model_kwargs: Additional keyword arguments for model initialization.
            
        Returns:
            The initialized orchestrator instance.
        """
        _LOG.info("Initializing training environment...")

        self._train_dataloader_factory = train_dataloader_factory
        self._val_dataloader_factory = val_dataloader_factory
        
        # 1. Initialize core training operator
        self.trainer = PiscesLxTrainingOperator(self.train_config)
        
        # 2. Initialize model
        model = self.trainer.initialize_model(model_class, **model_kwargs)
        
        # 3. Initialize optimizers
        self._setup_optimizers()
        
        # 4. Initialize quantization components (if enabled)
        if self.train_config.quantization.enable_quantization:
            self._setup_quantization()
        
        # 5. Initialize multi-task components (if enabled)
        if self.train_config.enable_multitask:
            self._setup_multitask()
        
        # 6. Initialize watermark components (if enabled)
        if getattr(self.train_config, 'enable_watermark', False):
            self._setup_watermark()
        
        # 7. Initialize training pipeline
        self.pipeline = TrainingPipelineOperator(self.train_config, trainer=self.trainer)
        
        # 8. Setup curriculum learning (if configured)
        if hasattr(self.train_config, 'curriculum') and self.train_config.curriculum:
            self._setup_curriculum_learning()
        
        # 9. Setup watermark pipeline (if watermark enabled)
        if getattr(self.train_config, 'enable_watermark', False):
            self._setup_watermark_pipeline()
        
        self.is_initialized = True
        self.current_phase = "ready"
        
        _LOG.info("Training environment initialization completed")
        return self
    
    def _setup_optimizers(self):
        """Setup optimizer components."""
        if self.trainer is None:
            return
        opt = self.trainer.initialize_optimizer()
        self.optimizers["torch_optimizer"] = opt
        _LOG.info("Optimizers setup completed", optimizers=list(self.optimizers.keys()))
    
    def _setup_quantization(self):
        """Setup quantization components."""
        if not getattr(self.train_config, "quantization", None) or not self.train_config.quantization.enable_quantization:
            return
        q = self.train_config.quantization
        q_info = q.to_dict() if hasattr(q, "to_dict") else (q.__dict__ if hasattr(q, "__dict__") else str(q))
        _LOG.warning("Quantization operators not wired; skipping", quantization=q_info)
    
    def _setup_multitask(self):
        """Setup multi-task learning components."""
        weights = getattr(self.train_config, "task_weights", None)
        if not isinstance(weights, dict):
            weights = {}
        self.multitask_ops["multitask"] = PiscesLxMultiTaskTrainingOperator(task_weights=weights)
        _LOG.info("Multitask components setup", multitask_ops=list(self.multitask_ops.keys()))
    def _setup_watermark(self):
        """Setup watermark components."""
        self.watermark_ops = {}
        
        # Training watermark integration
        enabled = bool(getattr(self.train_config, 'enable_content_watermark', True) or getattr(self.train_config, 'enable_weight_watermark', False))
        jurisdiction = str(getattr(self.train_config, "jurisdiction", "GLOBAL"))
        self.watermark_ops['integration'] = TrainingWatermarkIntegrationOperator(enabled=enabled, jurisdiction=jurisdiction)
        
        _LOG.info(f"Watermark components setup: {list(self.watermark_ops.keys())}")
    
    def _setup_watermark_pipeline(self):
        """Setup watermark pipeline."""
        if 'integration' in self.watermark_ops:
            self.watermark_pipeline = TrainingPipelineWatermarkOperator(enabled=True)
            _LOG.info("Watermark pipeline setup completed")
    
    def _setup_curriculum_learning(self):
        """Setup curriculum learning."""
        if hasattr(self.train_config, 'curriculum') and self.train_config.curriculum:
            self.pipeline.curriculum = CurriculumLearningOperator(
                difficulty_schedule=self.train_config.curriculum
            )
            _LOG.info("Curriculum learning setup completed")
    
    def add_training_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Add a training callback function.
        
        Args:
            callback: Callback function that receives (stage, kwargs) parameters.
        """
        if self.pipeline:
            self.pipeline.add_callback(callback)
            _LOG.info("Training callback added")

    def _execute_impl(self, inputs: Dict[str, Any], **kwargs):
        action = inputs.get("action") or inputs.get("mode") or "train"
        if action in ("train", "fit"):
            epochs = int(inputs.get("epochs", 1))
            resume_from = inputs.get("resume_from")
            return {"results": self.start_training(epochs=epochs, resume_from=resume_from)}
        if action in ("evaluate", "eval"):
            metrics = inputs.get("metrics")
            dataloader = inputs.get("dataloader") or self._get_val_dataloader()
            return {"metrics": self.evaluate_model(dataloader, metrics=metrics)}
        if action in ("status", "stats"):
            return {"status": self.get_training_status()}
        raise ValueError(f"Unsupported action: {action}")
    
    def start_training(self, epochs: int = 1, 
                      resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Start the training process.
        
        Args:
            epochs: Number of training epochs.
            resume_from: Path to checkpoint for resuming training.
            
        Returns:
            Training history dictionary.
        """
        if not self.is_initialized:
            raise RuntimeError("Training environment not initialized. Call initialize_training() first.")
        
        _LOG.info(f"Starting training for {epochs} epochs")
        self.current_phase = "training"
        self.training_history['timestamps']['start'] = datetime.now().isoformat()
        
        try:
            # Get dataloaders
            train_loader = self._get_train_dataloader()
            val_loader = self._get_val_dataloader()
            
            # Execute training
            training_results = self.pipeline.fit(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=epochs,
                resume_from=resume_from
            )
            
            # Record training history
            self.training_history['results'] = training_results
            self.training_history['timestamps']['end'] = datetime.now().isoformat()
            self.current_phase = "completed"
            
            _LOG.info("Training completed successfully")
            return training_results
            
        except Exception as e:
            self.current_phase = "failed"
            self.training_history['error'] = str(e)
            self.training_history['timestamps']['error'] = datetime.now().isoformat()
            _LOG.error(f"Training failed: {e}")
            raise
    
    def _get_train_dataloader(self):
        """Get training dataloader."""
        if self._train_dataloader_factory is not None:
            try:
                return self._train_dataloader_factory(self.train_config)
            except TypeError:
                return self._train_dataloader_factory()
        return self._build_default_dataloader(split="train")
    
    def _get_val_dataloader(self):
        """Get validation dataloader."""
        if self._val_dataloader_factory is not None:
            try:
                return self._val_dataloader_factory(self.train_config)
            except TypeError:
                return self._val_dataloader_factory()
        samples = self._load_text_samples(split="val")
        if not samples:
            _LOG.warning("No validation dataset found, skipping validation")
            return None
        return self._build_default_dataloader(split="val")

    def _build_default_dataloader(self, split: str):
        from torch.utils.data import Dataset, DataLoader

        class PiscesLxJsonlTextDataset(Dataset):
            def __init__(self, samples: List[str], seq_len: int, tokenizer: Optional[Any] = None):
                self.samples = samples
                self.seq_len = seq_len
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx: int):
                text = self.samples[idx]
                if self.tokenizer is not None:
                    enc = self.tokenizer(
                        text,
                        max_length=self.seq_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].squeeze(0)
                    attention_mask = enc["attention_mask"].squeeze(0)
                    pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
                else:
                    tokens = text.strip().split()
                    import hashlib
                    ids = []
                    for t in tokens[: self.seq_len]:
                        h = hashlib.md5(t.encode("utf-8")).digest()
                        ids.append(int.from_bytes(h[:4], "little") % 32000 + 1)
                    pad_id = 0
                    if len(ids) < self.seq_len:
                        ids = ids + [pad_id] * (self.seq_len - len(ids))
                    input_ids = torch.tensor(ids, dtype=torch.long)
                    attention_mask = (input_ids != pad_id).to(torch.long)

                labels = input_ids.clone()
                labels[labels == pad_id] = -100
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        samples = self._load_text_samples(split=split)
        if not samples:
            raise ValueError(f"No dataset samples available for split='{split}'. Configure TrainingConfig.data.datasets or pass dataloader factories.")

        tokenizer = getattr(self.trainer, "tokenizer", None) if self.trainer is not None else None
        dataset = PiscesLxJsonlTextDataset(samples, seq_len=self.train_config.data.sequence_length, tokenizer=tokenizer)
        dl_kwargs = {
            "batch_size": self.train_config.data.batch_size,
            "shuffle": (split == "train"),
            "num_workers": self.train_config.data.num_workers,
            "pin_memory": self.train_config.data.pin_memory,
            "drop_last": (split == "train"),
        }
        if self.train_config.data.num_workers > 0:
            dl_kwargs["prefetch_factor"] = self.train_config.data.prefetch_factor
        return DataLoader(dataset, **dl_kwargs)

    def _load_text_samples(self, split: str) -> List[str]:
        datasets_cfg = getattr(self.train_config.data, "datasets", []) or []
        texts: List[str] = []
        for ds in datasets_cfg:
            if not isinstance(ds, dict):
                continue
            ds_split = ds.get("split")
            if ds_split and ds_split != split:
                continue
            path = ds.get("path") or ds.get("file")
            if not path:
                continue
            p = Path(path)
            if not p.exists():
                _LOG.warning("Dataset path not found", path=str(p))
                continue
            if p.suffix.lower() in (".jsonl", ".json"):
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        text = obj.get("text") or obj.get("prompt") or obj.get("content")
                        if text:
                            texts.append(str(text))
            else:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        t = line.strip()
                        if t:
                            texts.append(t)
        return texts
    
    def evaluate_model(self, dataloader, metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            dataloader: Evaluation dataloader.
            metrics: List of evaluation metrics.
            
        Returns:
            Evaluation results dictionary.
        """
        if not self.is_initialized or not self.pipeline:
            raise RuntimeError("Training environment not properly initialized")
        
        _LOG.info("Starting model evaluation")
        self.current_phase = "evaluation"
        
        results = self.pipeline.evaluate(dataloader, metrics)
        
        self.current_phase = "ready"
        _LOG.info("Model evaluation completed")
        return results
    
    def export_trained_model(self, filepath: str, format: str = "torch") -> str:
        """
        Export the trained model.
        
        Args:
            filepath: Export file path.
            format: Export format.
            
        Returns:
            Export file path.
        """
        if not self.trainer:
            raise RuntimeError("No trained model available")
        
        export_path = Path(filepath)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.trainer.export_model(str(export_path), format)
        _LOG.info(f"Model exported to {export_path}")
        return str(export_path)
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get training status information.
        
        Returns:
            Training status dictionary.
        """
        status = {
            'phase': self.current_phase,
            'is_initialized': self.is_initialized,
            'config_summary': {
                'model_name': self.train_config.model_name,
                'max_steps': self.train_config.max_steps,
                'batch_size': self.train_config.data.batch_size,
                'learning_rate': self.train_config.optimizer.learning_rate
            },
            'components': {
                'trainer': self.trainer is not None,
                'pipeline': self.pipeline is not None,
                'optimizers': list(self.optimizers.keys()),
                'quantizers': list(self.quantizers.keys()),
                'multitask_ops': list(self.multitask_ops.keys())
            }
        }
        
        if self.trainer:
            status['training_progress'] = self.trainer.get_training_progress()
        
        return status
    
    def save_training_state(self, filepath: str):
        """
        Save training state.
        
        Args:
            filepath: State file path.
        """
        state = {
            'config': self.train_config.to_dict(),
            'training_history': self.training_history,
            'current_phase': self.current_phase,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Training state saved to {filepath}")
    
    def load_training_state(self, filepath: str):
        """
        Load training state.
        
        Args:
            filepath: State file path.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Restore configuration
        self.train_config = TrainingConfig.from_dict(state['config'])
        
        # Restore training history
        self.training_history = state.get('training_history', {})
        self.current_phase = state.get('current_phase', 'initialized')
        
        _LOG.info(f"Training state loaded from {filepath}")


