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
    - PiscesLxTrainingPipelineOperator: End-to-end pipeline management
    - Optimizer operators: GaLore, Lion, Sophia implementations
    - Quantization operators: QAT and post-training quantization
    - Multi-task operators: Task balancing and continual learning

Configuration Sources:
    Configurations can be provided as:
    - PiscesLxTrainingConfig object
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

from .config import PiscesLxTrainingConfig
from .core import PiscesLxTrainingOperator
from tools.data.dataset.raw_text import (
    PiscesLxToolsDataJsonSampleReader,
    PiscesLxToolsDataTextCollator,
    PiscesLxToolsDataRawTextDataset,
)
from .pipeline import PiscesLxTrainingPipelineOperator, PiscesLxCurriculumLearningOperator, PiscesLxMultiTaskTrainingOperator

from utils.paths import PiscesLxPaths
from utils.paths import PiscesLxPaths
_LOG = PiscesLxLogger("PiscesLx.Tools.Train", file_path=PiscesLxPaths.get_log_file("PiscesLx.Tools.Train"), enable_file=True)


@PiscesLxOperatorRegistrar()
class PiscesLxTrainOrchestrator(PiscesLxBaseOperator):
    """
    Training Orchestrator
    
    Central coordinator that manages the entire training lifecycle. Provides
    a unified interface for configuration, initialization, and execution of
    training workflows.
    
    Attributes:
        config: PiscesLxTrainingConfig instance with all parameters
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
    
    def __init__(self, config: Optional[Union[PiscesLxOperatorConfig, PiscesLxTrainingConfig, str, Dict[str, Any]]] = None):
        """
        Initialize training orchestrator.
        
        Args:
            config: PiscesLxTrainingConfig, config file path, or configuration dictionary
        """
        op_config = config if isinstance(config, PiscesLxOperatorConfig) else None
        super().__init__(op_config)

        self.train_config = self._normalize_train_config(config)
        
        self.trainer = None
        self.pipeline = None
        self.optimizers = {}
        self.quantizers = {}
        self.multitask_ops = {}
        self._teacher_provider = None
        
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
        
        self._dev_mode_manager = None
        self._dev_mode_ui = None
        
        self._init_dev_mode()
        
        if self.stage:
            _LOG.info(f"PiscesLxTrainOrchestrator initialized with stage={self.stage.value}")
        else:
            _LOG.info("PiscesLxTrainOrchestrator initialized")

    def _normalize_train_config(self, config: Optional[Union[PiscesLxOperatorConfig, PiscesLxTrainingConfig, str, Dict[str, Any]]]) -> PiscesLxTrainingConfig:
        if isinstance(config, PiscesLxTrainingConfig):
            return config
        
        if isinstance(config, str):
            return self._load_config_from_file(config)
        
        if isinstance(config, dict):
            return PiscesLxTrainingConfig.from_dict(config)
        
        if isinstance(config, PiscesLxOperatorConfig):
            params = getattr(config, "parameters", {}) or {}
            cfg = params.get("training_config", None)
            if isinstance(cfg, PiscesLxTrainingConfig):
                return cfg
            if isinstance(cfg, str):
                return self._load_config_from_file(cfg)
            if isinstance(cfg, dict):
                return PiscesLxTrainingConfig.from_dict(cfg)
            cfg_path = params.get("config_path")
            if isinstance(cfg_path, str) and cfg_path:
                return self._load_config_from_file(cfg_path)
        
        return PiscesLxTrainingConfig()
    
    def _init_dev_mode(self) -> None:
        """
        Initialize developer mode integration.
        
        This method checks if developer mode is enabled and sets up
        the necessary components for the vim-style command interface.
        The UI is started immediately with Live display to capture
        all subsequent logs.
        """
        try:
            from tools.dev.manager import PiscesLxDevModeManager
            
            self._dev_mode_manager = PiscesLxDevModeManager.get_instance()
            
            if self._dev_mode_manager.is_enabled():
                self._dev_mode_manager.start_ui()
                _LOG.info("Developer mode enabled and UI started with Live display")
            else:
                _LOG.debug("Developer mode disabled")
        except Exception as e:
            _LOG.warning(f"Failed to initialize developer mode: {e}")
            self._dev_mode_manager = None
    
    def _start_dev_mode_ui(self) -> None:
        """
        Start developer mode UI before training engine initialization.
        
        This method ensures the UI is started with Live display before
        training begins. The UI captures all subsequent logs.
        
        Note: UI is now started in _init_dev_mode() for earlier initialization.
        This method is kept for compatibility and ensures UI is ready.
        """
        if self._dev_mode_manager is None or not self._dev_mode_manager.is_enabled():
            return
        
        try:
            self._dev_mode_manager.start_ui()
            _LOG.info("Developer mode UI confirmed ready")
        except Exception as e:
            _LOG.warning(f"Failed to start developer mode UI: {e}")
    
    def _attach_dev_mode_trainer(self) -> None:
        """
        Attach developer mode to the trainer after initialization.
        
        This method connects the developer mode manager to the trainer
        instance, enabling real-time interaction with the training process.
        """
        if self._dev_mode_manager is None or not self._dev_mode_manager.is_enabled():
            return
        
        try:
            self._dev_mode_manager.attach(self.trainer)
            _LOG.info("Developer mode attached to trainer")
        except Exception as e:
            _LOG.warning(f"Failed to attach developer mode to trainer: {e}")
    
    def _detach_dev_mode(self) -> None:
        """
        Detach developer mode from the training process.
        """
        if self._dev_mode_manager is None:
            return
        
        try:
            self._dev_mode_manager.detach()
            _LOG.info("Developer mode detached")
        except Exception as e:
            _LOG.warning(f"Failed to detach developer mode: {e}")
    
    def _check_dev_mode_pause(self) -> bool:
        """
        Check if training should be paused via developer mode.
        
        Returns:
            bool: True if training should pause
        """
        if self._dev_mode_manager is None:
            return False
        return self._dev_mode_manager.is_paused()
    
    def _wait_dev_mode_resume(self) -> None:
        """
        Wait for developer mode resume signal.
        
        This method blocks until training is resumed via developer mode.
        """
        import time
        while self._check_dev_mode_pause():
            time.sleep(0.1)
    
    def _load_config_from_file(self, config_path: str) -> PiscesLxTrainingConfig:
        """Load configuration from file."""
        config_path = Path(config_path)
        if config_path.suffix.lower() == '.json':
            return PiscesLxTrainingConfig.load_from_json(str(config_path))
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            return PiscesLxTrainingConfig.load_from_yaml(str(config_path))
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    def _resolve_model_dataset_file(
        self,
        model_cfg_path: Optional[Path],
        params: Dict[str, Any],
    ) -> Optional[Path]:
        """Resolve dataset list file with model-specific priority.

        Priority:
        1. Explicit `model_txt` CLI param if provided
        2. `model.txt` next to the chosen model config
        3. `model.txt` in the model config's parent directory
        4. Legacy global `data_cache/model.txt`
        """
        candidates: List[Path] = []

        explicit = str(params.get("model_txt") or "").strip()
        if explicit:
            candidates.append(Path(explicit))

        if model_cfg_path is not None:
            candidates.append(model_cfg_path.with_name("model.txt"))
            candidates.append(model_cfg_path.parent / "model.txt")

        candidates.append(Path(PiscesLxPaths.get_cache_dir("data_cache")) / "model.txt")

        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate
            except Exception:
                continue
        return None

    def _apply_model_ink_config(self, raw_model_cfg: Dict[str, Any]) -> None:
        """Carry the model file's ``ink_optimizer`` block onto the training configuration.

        Without this the block is parsed from YAML and then dropped, so a tier cannot state
        its own optimiser state widths or coverage: see docs/audit/00 §47.
        """
        section = raw_model_cfg.get("ink_optimizer")
        if isinstance(section, dict):
            self.train_config.ink_optimizer = dict(section)

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

        # YvConfig may expose training_config as a plain dict or as a simple
        # attribute container. Normalize to a dict to make all keys apply.
        if not isinstance(model_train_cfg, dict):
            try:
                model_train_cfg = dict(vars(model_train_cfg))
            except Exception:
                try:
                    model_train_cfg = {
                        k: getattr(model_train_cfg, k)
                        for k in dir(model_train_cfg)
                        if (not k.startswith("_")) and (not callable(getattr(model_train_cfg, k, None)))
                    }
                except Exception:
                    return
        
        if "lr" in model_train_cfg:
            self.train_config.optimizer.learning_rate = model_train_cfg["lr"]
        # A stage file is authoritative over the convenience block embedded in the model
        # config: drop every key the stage already declared so the two files cannot
        # silently decide the same switch in opposite directions.
        declared = set(self.train_config.stage_declared_keys)
        if declared:
            model_train_cfg = {key: value for key, value in model_train_cfg.items()
                               if key not in declared}
            if not model_train_cfg:
                return

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

        if "max_grad_norm" in model_train_cfg:
            self.train_config.optimizer.max_grad_norm = model_train_cfg["max_grad_norm"]
        
        if "warmup_ratio" in model_train_cfg:
            self.train_config.scheduler.warmup_ratio = model_train_cfg["warmup_ratio"]

        if "min_lr_ratio" in model_train_cfg:
            self.train_config.scheduler.min_lr_ratio = model_train_cfg["min_lr_ratio"]

        if "warmup_steps" in model_train_cfg:
            try:
                self.train_config.scheduler.warmup_steps = int(model_train_cfg["warmup_steps"])
            except Exception:
                pass

        if "scheduler_name" in model_train_cfg:
            try:
                self.train_config.scheduler.name = str(model_train_cfg["scheduler_name"])
            except Exception:
                pass

        if "scheduler" in model_train_cfg and getattr(self.train_config, "scheduler", None) is not None:
            s = model_train_cfg["scheduler"]
            if not isinstance(s, dict):
                try:
                    s = dict(vars(s))
                except Exception:
                    s = {}
            try:
                if "name" in s:
                    self.train_config.scheduler.name = str(s["name"])
                if "warmup_steps" in s:
                    self.train_config.scheduler.warmup_steps = int(s["warmup_steps"])
                if "warmup_type" in s:
                    self.train_config.scheduler.warmup_type = str(s["warmup_type"])
                if "warmup_ratio" in s:
                    self.train_config.scheduler.warmup_ratio = float(s["warmup_ratio"])
                if "min_lr_ratio" in s:
                    self.train_config.scheduler.min_lr_ratio = float(s["min_lr_ratio"])
                if "decay_steps" in s:
                    self.train_config.scheduler.decay_steps = int(s["decay_steps"])
            except Exception:
                pass
        
        if "gradient_checkpointing" in model_train_cfg:
            self.train_config.gradient_checkpointing = model_train_cfg["gradient_checkpointing"]
        
        if "mixed_precision" in model_train_cfg:
            self.train_config.mixed_precision = model_train_cfg["mixed_precision"]
        
        if "flash_attention" in model_train_cfg:
            self.train_config.flash_attention = model_train_cfg["flash_attention"]

        if "packing" in model_train_cfg:
            self.train_config.packing = bool(model_train_cfg["packing"])
        
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
        
        if "distributed" in model_train_cfg:
            dist = model_train_cfg["distributed"]
            if "enabled" in dist:
                self.train_config.distributed = dist["enabled"]
            if "world_size" in dist:
                self.train_config.world_size = dist["world_size"]
            if "parallel_3d" in dist:
                self.train_config.parallel_3d = dist["parallel_3d"]

        # Nested configs for extreme memory control (QLoRA + LoRA)
        if "quantization" in model_train_cfg and getattr(self.train_config, "quantization", None) is not None:
            q = model_train_cfg["quantization"]
            if not isinstance(q, dict):
                try:
                    q = dict(vars(q))
                except Exception:
                    q = {}
            try:
                if "enabled" in q:
                    self.train_config.quantization.enable_quantization = bool(q["enabled"])
                if "enable_quantization" in q:
                    self.train_config.quantization.enable_quantization = bool(q["enable_quantization"])
                if "method" in q:
                    self.train_config.quantization.quant_method = str(q["method"])
                if "quant_method" in q:
                    self.train_config.quantization.quant_method = str(q["quant_method"])
                if "bits" in q:
                    self.train_config.quantization.bits = int(q["bits"])
                if "group_size" in q:
                    self.train_config.quantization.group_size = int(q["group_size"])
                if "symmetric" in q:
                    self.train_config.quantization.symmetric = bool(q["symmetric"])
                if "enable_fp8_linear" in q:
                    self.train_config.quantization.enable_fp8_linear = bool(q["enable_fp8_linear"])
            except Exception:
                pass

        if "lora" in model_train_cfg and getattr(self.train_config, "lora", None) is not None:
            l = model_train_cfg["lora"]
            if not isinstance(l, dict):
                try:
                    l = dict(vars(l))
                except Exception:
                    l = {}
            try:
                if "enabled" in l:
                    self.train_config.lora.enabled = bool(l["enabled"])
                if "r" in l:
                    self.train_config.lora.r = int(l["r"])
                if "lora_alpha" in l:
                    self.train_config.lora.lora_alpha = int(l["lora_alpha"])
                if "lora_dropout" in l:
                    self.train_config.lora.lora_dropout = float(l["lora_dropout"])
                if "target_modules" in l:
                    self.train_config.lora.target_modules = list(l["target_modules"]) if l["target_modules"] else None
                if "bias" in l:
                    self.train_config.lora.bias = str(l["bias"])
                if "task_type" in l:
                    self.train_config.lora.task_type = str(l["task_type"])
            except Exception:
                pass

        if "advanced_operators" in model_train_cfg:
            adv = model_train_cfg["advanced_operators"]
            if "moe_gradient" in adv:
                self.train_config.moe_gradient = adv["moe_gradient"]
            if "kfac" in adv:
                self.train_config.kfac = adv["kfac"]
            if "multitask" in adv:
                self.train_config.multitask = adv["multitask"]
            if "modality_scheduler" in adv:
                self.train_config.modality_scheduler = adv["modality_scheduler"]

        if "moe_gradient" in model_train_cfg:
            self.train_config.moe_gradient = model_train_cfg["moe_gradient"]
        if "kfac" in model_train_cfg:
            self.train_config.kfac = model_train_cfg["kfac"]
        if "multitask" in model_train_cfg:
            self.train_config.multitask = model_train_cfg["multitask"]
        if "modality_scheduler" in model_train_cfg:
            self.train_config.modality_scheduler = model_train_cfg["modality_scheduler"]

    # Stages that fit an existing base model: memory policies (4-bit base + adapters)
    # apply here. Pretraining has no base to adapt — its trunk weights are the product.
    ADAPTER_STAGES = frozenset({
        "sft", "alignment_dpo", "alignment_ppo", "alignment_orpo", "specialized",
        "rl_grpo", "rl_rlvr", "rl_dapo",
    })

    def _apply_flagship_memory_and_convergence_policy(self, model_cfg: Any, params: Dict[str, Any]) -> None:
        if model_cfg is None:
            return

        q = getattr(self.train_config, "quantization", None)
        l = getattr(self.train_config, "lora", None)
        opt = getattr(self.train_config, "optimizer", None)
        sch = getattr(self.train_config, "scheduler", None)

        # From-scratch stages must train every parameter in the configured dtype and
        # must save at the cadence the stage configuration declared.
        stage_name = str(getattr(self.stage, "value", self.stage) or "")
        base_checkpoint = str(params.get("resume_ckpt") or params.get("model_name_or_path") or "").strip()
        adapter_run = (stage_name in self.ADAPTER_STAGES) or bool(base_checkpoint)

        if adapter_run:
            if q is not None:
                q.enable_quantization = True
                q.quant_method = "nf4"
                q.bits = 4
                if getattr(q, "group_size", None) is None:
                    q.group_size = 128
            if l is not None:
                l.enabled = True

                hidden = int(getattr(model_cfg, "hidden_size", 0) or 0)
                n_layer = int(getattr(model_cfg, "n_layer", 0) or 0)
                scale = max(hidden, n_layer * 128)
                if scale <= 1024:
                    r, alpha = 32, 64
                elif scale <= 2048:
                    r, alpha = 64, 128
                else:
                    r, alpha = 128, 256

                l.r = int(getattr(l, "r", r) or r)
                if int(l.r) < r:
                    l.r = int(r)
                l.lora_alpha = int(getattr(l, "lora_alpha", alpha) or alpha)
                if int(l.lora_alpha) < alpha:
                    l.lora_alpha = int(alpha)

                targets = list(getattr(l, "target_modules", None) or [])
                if not targets:
                    targets = [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ]
                else:
                    if "k_proj" not in targets:
                        targets.append("k_proj")
                l.target_modules = targets
        else:
            # Training from scratch has nothing to adapt: a 4-bit quantised random trunk
            # with LoRA adapters on top is not a usable weight. This rule is authoritative
            # over both the stage file and any `training_config` embedded in the model
            # config, which previously re-enabled adapters after the stage was applied.
            if q is not None:
                q.enable_quantization = False
            if l is not None:
                l.enabled = False

        if opt is not None:
            # 8-bit states are a memory policy: they belong to adapter runs on a shared
            # GPU. A stage that declares plain AdamW must get plain AdamW, otherwise the
            # declared optimizer name was silently overwritten (and Ink's step semantics
            # then decide the quality of the weight).
            if adapter_run:
                opt.name = "adamw8bit"
            if getattr(opt, "use_fp4", False):
                opt.use_fp4 = False
            if getattr(opt, "use_galore", False):
                opt.use_galore = False

        if sch is not None:
            if str(getattr(sch, "name", "cosine") or "cosine").lower() not in {"cosine", "linear"}:
                sch.name = "cosine"
            # Warmup longer than the run itself never reaches the configured peak LR, so
            # the default is bounded by the declared step budget.
            max_steps = int(getattr(self.train_config, "max_steps", 0) or 0)
            warmup_cap = max(1, max_steps // 10) if max_steps > 0 else 100
            if int(getattr(sch, "warmup_steps", 0) or 0) <= 0:
                sch.warmup_steps = min(100, warmup_cap)
            elif max_steps > 0 and int(sch.warmup_steps) >= max_steps:
                sch.warmup_steps = warmup_cap
            sch.warmup_type = str(getattr(sch, "warmup_type", "exponential") or "exponential")
            if float(getattr(sch, "min_lr_ratio", 0.0) or 0.0) <= 0.0:
                sch.min_lr_ratio = 0.1

        # Only set defaults if not already explicitly configured from yaml/training_config
        if str(getattr(self.train_config, "gradient_checkpointing", None)) not in ("false", "False", False):
            self.train_config.gradient_checkpointing = True
        if str(getattr(self.train_config, "mixed_precision", "bf16") or "bf16") == "bf16":
            self.train_config.mixed_precision = "bf16"
        self.train_config.flash_attention = bool(getattr(self.train_config, "flash_attention", True))

        # The checkpoint cadence is declared by the stage configuration; a short host-tier
        # run must be able to ask for a checkpoint every few steps.
        if int(getattr(self.train_config, "save_steps", 0) or 0) <= 0:
            self.train_config.save_steps = 1000
        if int(getattr(self.train_config, "eval_steps", 0) or 0) <= 0:
            self.train_config.eval_steps = 500

        if int(getattr(self.train_config, "log_steps", 0) or 0) <= 0:
            self.train_config.log_steps = 10

        _LOG.info(
            "Flagship policy applied",
            adapter_run=adapter_run,
            quant_bits=int(getattr(self.train_config.quantization, "bits", 4) or 4),
            quant_method=str(getattr(self.train_config.quantization, "quant_method", "nf4") or "nf4"),
            lora_enabled=bool(getattr(self.train_config.lora, "enabled", False)),
            lora_r=int(getattr(self.train_config.lora, "r", 0) or 0),
            lora_alpha=int(getattr(self.train_config.lora, "lora_alpha", 0) or 0),
            optimizer=str(getattr(self.train_config.optimizer, "name", "") or ""),
            scheduler=str(getattr(self.train_config.scheduler, "name", "") or ""),
            warmup_steps=int(getattr(self.train_config.scheduler, "warmup_steps", 0) or 0),
            warmup_type=str(getattr(self.train_config.scheduler, "warmup_type", "") or ""),
            min_lr_ratio=float(getattr(self.train_config.scheduler, "min_lr_ratio", 0.0) or 0.0),
            grad_checkpointing=bool(getattr(self.train_config, "gradient_checkpointing", False)),
            mixed_precision=str(getattr(self.train_config, "mixed_precision", "") or ""),
            save_steps=int(getattr(self.train_config, "save_steps", 0) or 0),
            eval_steps=int(getattr(self.train_config, "eval_steps", 0) or 0),
        )

    def _apply_top_level_config(self, model_cfg: Any):
        """
        Apply top-level configuration from model config file.
        
        Top-level configs like galore_enabled, use_h2o_attention, use_mla are
        at the root level of the YAML file, not inside training_config.
        
        Args:
            model_cfg: YvConfig instance with all model parameters
        """
        if model_cfg is None:
            return
        
        # Apply GaLore configuration from top-level
        if hasattr(model_cfg, 'galore_enabled') and model_cfg.galore_enabled:
            self.train_config.optimizer.use_galore = True
            _LOG.info(f"GaLore enabled from top-level config: rank={getattr(model_cfg, 'galore_rank', 128)}")
        
        if hasattr(model_cfg, 'galore_rank'):
            self.train_config.optimizer.galore_rank = model_cfg.galore_rank
        if hasattr(model_cfg, 'galore_update_interval'):
            self.train_config.optimizer.galore_update_proj_gap = model_cfg.galore_update_interval
        if hasattr(model_cfg, 'galore_quantization_bits'):
            self.train_config.optimizer.galore_quantization_bits = model_cfg.galore_quantization_bits
        if hasattr(model_cfg, 'galore_lr_ratio'):
            self.train_config.optimizer.galore_lr_ratio = model_cfg.galore_lr_ratio
        if hasattr(model_cfg, 'galore_min_rank'):
            self.train_config.optimizer.galore_min_rank = model_cfg.galore_min_rank
        if hasattr(model_cfg, 'galore_max_rank'):
            self.train_config.optimizer.galore_max_rank = model_cfg.galore_max_rank
        if hasattr(model_cfg, 'galore_rank_adapt_interval'):
            self.train_config.optimizer.galore_rank_adapt_interval = model_cfg.galore_rank_adapt_interval
        if hasattr(model_cfg, 'galore_rank_adapt_threshold'):
            self.train_config.optimizer.galore_rank_adapt_threshold = model_cfg.galore_rank_adapt_threshold
        if hasattr(model_cfg, 'galore_memory_efficient'):
            self.train_config.optimizer.galore_memory_efficient = model_cfg.galore_memory_efficient
        if hasattr(model_cfg, 'galore_moe_expert_only'):
            self.train_config.optimizer.galore_moe_expert_only = model_cfg.galore_moe_expert_only
        
        # Apply FP4 training configuration from top-level
        if hasattr(model_cfg, 'use_fp4') and model_cfg.use_fp4:
            self.train_config.optimizer.use_fp4 = True
            _LOG.info(f"FP4 training enabled from top-level config: block_size={getattr(model_cfg, 'fp4_block_size', 16)}")
        
        if hasattr(model_cfg, 'fp4_block_size'):
            self.train_config.optimizer.fp4_block_size = model_cfg.fp4_block_size
        if hasattr(model_cfg, 'fp4_stochastic_rounding'):
            self.train_config.optimizer.fp4_stochastic_rounding = model_cfg.fp4_stochastic_rounding
        if hasattr(model_cfg, 'fp4_master_weights_dtype'):
            self.train_config.optimizer.fp4_master_weights_dtype = model_cfg.fp4_master_weights_dtype
        
        # Apply gradient checkpointing from top-level
        if hasattr(model_cfg, 'use_gradient_checkpointing'):
            self.train_config.gradient_checkpointing = model_cfg.use_gradient_checkpointing
        
        # Apply mixed precision from top-level (if not set in training_config)
        if hasattr(model_cfg, 'mixed_precision') and not hasattr(self.train_config, 'mixed_precision'):
            self.train_config.mixed_precision = model_cfg.mixed_precision
        
        # Log H2O and MLA status (these are used by model, not training engine)
        if hasattr(model_cfg, 'use_h2o_attention') and model_cfg.use_h2o_attention:
            _LOG.info("H2O Attention enabled in model config - will reduce KV cache memory")
        
        if hasattr(model_cfg, 'use_mla') and model_cfg.use_mla:
            _LOG.info(f"MLA enabled in model config - KV compression rank={getattr(model_cfg, 'kv_lora_rank', 512)}")
        
        _LOG.info("Applied top-level config from model config file")

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
            new_stage: The target PiscesLxTrainingStage (can be string or PiscesLxTrainingStage enum)
            **stage_config: Additional configuration overrides for the new stage
            
        Returns:
            Self for method chaining
            
        Example:
            >>> orchestrator.switch_stage(PiscesLxTrainingStage.SFT, learning_rate=5e-5)
            >>> orchestrator.switch_stage("alignment_dpo", beta=0.1)
        """
        from .config import PiscesLxTrainingStage
        
        if isinstance(new_stage, str):
            new_stage = PiscesLxTrainingStage(new_stage)
        
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
        self._explicit_train_config = False
        if isinstance(train_cfg_path, str) and train_cfg_path.strip():
            # An explicitly named stage file wins over the stage-derived default: the two
            # would otherwise both be applied and the derived one silently wins back.
            self._explicit_train_config = True
            self.train_config = self._load_config_from_file(train_cfg_path.strip())

        train_mode = str(params.get("train_mode") or "standard").strip() or "standard"
        if train_mode == "standard" and params.get("rlhf"):
            train_mode = "preference"
        if train_mode == "standard" and (params.get("save") is not None or params.get("bits") is not None):
            train_mode = "quant_export"

        enable_distill = bool(params.get("distill", False))
        teacher_model_path = params.get("teacher_model")
        teacher_name = params.get("teacher_name")
        teacher_api_key = params.get("teacher_api_key")
        teacher_type = str(params.get("teacher_type") or "local").strip() or "local"
        distill_alpha = float(params.get("distill_alpha") or 0.5)
        distill_temperature = float(params.get("distill_temperature") or 2.0)

        if enable_distill and teacher_model_path:
            _LOG.info(
                "Distillation enabled",
                teacher_model=str(teacher_model_path),
                teacher_name=str(teacher_name) if teacher_name else "default",
                teacher_api_key="***" if teacher_api_key else "none",
                teacher_type=teacher_type,
                distill_alpha=distill_alpha,
                distill_temperature=distill_temperature,
            )
            self.train_config.enable_distillation = True
            self.train_config.distill_teacher_path = str(teacher_model_path)
            self.train_config.distill_teacher_name = str(teacher_name) if teacher_name else None
            self.train_config.distill_teacher_api_key = str(teacher_api_key) if teacher_api_key else None
            self.train_config.distill_teacher_type = teacher_type
            self.train_config.distill_alpha = distill_alpha
            self.train_config.distill_temperature = distill_temperature

        stage_param = params.get("stage")
        if stage_param:
            from .config import PiscesLxTrainingStage
            try:
                if isinstance(stage_param, str):
                    stage = PiscesLxTrainingStage(stage_param)
                else:
                    stage = stage_param
                self.stage = stage
                if self.stage == PiscesLxTrainingStage.ALIGNMENT_DPO or self.stage == PiscesLxTrainingStage.ALIGNMENT_PPO:
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

        raw_model_cfg: Dict[str, Any] = {}
        try:
            with open(model_cfg_path, "r", encoding="utf-8") as f:
                raw_model_cfg = yaml.safe_load(f) or {}
        except Exception:
            raw_model_cfg = {}

        model_cfg = YvConfig.from_yaml(str(model_cfg_path))

        self._seed_training_moe_defaults_from_model_config(model_cfg)

        # Apply top-level config (galore_enabled, use_h2o_attention, etc.)
        self._apply_top_level_config(model_cfg)

        if dataset:
            dataset_list = [dataset]
        else:
            model_txt_path = self._resolve_model_dataset_file(model_cfg_path, params)
            if model_txt_path is None:
                _LOG.error(
                    "No model.txt found for this model! Please place model.txt next to the model config, "
                    "or use --dataset argument."
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
            _LOG.info(f"Using dataset list from {model_txt_path}")

        # Apply training_config from raw YAML. Do not rely on YvConfig having a training_config field,
        # because YvConfig.from_yaml filters unknown keys.
        try:
            if isinstance(raw_model_cfg, dict) and raw_model_cfg.get("training_config"):
                self._apply_model_ink_config(raw_model_cfg)
                self._apply_model_training_config(raw_model_cfg.get("training_config"))
            elif hasattr(model_cfg, "training_config") and getattr(model_cfg, "training_config"):
                self._apply_model_ink_config(getattr(model_cfg, "training_config", {}) or {})
                self._apply_model_training_config(getattr(model_cfg, "training_config"))
        except Exception:
            pass
        
        if self.stage is not None and not self._explicit_train_config:
            self.train_config.load_stage_config(self.stage)
            from .config import PiscesLxTrainingStage
            if self.stage in [PiscesLxTrainingStage.ALIGNMENT_DPO, PiscesLxTrainingStage.ALIGNMENT_PPO, PiscesLxTrainingStage.ALIGNMENT_ORPO]:
                self.train_config.output_dir = f".pisceslx/ckpt"
        
        try:
            self.train_config.apply_cli_overrides(params)
        except Exception:
            pass

        self._apply_flagship_memory_and_convergence_policy(model_cfg, params)

        model_cfg = self._apply_training_moe_overrides_to_model_config(model_cfg)
        
        if params.get("max_steps") is not None:
            self.train_config.max_steps = int(params["max_steps"])
            _LOG.info(f"Step budget overridden from the command line: max_steps={self.train_config.max_steps}")

        seq_len = int(getattr(self.train_config.data, "sequence_length", 2048))
        if params.get("seq_len") is not None:
            try:
                seq_len = int(params["seq_len"])
                self.train_config.data.sequence_length = seq_len
            except Exception:
                pass

        tokenizer = YvTokenizer()
        self._tokenizer = tokenizer
        pad_id = int(getattr(tokenizer, "pad_token_id", 0))
        use_packing = bool(getattr(self.train_config, "packing", False))


        dm = PiscesLxToolsDataDatasetManager()

        # Initialize distributed process group for multi-GPU training
        # Set by torchrun (LOCAL_RANK env) or --distributed flag
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None and not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
            world_size_env = int(os.environ.get("WORLD_SIZE", "1") or 1)
            if world_size_env <= 1:
                os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
                os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
            os.environ.setdefault("NCCL_ALGO", "Ring")
            os.environ.setdefault("NCCL_PROTO", "Simple")
            os.environ.setdefault("NCCL_MIN_NCHANNELS", "1")
            os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")
            try:
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                torch.distributed.init_process_group(backend=backend)
                if torch.cuda.is_available():
                    torch.cuda.set_device(int(local_rank))
                _LOG.info(
                    f"Distributed training initialized: backend={backend}, rank={torch.distributed.get_rank()}, "
                    f"world_size={torch.distributed.get_world_size()}, master_addr={os.environ.get('MASTER_ADDR')}:"
                    f"{os.environ.get('MASTER_PORT')}, local_rank={local_rank}"
                )
            except Exception as e:
                _LOG.warning(f"Failed to initialize distributed training: {e}")

        dl_kwargs: Dict[str, Any] = {
            "batch_size": int(getattr(self.train_config.data, "batch_size", 1)),
            "shuffle": True,
            "num_workers": int(getattr(self.train_config.data, "num_workers", 0)),
            "pin_memory": bool(getattr(self.train_config.data, "pin_memory", torch.cuda.is_available())),
            "drop_last": True,
            "collate_fn": PiscesLxToolsDataTextCollator(seq_len, pad_id, use_packing),
        }
        if dl_kwargs["num_workers"] > 0:
            dl_kwargs["prefetch_factor"] = int(getattr(self.train_config.data, "prefetch_factor", 2))
        try:
            cpu_count = os.cpu_count() or 1
            world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
            per_rank_budget = max(1, cpu_count // max(1, world_size))
            dl_kwargs["num_workers"] = min(int(dl_kwargs["num_workers"]), per_rank_budget)
            if dl_kwargs["num_workers"] == 0:
                dl_kwargs.pop("prefetch_factor", None)
        except Exception:
            pass

        # Do not reload model_cfg here; keep the same instance to avoid
        # desynchronizing applied training_config and model-level overrides.

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
            raw_samples = PiscesLxToolsDataJsonSampleReader.resolve(ds_name)
            if raw_samples is not None:
                texts = [PiscesLxToolsDataJsonSampleReader.extract_text(s) for s in raw_samples]
                _LOG.info(f"Loaded {len(texts)} samples from path: {ds_name}")
                if not texts:
                    _LOG.warning(f"No samples found in {ds_name}, skipping")
                    continue
                train_ds = PiscesLxToolsDataRawTextDataset(texts, tokenizer, seq_len)
            else:
                train_ds = dm.load(name=str(ds_name), subset=str(ds_name), split="train", max_samples=None, config=model_cfg)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                from torch.utils.data.distributed import DistributedSampler
                dl_kwargs["sampler"] = DistributedSampler(
                    train_ds,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=dl_kwargs.pop("shuffle", True),
                )
            train_loader = DataLoader(train_ds, **dl_kwargs)
            self._current_train_loader = train_loader

            # Scan dataset for modalities before model init
            from utils.modality_scanner import PiscesLxModalityScanner
            detected_modalities = PiscesLxModalityScanner.scan_dataset(train_ds)
            if detected_modalities == {'text'}:
                _LOG.info(f"Dataset {ds_name} is text-only, skipping multimodal encoders")
            else:
                _LOG.info(f"Dataset {ds_name} detected modalities: {detected_modalities}")

            if idx == 0:
                self.initialize_training(
                    model_class=YvModel,
                    train_dataloader_factory=lambda *_args, **_kwargs: train_loader,
                    val_dataloader_factory=None,
                    cfg=model_cfg,
                    modalities=detected_modalities,
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
                from .pipeline import PiscesLxTrainingInterruption
                if isinstance(e, PiscesLxTrainingInterruption):
                    if e.mode == PiscesLxTrainingInterruption.SKIP_DATASET:
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


    def _offload_model(self) -> None:
        """Offload the training engine model from GPU to free memory.

        Called between EnTA loop iterations so the next
        ``generate_batch()`` call can run without OOM.
        """
        import gc
        import torch

        if hasattr(self, "trainer") and self.trainer is not None:
            model = getattr(self.trainer, "model", None)
            if model is not None:
                model.cpu()
                _LOG.info("Training engine model offloaded to CPU")

        # Also clear any cached CUDA memory.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        _LOG.info("Memory cleaned after training batch")

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
        from opss.train.pref_align import PiscesLxPPOConfig, PiscesLxPreferenceAlignmentOperator

        cfg = YvConfig.from_yaml(str(model_cfg_path))
        self._apply_top_level_config(cfg)
        cfg = self._apply_training_moe_overrides_to_model_config(cfg)
        model = YvModel(cfg)
        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
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

        class _PiscesLxTorchTokenizerAdapter:
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

        tokenizer = _PiscesLxTorchTokenizerAdapter(base_tokenizer, max_length=rlhf_max_length)

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

        op_cfg = PiscesLxPPOConfig(
            learning_rate=float(params.get("rlhf_lr") or 1e-5),
            ppo_epochs=int(rlhf_epochs),
            mini_batch_size=int(params.get("rlhf_mini_batch_size") or 1),
            max_grad_norm=float(params.get("max_grad_norm") or 1.0),
        )

        op = PiscesLxPreferenceAlignmentOperator()
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
        self._apply_top_level_config(cfg)
        cfg = self._apply_training_moe_overrides_to_model_config(cfg)
        model = YvModel(cfg)

        try:
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
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

        from opss.quantize.pipeline import PiscesLxQuantizationPipelineOperator, PiscesLxQuantizationPipelineConfig

        if bits not in (4, 8):
            return {"status": "error", "reason": f"unsupported_bits:{bits}"}

        method = "gptq" if bits == 4 else "smoothquant"
        qcfg = PiscesLxQuantizationPipelineConfig(
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
        op = PiscesLxQuantizationPipelineOperator()
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

        # Set local_rank from environment for distributed training (torchrun sets LOCAL_RANK)
        if hasattr(self.train_config, 'local_rank'):
            local_rank_env = os.environ.get("LOCAL_RANK")
            if local_rank_env is not None:
                self.train_config.local_rank = int(local_rank_env)
        if hasattr(self.train_config, 'world_size'):
            world_size_env = os.environ.get("WORLD_SIZE")
            if world_size_env is not None:
                self.train_config.world_size = int(world_size_env)
                if int(world_size_env) > 1 and hasattr(self.train_config, 'distributed'):
                    self.train_config.distributed = True

        self._train_dataloader_factory = train_dataloader_factory
        self._val_dataloader_factory = val_dataloader_factory

        active_modalities = model_kwargs.get("modalities", {"text"})
        try:
            active_modalities = sorted(set(active_modalities or {"text"}))
        except Exception:
            active_modalities = ["text"]
        self.train_config.active_modalities = active_modalities
        self._apply_runtime_memory_policy(model_kwargs.get("cfg"), active_modalities)
        
        self._start_dev_mode_ui()
        
        self.trainer = PiscesLxTrainingOperator(self.train_config)
        
        model = self.trainer.initialize_model(model_class, **model_kwargs)
        
        if getattr(self.train_config, 'enable_distillation', False):
            self._setup_distillation()
        
        self._setup_optimizers()
        
        if self.train_config.quantization.enable_quantization:
            self._setup_quantization()
        
        if self.train_config.enable_multitask:
            self._setup_multitask()
        
        
        self.pipeline = PiscesLxTrainingPipelineOperator(self.train_config, trainer=self.trainer)
        
        if hasattr(self.train_config, 'curriculum') and self.train_config.curriculum:
            self._setup_curriculum_learning()
        
        
        self._attach_dev_mode_trainer()
        
        self.is_initialized = True
        self.current_phase = "ready"
        
        _LOG.info("Training environment initialization completed")
        return self

    def _apply_runtime_memory_policy(self, model_cfg: Any, active_modalities: List[str]) -> None:
        """Adapt per-step training cost to the active architecture on low-VRAM cards.

        Keeps the full auxiliary architecture enabled, but reduces the *simultaneous*
        activation footprint by shrinking per-device batch size and compensating with
        gradient accumulation. This preserves effective batch size far better than
        simply disabling subsystems.
        """
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
                return
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return

        cfg = model_cfg or getattr(self, "config", None)
        has_aux_arch = bool(getattr(cfg, "use_dual_inject", False) or getattr(cfg, "use_subconscious", False))
        is_multimodal = any(mod != "text" for mod in active_modalities)
        is_heavy_arch = (
            not bool(getattr(cfg, "backbone_allow_legacy_blocks", True))
            or bool(getattr(cfg, "use_mamba3", False))
            or int(getattr(cfg, "n_layer", 0) or 0) >= 16
        )
        if vram_gb > 16.5 or not (has_aux_arch or is_multimodal or is_heavy_arch):
            return

        original_batch = int(getattr(self.train_config.data, "batch_size", 1) or 1)
        original_accum = int(getattr(self.train_config, "gradient_accumulation_steps", 1) or 1)
        target_batch = 1 if (is_multimodal or is_heavy_arch) else 2
        if original_batch <= target_batch:
            return

        effective_batch = original_batch * original_accum
        new_accum = max(original_accum, (effective_batch + target_batch - 1) // target_batch)
        self.train_config.data.batch_size = target_batch
        self.train_config.gradient_accumulation_steps = new_accum

        if hasattr(self.train_config.data, "prefetch_factor"):
            self.train_config.data.prefetch_factor = min(int(getattr(self.train_config.data, "prefetch_factor", 2) or 2), 2)
        if hasattr(self.train_config.data, "num_workers"):
            # ``or 2`` turned an explicit ``num_workers: 0`` into 2 workers, which is
            # exactly the setting a low-memory host needs; only ``None`` defaults.
            requested = getattr(self.train_config.data, "num_workers", 2)
            self.train_config.data.num_workers = min(int(2 if requested is None else requested), 2)

        _LOG.info(
            "Applied low-VRAM runtime policy",
            gpu_vram_gb=round(vram_gb, 2),
            active_modalities=active_modalities,
            is_heavy_arch=is_heavy_arch,
            batch_size_before=original_batch,
            batch_size_after=target_batch,
            grad_accum_before=original_accum,
            grad_accum_after=new_accum,
        )
    
    def _setup_distillation(self):
        """Setup distillation components with teacher provider."""
        teacher_path = getattr(self.train_config, 'distill_teacher_path', None)
        teacher_name = getattr(self.train_config, 'distill_teacher_name', None)
        teacher_api_key = getattr(self.train_config, 'distill_teacher_api_key', None)
        teacher_type = getattr(self.train_config, 'distill_teacher_type', 'local')
        distill_alpha = getattr(self.train_config, 'distill_alpha', 0.5)
        distill_temperature = getattr(self.train_config, 'distill_temperature', 2.0)
        
        if not teacher_path:
            _LOG.warning("Distillation enabled but no teacher model path specified")
            return
        
        try:
            try:
                from tools.data.distill.offline_teacher_provider import (
                    PiscesLxTeacherProviderFactory,
                    PiscesLxTeacherConfig,
                    PiscesLxLocalTeacherProvider,
                    PiscesLxServerTeacherProvider,
                    PiscesLxRemoteTeacherProvider,
                )
            except ImportError as _imp_err:
                raise ImportError(
                    f"Offline distillation teacher providers unavailable from "
                    f"tools.data.distill.offline_teacher_provider: {_imp_err}. Teachers must be "
                    f"reached through offline-generated data, never inside the engine."
                ) from _imp_err
            
            provider_config = PiscesLxTeacherConfig(
                provider_type=teacher_type,
                alpha=distill_alpha,
                temperature=distill_temperature,
            )
            
            if teacher_type == "local":
                provider_config.model_path = teacher_path
                self._teacher_provider = PiscesLxLocalTeacherProvider(provider_config)
                _LOG.info(f"Local teacher provider created from: {teacher_path}")
            
            elif teacher_type == "server":
                provider_config.server_url = teacher_path
                self._teacher_provider = PiscesLxServerTeacherProvider(provider_config)
                _LOG.info(f"Server teacher provider created for: {teacher_path}")
            
            elif teacher_type == "remote":
                provider_config.base_url = teacher_path
                if teacher_name:
                    provider_config.model_name = teacher_name
                if teacher_api_key:
                    provider_config.api_key = teacher_api_key
                self._teacher_provider = PiscesLxRemoteTeacherProvider(
                    provider_config,
                    tokenizer=getattr(self, "_tokenizer", None),
                )
                _LOG.info(f"Remote API teacher provider created: base_url={teacher_path}, model={teacher_name}")
            
            else:
                _LOG.warning(f"Unknown teacher type: {teacher_type}, falling back to local")
                provider_config.model_path = teacher_path
                self._teacher_provider = PiscesLxLocalTeacherProvider(provider_config)
            
            if self.trainer is not None:
                self.trainer.set_teacher_provider(self._teacher_provider)
                _LOG.info("Teacher provider attached to trainer")
            
        except ImportError as e:
            _LOG.warning(f"Distillation provider modules not available: {e}")
        except Exception as e:
            _LOG.error(f"Failed to setup distillation: {e}")
    def _run_pre_training_phases(self, train_loader=None):
        """Run pre-training phases: knowledge store build, subconscious warmup, expert evolution."""
        self._maybe_build_knowledge_store()
        self._maybe_run_subconscious_warmup(train_loader)

    def _maybe_build_knowledge_store(self):
        """Build the optional external knowledge store if explicitly enabled."""
        model_cfg = getattr(self.trainer.model, "config", None) if self.trainer else None
        if model_cfg is None:
            return
        store_path = getattr(model_cfg, "memory_store_path", None) or getattr(model_cfg, "knowledge_store_path", None)
        if not store_path:
            return
        if os.path.isdir(store_path):
            _LOG.info(f"Knowledge store already exists at {store_path}, skipping build")
            return
        _LOG.info(f"Building knowledge store at {store_path}")
        try:
            from opss.knowledge.builder import PiscesLxKnowledgeBuilder
            cfg = {
                "store_path": store_path,
                "slots": getattr(model_cfg, "knowledge_store_slots", None) or getattr(model_cfg, "memory_knowledge_slots", 4096),
                "chunk_size": getattr(model_cfg, "knowledge_store_chunk_size", 512),
                "chunk_overlap": getattr(model_cfg, "knowledge_store_chunk_overlap", 64),
                "hidden_size": model_cfg.hidden_size,
                "index_type": getattr(model_cfg, "knowledge_store_index_type", "ivfpq"),
                "index_nlist": getattr(model_cfg, "knowledge_store_index_nlist", 4096),
                "index_m": getattr(model_cfg, "knowledge_store_index_m", 32),
            }
            builder = PiscesLxKnowledgeBuilder(cfg)
            builder.build()
            _LOG.info("Knowledge store build completed")
        except Exception as e:
            _LOG.warning(f"Failed to build knowledge store (non-fatal): {e}")

    def _maybe_run_subconscious_warmup(self, train_loader=None):
        """Warm-start the subconscious knowledge field using teacher embeddings from the data."""
        if self.trainer is None:
            return
        model = self.trainer.model
        model_cfg = getattr(model, "config", None)
        if model_cfg is None:
            return
        if not getattr(model_cfg, "use_subconscious", False) and not getattr(model_cfg, "use_dual_inject", True):
            return
        subconscious = getattr(model, "subconscious", None)
        if subconscious is None:
            return
        if train_loader is None:
            return
        _LOG.info("Starting subconscious knowledge field warm-start")
        try:
            from opss.subconscious_trainer import YvSubconsciousTrainer
            from opss.subconscious_trainer import PiscesLxSubconsciousTrainingConfig
            trainer = YvSubconsciousTrainer(subconscious, config=PiscesLxSubconsciousTrainingConfig())
            sample_batch = next(iter(train_loader))
            hidden = getattr(self.trainer, "get_hidden_states", None)
            if hidden is not None:
                teacher_embs = hidden(sample_batch)
                trainer.warm_start_knowledge_field(teacher_embs)
                _LOG.info("Subconscious knowledge field warm-start completed")
        except Exception as e:
            _LOG.warning(f"Subconscious warm-start skipped (non-fatal): {e}")

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
    def _setup_curriculum_learning(self):
        """Setup curriculum learning."""
        if hasattr(self.train_config, 'curriculum') and self.train_config.curriculum:
            self.pipeline.curriculum = PiscesLxCurriculumLearningOperator(
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
            if getattr(self.train_config, 'enable_distillation', False):
                _LOG.info("Distillation mode detected, using PiscesLxDistillationOperator")
                return self._run_distillation_training(epochs, resume_from)

            train_loader = self._get_train_dataloader()
            val_loader = self._get_val_dataloader()

            if self._check_dev_mode_pause():
                _LOG.info("Training paused via developer mode before start")
                self._wait_dev_mode_resume()

            self._run_pre_training_phases(train_loader)

            training_results = self.pipeline.fit(
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                epochs=epochs,
                resume_from=resume_from
            )

            self.training_history['results'] = training_results
            self.training_history['timestamps']['end'] = datetime.now().isoformat()
            self.current_phase = "completed"

            self._detach_dev_mode()

            _LOG.info("Training completed successfully")
            return training_results

        except Exception as e:
            self.current_phase = "failed"
            self.training_history['error'] = str(e)
            self.training_history['timestamps']['error'] = datetime.now().isoformat()
            _LOG.error(f"Training failed: {e}")
            self._detach_dev_mode()
            raise

    def _run_distillation_training(self, epochs: int, resume_from: Optional[str]) -> Dict[str, Any]:
        """Run distillation training using PiscesLxDistillationOperator."""
        _LOG.info("Starting distillation training...")

        teacher_provider = self._teacher_provider
        if teacher_provider is None:
            raise ValueError(
                "Distillation is enabled but teacher_provider is not set. "
                "Please ensure _setup_distillation() was called before start_training()."
            )

        _e_direct = None
        _e_lazy = None
        try:
            from opss.train.distill import PiscesLxDistillationConfig
        except ImportError as _e:
            _e_direct = _e
            try:
                from opss.train.distill import PiscesLxDistillationConfig
            except ImportError as _e2:
                _e_lazy = _e2
                raise ImportError(
                    "PiscesLxDistillationConfig not found. "
                    "Ensure opss/train/distill.py and opss/train/__init__.py are deployed. "
                    f"Direct import error: {_e_direct}. Lazy import error: {_e_lazy}."
                )

        distill_cfg = PiscesLxDistillationConfig(
            student_model_path=str(resume_from) if resume_from else ".pisceslx/ckpt",
            output_dir=getattr(self.train_config, 'output_dir', '.pisceslx/distill_output'),
            train_data=getattr(self.train_config.data, 'train_data_path', './data/train.jsonl'),
            val_data=getattr(self.train_config.data, 'val_data_path', './data/val.jsonl'),
            batch_size=int(getattr(self.train_config.data, 'batch_size', 4)),
            micro_batch_size=int(getattr(self.train_config.data, 'micro_batch_size', 1)),
            gradient_accumulation_steps=int(getattr(self.train_config, 'gradient_accumulation_steps', 4)),
            learning_rate=float(getattr(self.train_config.optimizer, 'learning_rate', 1e-5)),
            temperature=float(getattr(self.train_config, 'distill_temperature', 2.0)),
            alpha=float(getattr(self.train_config, 'distill_alpha', 0.5)),
            max_steps=int(getattr(self.train_config, 'max_steps', 10000)),
            max_grad_norm=float(getattr(self.train_config.optimizer, 'max_grad_norm', 1.0)),
            use_bf16=True,
            use_gradient_checkpointing=True,
        )

        from opss.train.distill_loss import PiscesLxDistillationLossConfig, PiscesLxDistillationLoss

        student_model = self.trainer.model if self.trainer else None
        if student_model is None:
            raise RuntimeError("Training engine model is not initialized. Cannot start distillation.")

        embed_attr = getattr(student_model, 'embed', None)
        if embed_attr is None:
            embed_attr = student_model.get_input_embeddings()
        model_vocab_size = embed_attr.weight.shape[0]

        loss_config = PiscesLxDistillationLossConfig(
            temperature=distill_cfg.temperature,
            alpha=distill_cfg.alpha,
            beta=distill_cfg.beta,
            gamma=distill_cfg.gamma,
            delta=distill_cfg.delta,
            epsilon=distill_cfg.epsilon,
            ignore_index=distill_cfg.ignore_index,
        )
        distill_loss_fn = PiscesLxDistillationLoss(loss_config)

        self.trainer.set_distill_context(distill_loss_fn, model_vocab_size)
        _LOG.info("Distillation context injected into training engine")

        train_loader = self._get_train_dataloader()
        val_loader = self._get_val_dataloader()

        training_results = self.pipeline.fit(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            resume_from=resume_from,
        )

        self.training_history['results'] = training_results
        self.training_history['timestamps']['end'] = datetime.now().isoformat()
        self.current_phase = "completed"
        self._detach_dev_mode()

        _LOG.info("Distillation training completed successfully")
        return training_results
    
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

        seq_len = int(getattr(self.train_config.data, "sequence_length", 0) or 0)
        batch_size = int(getattr(self.train_config.data, "batch_size", 1) or 1)
        num_workers = int(getattr(self.train_config.data, "num_workers", 0) or 0)
        pin_memory = bool(getattr(self.train_config.data, "pin_memory", False))
        prefetch_factor = int(getattr(self.train_config.data, "prefetch_factor", 2) or 2)
        use_packing = bool(getattr(self.train_config, "packing", False))
        use_pad_multiple = (
            getattr(getattr(self, "trainer", None), "device", torch.device("cpu")).type == "cuda"
            and batch_size > 1
        )
        pad_to_multiple_of = 8 if use_pad_multiple else None

        class PiscesLxJsonlTextDataset(Dataset):
            def __init__(self, samples: List[str], seq_len: int, tokenizer: Optional[Any] = None, packing: bool = False):
                self.samples = samples
                self.seq_len = seq_len
                self.tokenizer = tokenizer
                self.packing = bool(packing and self.tokenizer is not None and self.seq_len > 0)
                self.pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0) if self.tokenizer is not None else 0
                self._packed = None
                if self.packing:
                    self._packed = self._build_packed()

            def _encode_text(self, text: str) -> List[int]:
                enc = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_tensors=None
                )
                ids = enc.get("input_ids", [])
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    ids = ids[0]
                return list(ids) if ids is not None else []

            def _build_packed(self) -> List[List[int]]:
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if eos_id is None:
                    eos_id = getattr(self.tokenizer, "sep_token_id", None)
                packed = []
                buffer: List[int] = []
                for text in self.samples:
                    ids = self._encode_text(text)
                    if eos_id is not None:
                        ids = ids + [int(eos_id)]
                    if not ids:
                        continue
                    buffer.extend(ids)
                    while len(buffer) >= self.seq_len:
                        packed.append(buffer[: self.seq_len])
                        buffer = buffer[self.seq_len :]
                if buffer:
                    pad_len = self.seq_len - len(buffer)
                    packed.append(buffer + [self.pad_id] * pad_len)
                return packed

            def __len__(self):
                if self.packing and self._packed is not None:
                    return len(self._packed)
                return len(self.samples)

            def __getitem__(self, idx: int):
                if self.packing and self._packed is not None:
                    ids = self._packed[idx]
                    input_ids = torch.tensor(ids, dtype=torch.long)
                    attention_mask = (input_ids != self.pad_id).to(torch.long)
                    labels = input_ids.clone()
                    labels[labels == self.pad_id] = -100
                    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

                text = self.samples[idx]
                if self.tokenizer is not None:
                    return {"text": text}

                tokens = text.strip().split()
                import hashlib
                ids = []
                for t in tokens[: self.seq_len]:
                    h = hashlib.md5(t.encode("utf-8")).digest()
                    ids.append(int.from_bytes(h[:4], "little") % 32000 + 1)
                if len(ids) < self.seq_len:
                    ids = ids + [self.pad_id] * (self.seq_len - len(ids))
                input_ids = torch.tensor(ids, dtype=torch.long)
                attention_mask = (input_ids != self.pad_id).to(torch.long)
                labels = input_ids.clone()
                labels[labels == self.pad_id] = -100
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        samples = self._load_text_samples(split=split)
        if not samples:
            raise ValueError(f"No dataset samples available for split='{split}'. Configure PiscesLxTrainingConfig.data.datasets or pass dataloader factories.")

        tokenizer = getattr(self.trainer, "tokenizer", None) if self.trainer is not None else None
        dataset = PiscesLxJsonlTextDataset(
            samples,
            seq_len=seq_len,
            tokenizer=tokenizer,
            packing=use_packing
        )


        dl_kwargs = {
            "batch_size": batch_size,
            "shuffle": (split == "train"),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": (split == "train"),
            "collate_fn": PiscesLxToolsDataTextCollator(
                seq_len,
                int(getattr(tokenizer, "pad_token_id", 0) or 0),
                use_packing,
                tokenizer=tokenizer,
                pad_to_multiple_of=pad_to_multiple_of,
            ),
        }
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor
            dl_kwargs["persistent_workers"] = True
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
        self.train_config = PiscesLxTrainingConfig.from_dict(state['config'])
        
        # Restore training history
        self.training_history = state.get('training_history', {})
        self.current_phase = state.get('current_phase', 'initialized')
        
        _LOG.info(f"Training state loaded from {filepath}")


