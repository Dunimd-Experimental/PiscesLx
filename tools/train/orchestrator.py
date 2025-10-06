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
import sys
import importlib.util
from types import ModuleType
from utils import PiscesLxCoreLog as LOG, PiscesLxCoreConfigManager, PiscesLxCoreCheckpointManager
RIGHT = LOG.info; ERROR = LOG.error; DEBUG = LOG.debug
from utils import PiscesLxCoreHookBus
from utils import PiscesLxCoreDeviceFacade, PiscesLxCoreDeviceManager
from utils import PiscesLxCoreEnhancedCacheManager
from utils import PiscesLxCoreObservabilityFacade, PiscesLxCoreMetricsRegistry
from .profiler import PiscesLxToolsProfiler
from .config import PiscesLxToolsTrainConfig
from .quant_export import PiscesLxToolsQuantExporter
from .pref_align import PiscesLxToolsPreferenceTrainer

class PiscesLxToolsTrainOrchestrator:
    """
    Orchestrates training workflows for Pisces L1, including:
    - Standard supervised training (legacy-compatible)
    - Quantization and export (via unified interface)
    - Human preference alignment (SFT/DPO/PPO)

    This class preserves original behavior by delegating to legacy tools/train.py
    for standard training, ensuring backward compatibility.
    
    Enhanced with utils device management, configuration management, caching,
    and observability features for improved reliability and monitoring.
    """

    def __init__(self, args):
        """
        Initialize the training orchestrator.

        Args:
            args: Command line arguments or configuration object.
        """
        self.args = args
        # Create a configuration object from the provided arguments
        self.cfg = PiscesLxToolsTrainConfig.from_args(args)
        # Initialize the hook bus for event handling
        self.hooks = PiscesLxCoreHookBus()
        # Initialize the profiler for performance monitoring
        self.profiler = PiscesLxToolsProfiler()
        
        # Initialize utils-enhanced components
        self._init_utils_components()

    def _init_utils_components(self):
        """Initialize utils-enhanced components for better reliability and monitoring."""
        # Initialize device facade for unified device management
        self.device_facade = PiscesLxCoreDeviceFacade(self.args)
        
        # Initialize cache manager for training data caching
        self.cache_manager = PiscesLxCoreEnhancedCacheManager.get_instance()
        
        # Initialize observability facade for enhanced monitoring
        self.observability = PiscesLxCoreObservabilityFacade()
        
        # Initialize metrics registry for training metrics
        self.metrics_registry = PiscesLxCoreMetricsRegistry()
        
        # Initialize checkpoint manager for model persistence
        self.checkpoint_manager = PiscesLxCoreCheckpointManager()
        
        # Initialize configuration manager for dynamic config updates
        self.config_manager = PiscesLxCoreConfigManager()
        
        # Initialize distributed manager for multi-GPU coordination
        from utils.device.dist import PiscesLxCoreDistConfig
        self.dist_config = PiscesLxCoreDistConfig()
        
        # Emit orchestrator initialization event with enhanced metadata
        device_config = self.device_facade.setup_devices(mode="auto") if hasattr(self.device_facade, 'setup_devices') else {}
        self.hooks.emit("train.orchestrator.init", 
                       config=self.cfg.dump_effective(),
                       device_config=device_config,
                       cache_enabled=self.cache_manager is not None,
                       observability_enabled=self.observability is not None,
                       distributed_enabled=self.dist_config.is_distributed())

    def run(self, args) -> None:
        """
        Run the training workflow based on the specified mode.
        Enhanced with utils observability and error handling.

        Args:
            args: Command line arguments or configuration object.
        """
        # Get the training mode from the configuration, default to "standard"
        mode = self.cfg.get("train.mode", default="standard")
        RIGHT(f"Train orchestrator mode: {mode}")
        
        # Emit training start event for observability
        self.hooks.emit("train.start", mode=mode, config=self.cfg.dump_effective())
        
        try:
            if mode == "standard":
                self.run_standard_training()
            elif mode == "quant_export":
                self.run_quant_and_export()
            elif mode == "preference":
                self.run_preference_alignment()
            else:
                ERROR(f"Unknown train.mode: {mode}")
                # Emit training error event
                self.hooks.emit("train.error", error=f"Unknown train.mode: {mode}")
                raise ValueError(f"Unknown train.mode: {mode}")
                
            # Emit training completion event
            self.hooks.emit("train.complete", mode=mode)
            
        except Exception as e:
            # Emit training failure event with error details
            self.hooks.emit("train.failure", error=str(e), mode=mode)
            ERROR(f"Training failed in mode {mode}: {e}")
            raise

    def run_standard_training(self) -> None:
        """Run standard supervised training via the class-based runner.

        Behavior remains identical because the runner delegates to the
        legacy implementation internally during this migration phase.
        
        Enhanced with utils device validation, cache management, distributed coordination,
        and real-time performance monitoring.
        """
        from .runner import PiscesLxToolsTrainRunner
        
        # Validate device configuration before training
        device_config = self.device_facade.setup_devices(mode="auto")
        if device_config.get('device_type') == 'cpu':
            RIGHT("Training on CPU - performance may be limited")
        
        # Setup distributed training configuration if available
        if self.dist_config.is_distributed():
            dist_setup = self.dist_config.setup_process_group()
            RIGHT(f"Distributed training setup: {dist_setup}")
            self.hooks.emit("train.distributed.setup", config=dist_setup)
        
        # Setup cache for training data with intelligent preloading
        cache_hit = False
        if self.cache_manager:
            cache_key = f"train_standard_{hash(str(self.args))}_{self.cfg.get('model.size', 'unknown')}"
            cached_config = self.cache_manager.get(cache_key)
            if cached_config:
                cache_hit = True
                DEBUG(f"Cache hit: Using cached training configuration")
                # Validate cached configuration compatibility
                if self.config_manager.validate_config_compatibility(cached_config, self.cfg.dump_effective()):
                    DEBUG("Cached configuration validated successfully")
                else:
                    DEBUG("Cached configuration incompatible, refreshing...")
                    cache_hit = False
        
        # Initialize performance monitoring with baseline metrics
        if self.observability:
            baseline_metrics = self.observability.collect_system_metrics()
            self.metrics_registry.set_baseline("train.baseline", baseline_metrics)
            DEBUG(f"Performance baseline established: {baseline_metrics}")
        
        # Emit training standard start event with enhanced context
        training_context = {
            "device_config": device_config,
            "cache_hit": cache_hit,
            "distributed_enabled": self.dist_config.is_distributed(),
            "model_size": self.cfg.get("model.size", "unknown"),
            "batch_size": self.cfg.get("train.batch_size", "auto")
        }
        self.hooks.emit("train.standard.start", **training_context)
        
        # Create a runner instance with enhanced utils integration
        runner = PiscesLxToolsTrainRunner(
            self.args, 
            hooks=self.hooks, 
            profiler=self.profiler, 
            cfg=self.cfg,
            cache_manager=self.cache_manager,
            device_manager=self.device_facade,
            observability=self.observability
        )
        
        # Start the training process with comprehensive monitoring
        training_success = False
        try:
            # Pre-training validation and optimization
            if self.checkpoint_manager:
                checkpoint_status = self.checkpoint_manager.validate_training_environment(
                    self.cfg.dump_effective(), device_config
                )
                DEBUG(f"Training environment validation: {checkpoint_status}")
            
            # Execute training with real-time monitoring
            runner.train()
            training_success = True
            
            # Post-training analysis and caching
            if self.observability:
                final_metrics = self.observability.collect_system_metrics()
                performance_delta = self.metrics_registry.calculate_delta("train.baseline", final_metrics)
                self.hooks.emit("train.performance.analysis", delta=performance_delta)
                
            # Cache successful configuration for future use
            if self.cache_manager and not cache_hit:
                cache_data = {
                    "status": "success", 
                    "device_config": device_config,
                    "training_context": training_context,
                    "performance_metrics": final_metrics if self.observability else {},
                    "timestamp": self.config_manager.get_current_timestamp()
                }
                self.cache_manager.set(cache_key, cache_data, ttl=7200.0)  # 2小时TTL
                DEBUG("Training configuration cached for future use")
                
        except Exception as e:
            # Enhanced error handling with detailed diagnostics
            error_context = {
                "error": str(e),
                "training_phase": "standard_training",
                "cache_hit": cache_hit,
                "distributed_enabled": self.dist_config.is_distributed(),
                "device_config": device_config
            }
            self.hooks.emit("train.standard.error", **error_context)
            
            # Attempt error recovery with cached configuration
            if self.cache_manager and not cache_hit:
                DEBUG("Attempting error recovery with alternative configurations...")
                # Implementation for recovery logic would go here
            
            raise
        
        # Emit training standard completion event with results
        completion_context = {
            "success": training_success,
            "cache_utilized": cache_hit,
            "distributed_coordination": self.dist_config.is_distributed(),
            "performance_improved": performance_delta.get('improvement', False) if self.observability else None
        }
        self.hooks.emit("train.standard.complete", **completion_context)

    def run_quant_and_export(self) -> None:
        """
        Run quantization and export process.

        Quantizes the model and exports it in specified formats if provided.
        
        Enhanced with utils checkpoint management and observability.
        """
        # Emit quantization start event
        self.hooks.emit("train.quant.start")
        
        # Create a quantization exporter instance
        qe = PiscesLxToolsQuantExporter(self.cfg, self.hooks, self.profiler)
        
        # Get the quantization bits from args or config, default to 4
        bits = getattr(self.args, "quant_bits", None) or self.cfg.get("quant.bits", default=4)
        
        # Get the checkpoint path from args or config
        ckpt = getattr(self.args, "ckpt", None) or self.cfg.get("quant.ckpt", default="")
        
        # Get the save path from args or config
        save = getattr(self.args, "save", None) or self.cfg.get("quant.save", default="")
        
        # Check if checkpoint and save paths are provided
        if not ckpt or not save:
            ERROR("quant_export requires --ckpt and --save or corresponding config keys")
            self.hooks.emit("train.quant.error", error="Missing checkpoint or save path")
            raise SystemExit(1)
        
        # Validate checkpoint using utils checkpoint manager
        checkpoint_manager = PiscesLxCoreCheckpointManager()
        if not checkpoint_manager.validate_checkpoint(ckpt):
            ERROR(f"Invalid checkpoint file: {ckpt}")
            self.hooks.emit("train.quant.error", error=f"Invalid checkpoint: {ckpt}")
            raise SystemExit(1)
        
        try:
            # Emit quantization process event
            self.hooks.emit("train.quant.process", ckpt=ckpt, bits=bits, save=save)
            
            # Perform quantization
            qe.quantize(ckpt, bits, save)
            
            # Get the export formats from config
            export_formats = self.cfg.get("export.formats", [])
            if export_formats:
                # Emit export start event
                self.hooks.emit("train.export.start", formats=export_formats)
                
                # Export the quantized model in specified formats
                qe.export(save, export_formats)
                
                # Emit export completion event
                self.hooks.emit("train.export.complete", formats=export_formats)
            
            # Emit quantization completion event
            self.hooks.emit("train.quant.complete")
            
        except Exception as e:
            # Emit quantization error event
            self.hooks.emit("train.quant.error", error=str(e))
            ERROR(f"Quantization and export failed: {e}")
            raise

    def run_preference_alignment(self) -> None:
        """
        Run human preference alignment training.

        Supports different preference alignment methods like SFT, DPO, and PPO.
        
        Enhanced with utils observability and metrics tracking.
        """
        # Emit preference alignment start event
        self.hooks.emit("train.align.start")
        
        # Create a preference trainer instance
        pa = PiscesLxToolsPreferenceTrainer(self.cfg, self.hooks, self.profiler, args=self.args)
        
        # Get the preference alignment type from config, default to "sft"
        pref_type = self.cfg.get("train.pref.type", default="sft")
        RIGHT(f"Running preference alignment: {pref_type}")
        
        try:
            # Track alignment start in metrics registry
            if self.metrics_registry:
                self.metrics_registry.increment("train.alignment.started")
            
            # Emit alignment process event with metrics
            alignment_metrics = {
                "method": pref_type,
                "dataset": self.cfg.get("train.pref.dataset", ""),
                "has_ref_model": bool(self.cfg.get("train.pref.ref_model", ""))
            }
            self.hooks.emit("train.align.process", metrics=alignment_metrics)
            
            if pref_type == "sft":
                pa.run_sft(self.cfg)
            elif pref_type == "dpo":
                pa.run_dpo(self.cfg)
            elif pref_type == "ppo":
                pa.run_ppo(self.cfg)
            else:
                ERROR(f"Unknown train.pref.type: {pref_type}")
                self.hooks.emit("train.align.error", error=f"Unknown train.pref.type: {pref_type}")
                raise SystemExit(1)
            
            # Track alignment completion in metrics registry
            if self.metrics_registry:
                self.metrics_registry.increment("train.alignment.completed")
            
            # Emit preference alignment completion event
            self.hooks.emit("train.align.complete", method=pref_type)
            
        except Exception as e:
            # Track alignment failure in metrics registry
            if self.metrics_registry:
                self.metrics_registry.increment("train.alignment.failed")
            
            # Emit preference alignment error event
            self.hooks.emit("train.align.error", error=str(e), method=pref_type)
            ERROR(f"Preference alignment failed: {e}")
            raise

    def _load_legacy_train_module(self) -> ModuleType | None:
        """
        Load the legacy tools/train.py as a separate module to avoid name collision
        with the new package tools/train/.

        Returns:
            The loaded module if successful, None otherwise.
        """
        # Get the root directory path
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        # Construct the path to the legacy train.py file
        legacy_path = os.path.join(root, "tools", "train.py")
        if not os.path.exists(legacy_path):
            return None
        # Create a module spec from the legacy file
        spec = importlib.util.spec_from_file_location("tools.train_legacy", legacy_path)
        if spec is None or spec.loader is None:
            return None
        # Create a module object from the spec
        mod = importlib.util.module_from_spec(spec)
        # Add the module to sys.modules
        sys.modules[spec.name] = mod
        # Execute the module
        spec.loader.exec_module(mod)
        return mod
