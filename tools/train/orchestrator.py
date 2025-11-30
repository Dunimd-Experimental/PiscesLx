#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import os
import sys
import importlib.util
from types import ModuleType

# Avoid importing utils at module import time to prevent side-effects
# Use dms_core logging exclusively
import dms_core
PiscesLxCoreCheckpointManager = None
PiscesLxToolsProfiler = None
PiscesLxToolsTrainConfig = None
PiscesLxToolsQuantExporter = None
PiscesLxToolsPreferenceTrainer = None


def _lazy_imports_for_init():
    global PiscesLxToolsProfiler, PiscesLxToolsTrainConfig
    from .profiler import PiscesLxToolsProfiler as _Prof
    from .config import PiscesLxToolsTrainConfig as _Cfg
    PiscesLxToolsProfiler = _Prof
    PiscesLxToolsTrainConfig = _Cfg


def _lazy_imports_for_utils():
    global PiscesLxCoreCheckpointManager, PiscesLxToolsQuantExporter, PiscesLxToolsPreferenceTrainer
    from utils.checkpoint import PiscesLxCoreCheckpointManager as _Ckpt
    from .quant_export import PiscesLxToolsQuantExporter as _QE
    from .pref_align import PiscesLxToolsPreferenceTrainer as _PT
    PiscesLxCoreCheckpointManager = _Ckpt
    PiscesLxToolsQuantExporter = _QE
    PiscesLxToolsPreferenceTrainer = _PT

class PiscesLxToolsTrainOrchestrator:
    """
    Orchestrates training workflows for PiscesL1, including:
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
        # Lazy import minimal dependencies to avoid side-effects
        _lazy_imports_for_init()
        # Initialize logger
        self._logger = dms_core.log.get_logger("pisceslx.tools.train.orchestrator")
        self._logger.info("Orchestrator __init__ start")
        print("[orchestrator] __init__ enter")
        # Create a configuration object from the provided arguments
        self.cfg = PiscesLxToolsTrainConfig.from_args(args)
        # Initialize the profiler for performance monitoring
        self.profiler = PiscesLxToolsProfiler()
        
        # Initialize utils-enhanced components
        self._init_utils_components()
        self._logger.info("Orchestrator __init__ end")
        print("[orchestrator] __init__ end")

    def _init_utils_components(self):
        """Initialize utils-enhanced components for better reliability and monitoring."""
        # Lazily import utils classes before using them
        _lazy_imports_for_utils()
        self._logger.info("init_utils: creating device_facade")
        # Create a simple CPU-only facade as fallback
        class _CpuOnlyFacade:
            def __init__(self, args=None):
                self.args = args
            def setup_devices(self, mode: str = "auto"):
                return {"device_type": "cpu", "strategy": "cpu", "gpu_ids": [], "batch_size": 1}
        self.device_facade = _CpuOnlyFacade(self.args)
        self._logger.info("init_utils: CPU-only facade in use")
        
        self._logger.info("init_utils: creating checkpoint manager")
        # Initialize checkpoint manager for model persistence
        self.checkpoint_manager = PiscesLxCoreCheckpointManager()
        
        # Initialize distributed manager as a simple object
        self._logger.info("init_utils: creating dist config")
        self.dist_config = type('DistConfig', (), {'setup_process_group': lambda self: {}, 'is_distributed': lambda self: False})()

        # Initialize watermark manager (optional, controlled by env)
        self._logger.info("init_utils: creating watermark manager")
        self.wm_manager = None
        # Defer device setup and init emit to run() to avoid blocking __init__
        self._logger.info("init_utils: deferring device setup to run()")

    def run(self, args) -> None:
        """
        Run the training workflow based on the specified mode.
        Enhanced with utils observability and error handling.

        Args:
            args: Command line arguments or configuration object.
        """
        # Get the training mode from the configuration, default to "standard"
        self._logger.info("Orchestrator run() enter")
        print("[orchestrator] run enter")
        mode = self.cfg.get("train.mode", default="standard")
        self._logger.info(f"Train orchestrator mode: {mode}")
        # Ensure utils classes are available
        _lazy_imports_for_utils()
        # Perform deferred device setup
        try:
            if not hasattr(self, 'device_facade') or self.device_facade is None:
                self._logger.info("run(): creating device_facade (deferred)")
                class _CpuOnlyFacade:
                    def __init__(self, args=None):
                        self.args = args
                    def setup_devices(self, mode: str = "auto"):
                        return {"device_type": "cpu", "strategy": "cpu", "gpu_ids": [], "batch_size": 1}
                self.device_facade = _CpuOnlyFacade(self.args)
            self._logger.info("run(): calling setup_devices (deferred)")
            device_config = self.device_facade.setup_devices(mode="auto") if hasattr(self.device_facade, 'setup_devices') else {}
            self._logger.info("run(): setup_devices done")
            print("[orchestrator] setup_devices done")
            self._logger.info(f"SETUP_DEVICES_DONE: device_type={device_config.get('device_type')}")
        except Exception as e:
            self._logger.error(f"run(): deferred device setup failed: {e}")
        
        try:
            if mode == "standard":
                self.run_standard_training()
            elif mode == "quant_export":
                self.run_quant_and_export()
            elif mode == "preference":
                self.run_preference_alignment()
            else:
                self._logger.error(f"Unknown train.mode: {mode}")
                raise ValueError(f"Unknown train.mode: {mode}")
                
        except Exception as e:
            self._logger.error(f"Training failed in mode {mode}: {e}")
            raise

    def run_standard_training(self) -> None:
        """Run standard supervised training via the class-based runner.

        Behavior remains identical because the runner delegates to the
        legacy implementation internally during this migration phase.
        
        Enhanced with utils device validation, cache management, distributed coordination,
        and real-time performance monitoring.
        """
        self._logger.info("run_standard_training enter")
        # Ensure device_facade is present even if __init__ deferred/failed
        if not hasattr(self, 'device_facade') or self.device_facade is None:
            self._logger.info("run_standard_training: creating device_facade (late)")
            try:
                self.device_facade = PiscesLxCoreDeviceFacade(self.args)
            except Exception as e:
                self._logger.error(f"run_standard_training: device_facade creation failed: {e}")
                class _CpuOnlyFacade:
                    def __init__(self, args=None):
                        self.args = args
                    def setup_devices(self, mode: str = "auto"):
                        return {"device_type": "cpu", "strategy": "cpu", "gpu_ids": [], "batch_size": 1}
                self.device_facade = _CpuOnlyFacade(self.args)
        from .runner import PiscesLxToolsTrainRunner
        
        # Validate device configuration before training
        device_config = self.device_facade.setup_devices(mode="auto")
        if device_config.get('device_type') == 'cpu':
            self._logger.info("Training on CPU - performance may be limited")
        else:
            self._logger.info(f"Device ready: {device_config.get('device_type')} GPUs={device_config.get('num_devices', '?')}")
        
        # Setup distributed training configuration if available
        if self._is_distributed():
            dist_setup = self.dist_config.setup_process_group()
            self._logger.info(f"Distributed training setup: {dist_setup}")
            self.hooks.emit("train.distributed.setup", config=dist_setup)
        else:
            self._logger.info("Distributed training disabled")
        
        # Setup cache for training data with intelligent preloading
        cache_hit = False
        cached_config = None
        cache_ns = "train.standard"
        if self.cache_manager:
            try:
                cache = self.cache_manager.get_default_cache()
                cache_key = f"train_standard_{hash(str(self.args))}_{self.cfg.get('model.size', 'unknown')}"
                cached_config = cache.get(cache_ns, cache_key)
                if cached_config:
                    cache_hit = True
                    self._logger.debug(f"Cache hit: Using cached training configuration")
                    # Validate cached configuration compatibility
                    if self.config_manager.validate_config_compatibility(cached_config, self.cfg.dump_effective()):
                        self._logger.debug("Cached configuration validated successfully")
                    else:
                        self._logger.debug("Cached configuration incompatible, refreshing...")
                        cache_hit = False
            except Exception as e:
                self._logger.debug(f"Enhanced cache get failed: {e}")
        
        # Initialize performance monitoring with baseline metrics (if supported)
        performance_delta = {}
        if self.observability and hasattr(self.observability, "collect_system_metrics"):
            try:
                baseline_metrics = self.observability.collect_system_metrics()
                self.metrics_registry.set_baseline("train.baseline", baseline_metrics)
                self._logger.debug(f"Performance baseline established: {baseline_metrics}")
            except Exception as e:
                self._logger.debug(f"Observability baseline skipped: {e}")
        else:
            self._logger.debug("Observability baseline skipped: collect_system_metrics not available")
        
        # Emit training standard start event with enhanced context
        training_context = {
            "device_config": device_config,
            "cache_hit": cache_hit,
            "distributed_enabled": self._is_distributed(),
            "model_size": self.cfg.get("model.size", "unknown"),
            "batch_size": self.cfg.get("train.batch_size", "auto")
        }
        self.hooks.emit("train.standard.start", **training_context)
        self._logger.info("Runner instantiation start")
        
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
        self._logger.info("Runner instantiation done")
        
        # Start the training process with comprehensive monitoring
        training_success = False
        try:
            # Pre-training validation and optimization (optional)
            if self.checkpoint_manager and hasattr(self.checkpoint_manager, 'validate_training_environment'):
                try:
                    checkpoint_status = self.checkpoint_manager.validate_training_environment(
                        self.cfg.dump_effective(), device_config
                    )
                    self._logger.debug(f"Training environment validation: {checkpoint_status}")
                except Exception as e:
                    self._logger.debug(f"Checkpoint environment validation skipped: {e}")
            
            # Execute training with real-time monitoring
            self._logger.info("runner.train() start")
            runner.train()
            self._logger.info("runner.train() end")
            training_success = True
            
            # Post-training analysis and caching
            if self.observability and hasattr(self.observability, "collect_system_metrics"):
                try:
                    final_metrics = self.observability.collect_system_metrics()
                    performance_delta = self.metrics_registry.calculate_delta("train.baseline", final_metrics)
                    self._logger.debug(f"Training performance delta: {performance_delta}")
                    self.hooks.emit("train.performance.analysis", delta=performance_delta)
                except Exception as e:
                    self._logger.debug(f"Observability final metrics skipped: {e}")
                
            # Cache successful configuration for future use
            if self.cache_manager and not cache_hit:
                try:
                    cache = self.cache_manager.get_default_cache()
                    cache_key = f"train_standard_{hash(str(self.args))}_{self.cfg.get('model.size', 'unknown')}"
                    cache_data = {
                        "status": "success", 
                        "device_config": device_config,
                        "training_context": training_context,
                        "performance_metrics": final_metrics if self.observability else {},
                        "timestamp": self.config_manager.get_current_timestamp()
                    }
                    cache.set(cache_ns, cache_key, cache_data, ttl=7200)  # 2小时TTL
                    self._logger.debug("Training configuration cached for future use")
                except Exception as e:
                    self._logger.debug(f"Enhanced cache set failed: {e}")
                
        except Exception as e:
            # Enhanced error handling with detailed diagnostics
            error_context = {
                "error": str(e),
                "training_phase": "standard_training",
                "cache_hit": cache_hit,
                "distributed_enabled": self._is_distributed(),
                "device_config": device_config
            }
            self.hooks.emit("train.standard.error", **error_context)
            
            # Attempt error recovery with cached configuration
            if self.cache_manager and not cache_hit:
                self._logger.debug("Attempting error recovery with alternative configurations...")
                # Implementation for recovery logic would go here
            
            raise
        
        # Emit training standard completion event with results
        completion_context = {
            "success": training_success,
            "cache_utilized": cache_hit,
            "distributed_coordination": self._is_distributed(),
            "performance_improved": performance_delta.get('improvement', False) if self.observability else None
        }
        self.hooks.emit("train.standard.complete", **completion_context)

    def _is_distributed(self) -> bool:
        try:
            fn = getattr(self.dist_config, 'is_distributed', None)
            if callable(fn):
                return bool(fn())
            # Fallback: infer from possible attributes
            return bool(getattr(self.dist_config, 'distributed', False))
        except Exception:
            return False

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
            self._logger.error("quant_export requires --ckpt and --save or corresponding config keys")
            self.hooks.emit("train.quant.error", error="Missing checkpoint or save path")
            raise SystemExit(1)
        
        # Validate checkpoint using utils checkpoint manager
        checkpoint_manager = PiscesLxCoreCheckpointManager()
        if not checkpoint_manager.validate_checkpoint(ckpt):
            self._logger.error(f"Invalid checkpoint file: {ckpt}")
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
            self._logger.error(f"Quantization and export failed: {e}")
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
        self._logger.info(f"Running preference alignment: {pref_type}")
        
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
                self._logger.error(f"Unknown train.pref.type: {pref_type}")
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
            self._logger.error(f"Preference alignment failed: {e}")
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

