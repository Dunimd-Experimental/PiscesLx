#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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
from types import ModuleType
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
from utils import PiscesLxCoreDeviceFacade, PiscesLxCoreConfigManager, PiscesLxCoreCheckpointManager, PiscesLxCoreEnhancedCacheManager
from . import impl as _impl
from .impl import PiscesLxToolsTrainImpl

logger = PiscesLxCoreLog("PiscesLx.Tools.Train.Runner")

class PiscesLxToolsTrainRunner:
    """
    Encapsulates the complete training workflow (legacy-compatible).
    This class provides a wrapper for the legacy training implementation,
    allowing for a smooth transition while maintaining backward compatibility.
    """

    def __init__(self, args, hooks=None, profiler=None, cfg=None):
        """
        Initialize the training runner.

        Args:
            args: Training arguments.
            hooks (optional): Training hooks. Defaults to None.
            profiler (optional): Profiler instance. Defaults to None.
            cfg (optional): Configuration object. Defaults to None.
        """
        self.args = args
        self.hooks = hooks
        self.profiler = profiler
        self.cfg = cfg
        self._logger = PiscesLxCoreLog("PiscesLx.Tools.Train.Runner")
        # Instantiate class-based facade for unified style
        self._impl = PiscesLxToolsTrainImpl()
        try:
            self._impl.set_context(hooks=hooks, profiler=profiler, cfg=cfg)
        except Exception:
            # Best-effort context wiring; do not block training
            pass
        # Initialize utils components up-front for SSOT usage throughout
        self._init_utils_components()
        # Keep a reference to the module for helper delegations (no behavior change)
        self._impl_module = _impl
    
    def _init_utils_components(self):
        """Initialize utils-enhanced components for device, config, checkpoint, and observability."""
        # Device management facade
        self.device_facade = PiscesLxCoreDeviceFacade()
        # Config manager
        self.config_manager = PiscesLxCoreConfigManager()
        # Checkpoint manager
        self.checkpoint_manager = PiscesLxCoreCheckpointManager()
        # Optional: cache and observability for training-side hooks
        try:
            self.cache_manager = PiscesLxCoreEnhancedCacheManager.get_instance()
        except Exception:
            self.cache_manager = PiscesLxCoreEnhancedCacheManager()
        try:
            from utils import PiscesLxCoreObservabilityFacade, PiscesLxCoreMetricsRegistry
            self.observability = PiscesLxCoreObservabilityFacade()
            self.metrics_registry = PiscesLxCoreMetricsRegistry()
        except Exception:
            self.observability = None
            self.metrics_registry = None
        self._logger.info("PiscesLxToolsTrainRunner utils components initialized")

    def train(self) -> None:
        """
        Entry point to start training using the legacy implementation for now.
        Validates the training arguments before executing the training implementation.
        Enhanced with utils device validation and configuration management.

        Raises:
            Exception: If the training arguments are invalid.
        """
        self._logger.info("Starting training via PiscesLxToolsTrainRunner (delegating to legacy _train_impl)")
        
        # Validate device configuration before training
        device_config = self.device_facade.setup_devices(mode="auto")
        if device_config.get('device_type') == 'cpu':
            self._logger.info("Training on CPU - performance may be limited")
        
        # Validate configuration using utils config manager
        if hasattr(self, 'config_manager') and self.config_manager:
            validation_result = self.config_manager.validate(self.args)
            if not validation_result.is_valid:
                self._logger.error(f"Configuration validation failed: {validation_result.errors}")
                raise ValueError(f"Invalid configuration: {validation_result.errors}")
        
        # Validate arguments before running the implementation to preserve legacy behavior
        if hasattr(self._impl, "validate_args"):
            try:
                self.args = self._impl.validate_args(self.args)
            except Exception as e:
                self._logger.error(f"Invalid training arguments: {e}")
                raise
        
        try:
            # Delegate to class facade to run training
            self._impl.train(self.args)
            self._logger.info("Training completed successfully")
        except Exception as e:
            self._logger.error(f"Training failed: {e}")
            raise

    def setup_distributed_training(self):
        """
        Set up distributed training.
        Delegates the task to the implementation module.

        Returns:
            The result of the implementation's setup_distributed_training function.
            
        Enhanced with utils device management and configuration validation.
        """
        # Validate distributed configuration
        if hasattr(self, 'cfg') and self.cfg and hasattr(self, 'config_manager') and self.config_manager:
            dist_config = self.cfg.get("distributed", {})
            if dist_config:
                validation_result = self.config_manager.validate(dist_config)
                if not validation_result.is_valid:
                    self._logger.error(f"Distributed configuration validation failed: {validation_result.errors}")
        
        # Setup devices using utils device facade
        device_config = self.device_facade.setup_devices(mode="distributed")
        
        # Emit distributed setup start event
        if hasattr(self, 'hooks') and self.hooks:
            self.hooks.emit("train.distributed.setup.start", device_config=device_config)
        
        try:
            # Delegate to implementation
            dist_config = self._impl.setup_distributed_training()
            
            # Emit distributed setup completion event
            if hasattr(self, 'hooks') and self.hooks:
                self.hooks.emit("train.distributed.setup.complete", config=dist_config)
            
            return dist_config
            
        except Exception as e:
            # Emit distributed setup error event
            if hasattr(self, 'hooks') and self.hooks:
                self.hooks.emit("train.distributed.setup.error", error=str(e))
            self._logger.error(f"Distributed training setup failed: {e}")
            raise

    def create_dataloader(self, *a, **kw):
        """
        Create a data loader for training.

        Args:
            *a: Positional arguments passed to the implementation.
            **kw: Keyword arguments passed to the implementation.

        Returns:
            The data loader instance.
            
        Enhanced with utils device management and caching.
        """
        # Emit dataloader creation start event
        if hasattr(self, 'hooks') and self.hooks:
            self.hooks.emit("train.dataloader.create.start", args=a, kwargs=kw)
        
        try:
            # Validate device configuration for dataloader
            device_config = self.device_facade.setup_devices(mode="auto")
            
            # Create dataloader using implementation
            dataloader = self._impl.create_dataloader(*a, **kw)
            
            # Emit dataloader creation completion event
            if hasattr(self, 'hooks') and self.hooks:
                self.hooks.emit("train.dataloader.create.complete", device_config=device_config)
            
            return dataloader
            
        except Exception as e:
            # Emit dataloader creation error event
            if hasattr(self, 'hooks') and self.hooks:
                self.hooks.emit("train.dataloader.create.error", error=str(e))
            self._logger.error(f"DataLoader creation failed: {e}")
            raise

    def collate_fn(self, batch):
        """
        Collate a batch of data.
        Delegates the task to the implementation module.

        Args:
            batch: A batch of data to be collated.

        Returns:
            The collated batch of data.
            
        Enhanced with utils device management.
        """
        # Emit collate function start event
        if hasattr(self, 'hooks') and self.hooks:
            self.hooks.emit("train.collate.start", batch_size=len(batch))
        
        try:
            # Get device configuration for batch processing
            device_config = self.device_facade.setup_devices(mode="auto")
            
            # Use implementation's collate function
            collated_batch = self._impl.collate_fn(batch)
            
            # Emit collate function completion event
            if hasattr(self, 'hooks') and self.hooks:
                self.hooks.emit("train.collate.complete", device_config=device_config)
            
            return collated_batch
            
        except Exception as e:
            # Emit collate function error event
            if hasattr(self, 'hooks') and self.hooks:
                self.hooks.emit("train.collate.error", error=str(e))
            self._logger.error(f"Collate function failed: {e}")
            raise
