#!/usr/bin/env python3

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

# Core utilities

from utils.fs.core import PiscesLxCoreFS
from utils.validate import PiscesLxCoreValidator
from utils.config.loader import PiscesLxCoreConfigLoader
from utils.config.manager import PiscesLxCoreConfigManager
from utils.checkpoint import PiscesLxCoreCheckpointManager
from utils.concurrency import PiscesLxCoreTimeout, PiscesLxCoreRetry, PiscesLxCoreParallel

# Error taxonomy
from utils.error import (
    PiscesLxCoreError,
    PiscesLxCoreErrorCode,
    PiscesLxCoreValidationError,
    PiscesLxCoreConfigError,
    PiscesLxCoreIOError,
    PiscesLxCoreFilesystemError,
    PiscesLxCoreConcurrencyError,
    PiscesLxCoreTimeoutError,
    PiscesLxCoreNetworkError,
    PiscesLxCoreCacheError,
    PiscesLxCoreLogError,
    PiscesLxCoreHooksError,
    PiscesLxCoreObservabilityError,
    PiscesLxCoreReporterError,
    PiscesLxCoreMetricsError,
    PiscesLxCoreExporterError,
    PiscesLxCoreDeviceError,
    PiscesLxCoreNoGPUError,
    PiscesLxCoreGPUInsufficientError,
    PiscesLxCoreMemoryError,
)

# Cache module
from utils.cache.core import PiscesLxCoreCache
from utils.cache.enhanced import PiscesLxCoreEnhancedCache, PiscesLxCoreEnhancedCacheManager
from utils.cache import PiscesLxCoreCacheManagerFacade

# Log module
from utils.log.core import PiscesLxCoreLog, PiscesLxCoreLogManager
from utils.log.context import PiscesLxCoreLogContext
from utils.log.analytics import (
    PiscesLxCoreLogPatternAnalyzer,
    PiscesLxCoreLogPredictor,
    PiscesLxCoreLogForecaster,
    PiscesLxCoreLogCorrelator
)
from utils.log.config import (
    PiscesLxCoreLogConfig,
    PiscesLxCoreLogConfigBuilder
)

# Hooks module - import before device to ensure get_global_hook_bus is available
from utils.hooks.types import (
    PiscesLxCoreAlgorithmicListener,
    PiscesLxCoreFunctionListener,
    PiscesLxCoreEventMetrics,
    PiscesLxCoreExecutionResult
)
from utils.hooks.registry import PiscesLxCoreListenerRegistry, PiscesLxCoreRegistryEntry
from utils.hooks.executor import PiscesLxCoreHookExecutor
from utils.hooks.bus import PiscesLxCoreHookBus, PiscesLxCoreGlobalHookBusFacade

# Device module
from utils.device.config import PiscesLxCoreDeviceConfig
from utils.device.facade import PiscesLxCoreDeviceFacade
from utils.device.runner import PiscesLxCoreDeviceRunner
from utils.device.manager import PiscesLxCoreDeviceManager
from utils.device.cluster import PiscesLxCoreDeviceUnifiedPlanner
from utils.device.cpu_detector import PiscesLxCoreDeviceCpuDetector

from utils.device.smart_detector import PiscesLxCoreDeviceSmartDetector
from utils.device.nvidia_detector import PiscesLxCoreDeviceNvidiaDetector
from utils.device.dist import (
    PiscesLxCoreDistConfig,
    PiscesLxCoreDistPlan,
    PiscesLxCoreDistPlanner,
    PiscesLxCoreProcessGroupManager,
    PiscesLxCoreModelParallelizer,
    PiscesLxCoreClusterEnv,
    PiscesLxCoreTopologyOptimizer,
    PiscesLxCoreLaunchSpec,
    PiscesLxCoreDistributedSamplerBuilder,
)

# Observability module
from utils.observability.decorators import PiscesLxCoreDecorators
from utils.observability.metrics import PiscesLxCoreMetricsRegistry
from utils.observability.facade import PiscesLxCoreObservabilityFacade
from utils.observability.manager import PiscesLxCoreObservabilityManager
from utils.observability.service import PiscesLxCoreObservabilityService

# Quantization module
from utils.quantization import PiscesLxCoreQuantizer

# Config module
from utils.config.manager import PiscesLxCoreConfigManager, PiscesLxCoreConfigManagerFacade

__all__ = [
    # Core utilities
    'PiscesLxCoreFS',
    'PiscesLxCoreValidator',
    'PiscesLxCoreConfigLoader',
    'PiscesLxCoreConfigManager',
    'PiscesLxCoreConfigManagerFacade',
    'PiscesLxCoreTimeout', 
    'PiscesLxCoreRetry', 
    'PiscesLxCoreParallel',
    'PiscesLxCoreCheckpointManager',

    
    # Error taxonomy
    'PiscesLxCoreError',
    'PiscesLxCoreErrorCode',
    'PiscesLxCoreValidationError',
    'PiscesLxCoreConfigError',
    'PiscesLxCoreIOError',
    'PiscesLxCoreFilesystemError',
    'PiscesLxCoreConcurrencyError',
    'PiscesLxCoreTimeoutError',
    'PiscesLxCoreNetworkError',
    'PiscesLxCoreCacheError',
    'PiscesLxCoreLogError',
    'PiscesLxCoreHooksError',
    'PiscesLxCoreObservabilityError',
    'PiscesLxCoreReporterError',
    'PiscesLxCoreMetricsError',
    'PiscesLxCoreExporterError',
    'PiscesLxCoreDeviceError',
    'PiscesLxCoreNoGPUError',
    'PiscesLxCoreGPUInsufficientError',
    'PiscesLxCoreMemoryError',
    
    # Cache
    'PiscesLxCoreCache',
    'PiscesLxCoreEnhancedCache',
    'PiscesLxCoreEnhancedCacheManager',
    'PiscesLxCoreCacheManagerFacade',
    
    # Logging
    'PiscesLxCoreLog',
    'PiscesLxCoreLogManager',
    'PiscesLxCoreLogContext',
    
    # Log analytics
    'PiscesLxCoreLogPatternAnalyzer',
    'PiscesLxCoreLogPredictor',
    'PiscesLxCoreLogForecaster',
    'PiscesLxCoreLogCorrelator',
    
    # Log configuration
    'PiscesLxCoreLogConfig',
    'PiscesLxCoreLogConfigBuilder',
    
    # Device
    'PiscesLxCoreDeviceConfig',
    'PiscesLxCoreDeviceManager', 
    
    'PiscesLxCoreDeviceFacade',
    'PiscesLxCoreDeviceNvidiaDetector',
    'PiscesLxCoreDeviceCpuDetector',
    'PiscesLxCoreDeviceSmartDetector',
    'PiscesLxCoreDeviceRunner',
    # Device-specific error types are already exported in the taxonomy section above
    
    # Unified planner facade
    'PiscesLxCoreDeviceUnifiedPlanner',
    
    # Distributed
    'PiscesLxCoreDistConfig',
    'PiscesLxCoreDistPlan',
    'PiscesLxCoreDistPlanner',
    'PiscesLxCoreProcessGroupManager',
    'PiscesLxCoreModelParallelizer',
    'PiscesLxCoreClusterEnv',

    'PiscesLxCoreTopologyOptimizer',
    'PiscesLxCoreLaunchSpec',
    'PiscesLxCoreDistributedSamplerBuilder',
    
    # Hooks
    'PiscesLxCoreAlgorithmicListener',
    'PiscesLxCoreFunctionListener',
    'PiscesLxCoreEventMetrics',
    'PiscesLxCoreExecutionResult',
    'PiscesLxCoreListenerRegistry',
    'PiscesLxCoreRegistryEntry',
    'PiscesLxCoreHookExecutor',
    'PiscesLxCoreHookBus',
    'PiscesLxCoreGlobalHookBusFacade',
    
    # Observability
    'PiscesLxCoreObservabilityService',
    'PiscesLxCoreDecorators',
    'PiscesLxCoreMetricsRegistry',
    'PiscesLxCoreObservabilityFacade',
    'PiscesLxCoreObservabilityManager',
    
    # Quantization
    'PiscesLxCoreQuantizer',
]