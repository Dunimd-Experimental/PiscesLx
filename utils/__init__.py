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

from utils.fs.core import PiscesLxCoreFS
from utils.log.core import PiscesLxCoreLog
from utils.cache.core import PiscesLxCoreCacheManagerFacade
from utils.device.facade import PiscesLxCoreDeviceFacade
from utils.observability.metrics import PiscesLxCoreMetricsRegistry
from utils.observability.facade import PiscesLxCoreObservabilityFacade
from utils.watermark.manager import PiscesWatermarkManager as PiscesLxUtilsWatermarkManager
from utils.watermark.watermark import PiscesLxWatermark as PiscesLxUtilsWatermark
# Expose logits processor via utils facade
try:
    from utils.watermark.logits_processor import PiscesWatermarkLogitsProcessor as PiscesLxUtilsLogitsProcessor
except Exception:
    PiscesLxUtilsLogitsProcessor = None
from utils.watermark.protocol import (
    create_payload as PiscesLxUtilsCreatePayload,
    sign_payload as PiscesLxUtilsSignPayload,
    verify_payload as PiscesLxUtilsVerifyPayload,
)
from utils.config.manager import PiscesLxCoreConfigManager, PiscesLxCoreConfigManagerFacade
from utils.cache.enhanced import PiscesLxCoreEnhancedCache, PiscesLxCoreEnhancedCacheManager
from utils.checkpoint import PiscesLxCoreCheckpointManager
from utils.concurrency import (
    PiscesLxCoreTimeout,
    PiscesLxCoreRetry,
    PiscesLxCoreAsyncManager,
    PiscesLxCoreResourcePool,
    PiscesLxCoreConcurrencyManager,
    PiscesLxCoreParallel,
)
from utils.error import (
    PiscesLxCoreErrorCode,
    PiscesLxCoreError,
    PiscesLxCoreValidationError,
    PiscesLxCoreConfigError,
    PiscesLxCoreIOError,
    PiscesLxCoreFilesystemError,
    PiscesLxCoreConcurrencyError,
    PiscesLxCoreTimeoutError,
    PiscesLxCoreNetworkError,
    PiscesLxCoreMemoryError,
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
    PiscesLxCorePlatformDetectionError,
    PiscesLxCoreDistributedSetupError,
    PiscesLxCoreDeviceOrchestrationError,
    PiscesLxCoreConfigurationError,
)
from utils.quantization import PiscesLxCoreQuantizer, PiscesLxCoreQuantizationFacade
from utils.ul import PiscesLxCoreUL
from utils.validate import PiscesLxCoreValidator

__all__ = [
    # Config
    "PiscesLxCoreConfigManager",
    "PiscesLxCoreConfigManagerFacade",
    # Cache
    "PiscesLxCoreEnhancedCache",
    "PiscesLxCoreEnhancedCacheManager",
    "PiscesLxCoreCacheManagerFacade",
    # Device
    "PiscesLxCoreDeviceFacade",
    # Filesystem
    "PiscesLxCoreFS",
    # Logging
    "PiscesLxCoreLog",
    # Observability
    "PiscesLxCoreObservabilityFacade",
    # Metrics
    "PiscesLxCoreMetricsRegistry",
    # Watermark
    "PiscesLxUtilsWatermark",
    "PiscesLxUtilsWatermarkManager",
    "PiscesLxUtilsCreatePayload",
    "PiscesLxUtilsSignPayload",
    "PiscesLxUtilsVerifyPayload",
    "PiscesLxUtilsLogitsProcessor",
    # Checkpoint
    "PiscesLxCoreCheckpointManager",
    # Concurrency
    "PiscesLxCoreTimeout",
    "PiscesLxCoreRetry",
    "PiscesLxCoreAsyncManager",
    "PiscesLxCoreResourcePool",
    "PiscesLxCoreConcurrencyManager",
    "PiscesLxCoreParallel",
    # Error classes
    "PiscesLxCoreErrorCode",
    "PiscesLxCoreError",
    "PiscesLxCoreValidationError",
    "PiscesLxCoreConfigError",
    "PiscesLxCoreIOError",
    "PiscesLxCoreFilesystemError",
    "PiscesLxCoreConcurrencyError",
    "PiscesLxCoreTimeoutError",
    "PiscesLxCoreNetworkError",
    "PiscesLxCoreMemoryError",
    "PiscesLxCoreCacheError",
    "PiscesLxCoreLogError",
    "PiscesLxCoreHooksError",
    "PiscesLxCoreObservabilityError",
    "PiscesLxCoreReporterError",
    "PiscesLxCoreMetricsError",
    "PiscesLxCoreExporterError",
    "PiscesLxCoreDeviceError",
    "PiscesLxCoreNoGPUError",
    "PiscesLxCoreGPUInsufficientError",
    "PiscesLxCorePlatformDetectionError",
    "PiscesLxCoreDistributedSetupError",
    "PiscesLxCoreDeviceOrchestrationError",
    "PiscesLxCoreConfigurationError",
    # Quantization
    "PiscesLxCoreQuantizer",
    "PiscesLxCoreQuantizationFacade",
    # UL
    "PiscesLxCoreUL",
    # Validate
    "PiscesLxCoreValidator",
]