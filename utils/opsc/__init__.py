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
Operator System Core (OPSC) - Comprehensive Operator Abstraction Framework

This module provides a unified framework for operator abstraction, registration,
scheduling, and execution within the PiscesL1 ecosystem. The OPSC framework
implements a plugin-like architecture where computational units (operators) can
be developed independently and composed dynamically at runtime.

Architecture Overview:
    The OPSC framework follows a layered architecture pattern:

    1. Interface Layer (interface.py):
       - Defines abstract base classes and data structures
       - Establishes contracts for all operator implementations
       - Provides standardized result types and configuration

    2. Registry Layer (registry.py):
       - Centralized operator management
       - Version tracking and dependency resolution
       - Lifecycle management for operator instances

    3. Executor Layer (executor.py):
       - Runtime execution engine
       - Synchronous and asynchronous execution support
       - Thread pool management and resource allocation

    4. Core Layer (core.py):
       - System orchestration and coordination
       - Pipeline and batch processing
       - Health monitoring and system lifecycle

    5. Base Layer (base.py):
       - Concrete operator implementations
       - Common patterns (transform, filter, etc.)
       - Utility functions and decorators

Core Design Principles:
    1. Interface Segregation: Each operator implements only required interfaces
    2. Open/Closed: Open for extension, closed for modification
    3. Dependency Inversion: High-level modules depend on abstractions
    4. Single Responsibility: Each class has one clear purpose
    5. Composition over Inheritance: Flexible operator composition

Operator Lifecycle:
    1. Registration: Operator class registered with the registry
    2. Instantiation: Operator instance created with configuration
    3. Setup: Resources allocated, validation performed
    4. Execution: Core business logic executed
    5. Teardown: Resources released, cleanup performed

Thread Safety:
    - Registry operations are thread-safe via internal locking
    - Executor uses ThreadPoolExecutor for concurrent execution
    - Individual operators should implement their own synchronization

Usage Patterns:
    Basic Registration:
        core = PiscesLxOperatorCore()
        core.register_operator(MyOperator)
        result = core.execute('my_operator', {'data': value})

    Pipeline Execution:
        pipeline = [
            {'operator': 'transformer', 'inputs': {...}},
            {'operator': 'filter', 'inputs': {...}}
        ]
        results = core.execute_pipeline(pipeline)

    Custom Operator:
        class MyOperator(PiscesLxBaseOperator):
            @property
            def name(self) -> str:
                return "my_operator"

            @property
            def version(self) -> str:
                return "1.0.0"

            @property
            def description(self) -> str:
                return "Description of my operator"

            def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
                # Implementation here
                return PiscesLxOperatorResult(...)

Integration:
    - Logging: Integrated with PiscesLxCoreLogger via dc.py
    - Configuration: Supports PiscesLxOperatorConfig dataclasses
    - Monitoring: Compatible with PiscesLxCoreMetrics
    - Tracing: Supports PiscesLxCoreTracing integration

Version: 1.0.0
Author: Dunimd Team
License: Apache License 2.0
"""

from .interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorConfig,
    PiscesLxOperatorStatus
)
from .registry import PiscesLxOperatorRegistry
from .core import PiscesLxOperatorCore, PiscesLxOperatorSystemConfig
from .executor import PiscesLxOperatorExecutor, PiscesLxExecutionContext
from .base import (
    PiscesLxBaseOperator,
    PiscesLxTransformOperator,
    PiscesLxFilterOperator
)
from .distributed import (
    PiscesLxParallelismType,
    PiscesLxCommunicationType,
    PiscesLxParallelConfig,
    PiscesLxCommunicationStats,
    PiscesLxDistributedOperator,
    PiscesLxTensorParallelOperator,
    PiscesLxPipelineParallelOperator,
    PiscesLxExpertParallelOperator,
    PiscesLxDataParallelOperator,
    PiscesLxHybridParallelOperator
)
from .resources import (
    PiscesLxResourceType,
    PiscesLxAllocationPriority,
    PiscesLxResourceRequest,
    PiscesLxResourceAllocation,
    PiscesLxMemoryPool,
    PiscesLxCUDAScheduler,
    PiscesLxCUDAStream,
    PiscesLxCUDAEvent,
    PiscesLxGPUScheduler,
    PiscesLxResourceEstimator
)
from .heterogeneous import (
    PiscesLxDeviceTypePreference,
    PiscesLxExecutionTarget,
    PiscesLxHeterogeneousConfig,
    PiscesLxDeviceLoad,
    PiscesLxScheduledTask,
    PiscesLxOffloadManager,
    PiscesLxLoadBalancer,
    PiscesLxHeterogeneousExecutor
)

__all__ = [
    "PiscesLxOperatorInterface",
    "PiscesLxOperatorResult",
    "PiscesLxOperatorConfig",
    "PiscesLxOperatorStatus",
    "PiscesLxOperatorRegistry",
    "PiscesLxOperatorCore",
    "PiscesLxOperatorSystemConfig",
    "PiscesLxOperatorExecutor",
    "PiscesLxExecutionContext",
    "PiscesLxBaseOperator",
    "PiscesLxTransformOperator",
    "PiscesLxFilterOperator",
    "PiscesLxParallelismType",
    "PiscesLxCommunicationType",
    "PiscesLxParallelConfig",
    "PiscesLxCommunicationStats",
    "PiscesLxDistributedOperator",
    "PiscesLxTensorParallelOperator",
    "PiscesLxPipelineParallelOperator",
    "PiscesLxExpertParallelOperator",
    "PiscesLxDataParallelOperator",
    "PiscesLxHybridParallelOperator",
    "PiscesLxResourceType",
    "PiscesLxAllocationPriority",
    "PiscesLxResourceRequest",
    "PiscesLxResourceAllocation",
    "PiscesLxMemoryPool",
    "PiscesLxCUDAScheduler",
    "PiscesLxCUDAStream",
    "PiscesLxCUDAEvent",
    "PiscesLxGPUScheduler",
    "PiscesLxResourceEstimator",
    "PiscesLxDeviceTypePreference",
    "PiscesLxExecutionTarget",
    "PiscesLxHeterogeneousConfig",
    "PiscesLxDeviceLoad",
    "PiscesLxScheduledTask",
    "PiscesLxOffloadManager",
    "PiscesLxLoadBalancer",
    "PiscesLxHeterogeneousExecutor"
]
