#!/usr/bin/env/python3
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

from .task_decomposer import (
    POPSSTaskComplexity,
    POPSSTaskDependencyType,
    POPSSSubtask,
    POPSSTaskDecompositionResult,
    POPSSTaskDecomposer,
    POPSSTaskDecomposerConfig,
)

from .result_aggregator import (
    POPSSAggregationStrategy,
    POPSSResultConsistency,
    POPSSAggregatedResult,
    POPSSResultSource,
    POPSSResultAggregator,
    POPSSResultAggregatorConfig,
)

from .conflict_resolver import (
    POPSSConflictType,
    POPSSConflictSeverity,
    POPSSConflict,
    POPSSConflictResolution,
    POPSSConflictResolver,
    POPSSConflictResolverConfig,
)

from .context_manager import (
    POPSSContextScope,
    POPSSContextPriority,
    POPSSContextEntry,
    POPSSContextSnapshot,
    POPSSContextManager,
    POPSSContextManagerConfig,
)

__all__ = [
    "POPSSTaskComplexity",
    "POPSSTaskDependencyType",
    "POPSSSubtask",
    "POPSSTaskDecompositionResult",
    "POPSSTaskDecomposer",
    "POPSSTaskDecomposerConfig",
    "POPSSAggregationStrategy",
    "POPSSResultConsistency",
    "POPSSAggregatedResult",
    "POPSSResultSource",
    "POPSSResultAggregator",
    "POPSSResultAggregatorConfig",
    "POPSSConflictType",
    "POPSSConflictSeverity",
    "POPSSConflict",
    "POPSSConflictResolution",
    "POPSSConflictResolver",
    "POPSSConflictResolverConfig",
    "POPSSContextScope",
    "POPSSContextPriority",
    "POPSSContextEntry",
    "POPSSContextSnapshot",
    "POPSSContextManager",
    "POPSSContextManagerConfig",
]
