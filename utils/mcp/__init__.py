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

from .types import (
    PiscesLxCoreMCPToolMetadata,
    PiscesLxCoreMCPExecutionContext,
    PiscesLxCoreMCPModuleStats,
    PiscesLxCoreMCPModuleStatus,
    PiscesLxCoreMCPHealthStatus,
    PiscesLxCoreMCPPerformanceMetrics,
    PiscesLxCoreMCPConfiguration,
    PiscesLxCoreMCPFileWatcherConfig,
    PiscesLxCoreMCPMessageType,
    PiscesLxCoreMCPMessage,
    PiscesLxCoreAgenticAction,
    PiscesLxCoreAgenticObservation,
    PiscesLxCoreMCPProtocol
)

from .monitor import PiscesLxCoreMCPMonitor
from .session import PiscesLxCoreMCPSession
from .registry import PiscesLxCoreMCPRegistry
from .tools import PiscesLxCoreMCPTools
from .xml_utils import (
    PiscesLxCoreMCPAgenticCall, PiscesLxCoreMCPXMLParser, PiscesLxCoreMCPXMLGenerator
)
from .execution import (
    PiscesLxCoreMCPExecutionMode, PiscesLxCoreMCPExecutionStatus, PiscesLxCoreMCPExecutionResult, PiscesLxCoreMCPExecutionConfig,
    PiscesLxCoreMCPExecutionManager, get_execution_manager
)
from .remote_client import (
    _RemoteToolMetadata, _RemoteClientConfig,
    PiscesLxCoreMCPRemoteClient, PiscesLxCoreMCPArcticRemoteClient, PiscesLxCoreMCPRemoteClientPool,
    get_remote_client_pool, execute_remote_tool
)
from .tool_executor import (
    PiscesLxCoreMCPToolType, PiscesLxCoreMCPToolMetadata, PiscesLxCoreMCPExecutionContext,
    PiscesLxCoreMCPNativeToolExecutor, PiscesLxCoreMCPInternalToolExecutor, PiscesLxCoreMCPExternalToolExecutor,
    PiscesLxCoreMCPUnifiedToolExecutor, get_unified_tool_executor
)
from .core import PiscesLxCoreMCPPlaza
from .arctic_extensions import PiscesLxCoreMCPTreeSearchReasoner, create_arctic_reasoner

# Version information
from configs.version import PVERSION
__version__ = PVERSION
__author__ = "Wenze Wei"
__email__ = "wenze.wei@dunimd.com"

# Module exports - only classes, no functions
__all__ = [
    # Core classes
    "PiscesLxCoreMCPPlaza",
    "PiscesLxCoreMCPMonitor", 
    "PiscesLxCoreMCPSession",
    "PiscesLxCoreMCPRegistry",
    "PiscesLxCoreMCPTools",
    
    # XML utilities
    "PiscesLxCoreMCPAgenticCall",
    "PiscesLxCoreMCPXMLParser", 
    "PiscesLxCoreMCPXMLGenerator",
    
    # Execution
    "PiscesLxCoreMCPExecutionMode",
    "PiscesLxCoreMCPExecutionStatus", 
    "PiscesLxCoreMCPExecutionResult",
    "PiscesLxCoreMCPExecutionConfig",
    "PiscesLxCoreMCPExecutionManager",
    "get_execution_manager",
    
    # Remote client
    "PiscesLxCoreMCPRemoteClient",
    "PiscesLxCoreMCPArcticRemoteClient", 
    "PiscesLxCoreMCPRemoteClientPool",
    "get_remote_client_pool",
    "execute_remote_tool",
    
    # Tool executor
    "PiscesLxCoreMCPToolType",
    "PiscesLxCoreMCPToolMetadata",
    "PiscesLxCoreMCPExecutionContext", 
    "PiscesLxCoreMCPNativeToolExecutor",
    "PiscesLxCoreMCPInternalToolExecutor",
    "PiscesLxCoreMCPExternalToolExecutor",
    "PiscesLxCoreMCPUnifiedToolExecutor",
    "get_unified_tool_executor",
    
    # Data types
    "PiscesLxCoreMCPToolMetadata",
    "PiscesLxCoreMCPExecutionContext",
    "PiscesLxCoreMCPModuleStats",
    "PiscesLxCoreMCPModuleStatus",
    "PiscesLxCoreMCPHealthStatus",
    "PiscesLxCoreMCPPerformanceMetrics",
    "PiscesLxCoreMCPConfiguration",
    "PiscesLxCoreMCPFileWatcherConfig",
    "PiscesLxCoreMCPTreeSearchReasoner",
    "create_arctic_reasoner",
    "PiscesLxCoreMCPMessageType",
    "PiscesLxCoreMCPMessage",
    "PiscesLxCoreAgenticAction",
    "PiscesLxCoreAgenticObservation",
    "PiscesLxCoreMCPProtocol",
    
    # Version info
    "__version__",
    "__author__",
    "__email__"
]

# Default configuration
default_config = PiscesLxCoreMCPConfiguration()