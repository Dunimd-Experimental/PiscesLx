#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

"""
Remote MCP Client Support for PiscesL1

This module provides support for user-local MCP clients, enabling the model to
remotely invoke tools on the user's local machine while maintaining compatibility
with the existing XML-based tool calling format.
"""

from .client import ArcticRemoteMCPClient
from .protocol import ArcticRemoteMCPProtocol
from .router import ArcticMCPRemoteRouter
from .connector import ArcticMCPRemoteConnector
from model.mcp.translator import ArcticMCPTranslationLayer
from .types import RemoteToolCall, RemoteExecutionResult, RemoteClientConfig

__all__ = [
    "ArcticRemoteMCPClient",
    "ArcticRemoteMCPProtocol", 
    "ArcticMCPRemoteRouter",
    "ArcticMCPRemoteConnector",
    "RemoteToolCall",
    "RemoteExecutionResult", 
    "RemoteClientConfig"
]