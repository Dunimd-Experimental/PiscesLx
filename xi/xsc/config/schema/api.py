#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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
API configuration schema.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class XiApiHandshakeConfig:
    """
    API handshake configuration.
    
    Attributes:
        enabled: Whether handshake is enabled
        timeout: Handshake timeout in seconds
    """
    enabled: bool = True
    timeout: int = 5


@dataclass
class XiApiConfig:
    """
    API server configuration.
    
    Attributes:
        host: API server host
        port: API server port
        cors_origins: Allowed CORS origins
        timeout: Request timeout in seconds
        max_workers: Maximum worker threads
        handshake: Handshake configuration
    """
    host: str = "127.0.0.1"
    port: int = 3140
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    timeout: int = 120
    max_workers: int = 4
    handshake: Optional['XiApiHandshakeConfig'] = None
