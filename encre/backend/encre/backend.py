#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

from typing import Any

from encre.backends.base import BaseBackend
from encre.backends.anthropic import AnthropicBackend
from encre.backends.bedrock import BedrockBackend
from encre.backends.deepseek import DeepSeekBackend
from encre.backends.failover import FailoverBackend
from encre.backends.google import GoogleBackend
from encre.backends.groq import GroqBackend
from encre.backends.local import LocalBackend
from encre.backends.ollama import OllamaBackend
from encre.backends.openai import OpenAIBackend
from encre.backends.openai_compatible import OpenAICompatibleBackend
from encre.backends.router import RouterBackend
from encre.backends.openrouter import OpenRouterBackend
from encre.backends.novita import NovitaBackend
from encre.backends.aigateway import AIGatewayBackend
from encre.backends.glm import GLMBackend
from encre.backends.kimi import KimiBackend
from encre.backends.arcee import ArceeBackend
from encre.backends.gmi import GMIBackend
from encre.backends.minimax import MiniMaxBackend
from encre.backends.alibaba import AlibabaBackend
from encre.backends.kilocode import KiloCodeBackend
from encre.backends.xiaomi import XiaomiBackend
from encre.backends.tencent import TencentBackend
from encre.backends.huggingface import HuggingFaceBackend
from encre.backends.opencode import OpenCodeZenBackend, OpenCodeGoBackend
from encre.backends.lmstudio import LMStudioBackend
from encre.backends.github_copilot import GitHubCopilotBackend


def create_backend(type: str, **kwargs: Any) -> BaseBackend | None:
    if not type:
        return None
    backend_map: dict[str, type[BaseBackend]] = {
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "ollama": OllamaBackend,
        "deepseek": DeepSeekBackend,
        "google": GoogleBackend,
        "groq": GroqBackend,
        "local": LocalBackend,
        "bedrock": BedrockBackend,
        "openai_compatible": OpenAICompatibleBackend,
        "failover": FailoverBackend,
        "router": RouterBackend,
        "openrouter": OpenRouterBackend,
        "novita": NovitaBackend,
        "aigateway": AIGatewayBackend,
        "glm": GLMBackend,
        "kimi": KimiBackend,
        "arcee": ArceeBackend,
        "gmi": GMIBackend,
        "minimax": MiniMaxBackend,
        "alibaba": AlibabaBackend,
        "kilocode": KiloCodeBackend,
        "xiaomi": XiaomiBackend,
        "tencent": TencentBackend,
        "huggingface": HuggingFaceBackend,
        "opencode-zen": OpenCodeZenBackend,
        "opencode-go": OpenCodeGoBackend,
        "lmstudio": LMStudioBackend,
        "github-copilot": GitHubCopilotBackend,
    }
    cls = backend_map.get(type)
    if cls is None:
        raise ValueError(f"Unknown backend type: {type}. Available: {sorted(backend_map.keys())}")
    return cls(**kwargs)
