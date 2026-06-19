#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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



"""
Backends package -- LLM inference adapters for self-training.

This package provides a unified interface for running LLM inference in the
self-training pipeline.  All external cloud provider adapters (OpenAI,
Anthropic, Google, DeepSeek, Kimi, etc.) have been removed; the EnCRE
framework is dedicated to running PiscesL1 itself and any local /
OpenAI-compatible serving endpoint.

Available backends
------------------
- :class:`BaseBackend`  -- abstract base class.
- :class:`LocalBackend` -- Hugging Face transformers (CPU/GPU inference).
                            The default backend for self-training rollouts.
- :class:`OpenAICompatibleBackend` -- any OpenAI-protocol endpoint
                                       (vLLM, SGLang, llama.cpp, etc.).

Shared infrastructure
---------------------
- :class:`OpenAISSEBackend` -- base class for OpenAI-protocol streaming.
- :func:`retry_with_backoff` -- exponential-backoff retry.
- :class:`BackendRegistry` -- dynamic model metadata registry.
"""

from enta.backends.base import BaseBackend
from enta.backends.local import LocalBackend
from enta.backends.openai_compatible import OpenAICompatibleBackend
from enta.backends.openai_sse import OpenAISSEBackend
from enta.backends.remote_teacher import (
    JudgeVerdict,
    RemoteTeacherClient,
    RoundtableResult,
    TeacherAnswer,
    TeacherRoundtable,
    TeacherSpec,
    build_roundtable_from_config,
)
from enta.backends.retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    retry_with_backoff,
)

__all__ = [
    "DEFAULT_RETRY_CONFIG",
    "BaseBackend",
    "JudgeVerdict",
    "LocalBackend",
    "OpenAICompatibleBackend",
    "OpenAISSEBackend",
    "RemoteTeacherClient",
    "RetryConfig",
    "RoundtableResult",
    "TeacherAnswer",
    "TeacherRoundtable",
    "TeacherSpec",
    "build_roundtable_from_config",
    "retry_with_backoff",
]
