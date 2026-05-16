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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
Multi-Source Teacher Provider System for Knowledge Distillation

This module provides a unified interface for accessing teacher models from
multiple sources: local weights, private server deployments, and remote APIs
(OpenAI, Anthropic, etc.).

Key Features:
    - Unified interface for different teacher sources
    - Local model loading with memory optimization
    - Server-based teacher with custom API
    - Remote API integration (OpenAI, Anthropic, DeepSeek, etc.)
    - Automatic provider selection based on configuration

Architecture:
    TeacherProvider (ABC)
    ├── LocalTeacherProvider: Load model weights locally
    ├── ServerTeacherProvider: Connect to private inference server
    └── RemoteTeacherProvider: Use remote API (OpenAI-style)

Usage:
    from opss.train.distill_provider import (
        POPSSTeacherProviderFactory,
        POPSSTeacherConfig,
        POPSSTeacherProviderType,
    )

    config = POPSSTeacherConfig(
        provider_type=POPSSTeacherProviderType.LOCAL,
        model_path="./models/teacher-7b"
    )
    teacher = POPSSTeacherProviderFactory.create(config)
    outputs = teacher.get_all_outputs(input_ids)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file
from configs.version import VERSION


def _normalize_base_url(base_url: str) -> str:
    """Normalize base URL by stripping common endpoint suffixes.
    
    Users may accidentally pass the chat completions endpoint
    (e.g. https://api.deepseek.com/chat/completions) instead of the
    API base URL. The OpenAI client auto-appends /chat/completions,
    /models etc., so endpoint URLs as base would produce malformed requests.
    """
    import re
    base_url = base_url.rstrip().rstrip('/').rstrip(',')
    base_url = re.sub(r'(/v\d+)?/chat/completions$', '', base_url)
    base_url = re.sub(r'(/v\d+)?/completions$', '', base_url)
    base_url = re.sub(r'(/v\d+)?/models$', '', base_url)
    if not base_url.endswith('/v1'):
        base_url = base_url.rstrip('/')
    return base_url


class POPSSTeacherProviderType(Enum):
    """Teacher provider type enumeration."""
    LOCAL = "local"
    SERVER = "server"
    REMOTE = "remote"




@dataclass
class POPSSTeacherConfig:
    """Teacher model configuration for distillation.
    
    Attributes:
        provider_type: Type of provider (local/server/remote).
        model_path: Path to local model weights (for local provider).
        device: Device to load model on (for local provider).
        torch_dtype: Data type for model weights.
        server_url: URL of inference server (for server provider).
        api_key: API key for authentication.
        api_type: Type of remote API (openai/anthropic/deepseek/etc).
        model_name: Model name for remote API.
        base_url: Custom base URL for API endpoint.
        max_retries: Maximum retry attempts for network requests.
        timeout: Request timeout in seconds.
        batch_size: Batch size for API requests.
        offload_strategy: Memory offload strategy for local models.
        layer_indices: Specific layers to extract (None for all).
        output_hidden_states: Whether to output hidden states.
        output_attentions: Whether to output attentions.
        alpha: Distillation loss weight (0.0-1.0).
        temperature: Temperature for soft label distillation.
    """
    
    provider_type: str = "local"
    
    model_path: Optional[str] = None
    device: str = "cuda:0"
    torch_dtype: str = "float16"
    offload_strategy: str = "auto"
    
    server_url: Optional[str] = None
    api_key: Optional[str] = None
    
    api_type: Optional[str] = None
    model_name: Optional[str] = None
    remote_distill_mode: str = "logprobs"
    base_url: Optional[str] = None
    
    max_retries: int = 3
    timeout: int = 60
    batch_size: int = 8
    
    layer_indices: Optional[List[int]] = None
    output_hidden_states: bool = True
    output_attentions: bool = True
    
    alpha: float = 0.5
    temperature: float = 2.0


class POPSSTeacherProvider(ABC):
    """Abstract base class for teacher model providers.
    
    This class defines the unified interface for accessing teacher models
    from different sources. Implementations should handle the specific
    details of loading, connecting, or calling the teacher model.
    
    All methods should return tensors in a consistent format for the
    distillation loss computation.
    """
    
    def __init__(self, config: POPSSTeacherConfig):
        self.config = config
        self._LOG = PiscesLxLogger(
            "PiscesLx.Distill.Provider",
            file_path=get_log_file("PiscesLx.Distill.Provider"),
            enable_file=True
        )
    
    @abstractmethod
    def get_logits(self, input_ids: Tensor) -> Tensor:
        """Get teacher model logits.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            
        Returns:
            Logits tensor [batch, seq_len, vocab_size].
        """
        pass
    
    @abstractmethod
    def get_hidden_states(self, input_ids: Tensor) -> Optional[List[Tensor]]:
        """Get teacher model hidden states.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            
        Returns:
            List of hidden state tensors, one per layer.
            Each tensor: [batch, seq_len, hidden_size].
            Returns None if not supported by provider.
        """
        pass
    
    @abstractmethod
    def get_attentions(self, input_ids: Tensor) -> Optional[List[Tensor]]:
        """Get teacher model attention weights.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            
        Returns:
            List of attention tensors, one per layer.
            Each tensor: [batch, num_heads, seq_len, seq_len].
            Returns None if not supported by provider.
        """
        pass
    
    def get_all_outputs(self, input_ids: Tensor) -> Dict[str, Any]:
        """Get all teacher outputs for distillation.
        
        Args:
            input_ids: Input token IDs [batch, seq_len].
            
        Returns:
            Dictionary containing:
                - logits: [batch, seq_len, vocab_size]
                - hidden_states: Optional[List[Tensor]]
                - attentions: Optional[List[Tensor]]
        """
        return {
            "logits": self.get_logits(input_ids),
            "hidden_states": self.get_hidden_states(input_ids) if self.config.output_hidden_states else None,
            "attentions": self.get_attentions(input_ids) if self.config.output_attentions else None,
        }
    
    def is_available(self) -> bool:
        """Check if teacher provider is available."""
        return True
    
    def close(self):
        """Release resources."""
        pass


class POPSSLocalTeacherProvider(POPSSTeacherProvider):
    """Local teacher model provider.
    
    Loads teacher model weights directly into GPU memory.
    Best for smaller models (≤13B) with sufficient GPU memory.
    
    Features:
        - Fastest access speed (no network latency)
        - Full access to hidden states and attentions
        - No API costs
        - Data privacy guaranteed
    
    Limitations:
        - High GPU memory usage
        - Limited by local GPU capacity
    """
    
    def __init__(self, config: POPSSTeacherConfig, model_class=None, tokenizer=None):
        super().__init__(config)
        self.model = None
        self.tokenizer = tokenizer
        self.model_class = model_class
        self._initialized = False
        self._offloaded_layers = set()
        
    def _lazy_init(self):
        """Lazy initialization of model."""
        if self._initialized:
            return
            
        self._LOG.info(f"Loading local teacher model from {self.config.model_path}")
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
        
        if self.model_class is not None:
            self.model = self.model_class.from_pretrained(
                self.config.model_path,
                torch_dtype=dtype,
                device_map=self.config.offload_strategy if self.config.offload_strategy != "none" else None,
            )
        else:
            try:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_path,
                    torch_dtype=dtype,
                    device_map=self.config.offload_strategy if self.config.offload_strategy != "none" else None,
                    trust_remote_code=True,
                )
            except ImportError:
                raise ImportError("transformers library required for default model loading")
        
        if self.config.offload_strategy == "none":
            self.model = self.model.to(self.config.device)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self._initialized = True
        self._LOG.info("Local teacher model loaded successfully")
    
    def get_logits(self, input_ids: Tensor) -> Tensor:
        """Get logits from local model."""
        self._lazy_init()
        
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.model.device))
            return outputs.logits
    
    def get_hidden_states(self, input_ids: Tensor) -> Optional[List[Tensor]]:
        """Get hidden states from local model."""
        self._lazy_init()
        
        if not hasattr(self.model, 'output_hidden_states'):
            return None
        
        with torch.no_grad():
            outputs = self.model(
                input_ids.to(self.model.device),
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states
            
            if self.config.layer_indices is not None:
                return [hidden_states[i] for i in self.config.layer_indices]
            return list(hidden_states)
    
    def get_attentions(self, input_ids: Tensor) -> Optional[List[Tensor]]:
        """Get attentions from local model."""
        self._lazy_init()
        
        if not hasattr(self.model, 'output_attentions'):
            return None
        
        with torch.no_grad():
            outputs = self.model(
                input_ids.to(self.model.device),
                output_attentions=True,
            )
            attentions = outputs.attentions
            
            if self.config.layer_indices is not None:
                return [attentions[i] for i in self.config.layer_indices]
            return list(attentions)
    
    def close(self):
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
        self._initialized = False


class POPSSServerTeacherProvider(POPSSTeacherProvider):
    """Server-based teacher model provider.
    
    Connects to a private inference server for teacher outputs.
    Best for larger models (70B+) deployed on dedicated servers.
    
    Features:
        - Support for very large models
        - Custom API for hidden states and attentions
        - Fast internal network access
        - Data stays within organization
    
    Limitations:
        - Requires server deployment
        - Network latency
        - Server maintenance overhead
    """
    
    def __init__(self, config: POPSSTeacherConfig):
        super().__init__(config)
        self.session = None
        self._init_session()
    
    def _init_session(self):
        """Initialize HTTP session."""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            self.session = requests.Session()
            
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            if self.config.api_key:
                self.session.headers.update({
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                })
        except ImportError:
            raise ImportError("requests library required for server provider")
    
    def _request(self, input_ids: Tensor, **kwargs) -> Dict[str, Any]:
        """Make request to server."""
        payload = {
            "input_ids": input_ids.tolist(),
            **kwargs
        }
        
        response = self.session.post(
            f"{self.config.server_url}/v1/distill/forward",
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()
    
    def get_logits(self, input_ids: Tensor) -> Tensor:
        """Get logits from server."""
        result = self._request(
            input_ids,
            output_hidden_states=False,
            output_attentions=False,
        )
        return torch.tensor(result["logits"], dtype=torch.float32)
    
    def get_hidden_states(self, input_ids: Tensor) -> Optional[List[Tensor]]:
        """Get hidden states from server."""
        try:
            result = self._request(
                input_ids,
                output_hidden_states=True,
                output_attentions=False,
                layer_indices=self.config.layer_indices,
            )
            hidden_states = result.get("hidden_states", {})
            if not hidden_states:
                return None
            indices = sorted(map(int, hidden_states.keys()))
            return [torch.tensor(hidden_states[str(i)], dtype=torch.float32) for i in indices]
        except Exception as e:
            self._LOG.warning(f"Failed to get hidden states: {e}")
            return None
    
    def get_attentions(self, input_ids: Tensor) -> Optional[List[Tensor]]:
        """Get attentions from server."""
        try:
            result = self._request(
                input_ids,
                output_hidden_states=False,
                output_attentions=True,
                layer_indices=self.config.layer_indices,
            )
            attentions = result.get("attentions", {})
            if not attentions:
                return None
            indices = sorted(map(int, attentions.keys()))
            return [torch.tensor(attentions[str(i)], dtype=torch.float32) for i in indices]
        except Exception as e:
            self._LOG.warning(f"Failed to get attentions: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check server availability."""
        try:
            response = self.session.get(
                f"{self.config.server_url}/health",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close HTTP session."""
        if self.session is not None:
            self.session.close()
            self.session = None


class POPSSRemoteTeacherProvider(POPSSTeacherProvider):
    """Remote API teacher model provider.
    
    Uses external API services (OpenAI, Anthropic, etc.) as teacher.
    Best for accessing closed-source models or when no GPU is available.
    
    Features:
        - Access to most powerful closed-source models
        - No local GPU required
        - Simple deployment
        - Pay-as-you-go pricing
    
    Limitations:
        - Cannot access hidden states or attentions
        - API costs per request
        - Network latency
        - Data privacy concerns
    """
    
    def __init__(self, config: POPSSTeacherConfig, tokenizer=None):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.client = None
        self._client_type = "openai"
        self._init_client()
    
    def _init_client(self):
        """Initialize API client."""
        try:
            from openai import OpenAI
            
            api_type = self.config.api_type or "openai"
            
            if api_type == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.config.api_key)
                self._client_type = "anthropic"
            else:
                base_url = None
                if hasattr(self.config, 'base_url') and self.config.base_url:
                    base_url = self.config.base_url
                    base_url = _normalize_base_url(base_url)
                if isinstance(base_url, str) and "deepseek.com" in base_url and not base_url.endswith("/beta"):
                    base_url = base_url.rstrip("/") + "/beta"
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=base_url,
                )
                self._client_type = "openai"
            
            self._LOG.info(f"Initialized {api_type} API client")
        except ImportError as e:
            raise ImportError(f"openai/anthropic library required: {e}")
    
    def get_logits(self, input_ids: Tensor) -> Tensor:
        """Get approximate logits via logprobs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for remote provider logits")
        
        text = self.tokenizer.decode(input_ids[0])
        
        if self._client_type == "anthropic":
            return self._get_anthropic_logits(text, input_ids)
        else:
            return self._get_openai_logits(text, input_ids)
    
    def _get_openai_logits(self, text: str, input_ids: Tensor) -> Tensor:
        """Get logits from OpenAI-compatible API."""
        try:
            response = self.client.completions.create(
                model=self.config.model_name,
                prompt=text,
                max_tokens=0,
                logprobs=5,
                echo=True,
            )
            
            vocab_size = len(self.tokenizer) if self.tokenizer else 50000
            seq_len = input_ids.shape[1]
            logits = torch.full((1, seq_len, vocab_size), -100.0)
            
            if response.choices and response.choices[0].logprobs:
                logprobs_data = response.choices[0].logprobs
                if logprobs_data.top_logprobs:
                    for i, top_lp in enumerate(logprobs_data.top_logprobs):
                        if top_lp and i < seq_len:
                            for token, lp in top_lp.items():
                                try:
                                    token_id = self.tokenizer.encode(token)[-1]
                                    logits[0, i, token_id] = lp
                                except Exception:
                                    pass
            
            return logits
        except Exception as e:
            self._LOG.error(f"Failed to get OpenAI logits: {e}")
            vocab_size = len(self.tokenizer) if self.tokenizer else 50000
            return torch.zeros(1, input_ids.shape[1], vocab_size)
    
    def _get_anthropic_logits(self, text: str, input_ids: Tensor) -> Tensor:
        """Get logits from Anthropic API."""
        try:
            response = self.client.completions.create(
                model=self.config.model_name,
                prompt=text,
                max_tokens_to_sample=0,
            )
            
            vocab_size = len(self.tokenizer) if self.tokenizer else 50000
            return torch.zeros(1, input_ids.shape[1], vocab_size)
        except Exception as e:
            self._LOG.error(f"Failed to get Anthropic logits: {e}")
            vocab_size = len(self.tokenizer) if self.tokenizer else 50000
            return torch.zeros(1, input_ids.shape[1], vocab_size)
    
    def get_hidden_states(self, input_ids: Tensor) -> Optional[List[Tensor]]:
        """Remote API does not support hidden states."""
        self._LOG.warning("Remote API does not support hidden states")
        return None
    
    def get_attentions(self, input_ids: Tensor) -> Optional[List[Tensor]]:
        """Remote API does not support attentions."""
        self._LOG.warning("Remote API does not support attentions")
        return None
    
    def generate_for_distillation(
        self,
        prompts: List[str],
        max_tokens: int = 512,
    ) -> List[str]:
        """Generate text for contrastive distillation.
        
        Args:
            prompts: List of prompt strings.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            List of generated text strings.
        """
        results = []
        
        for prompt in prompts:
            try:
                if self._client_type == "anthropic":
                    response = self.client.messages.create(
                        model=self.config.model_name,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    results.append(response.content[0].text)
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.model_name,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    results.append(response.choices[0].message.content)
            except Exception as e:
                self._LOG.error(f"Generation failed: {e}")
                results.append("")
        
        return results
    
    def is_available(self) -> bool:
        """Check API availability via client state and essential config.
        
        No live API request is made here to avoid transient network/auth failures
        during initialization. Real API errors are caught during training steps.
        """
        if self.client is None:
            self._LOG.error("Remote teacher provider: API client is not initialized")
            return False
        
        if self._client_type == "anthropic":
            if not self.config.api_key:
                self._LOG.warning("Remote teacher provider: no api_key configured for Anthropic")
                return False
            return True
        
        if not self.config.model_name:
            self._LOG.warning("Remote teacher provider: no model_name configured")
            return False
        if not self.config.api_key:
            self._LOG.warning("Remote teacher provider: no api_key configured")
            return False
        return True


class POPSSTeacherProviderFactory:
    """Factory for creating teacher providers.
    
    This factory automatically creates the appropriate provider
    based on the configuration.
    """
    
    _registry: Dict[str, type] = {
        POPSSTeacherProviderType.LOCAL.value: POPSSLocalTeacherProvider,
        POPSSTeacherProviderType.SERVER.value: POPSSServerTeacherProvider,
        POPSSTeacherProviderType.REMOTE.value: POPSSRemoteTeacherProvider,
    }
    
    @classmethod
    def create(
        cls,
        config: POPSSTeacherConfig,
        model_class=None,
        tokenizer=None,
    ) -> POPSSTeacherProvider:
        """Create a teacher provider based on configuration.
        
        Args:
            config: Teacher configuration.
            model_class: Optional model class for local provider.
            tokenizer: Optional tokenizer for remote provider.
            
        Returns:
            Teacher provider instance.
        """
        provider_type = config.provider_type.lower()
        
        if provider_type not in cls._registry:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available: {list(cls._registry.keys())}"
            )
        
        provider_class = cls._registry[provider_type]
        
        if provider_type == POPSSTeacherProviderType.LOCAL.value:
            return provider_class(config, model_class=model_class, tokenizer=tokenizer)
        elif provider_type == POPSSTeacherProviderType.REMOTE.value:
            return provider_class(config, tokenizer=tokenizer)
        else:
            return provider_class(config)
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """Register a custom provider.
        
        Args:
            name: Provider name.
            provider_class: Provider class.
        """
        cls._registry[name] = provider_class


__all__ = [
    "POPSSTeacherProviderType",
    "POPSSTeacherConfig",
    "POPSSTeacherProvider",
    "POPSSLocalTeacherProvider",
    "POPSSServerTeacherProvider",
    "POPSSRemoteTeacherProvider",
    "POPSSTeacherProviderFactory",
]
