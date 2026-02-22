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

"""
Model Router for Multi-Size Model Management

This module provides intelligent routing between different model sizes
based on request parameters, resource availability, and routing policies.

Routing Strategies:
    - explicit: Use model specified in request
    - auto: Automatically select based on task complexity
    - load_balanced: Distribute across available models
    - fallback: Try larger models if smaller ones fail

Architecture:
    PiscesLxModelRouter
    ├── Route model_id to ModelSpec
    ├── Validate model availability
    ├── Manage model instances
    └── Handle fallback logic

Usage:
    >>> router = PiscesLxModelRouter(default_size="7B")
    >>> spec = router.resolve("piscesl1-671b")
    >>> print(spec.name)  # "671B"
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

from .config import ModelSpec, MODEL_SPECS, get_model_spec


class PiscesLxRoutingStrategy(Enum):
    """Model routing strategy enumeration."""
    EXPLICIT = "explicit"
    AUTO = "auto"
    LOAD_BALANCED = "load_balanced"
    FALLBACK = "fallback"


@dataclass
class PiscesLxRouteResult:
    """
    Result of model routing decision.
    
    Attributes:
        model_spec: Selected model specification
        model_id: Original model ID from request
        model_size: Resolved model size
        strategy: Routing strategy used
        fallback_chain: Fallback models if primary fails
        metadata: Additional routing metadata
    """
    model_spec: ModelSpec
    model_id: str
    model_size: str
    strategy: PiscesLxRoutingStrategy
    fallback_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PiscesLxModelRouter:
    """
    Intelligent Model Router for Multi-Size Model Management.
    
    This class provides routing logic to select the appropriate model
    based on request parameters, resource constraints, and routing policies.
    
    Features:
        - Parse model_id to extract model size
        - Support multiple routing strategies
        - Fallback to smaller models on resource constraints
        - Track model availability and load
    
    Model ID Format:
        - "piscesl1-7b" -> 7B model
        - "piscesl1-671b" -> 671B model
        - "piscesl1-1t" -> 1T model
        - "7B" -> 7B model (shorthand)
    
    Example:
        >>> router = PiscesLxModelRouter(default_size="7B")
        >>> result = router.resolve("piscesl1-671b")
        >>> print(result.model_size)  # "671B"
    """
    
    MODEL_ID_PATTERN = re.compile(
        r'^(piscesl1[-_])?([0-9.]+[BMKTbkmkt]?)$',
        re.IGNORECASE
    )
    
    SIZE_ALIASES = {
        "0.5b": "0.5B",
        "1b": "1B",
        "7b": "7B",
        "14b": "14B",
        "72b": "72B",
        "671b": "671B",
        "1t": "1T",
        "500m": "0.5B",
        "1b": "1B",
    }
    
    FALLBACK_ORDER = ["0.5B", "1B", "7B", "14B", "72B", "671B", "1T"]
    
    def __init__(
        self,
        default_size: str = "7B",
        strategy: PiscesLxRoutingStrategy = PiscesLxRoutingStrategy.EXPLICIT,
        available_sizes: Optional[Set[str]] = None,
        enable_fallback: bool = True,
    ):
        """
        Initialize the model router.
        
        Args:
            default_size: Default model size when not specified
            strategy: Routing strategy to use
            available_sizes: Set of available model sizes (None = all available)
            enable_fallback: Enable fallback to smaller models
        """
        self.default_size = default_size
        self.strategy = strategy
        self.available_sizes = available_sizes or set(MODEL_SPECS.keys())
        self.enable_fallback = enable_fallback
        
        self._LOG = logging.getLogger(self.__class__.__name__)
        self._model_load: Dict[str, int] = {size: 0 for size in MODEL_SPECS}
        self._route_history: List[Dict[str, Any]] = []
    
    def resolve(self, model_id: str) -> PiscesLxRouteResult:
        """
        Resolve model ID to model specification.
        
        Args:
            model_id: Model identifier (e.g., "piscesl1-7b", "671B")
        
        Returns:
            PiscesLxRouteResult with resolved model spec
        
        Raises:
            ValueError: If model ID cannot be resolved
        """
        model_size = self._parse_model_id(model_id)
        
        if model_size not in MODEL_SPECS:
            raise ValueError(f"Unknown model size: {model_size}")
        
        if model_size not in self.available_sizes:
            self._LOG.warning(f"Model size {model_size} not available, using fallback")
            model_size = self._find_fallback(model_size)
        
        model_spec = MODEL_SPECS[model_size]
        
        fallback_chain = []
        if self.enable_fallback:
            fallback_chain = self._build_fallback_chain(model_size)
        
        result = PiscesLxRouteResult(
            model_spec=model_spec,
            model_id=model_id,
            model_size=model_size,
            strategy=self.strategy,
            fallback_chain=fallback_chain,
            metadata={
                "available_sizes": list(self.available_sizes),
                "current_load": self._model_load.get(model_size, 0),
            }
        )
        
        self._record_route(result)
        
        return result
    
    def _parse_model_id(self, model_id: str) -> str:
        """
        Parse model ID to extract model size.
        
        Args:
            model_id: Model identifier string
        
        Returns:
            Normalized model size string
        """
        model_id_lower = model_id.lower().strip()
        
        if model_id_lower in self.SIZE_ALIASES:
            return self.SIZE_ALIASES[model_id_lower]
        
        match = self.MODEL_ID_PATTERN.match(model_id_lower)
        if match:
            size_part = match.group(2)
            normalized = self._normalize_size(size_part)
            if normalized:
                return normalized
        
        if model_id in MODEL_SPECS:
            return model_id
        
        self._LOG.debug(f"Could not parse model_id '{model_id}', using default")
        return self.default_size
    
    def _normalize_size(self, size_str: str) -> Optional[str]:
        """
        Normalize size string to standard format.
        
        Args:
            size_str: Size string (e.g., "7b", "671B", "1T")
        
        Returns:
            Normalized size string or None
        """
        size_str = size_str.upper()
        
        if size_str in MODEL_SPECS:
            return size_str
        
        size_lower = size_str.lower()
        if size_lower in self.SIZE_ALIASES:
            return self.SIZE_ALIASES[size_lower]
        
        return None
    
    def _find_fallback(self, requested_size: str) -> str:
        """
        Find available fallback model size.
        
        Args:
            requested_size: Requested model size
        
        Returns:
            Available fallback size
        """
        requested_idx = self.FALLBACK_ORDER.index(requested_size) if requested_size in self.FALLBACK_ORDER else -1
        
        for size in reversed(self.FALLBACK_ORDER[:max(0, requested_idx + 1)]):
            if size in self.available_sizes:
                return size
        
        for size in self.FALLBACK_ORDER:
            if size in self.available_sizes:
                return size
        
        return self.default_size
    
    def _build_fallback_chain(self, primary_size: str) -> List[str]:
        """
        Build fallback chain for a model size.
        
        Args:
            primary_size: Primary model size
        
        Returns:
            List of fallback model sizes
        """
        chain = []
        primary_idx = self.FALLBACK_ORDER.index(primary_size) if primary_size in self.FALLBACK_ORDER else len(self.FALLBACK_ORDER)
        
        for size in reversed(self.FALLBACK_ORDER[:primary_idx]):
            if size in self.available_sizes and size != primary_size:
                chain.append(size)
        
        return chain
    
    def _record_route(self, result: PiscesLxRouteResult):
        """Record routing decision for analytics."""
        self._route_history.append({
            "model_id": result.model_id,
            "model_size": result.model_size,
            "strategy": result.strategy.value,
        })
        
        self._model_load[result.model_size] = self._model_load.get(result.model_size, 0) + 1
    
    def get_model_spec(self, model_id: str) -> ModelSpec:
        """
        Get model specification for a model ID.
        
        Args:
            model_id: Model identifier
        
        Returns:
            ModelSpec for the model
        """
        result = self.resolve(model_id)
        return result.model_spec
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their specifications.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        for size in self.available_sizes:
            if size in MODEL_SPECS:
                spec = MODEL_SPECS[size]
                models.append({
                    "id": f"piscesl1-{size.lower()}",
                    "size": size,
                    "hidden_size": spec.hidden_size,
                    "num_layers": spec.num_layers,
                    "num_heads": spec.num_heads,
                    "context_length": spec.context_length,
                    "moe_experts": spec.moe_experts,
                    "moe_active": spec.moe_active,
                    "is_moe": spec.moe_experts > 0,
                })
        return models
    
    def update_availability(self, size: str, available: bool):
        """
        Update model availability status.
        
        Args:
            size: Model size
            available: Whether the model is available
        """
        if available:
            self.available_sizes.add(size)
        else:
            self.available_sizes.discard(size)
    
    def get_load_stats(self) -> Dict[str, int]:
        """Get current load statistics for all models."""
        return dict(self._model_load)
    
    def reset_load_stats(self):
        """Reset load statistics."""
        self._model_load = {size: 0 for size in MODEL_SPECS}
    
    def get_route_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent routing history.
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of routing records
        """
        return self._route_history[-limit:]
    
    def select_by_complexity(
        self,
        prompt_length: int,
        task_type: str = "general",
        max_context_needed: int = 0,
    ) -> PiscesLxRouteResult:
        """
        Auto-select model based on task complexity.
        
        This method implements intelligent model selection based on
        estimated task complexity, prompt length, and context requirements.
        
        Args:
            prompt_length: Length of input prompt in tokens
            task_type: Type of task (general, code, reasoning, creative)
            max_context_needed: Maximum context length needed
        
        Returns:
            PiscesLxRouteResult with selected model
        """
        if self.strategy != PiscesLxRoutingStrategy.AUTO:
            return self.resolve(f"piscesl1-{self.default_size.lower()}")
        
        selected_size = self._estimate_model_size(
            prompt_length=prompt_length,
            task_type=task_type,
            max_context_needed=max_context_needed,
        )
        
        model_id = f"piscesl1-{selected_size.lower()}"
        return self.resolve(model_id)
    
    def _estimate_model_size(
        self,
        prompt_length: int,
        task_type: str,
        max_context_needed: int,
    ) -> str:
        """
        Estimate appropriate model size based on requirements.
        
        Args:
            prompt_length: Prompt length in tokens
            task_type: Task type
            max_context_needed: Maximum context needed
        
        Returns:
            Estimated model size
        """
        context_requirement = max(prompt_length * 4, max_context_needed)
        
        suitable_models = []
        for size in self.available_sizes:
            spec = MODEL_SPECS.get(size)
            if spec and spec.context_length >= context_requirement:
                suitable_models.append((size, spec))
        
        if not suitable_models:
            for size in reversed(self.FALLBACK_ORDER):
                if size in self.available_sizes:
                    return size
            return self.default_size
        
        complexity_multiplier = {
            "general": 1.0,
            "code": 1.2,
            "reasoning": 1.5,
            "creative": 1.0,
            "analysis": 1.3,
        }.get(task_type, 1.0)
        
        suitable_models.sort(key=lambda x: x[1].hidden_size * x[1].num_layers)
        
        target_index = int(len(suitable_models) * complexity_multiplier * 0.5)
        target_index = min(target_index, len(suitable_models) - 1)
        
        return suitable_models[target_index][0]
    
    def __repr__(self) -> str:
        return (
            f"PiscesLxModelRouter("
            f"default_size={self.default_size}, "
            f"strategy={self.strategy.value}, "
            f"available={len(self.available_sizes)})"
        )
