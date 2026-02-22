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
Expert Factory - Dynamic Expert Creation

This module provides a factory for creating expert agents dynamically.
Supports both code-based and prompt-based expert creation.

Key Features:
    - Create experts from prompt files
    - Register custom expert classes
    - Batch creation of multiple experts
    - Dynamic class generation

Usage:
    from opss.agents.factory import POPSSExpertFactory
    
    # Create from prompt file
    reviewer = POPSSExpertFactory.create("code_reviewer")
    
    # Create with model client
    reviewer = POPSSExpertFactory.create(
        "code_reviewer", 
        model_client=my_model
    )
    
    # Batch create
    experts = POPSSExpertFactory.create_batch([
        "code_reviewer",
        "math_solver",
        "architect_evaluator"
    ])
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union

from utils.dc import PiscesLxLogger

from .base import (
    POPSSBaseAgent,
    POPSSAgentConfig,
    POPSSAgentContext,
    POPSSAgentResult,
    POPSSAgentThought,
    POPSSAgentState,
    POPSSAgentCapability,
    POPSSPromptBasedAgent,
)

try:
    from .loader import POPSSPromptLoader, POPSSPromptConfig
    PROMPT_LOADER_AVAILABLE = True
except ImportError:
    PROMPT_LOADER_AVAILABLE = False
    POPSSPromptLoader = None
    POPSSPromptConfig = None

T = TypeVar('T', bound=POPSSBaseAgent)


class POPSSExpertFactory:
    """
    Factory for creating expert agents dynamically.
    
    This class provides methods to create experts from prompt files
    or registered expert classes. Supports both code-based and
    prompt-based execution modes.
    
    Attributes:
        _expert_classes: Registry of expert classes
        _expert_instances: Cache of singleton instances
        _model_client: Default model client for prompt mode
    
    Usage:
        # Register custom expert
        POPSSExpertFactory.register("my_expert", MyExpertClass)
        
        # Create expert
        expert = POPSSExpertFactory.create("code_reviewer")
        
        # Batch create
        experts = POPSSExpertFactory.create_batch(["a", "b", "c"])
    """
    
    _expert_classes: Dict[str, Type[POPSSBaseAgent]] = {}
    _expert_instances: Dict[str, POPSSBaseAgent] = {}
    _model_client: Optional[Any] = None
    
    _LOG: PiscesLxLogger = None
    
    @classmethod
    def _get_logger(cls) -> PiscesLxLogger:
        """Get or create logger."""
        if cls._LOG is None:
            cls._LOG = get_logger("POPSSExpertFactory")
        return cls._LOG
    
    @classmethod
    def set_model_client(cls, client: Any) -> None:
        """
        Set default model client for prompt mode.
        
        Args:
            client: Model client instance
        """
        cls._model_client = client
        cls._get_logger().info("Model client set")
    
    @classmethod
    def register(
        cls, 
        expert_type: str, 
        expert_class: Type[POPSSBaseAgent],
        override: bool = False
    ) -> bool:
        """
        Register an expert class.
        
        Args:
            expert_type: Unique identifier for the expert
            expert_class: Expert class to register
            override: Whether to override existing registration
            
        Returns:
            True if registration succeeded
        """
        if expert_type in cls._expert_classes and not override:
            cls._get_logger().warning(
                f"Expert already registered: {expert_type}"
            )
            return False
        
        cls._expert_classes[expert_type] = expert_class
        cls._get_logger().info(f"Registered expert class: {expert_type}")
        return True
    
    @classmethod
    def unregister(cls, expert_type: str) -> bool:
        """
        Unregister an expert class.
        
        Args:
            expert_type: Expert type to unregister
            
        Returns:
            True if unregistration succeeded
        """
        if expert_type in cls._expert_classes:
            del cls._expert_classes[expert_type]
            cls._get_logger().info(f"Unregistered expert: {expert_type}")
            return True
        return False
    
    @classmethod
    def create(
        cls,
        expert_type: str,
        mode: str = "auto",
        config: Optional[POPSSAgentConfig] = None,
        model_client: Optional[Any] = None,
        singleton: bool = False,
        **kwargs
    ) -> POPSSBaseAgent:
        """
        Create an expert instance.
        
        Args:
            expert_type: Expert type identifier
            mode: Execution mode (auto/code/prompt/hybrid)
            config: Optional agent configuration
            model_client: Optional model client
            singleton: Whether to return singleton instance
            **kwargs: Additional configuration parameters
            
        Returns:
            Expert agent instance
        """
        if singleton and expert_type in cls._expert_instances:
            return cls._expert_instances[expert_type]
        
        effective_mode = cls._determine_mode(expert_type, mode)
        
        if effective_mode == "code" and expert_type in cls._expert_classes:
            expert = cls._create_from_class(expert_type, config, **kwargs)
        elif effective_mode in ["prompt", "hybrid"]:
            expert = cls._create_from_prompt(
                expert_type, 
                effective_mode, 
                config, 
                model_client,
                **kwargs
            )
        else:
            expert = cls._create_dynamic(expert_type, effective_mode, config, model_client)
        
        if singleton:
            cls._expert_instances[expert_type] = expert
        
        return expert
    
    @classmethod
    def _determine_mode(cls, expert_type: str, mode: str) -> str:
        """Determine effective execution mode."""
        if mode != "auto":
            return mode
        
        if expert_type in cls._expert_classes:
            return "code"
        
        if PROMPT_LOADER_AVAILABLE:
            try:
                POPSSPromptLoader.load(expert_type)
                return "prompt"
            except FileNotFoundError:
                pass
        
        return "prompt"
    
    @classmethod
    def _create_from_class(
        cls,
        expert_type: str,
        config: Optional[POPSSAgentConfig],
        **kwargs
    ) -> POPSSBaseAgent:
        """Create expert from registered class."""
        expert_class = cls._expert_classes[expert_type]
        
        config = config or POPSSAgentConfig(
            name=expert_type,
            agent_id=f"{expert_type}_{uuid.uuid4().hex[:8]}"
        )
        
        return expert_class(config, **kwargs)
    
    @classmethod
    def _create_from_prompt(
        cls,
        expert_type: str,
        mode: str,
        config: Optional[POPSSAgentConfig],
        model_client: Optional[Any],
        **kwargs
    ) -> POPSSBaseAgent:
        """Create expert from prompt file."""
        if not PROMPT_LOADER_AVAILABLE:
            raise RuntimeError("Prompt loader not available")
        
        config = config or POPSSAgentConfig(
            name=expert_type,
            expert_type=expert_type,
            mode=mode,
            agent_id=f"{expert_type}_{uuid.uuid4().hex[:8]}"
        )
        
        effective_client = model_client or cls._model_client
        
        return POPSSPromptBasedAgent(
            expert_type=expert_type,
            config=config,
            model_client=effective_client
        )
    
    @classmethod
    def _create_dynamic(
        cls,
        expert_type: str,
        mode: str,
        config: Optional[POPSSAgentConfig],
        model_client: Optional[Any]
    ) -> POPSSBaseAgent:
        """Create dynamic expert class."""
        config = config or POPSSAgentConfig(
            name=expert_type,
            expert_type=expert_type,
            mode=mode,
            agent_id=f"{expert_type}_{uuid.uuid4().hex[:8]}"
        )
        
        class DynamicExpert(POPSSBaseAgent):
            expert_type = expert_type
            mode = mode
        
        expert = DynamicExpert(config)
        
        if model_client or cls._model_client:
            expert.set_model_client(model_client or cls._model_client)
        
        return expert
    
    @classmethod
    def create_batch(
        cls,
        expert_types: List[str],
        mode: str = "auto",
        model_client: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, POPSSBaseAgent]:
        """
        Create multiple experts at once.
        
        Args:
            expert_types: List of expert type identifiers
            mode: Execution mode for all experts
            model_client: Optional model client
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary mapping expert_type to instance
        """
        experts = {}
        
        for expert_type in expert_types:
            try:
                experts[expert_type] = cls.create(
                    expert_type,
                    mode=mode,
                    model_client=model_client,
                    **kwargs
                )
            except Exception as e:
                cls._get_logger().error(
                    f"Failed to create {expert_type}: {e}"
                )
        
        return experts
    
    @classmethod
    def create_cluster(
        cls,
        cluster_name: str,
        model_client: Optional[Any] = None
    ) -> Dict[str, POPSSBaseAgent]:
        """
        Create all experts in a cluster.
        
        Args:
            cluster_name: Cluster name (code, reasoning, etc.)
            model_client: Optional model client
            
        Returns:
            Dictionary of experts in the cluster
        """
        if not PROMPT_LOADER_AVAILABLE:
            cls._get_logger().warning("Prompt loader not available")
            return {}
        
        experts_by_category = POPSSPromptLoader.list_by_category()
        
        if cluster_name not in experts_by_category:
            cls._get_logger().warning(f"Cluster not found: {cluster_name}")
            return {}
        
        return cls.create_batch(
            experts_by_category[cluster_name],
            mode="prompt",
            model_client=model_client
        )
    
    @classmethod
    def create_all(
        cls,
        model_client: Optional[Any] = None
    ) -> Dict[str, POPSSBaseAgent]:
        """
        Create all available experts.
        
        Args:
            model_client: Optional model client
            
        Returns:
            Dictionary of all experts
        """
        all_experts = {}
        
        for expert_type in cls.list_available():
            try:
                all_experts[expert_type] = cls.create(
                    expert_type,
                    mode="auto",
                    model_client=model_client
                )
            except Exception as e:
                cls._get_logger().error(
                    f"Failed to create {expert_type}: {e}"
                )
        
        return all_experts
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available expert types.
        
        Returns:
            List of expert type identifiers
        """
        expert_types = set(cls._expert_classes.keys())
        
        if PROMPT_LOADER_AVAILABLE:
            expert_types.update(POPSSPromptLoader.list_available())
        
        return sorted(expert_types)
    
    @classmethod
    def list_by_category(cls) -> Dict[str, List[str]]:
        """
        List experts grouped by category.
        
        Returns:
            Dictionary mapping category to expert types
        """
        categories = {}
        
        if PROMPT_LOADER_AVAILABLE:
            categories = POPSSPromptLoader.list_by_category()
        
        for expert_type in cls._expert_classes:
            expert = cls._expert_classes[expert_type]
            if hasattr(expert, 'expert_type') and expert.expert_type:
                category = getattr(expert, 'category', 'custom')
                if category not in categories:
                    categories[category] = []
                if expert_type not in categories[category]:
                    categories[category].append(expert_type)
        
        return categories
    
    @classmethod
    def get_expert_info(cls, expert_type: str) -> Dict[str, Any]:
        """
        Get information about an expert.
        
        Args:
            expert_type: Expert type identifier
            
        Returns:
            Expert information dictionary
        """
        info = {
            "expert_type": expert_type,
            "available": False,
            "mode": "unknown",
            "source": "unknown",
        }
        
        if expert_type in cls._expert_classes:
            info["available"] = True
            info["mode"] = "code"
            info["source"] = "registered_class"
            info["class_name"] = cls._expert_classes[expert_type].__name__
        
        if PROMPT_LOADER_AVAILABLE:
            try:
                prompt_config = POPSSPromptLoader.load(expert_type)
                info["available"] = True
                info["mode"] = "prompt"
                info["source"] = prompt_config.source
                info["version"] = prompt_config.version
                info["has_system_prompt"] = bool(prompt_config.system_prompt)
                info["has_behavior_prompt"] = bool(prompt_config.behavior_prompt)
                info["has_output_schema"] = bool(prompt_config.output_schema)
            except FileNotFoundError:
                pass
        
        return info
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear singleton instance cache."""
        cls._expert_instances.clear()
        cls._get_logger().info("Expert cache cleared")
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """
        Get information about the expert cache.
        
        Returns:
            Cache information dictionary
        """
        return {
            "cached_count": len(cls._expert_instances),
            "cached_types": list(cls._expert_instances.keys()),
            "registered_count": len(cls._expert_classes),
            "registered_types": list(cls._expert_classes.keys()),
            "has_model_client": cls._model_client is not None,
        }
