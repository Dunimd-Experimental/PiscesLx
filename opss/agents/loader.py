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
Prompt Loader - YAML-based Prompt Management System

This module provides a comprehensive prompt loading and management system
for agent experts. Prompts are stored in YAML files and loaded on demand.

Key Features:
    - Load prompts from YAML files
    - Cache loaded prompts for performance
    - Support hot reload without restart
    - Format prompts with variables
    - List all available experts

Usage:
    from opss.agents.loader import POPSSPromptLoader
    
    prompt = POPSSPromptLoader.load("code_reviewer")
    system_prompt = POPSSPromptLoader.get_system_prompt("code_reviewer")
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from utils.dc import PiscesLxLogger
from configs.version import VERSION


@dataclass
class POPSSPromptConfig:
    """
    Configuration for a single expert prompt.
    
    Attributes:
        expert_type: Unique identifier for the expert
        version: Prompt version string
        source: Source of the prompt (claude/openai/custom)
        language: Primary language for the expert
        system_prompt: System-level prompt for the model
        behavior_prompt: Behavior template with placeholders
        output_schema: JSON schema for structured output
        parameters: Configurable parameters for the prompt
        examples: Example input/output pairs
        metadata: Additional metadata
    """
    expert_type: str
    version: str = VERSION
    source: str = "custom"
    language: str = "python"
    
    system_prompt: str = ""
    behavior_prompt: str = ""
    output_schema: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'POPSSPromptConfig':
        """Create config from dictionary."""
        return cls(
            expert_type=data.get("expert_type", ""),
            version=data.get("version", "1.0.0"),
            source=data.get("source", "custom"),
            language=data.get("language", "python"),
            system_prompt=data.get("system_prompt", ""),
            behavior_prompt=data.get("behavior_prompt", ""),
            output_schema=data.get("output_schema", {}),
            parameters=data.get("parameters", {}),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
        )


class POPSSPromptLoader:
    """
    Prompt loader for YAML-based prompt management.
    
    This class provides methods to load, cache, and manage prompts
    stored in YAML files. Supports hot reload and variable formatting.
    
    Attributes:
        PROMPTS_DIR: Base directory for prompt files
        _cache: In-memory cache for loaded prompts
        _lock: Thread lock for concurrent access
    
    Usage:
        config = POPSSPromptLoader.load("code_reviewer")
        prompt = POPSSPromptLoader.format_prompt("code_reviewer", code="...")
    """
    
    PROMPTS_DIR = Path(__file__).parent / "prompts"
    
    _cache: Dict[str, POPSSPromptConfig] = {}
    _file_timestamps: Dict[str, float] = {}
    _lock = threading.RLock()
    
    _LOG: PiscesLxLogger = None
    
    @classmethod
    def _get_logger(cls) -> PiscesLxLogger:
        """Get or create logger."""
        if cls._LOG is None:
            cls._LOG = get_logger("POPSSPromptLoader")
        return cls._LOG
    
    @classmethod
    def load(cls, expert_type: str, use_cache: bool = True) -> POPSSPromptConfig:
        """
        Load prompt configuration for an expert.
        
        Args:
            expert_type: Unique identifier for the expert
            use_cache: Whether to use cached version if available
            
        Returns:
            POPSSPromptConfig for the expert
            
        Raises:
            FileNotFoundError: If prompt file not found
            yaml.YAMLError: If YAML parsing fails
        """
        with cls._lock:
            if use_cache and expert_type in cls._cache:
                return cls._cache[expert_type]
            
            prompt_file = cls._find_prompt_file(expert_type)
            
            if not prompt_file:
                raise FileNotFoundError(f"Prompt not found: {expert_type}")
            
            with open(prompt_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            config = POPSSPromptConfig.from_dict(data)
            
            # Replace {{VERSION}} placeholder with actual version
            if config.version and "{{VERSION}}" in config.version:
                from configs.version import VERSION
                config.version = config.version.replace("{{VERSION}}", VERSION)
            
            cls._cache[expert_type] = config
            cls._file_timestamps[expert_type] = prompt_file.stat().st_mtime
            
            cls._get_logger().debug(f"Loaded prompt: {expert_type}")
            return config
    
    @classmethod
    def _find_prompt_file(cls, expert_type: str) -> Optional[Path]:
        """Find prompt file by expert type."""
        for subdir in cls.PROMPTS_DIR.iterdir():
            if subdir.is_dir():
                prompt_file = subdir / f"{expert_type}.yaml"
                if prompt_file.exists():
                    return prompt_file
        return None
    
    @classmethod
    def load_all(cls, use_cache: bool = True) -> Dict[str, POPSSPromptConfig]:
        """
        Load all available prompts.
        
        Args:
            use_cache: Whether to use cached versions
            
        Returns:
            Dictionary mapping expert_type to config
        """
        all_prompts = {}
        
        with cls._lock:
            for subdir in cls.PROMPTS_DIR.iterdir():
                if subdir.is_dir() and subdir.name != "templates":
                    for prompt_file in subdir.glob("*.yaml"):
                        expert_type = prompt_file.stem
                        try:
                            all_prompts[expert_type] = cls.load(
                                expert_type, use_cache=use_cache
                            )
                        except Exception as e:
                            cls._get_logger().warning(
                                f"Failed to load {expert_type}: {e}"
                            )
        
        return all_prompts
    
    @classmethod
    def get_system_prompt(cls, expert_type: str) -> str:
        """
        Get system prompt for an expert.
        
        Args:
            expert_type: Expert identifier
            
        Returns:
            System prompt string
        """
        config = cls.load(expert_type)
        return config.system_prompt
    
    @classmethod
    def get_behavior_prompt(cls, expert_type: str) -> str:
        """
        Get behavior prompt template for an expert.
        
        Args:
            expert_type: Expert identifier
            
        Returns:
            Behavior prompt template string
        """
        config = cls.load(expert_type)
        return config.behavior_prompt
    
    @classmethod
    def get_output_schema(cls, expert_type: str) -> Dict[str, Any]:
        """
        Get output schema for an expert.
        
        Args:
            expert_type: Expert identifier
            
        Returns:
            JSON schema dictionary
        """
        config = cls.load(expert_type)
        return config.output_schema
    
    @classmethod
    def get_parameters(cls, expert_type: str) -> Dict[str, Any]:
        """
        Get configurable parameters for an expert.
        
        Args:
            expert_type: Expert identifier
            
        Returns:
            Parameters dictionary
        """
        config = cls.load(expert_type)
        return config.parameters
    
    @classmethod
    def format_prompt(
        cls, 
        expert_type: str, 
        **kwargs
    ) -> str:
        """
        Format behavior prompt with variables.
        
        Args:
            expert_type: Expert identifier
            **kwargs: Variables to substitute
            
        Returns:
            Formatted prompt string
        """
        config = cls.load(expert_type)
        behavior_prompt = config.behavior_prompt
        
        try:
            return behavior_prompt.format(**kwargs)
        except KeyError as e:
            cls._get_logger().warning(
                f"Missing variable {e} for {expert_type}"
            )
            return behavior_prompt
    
    @classmethod
    def reload(cls, expert_type: str) -> POPSSPromptConfig:
        """
        Reload prompt from file, bypassing cache.
        
        Args:
            expert_type: Expert identifier
            
        Returns:
            Freshly loaded config
        """
        with cls._lock:
            if expert_type in cls._cache:
                del cls._cache[expert_type]
            if expert_type in cls._file_timestamps:
                del cls._file_timestamps[expert_type]
        
        return cls.load(expert_type, use_cache=False)
    
    @classmethod
    def reload_all(cls) -> Dict[str, POPSSPromptConfig]:
        """
        Reload all prompts from files.
        
        Returns:
            Dictionary of all freshly loaded configs
        """
        with cls._lock:
            cls._cache.clear()
            cls._file_timestamps.clear()
        
        return cls.load_all(use_cache=False)
    
    @classmethod
    def check_for_updates(cls) -> List[str]:
        """
        Check for updated prompt files.
        
        Returns:
            List of expert types that have been updated
        """
        updated = []
        
        with cls._lock:
            for expert_type, cached_config in list(cls._cache.items()):
                prompt_file = cls._find_prompt_file(expert_type)
                
                if prompt_file:
                    current_mtime = prompt_file.stat().st_mtime
                    cached_mtime = cls._file_timestamps.get(expert_type, 0)
                    
                    if current_mtime > cached_mtime:
                        updated.append(expert_type)
        
        return updated
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available expert types.
        
        Returns:
            List of expert type identifiers
        """
        expert_types = []
        
        for subdir in cls.PROMPTS_DIR.iterdir():
            if subdir.is_dir() and subdir.name != "templates":
                for prompt_file in subdir.glob("*.yaml"):
                    expert_types.append(prompt_file.stem)
        
        return sorted(expert_types)
    
    @classmethod
    def list_by_category(cls) -> Dict[str, List[str]]:
        """
        List experts grouped by category.
        
        Returns:
            Dictionary mapping category to list of expert types
        """
        categories = {}
        
        for subdir in cls.PROMPTS_DIR.iterdir():
            if subdir.is_dir() and subdir.name != "templates":
                experts = [f.stem for f in subdir.glob("*.yaml")]
                if experts:
                    categories[subdir.name] = sorted(experts)
        
        return categories
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """
        Get information about the prompt cache.
        
        Returns:
            Cache statistics dictionary
        """
        with cls._lock:
            return {
                "cached_count": len(cls._cache),
                "cached_types": list(cls._cache.keys()),
                "file_timestamps": cls._file_timestamps.copy(),
            }
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the prompt cache."""
        with cls._lock:
            cls._cache.clear()
            cls._file_timestamps.clear()
            cls._get_logger().info("Prompt cache cleared")
    
    @classmethod
    def validate_prompt(cls, expert_type: str) -> Dict[str, Any]:
        """
        Validate a prompt configuration.
        
        Args:
            expert_type: Expert identifier
            
        Returns:
            Validation result dictionary
        """
        result = {
            "expert_type": expert_type,
            "valid": True,
            "errors": [],
            "warnings": [],
        }
        
        try:
            config = cls.load(expert_type)
            
            if not config.system_prompt:
                result["warnings"].append("Empty system_prompt")
            
            if not config.behavior_prompt:
                result["warnings"].append("Empty behavior_prompt")
            
            if not config.output_schema:
                result["warnings"].append("No output_schema defined")
            
            if not config.expert_type:
                result["errors"].append("Missing expert_type")
                result["valid"] = False
            
        except FileNotFoundError:
            result["errors"].append("Prompt file not found")
            result["valid"] = False
        except yaml.YAMLError as e:
            result["errors"].append(f"YAML parsing error: {e}")
            result["valid"] = False
        except Exception as e:
            result["errors"].append(f"Unexpected error: {e}")
            result["valid"] = False
        
        return result
    
    @classmethod
    def create_prompt_file(
        cls,
        expert_type: str,
        category: str,
        system_prompt: str,
        behavior_prompt: str = "",
        output_schema: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        source: str = "custom",
        version: str = VERSION,
    ) -> Path:
        """
        Create a new prompt file.
        
        Args:
            expert_type: Unique identifier for the expert
            category: Category directory (code, reasoning, etc.)
            system_prompt: System-level prompt
            behavior_prompt: Behavior template
            output_schema: JSON schema for output
            parameters: Configurable parameters
            source: Source identifier
            version: Prompt version
            
        Returns:
            Path to created file
        """
        category_dir = cls.PROMPTS_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        prompt_file = category_dir / f"{expert_type}.yaml"
        
        data = {
            "expert_type": expert_type,
            "version": version,
            "source": source,
            "system_prompt": system_prompt,
            "behavior_prompt": behavior_prompt,
            "output_schema": output_schema or {},
            "parameters": parameters or {},
            "metadata": {
                "created_at": datetime.now().isoformat(),
            }
        }
        
        with open(prompt_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        cls._get_logger().info(f"Created prompt file: {prompt_file}")
        return prompt_file
