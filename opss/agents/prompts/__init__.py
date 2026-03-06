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
PiscesL1 Expert Prompts Module

This module contains YAML-based prompt configurations for expert agents.
Prompts are organized by category:

Categories:
    - code: Code generation, review, debugging, testing
    - reasoning: Logical, mathematical, causal reasoning
    - evaluation: Architecture, performance, quality evaluation
    - planning: Task, resource, time planning
    - analysis: Data, text, log analysis
    - search: Web, code, semantic search
    - memory: Short-term, long-term, working memory
    - tool: Shell, file, database operations
    - collaboration: Coordination, dispatching, aggregation
    - domain: Math, physics, chemistry, biology, legal, medical
    - templates: Base templates and output schemas

Usage:
    from opss.agents.loader import POPSSPromptLoader
    
    # Load a specific prompt
    config = POPSSPromptLoader.load("code_reviewer")
    
    # List all available prompts
    prompts = POPSSPromptLoader.list_available()
    
    # List by category
    categories = POPSSPromptLoader.list_by_category()
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

CATEGORIES = [
    "code",
    "reasoning", 
    "evaluation",
    "planning",
    "analysis",
    "search",
    "memory",
    "tool",
    "collaboration",
    "domain",
    "templates",
]

__all__ = [
    "PROMPTS_DIR",
    "CATEGORIES",
]
