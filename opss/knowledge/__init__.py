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

from __future__ import annotations

"""
Knowledge Module

Offline knowledge store construction for Engram-style
Lookup-Computation Separation.

Components:
    - PiscesLxKnowledgeBuilder: Main builder pipeline
    - PiscesLxKnowledgeBuilderConfig: Builder configuration
    - _PiscesLxKnowledgeEncoder: Fixed 0.5B encoder (640-dim, 16 layers, 4 MoE)
    - _PiscesLxNTXentLoss: Contrastive learning loss

Pipeline:
    Text -> Chunking -> 0.5B Encoder -> 256-dim Projection
    -> NT-Xent Contrastive Refinement -> FAISS IVF-PQ Index
    -> mmap Knowledge Store

Usage:
    from opss.knowledge import PiscesLxKnowledgeBuilder, PiscesLxKnowledgeBuilderConfig

    config = PiscesLxKnowledgeBuilderConfig(
        knowledge_slots=100_000_000,
        store_path="./knowledge_store/7B",
    )
    builder = PiscesLxKnowledgeBuilder(config)
    builder.build_from_texts(texts=["...", "..."])
"""

from .builder import PiscesLxBuilder, PiscesLxKnowledgeBuilder, PiscesLxKnowledgeBuilderConfig

__all__ = [
    "PiscesLxKnowledgeBuilder",
    "PiscesLxKnowledgeBuilderConfig",
    "create_knowledge_builder",
]