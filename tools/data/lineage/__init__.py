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
PiscesL1 Data Lineage Tracking Module.

This module provides data lineage tracking capabilities:
- LineageTracker: Track data provenance and processing history
- LineageRecord: Individual sample lineage record

Key Features:
    - Complete data provenance tracking
    - Transformation history recording
    - Quality score tracking
    - Serialization and export

Usage:
    >>> from tools.data.lineage import PiscesLxDataLineageTracker
    >>> tracker = PiscesLxDataLineageTracker()
    >>> tracker.record_source("doc_001", "huggingface", "wikitext")
    >>> tracker.record_transformation("doc_001", "clean", {"lowercase": True})
    >>> tracker.record_quality("doc_001", 0.92)
    >>> report = tracker.generate_report("doc_001")
    >>> tracker.export_json("lineage_report.json")
"""

from .tracker import PiscesLxDataLineageTracker, PiscesLxDataLineageRecord

__all__ = [
    'PiscesLxDataLineageTracker',
    'PiscesLxDataLineageRecord',
]
