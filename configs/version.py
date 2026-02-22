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

"""
Version Information Module for PiscesL1 Large Language Model Framework.

This module defines the version identifiers and release metadata for the
PiscesL1 project. Version information is used throughout the framework for
compatibility checking, logging, checkpoint management, and API versioning.

Versioning Scheme:
    The PiscesL1 project follows Semantic Versioning (SemVer) with the format:
    MAJOR.MINOR.PATCH
    
    - MAJOR: Incompatible API changes or major architectural shifts
    - MINOR: New features added in a backward-compatible manner
    - PATCH: Backward-compatible bug fixes and minor improvements

Version Types:
    VERSION (str):
        The primary framework version identifier. This version tracks the
        overall PiscesL1 framework including model architecture, training
        pipeline, inference engine, and tooling.
        
        Format: "MAJOR.MINOR.PATCH"
        Example: "1.0.0" indicates the first stable release
        
        Used for:
        - Framework compatibility checks
        - Dependency management
        - Documentation versioning
        - Release tracking
    
    CVERSION (str):
        The core model configuration version. This version specifically tracks
        changes to model architecture configurations, hyperparameter defaults,
        and training recipe modifications.
        
        Format: "MAJOR.MINOR.PATCH"
        Example: "0.3.0" indicates the third minor configuration revision
        
        Used for:
        - Model configuration compatibility
        - Checkpoint loading validation
        - Training recipe versioning
        - Configuration migration guides
        
        Increment triggers:
        - Changes to model architecture parameters
        - Modifications to default hyperparameters
        - Updates to training/evaluation configurations

Constants:
    VERSION (str): Framework version string, e.g., "1.0.0"
    CVERSION (str): Configuration version string, e.g., "0.3.0"
    AUTHOR (str): Project author/maintainer identifier

Usage Examples:
    Basic version access:
        from configs.version import VERSION, CVERSION, AUTHOR
        print(f"PiscesL1 v{VERSION} (Config v{CVERSION})")
    
    Version comparison:
        from configs.version import VERSION
        from packaging import version
        
        if version.parse(VERSION) >= version.parse("1.0.0"):
            # Use new API features
            pass
    
    Checkpoint compatibility check:
        from configs.version import CVERSION
        import json
        
        with open('checkpoint.json') as f:
            ckpt = json.load(f)
            if ckpt.get('config_version') != CVERSION:
                raise ValueError("Incompatible checkpoint version")

Integration Points:
    - manage.py: Displays version in CLI help and logs
    - model/config.py: Validates configuration version on load
    - opss/train/checkpoint.py: Embeds version in checkpoint metadata
    - tools/benchmark/: Records version in benchmark results
    - utils/dc.py: Includes version in logging context

Version History:
    1.0.0 - Initial stable release with flagship model architecture
    0.3.0 - Configuration version for initial training recipes

Note:
    When updating versions, ensure backward compatibility is maintained
    or provide migration utilities for existing checkpoints and configurations.
"""

VERSION = "1.0.0"
CVERSION = "0.3.0"
AUTHOR = "Dunimd Team"
