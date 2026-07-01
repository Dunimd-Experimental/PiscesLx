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
Version Information Module for PiscesLx Large Language Model Framework.

This module defines all version identifiers and release metadata for the PiscesLx
project. Version information is critically important for maintaining compatibility,
tracking changes, managing checkpoint compatibility, and ensuring proper integration
across the entire PiscesLx ecosystem.

PURPOSE AND IMPORTANCE:
    Version tracking serves multiple critical functions in a large-scale ML framework:

    1. Compatibility Verification:
       - Ensures model checkpoints are compatible with current framework
       - Prevents loading checkpoints trained with incompatible configurations
       - Enables proper migration paths between major versions

    2. Change Management:
       - Tracks significant changes across releases
       - Helps developers understand evolution of the codebase
       - Facilitates rollback procedures when issues are discovered

    3. Dependency Management:
       - Guides dependency version requirements
       - Helps users understand which framework features are available
       - Informs upgrade decisions for production deployments

    4. Audit and Compliance:
       - Provides immutable record of software versions in use
       - Required for regulatory compliance in enterprise settings
       - Enables reproducible research and experiments

SEMANTIC VERSIONING SCHEME:
    PiscesLx follows Semantic Versioning (SemVer) with the format: MAJOR.MINOR.PATCH

    MAJOR Version:
        - Incremented when making incompatible API changes
        - Includes major architectural shifts in model design
        - Breaking changes to configuration schemas
        - Changes that require manual migration steps
        Example: 1.0.0 -> 2.0.0 indicates fundamental restructuring

    MINOR Version:
        - Incremented when adding functionality in a backward-compatible manner
        - New model architecture variants or sizes
        - Additional training stages or algorithms
        - New configuration options that don't break existing ones
        Example: 1.0.0 -> 1.1.0 indicates feature additions

    PATCH Version:
        - Incremented when making backward-compatible bug fixes
        - Performance improvements without API changes
        - Documentation updates
        - Minor configuration parameter tweaks for stability
        Example: 1.0.0 -> 1.0.1 indicates patch releases

VERSION IDENTIFIERS:

    VERSION (str):
        Primary framework version identifier tracking the overall PiscesLx system.

        Content Scope:
        - Model architecture implementation and variants
        - Training pipeline and optimization algorithms
        - Inference engine and serving infrastructure
        - APIs, utilities, and tooling
        - Configuration schema definitions
        - Documentation and examples

        Update Triggers:
        - Changes to model architecture (new attention mechanisms, etc.)
        - Modifications to training algorithms or optimization strategies
        - Breaking changes to public APIs
        - Changes to configuration file structure
        - New model size variants (0.5B, 1.5B, 7B, etc.)
        - Significant performance optimizations

        Usage Examples:
        # Version compatibility check
        from configs.version import VERSION
        if version.parse(VERSION) >= version.parse("2.0.0"):
            use_new_api()

        # Logging and diagnostics
        logger.info(f"PiscesLx Framework v{VERSION}")

        # Dependency management
        requirements = [f"piscesl1=={VERSION}"]

    CVERSION (str):
        Core model configuration version tracking model-specific parameters.

        Content Scope:
        - Model hyperparameters (hidden_size, n_layer, n_head, etc.)
        - MoE configuration (num_experts, top_k, capacity_factor)
        - Training recipe defaults (learning_rate, batch_size, etc.)
        - Knowledge density optimization parameters
        - Attention mechanism configurations

        Update Triggers:
        - Changes to any model architecture parameter
        - Modifications to default hyperparameter values
        - Updates to MoE routing algorithms
        - Changes to knowledge density optimization settings
        - Adjustments to attention mechanism configurations

        Usage Examples:
        # Checkpoint compatibility validation
        from configs.version import CVERSION
        checkpoint = load_checkpoint('model.pt')
        if checkpoint['config_version'] != CVERSION:
            raise CompatibilityError("Checkpoint version mismatch")

        # Configuration migration
        if config['cversion'] < CVERSION:
            config = migrate_config(config, target=CVERSION)

        # Training recipe versioning
        from configs.version import CVERSION
        logger.info(f"Training with config version {CVERSION}")

VERSION HISTORY AND CHANGELOG:

    Version 1.0.0:
        - Initial stable release of PiscesLx framework
        - Flagship model architecture with MoE and knowledge density optimization
        - Complete training pipeline with pre-training, SFT, and alignment stages
        - Support for model sizes: 0.5B, 1.5B, 7B, 32B, 64B, 70B, 128B, 314B, 671B, 1T
        - Integrated watermarking system for AI content traceability
        - Multimodal support for text, image, audio, and video processing

    Version 0.3.2 (Current CVERSION):
        - Third patch release for configuration stability
        - Refined MoE routing temperature schedules
        - Updated knowledge density optimization weights
        - Optimized attention residuals for deeper networks

    Version 0.3.0:
        - Second minor release for configuration schema
        - Added support for new alignment algorithms (ORPO)
        - Enhanced multimodal configuration options
        - Improved dataset registry with more sources

    Version 0.2.0:
        - Initial configuration versioning started
        - Base configuration schema for all model sizes
        - Training stage configurations established

INTEGRATION POINTS:

    Framework Entry Points:
        - manage.py: CLI tool displays version in help output and logs
        - __init__.py: Exports VERSION and CVERSION for public access

    Model Module:
        - model/config.py: Validates CVERSION on configuration load
        - model/architecture.py: Uses version for feature flags

    Training Module:
        - opss/train/checkpoint.py: Embeds CVERSION in checkpoint metadata
        - opss/train/validator.py: Checks version compatibility before training

    Inference Module:
        - opss/inference/engine.py: Uses VERSION for API compatibility
        - opss/inference/sampling.py: Version-specific sampling optimizations

    Utilities:
        - utils/logging.py: Includes version in log headers
        - utils/monitoring.py: Tracks version in metrics and diagnostics
        - tools/benchmark/: Records version in benchmark results

CHECKPOINT VERSIONING:

    Each model checkpoint includes metadata with version information:

    checkpoint_metadata = {
        "version": VERSION,           # Framework version
        "config_version": CVERSION,   # Configuration version
        "model_type": "piscesl1_7b", # Model variant
        "training_steps": 100000,     # Training progress
        "timestamp": "2026-01-15",    # Creation time
        "git_commit": "abc123",       # Source commit
    }

    Version Compatibility Rules:
    1. Checkpoint VERSION must match current framework VERSION for loading
    2. If VERSION matches but CVERSION differs, migration may be required
    3. Major VERSION mismatch requires full migration process
    4. Configuration files include their target VERSION and CVERSION

MIGRATION PROCEDURES:

    When version incompatibility is detected:

    1. Minor CVERSION Difference:
       - Automatic migration of configuration parameters
       - Parameter mapping and transformation
       - Validation of migrated configuration

    2. Major CVERSION Difference:
       - Manual review of configuration changes required
       - Documentation of breaking changes
       - Step-by-step migration guide provided

    3. Major VERSION Difference:
       - Full checkpoint regeneration recommended
       - If not possible, contact support for assistance
       - Extensive testing required after migration

DEVELOPMENT WORKFLOW:

    Version Update Process:
    1. Developer identifies need for version change
    2. Changes documented in CHANGELOG with rationale
    3. Pull request includes version update
    4. Code review verifies version bump is appropriate
    5. CI/CD validates no breaking changes (for minor/patch)
    6. Release process publishes new version

    Version Pinning:
    - Production deployments should pin to specific VERSION
    - Configuration files should specify target CVERSION
    - Checkpoints should record both VERSION and CVERSION

TESTING AND VALIDATION:

    Version-related testing includes:
    - Loading checkpoints with matching versions
    - Rejecting checkpoints with mismatched versions
    - Verifying version string format compliance
    - Testing migration procedures between versions
    - Validating version-dependent feature behavior

COMPLIANCE AND AUDIT:

    For enterprise and regulated environments:
    - Version information included in all audit logs
    - Checkpoint metadata includes version provenance
    - Configuration snapshots preserved with timestamps
    - Version history available for investigation

AUTHOR AND MAINTAINANCE:

    AUTHOR (str):
        Project author/maintainer identifier for the PiscesLx project.

        Purpose:
        - Primary contact for project-related inquiries
        - Attribution in academic citations
        - Bug reports and security vulnerabilities
        - Licensing and legal matters

        Current Value: "Dunimd Team"
        - Dunimd Team is the official maintainer
        - Responsible for releases, security, and documentation

BEST PRACTICES:

    1. Always Check Versions:
       from configs.version import VERSION, CVERSION
       assert config['version'] == CVERSION

    2. Log Version Information:
       logger.info(f"PiscesLx v{VERSION} (Config v{CVERSION})")

    3. Pin Versions in Production:
       requirements = [f"piscesl1=={VERSION}"]

    4. Record Versions in Experiments:
       experiment_metadata = {"framework_version": VERSION, "config_version": CVERSION}

    5. Validate Checkpoints:
       def load_checkpoint(path):
           ckpt = torch.load(path)
           assert ckpt['version'] == VERSION
           assert ckpt['config_version'] == CVERSION
           return ckpt

TROUBLESHOOTING:

    Common Issues and Resolutions:

    Issue: "Version mismatch when loading checkpoint"
    Cause: Checkpoint was saved with different framework version
    Solution: Use matching framework version or migrate checkpoint

    Issue: "Configuration version unknown"
    Cause: Configuration file targets future version
    Solution: Upgrade framework or regenerate configuration

    Issue: "AttributeError: module 'configs.version' has no attribute 'CVERSION'"
    Cause: Using outdated configs package
    Solution: Update piscesl1 package to latest version

REFERENCES:

    - Semantic Versioning Specification: https://semver.org/
    - Python Versioning Standards: PEP 440
    - ML Model Checkpoint Best Practices
    - Configuration Management in Large-Scale ML Systems
"""

VERSION = "1.0.0"
CVERSION = "0.4.0"
AUTHOR = "Dunimd Team"