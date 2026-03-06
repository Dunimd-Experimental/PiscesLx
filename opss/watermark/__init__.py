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
PiscesL1 Watermark Operators

This module provides comprehensive watermark operators for AI-generated content
provenance, ownership verification, and regulatory compliance.

Submodules:
    - config: Configuration classes for watermark settings and compliance
    - protocol_operator: Watermark framing protocol with SYNC+LEN+CRC32
    - dct_operator: DCT-based image watermarking
    - content_watermark_operator: Multi-modal content watermarking
    - weight_watermark_operator: Model weight watermarking
    - compliance_operator: Legal compliance validation
    - audit_operator: Audit trail management
    - orchestrator: Unified watermark orchestration

Supported Regulations:
    - GB/T 45225-2024 (China)
    - AI Act 2024 (European Union)
    - NIST AI RMF 1.0 (United States)
    - AI Safety Act 2024 (United Kingdom)
    - ISO/IEC 27090 (International)

Usage Examples:
    >>> from opss.watermark import (
    ...     POPSSWatermarkConfig,
    ...     POPSSWatermarkContentOperator,
    ...     POPSSWatermarkWeightOperator,
    ...     POPSSWatermarkOrchestrator
    ... )
    >>> 
    >>> # Configure watermark system
    >>> config = POPSSWatermarkConfig(
    ...     standard=POPSSComplianceStandard.GB_T_45225_2024,
    ...     jurisdiction=POPSSJurisdiction.CN,
    ...     watermark_strength=1e-5
    ... )
    >>> 
    >>> # Create orchestrator
    >>> orchestrator = POPSSWatermarkOrchestrator(config)
    >>> 
    >>> # Embed watermark
    >>> result = orchestrator.embed("Hello, World!", user_id="user123")

"""

import sys
from pathlib import Path

from configs.version import VERSION, AUTHOR

from .config import (
    POPSSWatermarkJurisdiction,
    POPSSComplianceStandard,
    POPSSWatermarkRiskLevel,
    POPSSWatermarkContentType,
    POPSSWatermarkConfig,
    POPSSWatermarkPayload,
    POPSSWatermarkAuditRecord,
    POPSSWatermarkDefaultConfigFactory,
    POPSSWatermarkComplianceValidator,
    get_default_config,
    validate_compliance
)

from .protocol_operator import (
    POPSSWatermarkProtocolOperator,
    POPSSFrameInfo,
    create_protocol_operator
)

from .dct_operator import (
    POPSSWatermarkDCTOperator
)

from .content_watermark_operator import (
    POPSSWatermarkContentOperator,
    POPSSContentWatermarkOperator,
    create_content_watermark_operator
)

from .weight_watermark_operator import (
    POPSSWatermarkWeightOperator,
    POPSSWeightWatermarkOperator,
    create_weight_watermark_operator
)

from .compliance_operator import (
    POPSSWatermarkComplianceOperator,
    POPSSComplianceOperator,
    create_compliance_operator
)

from .audit_operator import (
    POPSSWatermarkAuditOperator,
    POPSSAuditOperator,
    create_audit_operator
)

from .orchestrator import (
    POPSSWatermarkOrchestrator,
    create_watermark_orchestrator
)

__all__ = [
    "POPSSWatermarkJurisdiction",
    "POPSSComplianceStandard",
    "POPSSWatermarkRiskLevel",
    "POPSSWatermarkContentType",
    "POPSSWatermarkConfig",
    "POPSSWatermarkPayload",
    "POPSSWatermarkAuditRecord",
    "POPSSWatermarkDefaultConfigFactory",
    "POPSSWatermarkComplianceValidator",
    "POPSSWatermarkProtocolOperator",
    "POPSSFrameInfo",
    "POPSSWatermarkDCTOperator",
    "POPSSWatermarkContentOperator",
    "POPSSContentWatermarkOperator",
    "POPSSWatermarkWeightOperator",
    "POPSSWeightWatermarkOperator",
    "POPSSWatermarkComplianceOperator",
    "POPSSComplianceOperator",
    "POPSSWatermarkAuditOperator",
    "POPSSAuditOperator",
    "POPSSWatermarkOrchestrator",
    "get_default_config",
    "validate_compliance",
    "create_protocol_operator",
    "create_content_watermark_operator",
    "create_weight_watermark_operator",
    "create_compliance_operator",
    "create_audit_operator",
    "create_watermark_orchestrator",
]

__version__ = VERSION
__author__ = AUTHOR
