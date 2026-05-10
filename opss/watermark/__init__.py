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
import importlib
from pathlib import Path

from configs.version import VERSION, AUTHOR

_LAZY_SYMBOLS = {
    "POPSSWatermarkJurisdiction": (".config", "POPSSWatermarkJurisdiction"),
    "POPSSComplianceStandard": (".config", "POPSSComplianceStandard"),
    "POPSSWatermarkRiskLevel": (".config", "POPSSWatermarkRiskLevel"),
    "POPSSWatermarkContentType": (".config", "POPSSWatermarkContentType"),
    "POPSSWatermarkConfig": (".config", "POPSSWatermarkConfig"),
    "POPSSWatermarkPayload": (".config", "POPSSWatermarkPayload"),
    "POPSSWatermarkAuditRecord": (".config", "POPSSWatermarkAuditRecord"),
    "POPSSWatermarkDefaultConfigFactory": (".config", "POPSSWatermarkDefaultConfigFactory"),
    "POPSSWatermarkComplianceValidator": (".config", "POPSSWatermarkComplianceValidator"),
    "get_default_config": (".config", "get_default_config"),
    "validate_compliance": (".config", "validate_compliance"),
    "POPSSWatermarkProtocolOperator": (".protocol_operator", "POPSSWatermarkProtocolOperator"),
    "POPSSFrameInfo": (".protocol_operator", "POPSSFrameInfo"),
    "create_protocol_operator": (".protocol_operator", "create_protocol_operator"),
    "POPSSWatermarkDCTOperator": (".dct_operator", "POPSSWatermarkDCTOperator"),
    "POPSSWatermarkContentOperator": (".content_watermark_operator", "POPSSWatermarkContentOperator"),
    "POPSSContentWatermarkOperator": (".content_watermark_operator", "POPSSContentWatermarkOperator"),
    "create_content_watermark_operator": (".content_watermark_operator", "create_content_watermark_operator"),
    "POPSSWatermarkWeightOperator": (".weight_watermark_operator", "POPSSWatermarkWeightOperator"),
    "POPSSWeightWatermarkOperator": (".weight_watermark_operator", "POPSSWeightWatermarkOperator"),
    "create_weight_watermark_operator": (".weight_watermark_operator", "create_weight_watermark_operator"),
    "POPSSWatermarkComplianceOperator": (".compliance_operator", "POPSSWatermarkComplianceOperator"),
    "POPSSComplianceOperator": (".compliance_operator", "POPSSComplianceOperator"),
    "create_compliance_operator": (".compliance_operator", "create_compliance_operator"),
    "POPSSWatermarkAuditOperator": (".audit_operator", "POPSSWatermarkAuditOperator"),
    "POPSSAuditOperator": (".audit_operator", "POPSSAuditOperator"),
    "create_audit_operator": (".audit_operator", "create_audit_operator"),
    "POPSSWatermarkOrchestrator": (".orchestrator", "POPSSWatermarkOrchestrator"),
    "create_watermark_orchestrator": (".orchestrator", "create_watermark_orchestrator"),
}


def __getattr__(name):
    if name in _LAZY_SYMBOLS:
        submod, attr = _LAZY_SYMBOLS[name]
        module = importlib.import_module(submod, __name__)
        val = getattr(module, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
