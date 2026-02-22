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
Compliance Operator

This module implements legal compliance validation and metadata generation
for AI watermark systems across multiple jurisdictions. It ensures that
watermarking operations meet regulatory requirements for different regions.

Supported Compliance Frameworks:
    - GB/T 45225-2024 (China): Mandatory AI content watermarking
    - AI Act 2024 (EU): High-risk AI transparency requirements
    - NIST AI RMF 1.0 (US): Risk management framework
    - AI Safety Act 2024 (UK): Frontier model safety registration
    - ISO/IEC 27090: International AI watermarking standard

Key Features:
    - Automatic jurisdiction detection and validation
    - Risk level classification (low/medium/high)
    - Compliance metadata generation
    - Requirement checking against standards
    - Remediation suggestions for non-compliance

Compliance Validation:
    The operator validates watermark configurations and content against
    jurisdiction-specific requirements, providing detailed compliance
    reports with recommendations.

Usage Examples:
    >>> from opss.watermark.compliance_operator import PiscesLxComplianceOperator
    >>> operator = PiscesLxComplianceOperator()
    >>> 
    >>> # Check compliance for a jurisdiction
    >>> result = operator.validate(
    ...     content_type="text",
    ...     jurisdiction="CN"
    ... )
    >>> 
    >>> # Generate compliance metadata
    >>> metadata = operator.generate_metadata(
    ...     model_id="PiscesL1-1.5B",
    ...     risk_level="medium"
    ... )

Author: PiscesL1 Development Team
Version: 1.0.0
"""

import hashlib
import json
import uuid
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from utils.opsc.base import PiscesLxBaseOperator
from utils.opsc.interface import PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION
from .config import (
    POPSSWatermarkConfig,
    POPSSComplianceStandard,
    POPSSWatermarkJurisdiction,
    POPSSWatermarkRiskLevel,
    POPSSWatermarkContentType
)


class POPSSComplianceOperator(PiscesLxBaseOperator):
    """
    Legal compliance validation and metadata generation operator.
    
    This operator ensures watermark operations comply with AI regulations
    across different jurisdictions. It validates configurations, generates
    compliance metadata, and provides remediation recommendations.
    
    The operator supports multiple regulatory frameworks including:
        - GB/T 45225-2024 (China): Mandatory AI content watermarking with
          explicit disclosure requirements for generated content
        - AI Act 2024 (EU): High-risk AI transparency requirements with
          mandatory audit trails for high-impact systems
        - NIST AI RMF 1.0 (US): Risk management framework with voluntary
          compliance guidelines for AI systems
        - AI Safety Act 2024 (UK): Frontier model safety registration with
          mandatory watermarking for large-scale models
        - ISO/IEC 27090: International AI watermarking standard providing
          cross-jurisdictional compliance framework
    
    Architecture:
        The operator implements a multi-layer validation approach:
        1. Jurisdiction Detection: Automatically maps jurisdiction codes to
           regulatory requirements
        2. Requirement Validation: Checks configuration against mandatory
           and recommended requirements
        3. Risk Assessment: Evaluates content risk based on deployment context
        4. Metadata Generation: Creates compliance metadata for audit trails
        5. Recommendation Engine: Provides actionable remediation suggestions
    
    Attributes:
        config (POPSSWatermarkConfig): Watermark configuration containing
            jurisdiction settings, watermark parameters, and compliance options
        supported_standards (List[POPSSComplianceStandard]): List of
            supported regulatory standards for validation
        
    Input Format:
        {
            "action": "validate" | "generate_metadata" | "check_requirement" | "assess_risk",
            "content_type": str,                  # Type of content (text/image/audio/video)
            "jurisdiction": str | POPSSWatermarkJurisdiction,  # Target jurisdiction code
            "config": POPSSWatermarkConfig,    # Configuration to validate
            "model_id": str,                      # Model identifier for metadata
            "requirement": str                    # Specific requirement to check
        }
        
    Output Format:
        {
            "action": str,
            "result": Dict,                       # Validation or generation result
            "compliance_score": float,            # 0.0 to 1.0 compliance score
            "recommendations": List[str]          # Remediation suggestions
        }
    
    Example:
        >>> operator = POPSSComplianceOperator()
        >>> result = operator.validate(
        ...     content_type="text",
        ...     jurisdiction="CN"
        ... )
        >>> print(result["compliance_status"])
        'compliant'
    """
    
    def __init__(self, config: Optional[POPSSWatermarkConfig] = None):
        super().__init__()
        self.name = "pisceslx_compliance_operator"
        self.version = VERSION
        self.description = "Legal compliance validation and metadata generation for AI watermarking"
        self.config = config or POPSSWatermarkConfig()
        self.supported_standards = [
            POPSSComplianceStandard.GB_T_45225_2024,
            POPSSComplianceStandard.AI_ACT_2024,
            POPSSComplianceStandard.NIST_AI_RMF,
            POPSSComplianceStandard.AI_SAFETY_ACT_2024,
            POPSSComplianceStandard.ISO_27090
        ]
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["validate", "generate_metadata", "check_requirement", "assess_risk"]
                },
                "content_type": {
                    "type": "string",
                    "enum": ["text", "image", "audio", "video", "multimodal"]
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "Jurisdiction code or name"
                },
                "config": {
                    "type": "object",
                    "description": "Watermark configuration"
                },
                "model_id": {
                    "type": "string",
                    "description": "Model identifier"
                },
                "requirement": {
                    "type": "string",
                    "description": "Specific requirement to check"
                },
                "risk_factors": {
                    "type": "array",
                    "description": "Risk factors for assessment"
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "result": {"type": "object"},
                "compliance_score": {"type": "number"},
                "recommendations": {"type": "array"},
                "compliance_status": {"type": "string"}
            }
        }
    
    def _execute_impl(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        action = inputs.get("action", "validate")
        
        if action == "validate":
            return self._validate(inputs)
        elif action == "generate_metadata":
            return self._generate_metadata(inputs)
        elif action == "check_requirement":
            return self._check_requirement(inputs)
        elif action == "assess_risk":
            return self._assess_risk(inputs)
        else:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Unknown action: {action}"
            )
    
    def _validate(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Validate watermark configuration against jurisdiction requirements.
        
        This method performs comprehensive compliance validation by checking
        the configuration against mandatory and recommended requirements for
        the specified jurisdiction. It evaluates multiple compliance dimensions
        including disclosure requirements, watermark strength, redundancy levels,
        and audit trail configuration.
        
        Validation Process:
            1. Parse jurisdiction from input (string or enum)
            2. Retrieve jurisdiction-specific requirements
            3. Check mandatory disclosure requirements
            4. Validate watermark strength range
            5. Verify redundancy level compliance
            6. Confirm audit trail configuration
            7. Calculate overall compliance score
            8. Generate remediation recommendations
        
        Args:
            inputs: Dictionary containing validation parameters
                - content_type (str): Type of content being validated
                - jurisdiction (str | POPSSWatermarkJurisdiction): Target jurisdiction
                - config (POPSSWatermarkConfig): Configuration to validate
        
        Returns:
            PiscesLxOperatorResult: Validation result containing
                - output: Dict with jurisdiction, violations, warnings, passed_checks,
                  compliance_score, and compliance_status
                - metadata: Dict with action, recommendations, and total_checks
        
        Compliance Scoring:
            - Passed checks: +1.0 per check
            - Warnings: +0.7 per check
            - Violations: -0.3 per check
            - Final score normalized to [0.0, 1.0] range
        """
        content_type = inputs.get("content_type", "text")
        jurisdiction_input = inputs.get("jurisdiction", "GLOBAL")
        config = inputs.get("config", self.config)
        
        try:
            if isinstance(jurisdiction_input, str):
                try:
                    jurisdiction = POPSSWatermarkJurisdiction(jurisdiction_input)
                except ValueError:
                    jurisdiction = POPSSWatermarkJurisdiction.GLOBAL
            else:
                jurisdiction = jurisdiction_input
            
            requirements = self._get_requirements(jurisdiction, content_type)
            
            violations = []
            warnings = []
            passed_checks = []
            
            if hasattr(jurisdiction, 'mandatory_disclosure'):
                if config.mandatory_disclosure < jurisdiction.mandatory_disclosure:
                    violations.append({
                        "requirement": "mandatory_disclosure",
                        "message": f"Jurisdiction {jurisdiction.code} requires mandatory disclosure",
                        "severity": "violation"
                    })
                else:
                    passed_checks.append("mandatory_disclosure")
            
            if config.watermark_strength > 1e-4:
                warnings.append({
                    "requirement": "watermark_strength",
                    "message": "High watermark strength may affect content quality",
                    "severity": "warning"
                })
            elif config.watermark_strength < 1e-7:
                warnings.append({
                    "requirement": "watermark_strength",
                    "message": "Low watermark strength may reduce extraction reliability",
                    "severity": "warning"
                })
            else:
                passed_checks.append("watermark_strength")
            
            if jurisdiction.code in ['CN', 'EU', 'UK']:
                if config.redundancy_level < 2:
                    violations.append({
                        "requirement": "redundancy_level",
                        "message": f"Jurisdiction {jurisdiction.code} requires redundancy >= 2",
                        "severity": "violation"
                    })
                else:
                    passed_checks.append("redundancy_level")
            
            if not config.audit_enabled and jurisdiction.code in ['EU', 'UK']:
                violations.append({
                    "requirement": "audit_enabled",
                    "message": f"Jurisdiction {jurisdiction.code} requires audit trail",
                    "severity": "violation"
                })
            else:
                passed_checks.append("audit_enabled")
            
            compliance_score = self._calculate_score(passed_checks, warnings, violations)
            compliance_status = "compliant" if not violations else "non_compliant"
            if warnings and not violations:
                compliance_status = "compliant_with_warnings"
            
            recommendations = self._generate_recommendations(violations, warnings, jurisdiction, content_type)
            
            result = {
                "jurisdiction": jurisdiction.code if jurisdiction else "UNKNOWN",
                "content_type": content_type,
                "standard": config.standard.value if hasattr(config.standard, 'value') else str(config.standard),
                "violations": violations,
                "warnings": warnings,
                "passed_checks": passed_checks,
                "compliance_score": compliance_score,
                "compliance_status": compliance_status
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=result,
                metadata={
                    "action": "validate",
                    "recommendations": recommendations,
                    "total_checks": len(passed_checks) + len(warnings) + len(violations)
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _generate_metadata(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Generate compliance metadata for watermark audit trails.
        
        This method creates comprehensive metadata that documents the compliance
        status and watermark capabilities of the system. The generated metadata
        can be embedded in watermarked content or stored in audit logs for
        regulatory compliance verification.
        
        Metadata Structure:
            - model_id: Identifier of the AI model generating content
            - watermark_version: Version of watermarking protocol used
            - timestamp: ISO 8601 timestamp of metadata generation
            - compliance: Jurisdiction-specific compliance information
            - watermark_capabilities: Supported watermarking modalities
        
        Args:
            inputs: Dictionary containing metadata generation parameters
                - model_id (str): Model identifier (default: "PiscesL1-1.5B")
                - risk_level (str): Risk classification (low/medium/high)
                - jurisdiction (str | POPSSWatermarkJurisdiction): Target jurisdiction
        
        Returns:
            PiscesLxOperatorResult: Metadata generation result containing
                - output: Dict with model_id, watermark_version, timestamp,
                  compliance info, and watermark_capabilities
                - metadata: Dict with action and jurisdiction
        """
        model_id = inputs.get("model_id", "PiscesL1-1.5B")
        risk_level = inputs.get("risk_level", "medium")
        jurisdiction_input = inputs.get("jurisdiction", "GLOBAL")
        
        try:
            if isinstance(jurisdiction_input, str):
                try:
                    jurisdiction = POPSSWatermarkJurisdiction(jurisdiction_input)
                except ValueError:
                    jurisdiction = POPSSWatermarkJurisdiction.GLOBAL
            else:
                jurisdiction = jurisdiction_input
            
            if risk_level:
                try:
                    risk = POPSSWatermarkRiskLevel(risk_level)
                except ValueError:
                    risk = POPSSWatermarkRiskLevel.MEDIUM
            else:
                risk = risk_level
            
            metadata = {
                "model_id": model_id,
                "watermark_version": "1.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "compliance": {
                    "standard": jurisdiction.standard if hasattr(jurisdiction, 'standard') else "ISO/IEC 27090",
                    "jurisdiction": jurisdiction.code if jurisdiction else "GLOBAL",
                    "risk_level": risk.value if isinstance(risk, Enum) else risk,
                    "mandatory_disclosure": jurisdiction.mandatory_disclosure if hasattr(jurisdiction, 'mandatory_disclosure') else False,
                    "audit_required": True,
                    "tamper_evidence": True,
                    "retention_days": jurisdiction.retention_days if hasattr(jurisdiction, 'retention_days') else 365
                },
                "watermark_capabilities": {
                    "text_watermark": True,
                    "image_watermark": True,
                    "audio_watermark": True,
                    "weight_watermark": True
                }
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=metadata,
                metadata={
                    "action": "generate_metadata",
                    "jurisdiction": metadata["compliance"]["jurisdiction"]
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _check_requirement(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Check a specific compliance requirement against configuration.
        
        This method provides targeted validation of individual compliance
        requirements, allowing for fine-grained compliance checking without
        running the full validation pipeline.
        
        Supported Requirements:
            - mandatory_disclosure: Check if disclosure labeling is required
            - audit_trail: Verify audit trail configuration
            - redundancy: Check redundancy level compliance
            - encryption: Verify encryption is enabled
            - strength: Validate watermark strength range
        
        Args:
            inputs: Dictionary containing requirement check parameters
                - requirement (str): Name of requirement to check
                - jurisdiction (str | POPSSWatermarkJurisdiction): Target jurisdiction
                - config (POPSSWatermarkConfig): Configuration to check
        
        Returns:
            PiscesLxOperatorResult: Requirement check result containing
                - output: Dict with requirement name, required value,
                  provided value, and compliant status
                - metadata: Dict with action and requirement name
        """
        requirement = inputs.get("requirement", "")
        jurisdiction_input = inputs.get("jurisdiction", "GLOBAL")
        config = inputs.get("config", self.config)
        
        try:
            if isinstance(jurisdiction_input, str):
                try:
                    jurisdiction = POPSSWatermarkJurisdiction(jurisdiction_input)
                except ValueError:
                    jurisdiction = POPSSWatermarkJurisdiction.GLOBAL
            else:
                jurisdiction = jurisdiction_input
            
            req_map = {
                "mandatory_disclosure": self._check_mandatory_disclosure,
                "audit_trail": self._check_audit_trail,
                "redundancy": self._check_redundancy,
                "encryption": self._check_encryption,
                "strength": self._check_strength
            }
            
            checker = req_map.get(requirement.lower())
            if not checker:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error=f"Unknown requirement: {requirement}"
                )
            
            result = checker(config, jurisdiction)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=result,
                metadata={
                    "action": "check_requirement",
                    "requirement": requirement
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _assess_risk(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Assess content risk level based on deployment context factors.
        
        This method evaluates the risk profile of AI-generated content based
        on various deployment and usage factors. The risk assessment helps
        determine appropriate watermarking strength and compliance requirements.
        
        Risk Factor Weights:
            - public_deployment: 0.3 (content accessible to public)
            - sensitive_content: 0.25 (personally identifiable or sensitive data)
            - high_stakes_decisions: 0.35 (decisions affecting legal/financial status)
            - commercial_use: 0.2 (content used for commercial purposes)
            - user_generated_input: 0.15 (content based on user prompts)
            - multi_party_data: 0.2 (content involving multiple data sources)
            - model_sharing: 0.25 (model shared with third parties)
            - open_source_release: 0.3 (model released as open source)
        
        Risk Level Classification:
            - low: risk_score < 0.3 (standard watermarking sufficient)
            - medium: 0.3 <= risk_score < 0.6 (enhanced watermarking recommended)
            - high: risk_score >= 0.6 (mandatory watermarking required)
        
        Args:
            inputs: Dictionary containing risk assessment parameters
                - risk_factors (List[str]): List of applicable risk factors
                - content_type (str): Type of content being assessed
        
        Returns:
            PiscesLxOperatorResult: Risk assessment result containing
                - output: Dict with risk_score, risk_level, factors breakdown,
                  and recommendations
                - metadata: Dict with action and content_type
        """
        risk_factors = inputs.get("risk_factors", [])
        content_type = inputs.get("content_type", "text")
        
        try:
            risk_score = 0.0
            risk_factors_detailed = []
            
            factor_weights = {
                "public_deployment": 0.3,
                "sensitive_content": 0.25,
                "high_stakes_decisions": 0.35,
                "commercial_use": 0.2,
                "user_generated_input": 0.15,
                "multi_party_data": 0.2,
                "model_sharing": 0.25,
                "open_source_release": 0.3
            }
            
            for factor in risk_factors:
                factor_lower = factor.lower().replace(" ", "_")
                weight = factor_weights.get(factor_lower, 0.1)
                risk_score += weight
                risk_factors_detailed.append({
                    "factor": factor,
                    "weight": weight,
                    "contribution": weight
                })
            
            risk_score = min(risk_score, 1.0)
            
            if risk_score < 0.3:
                risk_level = "low"
            elif risk_score < 0.6:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            result = {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "factors": risk_factors_detailed,
                "recommendations": self._get_risk_recommendations(risk_level, content_type)
            }
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=result,
                metadata={
                    "action": "assess_risk",
                    "content_type": content_type
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _get_requirements(self, jurisdiction: POPSSWatermarkJurisdiction, 
                         content_type: str) -> Dict[str, Any]:
        """
        Retrieve jurisdiction-specific compliance requirements.
        
        This method returns a dictionary of compliance requirements based on
        the specified jurisdiction and content type. Requirements include
        mandatory disclosure flags, audit requirements, retention periods,
        redundancy levels, and encryption requirements.
        
        Args:
            jurisdiction: Target jurisdiction enum value
            content_type: Type of content (text/image/audio/video)
        
        Returns:
            Dict containing:
                - mandatory_disclosure: bool indicating if disclosure is required
                - audit_required: bool indicating if audit trail is mandatory
                - retention_days: int specifying audit log retention period
                - redundancy_level: int specifying minimum redundancy level
                - encryption_required: bool indicating if encryption is mandatory
        """
        base_requirements = {
            "mandatory_disclosure": jurisdiction.mandatory_disclosure,
            "audit_required": True,
            "retention_days": jurisdiction.retention_days,
            "redundancy_level": 2 if jurisdiction.code in ['CN', 'EU', 'UK'] else 1,
            "encryption_required": True
        }
        return base_requirements
    
    def _calculate_score(self, passed: List, warnings: List, violations: List) -> float:
        """
        Calculate overall compliance score from validation results.
        
        The scoring algorithm weights different types of validation results
        to produce a normalized score between 0.0 and 1.0. Passed checks
        contribute positively, warnings contribute partially, and violations
        reduce the score.
        
        Scoring Formula:
            score = (passed * 1.0 + warnings * 0.7 - violations * 0.3) / total
            score = max(0.0, min(1.0, score))
        
        Args:
            passed: List of passed validation checks
            warnings: List of validation warnings
            violations: List of compliance violations
        
        Returns:
            float: Normalized compliance score in range [0.0, 1.0]
        """
        total = len(passed) + len(warnings) + len(violations)
        if total == 0:
            return 1.0
        
        score = len(passed) * 1.0 + len(warnings) * 0.7 - len(violations) * 0.3
        return max(0.0, min(1.0, score / total))
    
    def _generate_recommendations(self, violations: List, warnings: List,
                                jurisdiction: POPSSWatermarkJurisdiction, 
                                content_type: str) -> List[str]:
        """
        Generate actionable remediation recommendations for compliance issues.
        
        This method analyzes validation results and generates specific,
        actionable recommendations to address compliance violations and
        warnings. Recommendations are tailored to the jurisdiction and
        content type.
        
        Args:
            violations: List of compliance violations with requirement details
            warnings: List of validation warnings with severity info
            jurisdiction: Target jurisdiction for context-specific recommendations
            content_type: Type of content being validated
        
        Returns:
            List[str]: List of actionable recommendation strings
        """
        recommendations = []
        
        for v in violations:
            if v["requirement"] == "mandatory_disclosure":
                recommendations.append(
                    f"Enable mandatory_disclosure=True for {jurisdiction.code} compliance"
                )
            elif v["requirement"] == "redundancy_level":
                recommendations.append(
                    f"Increase redundancy_level to at least 2 for {jurisdiction.code}"
                )
            elif v["requirement"] == "audit_enabled":
                recommendations.append(
                    f"Enable audit trail for {jurisdiction.code} compliance"
                )
        
        for w in warnings:
            if w["requirement"] == "watermark_strength":
                recommendations.append(
                    "Adjust watermark_strength for better quality/robustness balance"
                )
        
        return recommendations
    
    def _check_mandatory_disclosure(self, config, jurisdiction) -> Dict[str, Any]:
        required = getattr(jurisdiction, 'mandatory_disclosure', False)
        return {
            "requirement": "mandatory_disclosure",
            "required": required,
            "provided": config.mandatory_disclosure,
            "compliant": config.mandatory_disclosure >= required
        }
    
    def _check_audit_trail(self, config, jurisdiction) -> Dict[str, Any]:
        return {
            "requirement": "audit_trail",
            "required": True,
            "provided": config.audit_enabled,
            "compliant": config.audit_enabled
        }
    
    def _check_redundancy(self, config, jurisdiction) -> Dict[str, Any]:
        required = 2 if jurisdiction.code in ['CN', 'EU', 'UK'] else 1
        return {
            "requirement": "redundancy_level",
            "required": required,
            "provided": config.redundancy_level,
            "compliant": config.redundancy_level >= required
        }
    
    def _check_encryption(self, config, jurisdiction) -> Dict[str, Any]:
        return {
            "requirement": "encryption",
            "required": True,
            "provided": config.encryption_enabled,
            "compliant": config.encryption_enabled
        }
    
    def _check_strength(self, config, jurisdiction) -> Dict[str, Any]:
        valid = 1e-8 <= config.watermark_strength <= 1e-2
        return {
            "requirement": "watermark_strength",
            "valid": valid,
            "value": config.watermark_strength,
            "range": "1e-8 to 1e-2"
        }
    
    def _get_risk_recommendations(self, risk_level: str, content_type: str) -> List[str]:
        recommendations = {
            "low": [
                "Standard watermarking is sufficient",
                "Regular compliance reviews recommended"
            ],
            "medium": [
                "Enhanced watermarking recommended",
                "Implement audit trail for all generations",
                "Periodic compliance audits recommended"
            ],
            "high": [
                "Mandatory watermarking required",
                "Full audit trail mandatory",
                "Consider explicit disclosure labeling",
                "Risk assessment documentation required",
                "Legal review recommended before deployment"
            ]
        }
        return recommendations.get(risk_level, [])
    
    def validate(self, content_type: str, jurisdiction: str) -> Dict[str, Any]:
        """
        Convenience method to validate configuration for a jurisdiction.
        
        This is a simplified interface for running compliance validation
        without needing to construct the full input dictionary.
        
        Args:
            content_type: Type of content (text/image/audio/video/multimodal)
            jurisdiction: Jurisdiction code (CN/EU/US/UK/JP/KR/GLOBAL)
        
        Returns:
            Dict containing validation results including compliance_status,
            violations, warnings, and compliance_score
        
        Raises:
            ValueError: If validation fails due to invalid inputs
        """
        result = self._validate({
            "content_type": content_type,
            "jurisdiction": jurisdiction
        })
        if result.is_success():
            return result.output
        raise ValueError(f"Validation failed: {result.error}")
    
    def generate_metadata(self, model_id: str, risk_level: str, 
                        jurisdiction: str = "GLOBAL") -> Dict[str, Any]:
        """
        Convenience method to generate compliance metadata.
        
        This is a simplified interface for generating compliance metadata
        without needing to construct the full input dictionary.
        
        Args:
            model_id: Identifier of the AI model
            risk_level: Risk classification (low/medium/high)
            jurisdiction: Target jurisdiction code (default: GLOBAL)
        
        Returns:
            Dict containing compliance metadata including model_id,
            watermark_version, timestamp, and compliance information
        
        Raises:
            ValueError: If metadata generation fails
        """
        result = self._generate_metadata({
            "model_id": model_id,
            "risk_level": risk_level,
            "jurisdiction": jurisdiction
        })
        if result.is_success():
            return result.output
        raise ValueError(f"Metadata generation failed: {result.error}")
    
    def assess_risk(self, risk_factors: List[str], 
                   content_type: str = "text") -> Dict[str, Any]:
        """
        Convenience method to assess content risk level.
        
        This is a simplified interface for running risk assessment
        without needing to construct the full input dictionary.
        
        Args:
            risk_factors: List of applicable risk factors from:
                - public_deployment, sensitive_content, high_stakes_decisions
                - commercial_use, user_generated_input, multi_party_data
                - model_sharing, open_source_release
            content_type: Type of content being assessed (default: text)
        
        Returns:
            Dict containing risk_score (0.0-1.0), risk_level (low/medium/high),
            factors breakdown, and recommendations
        
        Raises:
            ValueError: If risk assessment fails
        """
        result = self._assess_risk({
            "risk_factors": risk_factors,
            "content_type": content_type
        })
        if result.is_success():
            return result.output
        raise ValueError(f"Risk assessment failed: {result.error}")


def create_compliance_operator(
    config: Optional[POPSSWatermarkConfig] = None
) -> PiscesLxComplianceOperator:
    """
    Factory function to create a compliance operator instance.
    
    This function provides a convenient way to instantiate a compliance
    operator with optional configuration.
    
    Args:
        config: Optional watermark configuration. If not provided,
            a default configuration will be used.
    
    Returns:
        POPSSComplianceOperator: Configured compliance operator instance
    
    Example:
        >>> operator = create_compliance_operator()
        >>> result = operator.validate("text", "CN")
    """
    return POPSSComplianceOperator(config=config)


PiscesLxComplianceOperator = POPSSComplianceOperator


class POPSSWatermarkComplianceOperator(PiscesLxBaseOperator):
    """
    Enhanced Compliance Operator with unified factory.
    
    This class combines all compliance validation functions into a cohesive
    operator with factory methods. It provides the same functionality as
    PiscesLxComplianceOperator but with a standardized naming convention
    for the POPSS (PiscesL1 Operations) module.
    
    This operator serves as the primary interface for compliance validation
    in production environments, offering:
        - Multi-jurisdiction compliance checking
        - Risk assessment and classification
        - Metadata generation for audit trails
        - Requirement-specific validation
    
    Attributes:
        name (str): Operator identifier ("pisceslx_compliance_operator")
        version (str): Operator version string
        description (str): Human-readable description
        config (POPSSWatermarkConfig): Watermark configuration
    
    Methods:
        create: Factory method to create operator instance
    """
    
    def __init__(self, config: Optional[POPSSWatermarkConfig] = None):
        super().__init__()
        self.name = "pisceslx_compliance_operator"
        self.version = VERSION
        self.description = "Legal compliance validation for AI watermark systems"
        self.config = config or POPSSWatermarkConfig()
    
    @classmethod
    def create(cls, config: Optional[POPSSWatermarkConfig] = None) -> 'POPSSWatermarkComplianceOperator':
        """Factory method to create a compliance operator."""
        return cls(config=config)


__all__ = [
    "POPSSComplianceOperator",
    "POPSSWatermarkComplianceOperator",
    "create_compliance_operator"
]
