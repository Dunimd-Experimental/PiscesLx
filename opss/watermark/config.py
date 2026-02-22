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
Watermark Configuration Operator

This module implements comprehensive watermark configuration supporting multiple
jurisdictions and AI regulations. It provides configuration classes for
regulatory compliance across different legal frameworks.

Supported Regulations:
    - GB/T 45225-2024 (China): Mandatory AI content watermarking
    - AI Act 2024 (EU): High-risk AI transparency requirements
    - NIST AI RMF 1.0 (US): Risk management framework
    - AI Safety Act 2024 (UK): Frontier model safety registration
    - AI Guidelines 2024 (Japan): Voluntary AI labeling
    - AI Act 2024 (Korea): Classified risk management

Key Features:
    - Multi-jurisdiction compliance configuration
    - Regulatory standard detection and validation
    - Risk level classification
    - Watermark strength tuning per jurisdiction
    - Audit trail configuration
    - Encryption and security settings

Compliance Metadata Structure:
    {
        "standard": "GB/T 45225-2024",
        "jurisdiction": "CN",
        "risk_level": "low|medium|high",
        "mandatory_disclosure": True,
        "tamper_evidence": True,
        "audit_required": True,
        "retention_days": 365
    }

Usage Examples:
    >>> from opss.watermark.config import (
    ...     POPSSWatermarkConfig,
    ...     POPSSComplianceStandard,
    ...     POPSSWatermarkJurisdiction
    ... )
    >>> config = POPSSWatermarkConfig(
    ...     standard=POPSSComplianceStandard.GB_T_45225_2024,
    ...     watermark_strength=1e-5,
    ...     redundancy_level=3
    ... )
"""


import os
import json
import hashlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime
from configs.version import VERSION


class POPSSWatermarkJurisdiction(Enum):
    """
    Supported jurisdictions for watermark compliance.
    
    Each jurisdiction represents a legal framework with specific requirements
    for AI-generated content watermarking and disclosure.
    """
    CN = "CN"
    EU = "EU"
    US = "US"
    UK = "UK"
    JP = "JP"
    KR = "KR"
    GLOBAL = "GLOBAL"
    
    @property
    def name_full(self) -> str:
        names = {
            "CN": "China",
            "EU": "European Union",
            "US": "United States",
            "UK": "United Kingdom",
            "JP": "Japan",
            "KR": "South Korea",
            "GLOBAL": "Global"
        }
        return names.get(self.value, self.value)
    
    @property
    def standard(self) -> str:
        standards = {
            "CN": "GB/T 45225-2024",
            "EU": "AI Act 2024",
            "US": "NIST AI RMF 1.0",
            "UK": "AI Safety Act 2024",
            "JP": "AI Guidelines 2024",
            "KR": "AI Act 2024",
            "GLOBAL": "ISO/IEC 27090"
        }
        return standards.get(self.value, "Unknown")
    
    @property
    def mandatory_disclosure(self) -> bool:
        requirements = {
            "CN": True,
            "EU": True,
            "US": False,
            "UK": True,
            "JP": False,
            "KR": True,
            "GLOBAL": False
        }
        return requirements.get(self.value, False)
    
    @property
    def retention_days(self) -> int:
        periods = {
            "CN": 365,
            "EU": 2555,
            "US": 365,
            "UK": 730,
            "JP": 365,
            "KR": 365,
            "GLOBAL": 365
        }
        return periods.get(self.value, 365)
    
    @property
    def code(self) -> str:
        return self.value


class POPSSComplianceStandard(Enum):
    """
    Enumeration of supported AI regulation standards.
    
    Each standard defines specific requirements for watermarking,
    disclosure, and audit trails in AI-generated content.
    
    Standards:
        GB_T_45225_2024: Chinese national standard for AI watermarking
        AI_ACT_2024: European Union AI Act requirements
        NIST_AI_RMF: US National Institute of Standards AI Risk Management
        AI_SAFETY_ACT_2024: UK AI Safety Act frontier model requirements
        ISO_27090: International standard for AI watermarking
    """
    GB_T_45225_2024 = "GB/T 45225-2024"
    AI_ACT_2024 = "AI Act 2024"
    NIST_AI_RMF = "NIST AI RMF 1.0"
    AI_SAFETY_ACT_2024 = "AI Safety Act 2024"
    ISO_27090 = "ISO/IEC 27090"
    VOLUNTARY_BEST_PRACTICE = "Voluntary Best Practice"


class POPSSWatermarkRiskLevel(Enum):
    """
    Risk classification levels for AI-generated content.
    
    Risk levels determine the stringency of watermarking requirements
    and disclosure obligations under various regulations.
    
    Levels:
        LOW: Minimal watermarking requirements, voluntary disclosure
        MEDIUM: Standard watermarking with user disclosure option
        HIGH: Mandatory watermarking with explicit disclosure required
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class POPSSWatermarkContentType(Enum):
    """
    Content types requiring watermarking.
    
    Different content types may have different embedding strategies
    and regulatory requirements.
    """
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


@dataclass
class POPSSWatermarkConfig:
    """
    Comprehensive watermark configuration for multi-jurisdiction compliance.
    
    This configuration class encapsulates all parameters required for
    watermark embedding, verification, and audit trails across different
    legal frameworks.
    
    Attributes:
        standard (POPSSComplianceStandard): Regulatory compliance standard
        jurisdiction (POPSSWatermarkJurisdiction): Target jurisdiction
        risk_level (POPSSWatermarkRiskLevel): Content risk classification
        watermark_strength (float): Embedding strength (1e-6 to 1e-3)
        redundancy_level (int): Redundancy encoding level (1-5)
        encryption_enabled (bool): Enable payload encryption
        encryption_algorithm (str): Encryption algorithm (AES-256-GCM)
        verify_threshold (float): Verification confidence threshold
        audit_enabled (bool): Enable audit trail generation
        mandatory_disclosure (bool): Force explicit disclosure labels
        tamper_evidence (bool): Enable tamper detection
        retention_days (int): Audit log retention period
        owner_id (str): Unique owner identifier
        model_id (str): Model identifier for watermark payload
        version (str): Watermark version string
        
    Example:
        config = POPSSWatermarkConfig(
            standard=POPSSComplianceStandard.GB_T_45225_2024,
            jurisdiction=POPSSWatermarkJurisdiction.CN,
            risk_level=POPSSWatermarkRiskLevel.MEDIUM,
            watermark_strength=1e-5,
            redundancy_level=3,
            encryption_enabled=True,
            audit_enabled=True
        )
    """
    standard: POPSSComplianceStandard = POPSSComplianceStandard.GB_T_45225_2024
    jurisdiction: POPSSWatermarkJurisdiction = POPSSWatermarkJurisdiction.GLOBAL
    risk_level: POPSSWatermarkRiskLevel = POPSSWatermarkRiskLevel.MEDIUM
    watermark_strength: float = 1e-5
    redundancy_level: int = 3
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    verify_threshold: float = 0.02
    audit_enabled: bool = True
    mandatory_disclosure: bool = False
    tamper_evidence: bool = True
    retention_days: int = 365
    owner_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = "PiscesL1-1.5B"
    version: str = VERSION
    enable_text_watermark: bool = True
    enable_image_watermark: bool = True
    enable_audio_watermark: bool = True
    enable_weight_watermark: bool = True
    
    def __post_init__(self):
        self._apply_jurisdiction_defaults()
        self._validate_config()
    
    def _apply_jurisdiction_defaults(self):
        """Apply jurisdiction-specific default configuration."""
        if hasattr(self.jurisdiction, 'mandatory_disclosure'):
            if not self.mandatory_disclosure:
                self.mandatory_disclosure = self.jurisdiction.mandatory_disclosure
            self.retention_days = self.jurisdiction.retention_days
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not 1e-8 <= self.watermark_strength <= 1e-2:
            raise ValueError(f"watermark_strength must be between 1e-8 and 1e-2, got {self.watermark_strength}")
        if not 1 <= self.redundancy_level <= 5:
            raise ValueError(f"redundancy_level must be between 1 and 5, got {self.redundancy_level}")
        if not 0 < self.verify_threshold < 1:
            raise ValueError(f"verify_threshold must be between 0 and 1, got {self.verify_threshold}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = asdict(self)
        result['standard'] = self.standard.value
        result['jurisdiction'] = self.jurisdiction.value if isinstance(self.jurisdiction, Enum) else self.jurisdiction
        result['risk_level'] = self.risk_level.value if isinstance(self.risk_level, Enum) else self.risk_level
        return result
    
    def to_json(self, filepath: Optional[str] = None) -> Optional[str]:
        """Save configuration to JSON file or return as string."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, ensure_ascii=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'POPSSWatermarkConfig':
        """Create configuration from dictionary."""
        if 'standard' in config_dict:
            try:
                config_dict['standard'] = POPSSComplianceStandard(config_dict['standard'])
            except ValueError:
                pass
        if 'jurisdiction' in config_dict:
            try:
                if isinstance(config_dict['jurisdiction'], str):
                    config_dict['jurisdiction'] = POPSSWatermarkJurisdiction(config_dict['jurisdiction'])
            except ValueError:
                pass
        if 'risk_level' in config_dict:
            try:
                config_dict['risk_level'] = POPSSWatermarkRiskLevel(config_dict['risk_level'])
            except ValueError:
                pass
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'POPSSWatermarkConfig':
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = "PISCESLX_WM_") -> 'POPSSWatermarkConfig':
        """Create configuration from environment variables."""
        config_dict = {}
        env_mappings = {
            f"{prefix}STANDARD": "standard",
            f"{prefix}JURISDICTION": "jurisdiction",
            f"{prefix}RISK_LEVEL": "risk_level",
            f"{prefix}STRENGTH": "watermark_strength",
            f"{prefix}REDUNDANCY": "redundancy_level",
            f"{prefix}ENCRYPT": "encryption_enabled",
            f"{prefix}THRESHOLD": "verify_threshold",
            f"{prefix}AUDIT": "audit_enabled",
            f"{prefix}DISCLOSURE": "mandatory_disclosure",
            f"{prefix}RETENTION": "retention_days",
            f"{prefix}OWNER_ID": "owner_id",
            f"{prefix}MODEL_ID": "model_id",
        }
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if config_key in ['encryption_enabled', 'audit_enabled', 'mandatory_disclosure']:
                    config_dict[config_key] = value.lower() in ['true', '1', 'yes']
                elif config_key in ['watermark_strength', 'verify_threshold']:
                    config_dict[config_key] = float(value)
                elif config_key in ['redundancy_level', 'retention_days']:
                    config_dict[config_key] = int(value)
                else:
                    config_dict[config_key] = value
        return cls.from_dict(config_dict) if config_dict else cls()


@dataclass
class POPSSWatermarkPayload:
    """
    Watermark payload structure for content marking.
    
    This class defines the standardized payload format embedded in
    AI-generated content for traceability and compliance.
    
    Payload Fields:
        model_id (str): Unique model identifier
        version (str): Model version string
        timestamp (str): ISO format generation timestamp
        session_id (str): Unique session identifier
        user_hash (str): Hashed user identifier (privacy-preserving)
        content_hash (str): Hash of marked content
        trace_chain (str): Audit trail chain identifier
        compliance (Dict): Compliance metadata
        encoding_version (str): Payload encoding version
        signature (str): Digital signature for verification
        
    Example:
        payload = POPSSWatermarkPayload(
            model_id="PiscesL1-1.5B",
            user_hash="a1b2c3d4e5f6",
            content_hash="...")
    """
    model_id: str
    version: str = VERSION
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_hash: str = ""
    content_hash: str = ""
    trace_chain: str = field(default_factory=lambda: str(uuid.uuid4()))
    compliance: Dict[str, Any] = field(default_factory=dict)
    encoding_version: str = VERSION
    signature: str = ""
    
    def __post_init__(self):
        if not self.user_hash:
            self.user_hash = hashlib.sha256(self.session_id.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert payload to JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'), sort_keys=True)
    
    def compute_content_hash(self, content: Any) -> str:
        """Compute and set content hash."""
        if isinstance(content, str):
            self.content_hash = hashlib.sha256(content.encode()).hexdigest()[:32]
        elif hasattr(content, 'shape'):
            self.content_hash = hashlib.sha256(str(content.shape).encode()).hexdigest()[:32]
        else:
            self.content_hash = hashlib.sha256(str(content).encode()).hexdigest()[:32]
        return self.content_hash


@dataclass
class POPSSWatermarkAuditRecord:
    """
    Audit record for watermark operations.
    
    This class structures audit trail entries for compliance verification
    and forensic analysis of watermark operations.
    
    Audit Fields:
        operation (str): Type of operation (embed, verify, extract)
        content_type (str): Type of content processed
        operator_id (str): Operator identifier
        compliance_standard (str): Applied compliance standard
        timestamp (str): ISO format operation timestamp
        user_hash (str): Hashed user identifier
        content_hash (str): Hash of processed content
        watermark_id (str): Unique watermark identifier
        result (str): Operation result status
        jurisdiction (str): Applied jurisdiction
        metadata (Dict): Additional operation metadata
        
    Example:
        record = POPSSWatermarkAuditRecord(
            operation="embed",
            content_type="text",
            watermark_id="wm-12345",
            result="success"
        )
    """
    operation: str
    content_type: str
    watermark_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    user_hash: str = ""
    content_hash: str = ""
    operator_id: str = "PiscesLxWatermarkOperator-v1.0"
    compliance_standard: str = "GB/T 45225-2024"
    result: str = "success"
    jurisdiction: str = "GLOBAL"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit record to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert audit record to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def compute_hash_chain(self, previous_hash: Optional[str] = None) -> str:
        """Compute hash chain for audit integrity."""
        chain_data = self.to_json()
        if previous_hash:
            chain_data = previous_hash + chain_data
        return hashlib.sha256(chain_data.encode()).hexdigest()


class POPSSWatermarkDefaultConfigFactory:
    """
    Factory class for creating default watermark configurations.
    
    This class provides static methods to generate pre-configured watermark
    settings for different jurisdictions and compliance standards.
    
    Attributes:
        None (all methods are static)
        
    Methods:
        create_for_jurisdiction: Create config for a specific jurisdiction
        create_for_standard: Create config for a compliance standard
        create_default: Create generic default configuration
    """
    
    @staticmethod
    def create_for_jurisdiction(jurisdiction: Union[str, POPSSWatermarkJurisdiction]) -> POPSSWatermarkConfig:
        """
        Create default watermark configuration for a specific jurisdiction.
        
        Args:
            jurisdiction: Jurisdiction code or enum value (CN, EU, US, UK, JP, KR, GLOBAL)
            
        Returns:
            POPSSWatermarkConfig: Pre-configured configuration for the jurisdiction
            
        Example:
            >>> config = POPSSWatermarkDefaultConfigFactory.create_for_jurisdiction("CN")
        """
        if isinstance(jurisdiction, str):
            try:
                jurisdiction = POPSSWatermarkJurisdiction(jurisdiction)
            except ValueError:
                jurisdiction = POPSSWatermarkJurisdiction.GLOBAL
        
        configs = {
            POPSSWatermarkJurisdiction.CN: POPSSWatermarkConfig(
                standard=POPSSComplianceStandard.GB_T_45225_2024,
                jurisdiction=POPSSWatermarkJurisdiction.CN,
                risk_level=POPSSWatermarkRiskLevel.MEDIUM,
                watermark_strength=1e-5,
                redundancy_level=3,
                encryption_enabled=True,
                audit_enabled=True,
                mandatory_disclosure=True,
                verify_threshold=0.02
            ),
            POPSSWatermarkJurisdiction.EU: POPSSWatermarkConfig(
                standard=POPSSComplianceStandard.AI_ACT_2024,
                jurisdiction=POPSSWatermarkJurisdiction.EU,
                risk_level=POPSSWatermarkRiskLevel.HIGH,
                watermark_strength=5e-6,
                redundancy_level=3,
                encryption_enabled=True,
                audit_enabled=True,
                mandatory_disclosure=True,
                verify_threshold=0.03
            ),
            POPSSWatermarkJurisdiction.US: POPSSWatermarkConfig(
                standard=POPSSComplianceStandard.NIST_AI_RMF,
                jurisdiction=POPSSWatermarkJurisdiction.US,
                risk_level=POPSSWatermarkRiskLevel.MEDIUM,
                watermark_strength=1e-5,
                redundancy_level=2,
                encryption_enabled=True,
                audit_enabled=True,
                mandatory_disclosure=False,
                verify_threshold=0.02
            ),
            POPSSWatermarkJurisdiction.UK: POPSSWatermarkConfig(
                standard=POPSSComplianceStandard.AI_SAFETY_ACT_2024,
                jurisdiction=POPSSWatermarkJurisdiction.UK,
                risk_level=POPSSWatermarkRiskLevel.HIGH,
                watermark_strength=5e-6,
                redundancy_level=4,
                encryption_enabled=True,
                audit_enabled=True,
                mandatory_disclosure=True,
                verify_threshold=0.025
            ),
            POPSSWatermarkJurisdiction.GLOBAL: POPSSWatermarkConfig()
        }
        
        return configs.get(jurisdiction, POPSSWatermarkConfig())
    
    @staticmethod
    def create_for_standard(standard: Union[str, POPSSComplianceStandard]) -> POPSSWatermarkConfig:
        """
        Create default watermark configuration for a compliance standard.
        
        Args:
            standard: Compliance standard string or enum value
            
        Returns:
            POPSSWatermarkConfig: Pre-configured configuration for the standard
        """
        if isinstance(standard, str):
            try:
                standard = POPSSComplianceStandard(standard)
            except ValueError:
                return POPSSWatermarkConfig()
        
        jurisdiction_map = {
            POPSSComplianceStandard.GB_T_45225_2024: POPSSWatermarkJurisdiction.CN,
            POPSSComplianceStandard.AI_ACT_2024: POPSSWatermarkJurisdiction.EU,
            POPSSComplianceStandard.NIST_AI_RMF: POPSSWatermarkJurisdiction.US,
            POPSSComplianceStandard.AI_SAFETY_ACT_2024: POPSSWatermarkJurisdiction.UK,
            POPSSComplianceStandard.ISO_27090: POPSSWatermarkJurisdiction.GLOBAL,
        }
        
        jurisdiction = jurisdiction_map.get(standard, POPSSWatermarkJurisdiction.GLOBAL)
        return POPSSWatermarkDefaultConfigFactory.create_for_jurisdiction(jurisdiction)
    
    @staticmethod
    def create_default() -> POPSSWatermarkConfig:
        """
        Create generic default watermark configuration.
        
        Returns:
            POPSSWatermarkConfig: Default configuration with balanced settings
        """
        return POPSSWatermarkConfig()


class POPSSWatermarkComplianceValidator:
    """
    Validator class for watermark configuration compliance.
    
    This class provides methods to validate watermark configurations
    against jurisdiction-specific requirements and generate compliance reports.
    
    Attributes:
        None (all methods are instance or static)
        
    Methods:
        validate: Validate configuration against requirements
        check_jurisdiction: Check jurisdiction-specific requirements
        generate_report: Generate detailed compliance report
    """
    
    def __init__(self, jurisdiction: Optional[Union[str, POPSSWatermarkJurisdiction]] = None):
        """
        Initialize the compliance validator.
        
        Args:
            jurisdiction: Target jurisdiction for validation (optional)
        """
        if isinstance(jurisdiction, str):
            try:
                jurisdiction = POPSSWatermarkJurisdiction(jurisdiction)
            except ValueError:
                jurisdiction = None
        self._jurisdiction = jurisdiction
    
    def validate(self, config: POPSSWatermarkConfig) -> Dict[str, Any]:
        """
        Validate a watermark configuration against requirements.
        
        Args:
            config: Watermark configuration to validate
            
        Returns:
            Dictionary containing validation results and recommendations
        """
        results = {
            "valid": True,
            "jurisdiction": config.jurisdiction.code if config.jurisdiction else "UNKNOWN",
            "requirements": [],
            "recommendations": [],
            "compliance_score": 1.0
        }
        
        if config.jurisdiction and hasattr(config.jurisdiction, 'mandatory_disclosure'):
            if config.mandatory_disclosure < config.jurisdiction.mandatory_disclosure:
                results["requirements"].append(
                    f"Jurisdiction {config.jurisdiction.code} requires mandatory_disclosure=True"
                )
                results["compliance_score"] -= 0.2
                results["valid"] = False
        
        if config.watermark_strength > 1e-4:
            results["recommendations"].append(
                "High watermark strength may affect content quality"
            )
            results["compliance_score"] -= 0.05
        elif config.watermark_strength < 1e-7:
            results["recommendations"].append(
                "Low watermark strength may reduce extraction reliability"
            )
            results["compliance_score"] -= 0.1
        
        if config.redundancy_level < 2 and config.jurisdiction and config.jurisdiction.code in ['CN', 'EU', 'UK']:
            results["requirements"].append(
                f"Jurisdiction {config.jurisdiction.code} recommends redundancy_level >= 2"
            )
            results["compliance_score"] -= 0.1
        
        return results
    
    def check_jurisdiction_requirements(self, config: POPSSWatermarkConfig) -> List[str]:
        """
        Check jurisdiction-specific requirements.
        
        Args:
            config: Watermark configuration to check
            
        Returns:
            List of unmet requirements
        """
        requirements = []
        
        if config.jurisdiction and hasattr(config.jurisdiction, 'mandatory_disclosure'):
            if not config.mandatory_disclosure:
                requirements.append(
                    f"Jurisdiction {config.jurisdiction.code} requires mandatory_disclosure=True"
                )
        
        if config.redundancy_level < 2 and config.jurisdiction and config.jurisdiction.code in ['CN', 'EU', 'UK']:
            requirements.append(
                f"Jurisdiction {config.jurisdiction.code} recommends redundancy_level >= 2"
            )
        
        return requirements
    
    def generate_report(self, config: POPSSWatermarkConfig) -> Dict[str, Any]:
        """
        Generate a detailed compliance report.
        
        Args:
            config: Watermark configuration to analyze
            
        Returns:
            Comprehensive compliance report dictionary
        """
        validation = self.validate(config)
        
        report = {
            "configuration": config.to_dict(),
            "validation": validation,
            "jurisdiction_info": {
                "code": config.jurisdiction.code if config.jurisdiction else "UNKNOWN",
                "name": config.jurisdiction.name_full if config.jurisdiction else "Unknown",
                "standard": config.jurisdiction.standard if config.jurisdiction else "Unknown",
                "mandatory_disclosure": config.jurisdiction.mandatory_disclosure if config.jurisdiction else False,
                "retention_days": config.jurisdiction.retention_days if config.jurisdiction else 365
            },
            "timestamp": datetime.utcnow().isoformat(),
            "compliance_score": validation["compliance_score"]
        }
        
        return report


def get_default_config(jurisdiction: Optional[Union[str, POPSSWatermarkJurisdiction]] = None) -> POPSSWatermarkConfig:
    """
    Get default watermark configuration for a jurisdiction.
    
    .. deprecated::
        Use :class:`POPSSWatermarkDefaultConfigFactory` instead.
    
    Args:
        jurisdiction: Jurisdiction code or enum value
        
    Returns:
        Pre-configured POPSSWatermarkConfig instance
    """
    return POPSSWatermarkDefaultConfigFactory.create_for_jurisdiction(jurisdiction)


def validate_compliance(config: POPSSWatermarkConfig, 
                       jurisdiction: Optional[Union[str, POPSSWatermarkJurisdiction]] = None) -> Dict[str, Any]:
    """
    Validate configuration against jurisdiction requirements.
    
    .. deprecated::
        Use :class:`POPSSWatermarkComplianceValidator` instead.
    
    Args:
        config: Watermark configuration to validate
        jurisdiction: Target jurisdiction for validation
        
    Returns:
        Dictionary containing validation results and recommendations
    """
    validator = POPSSWatermarkComplianceValidator(jurisdiction)
    return validator.validate(config)


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
    "get_default_config",
    "validate_compliance"
]
