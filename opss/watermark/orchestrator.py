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
Watermark Orchestrator

This module implements the watermark orchestrator that coordinates all watermark
operators into a unified pipeline. It provides a high-level interface for
content watermarking, weight watermarking, and compliance verification.

Orchestration Pipeline:
    ┌─────────────────────────────────────────────────────────┐
    │                    Input Content                        │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │              Compliance Validation                       │
    │  - Check jurisdiction requirements                      │
    │  - Validate configuration                             │
    │  - Generate compliance metadata                         │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │              Content Watermarking                       │
    │  - Text: Zero-width character embedding                │
    │  - Image: DCT frequency domain embedding                │
    │  - Audio: STFT ultrasonic embedding                     │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │              Weight Watermarking                         │
    │  - Layer selection and codebook generation              │
    │  - Regularization loss computation                      │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │              Audit Trail Logging                        │
    │  - Operation logging                                    │
    │  - Hash chain maintenance                              │
    │  - Report generation                                   │
    └─────────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    Output                              │
    │  - Watermarked content                                 │
    │  - Verification results                                │
    │  - Audit trail                                         │
    └─────────────────────────────────────────────────────────┘

Key Features:
    - Unified interface for all watermark operations
    - Configuration-driven pipeline execution
    - Automatic operator coordination
    - Error handling and recovery
    - Performance monitoring
    - Comprehensive metadata and audit trails

Usage Examples:
    >>> from opss.watermark.orchestrator import POPSSWatermarkOrchestrator
    >>> orchestrator = POPSSWatermarkOrchestrator()
    >>> 
    >>> # Configure for a jurisdiction
    >>> orchestrator.configure(jurisdiction="CN", strength=1e-5)
    >>> 
    >>> # Embed watermark in content
    >>> result = orchestrator.embed(content, metadata={"user_id": "user123"})
    >>> 
    >>> # Verify watermark
    >>> verified = orchestrator.verify(content)
"""

import time
import torch
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.opsc.base import PiscesLxBaseOperator
from utils.opsc.interface import PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION
from .config import (
    POPSSWatermarkConfig,
    POPSSWatermarkJurisdiction,
    POPSSComplianceStandard,
    POPSSWatermarkRiskLevel,
    get_default_config
)
from .content_watermark_operator import POPSSContentWatermarkOperator
from .weight_watermark_operator import POPSSWeightWatermarkOperator
from .compliance_operator import POPSSComplianceOperator
from .audit_operator import POPSSAuditOperator


class POPSSWatermarkOrchestrator(PiscesLxBaseOperator):
    """
    Watermark orchestrator for unified watermark operations.
    
    This orchestrator coordinates all watermark operators to provide a complete
    watermarking solution including content watermarking, weight watermarking,
    compliance validation, and audit logging. It serves as the primary interface
    for all watermark-related operations in the PiscesL1 system.
    
    Architecture:
        The orchestrator implements a pipeline-based architecture where each
        operation flows through multiple stages:
        
        1. Input Validation: Verify input format and content type
        2. Compliance Check: Validate against jurisdiction requirements
        3. Content Watermarking: Apply modality-specific watermarking
        4. Weight Watermarking: Inject ownership patterns into model weights
        5. Audit Logging: Record operation for compliance audit trails
    
    Supported Operations:
        - embed: Embed watermark in text, image, or audio content
        - verify: Extract and verify watermark from content
        - train_embed: Get regularization loss for weight watermarking
        - validate_compliance: Check configuration against regulations
        - full_pipeline: Execute complete watermarking pipeline
    
    Modality Support:
        - Text: Zero-width character embedding with protocol framing
        - Image: DCT-based frequency domain watermarking
        - Audio: STFT-based ultrasonic watermarking
        - Weight: Codebook-based regularization for model ownership
    
    Attributes:
        config (POPSSWatermarkConfig): Watermark configuration containing
            jurisdiction settings, strength parameters, and compliance options
        content_operator (POPSSContentWatermarkOperator): Handles content
            watermarking for text, images, and audio
        weight_operator (POPSSWeightWatermarkOperator): Handles model weight
            watermarking for ownership verification
        compliance_operator (POPSSComplianceOperator): Validates configurations
            against regulatory requirements
        audit_operator (POPSSAuditOperator): Manages audit trail logging
        _stats (Dict): Operation statistics for monitoring
        
    Input Format:
        {
            "action": "embed" | "verify" | "train_embed" | "validate_compliance" | "full_pipeline",
            "content": str | torch.Tensor,          # Content to watermark
            "model": torch.nn.Module,              # Model for weight watermarking
            "metadata": Dict[str, Any],           # Watermark metadata
            "user_id": str,                       # User identifier
            "result_callback": Callable,           # Optional callback
        }
        
    Output Format:
        {
            "action": str,
            "watermarked_content": Any,
            "watermark_id": str,
            "compliance_report": Dict,
            "audit_id": str
        }
    
    Example:
        >>> orchestrator = POPSSWatermarkOrchestrator()
        >>> orchestrator.configure(jurisdiction="CN", strength=1e-5)
        >>> result = orchestrator.embed("Hello, World!", user_id="user123")
        >>> print(result["watermark_id"])
    """
    
    def __init__(self, config: Optional[POPSSWatermarkConfig] = None):
        super().__init__()
        self.name = "popss_watermark_orchestrator"
        self.version = VERSION
        self.description = "Unified watermark orchestrator for content and weight watermarking"
        self.config = config or POPSSWatermarkConfig()
        self.content_operator = POPSSContentWatermarkOperator(self.config)
        self.weight_operator = POPSSWeightWatermarkOperator(self.config)
        self.compliance_operator = POPSSComplianceOperator(self.config)
        self.audit_operator = POPSSAuditOperator(self.config)
        self._stats = {
            "total_operations": 0,
            "embed_operations": 0,
            "verify_operations": 0,
            "avg_processing_time": 0.0
        }
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["embed", "verify", "train_embed", "validate_compliance", "full_pipeline"]
                },
                "content": {
                    "type": ["string", "tensor"],
                    "description": "Content to watermark or verify"
                },
                "model": {
                    "type": "object",
                    "description": "PyTorch model for weight watermarking"
                },
                "metadata": {
                    "type": "object",
                    "description": "Watermark metadata"
                },
                "user_id": {
                    "type": "string",
                    "description": "User identifier"
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "Target jurisdiction"
                },
                "config_override": {
                    "type": "object",
                    "description": "Configuration override"
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "watermarked_content": {"type": "any"},
                "watermark_id": {"type": "string"},
                "compliance_report": {"type": "object"},
                "audit_id": {"type": "string"},
                "processing_time": {"type": "number"}
            }
        }
    
    def _execute_impl(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        start_time = time.time()
        
        action = inputs.get("action", "embed")
        
        if action == "embed":
            result = self._embed(inputs)
        elif action == "verify":
            result = self._verify(inputs)
        elif action == "train_embed":
            result = self._train_embed(inputs)
        elif action == "validate_compliance":
            result = self._validate_compliance(inputs)
        elif action == "full_pipeline":
            result = self._full_pipeline(inputs)
        else:
            result = PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Unknown action: {action}"
            )
        
        processing_time = time.time() - start_time
        self._stats["total_operations"] += 1
        self._stats["avg_processing_time"] = (
            (self._stats["avg_processing_time"] * (self._stats["total_operations"] - 1) + processing_time) /
            self._stats["total_operations"]
        )
        
        if result.is_success():
            if "metadata" not in result.metadata:
                result.metadata["processing_time"] = processing_time
            else:
                result.metadata["processing_time"] = processing_time
        
        return result
    
    def _embed(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Embed watermark in content.
        
        This method orchestrates the watermark embedding process by routing
        the content to the appropriate operator based on content type. It
        handles text, image, and audio content with modality-specific
        watermarking techniques.
        
        Embedding Process:
            1. Validate content is provided
            2. Construct payload with model_id and user_id
            3. Route to appropriate content operator method
            4. Log operation in audit trail if enabled
            5. Return watermarked content and metadata
        
        Args:
            inputs: Dictionary containing embedding parameters
                - content (str | torch.Tensor): Content to watermark
                - metadata (Dict): Additional metadata to embed
                - user_id (str): User identifier for traceability
        
        Returns:
            PiscesLxOperatorResult: Embedding result containing
                - output: Dict with watermarked_content, watermark_id, and payload
                - metadata: Dict with action and content_type
        """
        content = inputs.get("content")
        metadata = inputs.get("metadata", {})
        user_id = inputs.get("user_id", "anonymous")
        
        if not content:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Content is required"
            )
        
        try:
            embed_inputs = {
                "action": "embed",
                "content": content,
                "payload": {
                    "model_id": self.config.model_id,
                    "user_id": user_id
                },
                "metadata": metadata
            }
            
            if self.config.enable_text_watermark and isinstance(content, str):
                embed_result = self.content_operator._embed_text(embed_inputs)
            elif self.config.enable_image_watermark and isinstance(content, torch.Tensor):
                embed_result = self.content_operator._embed_image(embed_inputs)
            else:
                embed_result = self.content_operator._embed(inputs)
            
            if not embed_result.is_success():
                return embed_result
            
            watermark_id = embed_result.output.get("watermark_id", "")
            
            if self.config.audit_enabled:
                self.audit_operator.log_operation(
                    operation="embed",
                    content_type=self._detect_content_type(content),
                    watermark_id=watermark_id,
                    result="success",
                    metadata={
                        "model_id": self.config.model_id,
                        "compliance_standard": self.config.standard.value
                    }
                )
            
            self._stats["embed_operations"] += 1
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "watermarked_content": embed_result.output.get("watermarked_content", embed_result.output.get("content")),
                    "watermark_id": watermark_id,
                    "payload": embed_result.output.get("payload", {})
                },
                metadata={
                    "action": "embed",
                    "content_type": self._detect_content_type(content)
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _verify(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Verify watermark in content.
        
        This method extracts and verifies watermarks from content, returning
        the embedded payload if a valid watermark is found. It supports all
        content modalities and logs verification attempts in the audit trail.
        
        Verification Process:
            1. Validate content is provided
            2. Detect content type (text/image/audio)
            3. Log verification attempt in audit trail
            4. Extract watermark using content operator
            5. Log verification result in audit trail
            6. Return verification status and payload
        
        Args:
            inputs: Dictionary containing verification parameters
                - content (str | torch.Tensor): Content to verify
        
        Returns:
            PiscesLxOperatorResult: Verification result containing
                - output: Dict with watermark_found, payload, and verified status
                - metadata: Dict with action and content_type
        """
        content = inputs.get("content")
        
        if not content:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Content is required"
            )
        
        try:
            content_type = self._detect_content_type(content)
            
            if self.config.audit_enabled:
                self.audit_operator.log_operation(
                    operation="verify",
                    content_type=content_type,
                    result="pending",
                    metadata={"action": "verify"}
                )
            
            extract_result = self.content_operator._extract(inputs, content_type)
            
            watermark_found = extract_result.output.get("watermark_found", False)
            
            if self.config.audit_enabled:
                self.audit_operator.log_operation(
                    operation="verify",
                    content_type=content_type,
                    watermark_id=extract_result.output.get("payload", {}).get("trace_chain", ""),
                    result="success" if watermark_found else "failed",
                    metadata={
                        "watermark_found": watermark_found,
                        "payload": extract_result.output.get("payload", {})
                    }
                )
            
            self._stats["verify_operations"] += 1
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "watermark_found": watermark_found,
                    "payload": extract_result.output.get("payload", {}),
                    "verified": watermark_found
                },
                metadata={
                    "action": "verify",
                    "content_type": content_type
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _train_embed(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Get weight regularization loss for training-time watermarking.
        
        This method prepares the weight watermarking operator and computes
        the regularization loss that should be added to the training objective.
        The loss encourages model weights to encode ownership patterns.
        
        Training Process:
            1. Validate model is a PyTorch Module
            2. Configure weight operator with owner_id and strength
            3. Select layers for watermarking if not already selected
            4. Compute regularization loss from weight operator
            5. Return loss for gradient descent
        
        Args:
            inputs: Dictionary containing training parameters
                - model (torch.nn.Module): Model to watermark
        
        Returns:
            PiscesLxOperatorResult: Training result containing
                - output: Dict with regularization_loss and layers_watermarked count
                - metadata: Dict with action and strength
        
        Note:
            The regularization loss should be added to the main training loss:
            total_loss = task_loss + lambda * watermark_loss
        """
        model = inputs.get("model")
        
        if not isinstance(model, torch.nn.Module):
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error="Model is required for weight watermarking"
            )
        
        try:
            self.weight_operator.configure(
                owner_id=self.config.owner_id,
                strength=self.config.watermark_strength,
                threshold=self.config.verify_threshold
            )
            
            if not self.weight_operator._selected_layers:
                self.weight_operator._select_layers({"model": model})
            
            regularization_result = self.weight_operator._regularize({"model": model})
            
            if not regularization_result.is_success():
                return regularization_result
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "regularization_loss": regularization_result.output["regularization_loss"],
                    "layers_watermarked": len(self.weight_operator._selected_layers)
                },
                metadata={
                    "action": "train_embed",
                    "strength": self.config.watermark_strength
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _validate_compliance(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Validate configuration against jurisdiction compliance requirements.
        
        This method checks the current watermark configuration against
        regulatory requirements for the specified jurisdiction, providing
        a compliance report with any violations or warnings.
        
        Validation Process:
            1. Log compliance check in audit trail
            2. Delegate to compliance operator for validation
            3. Log validation result in audit trail
            4. Return compliance report
        
        Args:
            inputs: Dictionary containing validation parameters
                - jurisdiction (str): Target jurisdiction code
                - content_type (str): Type of content to validate for
        
        Returns:
            PiscesLxOperatorResult: Compliance validation result containing
                - output: Dict with jurisdiction, violations, warnings,
                  compliance_score, and compliance_status
                - metadata: Dict with action and recommendations
        """
        jurisdiction = inputs.get("jurisdiction", "GLOBAL")
        content_type = inputs.get("content_type", "text")
        
        try:
            if self.config.audit_enabled:
                self.audit_operator.log_operation(
                    operation="verify",
                    content_type="compliance",
                    result="pending",
                    metadata={"action": "validate_compliance"}
                )
            
            compliance_result = self.compliance_operator._validate({
                "content_type": content_type,
                "jurisdiction": jurisdiction,
                "config": self.config
            })
            
            if self.config.audit_enabled:
                self.audit_operator.log_operation(
                    operation="verify",
                    content_type="compliance",
                    result="success",
                    metadata={
                        "compliance_report": compliance_result.output
                    }
                )
            
            return compliance_result
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _full_pipeline(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        """
        Execute complete watermarking pipeline.
        
        This method runs the full watermarking workflow including content
        watermarking, weight watermarking (if model provided), and compliance
        validation. It provides a one-stop interface for comprehensive
        watermarking operations.
        
        Pipeline Stages:
            1. Compliance Validation: Check configuration against requirements
            2. Content Watermarking: Embed watermark in provided content
            3. Weight Watermarking: Compute regularization loss if model provided
            4. Result Aggregation: Combine all results into unified output
        
        Args:
            inputs: Dictionary containing pipeline parameters
                - content (str | torch.Tensor): Content to watermark
                - model (torch.nn.Module): Optional model for weight watermarking
                - metadata (Dict): Additional metadata to embed
                - user_id (str): User identifier for traceability
        
        Returns:
            PiscesLxOperatorResult: Pipeline result containing
                - output: Dict with watermarked_content, weight_loss,
                  compliance_report, and watermark_id
                - metadata: Dict with action and completion status
        """
        content = inputs.get("content")
        model = inputs.get("model")
        metadata = inputs.get("metadata", {})
        user_id = inputs.get("user_id", "anonymous")
        
        results = {
            "watermarked_content": None,
            "weight_loss": None,
            "compliance_report": None,
            "watermark_id": None,
            "audit_id": None
        }
        
        try:
            compliance_result = self._validate_compliance(inputs)
            results["compliance_report"] = compliance_result.output if compliance_result.is_success() else {}
            
            if content:
                embed_result = self._embed({
                    "content": content,
                    "metadata": metadata,
                    "user_id": user_id
                })
                if embed_result.is_success():
                    results["watermarked_content"] = embed_result.output.get("watermarked_content")
                    results["watermark_id"] = embed_result.output.get("watermark_id")
            
            if model and self.config.enable_weight_watermark:
                train_result = self._train_embed({"model": model})
                if train_result.is_success():
                    results["weight_loss"] = train_result.output.get("regularization_loss")
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=results,
                metadata={
                    "action": "full_pipeline",
                    "content_watermarked": results["watermarked_content"] is not None,
                    "weight_watermarked": results["weight_loss"] is not None
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _detect_content_type(self, content: Any) -> str:
        """
        Detect the type of content for routing to appropriate operator.
        
        This method analyzes the content object to determine its type,
        enabling the orchestrator to route to the correct watermarking
        method.
        
        Detection Logic:
            - str -> "text"
            - torch.Tensor with 3D shape -> "image"
            - torch.Tensor with 1D or 2D shape -> "audio"
            - Other -> "unknown"
        
        Args:
            content: Content object to analyze
        
        Returns:
            str: Content type string (text/image/audio/unknown)
        """
        if isinstance(content, str):
            return "text"
        elif isinstance(content, torch.Tensor):
            shape = content.shape
            if len(shape) == 3:
                return "image"
            elif len(shape) in [1, 2]:
                return "audio"
        return "unknown"
    
    def configure(self, jurisdiction: str = None, strength: float = None,
                owner_id: str = None, threshold: float = None) -> None:
        """
        Configure the orchestrator with custom parameters.
        
        This method allows runtime configuration of the orchestrator
        and its sub-operators. Any provided parameters will override
        the current configuration values.
        
        Args:
            jurisdiction: Jurisdiction code (CN/EU/US/UK/JP/KR/GLOBAL)
            strength: Watermark embedding strength (1e-8 to 1e-2)
            owner_id: Unique owner identifier for weight watermarking
            threshold: Verification threshold for watermark detection
        
        Note:
            After configuration, all sub-operators are reinitialized
            with the updated configuration.
        """
        if jurisdiction:
            try:
                self.config.jurisdiction = POPSSWatermarkJurisdiction(jurisdiction)
            except ValueError:
                pass
        if strength:
            self.config.watermark_strength = strength
        if owner_id:
            self.config.owner_id = owner_id
        if threshold:
            self.config.verify_threshold = threshold
        
        self.content_operator = POPSSContentWatermarkOperator(self.config)
        self.weight_operator = POPSSWeightWatermarkOperator(self.config)
        self.compliance_operator = POPSSComplianceOperator(self.config)
    
    def embed(self, content: Any, metadata: Optional[Dict[str, Any]] = None,
             user_id: str = "anonymous") -> Dict[str, Any]:
        """
        Embed watermark in content.
        
        This is a convenience method that wraps the internal _embed operation
        and returns a simplified result dictionary.
        
        Args:
            content: Content to watermark (str for text, torch.Tensor for image/audio)
            metadata: Optional additional metadata to embed in watermark
            user_id: User identifier for traceability (default: "anonymous")
        
        Returns:
            Dict containing:
                - watermarked_content: The watermarked content
                - watermark_id: Unique identifier for this watermark
                - payload: The embedded payload dictionary
        
        Raises:
            ValueError: If embedding fails
        
        Example:
            >>> result = orchestrator.embed("Hello", user_id="user123")
            >>> watermarked_text = result["watermarked_content"]
        """
        result = self._embed({
            "content": content,
            "metadata": metadata or {},
            "user_id": user_id
        })
        if result.is_success():
            return {
                "watermarked_content": result.output["watermarked_content"],
                "watermark_id": result.output["watermark_id"],
                "payload": result.output.get("payload", {})
            }
        raise ValueError(f"Embedding failed: {result.error}")
    
    def verify(self, content: Any) -> Dict[str, Any]:
        """
        Verify watermark in content.
        
        This is a convenience method that wraps the internal _verify operation
        and returns a simplified result dictionary.
        
        Args:
            content: Content to verify (str for text, torch.Tensor for image/audio)
        
        Returns:
            Dict containing:
                - watermark_found: Boolean indicating if watermark was detected
                - payload: Extracted payload dictionary (empty if not found)
                - verified: Boolean indicating verification success
        
        Raises:
            ValueError: If verification fails
        
        Example:
            >>> result = orchestrator.verify(watermarked_text)
            >>> if result["watermark_found"]:
            ...     print(f"User: {result['payload']['user_id']}")
        """
        result = self._verify({"content": content})
        if result.is_success():
            return {
                "watermark_found": result.output["watermark_found"],
                "payload": result.output.get("payload", {}),
                "verified": result.output["verified"]
            }
        raise ValueError(f"Verification failed: {result.error}")
    
    def get_regularization_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Get weight regularization loss for training.
        
        This is a convenience method that wraps the internal _train_embed
        operation and returns the regularization loss tensor directly.
        
        Args:
            model: PyTorch model to compute regularization loss for
        
        Returns:
            torch.Tensor: Regularization loss to add to training objective
        
        Raises:
            ValueError: If regularization computation fails
        
        Note:
            Add this loss to your training loop:
            >>> loss = task_loss + 0.01 * orchestrator.get_regularization_loss(model)
        """
        result = self._train_embed({"model": model})
        if result.is_success():
            return result.output["regularization_loss"]
        raise ValueError(f"Regularization failed: {result.error}")
    
    def validate_compliance(self, jurisdiction: str, 
                          content_type: str = "text") -> Dict[str, Any]:
        """
        Validate compliance for a jurisdiction.
        
        This is a convenience method that wraps the internal _validate_compliance
        operation and returns the compliance report directly.
        
        Args:
            jurisdiction: Jurisdiction code to validate against
            content_type: Type of content to validate for (default: "text")
        
        Returns:
            Dict containing:
                - jurisdiction: The validated jurisdiction
                - compliance_status: "compliant", "non_compliant", or "compliant_with_warnings"
                - violations: List of compliance violations
                - warnings: List of compliance warnings
                - compliance_score: Float score from 0.0 to 1.0
        
        Raises:
            ValueError: If compliance validation fails
        
        Example:
            >>> report = orchestrator.validate_compliance("CN", "text")
            >>> if report["compliance_status"] == "compliant":
            ...     print("Configuration meets CN requirements")
        """
        result = self._validate_compliance({
            "jurisdiction": jurisdiction,
            "content_type": content_type
        })
        if result.is_success():
            return result.output
        raise ValueError(f"Compliance validation failed: {result.error}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics.
        
        This method returns operational statistics collected during
        orchestrator usage, useful for monitoring and debugging.
        
        Returns:
            Dict containing:
                - total_operations: Total number of operations performed
                - embed_operations: Number of embed operations
                - verify_operations: Number of verify operations
                - avg_processing_time: Average processing time in seconds
                - config: Current configuration dictionary
        
        Example:
            >>> stats = orchestrator.get_stats()
            >>> print(f"Total operations: {stats['total_operations']}")
        """
        return {
            "total_operations": self._stats["total_operations"],
            "embed_operations": self._stats["embed_operations"],
            "verify_operations": self._stats["verify_operations"],
            "avg_processing_time": self._stats["avg_processing_time"],
            "config": self.config.to_dict()
        }


def create_watermark_orchestrator(
    config: Optional[POPSSWatermarkConfig] = None
) -> POPSSWatermarkOrchestrator:
    """
    Factory function to create a watermark orchestrator instance.
    
    This function provides a convenient way to instantiate a watermark
    orchestrator with optional configuration.
    
    Args:
        config: Optional watermark configuration. If not provided,
            a default configuration will be used.
    
    Returns:
        POPSSWatermarkOrchestrator: Configured orchestrator instance
    
    Example:
        >>> orchestrator = create_watermark_orchestrator()
        >>> result = orchestrator.embed("Hello, World!")
    """
    return POPSSWatermarkOrchestrator(config=config)


class POPSSWatermarkOrchestratorEnhanced(PiscesLxBaseOperator):
    """
    Enhanced Watermark Orchestrator with unified factory.
    
    This class combines all orchestration functions into a cohesive
    operator with factory methods for unified watermark management.
    It provides the same functionality as POPSSWatermarkOrchestrator
    but with additional enhancements for production use.
    
    This orchestrator serves as the primary production interface for
    watermark operations, offering:
        - Multi-modal content watermarking (text, image, audio)
        - Model weight watermarking for ownership verification
        - Multi-jurisdiction compliance validation
        - Comprehensive audit trail management
        - Performance monitoring and statistics
    
    Class Attributes:
        name (str): Operator identifier
        version (str): Operator version string
        description (str): Human-readable description
    
    Instance Attributes:
        config (POPSSWatermarkConfig): Watermark configuration
        content_operator: Content watermarking operator
        weight_operator: Weight watermarking operator
        compliance_operator: Compliance validation operator
        audit_operator: Audit trail operator
        _stats (Dict): Operation statistics
    
    Methods:
        create: Factory method to create orchestrator instance
    """
    
    def __init__(self, config: Optional[POPSSWatermarkConfig] = None):
        super().__init__()
        self.name = "popss_watermark_orchestrator_enhanced"
        self.version = VERSION
        self.description = "Unified watermark orchestration for multi-modal content"
        self.config = config or POPSSWatermarkConfig()
        self.content_operator = POPSSContentWatermarkOperator(config=config)
        self.weight_operator = POPSSWeightWatermarkOperator(config=config)
        self.compliance_operator = POPSSComplianceOperator(config=config)
        self.audit_operator = POPSSAuditOperator(config=config)
        self._stats = {
            "total_operations": 0,
            "embed_operations": 0,
            "verify_operations": 0,
            "avg_processing_time": 0.0
        }
    
    @classmethod
    def create(cls, config: Optional[POPSSWatermarkConfig] = None) -> 'POPSSWatermarkOrchestratorEnhanced':
        """Factory method to create a watermark orchestrator."""
        return cls(config=config)


__all__ = [
    "POPSSWatermarkOrchestrator",
    "create_watermark_orchestrator"
]
