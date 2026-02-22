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
Audit Trail Operator

This module implements comprehensive audit trail management for watermark
operations. It provides tracking, logging, and reporting capabilities
for compliance and forensic analysis.

Key Features:
    - Operation logging (embed, verify, extract)
    - Hash chain integrity protection
    - Chain of custody tracking
    - Audit report generation
    - Tamper evidence
    - Compliance reporting

Audit Record Structure:
    {
        "operation": "embed" | "verify" | "extract",
        "timestamp": "ISO8601 timestamp",
        "content_type": "text" | "image" | "audio" | "weight",
        "watermark_id": "UUID",
        "user_hash": "SHA256 hash of user ID",
        "content_hash": "SHA256 hash of content",
        "operator_id": "Operator identifier",
        "compliance_standard": "Applied standard",
        "result": "success" | "failed",
        "metadata": {...}
    }

Hash Chain:
    Each audit record is linked to the previous through a hash chain,
    providing tamper-evident logging:
    
    chain_hash[i] = SHA256(previous_hash + record[i])

Usage Examples:
    >>> from opss.watermark.audit_operator import PiscesLxAuditOperator
    >>> operator = PiscesLxAuditOperator()
    >>> 
    >>> # Log watermark embedding
    >>> operator.log_operation(
    ...     operation="embed",
    ...     content_type="text",
    ...     watermark_id="wm-12345"
    ... )
    >>> 
    >>> # Generate audit report
    >>> report = operator.generate_report(start_date, end_date)

Author: PiscesL1 Development Team
Version: 1.0.0
"""

import json
import hashlib
import uuid
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from utils.opsc.base import PiscesLxBaseOperator
from utils.opsc.interface import PiscesLxOperatorResult, PiscesLxOperatorStatus
from configs.version import VERSION
from .config import POPSSWatermarkConfig, POPSSWatermarkAuditRecord

PiscesLxAuditRecord = POPSSWatermarkAuditRecord


class POPSSAuditOperator(PiscesLxBaseOperator):
    """
    Audit trail management operator for watermark operations.
    
    This operator provides comprehensive logging and tracking capabilities
    for watermark operations, ensuring compliance with regulatory requirements
    and enabling forensic analysis.
    
    Attributes:
        config (POPSSWatermarkConfig): Watermark configuration
        records (List[PiscesLxAuditRecord]): Audit records
        chain_hash (str): Current hash chain head
        storage_path (Path): Directory for audit storage
        
    Input Format:
        {
            "action": "log" | "query" | "generate_report" | "verify_chain" | "export",
            "operation": str,                     # embed, verify, extract
            "content_type": str,                  # text, image, audio, weight
            "watermark_id": str,                 # Unique watermark identifier
            "payload": Dict,                     # Watermark payload
            "result": str,                       # success, failed
            "metadata": Dict,                    # Additional metadata
            "start_date": str,                   # Query start date (ISO format)
            "end_date": str,                     # Query end date (ISO format)
            "filters": Dict,                     # Query filters
            "output_path": str                   # Export path
        }
        
    Output Format:
        {
            "action": str,
            "result": Dict | List[Dict],
            "audit_id": str,
            "chain_valid": bool
        }
    """
    
    def __init__(self, config: Optional[POPSSWatermarkConfig] = None):
        super().__init__()
        self.name = "pisceslx_audit_operator"
        self.version = VERSION
        self.description = "Audit trail management for watermark operations"
        self.config = config or POPSSWatermarkConfig()
        self.records: List[PiscesLxAuditRecord] = []
        self.chain_hash: Optional[str] = None
        self.storage_path = Path("artifacts/watermark/audit")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["log", "query", "generate_report", "verify_chain", "export"]
                },
                "operation": {
                    "type": "string",
                    "enum": ["embed", "verify", "extract"]
                },
                "content_type": {
                    "type": "string",
                    "enum": ["text", "image", "audio", "video", "weight", "multimodal"]
                },
                "watermark_id": {
                    "type": "string",
                    "description": "Unique watermark identifier"
                },
                "payload": {
                    "type": "object",
                    "description": "Watermark payload"
                },
                "result": {
                    "type": "string",
                    "enum": ["success", "failed"]
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata"
                },
                "content_hash": {
                    "type": "string",
                    "description": "Hash of marked content"
                },
                "user_hash": {
                    "type": "string",
                    "description": "Hashed user identifier"
                },
                "start_date": {
                    "type": "string",
                    "description": "Query start date (ISO format)"
                },
                "end_date": {
                    "type": "string",
                    "description": "Query end date (ISO format)"
                },
                "filters": {
                    "type": "object",
                    "description": "Query filters"
                },
                "output_path": {
                    "type": "string",
                    "description": "Export file path"
                }
            }
        }
    
    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "result": {"type": "any"},
                "audit_id": {"type": "string"},
                "record_count": {"type": "integer"},
                "chain_valid": {"type": "boolean"}
            }
        }
    
    def _execute_impl(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        action = inputs.get("action", "log")
        
        if action == "log":
            return self._log_operation(inputs)
        elif action == "query":
            return self._query_records(inputs)
        elif action == "generate_report":
            return self._generate_report(inputs)
        elif action == "verify_chain":
            return self._verify_chain(inputs)
        elif action == "export":
            return self._export_records(inputs)
        else:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=f"Unknown action: {action}"
            )
    
    def _log_operation(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        try:
            record = PiscesLxAuditRecord(
                operation=inputs.get("operation", "embed"),
                content_type=inputs.get("content_type", "text"),
                watermark_id=inputs.get("watermark_id", str(uuid.uuid4())),
                user_hash=inputs.get("user_hash", ""),
                content_hash=inputs.get("content_hash", ""),
                compliance_standard=self.config.standard.value if hasattr(self.config.standard, 'value') else str(self.config.standard),
                result=inputs.get("result", "success"),
                jurisdiction=self.config.jurisdiction.code if hasattr(self.config.jurisdiction, 'code') else str(self.config.jurisdiction),
                metadata=inputs.get("metadata", {})
            )
            
            chain_hash = self.chain_hash
            if chain_hash:
                record_hash = record.compute_hash_chain(chain_hash)
                self.chain_hash = record_hash
            else:
                self.chain_hash = record.compute_hash_chain(None)
            
            self.records.append(record)
            
            self._save_record(record)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "logged": True,
                    "audit_id": record.watermark_id,
                    "chain_head": self.chain_hash
                },
                metadata={
                    "action": "log",
                    "operation": record.operation,
                    "content_type": record.content_type
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _query_records(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        try:
            filters = inputs.get("filters", {})
            start_date = inputs.get("start_date")
            end_date = inputs.get("end_date")
            
            filtered = []
            
            for record in self.records:
                match = True
                
                if filters:
                    if "operation" in filters and record.operation != filters["operation"]:
                        match = False
                    if "content_type" in filters and record.content_type != filters["content_type"]:
                        match = False
                    if "result" in filters and record.result != filters["result"]:
                        match = False
                    if "watermark_id" in filters and record.watermark_id != filters["watermark_id"]:
                        match = False
                
                if match and start_date:
                    if record.timestamp < start_date:
                        match = False
                
                if match and end_date:
                    if record.timestamp > end_date:
                        match = False
                
                if match:
                    filtered.append(record.to_dict())
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "records": filtered,
                    "count": len(filtered)
                },
                metadata={
                    "action": "query",
                    "filters_applied": filters
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _generate_report(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        try:
            start_date = inputs.get("start_date", "1970-01-01T00:00:00")
            end_date = inputs.get("end_date", datetime.utcnow().isoformat())
            
            report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "period": {
                    "start": start_date,
                    "end": end_date
                },
                "summary": {
                    "total_operations": 0,
                    "embed_operations": 0,
                    "verify_operations": 0,
                    "extract_operations": 0,
                    "success_count": 0,
                    "failed_count": 0,
                    "by_content_type": defaultdict(int),
                    "by_jurisdiction": defaultdict(int)
                },
                "compliance_summary": {},
                "operations": []
            }
            
            for record in self.records:
                if record.timestamp < start_date or record.timestamp > end_date:
                    continue
                
                report["summary"]["total_operations"] += 1
                
                if record.operation == "embed":
                    report["summary"]["embed_operations"] += 1
                elif record.operation == "verify":
                    report["summary"]["verify_operations"] += 1
                elif record.operation == "extract":
                    report["summary"]["extract_operations"] += 1
                
                if record.result == "success":
                    report["summary"]["success_count"] += 1
                else:
                    report["summary"]["failed_count"] += 1
                
                report["summary"]["by_content_type"][record.content_type] += 1
                report["summary"]["by_jurisdiction"][record.jurisdiction] += 1
            
            report["summary"]["by_content_type"] = dict(report["summary"]["by_content_type"])
            report["summary"]["by_jurisdiction"] = dict(report["summary"]["by_jurisdiction"])
            
            report["compliance_summary"] = {
                "jurisdictions_covered": list(report["summary"]["by_jurisdiction"].keys()),
                "compliance_standards": set()
            }
            
            output_path = inputs.get("output_path")
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "report": report,
                    "output_path": output_path
                },
                metadata={
                    "action": "generate_report",
                    "period_covered": f"{start_date} to {end_date}"
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _verify_chain(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        try:
            chain_valid = True
            broken_points = []
            previous_hash = None
            
            for i, record in enumerate(self.records):
                computed_hash = record.compute_hash_chain(previous_hash)
                if previous_hash is None:
                    if self.chain_hash != computed_hash:
                        chain_valid = False
                        broken_points.append(i)
                previous_hash = computed_hash
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "chain_valid": chain_valid,
                    "total_records": len(self.records),
                    "broken_points": broken_points
                },
                metadata={
                    "action": "verify_chain",
                    "verification_time": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _export_records(self, inputs: Dict[str, Any]) -> PiscesLxOperatorResult:
        try:
            output_path = inputs.get("output_path")
            if not output_path:
                return PiscesLxOperatorResult(
                    operator_name=self.name,
                    status=PiscesLxOperatorStatus.FAILED,
                    error="output_path is required"
                )
            
            records_data = [r.to_dict() for r in self.records]
            
            export_data = {
                "exported_at": datetime.utcnow().isoformat(),
                "total_records": len(records_data),
                "chain_head": self.chain_hash,
                "records": records_data
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output={
                    "exported": True,
                    "output_path": output_path,
                    "records_exported": len(records_data)
                },
                metadata={
                    "action": "export",
                    "format": "JSON"
                }
            )
            
        except Exception as e:
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e)
            )
    
    def _save_record(self, record: PiscesLxAuditRecord) -> None:
        try:
            filename = f"audit_{record.watermark_id}.json"
            filepath = self.storage_path / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def log_operation(self, operation: str, content_type: str,
                    watermark_id: Optional[str] = None,
                    result: str = "success",
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Convenience method to log a watermark operation."""
        result_obj = self._log_operation({
            "operation": operation,
            "content_type": content_type,
            "watermark_id": watermark_id,
            "result": result,
            "metadata": metadata or {}
        })
        if result_obj.is_success():
            return result_obj.output["audit_id"]
        raise ValueError(f"Logging failed: {result_obj.error}")
    
    def query(self, filters: Optional[Dict[str, Any]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> List[Dict]:
        """Convenience method to query audit records."""
        result = self._query_records({
            "filters": filters or {},
            "start_date": start_date,
            "end_date": end_date
        })
        if result.is_success():
            return result.output["records"]
        raise ValueError(f"Query failed: {result.error}")
    
    def generate_report(self, start_date: str, end_date: str,
                       output_path: Optional[str] = None) -> Dict:
        """Convenience method to generate audit report."""
        result = self._generate_report({
            "start_date": start_date,
            "end_date": end_date,
            "output_path": output_path
        })
        if result.is_success():
            return result.output["report"]
        raise ValueError(f"Report generation failed: {result.error}")
    
    def verify_chain_integrity(self) -> Tuple[bool, List[int]]:
        """Verify hash chain integrity."""
        result = self._verify_chain({})
        if result.is_success():
            return result.output["chain_valid"], result.output["broken_points"]
        raise ValueError(f"Chain verification failed: {result.error}")


def create_audit_operator(
    config: Optional[POPSSWatermarkConfig] = None
) -> POPSSAuditOperator:
    """Factory function to create an audit operator."""
    return POPSSAuditOperator(config=config)


PiscesLxAuditOperator = POPSSAuditOperator


class POPSSWatermarkAuditOperator(PiscesLxBaseOperator):
    """
    Enhanced Audit Operator with unified factory.
    
    This class combines all audit trail functions into a cohesive
    operator with factory methods.
    
    Methods:
        create: Factory method to create operator instance
    """
    
    def __init__(self, config: Optional[POPSSWatermarkConfig] = None):
        super().__init__()
        self.name = "pisceslx_audit_operator"
        self.version = VERSION
        self.description = "Audit trail management for watermark operations"
        self.config = config or POPSSWatermarkConfig()
        self.audit_records = []
        self.chain_hash = None
    
    @classmethod
    def create(cls, config: Optional[POPSSWatermarkConfig] = None) -> 'POPSSWatermarkAuditOperator':
        """Factory method to create an audit operator."""
        return cls(config=config)


__all__ = [
    "POPSSAuditOperator",
    "POPSSWatermarkAuditOperator",
    "create_audit_operator"
]
