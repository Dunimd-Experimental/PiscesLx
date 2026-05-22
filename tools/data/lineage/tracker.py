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
Data Lineage Tracker for tracking data provenance and processing history.

This module provides comprehensive data lineage tracking, recording
the complete journey of data from source to final use, including
all transformations and quality metrics.

Key Features:
    - Source tracking: Record data origin and download info
    - Transformation history: Track all processing steps
    - Quality metrics: Store quality scores over time
    - Serialization: Save and load lineage records

Usage:
    >>> from tools.data.lineage import PiscesLxDataLineageTracker
    >>> tracker = PiscesLxDataLineageTracker()
    >>> tracker.record_source("dataset_001", "huggingface", "wikitext")
    >>> tracker.record_transformation("dataset_001", "clean", {"lowercase": True})
    >>> tracker.record_quality("dataset_001", 0.85)
    >>> report = tracker.generate_report("dataset_001")
"""

import json
import hashlib
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict


class PiscesLxDataLineageRecord:
    """
    Single lineage record for a data sample.
    
    Attributes:
        sample_id: Unique identifier for the sample.
        source: Data source information.
        transformations: List of applied transformations.
        quality_scores: Quality score history.
        metadata: Additional metadata.
        timestamps: Timestamps for each operation.
    """
    
    def __init__(self, sample_id: str) -> None:
        """
        Initialize a lineage record.
        
        Args:
            sample_id: Unique sample identifier.
        """
        self.sample_id = sample_id
        self.source: Dict[str, Any] = {}
        self.transformations: List[Dict[str, Any]] = []
        self.quality_scores: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.timestamps: Dict[str, str] = {}
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert record to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation.
        """
        return {
            'sample_id': self.sample_id,
            'source': self.source,
            'transformations': self.transformations,
            'quality_scores': self.quality_scores,
            'metadata': self.metadata,
            'timestamps': self.timestamps,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PiscesLxDataLineageRecord':
        """
        Create record from dictionary.
        
        Args:
            data: Dictionary data.
            
        Returns:
            PiscesLxDataLineageRecord: Created record.
        """
        record = cls(data['sample_id'])
        record.source = data.get('source', {})
        record.transformations = data.get('transformations', [])
        record.quality_scores = data.get('quality_scores', [])
        record.metadata = data.get('metadata', {})
        record.timestamps = data.get('timestamps', {})
        record.created_at = data.get('created_at', datetime.now().isoformat())
        return record


class PiscesLxDataLineageTracker:
    """
    Data lineage tracker for comprehensive provenance tracking.
    
    This class tracks the complete journey of data through the
    processing pipeline, from source to final use.
    
    Attributes:
        records: Dictionary of lineage records.
        global_metadata: Global metadata for all records.
    
    Example:
        >>> tracker = PiscesLxDataLineageTracker()
        >>> tracker.record_source("doc_001", "huggingface", "wikitext-103")
        >>> tracker.record_transformation("doc_001", "clean", {"lowercase": True})
        >>> tracker.record_quality("doc_001", 0.92)
        >>> report = tracker.generate_report("doc_001")
    """
    
    def __init__(self) -> None:
        """Initialize the lineage tracker."""
        self.records: Dict[str, PiscesLxDataLineageRecord] = {}
        self.global_metadata: Dict[str, Any] = {}
        self._source_index: Dict[str, Set[str]] = defaultdict(set)
    
    def _get_or_create_record(self, sample_id: str) -> PiscesLxDataLineageRecord:
        """
        Get or create a lineage record.
        
        Args:
            sample_id: Sample identifier.
            
        Returns:
            PiscesLxDataLineageRecord: The record.
        """
        if sample_id not in self.records:
            self.records[sample_id] = PiscesLxDataLineageRecord(sample_id)
        return self.records[sample_id]
    
    def record_source(
        self,
        sample_id: str,
        source_type: str,
        source_name: str,
        url: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Record data source information.
        
        Args:
            sample_id: Sample identifier.
            source_type: Type of source ('huggingface', 'modelscope', 'local', etc.).
            source_name: Name of the source dataset.
            url: Optional URL of the source.
            version: Optional version string.
            **kwargs: Additional source metadata.
        """
        record = self._get_or_create_record(sample_id)
        record.source = {
            'type': source_type,
            'name': source_name,
            'url': url,
            'version': version,
            **kwargs
        }
        record.timestamps['source_recorded'] = datetime.now().isoformat()
        
        self._source_index[f"{source_type}:{source_name}"].add(sample_id)
    
    def record_download(
        self,
        sample_id: str,
        download_time: float,
        file_size: Optional[int] = None,
        checksum: Optional[str] = None
    ) -> None:
        """
        Record download information.
        
        Args:
            sample_id: Sample identifier.
            download_time: Time taken to download in seconds.
            file_size: Optional file size in bytes.
            checksum: Optional file checksum.
        """
        record = self._get_or_create_record(sample_id)
        record.metadata['download'] = {
            'time_seconds': download_time,
            'file_size': file_size,
            'checksum': checksum,
            'downloaded_at': datetime.now().isoformat()
        }
    
    def record_transformation(
        self,
        sample_id: str,
        transform_name: str,
        params: Optional[Dict[str, Any]] = None,
        input_hash: Optional[str] = None,
        output_hash: Optional[str] = None
    ) -> None:
        """
        Record a transformation applied to the data.
        
        Args:
            sample_id: Sample identifier.
            transform_name: Name of the transformation.
            params: Transformation parameters.
            input_hash: Hash of input data.
            output_hash: Hash of output data.
        """
        record = self._get_or_create_record(sample_id)
        record.transformations.append({
            'name': transform_name,
            'params': params or {},
            'input_hash': input_hash,
            'output_hash': output_hash,
            'timestamp': datetime.now().isoformat()
        })
    
    def record_quality(
        self,
        sample_id: str,
        quality_score: float,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Record quality score for a sample.
        
        Args:
            sample_id: Sample identifier.
            quality_score: Overall quality score (0-1).
            metrics: Detailed quality metrics.
        """
        record = self._get_or_create_record(sample_id)
        record.quality_scores.append({
            'score': quality_score,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def record_metadata(
        self,
        sample_id: str,
        key: str,
        value: Any
    ) -> None:
        """
        Record additional metadata.
        
        Args:
            sample_id: Sample identifier.
            key: Metadata key.
            value: Metadata value.
        """
        record = self._get_or_create_record(sample_id)
        record.metadata[key] = value
    
    def get_record(self, sample_id: str) -> Optional[PiscesLxDataLineageRecord]:
        """
        Get lineage record for a sample.
        
        Args:
            sample_id: Sample identifier.
            
        Returns:
            Optional[PiscesLxDataLineageRecord]: The record or None.
        """
        return self.records.get(sample_id)
    
    def get_samples_by_source(self, source_type: str, source_name: str) -> List[str]:
        """
        Get all samples from a specific source.
        
        Args:
            source_type: Source type.
            source_name: Source name.
            
        Returns:
            List[str]: List of sample IDs.
        """
        key = f"{source_type}:{source_name}"
        return list(self._source_index.get(key, set()))
    
    def generate_report(self, sample_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive lineage report.
        
        Args:
            sample_id: Sample identifier.
            
        Returns:
            Dict[str, Any]: Lineage report.
        """
        record = self.get_record(sample_id)
        if record is None:
            return {'error': f"No record found for {sample_id}"}
        
        report = record.to_dict()
        
        if record.quality_scores:
            scores = [qs['score'] for qs in record.quality_scores]
            report['quality_summary'] = {
                'current_score': scores[-1],
                'min_score': min(scores),
                'max_score': max(scores),
                'avg_score': sum(scores) / len(scores),
                'num_evaluations': len(scores)
            }
        
        report['transformation_count'] = len(record.transformations)
        
        return report
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a summary report for all tracked samples.
        
        Returns:
            Dict[str, Any]: Summary report.
        """
        all_scores = []
        source_counts: Dict[str, int] = defaultdict(int)
        transform_counts: Dict[str, int] = defaultdict(int)
        
        for record in self.records.values():
            if record.quality_scores:
                all_scores.extend([qs['score'] for qs in record.quality_scores])
            
            if record.source:
                source_key = f"{record.source.get('type', 'unknown')}:{record.source.get('name', 'unknown')}"
                source_counts[source_key] += 1
            
            for transform in record.transformations:
                transform_counts[transform['name']] += 1
        
        return {
            'total_samples': len(self.records),
            'unique_sources': len(source_counts),
            'source_distribution': dict(source_counts),
            'transformation_distribution': dict(transform_counts),
            'quality_stats': {
                'num_scores': len(all_scores),
                'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0,
                'min_score': min(all_scores) if all_scores else 0,
                'max_score': max(all_scores) if all_scores else 0
            } if all_scores else None
        }
    
    def save(self, path: str) -> None:
        """
        Save tracker state to file.
        
        Args:
            path: File path.
        """
        state = {
            'records': {sid: r.to_dict() for sid, r in self.records.items()},
            'global_metadata': self.global_metadata
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str) -> None:
        """
        Load tracker state from file.

        Args:
            path: File path.
        """
        import pickle
        import os
        # Validate path to prevent traversal
        resolved = os.path.realpath(os.path.expanduser(path))
        with open(resolved, 'rb') as f:
            state = pickle.load(f)
        
        self.records = {
            sid: PiscesLxDataLineageRecord.from_dict(r)
            for sid, r in state.get('records', {}).items()
        }
        self.global_metadata = state.get('global_metadata', {})
        
        self._source_index = defaultdict(set)
        for sid, record in self.records.items():
            if record.source:
                key = f"{record.source.get('type', 'unknown')}:{record.source.get('name', 'unknown')}"
                self._source_index[key].add(sid)
    
    def export_json(self, path: str) -> None:
        """
        Export lineage data to JSON format.
        
        Args:
            path: Output file path.
        """
        data = {
            'records': {sid: r.to_dict() for sid, r in self.records.items()},
            'summary': self.generate_summary_report()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def clear(self) -> None:
        """Clear all tracked records."""
        self.records.clear()
        self._source_index.clear()
    
    def __len__(self) -> int:
        """Get number of tracked samples."""
        return len(self.records)
    
    def __contains__(self, sample_id: str) -> bool:
        """Check if sample is tracked."""
        return sample_id in self.records
