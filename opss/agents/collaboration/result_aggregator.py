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

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from utils.dc import PiscesLxLogger

class POPSSAggregationStrategy(Enum):
    CONCATENATE = "concatenate"
    SUMMARIZE = "summarize"
    MERGE = "merge"
    VOTE = "vote"
    WEIGHTED = "weighted"
    PRIORITY = "priority"

class POPSSResultConsistency(Enum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    PARTIAL = "partial"
    CONFLICTING = "conflicting"

@dataclass
class POPSSAggregatedResult:
    aggregation_id: str
    task_id: str
    
    combined_output: str = ""
    structured_output: Dict[str, Any] = field(default_factory=dict)
    
    source_results: Dict[str, Any] = field(default_factory=dict)
    aggregation_strategy: str = ""
    
    consistency_status: str = ""
    confidence_score: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class POPSSResultSource:
    source_id: str
    agent_id: str
    agent_type: str
    
    result: Any
    weight: float = 1.0
    confidence: float = 0.5
    
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class POPSSResultAggregatorConfig:
    default_strategy: POPSSAggregationStrategy = POPSSAggregationStrategy.CONCATENATE
    enable_consistency_check: bool = True
    enable_conflict_resolution: bool = True
    
    min_confidence_threshold: float = 0.3
    weight_by_agent_type: Dict[str, float] = field(default_factory=lambda: {
        "analysis": 1.2,
        "research": 1.1,
        "code": 1.0,
        "creative": 0.9,
        "general": 0.8,
    })
    
    max_result_length: int = 10000
    enable_deduplication: bool = True
    enable_summarization: bool = True

class POPSSResultAggregator:
    def __init__(self, config: Optional[POPSSResultAggregatorConfig] = None):
        self.config = config or POPSSResultAggregatorConfig()
        self._LOG = self._configure_logging()
        
        self._aggregation_history: List[Dict[str, Any]] = []
        
        self._async_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="piscesl1_result_aggregator"
        )
        
        self._LOG.info("POPSSResultAggregator initialized")
    
    def _configure_logging(self) -> PiscesLxLogger:
        logger = get_logger("PiscesLx.Core.Agents.Collaboration.ResultAggregator")
        return logger
    
    async def aggregate(
        self,
        task_id: str,
        results: List[POPSSResultSource],
        strategy: Optional[POPSSAggregationStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> POPSSAggregatedResult:
        aggregation_id = f"agg_{uuid.uuid4().hex[:12]}"
        
        self._LOG.info(f"Aggregating {len(results)} results for task: {task_id}")
        
        if not results:
            return POPSSAggregatedResult(
                aggregation_id=aggregation_id,
                task_id=task_id,
                combined_output="No results to aggregate",
                aggregation_strategy="empty",
                consistency_status=POPSSResultConsistency.PARTIAL.value,
                confidence_score=0.0,
            )
        
        effective_strategy = strategy or self.config.default_strategy
        
        source_results = {}
        for source in results:
            source_results[source.source_id] = {
                'agent_id': source.agent_id,
                'agent_type': source.agent_type,
                'result': source.result,
                'weight': source.weight,
                'confidence': source.confidence,
                'metadata': source.metadata,
            }
        
        consistency_check = None
        if self.config.enable_consistency_check:
            consistency_check = await self._check_consistency(source_results)
        
        combined_output = await self._combine_results(source_results, effective_strategy)
        
        structured_output = await self._structure_results(source_results, effective_strategy)
        
        if consistency_check:
            consistency_status = consistency_check['status']
            confidence_score = consistency_check['confidence']
        else:
            consistency_status = POPSSResultConsistency.CONSISTENT.value
            confidence_score = sum(s.confidence for s in results) / len(results)
        
        aggregated_result = POPSSAggregatedResult(
            aggregation_id=aggregation_id,
            task_id=task_id,
            combined_output=combined_output,
            structured_output=structured_output,
            source_results=source_results,
            aggregation_strategy=effective_strategy.value,
            consistency_status=consistency_status,
            confidence_score=confidence_score,
            metadata={
                'consistency_check': consistency_check,
                'context': context or {},
                'result_count': len(results),
                'timestamp': datetime.now().isoformat(),
            }
        )
        
        self._aggregation_history.append({
            'aggregation_id': aggregation_id,
            'task_id': task_id,
            'result_count': len(results),
            'strategy': effective_strategy.value,
            'consistency': consistency_status,
            'timestamp': datetime.now().isoformat(),
        })
        
        return aggregated_result
    
    async def _check_consistency(self, source_results: Dict[str, Any]) -> Dict[str, Any]:
        results_text = []
        for source_id, data in source_results.items():
            result = data['result']
            if isinstance(result, str):
                results_text.append(result)
            elif isinstance(result, dict):
                results_text.append(str(result))
        
        if len(results_text) < 2:
            return {
                'status': POPSSResultConsistency.CONSISTENT.value,
                'confidence': 1.0,
                'details': 'Single result, consistency check skipped',
            }
        
        identical_count = 0
        comparison_pairs = 0
        
        for i in range(len(results_text)):
            for j in range(i + 1, len(results_text)):
                if results_text[i] == results_text[j]:
                    identical_count += 1
                comparison_pairs += 1
        
        similarity_ratio = identical_count / max(comparison_pairs, 1)
        
        if similarity_ratio >= 0.9:
            return {
                'status': POPSSResultConsistency.CONSISTENT.value,
                'confidence': 0.95,
                'similarity_ratio': similarity_ratio,
            }
        elif similarity_ratio >= 0.5:
            return {
                'status': POPSSResultConsistency.PARTIAL.value,
                'confidence': 0.6,
                'similarity_ratio': similarity_ratio,
            }
        else:
            return {
                'status': POPSSResultConsistency.CONFLICTING.value,
                'confidence': 0.3,
                'similarity_ratio': similarity_ratio,
            }
    
    async def _combine_results(
        self,
        source_results: Dict[str, Any],
        strategy: POPSSAggregationStrategy
    ) -> str:
        if strategy == POPSSAggregationStrategy.CONCATENATE:
            combined = []
            for source_id, data in source_results.items():
                result = data['result']
                if isinstance(result, str):
                    combined.append(f"## Source: {data['agent_id']}\n{result}")
                elif result is not None:
                    combined.append(f"## Source: {data['agent_id']}\n{str(result)}")
            
            return "\n\n".join(combined)
        
        elif strategy == POPSSAggregationStrategy.SUMMARIZE:
            summaries = []
            for source_id, data in source_results.items():
                result = data['result']
                if isinstance(result, str):
                    summary = result[:500] + ("..." if len(result) > 500 else "")
                    summaries.append(summary)
                elif result is not None:
                    summaries.append(str(result)[:500])
            
            return f"## Aggregated Summary\n\n" + "\n---\n".join(summaries)
        
        elif strategy == POPSSAggregationStrategy.WEIGHTED:
            weighted_results = []
            for source_id, data in source_results.items():
                weight = data.get('weight', 1.0)
                confidence = data.get('confidence', 0.5)
                final_weight = weight * confidence
                
                result = data['result']
                if isinstance(result, str):
                    weighted_results.append((final_weight, result))
            
            weighted_results.sort(key=lambda x: x[0], reverse=True)
            
            top_results = [r[1] for r in weighted_results[:3]]
            return f"## Weighted Results (Top {len(top_results)})\n\n" + "\n---\n".join(top_results)
        
        elif strategy == POPSSAggregationStrategy.PRIORITY:
            sorted_sources = sorted(
                source_results.items(),
                key=lambda x: (x[1].get('weight', 1.0), x[1].get('confidence', 0.5)),
                reverse=True
            )
            
            priority_results = []
            for source_id, data in sorted_sources[:3]:
                result = data['result']
                if isinstance(result, str):
                    priority_results.append(result)
            
            return f"## Priority Results\n\n" + "\n\n".join(priority_results)
        
        elif strategy == POPSSAggregationStrategy.VOTE:
            vote_counts = {}
            for source_id, data in source_results.items():
                result = data['result']
                if isinstance(result, str):
                    result_key = result[:100]
                    vote_counts[result_key] = vote_counts.get(result_key, 0) + 1
            
            sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
            
            top_votes = [f"Votes: {v[1]} - {v[0][:200]}" for v in sorted_votes[:3]]
            
            return f"## Voting Results\n\n" + "\n---\n".join(top_votes)
        
        elif strategy == POPSSAggregationStrategy.MERGE:
            merged = {}
            for source_id, data in source_results.items():
                result = data['result']
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key not in merged:
                            merged[key] = []
                        merged[key].append({
                            'source': data['agent_id'],
                            'value': value,
                        })
            
            return f"## Merged Results\n\n{str(merged)}"
        
        return str(source_results)
    
    async def _structure_results(
        self,
        source_results: Dict[str, Any],
        strategy: POPSSAggregationStrategy
    ) -> Dict[str, Any]:
        structured = {
            'summary': {},
            'details': [],
            'metadata': {
                'source_count': len(source_results),
                'strategy': strategy.value,
            }
        }
        
        confidence_scores = []
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for source_id, data in source_results.items():
            confidence = data.get('confidence', 0.5)
            weight = data.get('weight', 1.0)
            confidence_scores.append(confidence)
            total_weight += weight
            weighted_confidence += confidence * weight
            
            structured['details'].append({
                'source_id': source_id,
                'agent_id': data['agent_id'],
                'agent_type': data['agent_type'],
                'confidence': confidence,
                'weight': weight,
                'result_type': type(data['result']).__name__,
            })
        
        structured['summary'] = {
            'average_confidence': sum(confidence_scores) / max(len(confidence_scores), 1),
            'weighted_confidence': weighted_confidence / max(total_weight, 1),
            'confidence_range': {
                'min': min(confidence_scores) if confidence_scores else 0,
                'max': max(confidence_scores) if confidence_scores else 0,
            }
        }
        
        return structured
    
    async def validate_result(self, result: POPSSResultSource) -> Tuple[bool, Optional[str]]:
        if result.result is None:
            return False, "Result is None"
        
        if isinstance(result.result, str) and len(result.result) > self.config.max_result_length:
            return False, f"Result exceeds maximum length"
        
        if result.confidence < self.config.min_confidence_threshold:
            return False, f"Confidence below threshold: {result.confidence}"
        
        return True, None
    
    def deduplicate_results(self, results: List[POPSSResultSource]) -> List[POPSSResultSource]:
        if not self.config.enable_deduplication:
            return results
        
        seen_contents = {}
        unique_results = []
        
        for result in results:
            content_key = str(result.result)[:100]
            
            if content_key not in seen_contents:
                seen_contents[content_key] = result.source_id
                unique_results.append(result)
        
        return unique_results
    
    def get_aggregation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._aggregation_history[-limit:]
    
    def get_metrics(self) -> Dict[str, Any]:
        history = self._aggregation_history
        
        consistency_counts = {}
        strategy_counts = {}
        
        for entry in history:
            consistency = entry.get('consistency', 'unknown')
            strategy = entry.get('strategy', 'unknown')
            
            consistency_counts[consistency] = consistency_counts.get(consistency, 0) + 1
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_aggregations': len(history),
            'consistency_distribution': consistency_counts,
            'strategy_distribution': strategy_counts,
            'average_result_count': (
                sum(e.get('result_count', 0) for e in history) / max(len(history), 1)
            ),
        }
    
    def shutdown(self):
        self._async_executor.shutdown(wait=True)
        self._LOG.info("POPSSResultAggregator shutdown")
