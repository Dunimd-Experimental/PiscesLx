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

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

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
        logger = PiscesLxLogger("PiscesLx.Opss.Agents",file_path=get_log_file("PiscesLx.Opss.Agents"), enable_file=True)
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
            
            return "## Aggregated Summary\n\n" + "\n---\n".join(summaries)
        
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
            
            return "## Priority Results\n\n" + "\n\n".join(priority_results)
        
        elif strategy == POPSSAggregationStrategy.VOTE:
            vote_counts = {}
            for source_id, data in source_results.items():
                result = data['result']
                if isinstance(result, str):
                    result_key = result[:100]
                    vote_counts[result_key] = vote_counts.get(result_key, 0) + 1
            
            sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
            
            top_votes = [f"Votes: {v[1]} - {v[0][:200]}" for v in sorted_votes[:3]]
            
            return "## Voting Results\n\n" + "\n---\n".join(top_votes)
        
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
            return False, "Result exceeds maximum length"
        
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


class POPSSParallelResultAggregator:
    def __init__(self, config: Optional[POPSSResultAggregatorConfig] = None):
        self.config = config or POPSSResultAggregatorConfig()
        self._LOG = PiscesLxLogger("PiscesLx.Opss.Agents", file_path=get_log_file("PiscesLx.Opss.Agents"), enable_file=True)
        self._async_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="piscesl1_parallel_aggregator")
        self._LOG.info("POPSSParallelResultAggregator initialized")

    async def merge_parallel_results(
        self,
        results: Dict[str, List[POPSSResultSource]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, POPSSAggregatedResult]:
        merged: Dict[str, POPSSAggregatedResult] = {}
        for group_key, group_results in results.items():
            merged[group_key] = await self._merge_group(group_key, group_results)
        return merged

    async def _merge_group(self, group_key: str, results: List[POPSSResultSource]) -> POPSSAggregatedResult:
        agg_id = f"pra_{uuid.uuid4().hex[:12]}"
        if not results:
            return POPSSAggregatedResult(
                aggregation_id=agg_id,
                task_id=group_key,
                combined_output="",
                aggregation_strategy="parallel_merge_empty",
                consistency_status=POPSSResultConsistency.PARTIAL.value,
                confidence_score=0.0,
            )
        consensus = await self._voting_consensus(results)
        confidence_merged = await self._confidence_weighted_merge(results)
        conflicts = self._detect_conflicts(results)
        resolved = await self._resolve_conflicts(conflicts, results, consensus)

        combined = resolved if resolved else confidence_merged
        combined_agg = POPSSAggregatedResult(
            aggregation_id=agg_id,
            task_id=group_key,
            combined_output=combined.get("combined_output", str(results)),
            structured_output=combined.get("structured_output", {}),
            source_results={s.source_id: {
                "agent_id": s.agent_id,
                "agent_type": s.agent_type,
                "result": s.result,
                "weight": s.weight,
                "confidence": s.confidence,
            } for s in results},
            aggregation_strategy="parallel_merge",
            consistency_status=consensus["consensus_status"],
            confidence_score=consensus["consensus_confidence"],
            metadata={
                "result_count": len(results),
                "consensus": consensus,
                "conflicts_detected": len(conflicts),
                "conflict_strategy": "majority_vote",
            },
        )
        return combined_agg

    async def _voting_consensus(self, results: List[POPSSResultSource]) -> Dict[str, Any]:
        if not results:
            return {"consensus_status": POPSSResultConsistency.PARTIAL.value, "consensus_confidence": 0.0, "winner": None}

        outputs: List[str] = []
        for r in results:
            output = r.result
            if isinstance(output, str):
                outputs.append(output)
            elif isinstance(output, dict):
                outputs.append(str(output))
            else:
                outputs.append(str(output) if output is not None else "")

        output_counts: Dict[str, int] = {}
        output_confidences: Dict[str, List[float]] = defaultdict(list)

        for r, out in zip(results, outputs):
            key = out[:200] if out else "__empty__"
            output_counts[key] = output_counts.get(key, 0) + 1
            output_confidences[key].append(r.confidence)

        max_count = max(output_counts.values()) if output_counts else 0
        total = len(results)
        majority_ratio = max_count / max(total, 1)
        winner_key = max(output_counts, key=output_counts.get) if output_counts else "__empty__"
        avg_confidence = sum(output_confidences.get(winner_key, [0.5])) / max(len(output_confidences.get(winner_key, [1])), 1)

        if majority_ratio >= 0.67:
            status = POPSSResultConsistency.CONSISTENT.value
        elif majority_ratio >= 0.5:
            status = POPSSResultConsistency.PARTIAL.value
        else:
            status = POPSSResultConsistency.INCONSISTENT.value

        return {
            "consensus_status": status,
            "consensus_confidence": avg_confidence * majority_ratio,
            "winner": winner_key if winner_key != "__empty__" else None,
            "vote_counts": dict(output_counts),
            "majority_ratio": majority_ratio,
            "total_voters": total,
        }

    async def _confidence_weighted_merge(self, results: List[POPSSResultSource]) -> Dict[str, Any]:
        if not results:
            return {"combined_output": "", "structured_output": {}}

        weighted_texts: List[Tuple[float, str]] = []
        structured_parts: Dict[str, List[Tuple[float, Any]]] = defaultdict(list)

        for r in results:
            weight = r.weight * r.confidence
            if isinstance(r.result, str):
                weighted_texts.append((weight, r.result))
            elif isinstance(r.result, dict):
                for key, value in r.result.items():
                    structured_parts[key].append((weight, value))

        weighted_texts.sort(key=lambda x: x[0], reverse=True)
        combined = "\n\n---\n\n".join(f"[confidence:{w:.2f}] {t}" for w, t in weighted_texts)

        merged_structured: Dict[str, Any] = {}
        for key, values in structured_parts.items():
            values.sort(key=lambda x: x[0], reverse=True)
            merged_structured[key] = values[0][1] if values else None

        return {
            "combined_output": combined,
            "structured_output": merged_structured,
        }

    def _detect_conflicts(self, results: List[POPSSResultSource]) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        if len(results) < 2:
            return conflicts

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                r1, r2 = results[i], results[j]
                out1 = str(r1.result) if r1.result else ""
                out2 = str(r2.result) if r2.result else ""

                if out1 and out2 and out1 != out2 and r1.confidence > 0.5 and r2.confidence > 0.5:
                    confidence_gap = abs(r1.confidence - r2.confidence)
                    if confidence_gap < 0.3:
                        conflicts.append({
                            "source_a": r1.source_id,
                            "source_b": r2.source_id,
                            "agent_a": r1.agent_id,
                            "agent_b": r2.agent_id,
                            "output_a": out1[:200],
                            "output_b": out2[:200],
                            "confidence_a": r1.confidence,
                            "confidence_b": r2.confidence,
                        })
        return conflicts

    async def _resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        results: List[POPSSResultSource],
        consensus: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not conflicts:
            return None

        agent_type_weights = self.config.weight_by_agent_type
        resolution_startegies = ["majority_vote", "weighted_confidence", "highest_confidence"]

        for strategy in resolution_startegies:
            if strategy == "majority_vote":
                if consensus.get("consensus_status") == POPSSResultConsistency.CONSISTENT.value:
                    break
            elif strategy == "weighted_confidence":
                scored: List[Tuple[float, POPSSResultSource]] = []
                for r in results:
                    base_weight = agent_type_weights.get(r.agent_type, 1.0)
                    scored.append((r.weight * r.confidence * base_weight, r))
                scored.sort(key=lambda x: x[0], reverse=True)
                if scored:
                    winner = scored[0][1]
                    return {
                        "combined_output": str(winner.result) if winner.result else "",
                        "structured_output": {},
                        "resolution_strategy": "weighted_confidence",
                    }
            elif strategy == "highest_confidence":
                best = max(results, key=lambda r: r.confidence)
                return {
                    "combined_output": str(best.result) if best.result else "",
                    "structured_output": {},
                    "resolution_strategy": "highest_confidence",
                }

        return None

    async def aggregate_grouped(
        self,
        grouped_results: Dict[str, List[POPSSResultSource]],
        top_n: int = 5
    ) -> List[POPSSAggregatedResult]:
        all_aggregated: List[POPSSAggregatedResult] = []
        merged = await self.merge_parallel_results(grouped_results)
        for group_key, agg_result in merged.items():
            all_aggregated.append(agg_result)
        all_aggregated.sort(key=lambda a: a.confidence_score, reverse=True)
        return all_aggregated[:top_n]

    def get_parallel_metrics(self) -> Dict[str, Any]:
        return {
            "type": "POPSSParallelResultAggregator",
            "default_strategy": "parallel_merge",
            "conflict_resolution": "majority_vote -> weighted_confidence -> highest_confidence",
            "consensus_threshold": "67% consistent, 50% partial",
        }

    def shutdown(self):
        self._async_executor.shutdown(wait=True)
        self._LOG.info("POPSSParallelResultAggregator shutdown")
