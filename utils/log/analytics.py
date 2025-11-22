#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import re
import time
import json
import threading
from .core import PiscesLxCoreLog
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable

class PiscesLxCoreLogPatternAnalyzer:
    """Log pattern analyzer used to identify and analyze patterns in logs."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize the log pattern analyzer.

        Args:
            window_size (int): The size of the analysis window, defaulting to 1000 log entries.
        """
        self.window_size = window_size
        self.log_buffer = deque(maxlen=window_size)
        self.patterns: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "first_seen": None,
            "last_seen": None,
            "severity_distribution": defaultdict(int),
            "components": set()
        })
        self._lock = threading.Lock()
        
    def analyze_log(self, log_record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single log record.

        Args:
            log_record (Dict[str, Any]): A log record.

        Returns:
            Dict[str, Any]: The analysis result.
        """
        with self._lock:
            # Add the log record to the buffer
            self.log_buffer.append(log_record)
            
            # Extract the pattern from the log record
            pattern = self._extract_pattern(log_record)
            
            # Update the statistics of the pattern
            self._update_pattern_stats(pattern, log_record)
            
            # Return the analysis result
            return {
                "pattern": pattern,
                "pattern_stats": dict(self.patterns[pattern]),
                "anomaly_score": self._calculate_anomaly_score(pattern)
            }
    
    def _extract_pattern(self, log_record: Dict[str, Any]) -> str:
        """Extract the pattern from a log record.

        Args:
            log_record (Dict[str, Any]): A log record.

        Returns:
            str: The extracted pattern.
        """
        # Extract the pattern based on the event type and component
        event = log_record.get("event", "UNKNOWN")
        component = log_record.get("component", "UNKNOWN")
        return f"{component}:{event}"
    
    def _update_pattern_stats(self, pattern: str, log_record: Dict[str, Any]) -> None:
        """Update the statistical information of a pattern.

        Args:
            pattern (str): The pattern.
            log_record (Dict[str, Any]): A log record.
        """
        stats = self.patterns[pattern]
        stats["count"] += 1
        stats["last_seen"] = time.time()
        
        if stats["first_seen"] is None:
            stats["first_seen"] = stats["last_seen"]
            
        # Update the severity distribution
        level = log_record.get("level", "INFO")
        stats["severity_distribution"][level] += 1
        
        # Update the component information
        component = log_record.get("component", "UNKNOWN")
        stats["components"].add(component)
    
    def _calculate_anomaly_score(self, pattern: str) -> float:
        """Calculate the anomaly score of a pattern.

        Args:
            pattern (str): The pattern.

        Returns:
            float: The anomaly score (ranging from 0 to 1).
        """
        if pattern not in self.patterns:
            return 0.0
            
        stats = self.patterns[pattern]
        count = stats["count"]
        
        # If the pattern appears rarely, it might be an anomaly
        if count < 5:
            return min(1.0, (5 - count) / 5.0)
            
        # If the severity distribution is uneven, it might be an anomaly
        severity_dist = stats["severity_distribution"]
        total = sum(severity_dist.values())
        if total > 0:
            entropy = 0.0
            for count in severity_dist.values():
                p = count / total
                if p > 0:
                    entropy -= p * (p ** 0.5)  # Simplified entropy calculation
            return min(1.0, entropy)
            
        return 0.0
    
    def get_top_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most common log patterns.

        Args:
            limit (int): The limit on the number of patterns to return.

        Returns:
            List[Dict[str, Any]]: A list of patterns.
        """
        with self._lock:
            sorted_patterns = sorted(
                self.patterns.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            
            result = []
            for pattern, stats in sorted_patterns[:limit]:
                result.append({
                    "pattern": pattern,
                    "count": stats["count"],
                    "first_seen": stats["first_seen"],
                    "last_seen": stats["last_seen"],
                    "severity_distribution": dict(stats["severity_distribution"]),
                    "components": list(stats["components"])
                })
                
            return result


class PiscesLxCoreLogPredictor:
    """Log predictor used to predict future log events."""
    
    def __init__(self, pattern_analyzer: PiscesLxCoreLogPatternAnalyzer):
        """Initialize the log predictor.

        Args:
            pattern_analyzer (PiscesLxCoreLogPatternAnalyzer): The pattern analyzer.
        """
        self.pattern_analyzer = pattern_analyzer
        self.prediction_models: Dict[str, Any] = {}
        
    def train(self, log_records: List[Dict[str, Any]]) -> None:
        """Train the prediction model.

        Args:
            log_records (List[Dict[str, Any]]): The training data.
        """
        # Train a simple prediction model based on historical pattern frequencies
        pattern_counts = defaultdict(int)
        pattern_timestamps = defaultdict(list)
        
        current_time = time.time()
        
        for record in log_records:
            # Analyze each log record
            analysis = self.pattern_analyzer.analyze_log(record)
            pattern = analysis["pattern"]
            
            # Update the count and timestamp
            pattern_counts[pattern] += 1
            pattern_timestamps[pattern].append(current_time)
        
        # Build a simple prediction model
        total_logs = len(log_records)
        if total_logs > 0:
            for pattern, count in pattern_counts.items():
                # Calculate the probability of the pattern occurring
                probability = count / total_logs
                
                # Calculate the average interval time (simplified calculation)
                timestamps = pattern_timestamps[pattern]
                if len(timestamps) > 1:
                    intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                    avg_interval = sum(intervals) / len(intervals)
                else:
                    avg_interval = 3600  # Default 1-hour interval
                
                self.prediction_models[pattern] = {
                    "probability": probability,
                    "count": count,
                    "average_interval_seconds": avg_interval,
                    "last_seen": max(timestamps) if timestamps else current_time
                }
        
        try:
            PiscesLxCoreLog.info("log_predictor_model_trained", {
                "event": "analytics.predictor.trained",
                "total_patterns": len(self.prediction_models),
                "total_logs": total_logs,
                "training_time": time.time()
            })
        except Exception:
            pass
    
    def predict_next_events(self, count: int = 5) -> List[Dict[str, Any]]:
        """Predict the next likely events.

        Args:
            count (int): The number of events to predict.

        Returns:
            List[Dict[str, Any]]: A list of predicted events.
        """
        # Make predictions based on the trained model
        predictions = []
        current_time = time.time()
        
        # Patterns sorted by probability
        sorted_patterns = sorted(
            self.prediction_models.items(),
            key=lambda x: x[1]["probability"],
            reverse=True
        )
        
        for i, (pattern, model_data) in enumerate(sorted_patterns[:count]):
            # Calculate the predicted time
            last_seen = model_data.get("last_seen", current_time)
            avg_interval = model_data.get("average_interval_seconds", 3600)
            
            # Predict the next occurrence time based on the average interval and a random factor
            random_factor = 0.8 + (hash(pattern) % 100) / 100.0 * 0.4  # Random factor between 0.8 and 1.2
            predicted_time = last_seen + avg_interval * random_factor
            
            # If the predicted time is in the past, adjust it to the future
            if predicted_time < current_time:
                predicted_time = current_time + (i + 1) * 300  # Future intervals of 5 minutes
            
            # Get pattern information
            pattern_info = None
            if pattern in self.pattern_analyzer.patterns:
                pattern_info = dict(self.pattern_analyzer.patterns[pattern])
                pattern_info["components"] = list(pattern_info.get("components", set()))
            
            predictions.append({
                "predicted_pattern": pattern,
                "probability": model_data["probability"],
                "predicted_time": predicted_time,
                "pattern_info": pattern_info,
                "confidence": min(1.0, model_data["count"] / 100.0)  # Confidence based on historical data volume
            })
        
        try:
            PiscesLxCoreLog.info("log_events_predicted", {
                "event": "analytics.predictor.prediction",
                "predictions_count": len(predictions),
                "top_pattern": predictions[0]["predicted_pattern"] if predictions else None,
                "prediction_time": current_time
            })
        except Exception:
            pass
        
        return predictions


class PiscesLxCoreLogForecaster:
    """Log error predictor that forecasts future possible errors based on historical log patterns."""
    
    def __init__(self, logger: Optional[PiscesLxCoreLog] = None):
        """Initialize the log error predictor.

        Args:
            logger (Optional[PiscesLxCoreLog]): The logger.
        """
        self.logger = logger or PiscesLxCoreLog(name="log_forecaster")
        self.error_patterns: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
    def register_error_pattern(self, pattern: str, timestamp: float) -> None:
        """Register an error pattern.

        Args:
            pattern (str): The error pattern.
            timestamp (float): The timestamp.
        """
        with self._lock:
            self.error_patterns[pattern].append(timestamp)
            # Keep a maximum of 100 timestamps to limit memory usage
            if len(self.error_patterns[pattern]) > 100:
                self.error_patterns[pattern] = self.error_patterns[pattern][-100:]
    
    def forecast_errors(self, horizon_hours: int = 24) -> Dict[str, Dict[str, Any]]:
        """Forecast future possible errors.

        Args:
            horizon_hours (int): The forecast time horizon in hours.

        Returns:
            Dict[str, Dict[str, Any]]: The error forecast results.
        """
        predictions = {}
        current_time = time.time()
        horizon_seconds = horizon_hours * 3600
        
        with self._lock:
            for pattern, timestamps in self.error_patterns.items():
                if len(timestamps) < 2:
                    continue
                    
                # Calculate the event intervals
                intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                avg_interval = sum(intervals) / len(intervals)
                
                # Predict the next occurrence time
                last_occurrence = timestamps[-1]
                next_predicted = last_occurrence + avg_interval
                
                # Calculate the probability within the forecast time horizon
                if next_predicted <= current_time + horizon_seconds:
                    # Simplified probability calculation
                    probability = min(1.0, 1.0 / (1.0 + (next_predicted - current_time) / 3600))
                    
                    predictions[pattern] = {
                        "next_predicted_time": next_predicted,
                        "probability": probability,
                        "average_interval_hours": avg_interval / 3600,
                        "occurrence_count": len(timestamps)
                    }
                    
                    # Record the forecast log
                    if self.logger:
                        self.logger.info("ERROR_FORECAST", {
                            "pattern": pattern,
                            "next_predicted_time": next_predicted,
                            "probability": probability,
                            "message": f"Predicted occurrence of '{pattern}' with {probability:.2%} probability"
                        })
        
        return predictions


class PiscesLxCoreLogCorrelator:
    """Log correlator used to correlate logs from different sources."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize the log correlator.

        Args:
            window_size (int): The size of the correlation window.
        """
        self.window_size = window_size
        self.log_streams: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.correlations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = threading.Lock()
        
    def add_log(self, stream_id: str, log_record: Dict[str, Any]) -> None:
        """Add a log record to the specified stream.

        Args:
            stream_id (str): The stream ID.
            log_record (Dict[str, Any]): The log record.
        """
        with self._lock:
            self.log_streams[stream_id].append(log_record)
            self._find_correlations(stream_id, log_record)
    
    def _find_correlations(self, stream_id: str, log_record: Dict[str, Any]) -> None:
        """Find correlations between log records.

        Args:
            stream_id (str): The stream ID.
            log_record (Dict[str, Any]): The log record.
        """
        # Find correlations based on timestamp and trace_id
        trace_id = log_record.get("trace_id")
        timestamp = log_record.get("timestamp", time.time())
        
        if trace_id:
            # Find other logs with the same trace_id
            for other_stream_id, logs in self.log_streams.items():
                if other_stream_id == stream_id:
                    continue
                    
                for other_log in logs:
                    if other_log.get("trace_id") == trace_id:
                        correlation_id = f"{trace_id}_{min(stream_id, other_stream_id)}_{max(stream_id, other_stream_id)}"
                        self.correlations[correlation_id].append({
                            "stream1": stream_id,
                            "log1": log_record,
                            "stream2": other_stream_id,
                            "log2": other_log,
                            "correlation_type": "trace_id_match"
                        })
    
    def get_correlations(self, correlation_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get log correlations.

        Args:
            correlation_id (Optional[str]): The correlation ID. If specified, only return the correlation for this ID.

        Returns:
            Dict[str, List[Dict[str, Any]]]: The correlation information.
        """
        with self._lock:
            if correlation_id:
                return {correlation_id: self.correlations.get(correlation_id, [])}
            return dict(self.correlations)