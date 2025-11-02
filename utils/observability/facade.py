#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

import time
from ..log.core import PiscesLxCoreLog
from typing import Any, Dict, List, Optional
from .service import PiscesLxCoreObservabilityService
from .manager import PiscesLxCoreObservabilityManager
from .config import create_observability_config, PiscesL1CoreMonitoringConfig

class PiscesLxCoreObservabilityFacade:
    """A unified facade class for observability operations.

    Provides simplified configuration interfaces for monitoring training, inference, and distributed processes.
    """
    
    def __init__(self, args: Any = None) -> None:
        """Initialize the observability facade class.
        
        Args:
            args: Command line arguments or configuration object. Defaults to None.
        """
        self.args = args or {}
        self.logger = PiscesLxCoreLog("OBSERVABILITY")
        # Initialize core service and manager
        self._service = PiscesLxCoreObservabilityService.instance()
        self._manager = PiscesLxCoreObservabilityManager()
        # Store active monitoring sessions
        self._active_sessions = {}
        # Log the initialization of the observability facade
        try:
            self.logger.info("Observability facade initialized for tools/services monitoring", {"scope": "tools_services"})
        except Exception as e:
            self.logger.error("Failed to log observability facade initialization", error=str(e), error_class=type(e).__name__)
    
    def observe_training(self, model_size: str = "7B", mode: str = "auto") -> Dict[str, Any]:
        """Configure monitoring for the training process.
        
        Args:
            model_size: Model size, e.g., "7B", "70B", "314B". Defaults to "7B".
            mode: Monitoring mode, options are "auto", "detailed", "minimal". Defaults to "auto".
            
        Returns:
            Dict[str, Any]: Training monitoring configuration containing session ID, model size, mode, etc.
            
        Examples:
            >>> obs = PiscesLxCoreObservabilityFacade()
            >>> config = obs.observe_training(model_size="70B", mode="auto")
        """
        self.logger.info("Configuring training monitoring", {"model": model_size, "mode": mode})
        # Retrieve the unified configuration manager
        from utils.config.manager import PiscesLxCoreConfigManager
        config_manager = PiscesLxCoreConfigManager.get_instance()
        base_config = config_manager.get("observability", {})
        config = create_observability_config(base_config, "training", model_size, mode)
        # Start a new monitoring session
        session_id = self._start_monitoring_session(config)
        
        result = {
            "session_id": session_id,
            "model_size": model_size,
            "mode": mode,
            "monitoring_level": config.monitoring_level,
            "metrics": config.metrics,
            "sampling_rate": config.sampling_rate,
            "status": "active"
        }
        
        self.logger.success("Training monitoring activated", {"session_id": session_id, "level": config.monitoring_level})
        return result
    
    def observe_inference(self, batch_size: int = 1, sequence_length: int = 2048, 
                         mode: str = "auto") -> Dict[str, Any]:
        """Configure monitoring for the inference process.
        
        Args:
            batch_size: Batch size. Defaults to 1.
            sequence_length: Sequence length. Defaults to 2048.
            mode: Monitoring mode, options are "auto", "performance", "memory". Defaults to "auto".
            
        Returns:
            Dict[str, Any]: Inference monitoring configuration.
        """
        self.logger.info("Configuring inference monitoring", {"batch": batch_size, "seq_len": sequence_length})
        # Retrieve the unified configuration manager
        from utils.config.manager import PiscesLxCoreConfigManager
        config_manager = PiscesLxCoreConfigManager.get_instance()
        base_config = config_manager.get("observability", {})
        config = create_observability_config(base_config, "inference", batch_size, mode)
        # Start a new monitoring session
        session_id = self._start_monitoring_session(config)
        
        result = {
            "session_id": session_id,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "inference_load": config.batch_size * config.sequence_length,
            "mode": mode,
            "metrics": config.metrics,
            "latency_tracking": config.latency_tracking,
            "memory_tracking": config.memory_tracking,
            "throughput_tracking": config.throughput_tracking,
            "status": "active"
        }
        
        return result
    
    def observe_distributed(self, world_size: int = None, rank: int = None, 
                           mode: str = "auto") -> Dict[str, Any]:
        """Configure monitoring for the distributed training process.
        
        Args:
            world_size: Total number of nodes. Auto-detected if None. Defaults to None.
            rank: Rank of the current node. Auto-detected if None. Defaults to None.
            mode: Monitoring mode, options are "auto", "cluster", "node". Defaults to "auto".
            
        Returns:
            Dict[str, Any]: Distributed monitoring configuration.
        """
        self.logger.info("Configuring distributed monitoring", {"world_size": world_size, "rank": rank})
        # Retrieve the unified configuration manager
        from utils.config.manager import PiscesLxCoreConfigManager
        config_manager = PiscesLxCoreConfigManager.get_instance()
        base_config = config_manager.get("observability", {})
        config = create_observability_config(base_config, "distributed", world_size, mode)
        # Start a new monitoring session
        session_id = self._start_monitoring_session(config)
        
        result = {
            "session_id": session_id,
            "world_size": config.world_size,
            "rank": config.rank,
            "mode": mode,
            "metrics": config.metrics,
            "communication_tracking": config.communication_tracking,
            "synchronization_tracking": config.synchronization_tracking,
            "fault_tolerance": config.fault_tolerance,
            "status": "active"
        }
        
        return result
    
    def get_device_report(self, detailed: bool = False) -> Dict[str, Any]:
        """Generate a standardized device detection report.
        
        Args:
            detailed: Whether to generate a detailed report. Defaults to False.
            
        Returns:
            Dict[str, Any]: Standardized device detection report. If an error occurs, the report contains an error status.
            
        Raises:
            Exception: If an error occurs while generating the report, it will be logged.
        """
        try:
            # Fetch current metrics from the service
            metrics = self._service.metrics_collector.get_current_metrics()
            registry = self._service.metrics_registry()
            
            # Construct the basic report
            report = {
                "status": "healthy" if metrics.health_score >= 90 else "degraded",
                "timestamp": metrics.timestamp,
                "gpu_info": self._service.runtime_meta.get("gpu_info", []),
                "summary": {
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "throughput": metrics.throughput,
                    "p95_latency": metrics.p95_latency,
                    "error_rate": metrics.error_rate,
                    "health_score": metrics.health_score,
                }
            }
            
            if detailed:
                # Add detailed metrics and trend analysis
                snapshot = registry.snapshot()
                report["detailed_metrics"] = snapshot
                report["trend_analysis"] = {
                    "drift_detected": metrics.p95_drift > 0.1,
                    "drift_magnitude": metrics.p95_drift
                }
            
            return report
            
        except Exception as e:
            self.logger.error("Failed to generate device report", error=str(e), error_class=type(e).__name__)
            return {"status": "error", "error": str(e)}
    
    def stop_monitoring(self, session_id: str = None) -> bool:
        """Stop monitoring sessions.
        
        Args:
            session_id: Session ID. If None, stop all sessions. Defaults to None.
            
        Returns:
            bool: True if the sessions were successfully stopped, False otherwise.
        """
        try:
            if session_id:
                # Stop the specified session
                if session_id in self._active_sessions:
                    self._manager.stop_monitoring()
                    del self._active_sessions[session_id]
                    self.logger.info("Monitoring session stopped", {"session_id": session_id})
                    return True
                else:
                    self.logger.warning("Session does not exist", session_id=session_id)
                    return False
            else:
                # Stop all active sessions
                for sid in list(self._active_sessions.keys()):
                    self.stop_monitoring(sid)
                self.logger.info("All monitoring sessions stopped")
                return True
                
        except Exception as e:
            self.logger.error("Failed to stop monitoring", error=str(e), error_class=type(e).__name__)
            return False
    
    def record_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record training metrics to the metrics registry.
        
        Args:
            metrics: Dictionary containing training metrics.
            
        Raises:
            Exception: If an error occurs while recording metrics, it will be logged.
        """
        try:
            registry = self._service.metrics_registry()
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    counter = registry.counter(f"training_{key}")
                    counter.inc(value)
        except Exception as e:
            self.logger.error("Failed to record training metrics", error=str(e), error_class=type(e).__name__)
    
    def get_active_sessions(self) -> List[str]:
        """Get the list of active monitoring session IDs.
        
        Returns:
            List[str]: List of active session IDs.
        """
        return list(self._active_sessions.keys())
    
    def get_session_config(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the configuration of a specific monitoring session.
        
        Args:
            session_id: Session ID.
            
        Returns:
            Optional[Dict[str, Any]]: Session configuration if the session exists, None otherwise.
        """
        return self._active_sessions.get(session_id)
    
    def _start_monitoring_session(self, config: PiscesL1CoreMonitoringConfig) -> str:
        """Internal method to start a monitoring session.
        
        Args:
            config: Monitoring configuration object.
            
        Returns:
            str: Session ID of the started monitoring session.
            
        Raises:
            RuntimeError: If failed to start the monitoring session.
        """
        session_id = config.session_id
        # Convert the configuration object to a dictionary for the manager
        config_dict = {
            "session_id": session_id,
            "interval": config.interval,
            "cache_enabled": config.cache_enabled,
            "metrics": config.metrics,
            "monitoring_level": config.monitoring_level
        }
        # Start the monitoring session
        success = self._manager.start_monitoring(config_dict)
        if not success:
            raise RuntimeError(f"Failed to start monitoring session {session_id}")
        # Record the active session
        self._active_sessions[session_id] = {
            "config": config,
            "start_time": time.time(),
            "type": self._get_session_type(config)
        }
        
        return session_id
    
    def _get_session_type(self, config: PiscesL1CoreMonitoringConfig) -> str:
        """Determine the type of a monitoring session based on its session ID.
        
        Args:
            config: Monitoring configuration object.
            
        Returns:
            str: Session type, which can be "training", "inference", "distributed", or "unknown".
        """
        if "training" in config.session_id:
            return "training"
        elif "inference" in config.session_id:
            return "inference"
        elif "distributed" in config.session_id:
            return "distributed"
        else:
            return "unknown"