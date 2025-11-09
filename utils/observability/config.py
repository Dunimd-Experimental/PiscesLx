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

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("PiscesLx.Core.Observability.Config")

@dataclass
class PiscesL1CoreMonitoringConfig:
    """A data class encapsulating monitoring configuration parameters.

    Attributes:
        session_id (str): Session ID for monitoring. Defaults to an empty string.
        mode (str): Monitoring mode. Defaults to "auto".
        monitoring_level (str): Level of monitoring. Defaults to "performance".
        model_size (str): Size of the model. Defaults to "7B".
        batch_size (int): Batch size. Defaults to 1.
        sequence_length (int): Sequence length. Defaults to 2048.
        world_size (int): Number of processes in distributed training. Defaults to 1.
        rank (int): Rank of the current process. Defaults to 0.
        interval (float): Monitoring interval in seconds. Defaults to 10.0.
        cache_enabled (bool): Indicates whether caching is enabled. Defaults to True.
        sampling_rate (float): Sampling rate for monitoring. Defaults to 1.0.
        latency_tracking (bool): Indicates whether to track latency. Defaults to True.
        memory_tracking (bool): Indicates whether to track memory. Defaults to True.
        throughput_tracking (bool): Indicates whether to track throughput. Defaults to True.
        communication_tracking (bool): Indicates whether to track communication. Defaults to False.
        synchronization_tracking (bool): Indicates whether to track synchronization. Defaults to False.
        fault_tolerance (bool): Indicates whether fault tolerance is enabled. Defaults to False.
        metrics (List[str]): List of metrics to monitor. Defaults to an empty list.
    """
    
    logger = logger
    
    # Basic configuration
    session_id: str = ""
    mode: str = "auto"
    monitoring_level: str = "performance"
    
    # Model-related
    model_size: str = "7B"
    batch_size: int = 1
    sequence_length: int = 2048
    
    # Distributed-related
    world_size: int = 1
    rank: int = 0
    
    # Runtime configuration
    interval: float = 10.0
    cache_enabled: bool = True
    sampling_rate: float = 1.0
    
    # Feature switches
    latency_tracking: bool = True
    memory_tracking: bool = True
    throughput_tracking: bool = True
    communication_tracking: bool = False
    synchronization_tracking: bool = False
    fault_tolerance: bool = False
    
    # Metrics list
    metrics: List[str] = field(default_factory=list)

@dataclass
class PiscesL1CoreMonitoringConfig:
    """A data class encapsulating monitoring configuration parameters.

    Attributes:
        session_id (str): Session ID for monitoring. Defaults to an empty string.
        mode (str): Monitoring mode. Defaults to "auto".
        monitoring_level (str): Level of monitoring. Defaults to "performance".
        model_size (str): Size of the model. Defaults to "7B".
        batch_size (int): Batch size. Defaults to 1.
        sequence_length (int): Sequence length. Defaults to 2048.
        world_size (int): Number of processes in distributed training. Defaults to 1.
        rank (int): Rank of the current process. Defaults to 0.
        interval (float): Monitoring interval in seconds. Defaults to 10.0.
        cache_enabled (bool): Indicates whether caching is enabled. Defaults to True.
        sampling_rate (float): Sampling rate for monitoring. Defaults to 1.0.
        latency_tracking (bool): Indicates whether to track latency. Defaults to True.
        memory_tracking (bool): Indicates whether to track memory. Defaults to True.
        throughput_tracking (bool): Indicates whether to track throughput. Defaults to True.
        communication_tracking (bool): Indicates whether to track communication. Defaults to False.
        synchronization_tracking (bool): Indicates whether to track synchronization. Defaults to False.
        fault_tolerance (bool): Indicates whether fault tolerance is enabled. Defaults to False.
        metrics (List[str]): List of metrics to monitor. Defaults to an empty list.
    """
    
    logger = logger
    
    # Basic configuration
    session_id: str = ""
    mode: str = "auto"
    monitoring_level: str = "performance"
    
    # Model-related
    model_size: str = "7B"
    batch_size: int = 1
    sequence_length: int = 2048
    
    # Distributed-related
    world_size: int = 1
    rank: int = 0
    
    # Runtime configuration
    interval: float = 10.0
    cache_enabled: bool = True
    sampling_rate: float = 1.0
    
    # Feature switches
    latency_tracking: bool = True
    memory_tracking: bool = True
    throughput_tracking: bool = True
    communication_tracking: bool = False
    synchronization_tracking: bool = False
    fault_tolerance: bool = False
    
    # Metrics list
    metrics: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generates a session ID if none is provided.
        
        Logs an info message upon successful session ID generation,
        or an error message if an exception occurs.
        """
        try:
            if not self.session_id:
                self.session_id = f"monitor_{int(time.time())}"
                self.logger.info(f"Generated session_id: {self.session_id}")
        except Exception as e:
            self.logger.error(f"Failed to generate session_id: {str(e)}")

def create_observability_config(mode: str = "training", **kwargs) -> Dict[str, Any]:
    """Creates a simplified observability configuration.

    Args:
        mode (str): Configuration mode ("training", "inference", or "distributed"). Defaults to "training".
        **kwargs: Additional configuration parameters.

    Returns:
        Dict[str, Any]: Observability configuration dictionary.
    """
    from utils.config.manager import PiscesLxCoreConfigManager
    
    config_manager = PiscesLxCoreConfigManager()
    base_config = config_manager.get("observability", {})
    
    configs = {
        "training": {
            "mode": "training",
            "metrics_interval": kwargs.get("metrics_interval", 30),
            "enable_gpu_monitoring": kwargs.get("enable_gpu_monitoring", True),
            "enable_memory_tracking": kwargs.get("enable_memory_tracking", True),
            "alert_thresholds": {
                "gpu_memory_usage": kwargs.get("gpu_memory_threshold", 0.9),
                "cpu_usage": kwargs.get("cpu_threshold", 0.8),
                "memory_usage": kwargs.get("memory_threshold", 0.85),
                "disk_usage": kwargs.get("disk_threshold", 0.9)
            }
        },
        "inference": {
            "mode": "inference",
            "metrics_interval": kwargs.get("metrics_interval", 60),
            "enable_gpu_monitoring": kwargs.get("enable_gpu_monitoring", True),
            "enable_memory_tracking": kwargs.get("enable_memory_tracking", False),
            "alert_thresholds": {
                "gpu_memory_usage": kwargs.get("gpu_memory_threshold", 0.95),
                "cpu_usage": kwargs.get("cpu_threshold", 0.7),
                "memory_usage": kwargs.get("memory_threshold", 0.8),
                "disk_usage": kwargs.get("disk_threshold", 0.8)
            }
        },
        "distributed": {
            "mode": "distributed",
            "metrics_interval": kwargs.get("metrics_interval", 15),
            "enable_gpu_monitoring": kwargs.get("enable_gpu_monitoring", True),
            "enable_memory_tracking": kwargs.get("enable_memory_tracking", True),
            "alert_thresholds": {
                "gpu_memory_usage": kwargs.get("gpu_memory_threshold", 0.85),
                "cpu_usage": kwargs.get("cpu_threshold", 0.75),
                "memory_usage": kwargs.get("memory_threshold", 0.8),
                "disk_usage": kwargs.get("disk_threshold", 0.85)
            }
        }
    }
    
    return configs.get(mode, configs["training"])

class PiscesL1CoreMonitoringConfig:
    """A class for creating different types of monitoring configurations."""

    @staticmethod
    def create_training_config(model_size: str = "7B", mode: str = "auto") -> 'PiscesL1CoreMonitoringConfig':
        """Creates a monitoring configuration for training.

        Args:
            model_size (str): Size of the model. Defaults to "7B".
            mode (str): Monitoring mode. Defaults to "auto".

        Returns:
            PiscesL1CoreMonitoringConfig: A monitoring configuration object for training.
        """
        from utils.config.manager import PiscesL1CoreConfigManager
        level = PiscesL1CoreConfigManager._determine_level(model_size, mode)
        metrics = PiscesL1CoreConfigManager._get_training_metrics(level)
        sampling_rate = PiscesL1CoreConfigManager.SAMPLING_RATES[level]
        
        return PiscesL1CoreMonitoringConfig(
            session_id=f"training_{int(time.time())}",
            mode=mode,
            monitoring_level=level,
            model_size=model_size,
            metrics=metrics,
            sampling_rate=sampling_rate,
            latency_tracking=True,
            memory_tracking=True,
            throughput_tracking=True
        )
    
    @staticmethod
    def create_inference_config(batch_size: int = 1, sequence_length: int = 2048, 
                              mode: str = "auto") -> 'PiscesL1CoreMonitoringConfig':
        """Creates a monitoring configuration for inference.

        Args:
            batch_size (int): Batch size. Defaults to 1.
            sequence_length (int): Sequence length. Defaults to 2048.
            mode (str): Monitoring mode. Defaults to "auto".

        Returns:
            PiscesL1CoreMonitoringConfig: A monitoring configuration object for inference.
        """
        from utils.config.manager import PiscesL1CoreConfigManager
        inference_load = batch_size * sequence_length
        
        if mode == "auto":
            mode = "performance" if inference_load > 8192 else "minimal"
        
        metrics = PiscesL1CoreConfigManager._get_inference_metrics(mode)
        
        return PiscesL1CoreMonitoringConfig(
            session_id=f"inference_{int(time.time())}",
            mode=mode,
            batch_size=batch_size,
            sequence_length=sequence_length,
            metrics=metrics,
            latency_tracking=mode in ["performance", "detailed"],
            memory_tracking=mode in ["memory", "detailed"],
            throughput_tracking=True
        )
    
    @staticmethod
    def create_distributed_config(world_size: int = None, rank: int = None, 
                                mode: str = "auto") -> 'PiscesL1CoreMonitoringConfig':
        """Creates a monitoring configuration for distributed training.

        Args:
            world_size (int, optional): Number of processes in distributed training. 
                If None, it will be retrieved from the environment variable 'WORLD_SIZE'. Defaults to None.
            rank (int, optional): Rank of the current process. 
                If None, it will be retrieved from the environment variable 'RANK'. Defaults to None.
            mode (str): Monitoring mode. Defaults to "auto".

        Returns:
            PiscesL1CoreMonitoringConfig: A monitoring configuration object for distributed training.
        """
        from utils.config.manager import PiscesL1CoreConfigManager
        # Automatic environment detection
        world_size = world_size or int(os.environ.get('WORLD_SIZE', 1))
        rank = rank or int(os.environ.get('RANK', 0))
        
        # Intelligent distributed configuration
        if world_size == 1:
            monitoring_scope = "single_node"
        elif world_size <= 4:
            monitoring_scope = "small_cluster"
        else:
            monitoring_scope = "large_cluster"
        
        metrics = PiscesL1CoreConfigManager._get_distributed_metrics(monitoring_scope)
        
        return PiscesL1CoreMonitoringConfig(
            session_id=f"distributed_{rank}_{int(time.time())}",
            mode=mode,
            world_size=world_size,
            rank=rank,
            metrics=metrics,
            communication_tracking=world_size > 1,
            synchronization_tracking=world_size > 1,
            fault_tolerance=monitoring_scope in ["small_cluster", "large_cluster"]
        )
    
    @staticmethod
    def _determine_level(model_size: str, mode: str) -> str:
        """Intelligently determines the monitoring level.

        Args:
            model_size (str): Size of the model.
            mode (str): Monitoring mode. If not "auto", it will be returned directly.

        Returns:
            str: The determined monitoring level.
        """
        from utils.config.manager import PiscesL1CoreConfigManager
        if mode != "auto":
            return mode
        
        # Parse the model size
        if model_size.endswith("B"):
            try:
                size_num = float(model_size[:-1])
                for level, threshold in PiscesL1CoreConfigManager.MODEL_THRESHOLDS.items():
                    if size_num < threshold:
                        return level
            except ValueError:
                PiscesL1CoreMonitoringConfig.logger.warning(
                    f"Failed to parse model size '{model_size}'. Using default monitoring level 'performance'."
                )
        
        return "performance"
    
    @staticmethod
    def _get_training_metrics(level: str) -> List[str]:
        """Gets the training monitoring metrics based on the monitoring level.

        Args:
            level (str): Monitoring level.

        Returns:
            List[str]: A list of training monitoring metrics.
        """
        base = ["loss", "learning_rate", "iteration_time"]
        perf = ["cpu_usage", "memory_usage", "throughput", "p95_latency", "error_rate"]
        gpu = ["gpu_utilization", "gpu_memory_usage"]
        
        if level == "minimal":
            return base
        elif level == "performance":
            return base + perf + gpu
        else:  # detailed, premium
            return base + perf + gpu
    
    @staticmethod
    def _get_inference_metrics(mode: str) -> List[str]:
        """Gets the inference monitoring metrics based on the monitoring mode.

        Args:
            mode (str): Monitoring mode.

        Returns:
            List[str]: A list of inference monitoring metrics.
        """
        base = ["latency", "throughput", "p95_latency", "error_rate"]
        
        if mode in ("performance", "detailed"):
            return base + ["gpu_utilization", "memory_usage"]
        elif mode == "memory":
            return base + ["memory_usage"]
        return base
    
    @staticmethod
    def _get_distributed_metrics(scope: str) -> List[str]:
        """Gets the distributed monitoring metrics based on the monitoring scope.

        Args:
            scope (str): Monitoring scope, e.g., "single_node", "small_cluster", "large_cluster".

        Returns:
            List[str]: A list of distributed monitoring metrics.
        """
        base = ["local_loss", "local_throughput", "node_health"]
        
        if scope == "single_node":
            return base
        else:  # small_cluster, large_cluster
            return base + ["sync_time"]
