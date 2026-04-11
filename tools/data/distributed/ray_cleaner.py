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
Ray Distributed Cleaner for large-scale data processing.

This module provides distributed data cleaning capabilities using Ray,
enabling processing of TB-scale datasets across multiple nodes.

Key Features:
    - Distributed batch processing
    - Automatic work distribution
    - Progress tracking
    - Fault tolerance

Usage:
    >>> from tools.data.distributed import PiscesLxDataRayDistributedCleaner
    >>> cleaner = PiscesLxDataRayDistributedCleaner(parallelism=8)
    >>> cleaned_ds = cleaner.clean(dataset_path, clean_fn)
"""

from typing import Any, Callable, Dict, List, Optional, Union
import os


class PiscesLxDataRayDistributedCleaner:
    """
    Ray-based distributed data cleaner.
    
    This class provides distributed data cleaning using Ray for
    parallel processing across multiple CPUs or nodes.
    
    Attributes:
        parallelism: Number of parallel workers.
        batch_size: Batch size for processing.
        progress_tracking: Whether to track progress.
    
    Example:
        >>> cleaner = PiscesLxDataRayDistributedCleaner(parallelism=8)
        >>> def clean_fn(batch):
        ...     return [text.lower() for text in batch]
        >>> cleaned = cleaner.clean("data.parquet", clean_fn)
    """
    
    def __init__(
        self,
        parallelism: int = 8,
        batch_size: int = 1000,
        progress_tracking: bool = True,
        num_cpus: Optional[int] = None
    ) -> None:
        """
        Initialize the Ray distributed cleaner.
        
        Args:
            parallelism: Number of parallel workers. Defaults to 8.
            batch_size: Batch size for processing. Defaults to 1000.
            progress_tracking: Enable progress tracking. Defaults to True.
            num_cpus: Number of CPUs to use. None for all available.
        """
        self.parallelism = parallelism
        self.batch_size = batch_size
        self.progress_tracking = progress_tracking
        self.num_cpus = num_cpus
        
        self._ray_available = self._check_ray()
        self._initialized = False
    
    def _check_ray(self) -> bool:
        """
        Check if Ray is available.
        
        Returns:
            bool: True if Ray is installed.
        """
        try:
            import ray
            return True
        except ImportError:
            return False
    
    def _init_ray(self) -> None:
        """Initialize Ray if not already initialized."""
        if not self._ray_available:
            raise ImportError("Ray is not installed. Install with: pip install ray")
        
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus, ignore_reinit_error=True)
        self._initialized = True
    
    def _shutdown_ray(self) -> None:
        """Shutdown Ray if initialized by this instance."""
        if self._ray_available and self._initialized:
            try:
                import ray
                ray.shutdown()
                self._initialized = False
            except Exception:
                pass
    
    def clean(
        self,
        data_source: Union[str, Any],
        clean_fn: Callable[[Dict], Dict],
        output_path: Optional[str] = None
    ) -> Any:
        """
        Clean data using distributed processing.
        
        Args:
            data_source: Path to data file or Ray dataset.
            clean_fn: Function to apply to each batch.
            output_path: Optional path to save cleaned data.
            
        Returns:
            Any: Cleaned Ray dataset.
        """
        if not self._ray_available:
            raise ImportError("Ray is not installed. Install with: pip install ray")
        
        self._init_ray()
        
        import ray
        
        if isinstance(data_source, str):
            if data_source.endswith('.parquet'):
                ds = ray.data.read_parquet(data_source)
            elif data_source.endswith('.json') or data_source.endswith('.jsonl'):
                ds = ray.data.read_json(data_source)
            elif data_source.endswith('.csv'):
                ds = ray.data.read_csv(data_source)
            else:
                ds = ray.data.read_parquet(data_source)
        else:
            ds = data_source
        
        cleaned_ds = ds.map_batches(
            clean_fn,
            batch_size=self.batch_size,
            parallelism=self.parallelism
        )
        
        if output_path:
            cleaned_ds.write_parquet(output_path)
        
        return cleaned_ds
    
    def clean_partitioned(
        self,
        data_path: str,
        clean_fn: Callable[[Dict], Dict],
        output_path: str,
        num_partitions: Optional[int] = None
    ) -> None:
        """
        Clean partitioned data.
        
        Args:
            data_path: Path to partitioned data directory.
            clean_fn: Function to apply to each batch.
            output_path: Path to save cleaned data.
            num_partitions: Number of output partitions.
        """
        if num_partitions is None:
            num_partitions = self.parallelism
        
        cleaned_ds = self.clean(data_path, clean_fn)
        cleaned_ds.write_parquet(output_path, num_rows_per_file=100000)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get Ray cluster statistics.
        
        Returns:
            Dict[str, Any]: Cluster statistics.
        """
        if not self._ray_available:
            return {'ray_available': False}
        
        import ray
        
        if not ray.is_initialized():
            return {'ray_available': True, 'initialized': False}
        
        return {
            'ray_available': True,
            'initialized': True,
            'num_nodes': len(ray.nodes()),
            'num_cpus': ray.available_resources().get('CPU', 0),
            'cluster_resources': ray.cluster_resources()
        }
    
    def __enter__(self) -> 'PiscesLxDataRayDistributedCleaner':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self._shutdown_ray()
