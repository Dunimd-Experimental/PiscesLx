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
Dask Distributed Loader for large-scale data loading.

This module provides distributed data loading capabilities using Dask,
enabling memory-efficient loading of TB-scale datasets.

Key Features:
    - Lazy loading with delayed computation
    - Memory-mapped data access
    - Chunked processing
    - Multiple format support

Usage:
    >>> from tools.data.distributed import PiscesLxDataDaskDistributedLoader
    >>> loader = PiscesLxDataDaskDistributedLoader(chunk_size='256MB')
    >>> ddf = loader.load("large_dataset.parquet")
    >>> for chunk in loader.iter_chunks(ddf, batch_size=10000):
    ...     process(chunk)
"""

from typing import Any, Dict, Iterator, List, Optional, Union
import os


class PiscesLxDataDaskDistributedLoader:
    """
    Dask-based distributed data loader.
    
    This class provides lazy loading and distributed processing
    capabilities for large-scale datasets using Dask.
    
    Attributes:
        chunk_size: Size of chunks for lazy loading.
        memory_limit: Memory limit per worker.
        n_workers: Number of Dask workers.
    
    Example:
        >>> loader = PiscesLxDataDaskDistributedLoader(chunk_size='256MB')
        >>> ddf = loader.load("data.parquet")
        >>> batch = loader.get_batch(ddf, 0, 1000)
    """
    
    def __init__(
        self,
        chunk_size: str = '256MB',
        memory_limit: Optional[str] = None,
        n_workers: Optional[int] = None
    ) -> None:
        """
        Initialize the Dask distributed loader.
        
        Args:
            chunk_size: Chunk size for lazy loading. Defaults to '256MB'.
            memory_limit: Memory limit per worker. None for auto.
            n_workers: Number of workers. None for auto.
        """
        self.chunk_size = chunk_size
        self.memory_limit = memory_limit
        self.n_workers = n_workers
        
        self._dask_available = self._check_dask()
        self._client = None
    
    def _check_dask(self) -> bool:
        """
        Check if Dask is available.
        
        Returns:
            bool: True if Dask is installed.
        """
        try:
            import dask
            import dask.dataframe as dd
            return True
        except ImportError:
            return False
    
    def _get_client(self) -> Any:
        """
        Get or create Dask client.
        
        Returns:
            Any: Dask client.
        """
        if not self._dask_available:
            raise ImportError("Dask is not installed. Install with: pip install dask")
        
        if self._client is None:
            try:
                from dask.distributed import Client
                self._client = Client(
                    n_workers=self.n_workers,
                    memory_limit=self.memory_limit
                )
            except ImportError:
                pass
        
        return self._client
    
    def load(
        self,
        path: str,
        format: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> Any:
        """
        Load data lazily using Dask.
        
        Args:
            path: Path to data file or directory.
            format: Data format ('parquet', 'csv', 'json'). Auto-detected if None.
            columns: Columns to load. None for all columns.
            
        Returns:
            Any: Dask DataFrame.
        """
        if not self._dask_available:
            raise ImportError("Dask is not installed. Install with: pip install dask")
        
        import dask.dataframe as dd
        
        if format is None:
            if path.endswith('.parquet'):
                format = 'parquet'
            elif path.endswith('.csv'):
                format = 'csv'
            elif path.endswith('.json') or path.endswith('.jsonl'):
                format = 'json'
            else:
                format = 'parquet'
        
        if format == 'parquet':
            return dd.read_parquet(
                path,
                columns=columns,
                chunksize=self.chunk_size
            )
        elif format == 'csv':
            return dd.read_csv(
                path,
                usecols=columns,
                blocksize=self._parse_size(self.chunk_size)
            )
        elif format == 'json':
            return dd.read_json(
                path,
                lines=True,
                blocksize=self._parse_size(self.chunk_size)
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _parse_size(self, size_str: str) -> int:
        """
        Parse size string to bytes.
        
        Args:
            size_str: Size string like '256MB'.
            
        Returns:
            int: Size in bytes.
        """
        size_str = size_str.upper().strip()
        multipliers = {
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3,
            'TB': 1024 ** 4,
        }
        
        for suffix, mult in multipliers.items():
            if size_str.endswith(suffix):
                return int(float(size_str[:-len(suffix)]) * mult)
        
        return int(size_str)
    
    def get_batch(
        self,
        ddf: Any,
        start: int,
        end: int
    ) -> Any:
        """
        Get a batch of rows from Dask DataFrame.
        
        Args:
            ddf: Dask DataFrame.
            start: Start index.
            end: End index.
            
        Returns:
            Any: Pandas DataFrame batch.
        """
        return ddf.iloc[start:end].compute()
    
    def iter_chunks(
        self,
        ddf: Any,
        batch_size: int = 10000
    ) -> Iterator[Any]:
        """
        Iterate over DataFrame in chunks.
        
        Args:
            ddf: Dask DataFrame.
            batch_size: Number of rows per batch.
            
        Yields:
            Any: Pandas DataFrame batch.
        """
        total_rows = len(ddf)
        
        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            yield self.get_batch(ddf, start, end)
    
    def save(
        self,
        ddf: Any,
        path: str,
        format: str = 'parquet'
    ) -> None:
        """
        Save Dask DataFrame to file.
        
        Args:
            ddf: Dask DataFrame.
            path: Output path.
            format: Output format.
        """
        if format == 'parquet':
            ddf.to_parquet(path)
        elif format == 'csv':
            ddf.to_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_stats(self, ddf: Any) -> Dict[str, Any]:
        """
        Get DataFrame statistics.
        
        Args:
            ddf: Dask DataFrame.
            
        Returns:
            Dict[str, Any]: Statistics.
        """
        return {
            'npartitions': ddf.npartitions,
            'columns': list(ddf.columns),
            'dtypes': ddf.dtypes.to_dict(),
            'memory_usage': ddf.memory_usage(deep=True).compute().to_dict()
        }
    
    def close(self) -> None:
        """Close Dask client."""
        if self._client is not None:
            self._client.close()
            self._client = None
    
    def __enter__(self) -> 'PiscesLxDataDaskDistributedLoader':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
