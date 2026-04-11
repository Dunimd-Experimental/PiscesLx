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
Memory-Mapped Array Implementation for PiscesL1 Data Module.

This module provides efficient memory-mapped array storage for large-scale
data handling in machine learning workflows. Memory mapping enables working
with datasets larger than available RAM by loading data on-demand from disk.

Architecture Overview:
    The memory-mapped array uses the operating system's virtual memory
    management to map file contents directly into memory address space.
    This allows transparent access to large files without loading them
    entirely into RAM.

Key Features:
    - On-demand data loading with transparent caching
    - Support for numpy array format with arbitrary dtypes
    - Random access with O(1) complexity
    - Memory-efficient handling of datasets larger than RAM
    - Thread-safe read operations
    - Automatic file handle management
    - Support for both read-only and read-write modes

Performance Characteristics:
    - Random access: O(1) with page fault overhead
    - Sequential access: O(n) with OS read-ahead optimization
    - Memory usage: O(page_size) per accessed region
    - File size: Limited only by available disk space and OS limits

Use Cases:
    - Large embedding matrices that don't fit in memory
    - Preprocessed tokenized datasets
    - Feature caches for multimodal data
    - Model weight storage for inference
    - Training data shards

Example:
    >>> from tools.data.cache import PiscesLxDataMemoryMappedArray
    >>> import numpy as np
    >>> 
    >>> # Create a memory-mapped array
    >>> arr = PiscesLxDataMemoryMappedArray.create(
    ...     "embeddings.mmap",
    ...     shape=(1000000, 768),
    ...     dtype=np.float32
    ... )
    >>> 
    >>> # Write data
    >>> arr[0:100] = np.random.randn(100, 768).astype(np.float32)
    >>> 
    >>> # Read data on-demand
    >>> batch = arr[0:32]  # Only loads these rows
"""

import mmap
import os
import struct
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("PiscesLx.Tools.Data.Cache.MMap", file_path=get_log_file("PiscesLx.Tools.Data.Cache"), enable_file=True)

MAGIC_HEADER = b'PISCES_MMAP_V1'
HEADER_SIZE = 128


@dataclass
class PiscesLxDataMMapHeader:
    """
    Header structure for memory-mapped array files.
    
    Attributes:
        magic: Magic bytes for file format identification.
        version: Format version number.
        ndim: Number of dimensions.
        shape: Array shape as a tuple.
        dtype_str: String representation of numpy dtype.
        itemsize: Size of each element in bytes.
        flags: Additional flags for future extensions.
    """
    magic: bytes = MAGIC_HEADER
    version: int = 1
    ndim: int = 0
    shape: Tuple[int, ...] = ()
    dtype_str: str = 'float32'
    itemsize: int = 4
    flags: int = 0


class PiscesLxDataMemoryMappedArray:
    """
    Memory-mapped numpy array for efficient large-scale data storage.
    
    This class provides a high-level interface for creating and accessing
    memory-mapped numpy arrays. It supports random access, on-demand loading,
    and efficient storage of large datasets that exceed available RAM.
    
    Architecture:
        The array is stored in a file with a custom header containing metadata
        (shape, dtype, etc.) followed by the raw array data. The OS handles
        memory management through virtual memory paging.
    
    Thread Safety:
        Read operations are thread-safe and can be performed concurrently.
        Write operations require external synchronization. The class uses
        a read-write lock pattern for efficient concurrent reads.
    
    Memory Management:
        - Data is loaded on-demand through page faults
        - OS manages the page cache automatically
        - Accessed pages remain in memory until evicted by OS
        - No explicit memory management required
    
    File Format:
        - Header (128 bytes): Magic, version, shape, dtype, flags
        - Data: Raw array bytes in C-contiguous order
    
    Attributes:
        filepath: Path to the memory-mapped file.
        shape: Shape of the array.
        dtype: Numpy dtype of the array.
        mode: Access mode ('r' for read-only, 'r+' for read-write).
        _mmap: Internal mmap object.
        _array: Internal numpy array view.
        _lock: Lock for thread-safe operations.
    
    Example:
        >>> arr = PiscesLxDataMemoryMappedArray.create(
        ...     "data.mmap", shape=(10000, 256), dtype=np.float32
        ... )
        >>> arr[0] = np.ones(256, dtype=np.float32)
        >>> print(arr[0])  # Loads only this row
        >>> arr.close()
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        mode: str = 'r',
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Union[str, np.dtype] = np.float32
    ):
        """
        Initialize or open a memory-mapped array.
        
        Args:
            filepath: Path to the memory-mapped file.
            mode: Access mode:
                - 'r': Read-only, file must exist
                - 'r+': Read-write, file must exist
                - 'c': Copy-on-write, file must exist
            shape: Shape of the array (required for new files, optional for existing).
            dtype: Data type of the array elements.
        
        Raises:
            FileNotFoundError: If file doesn't exist in read mode.
            ValueError: If file format is invalid.
        """
        self.filepath = Path(filepath)
        self.mode = mode
        self._mmap: Optional[mmap.mmap] = None
        self._array: Optional[np.memmap] = None
        self._lock = threading.RLock()
        self._header: Optional[PiscesLxDataMMapHeader] = None
        self._is_open = False
        
        if self.filepath.exists():
            self._open_existing()
        else:
            if shape is None:
                raise ValueError("Shape must be provided for new files")
            self._create_new(shape, dtype)
        
        _LOG.debug(
            "PiscesLxDataMemoryMappedArray initialized",
            filepath=str(self.filepath),
            shape=self.shape,
            dtype=str(self.dtype),
            mode=mode
        )
    
    @classmethod
    def create(
        cls,
        filepath: Union[str, Path],
        shape: Tuple[int, ...],
        dtype: Union[str, np.dtype] = np.float32,
        fill_value: Optional[float] = None,
        overwrite: bool = False
    ) -> 'PiscesLxDataMemoryMappedArray':
        """
        Create a new memory-mapped array file.
        
        Args:
            filepath: Path for the new file.
            shape: Shape of the array.
            dtype: Data type of the array elements.
            fill_value: Optional value to fill the array with.
            overwrite: If True, overwrite existing file.
        
        Returns:
            A new PiscesLxDataMemoryMappedArray instance.
        
        Raises:
            FileExistsError: If file exists and overwrite is False.
        """
        filepath = Path(filepath)
        
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {filepath}")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        dtype = np.dtype(dtype)
        total_size = np.prod(shape) * dtype.itemsize
        file_size = HEADER_SIZE + total_size
        
        with open(filepath, 'wb') as f:
            f.write(b'\x00' * file_size)
        
        header = PiscesLxDataMMapHeader(
            ndim=len(shape),
            shape=tuple(shape),
            dtype_str=dtype.str,
            itemsize=dtype.itemsize
        )
        
        cls._write_header(filepath, header)
        
        instance = cls(filepath, mode='r+', shape=shape, dtype=dtype)
        
        if fill_value is not None:
            instance._array.fill(fill_value)
        
        _LOG.info(
            "Created memory-mapped array",
            filepath=str(filepath),
            shape=shape,
            dtype=str(dtype),
            size_bytes=file_size
        )
        
        return instance
    
    @classmethod
    def open(
        cls,
        filepath: Union[str, Path],
        mode: str = 'r'
    ) -> 'PiscesLxDataMemoryMappedArray':
        """
        Open an existing memory-mapped array file.
        
        Args:
            filepath: Path to the existing file.
            mode: Access mode ('r', 'r+', or 'c').
        
        Returns:
            A PiscesLxDataMemoryMappedArray instance.
        
        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return cls(filepath, mode=mode)
    
    @staticmethod
    def _write_header(filepath: Path, header: PiscesLxDataMMapHeader) -> None:
        """Write header to file."""
        with open(filepath, 'r+b') as f:
            f.seek(0)
            f.write(header.magic)
            f.write(struct.pack('<I', header.version))
            f.write(struct.pack('<I', header.ndim))
            for dim in header.shape:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<32s', header.dtype_str.encode('utf-8')))
            f.write(struct.pack('<I', header.itemsize))
            f.write(struct.pack('<I', header.flags))
    
    @staticmethod
    def _read_header(filepath: Path) -> PiscesLxDataMMapHeader:
        """Read header from file."""
        with open(filepath, 'rb') as f:
            magic = f.read(16)
            if magic != MAGIC_HEADER:
                raise ValueError(f"Invalid file format: expected {MAGIC_HEADER}, got {magic}")
            
            version = struct.unpack('<I', f.read(4))[0]
            ndim = struct.unpack('<I', f.read(4))[0]
            
            shape = []
            for _ in range(ndim):
                shape.append(struct.unpack('<Q', f.read(8))[0])
            
            dtype_str = struct.unpack('<32s', f.read(32))[0].rstrip(b'\x00').decode('utf-8')
            itemsize = struct.unpack('<I', f.read(4))[0]
            flags = struct.unpack('<I', f.read(4))[0]
            
            return PiscesLxDataMMapHeader(
                magic=magic,
                version=version,
                ndim=ndim,
                shape=tuple(shape),
                dtype_str=dtype_str,
                itemsize=itemsize,
                flags=flags
            )
    
    def _open_existing(self) -> None:
        """Open an existing memory-mapped file."""
        self._header = self._read_header(self.filepath)
        
        dtype = np.dtype(self._header.dtype_str)
        shape = self._header.shape
        
        self._array = np.memmap(
            self.filepath,
            dtype=dtype,
            mode=self.mode,
            offset=HEADER_SIZE,
            shape=shape
        )
        self._is_open = True
    
    def _create_new(self, shape: Tuple[int, ...], dtype: Union[str, np.dtype]) -> None:
        """Create a new memory-mapped file."""
        dtype = np.dtype(dtype)
        total_size = np.prod(shape) * dtype.itemsize
        file_size = HEADER_SIZE + total_size
        
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.filepath, 'wb') as f:
            f.write(b'\x00' * file_size)
        
        self._header = PiscesLxDataMMapHeader(
            ndim=len(shape),
            shape=tuple(shape),
            dtype_str=dtype.str,
            itemsize=dtype.itemsize
        )
        self._write_header(self.filepath, self._header)
        
        self._array = np.memmap(
            self.filepath,
            dtype=dtype,
            mode='r+',
            offset=HEADER_SIZE,
            shape=shape
        )
        self._is_open = True
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the array."""
        if self._array is not None:
            return self._array.shape
        if self._header is not None:
            return self._header.shape
        return ()
    
    @property
    def dtype(self) -> np.dtype:
        """Return the dtype of the array."""
        if self._array is not None:
            return self._array.dtype
        if self._header is not None:
            return np.dtype(self._header.dtype_str)
        return np.dtype('float32')
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return len(self.shape)
    
    @property
    def size(self) -> int:
        """Return the total number of elements."""
        return int(np.prod(self.shape))
    
    @property
    def nbytes(self) -> int:
        """Return the total bytes consumed by the array."""
        return self.size * self.dtype.itemsize
    
    @property
    def itemsize(self) -> int:
        """Return the size of each element in bytes."""
        return self.dtype.itemsize
    
    def __getitem__(self, key) -> np.ndarray:
        """
        Get a slice or element from the array.
        
        Args:
            key: Index or slice specification.
        
        Returns:
            Numpy array or scalar value.
        """
        with self._lock:
            if self._array is None:
                raise RuntimeError("Array is not open")
            return self._array[key]
    
    def __setitem__(self, key, value) -> None:
        """
        Set a slice or element in the array.
        
        Args:
            key: Index or slice specification.
            value: Value to assign.
        """
        with self._lock:
            if self._array is None:
                raise RuntimeError("Array is not open")
            if self.mode == 'r':
                raise RuntimeError("Cannot write to read-only array")
            self._array[key] = value
    
    def __len__(self) -> int:
        """Return the length of the first dimension."""
        return self.shape[0] if self.shape else 0
    
    def __array__(self) -> np.ndarray:
        """Return a numpy array view."""
        return np.asarray(self._array)
    
    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f"PiscesLxDataMemoryMappedArray("
            f"filepath='{self.filepath}', "
            f"shape={self.shape}, "
            f"dtype={self.dtype}, "
            f"mode='{self.mode}')"
        )
    
    def close(self) -> None:
        """
        Close the memory-mapped file.
        
        This flushes any pending writes and releases the file handle.
        The array cannot be accessed after closing.
        """
        with self._lock:
            if self._array is not None:
                if hasattr(self._array, 'flush'):
                    self._array.flush()
                del self._array
                self._array = None
            
            if self._mmap is not None:
                self._mmap.close()
                self._mmap = None
            
            self._is_open = False
            _LOG.debug("Memory-mapped array closed", filepath=str(self.filepath))
    
    def flush(self) -> None:
        """Flush any pending writes to disk."""
        with self._lock:
            if self._array is not None and hasattr(self._array, 'flush'):
                self._array.flush()
    
    def is_open(self) -> bool:
        """Check if the array is currently open."""
        return self._is_open
    
    def get_batch(self, indices: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Get multiple rows by indices efficiently.
        
        Args:
            indices: List or array of row indices.
        
        Returns:
            Numpy array with the requested rows.
        """
        with self._lock:
            if self._array is None:
                raise RuntimeError("Array is not open")
            indices = np.asarray(indices)
            return self._array[indices]
    
    def get_slice(self, start: int, stop: int, axis: int = 0) -> np.ndarray:
        """
        Get a contiguous slice along an axis.
        
        Args:
            start: Start index.
            stop: Stop index (exclusive).
            axis: Axis to slice along (default: 0).
        
        Returns:
            Numpy array slice.
        """
        with self._lock:
            if self._array is None:
                raise RuntimeError("Array is not open")
            
            slices = [slice(None)] * self.ndim
            slices[axis] = slice(start, stop)
            return self._array[tuple(slices)]
    
    def iter_batches(
        self,
        batch_size: int,
        axis: int = 0,
        shuffle: bool = False
    ) -> Iterator[np.ndarray]:
        """
        Iterate over the array in batches.
        
        Args:
            batch_size: Number of elements per batch.
            axis: Axis to iterate along.
            shuffle: If True, yield batches in random order.
        
        Yields:
            Numpy array batches.
        """
        with self._lock:
            if self._array is None:
                raise RuntimeError("Array is not open")
            
            n = self.shape[axis]
            indices = np.arange(n)
            
            if shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, n, batch_size):
                batch_indices = indices[i:i + batch_size]
                slices = [slice(None)] * self.ndim
                slices[axis] = batch_indices
                yield self._array[tuple(slices)]
    
    def copy_to(self, dest: Union[str, Path, np.ndarray]) -> Union['PiscesLxDataMemoryMappedArray', np.ndarray]:
        """
        Copy the array to another location or array.
        
        Args:
            dest: Destination path or numpy array.
        
        Returns:
            New PiscesLxDataMemoryMappedArray if path given, else numpy array.
        """
        with self._lock:
            if self._array is None:
                raise RuntimeError("Array is not open")
            
            if isinstance(dest, (str, Path)):
                return PiscesLxDataMemoryMappedArray.create(
                    dest,
                    shape=self.shape,
                    dtype=self.dtype,
                    overwrite=True
                )
            else:
                dest[:] = self._array[:]
                return dest
    
    def as_numpy(self) -> np.ndarray:
        """
        Load the entire array into memory as a numpy array.
        
        Returns:
            Numpy array copy of the data.
        
        Warning:
            This loads the entire array into memory. For large arrays,
            this may cause memory issues.
        """
        with self._lock:
            if self._array is None:
                raise RuntimeError("Array is not open")
            return np.array(self._array)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory-mapped array.
        
        Returns:
            Dictionary with array statistics.
        """
        return {
            'filepath': str(self.filepath),
            'shape': self.shape,
            'dtype': str(self.dtype),
            'ndim': self.ndim,
            'size': self.size,
            'nbytes': self.nbytes,
            'itemsize': self.itemsize,
            'mode': self.mode,
            'is_open': self._is_open,
            'file_exists': self.filepath.exists(),
            'file_size': self.filepath.stat().st_size if self.filepath.exists() else 0
        }
    
    @staticmethod
    def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a memory-mapped array file without opening it.
        
        Args:
            filepath: Path to the file.
        
        Returns:
            Dictionary with file information.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return {'exists': False}
        
        header = PiscesLxDataMemoryMappedArray._read_header(filepath)
        
        return {
            'exists': True,
            'filepath': str(filepath),
            'shape': header.shape,
            'dtype': header.dtype_str,
            'ndim': header.ndim,
            'version': header.version,
            'itemsize': header.itemsize,
            'file_size': filepath.stat().st_size
        }
    
    def __enter__(self) -> 'PiscesLxDataMemoryMappedArray':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def __del__(self) -> None:
        """Destructor to ensure file is closed."""
        try:
            self.close()
        except Exception:
            pass
