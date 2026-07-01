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

"""
Exact Deduplication for identifying exact duplicates.

This module provides exact deduplication using cryptographic hashes
(MD5, SHA256) to identify byte-identical documents. This is useful
for removing exact duplicates before applying approximate deduplication.

Key Features:
    - Cryptographic hash-based deduplication
    - Support for MD5 and SHA256 algorithms
    - Streaming/incremental processing
    - Memory-efficient hash storage

Usage:
    >>> from tools.data.dedup import PiscesLxDataExactDeduplicator
    >>> dedup = PiscesLxDataExactDeduplicator(algorithm='sha256')
    >>> 
    >>> # Add documents
    >>> for idx, text in enumerate(documents):
    ...     if dedup.add(idx, text):
    ...         print(f"Document {idx} is unique")
    >>> 
    >>> # Get unique indices
    >>> unique_indices = dedup.get_unique_indices()
"""

import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Set


class PiscesLxDataExactDeduplicator:
    """
    Exact deduplicator using cryptographic hashes.
    
    This class identifies byte-identical documents using hash functions.
    It's fast and memory-efficient, suitable for removing exact duplicates
    before applying more expensive approximate deduplication.
    
    Attributes:
        algorithm: Hash algorithm ('md5' or 'sha256').
        hash_to_idx: Mapping from hash to document indices.
        idx_to_hash: Mapping from index to hash.
    
    Example:
        >>> dedup = PiscesLxDataExactDeduplicator(algorithm='sha256')
        >>> dedup.add(0, "Hello world")
        >>> dedup.add(1, "Hello world")  # Duplicate
        >>> dedup.add(2, "Different text")
        >>> print(dedup.get_unique_indices())  # [0, 2]
    """
    
    def __init__(self, algorithm: str = 'sha256') -> None:
        """
        Initialize the exact deduplicator.
        
        Args:
            algorithm: Hash algorithm to use. Either 'md5' or 'sha256'.
                Defaults to 'sha256' for better collision resistance.
        """
        if algorithm not in ('md5', 'sha256'):
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'md5' or 'sha256'.")
        
        self.algorithm = algorithm
        self.hash_to_idx: Dict[str, int] = {}
        self.idx_to_hash: Dict[int, str] = {}
        self._duplicates: Set[int] = set()
    
    def _compute_hash(self, text: str) -> str:
        """
        Compute hash of text.
        
        Args:
            text: Input text.
            
        Returns:
            str: Hexadecimal hash string.
        """
        encoded = text.encode('utf-8')
        if self.algorithm == 'md5':
            return hashlib.md5(encoded).hexdigest()
        else:
            return hashlib.sha256(encoded).hexdigest()
    
    def add(self, idx: int, text: str) -> bool:
        """
        Add a document to the deduplicator.
        
        Args:
            idx: Document index/identifier.
            text: Document text.
            
        Returns:
            bool: True if document is unique, False if duplicate.
        """
        text_hash = self._compute_hash(text)
        
        if text_hash in self.hash_to_idx:
            self._duplicates.add(idx)
            self.idx_to_hash[idx] = text_hash
            return False
        
        self.hash_to_idx[text_hash] = idx
        self.idx_to_hash[idx] = text_hash
        return True
    
    def add_batch(self, indices: List[int], texts: List[str]) -> List[bool]:
        """
        Add multiple documents at once.
        
        Args:
            indices: List of document indices.
            texts: List of document texts.
            
        Returns:
            List[bool]: List of uniqueness flags.
        """
        if len(indices) != len(texts):
            raise ValueError("Indices and texts must have the same length")
        
        results = []
        for idx, text in zip(indices, texts):
            results.append(self.add(idx, text))
        
        return results
    
    def is_duplicate(self, idx: int) -> bool:
        """
        Check if a document is a duplicate.
        
        Args:
            idx: Document index.
            
        Returns:
            bool: True if duplicate, False if unique.
        """
        return idx in self._duplicates
    
    def get_original(self, idx: int) -> Optional[int]:
        """
        Get the original (first) document index for a duplicate.
        
        Args:
            idx: Duplicate document index.
            
        Returns:
            Optional[int]: Original document index, or None if not a duplicate.
        """
        if idx not in self._duplicates:
            return None
        
        text_hash = self.idx_to_hash.get(idx)
        if text_hash is None:
            return None
        
        return self.hash_to_idx.get(text_hash)
    
    def get_unique_indices(self) -> List[int]:
        """
        Get indices of unique documents.
        
        Returns:
            List[int]: List of unique document indices.
        """
        return sorted(set(self.idx_to_hash.keys()) - self._duplicates)
    
    def get_duplicate_indices(self) -> List[int]:
        """
        Get indices of duplicate documents.
        
        Returns:
            List[int]: List of duplicate document indices.
        """
        return sorted(self._duplicates)
    
    def get_duplicate_groups(self) -> Dict[int, List[int]]:
        """
        Get groups of duplicate documents.
        
        Returns:
            Dict[int, List[int]]: Mapping from original index to duplicate indices.
        """
        groups: Dict[int, List[int]] = defaultdict(list)
        
        for dup_idx in self._duplicates:
            original_idx = self.get_original(dup_idx)
            if original_idx is not None:
                groups[original_idx].append(dup_idx)
        
        return dict(groups)
    
    def get_stats(self) -> dict:
        """
        Get deduplication statistics.
        
        Returns:
            dict: Statistics including total, unique, and duplicate counts.
        """
        return {
            'total_documents': len(self.idx_to_hash),
            'unique_documents': len(self.hash_to_idx),
            'duplicate_documents': len(self._duplicates),
            'duplicate_ratio': len(self._duplicates) / max(len(self.idx_to_hash), 1),
            'algorithm': self.algorithm
        }
    
    def clear(self) -> None:
        """Clear all stored data."""
        self.hash_to_idx.clear()
        self.idx_to_hash.clear()
        self._duplicates.clear()
    
    def __len__(self) -> int:
        """Get total number of documents."""
        return len(self.idx_to_hash)
    
    def __contains__(self, idx: int) -> bool:
        """Check if document exists."""
        return idx in self.idx_to_hash
