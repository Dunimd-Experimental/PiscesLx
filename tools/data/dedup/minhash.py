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
MinHash LSH Deduplication for efficient near-duplicate detection.

This module implements MinHash with Locality-Sensitive Hashing (LSH) for
efficiently identifying near-duplicate documents in large datasets.
This approach provides O(1) similarity queries with configurable
false positive/negative rates.

Key Features:
    - Sub-linear similarity search via LSH indexing
    - Configurable similarity threshold
    - Streaming/incremental deduplication support
    - Memory-efficient MinHash signature generation

Usage:
    >>> from tools.data.dedup import PiscesLxDataMinHashDeduplicator
    >>> dedup = PiscesLxDataMinHashDeduplicator(threshold=0.8, num_perm=128)
    >>> 
    >>> # Add documents
    >>> for idx, text in enumerate(documents):
    ...     dedup.add(idx, text)
    >>> 
    >>> # Get unique indices
    >>> unique_indices = dedup.get_unique_indices()
"""

import hashlib
import pickle
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union
import numpy as np


class PiscesLxDataMinHashDeduplicator:
    """
    MinHash LSH deduplicator for near-duplicate document detection.
    
    This class implements MinHash signatures combined with LSH banding
    to efficiently identify similar documents without pairwise comparison.
    
    Attributes:
        threshold: Jaccard similarity threshold for considering duplicates.
        num_perm: Number of permutation functions for MinHash.
        num_bands: Number of bands for LSH indexing.
        rows_per_band: Number of rows per band.
        seed: Random seed for hash functions.
    
    Example:
        >>> dedup = PiscesLxDataMinHashDeduplicator(threshold=0.8)
        >>> dedup.add(0, "This is a sample document")
        >>> dedup.add(1, "This is another sample document")
        >>> unique = dedup.get_unique_indices()
        >>> print(f"Found {len(unique)} unique documents")
    """
    
    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        num_bands: Optional[int] = None,
        seed: int = 42
    ) -> None:
        """
        Initialize the MinHash LSH deduplicator.
        
        Args:
            threshold: Jaccard similarity threshold. Defaults to 0.8.
            num_perm: Number of MinHash permutations. Defaults to 128.
                Higher values = more accurate but slower.
            num_bands: Number of LSH bands. If None, auto-calculated.
            seed: Random seed for reproducibility. Defaults to 42.
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.seed = seed
        
        if num_bands is None:
            self.num_bands = self._optimal_bands(threshold, num_perm)
        else:
            self.num_bands = num_bands
        
        self.rows_per_band = num_perm // self.num_bands
        
        self._hash_functions = self._generate_hash_functions()
        self._lsh_index: Dict[int, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
        self._signatures: Dict[int, np.ndarray] = {}
        self._documents: Dict[int, str] = {}
    
    def _optimal_bands(self, threshold: float, num_perm: int) -> int:
        """
        Calculate optimal number of bands for given threshold.
        
        Uses the formula: bands = (threshold^(-1/r) - 1) where r = rows_per_band
        
        Args:
            threshold: Target similarity threshold.
            num_perm: Number of permutations.
            
        Returns:
            int: Optimal number of bands.
        """
        for bands in range(2, num_perm):
            rows = num_perm // bands
            if rows < 1:
                continue
            prob = (1.0 / bands) ** (1.0 / rows)
            if prob >= threshold:
                return bands
        return max(2, num_perm // 8)
    
    def _generate_hash_functions(self) -> List[callable]:
        """
        Generate hash functions for MinHash.
        
        Returns:
            List[callable]: List of hash functions.
        """
        rng = np.random.default_rng(self.seed)
        functions = []
        
        for i in range(self.num_perm):
            a = rng.integers(1, 2**31 - 1)
            b = rng.integers(0, 2**31 - 1)
            p = 2**31 - 1
            
            def make_hash(a_val, b_val, p_val):
                def hash_func(x: int) -> int:
                    return ((a_val * x + b_val) % p_val)
                return hash_func
            
            functions.append(make_hash(a, b, p))
        
        return functions
    
    def _tokenize(self, text: str, n: int = 3) -> Set[str]:
        """
        Tokenize text into n-grams.
        
        Args:
            text: Input text.
            n: N-gram size. Defaults to 3 (trigrams).
            
        Returns:
            Set[str]: Set of n-grams.
        """
        text = text.lower().strip()
        tokens = set()
        for i in range(len(text) - n + 1):
            tokens.add(text[i:i+n])
        return tokens
    
    def _compute_signature(self, tokens: Set[str]) -> np.ndarray:
        """
        Compute MinHash signature for a set of tokens.
        
        Args:
            tokens: Set of tokens.
            
        Returns:
            np.ndarray: MinHash signature array.
        """
        if not tokens:
            return np.full(self.num_perm, 2**31 - 1, dtype=np.int64)
        
        signature = np.full(self.num_perm, 2**31 - 1, dtype=np.int64)
        
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16) % (2**31 - 1)
            
            for i, hash_func in enumerate(self._hash_functions):
                h = hash_func(token_hash)
                if h < signature[i]:
                    signature[i] = h
        
        return signature
    
    def _hash_band(self, signature: np.ndarray, band_idx: int) -> str:
        """
        Hash a band of the signature for LSH.
        
        Args:
            signature: MinHash signature.
            band_idx: Band index.
            
        Returns:
            str: Hash string for the band.
        """
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        band = signature[start:end]
        return hashlib.md5(band.tobytes()).hexdigest()
    
    def add(self, idx: int, text: str) -> bool:
        """
        Add a document to the deduplicator.
        
        Args:
            idx: Document index/identifier.
            text: Document text.
            
        Returns:
            bool: True if document is unique, False if duplicate.
        """
        tokens = self._tokenize(text)
        signature = self._compute_signature(tokens)
        
        for band_idx in range(self.num_bands):
            band_hash = self._hash_band(signature, band_idx)
            self._lsh_index[band_idx][band_hash].add(idx)
        
        self._signatures[idx] = signature
        self._documents[idx] = text
        
        return not self._is_duplicate(idx, signature)
    
    def _is_duplicate(self, idx: int, signature: np.ndarray) -> bool:
        """
        Check if a document is a duplicate.
        
        Args:
            idx: Document index.
            signature: MinHash signature.
            
        Returns:
            bool: True if duplicate found.
        """
        candidates: Set[int] = set()
        
        for band_idx in range(self.num_bands):
            band_hash = self._hash_band(signature, band_idx)
            candidates.update(self._lsh_index[band_idx][band_hash])
        
        candidates.discard(idx)
        
        for candidate_idx in candidates:
            if candidate_idx in self._signatures:
                similarity = self._jaccard_similarity(signature, self._signatures[candidate_idx])
                if similarity >= self.threshold:
                    return True
        
        return False
    
    def _jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from MinHash signatures.
        
        Args:
            sig1: First signature.
            sig2: Second signature.
            
        Returns:
            float: Estimated Jaccard similarity.
        """
        return np.mean(sig1 == sig2)
    
    def get_duplicates(self, idx: int) -> List[int]:
        """
        Get all documents similar to the given document.
        
        Args:
            idx: Document index.
            
        Returns:
            List[int]: List of similar document indices.
        """
        if idx not in self._signatures:
            return []
        
        signature = self._signatures[idx]
        candidates: Set[int] = set()
        
        for band_idx in range(self.num_bands):
            band_hash = self._hash_band(signature, band_idx)
            candidates.update(self._lsh_index[band_idx][band_hash])
        
        candidates.discard(idx)
        
        duplicates = []
        for candidate_idx in candidates:
            if candidate_idx in self._signatures:
                similarity = self._jaccard_similarity(signature, self._signatures[candidate_idx])
                if similarity >= self.threshold:
                    duplicates.append(candidate_idx)
        
        return duplicates
    
    def get_unique_indices(self) -> List[int]:
        """
        Get indices of unique documents (first occurrence kept).
        
        Returns:
            List[int]: List of unique document indices.
        """
        unique: Set[int] = set()
        duplicates: Set[int] = set()
        
        for idx in sorted(self._signatures.keys()):
            if idx in duplicates:
                continue
            
            unique.add(idx)
            
            for dup_idx in self.get_duplicates(idx):
                duplicates.add(dup_idx)
        
        return sorted(unique)
    
    def get_duplicate_clusters(self) -> List[List[int]]:
        """
        Get clusters of similar documents.
        
        Returns:
            List[List[int]]: List of duplicate clusters.
        """
        visited: Set[int] = set()
        clusters: List[List[int]] = []
        
        for idx in sorted(self._signatures.keys()):
            if idx in visited:
                continue
            
            cluster = [idx] + self.get_duplicates(idx)
            clusters.append(cluster)
            visited.update(cluster)
        
        return clusters
    
    def save(self, path: str) -> None:
        """
        Save deduplicator state to file.
        
        Args:
            path: File path.
        """
        state = {
            'threshold': self.threshold,
            'num_perm': self.num_perm,
            'num_bands': self.num_bands,
            'rows_per_band': self.rows_per_band,
            'seed': self.seed,
            'signatures': self._signatures,
            'documents': self._documents,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: str) -> None:
        """
        Load deduplicator state from file.
        
        Args:
            path: File path.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.threshold = state['threshold']
        self.num_perm = state['num_perm']
        self.num_bands = state['num_bands']
        self.rows_per_band = state['rows_per_band']
        self.seed = state['seed']
        self._signatures = state['signatures']
        self._documents = state['documents']
        
        self._hash_functions = self._generate_hash_functions()
        self._lsh_index = defaultdict(lambda: defaultdict(set))
        
        for idx, signature in self._signatures.items():
            for band_idx in range(self.num_bands):
                band_hash = self._hash_band(signature, band_idx)
                self._lsh_index[band_idx][band_hash].add(idx)
    
    def __len__(self) -> int:
        """Get number of documents."""
        return len(self._signatures)
    
    def __contains__(self, idx: int) -> bool:
        """Check if document exists."""
        return idx in self._signatures
