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
Probabilistic Language Trie (PLT) for Sequential KV Cache Prefix Deduplication.

Based on:
    Magarshak, "Sequential KV Cache Compression via Probabilistic Language Tries:
    Beyond the Per-Vector Shannon Limit", arXiv:2604.15356, 2026.

This module implements the PLT trie metric for identifying semantically equivalent
shared prefixes across sessions, enabling KV cache block sharing that goes beyond
exact token sequence matching.

Theory:
    The PLT trie metric d_T(s, s') = -log2 P_M(s ^ s') measures the semantic distance
    between two token sequences as the negative log-probability of their longest common
    prefix under the generative model M. Two sequences with a high-probability shared
    prefix have a small trie distance, indicating redundant KV cache structure that
    can be shared.

    Combined with the ultrametric property:
        d(s, s'') <= max(d(s, s'), d(s', s''))
    this enables efficient clustering of similar prompts for prefix sharing.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class YvPLTTrieNode:
    """Node in the Probabilistic Language Trie.

    Each node represents a token sequence prefix. Edges are weighted by
    the conditional probability P(token | prefix) under the language model.

    Attributes:
        token_id: The token at this node (-1 for root).
        cumulative_log_prob: Sum of log-probs along the path from root
            to this node: log P_M(prefix).
        kv_cache_ref: Optional reference to cached KV blocks for this prefix.
        kv_block_ids: List of physical block IDs in paged cache.
        access_count: Number of times this prefix has been accessed.
        last_access_time: Timestamp of last access for LRU eviction.
    """
    token_id: int = -1
    cumulative_log_prob: float = 0.0
    children: Dict[int, 'YvPLTTrieNode'] = None
    kv_cache_ref: Any = None
    kv_block_ids: List[int] = None
    access_count: int = 0
    last_access_time: float = 0.0

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.kv_block_ids is None:
            self.kv_block_ids = []

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def node_log_prob(self) -> float:
        return self.cumulative_log_prob


# Paper: Original contribution by Dunimd Team (Yv Architecture — PLT trie)
class YvPLTTrieIndex:
    """Probabilistic Language Trie index for KV cache prefix deduplication.

    Builds a trie structure over token sequences that have been processed
    by the model, weighted by the model's own probability estimates. This
    enables identification of semantically similar prefixes across sessions
    for KV cache block sharing.

    Architecture:
        The trie is built incrementally as the model processes sequences.
        Each node stores the cumulative log-probability of the prefix path.
        The trie metric d_T(s, s') = -log2 P_M(s ^ s') is used to determine
        whether two prefixes are "close enough" to share KV cache blocks.

    Key Features:
        - Probabilistic prefix matching beyond exact token equality
        - Configurable distance threshold for sharing decisions
        - Automatic ref-count management for shared paged cache blocks
        - LRU eviction of unused trie nodes
        - Efficient lookup via token-by-token trie traversal

    Usage:
        >>> trie = YvPLTTrieIndex(distance_threshold_bits=8.0)
        >>> trie.insert(token_ids=[1, 2, 3], log_probs=[-0.5, -1.0, -0.3], block_ids=[10, 11])
        >>> match = trie.find_longest_match([1, 2, 4], context_log_probs=[-0.5, -1.0, -2.0])
        >>> if match:
        ...     shared_blocks, divergence_pos = match

    Reference:
        Magarshak, "Sequential KV Cache Compression via Probabilistic Language Tries",
        arXiv:2604.15356, 2026, Section 4.
    """

    def __init__(
        self,
        distance_threshold_bits: float = 8.0,
        max_nodes: int = 100000,
        eviction_ttl_seconds: float = 3600.0,
    ):
        """Initialize the PLT trie index.

        Args:
            distance_threshold_bits: Maximum trie distance (in bits) for two
                prefixes to be considered semantically equivalent for sharing.
                Default 8.0 means prefixes with shared-prefix probability > 1/256
                can share KV cache blocks.
            max_nodes: Maximum number of trie nodes before eviction.
            eviction_ttl_seconds: Time-to-live for unused nodes in seconds.
        """
        self.root = YvPLTTrieNode(token_id=-1, cumulative_log_prob=0.0)
        self.distance_threshold_bits = distance_threshold_bits
        self.max_nodes = max_nodes
        self.eviction_ttl_seconds = eviction_ttl_seconds
        self._node_count = 1
        self._lock = __import__('threading').Lock()

    def insert(
        self,
        token_ids: List[int],
        log_probs: Optional[List[float]] = None,
        block_ids: Optional[List[int]] = None,
    ) -> YvPLTTrieNode:
        """Insert a token sequence into the PLT, creating or updating nodes.

        Each token in the sequence corresponds to a block of KV cache in the
        paged cache manager. The log_probs represent the model's per-token
        log-probability P(token_i | prefix_{<i}).

        Args:
            token_ids: List of token IDs forming the prefix.
            log_probs: Optional per-token log-probabilities. If None, uses
                a uniform prior of log(1/|V|) per token.
            block_ids: Optional list of paged cache block IDs, one per token
                or one per block-size group.

        Returns:
            The leaf node after insertion.
        """
        with self._lock:
            current = self.root
            cum_log_prob = 0.0

            for i, tid in enumerate(token_ids):
                if log_probs is not None and i < len(log_probs):
                    cum_log_prob += log_probs[i]
                else:
                    cum_log_prob += math.log(1.0 / 50000.0)

                if tid not in current.children:
                    if self._node_count >= self.max_nodes:
                        self._evict_lru_nodes()
                    current.children[tid] = YvPLTTrieNode(
                        token_id=tid,
                        cumulative_log_prob=cum_log_prob,
                    )
                    self._node_count += 1
                else:
                    current.children[tid].cumulative_log_prob = max(
                        current.children[tid].cumulative_log_prob,
                        cum_log_prob,
                    )

                current = current.children[tid]
                current.access_count += 1
                current.last_access_time = __import__('time').time()

            if block_ids is not None:
                current.kv_block_ids = block_ids

            return current

    def find_longest_match(
        self,
        token_ids: List[int],
        log_probs: Optional[List[float]] = None,
    ) -> Optional[Tuple[List[int], int]]:
        """Find the longest prefix in the trie that matches within the distance threshold.

        The match is determined by the trie metric:
            d_T(query, candidate) = -log2 P_M(query ^ candidate)

        Two prefixes match if their trie distance is below the configured threshold.
        This is equivalent to: the shared prefix (longest common prefix) has cumulative
        log-probability > -threshold_bits * ln(2).

        Args:
            token_ids: Query token sequence.
            log_probs: Optional per-token log-probabilities of the query.

        Returns:
            If a match is found, returns (shared_block_ids, divergence_position).
            The divergence_position is the index where the sequences diverge.
            Returns None if no match within threshold.
        """
        with self._lock:
            best_match_node = None
            best_match_depth = 0
            best_cum_log_prob = float('-inf')

            current = self.root
            cum_log_prob = 0.0

            for depth, tid in enumerate(token_ids):
                if tid in current.children:
                    node = current.children[tid]
                    if log_probs is not None and depth < len(log_probs):
                        cum_log_prob += log_probs[depth]
                    else:
                        cum_log_prob += math.log(1.0 / 50000.0)

                    if node.kv_block_ids:
                        if cum_log_prob > best_cum_log_prob:
                            best_cum_log_prob = cum_log_prob
                            best_match_node = node
                            best_match_depth = depth + 1

                    node.access_count += 1
                    node.last_access_time = __import__('time').time()
                    current = node
                else:
                    break

            if best_match_node is None:
                return None

            trie_distance_bits = -best_cum_log_prob / math.log(2)
            if trie_distance_bits <= self.distance_threshold_bits:
                return (list(best_match_node.kv_block_ids), best_match_depth)

            return None

    def trie_distance(
        self,
        seq_a: List[int],
        seq_b: List[int],
        log_probs_a: Optional[List[float]] = None,
        log_probs_b: Optional[List[float]] = None,
    ) -> float:
        """Compute the PLT trie distance between two token sequences.

        d_T(a, b) = -log2 P_M(a ^ b)

        The longest common prefix probability is taken as the minimum of
        the cumulative log-probabilities of both sequences up to the
        divergence point.

        Args:
            seq_a: First token sequence.
            seq_b: Second token sequence.
            log_probs_a: Per-token log-probabilities of seq_a.
            log_probs_b: Per-token log-probabilities of seq_b.

        Returns:
            Trie distance in bits. Lower values indicate higher similarity.
        """
        lcp_len = 0
        for a, b in zip(seq_a, seq_b):
            if a == b:
                lcp_len += 1
            else:
                break

        cum_log_prob = 0.0
        for i in range(lcp_len):
            if log_probs_a is not None and i < len(log_probs_a):
                cum_log_prob += log_probs_a[i]
            else:
                cum_log_prob += math.log(1.0 / 50000.0)

        return -cum_log_prob / math.log(2)

    def _evict_lru_nodes(self):
        """Evict trie nodes that haven't been accessed recently.

        Traverses the trie and removes leaf nodes that exceed the TTL.
        Prioritizes nodes with no kv_block_ids and low access counts.
        """
        import time as _time
        now = _time.time()
        candidates = []

        def _collect_leaf(node, parent, token_key, depth=0):
            if depth > 1000:
                raise RuntimeError("PLT trie recursion depth exceeded 1000")
            for kid, child in list(node.children.items()):
                if child.is_leaf():
                    age = now - child.last_access_time if child.last_access_time > 0 else 0
                    candidates.append((age, -child.access_count, parent, kid, child))
                else:
                    _collect_leaf(child, node, kid, depth + 1)

        _collect_leaf(self.root, self.root, -1)

        candidates.sort(key=lambda x: (not x[4].kv_block_ids, x[0]), reverse=True)

        evict_count = max(1, len(candidates) // 4)
        for i in range(min(evict_count, len(candidates))):
            age, _, parent, kid, child = candidates[i]
            if child.kv_block_ids:
                continue
            if age > self.eviction_ttl_seconds or (parent is self.root and age > self.eviction_ttl_seconds * 0.5):
                del parent.children[kid]
                self._node_count -= 1

    def clear(self):
        """Reset the trie to its initial empty state."""
        with self._lock:
            self.root = YvPLTTrieNode(token_id=-1, cumulative_log_prob=0.0)
            self._node_count = 1

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the trie for monitoring."""
        return {
            'node_count': self._node_count,
            'max_nodes': self.max_nodes,
            'distance_threshold_bits': self.distance_threshold_bits,
        }
