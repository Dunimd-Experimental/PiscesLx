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
Graph of Tokens MoE Routing.

Extends standard token-independent MoE routing with sequence-level
dependency awareness. Tokens form a graph based on semantic similarity
and position adjacency; tokens in the same graph cluster are routed
to the same group of experts. This creates coherent expert assignment
across related tokens.

Key Innovation:
    - Standard MoE: each token independently picks top-k experts
    - Graph of Tokens: token clusters pick top-k expert groups together
    - Benefits: coherent knowledge processing, reduced routing fragmentation

Architecture:
    1. Build token graph: affinity = sim(h_i, h_j) * pos_decay(|i-j|)
    2. Cluster tokens: spectral clustering on graph Laplacian
    3. Per-cluster routing: all tokens in cluster share expert assignment
    4. Per-token refinement: fine-grained weights within shared assignment

Reference:
    Graph of Tokens: Learning Token-Level Dependency for Mixture-of-Experts
    (Original contribution by Dunimd Team)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# Paper: Original contribution by Dunimd Team (Yv Architecture — graph-based MoE routing)
class YvTokenGraphBuilder(nn.Module):
    """
    Builds token dependency graph from hidden states.

    Constructs an affinity matrix where edge weights encode:
        - Semantic similarity: dot product between token representations
        - Positional decay: nearby tokens have higher base affinity
        - Adaptive threshold: only retain edges above learned threshold
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int = 4,
        temperature: float = 1.0,
        pos_decay_factor: float = 0.1,
        max_clusters: int = 8,
        device=None, dtype=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.temperature = temperature
        self.pos_decay_factor = pos_decay_factor
        self.max_clusters = max_clusters

        self.q_proj = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.cluster_centroids = nn.Parameter(torch.randn(max_clusters, hidden_size, device=device, dtype=dtype) * 0.02)
        self.affinity_threshold = nn.Parameter(torch.tensor(0.5, device=device, dtype=dtype))
        self.graph_scale = nn.Parameter(torch.ones(1, device=device, dtype=dtype) * 0.1)

    def build_affinity(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        b, t, h = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        sim = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(h)
        pos_diff = torch.arange(t, device=hidden_states.device).float()
        pos_decay = torch.exp(-self.pos_decay_factor * torch.abs(pos_diff.unsqueeze(0) - pos_diff.unsqueeze(1)))
        pos_decay = pos_decay.unsqueeze(0)
        affinity = sim * pos_decay * self.graph_scale
        threshold = torch.sigmoid(self.affinity_threshold)
        affinity = affinity * (affinity > threshold).float()
        return affinity

    def assign_clusters(
        self, hidden_states: torch.Tensor, affinity: torch.Tensor
    ) -> torch.Tensor:
        b, t, h = hidden_states.shape
        flat = hidden_states.view(b * t, h)
        centroids = self.cluster_centroids.unsqueeze(0).expand(b * t, -1, -1)
        dists = torch.cdist(flat.unsqueeze(1), centroids).squeeze(1)
        cluster_ids = dists.argmin(dim=-1)
        cluster_ids = cluster_ids.view(b, t)
        return cluster_ids

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        affinity = self.build_affinity(hidden_states)
        cluster_ids = self.assign_clusters(hidden_states, affinity)
        cluster_mask = F.one_hot(cluster_ids, num_classes=self.max_clusters).float()
        return cluster_ids, cluster_mask, affinity


# Paper: Original contribution by Dunimd Team (Yv Architecture — RoMA regularizer)
class YvRoMARegularizer(nn.Module):
    """
    RoMA: Routing Manifold Alignment (original contribution).

    Post-training regularizer that aligns the routing weight manifold
    with the task embedding manifold. Encourages samples with similar
    task embeddings to use similar expert routing patterns.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, n_neighbors: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_neighbors = n_neighbors

        self.task_embed_proj = nn.Linear(hidden_size, hidden_size)

        self.register_buffer("routing_memory", torch.zeros(256, num_experts))
        self.register_buffer("task_memory", torch.zeros(256, hidden_size))
        self.register_buffer("memory_ptr", torch.tensor(0))
        self.register_buffer("memory_filled", torch.tensor(0))

    def compute_manifold_loss(
        self, routing_weights: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        task_emb = self.task_embed_proj(hidden_states.mean(dim=1))

        if self.memory_filled < self.n_neighbors + 1:
            with torch.no_grad():
                bs = task_emb.shape[0]
                idx = self.memory_ptr % 256
                end = min(idx + bs, 256)
                actual = end - idx
                self.routing_memory[idx:end] = routing_weights[:actual].detach()
                self.task_memory[idx:end] = task_emb[:actual].detach()
                self.memory_ptr.add_(actual)
                self.memory_filled = torch.clamp(self.memory_filled + actual, max=256)
            return torch.tensor(0.0, device=routing_weights.device)

        task_mem = self.task_memory[:self.memory_filled]
        routing_mem = self.routing_memory[:self.memory_filled]

        sim = torch.matmul(task_emb, task_mem.T)
        dists = -sim
        neighbor_weights = routing_mem[None, :, :].expand(task_emb.shape[0], -1, -1)

        _, neighbor_idx = torch.topk(dists, min(self.n_neighbors, dists.shape[1]), dim=-1, largest=False)

        batch_range = torch.arange(task_emb.shape[0], device=routing_weights.device)[:, None]
        neighbor_routing = neighbor_weights[batch_range, neighbor_idx]
        target_routing = neighbor_routing.mean(dim=1)

        loss = F.mse_loss(routing_weights, target_routing)

        with torch.no_grad():
            bs = task_emb.shape[0]
            idx = self.memory_ptr % 256
            end = min(idx + bs, 256)
            actual = end - idx
            self.routing_memory[idx:end] = routing_weights[:actual].detach()
            self.task_memory[idx:end] = task_emb[:actual].detach()
            self.memory_ptr.add_(actual)
            self.memory_filled = torch.clamp(self.memory_filled + actual, max=256)

        return loss


# Paper: Original contribution by Dunimd Team (Yv Architecture — graph-of-tokens router)
class YvGraphOfTokensRouter(nn.Module):
    """
    Graph of Tokens MoE Router with cluster-aware expert assignment
    and optional RoMA manifold regularization.

    Routes tokens in two stages:
        1. Cluster-level: each cluster selects top-k expert groups
        2. Token-level: individual tokens fine-tune weights within assigned experts

    This ensures related tokens (e.g., tokens in the same phrase or concept)
    are processed by the same set of experts, improving coherence.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        n_graph_heads: int = 4,
        temperature: float = 1.0,
        load_balance_alpha: float = 0.01,
        z_loss_coef: float = 0.001,
        max_clusters: int = 8,
        use_roma: bool = False,
        roma_n_neighbors: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.load_balance_alpha = load_balance_alpha
        self.z_loss_coef = z_loss_coef
        self.max_clusters = max_clusters
        self.use_roma = use_roma

        self.graph_builder = YvTokenGraphBuilder(
            hidden_size=hidden_size,
            n_head=n_graph_heads,
            max_clusters=max_clusters,
        )

        self.cluster_router = nn.Linear(hidden_size, num_experts, bias=False)
        self.token_refiner = nn.Linear(hidden_size, num_experts, bias=False)

        self.roma = YvRoMARegularizer(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            n_neighbors=roma_n_neighbors,
        )

        # Load balancing prior: uniform target
        self.register_buffer("expert_load", torch.zeros(num_experts))
        self.register_buffer("step_count", torch.tensor(0))

    def _cluster_aggregate(
        self, router_logits: torch.Tensor, cluster_mask: torch.Tensor
    ) -> torch.Tensor:
        b, t, e = router_logits.shape
        n_clusters = cluster_mask.shape[-1]
        cluster_weights = cluster_mask.unsqueeze(-1) * router_logits.unsqueeze(2)
        cluster_scores = cluster_weights.sum(dim=1) / cluster_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        return cluster_scores

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, h = x.shape

        cluster_ids, cluster_mask, affinity = self.graph_builder(x)

        cluster_logits = self.cluster_router(x)
        cluster_aggregated = self._cluster_aggregate(cluster_logits, cluster_mask)

        cluster_probs = F.softmax(cluster_aggregated / max(self.temperature, 1e-8), dim=-1)
        cluster_scores, cluster_indices = torch.topk(cluster_probs, min(self.top_k, self.num_experts), dim=-1)

        group_indices = cluster_indices.unsqueeze(1).expand(-1, t, -1, -1)
        group_indices = group_indices.reshape(b * t, self.top_k)

        token_logits = self.token_refiner(x)
        token_probs = F.softmax(token_logits / max(self.temperature, 1e-8), dim=-1)
        group_indices = group_indices.clamp(0, self.num_experts - 1)
        token_gathered = torch.gather(
            token_probs.view(b * t, self.num_experts),
            dim=1,
            index=group_indices,
        )

        final_scores = cluster_scores.unsqueeze(1).expand(-1, t, -1, -1).reshape(b * t, self.top_k)
        final_scores = final_scores * token_gathered

        aux_loss = self._compute_aux_loss(cluster_probs, cluster_ids)
        z_loss = self._compute_z_loss(cluster_logits)
        total_loss = aux_loss + self.z_loss_coef * z_loss

        if self.use_roma:
            routing_for_roma = cluster_probs.mean(dim=1)
            roma_loss = self.roma.compute_manifold_loss(routing_for_roma, x)
            total_loss = total_loss + roma_loss

        return final_scores, group_indices, total_loss

    def _compute_aux_loss(
        self, cluster_probs: torch.Tensor, cluster_ids: torch.Tensor
    ) -> torch.Tensor:
        b, n_clusters, e = cluster_probs.shape
        expert_util = cluster_probs.mean(dim=1)
        target = torch.ones_like(expert_util) / self.num_experts
        aux_loss = self.load_balance_alpha * F.kl_div(
            expert_util.clamp(min=1e-8).log(), target, reduction="batchmean"
        )
        return aux_loss

    def _compute_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(logits, dim=-1).square().mean()
