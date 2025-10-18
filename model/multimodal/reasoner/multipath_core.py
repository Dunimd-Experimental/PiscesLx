#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcticMultiPathReasoningEngine(nn.Module):
    """
    A PyTorch module implementing the Arctic Multi-Path Reasoning Engine.
    This engine is designed to perform multi-modal reasoning with dynamic path selection and abstraction.
    """
    def __init__(self, cfg):
        """
        Initialize the ArcticMultiPathReasoningEngine.

        Args:
            cfg: Configuration object containing necessary parameters like hidden_size, vocab_size, etc.
        """
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.reasoning_heads = 8

        # Transformer encoder layers for abstraction
        self.abstraction_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=cfg.n_head,
                dim_feedforward=self.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(3)
        ])

        # Controller to determine reasoning depth
        self.depth_controller = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # Multi-head attention for multi-path reasoning
        self.multi_path_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.reasoning_heads,
            dropout=0.1,
            batch_first=True
        )

        # Controller for pruning reasoning paths
        self.path_pruning_controller = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, self.reasoning_heads),
            nn.Sigmoid()
        )

        # Components for chunked attention with LSH
        self.chunked_attention = nn.ModuleDict({
            'lsh_proj': nn.Linear(self.hidden_size, self.hidden_size // 4),
            'chunk_q': nn.Linear(self.hidden_size, self.hidden_size),
            'chunk_k': nn.Linear(self.hidden_size, self.hidden_size),
            'chunk_v': nn.Linear(self.hidden_size, self.hidden_size),
            'output_proj': nn.Linear(self.hidden_size, self.hidden_size),
            'layer_norm': nn.LayerNorm(self.hidden_size)
        })

        # Linear attention module
        self.linear_attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        # Module to verify the facts
        self.fact_verifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        # Meta-cognitive module using GRU
        self.meta_cognitive = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Head to estimate uncertainty
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Different reasoning streams
        self.reasoning_streams = nn.ModuleDict({
            'hypothesis': nn.Linear(self.hidden_size, self.vocab_size),
            'evidence': nn.Linear(self.hidden_size, self.vocab_size),
            'conclusion': nn.Linear(self.hidden_size, self.vocab_size),
            'reflection': nn.Linear(self.hidden_size, 3)
        })

        # Final thinking head
        self.thinking_head = nn.Linear(self.hidden_size, self.vocab_size)

        # Tokens for reasoning process
        self.reasoning_tokens = {
            'start_hypothesis': None,
            'start_evidence': None,
            'start_conclusion': None,
            'hypothesis_split': None,
            'hypothesis_merge': None
        }

    def initialize_reasoning_tokens(self, tokenizer):
        """
        Initialize reasoning tokens using the provided tokenizer.
        If tokenizer fails or is None, set default reasoning tokens.

        Args:
            tokenizer: Tokenizer object with convert_tokens_to_ids method.
        """
        if tokenizer is not None:
            try:
                self.reasoning_tokens = {
                    'start_hypothesis': tokenizer.convert_tokens_to_ids('<|start_hypothesis|>'),
                    'start_evidence': tokenizer.convert_tokens_to_ids('<|start_evidence|>'),
                    'start_conclusion': tokenizer.convert_tokens_to_ids('<|start_conclusion|>'),
                    'hypothesis_split': tokenizer.convert_tokens_to_ids('<|hypothesis_split|>'),
                    'hypothesis_merge': tokenizer.convert_tokens_to_ids('<|hypothesis_merge|>')
                }
            except Exception:
                self._set_default_reasoning_tokens()
        else:
            self._set_default_reasoning_tokens()

    def _set_default_reasoning_tokens(self):
        """
        Set default reasoning tokens based on the vocabulary size.
        """
        self.reasoning_tokens = {
            'start_hypothesis': self.vocab_size - 5,
            'start_evidence': self.vocab_size - 4,
            'start_conclusion': self.vocab_size - 3,
            'hypothesis_split': self.vocab_size - 2,
            'hypothesis_merge': self.vocab_size - 1
        }

    def resize_vocab(self, new_vocab_size):
        """
        Resize the vocabulary size and update the thinking head accordingly.

        Args:
            new_vocab_size (int): The new vocabulary size.
        """
        old_head = self.thinking_head
        new_head = nn.Linear(self.hidden_size, new_vocab_size, bias=False, device=old_head.weight.device, dtype=old_head.weight.dtype)
        num_to_copy = min(old_head.out_features, new_vocab_size)
        new_head.weight.data[:num_to_copy, :] = old_head.weight.data[:num_to_copy, :]
        self.thinking_head = new_head
        self.vocab_size = new_vocab_size

    def forward(self, hidden_states, input_ids=None, labels=None):
        """
        Forward pass of the ArcticMultiPathReasoningEngine.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape [batch_size, seq_len, hidden_size].
            input_ids (torch.Tensor, optional): Input token IDs of shape [batch_size, seq_len]. Defaults to None.
            labels (torch.Tensor, optional): Ground truth labels of shape [batch_size, seq_len]. Defaults to None.

        Returns:
            dict: A dictionary containing thinking logits, loss, uncertainty scores, fact consistency,
                  reasoning outputs, and reflection logits.
        """
        if any(v is None for v in self.reasoning_tokens.values()):
            self.initialize_reasoning_tokens(None)

        device = hidden_states.device
        batch_size, seq_len, _ = hidden_states.shape
        enable_full_reasoning = getattr(self.cfg, 'enable_dynamic_fusion', True)
        use_linear_attention = seq_len > 512

        if enable_full_reasoning:
            with torch.amp.autocast('cuda'):
                abstract_states = []
                current_states = hidden_states
                complexity_score = self._calculate_problem_complexity(hidden_states)
                base_num_layers = len(self.abstraction_layers)

                # Determine the number of abstraction layers based on complexity and sequence length
                if complexity_score < 0.3 and seq_len <= 256:
                    num_layers = 1
                elif complexity_score < 0.6 and seq_len <= 512:
                    num_layers = 2
                elif seq_len > 1024:
                    num_layers = min(2, base_num_layers)
                else:
                    num_layers = base_num_layers

                # Apply abstraction layers
                for i, layer in enumerate(self.abstraction_layers[:num_layers]):
                    current_states = layer(current_states)
                    abstract_states.append(current_states)
                    if i > 0:
                        abstraction_gain = self._calculate_abstraction_gain(abstract_states[-2], abstract_states[-1])
                        if abstraction_gain < 0.1:
                            break
                        if self._check_convergence(abstract_states[-2], abstract_states[-1]):
                            break

                # Apply different attention mechanisms based on sequence length
                if use_linear_attention:
                    if hasattr(self, 'chunked_attention'):
                        q = self.chunked_attention['chunk_q'](current_states)
                        k = self.chunked_attention['chunk_k'](current_states)
                        v = self.chunked_attention['chunk_v'](current_states)
                        multi_path_states = self._lsh_chunked_attention(q, k, v)
                    else:
                        multi_path_states = self.linear_attention(current_states)
                else:
                    raw_multi_path_states, _ = self.multi_path_attention(
                        current_states, current_states, current_states
                    )
                    path_importance = self.path_pruning_controller(
                        current_states.mean(dim=1, keepdim=True)
                    ).squeeze(1)
                    k = max(2, self.reasoning_heads // 2)
                    top_k_values, top_k_indices = torch.topk(path_importance, k, dim=-1)
                    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
                    selected_states = raw_multi_path_states[batch_indices, :, top_k_indices]
                    weights = F.softmax(top_k_values, dim=-1).unsqueeze(-1).unsqueeze(-1)
                    multi_path_states = (selected_states * weights).sum(dim=1)

                # Generate reasoning outputs for each stream
                reasoning_outputs = {}
                for stream_name, head in self.reasoning_streams.items():
                    if stream_name != 'reflection':
                        key_positions = self._get_key_positions(multi_path_states, seq_len)
                        if key_positions is not None:
                            selected_states = multi_path_states[:, key_positions, :]
                            reasoning_outputs[stream_name] = head(selected_states)
                        else:
                            reasoning_outputs[stream_name] = head(multi_path_states)

                # Compute verification scores
                verification_scores = []
                if len(abstract_states) > 0:
                    combined = torch.cat([abstract_states[-1], multi_path_states], dim=-1)
                    score = self.fact_verifier(combined)
                    verification_scores.append(score)

                # Compute meta-cognitive outputs and uncertainty scores
                pooled_states = multi_path_states.mean(dim=1, keepdim=True)
                meta_output, _ = self.meta_cognitive(pooled_states)
                reflection_logits = self.reasoning_streams['reflection'](meta_output.squeeze(1))
                uncertainty_scores = self.uncertainty_head(pooled_states)

                # Collapse reasoning paths
                collapsed_output = self._efficient_path_collapse(
                    reasoning_outputs, uncertainty_scores, verification_scores, input_ids,
                    path_importance if not use_linear_attention else None
                )
                thinking_output = self.thinking_head(collapsed_output)

                # Compute reasoning loss
                loss = self._compute_reasoning_loss(
                    reasoning_outputs, collapsed_output, uncertainty_scores, labels
                )

                # Compute fact consistency
                if 'hypothesis' in reasoning_outputs and 'evidence' in reasoning_outputs:
                    hypothesis_tokens = reasoning_outputs['hypothesis']
                    evidence_tokens = reasoning_outputs['evidence']
                    pooled_hyp = hypothesis_tokens.mean(dim=1, keepdim=True)
                    pooled_evi = evidence_tokens.mean(dim=1, keepdim=True)
                    fact_consistency = self.fact_verifier(torch.cat([pooled_hyp, pooled_evi], dim=-1))
                else:
                    fact_consistency = torch.ones(batch_size, 1, device=device) * 0.8

                # Compute additional loss if labels are provided
                if labels is not None:
                    shift_logits = thinking_output[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.view(-1)
                    valid_mask = (flat_labels != -100)
                    if valid_mask.any():
                        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                        chunk_size = min(1024, valid_indices.size(0))
                        reasoning_losses = []
                        for i in range(0, valid_indices.size(0), chunk_size):
                            end_idx = min(i + chunk_size, valid_indices.size(0))
                            chunk_indices = valid_indices[i:end_idx]
                            if chunk_indices.size(0) > 512:
                                sub_losses = []
                                for j in range(0, chunk_indices.size(0), 512):
                                    sub_end = min(j + 512, chunk_indices.size(0))
                                    sub_indices = chunk_indices[j:sub_end]
                                    sub_loss = F.cross_entropy(
                                        flat_logits[sub_indices],
                                        flat_labels[sub_indices],
                                        reduction='mean'
                                    )
                                    sub_losses.append(sub_loss)
                                    del sub_indices
                                    torch.cuda.empty_cache()
                                chunk_loss = torch.stack(sub_losses).mean()
                                del sub_losses
                            else:
                                chunk_loss = F.cross_entropy(
                                    flat_logits[chunk_indices],
                                    flat_labels[chunk_indices],
                                    reduction='mean'
                                )
                            reasoning_losses.append(chunk_loss)
                            del chunk_indices, chunk_loss
                            torch.cuda.empty_cache()
                        if reasoning_losses:
                            reasoning_loss = torch.stack(reasoning_losses).mean()
                            loss_weight = 0.1 if enable_full_reasoning else 0.02
                            loss = loss + loss_weight * reasoning_loss
                        del reasoning_losses, valid_indices
                        torch.cuda.empty_cache()
                    del flat_logits, flat_labels, valid_mask
                    torch.cuda.empty_cache()

                return {
                    "thinking_logits": thinking_output,
                    "loss": loss,
                    "uncertainty_scores": uncertainty_scores,
                    "fact_consistency": fact_consistency.expand(batch_size, seq_len, 1),
                    "reasoning_outputs": reasoning_outputs,
                    "reflection_logits": reflection_logits
                }

        # Fallback path
        uncertainty_scores = self.uncertainty_head(hidden_states)
        collapsed_output = self._efficient_path_collapse({}, uncertainty_scores, [], input_ids, None)
        reflection_logits = self.reasoning_streams['reflection'](hidden_states.mean(dim=1))
        loss = None
        fact_consistency = torch.ones(batch_size, 1, device=device) * 0.8
        return {
            "thinking_logits": self.thinking_head(collapsed_output),
            "loss": loss,
            "uncertainty_scores": uncertainty_scores,
            "fact_consistency": fact_consistency,
            "reasoning_outputs": {},
            "reflection_logits": reflection_logits
        }

    def _efficient_path_collapse(self, reasoning_outputs, uncertainty_scores, verification_scores, input_ids, path_importance=None):
        """
        Efficiently collapse multiple reasoning paths into a single output.

        Args:
            reasoning_outputs (dict): Dictionary containing reasoning outputs for each stream.
            uncertainty_scores (torch.Tensor): Tensor containing uncertainty scores.
            verification_scores (list): List of verification scores.
            input_ids (torch.Tensor): Input token IDs.
            path_importance (torch.Tensor, optional): Tensor containing path importance scores. Defaults to None.

        Returns:
            torch.Tensor: Collapsed output tensor.
        """
        if path_importance is not None:
            weights = F.softmax(path_importance, dim=-1)
        else:
            weights = 1.0 - uncertainty_scores.mean(dim=1, keepdim=True)
            weights = F.softmax(weights, dim=0)

        output_list = [output for stream_name, output in reasoning_outputs.items() if stream_name != 'reflection']
        if len(output_list) == 0:
            return uncertainty_scores.expand(-1, -1, 1)

        stacked_outputs = torch.stack(output_list, dim=0)
        weighted_outputs = stacked_outputs * weights.unsqueeze(-1).unsqueeze(-1)
        combined_output = weighted_outputs.sum(dim=0)

        if verification_scores:
            verification_confidence = torch.mean(torch.stack(verification_scores), dim=0)
            combined_output = combined_output * verification_confidence.unsqueeze(-1)
        return combined_output

    def _create_reasoning_masks(self, input_ids):
        """
        Create reasoning masks based on input token IDs and reasoning tokens.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len].

        Returns:
            dict: Dictionary containing reasoning masks for hypothesis, evidence, and conclusion.
        """
        if input_ids is None:
            return {}
        masks = {}
        start_hyp = (input_ids == self.reasoning_tokens['start_hypothesis']).cumsum(dim=1) > 0
        end_hyp = (input_ids == self.reasoning_tokens['start_evidence']).cumsum(dim=1) > 0
        masks['hypothesis'] = start_hyp & ~end_hyp
        start_ev = (input_ids == self.reasoning_tokens['start_evidence']).cumsum(dim=1) > 0
        end_ev = (input_ids == self.reasoning_tokens['start_conclusion']).cumsum(dim=1) > 0
        masks['evidence'] = start_ev & ~end_ev
        start_con = (input_ids == self.reasoning_tokens['start_conclusion']).cumsum(dim=1) > 0
        end_con = (input_ids == self.reasoning_tokens['hypothesis_merge']).cumsum(dim=1) > 0
        masks['conclusion'] = start_con & ~end_con
        return masks

    def _calculate_problem_complexity(self, hidden_states):
        """
        Calculate the complexity of the input problem based on sequence length, semantic variance, and information density.

        Args:
            hidden_states (torch.Tensor): Input hidden states of shape [batch_size, seq_len, hidden_size].

        Returns:
            float: Problem complexity score.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        length_complexity = min(seq_len / 512, 1.0)
        mean_state = hidden_states.mean(dim=1, keepdim=True)
        semantic_variance = torch.var(hidden_states - mean_state, dim=[1,2]).mean()
        diversity_complexity = torch.sigmoid(semantic_variance * 10)
        state_norms = torch.norm(hidden_states, dim=-1)
        normalized_norms = state_norms / (state_norms.max(dim=1, keepdim=True)[0] + 1e-8)
        information_density = -torch.sum(normalized_norms * torch.log(normalized_norms + 1e-8), dim=1).mean() / torch.log(torch.tensor(seq_len))
        complexity = (length_complexity * 0.3 + diversity_complexity * 0.4 + information_density * 0.3)
        return complexity.item()

    def _calculate_abstraction_gain(self, prev_states, curr_states):
        """
        Calculate the gain in abstraction between two consecutive states.

        Args:
            prev_states (torch.Tensor): Previous abstraction states.
            curr_states (torch.Tensor): Current abstraction states.

        Returns:
            float: Abstraction gain score.
        """
        prev_pooled = F.normalize(prev_states.mean(dim=1), p=2, dim=-1)
        curr_pooled = F.normalize(curr_states.mean(dim=1), p=2, dim=-1)
        semantic_shift = 1 - F.cosine_similarity(prev_pooled, curr_pooled, dim=-1).mean()
        prev_variance = torch.var(prev_states, dim=[1,2]).mean()
        curr_variance = torch.var(curr_states, dim=[1,2]).mean()
        compression_ratio = abs(curr_variance - prev_variance) / (prev_variance + 1e-8)
        abstraction_gain = semantic_shift * 0.7 + compression_ratio * 0.3
        return abstraction_gain.item()

    def _check_convergence(self, prev_states, curr_states, threshold=0.001):
        """
        Check if the abstraction process has converged by comparing two consecutive states.

        Args:
            prev_states (torch.Tensor): Previous abstraction states.
            curr_states (torch.Tensor): Current abstraction states.
            threshold (float, optional): Convergence threshold. Defaults to 0.001.

        Returns:
            bool: True if the process has converged, False otherwise.
        """
        diff = torch.abs(prev_states - curr_states).mean()
        return diff < threshold

    def _get_key_positions(self, states, seq_len, ratio=0.3):
        """
        Get key positions from the input sequence based on the sequence length and ratio.

        Args:
            states (torch.Tensor): Input states.
            seq_len (int): Sequence length.
            ratio (float, optional): Ratio to determine the number of key positions. Defaults to 0.3.

        Returns:
            torch.Tensor or None: Tensor containing key positions if sequence length is greater than 128, None otherwise.
        """
        if seq_len <= 128:
            return None
        num_key_positions = max(32, int(seq_len * ratio))
        step = seq_len // num_key_positions
        key_positions = torch.arange(0, seq_len, step, device=states.device)
        if len(key_positions) > num_key_positions:
            key_positions = key_positions[:num_key_positions]
        elif len(key_positions) < num_key_positions:
            additional = torch.arange(seq_len - (num_key_positions - len(key_positions)), seq_len, device=states.device)
            key_positions = torch.cat([key_positions, additional])
        return key_positions

    def _lsh_chunked_attention(self, query, key, value, chunk_size=64, num_hashes=8):
        """
        Perform chunked attention with Locality-Sensitive Hashing (LSH).

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.
            value (torch.Tensor): Value tensor.
            chunk_size (int, optional): Size of each chunk. Defaults to 64.
            num_hashes (int, optional): Number of hashes for LSH. Defaults to 8.

        Returns:
            torch.Tensor: Output tensor after chunked attention.
        """
        batch_size, seq_len, hidden_size = query.shape
        device = query.device
        query_proj = self.chunked_attention['lsh_proj'](query)
        key_proj = self.chunked_attention['lsh_proj'](key)
        hash_matrix = torch.randn(num_hashes, hidden_size // 4, device=device)
        query_hashes = torch.sign(torch.matmul(query_proj, hash_matrix.t()))
        key_hashes = torch.sign(torch.matmul(key_proj, hash_matrix.t()))
        query_buckets = torch.sum(query_hashes * (2 ** torch.arange(num_hashes, device=device)), dim=-1)
        key_buckets = torch.sum(key_hashes * (2 ** torch.arange(num_hashes, device=device)), dim=-1)
        _, query_sorted_indices = torch.sort(query_buckets.view(-1))
        _, key_sorted_indices = torch.sort(key_buckets.view(-1))
        output = torch.zeros_like(query)
        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)
            chunk_query = query[:, i:end_idx, :]
            chunk_key = key[:, i:end_idx, :]
            chunk_value = value[:, i:end_idx, :]
            q = self.chunked_attention['chunk_q'](chunk_query)
            k = self.chunked_attention['chunk_k'](chunk_key)
            v = self.chunked_attention['chunk_v'](chunk_value)
            scores = torch.matmul(q, k.transpose(-2, -1)) / (hidden_size ** 0.5)
            attention_weights = F.softmax(scores, dim=-1)
            chunk_output = torch.matmul(attention_weights, v)
            output[:, i:end_idx, :] = self.chunked_attention['output_proj'](chunk_output)
        return self.chunked_attention['layer_norm'](output + query)

    def _compute_reasoning_loss(self, reasoning_outputs, collapsed_output, uncertainty_scores, labels, input_ids=None, fact_consistency=None):
        """
        Compute the multi-objective reasoning loss.

        Args:
            reasoning_outputs (dict): Dictionary containing reasoning outputs for each stream.
            collapsed_output (torch.Tensor): Collapsed output tensor.
            uncertainty_scores (torch.Tensor): Tensor containing uncertainty scores.
            labels (torch.Tensor): Ground truth labels.
            input_ids (torch.Tensor, optional): Input token IDs. Defaults to None.
            fact_consistency (torch.Tensor, optional): Tensor containing fact consistency scores. Defaults to None.

        Returns:
            torch.Tensor: Total reasoning loss.
        """
        device = collapsed_output.device
        total_loss = torch.tensor(0.0, device=device)

        # 1) Sequence CE on collapsed_output
        if labels is not None and collapsed_output is not None:
            if collapsed_output.dim() == 3 and labels.dim() == 2:
                shift_logits = collapsed_output[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='mean'
                )
                total_loss = total_loss + ce_loss

        # 2) Stream-specific CE using masks
        if input_ids is not None and labels is not None and isinstance(reasoning_outputs, dict) and len(reasoning_outputs) > 0:
            masks = self._create_reasoning_masks(input_ids)
            for name in ('hypothesis', 'evidence', 'conclusion'):
                if name in reasoning_outputs and name in masks:
                    mask = masks[name]
                    # Expect stream logits same seq length; select masked positions
                    stream_logits = reasoning_outputs[name]
                    if stream_logits.dim() == 3 and labels.dim() == 2:
                        # Align shapes: [B, T, V] and [B, T]
                        if mask.shape == labels.shape:
                            sel_logits = stream_logits[mask]
                            sel_labels = labels[mask]
                            if sel_logits.numel() > 0 and sel_labels.numel() > 0:
                                stream_ce = F.cross_entropy(
                                    sel_logits,
                                    sel_labels,
                                    ignore_index=-100,
                                    reduction='mean'
                                )
                                total_loss = total_loss + 0.2 * stream_ce  # small weight per stream

        # 3) Uncertainty regularization (encourage lower uncertainty)
        if uncertainty_scores is not None and torch.is_tensor(uncertainty_scores):
            unc_reg = uncertainty_scores.mean()
            total_loss = total_loss + 0.1 * unc_reg

        # 4) Fact consistency towards 1.0
        if fact_consistency is not None and torch.is_tensor(fact_consistency):
            target = torch.ones_like(fact_consistency)
            cons_loss = F.mse_loss(fact_consistency, target)
            total_loss = total_loss + 0.05 * cons_loss

        return total_loss