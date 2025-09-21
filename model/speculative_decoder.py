#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from utils import DEBUG

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding"""
    num_candidates: int = 4  # Number of candidate tokens to generate in parallel
    draft_length: int = 5    # Length of draft sequence
    acceptance_threshold: float = 0.8  # Threshold for accepting draft tokens
    temperature: float = 0.7   # Temperature for sampling
    top_k: int = 50           # Top-k sampling
    top_p: float = 0.9        # Top-p sampling

class SpeculativeDecoder(nn.Module):
    """Multi-path speculative decoder for efficient autoregressive generation"""
    
    def __init__(self, config: SpeculativeConfig, model: nn.Module, tokenizer=None, on_stats: Optional[Any]=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        # Optional stats callback: callable(dict) -> None
        self.on_stats = on_stats
        
        # Create lightweight draft model (smaller version of main model)
        self.draft_model = self._create_draft_model()
        
        # Verification head for parallel token validation
        self.verification_head = nn.Linear(model.config.hidden_size, config.num_candidates)
        
    def _create_draft_model(self) -> nn.Module:
        """Create a lightweight draft model for fast token generation.
        Avoids tight coupling to self.model.config class shape.
        """
        vocab_size = getattr(self.model.config, 'vocab_size', 65536)
        base_hidden = getattr(self.model.config, 'hidden_size', 2048)
        base_layers = getattr(self.model.config, 'num_layers', getattr(self.model.config, 'n_layer', 24))
        base_heads = getattr(self.model.config, 'num_heads', getattr(self.model.config, 'n_head', 16))

        hidden_size = max(512, base_hidden // 2)
        num_layers = max(2, base_layers // 4)
        nhead = max(4, min(8, base_heads // 2))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # A minimal language head
        draft = nn.ModuleDict({
            'embed': nn.Embedding(vocab_size, hidden_size),
            'encoder': encoder,
            'lm': nn.Linear(hidden_size, vocab_size)
        })

        def forward_fn(input_ids: torch.Tensor):
            x = draft['embed'](input_ids)
            x = draft['encoder'](x)
            logits = draft['lm'](x)
            return logits

        draft.forward = forward_fn  # type: ignore
        return draft
        
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        cache_manager=None,
        **model_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate tokens using multi-path speculative decoding
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            max_length: Maximum generation length
            **model_kwargs: Additional model arguments
            
        Returns:
            Generated token IDs and generation statistics
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated_ids = input_ids.clone()
        stats = {
            'method': 'speculative',
            'total_draft_tokens': 0,
            'accepted_tokens': 0,
            'rejected_tokens': 0,
            'draft_acceptance_rate': 0.0,
            'speedup': 1.0,
            'iter_accept': [],
            'total_time_ms': 0.0
        }
        _t0 = time.time()
        
        # Generation loop
        while generated_ids.shape[1] < max_length:
            # Optional: simple cache by draft length
            if cache_manager is not None:
                cached = cache_manager.get_speculative_cache(self.config.draft_length)
                if cached is not None:
                    return cached, {'from_cache': True}
            
            # Step 1: Generate draft sequence with draft model
            draft_ids, draft_logits = self._generate_draft_sequence(
                generated_ids, attention_mask, cache_manager=cache_manager, **model_kwargs
            )
            
            # Step 2: Parallel verification of draft tokens
            accepted_ids, num_accepted = self._verify_draft_tokens(
                generated_ids, draft_ids, draft_logits, attention_mask, **model_kwargs
            )
            
            # Step 3: Update generated sequence
            generated_ids = torch.cat([generated_ids, accepted_ids], dim=1)
            # Update attention mask if provided
            if attention_mask is None:
                attention_mask = torch.ones_like(generated_ids, dtype=torch.long, device=device)
            else:
                add_len = accepted_ids.shape[1]
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], add_len), device=device, dtype=attention_mask.dtype)
                ], dim=1)
            
            # Update statistics
            stats['total_draft_tokens'] += draft_ids.shape[1]
            stats['accepted_tokens'] += num_accepted
            stats['rejected_tokens'] += (draft_ids.shape[1] - num_accepted)
            stats['iter_accept'].append(int(num_accepted))
            
            # Early backoff: if repeated zero-accept, do a single-token fallback step then resume
            if num_accepted == 0:
                # Accumulate streak in local state on stats (ephemeral)
                streak = stats.get('_zero_accept_streak', 0) + 1
                stats['_zero_accept_streak'] = streak
                if streak >= 2:
                    # Fallback to main model single token step
                    with torch.no_grad():
                        outputs = self.model(generated_ids, attention_mask=attention_mask, **model_kwargs)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        next_logits = logits[:, -1, :]
                        # Apply temperature/top-k/top-p from current config
                        if self.config.temperature > 0:
                            next_logits = next_logits / self.config.temperature
                        if self.config.top_k > 0:
                            top_k_logits, top_k_indices = torch.topk(next_logits, min(self.config.top_k, next_logits.size(-1)))
                            next_logits = torch.full_like(next_logits, float('-inf'))
                            next_logits.scatter_(-1, top_k_indices, top_k_logits)
                        if self.config.top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > self.config.top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                            next_logits[indices_to_remove] = float('-inf')
                        probs = F.softmax(next_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        generated_ids = torch.cat([generated_ids, next_token], dim=1)
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)
                        ], dim=1)
                    # Reset streak and continue speculative
                    stats['_zero_accept_streak'] = 0
                    continue
                else:
                    # Give speculative one more chance in next iteration
                    continue
                
        # Calculate final statistics
        if stats['total_draft_tokens'] > 0:
            stats['draft_acceptance_rate'] = stats['accepted_tokens'] / stats['total_draft_tokens']
            stats['speedup'] = 1.0 + (stats['accepted_tokens'] / max(1, stats['rejected_tokens']))
        stats['total_time_ms'] = (time.time() - _t0) * 1000.0
        
        # Cache the result if cache_manager is not None:
        if cache_manager is not None:
            cache_manager.set_speculative_cache(self.config.draft_length, generated_ids)
        
        # Telemetry: log summary and optionally call user callback
        try:
            DEBUG(f"[SpecDecode] draft_len={self.config.draft_length}, candidates={self.config.num_candidates}, "
                  f"accept_rate={stats['draft_acceptance_rate']:.3f}, speedup={stats['speedup']:.2f}, "
                  f"time_ms={stats['total_time_ms']:.1f}, iters={len(stats['iter_accept'])}")
        except Exception:
            pass
        if self.on_stats is not None:
            try:
                self.on_stats(stats)
            except Exception:
                pass

        return generated_ids, stats
        
    def _generate_draft_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_manager=None,
        **model_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a multi-step draft sequence using the draft model.
        Returns (draft_seq, draft_step_logits) where
        - draft_seq: [B, L]
        - draft_step_logits: [B, L, V]
        """
        bsz = input_ids.shape[0]
        device = input_ids.device
        draft_len = max(1, self.config.draft_length)
        cur_ids = input_ids
        draft_tokens: List[torch.Tensor] = []
        step_logits_list: List[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(draft_len):
                logits = self.draft_model(cur_ids)  # [B, T, V]
                step_logits = logits[:, -1:, :]    # [B, 1, V]
                # sample one token per step under current config
                step_token = self._sample_candidates(step_logits).unsqueeze(1)[:, 0:1]
                draft_tokens.append(step_token)
                cur_ids = torch.cat([cur_ids, step_token], dim=1)
                step_logits_list.append(step_logits)  # keep per-step logits
        draft_seq = torch.cat(draft_tokens, dim=1).to(device)  # [B, L]
        draft_step_logits = torch.cat(step_logits_list, dim=1) if step_logits_list else torch.zeros(bsz, 0, getattr(self.model.config, 'vocab_size', 65536), device=device)
        return draft_seq, draft_step_logits
            
    def _sample_candidates(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample multiple candidate tokens from logits"""
        batch_size = logits.shape[0]
        
        # Apply temperature
        if self.config.temperature > 0:
            logits = logits / self.config.temperature
            
        # Apply top-k filtering
        if self.config.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
            logits = torch.full_like(logits, float('-inf'))
            logits.scatter_(-1, top_k_indices, top_k_logits)
            
        # Apply top-p filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
        # Convert to probabilities and sample
        probs = F.softmax(logits, dim=-1)
        
        # Sample multiple candidates
        candidates = torch.multinomial(probs.squeeze(1), self.config.num_candidates, replacement=False)
        
        return candidates
        
    def _verify_draft_tokens(
        self,
        input_ids: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ) -> Tuple[torch.Tensor, int]:
        """Verify draft sequence via a single main-model forward and accept run until rejection.
        draft_logits is expected to be per-step logits of shape [B, L, V]."""
        device = input_ids.device
        bsz = input_ids.shape[0]
        assert bsz == 1, "Current speculative verifier expects batch_size=1"

        L = draft_ids.shape[1]
        if L == 0:
            return draft_ids[:, :0], 0

        with torch.no_grad():
            # Base forward on current prefix with KV-Cache enabled
            base_out = self.model(input_ids, attention_mask=None, use_cache=True, **model_kwargs)
            logits_base = base_out['logits'] if isinstance(base_out, dict) else base_out
            past_kv = base_out.get('past_key_values', None) if isinstance(base_out, dict) else None

            # Prepare draft per-step probabilities if available
            if draft_logits is not None and draft_logits.ndim == 3 and draft_logits.shape[1] == L:
                draft_probs_steps = F.softmax(draft_logits, dim=-1)
            else:
                draft_probs_steps = None

            num_accept = 0
            cur_ids = input_ids
            for i in range(L):
                token_i = draft_ids[:, i:i+1]  # [B,1]
                # Incremental step: feed only the new token with past KV
                step_out = self.model(token_i, attention_mask=None, use_cache=True, past_key_values=past_kv, **model_kwargs)
                step_logits = step_out['logits'] if isinstance(step_out, dict) else step_out
                past_kv = step_out.get('past_key_values', None) if isinstance(step_out, dict) else None
                step_prob = F.softmax(step_logits[:, -1, :], dim=-1)
                p_main = step_prob[0, token_i.item()]
                if draft_probs_steps is not None:
                    p_draft = draft_probs_steps[0, i, token_i.item()]
                else:
                    vocab = step_prob.shape[-1]
                    p_draft = step_prob.new_tensor(1.0 / vocab)
                accept_p = min(1.0, float((p_main + 1e-8) / (p_draft + 1e-8)))
                if accept_p >= self.config.acceptance_threshold:
                    num_accept += 1
                else:
                    break

        if num_accept > 0:
            return draft_ids[:, :num_accept], num_accept

        # Fallback: sample one token from main model at current prefix
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=None, **model_kwargs)
            next_logits = outputs['logits'][:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
        return token, 1
                
    def parallel_decode_step(
        self,
        input_ids: torch.Tensor,
        candidate_tokens: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Perform parallel decoding step with multiple candidate verification
        
        Args:
            input_ids: Current input sequence
            candidate_tokens: List of candidate token sequences
            attention_mask: Attention mask
            **model_kwargs: Additional model arguments
            
        Returns:
            Best accepted token sequence
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Prepare all candidate sequences
        all_candidates = []
        for candidates in candidate_tokens:
            candidate_seq = torch.cat([input_ids, candidates], dim=1)
            all_candidates.append(candidate_seq)
            
        # Batch all candidates for parallel processing
        candidate_batch = torch.cat(all_candidates, dim=0)
        
        with torch.no_grad():
            # Process all candidates in parallel
            outputs = self.model(candidate_batch, attention_mask=None, **model_kwargs)
            logits = outputs['logits']
            
            # Score each candidate sequence
            candidate_scores = []
            for i in range(len(candidate_tokens)):
                candidate_logits = logits[i * batch_size:(i + 1) * batch_size]
                # Calculate sequence score (average log probability)
                log_probs = F.log_softmax(candidate_logits, dim=-1)
                scores = log_probs.mean(dim=-1).mean(dim=-1)  # Average over sequence and vocab
                candidate_scores.append(scores)
                
            # Select best candidate
            best_idx = torch.argmax(torch.stack(candidate_scores))
            best_candidates = candidate_tokens[best_idx]
            
            return best_candidates

class AdaptiveSpeculativeDecoder(SpeculativeDecoder):
    """Adaptive speculative decoder that adjusts parameters based on performance"""
    
    def __init__(self, config: SpeculativeConfig, model: nn.Module, tokenizer=None):
        super().__init__(config, model, tokenizer)
        self.performance_history = []
        self.adaptation_interval = 10  # Adapt every N generations
        
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        **model_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate with adaptive parameter adjustment"""
        # Adapt parameters based on performance history
        if len(self.performance_history) >= self.adaptation_interval:
            self._adapt_parameters()
            
        # Generate with current parameters
        generated_ids, stats = super().speculative_generate(
            input_ids, attention_mask, max_length, **model_kwargs
        )
        
        # Record performance
        self.performance_history.append({
            'acceptance_rate': stats['draft_acceptance_rate'],
            'speedup': stats['speedup'],
            'num_candidates': self.config.num_candidates,
            'draft_length': self.config.draft_length
        })
        
        return generated_ids, stats
        
    def _adapt_parameters(self):
        """Adapt speculative decoding parameters based on performance history"""
        if len(self.performance_history) < self.adaptation_interval:
            return
            
        # Calculate recent performance metrics
        recent_history = self.performance_history[-self.adaptation_interval:]
        avg_acceptance_rate = sum(h['acceptance_rate'] for h in recent_history) / len(recent_history)
        avg_speedup = sum(h['speedup'] for h in recent_history) / len(recent_history)
        
        # Adapt number of candidates
        if avg_acceptance_rate > 0.8 and avg_speedup < 2.0:
            # High acceptance but low speedup - increase candidates
            self.config.num_candidates = min(8, self.config.num_candidates + 1)
        elif avg_acceptance_rate < 0.5:
            # Low acceptance rate - reduce candidates
            self.config.num_candidates = max(2, self.config.num_candidates - 1)
            
        # Adapt draft length
        if avg_acceptance_rate > 0.9:
            # Very high acceptance - increase draft length
            self.config.draft_length = min(10, self.config.draft_length + 1)
        elif avg_acceptance_rate < 0.6:
            # Low acceptance - reduce draft length
            self.config.draft_length = max(2, self.config.draft_length - 1)

        # Adapt sampling temperature and top_p for better verification match
        # Heuristic: if acceptance低，降低温度/减小top_p；若接受很高且速度不高，稍增温度以探索更多候选
        if avg_acceptance_rate < 0.5:
            self.config.temperature = max(0.5, round(self.config.temperature * 0.9, 2))
            self.config.top_p = max(0.7, round(self.config.top_p - 0.05, 2))
        elif avg_acceptance_rate > 0.85 and avg_speedup < 1.8:
            self.config.temperature = min(1.2, round(self.config.temperature * 1.05, 2))
            self.config.top_p = min(0.98, round(self.config.top_p + 0.02, 2))
        
        # Clear old history
        self.performance_history = self.performance_history[-self.adaptation_interval//2:]