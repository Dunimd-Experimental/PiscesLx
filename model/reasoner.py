#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PiscesReasoner(nn.Module):
    """
    Pisces Reasoner Module.
    
    Features:
    1. Controllable Chain-of-Thought (CoT) generation.
    2. Self-reflection mechanism for error correction.
    3. Dynamic thinking budget based on predicted difficulty.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size

        # Heads for different reasoning tasks
        self.thinking_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.difficulty_head = nn.Linear(self.hidden_size, 3)  # easy, medium, hard
        self.reflection_head = nn.Linear(self.hidden_size, 2)  # correct, incorrect

        # Special token IDs, to be set after tokenizer is updated
        self.start_thinking_id = None
        self.end_thinking_id = None

    def resize_vocab(self, new_vocab_size):
        """Resizes the thinking_head to match the new vocabulary size."""
        old_head = self.thinking_head
        new_head = nn.Linear(self.hidden_size, new_vocab_size, bias=False, device=old_head.weight.device, dtype=old_head.weight.dtype)

        num_to_copy = min(old_head.out_features, new_vocab_size)
        new_head.weight.data[:num_to_copy, :] = old_head.weight.data[:num_to_copy, :]
        
        self.thinking_head = new_head
        self.vocab_size = new_vocab_size

    def forward(self, hidden_states, input_ids=None, labels=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] from the base model.
            input_ids: [batch, seq_len]
            labels: [batch, seq_len]
        """
        if self.start_thinking_id is None or self.end_thinking_id is None:
            # Not yet configured for reasoning, return no-op
            return {"loss": torch.tensor(0.0, device=hidden_states.device, requires_grad=True)}

        batch_size, seq_len, _ = hidden_states.shape

        # 1. Predict difficulty (e.g., for dynamic budget)
        # Using hidden state of the first token (e.g., [CLS])
        difficulty_logits = self.difficulty_head(hidden_states[:, 0])

        # 2. Generate thinking chain
        thinking_logits = self.thinking_head(hidden_states)

        # 3. Reflect on the answer (self-correction)
        # Using hidden state of the last token
        reflection_logits = self.reflection_head(hidden_states[:, -1])

        loss = None
        if labels is not None:
            # Calculate loss only for the tokens within the thinking block
            # Create a mask for tokens between <|start_thinking|> and <|end_thinking|>
            start_mask = (input_ids == self.start_thinking_id).cumsum(dim=1) > 0
            end_mask = (input_ids == self.end_thinking_id).cumsum(dim=1) == 0
            cot_mask = start_mask & end_mask
            cot_mask |= (input_ids == self.end_thinking_id)
            
            if cot_mask.any():
                thinking_loss = F.cross_entropy(thinking_logits[cot_mask], labels[cot_mask])
            else:
                thinking_loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
                
            # A simple heuristic for reflection loss, assuming the last two tokens
            # determine correctness. This should be refined based on data format.
            reflection_labels = (labels[:, -1] == labels[:, -2]).long()
            reflection_loss = F.cross_entropy(reflection_logits, reflection_labels)
            
            loss = thinking_loss + reflection_loss

        return {
            "thinking_logits": thinking_logits,
            "difficulty_logits": difficulty_logits,
            "reflection_logits": reflection_logits,
            "loss": loss
        }


class TreeSearchReasoner:
    """
    Lightweight tree search for inference using self-consistency.
    Generates multiple reasoning paths and selects the best one by voting.
    """
    def __init__(self, model, tokenizer, num_samples=5, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

    @torch.no_grad()
    def generate(self, prompt):
        """Generate multiple answers and return the most consistent one."""
        answers = []
        for _ in range(self.num_samples):
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            
            outputs = self.model.generate(
                inputs,
                max_length=self.max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answers.append(answer)
        
        # Simple majority vote
        if not answers:
            return ""
        return max(set(answers), key=answers.count) 