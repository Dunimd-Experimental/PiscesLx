#!/usr/bin/env python3

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
        """
        Initialize the PiscesReasoner module.

        Args:
            cfg: Configuration object containing model parameters.
        """
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
        """
        Resizes the thinking_head to match the new vocabulary size.

        Args:
            new_vocab_size (int): The new size of the vocabulary.
        """
        old_head = self.thinking_head
        new_head = nn.Linear(self.hidden_size, new_vocab_size, bias=False, device=old_head.weight.device, dtype=old_head.weight.dtype)

        num_to_copy = min(old_head.out_features, new_vocab_size)
        new_head.weight.data[:num_to_copy, :] = old_head.weight.data[:num_to_copy, :]
        
        self.thinking_head = new_head
        self.vocab_size = new_vocab_size

    def forward(self, hidden_states, input_ids=None, labels=None):
        """
        Forward pass of the PiscesReasoner module.

        Args:
            hidden_states (torch.Tensor): [batch, seq_len, hidden_size] from the base model.
            input_ids (torch.Tensor, optional): [batch, seq_len]. Defaults to None.
            labels (torch.Tensor, optional): [batch, seq_len]. Defaults to None.

        Returns:
            dict: A dictionary containing thinking_logits, difficulty_logits, reflection_logits, and loss.
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
                
            # Reflection labels: use token-level accuracy across all valid tokens
            with torch.no_grad():
                # Get predictions for all valid tokens (non-padding)
                predictions = torch.argmax(thinking_logits, dim=-1)
                # Create mask for valid tokens (non-padding)
                valid_mask = (labels != 0)  # Assuming 0 is padding token
                # Calculate token-level accuracy
                correct_tokens = (predictions == labels) & valid_mask
                token_accuracy = correct_tokens.float().sum(dim=1) / valid_mask.float().sum(dim=1)
                # Convert to binary labels: high accuracy (>0.8) = correct, low = incorrect
                reflection_labels = (token_accuracy > 0.8).long()
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
        """
        Initialize the TreeSearchReasoner.

        Args:
            model: The model used for generation.
            tokenizer: The tokenizer used for encoding and decoding.
            num_samples (int, optional): Number of samples to generate. Defaults to 5.
            max_length (int, optional): Maximum length of the generated output. Defaults to 512.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

    @torch.no_grad()
    def generate(self, prompt):
        """
        Generate multiple answers and return the most consistent one.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The most consistent answer from the generated samples.
        """
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