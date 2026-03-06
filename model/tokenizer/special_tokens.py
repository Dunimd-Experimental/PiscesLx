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

"""Special tokens definition for Yv tokenizer with comprehensive categorization.

This module defines all special tokens used across the Yv model architecture,
organized into logical categories for standard operations, multimodal processing,
reasoning tasks, agent interactions, and control flow. It provides efficient
token-to-ID mappings and serialization support.

Architecture Overview:
    The special tokens module consists of two main components:
    
    1. **YvSpecialTokenType** (Enum):
       - Categorizes tokens into five types: STANDARD, MULTIMODAL, REASONING,
         AGENT, and CONTROL
       - Used for token type queries and filtering
    
    2. **YvSpecialTokens** (Dataclass):
       - Defines all special token strings with sensible defaults
       - Maintains bidirectional token-to-ID and ID-to-token mappings
       - Provides category-based token retrieval methods
       - Supports serialization via to_dict/from_dict methods

Token Categories:
    **Standard Tokens** (7 tokens):
        - BOS/EOS: Sequence boundary markers
        - PAD: Padding for batch processing
        - UNK: Unknown token fallback
        - MASK: Masked language modeling
        - SEP/CLS: Sentence separation and classification
    
    **Multimodal Tokens** (4 tokens):
        - <image>: Image modality marker
        - <audio>: Audio modality marker
        - <video>: Video modality marker
        - <document>: Document modality marker
    
    **Reasoning Tokens** (10 tokens):
        - <think/>, </think/: Structured thinking blocks
        - <answer/>, </answer>: Answer generation markers
        - <verify/>, </verify>: Verification blocks
        - <reasoning/>, </reasoning>: Reasoning sections
        - <reflection/>, </reflection>: Reflection blocks
    
    **Agent Tokens** (6 tokens):
        - <tool/>, </tool>: Tool invocation markers
        - <action/>, </action>: Action execution markers
        - <observation/>, </observation>: Observation markers

Example:
    >>> from model.tokenizer.special_tokens import YvSpecialTokens, YvSpecialTokenType
    >>> 
    >>> # Create with defaults
    >>> special = YvSpecialTokens()
    >>> 
    >>> # Get token ID
    >>> bos_id = special.token_to_id("<s>")
    >>> print(f"BOS token ID: {bos_id}")
    >>> 
    >>> # Check token type
    >>> token_type = special.get_token_type("<image>")
    >>> print(f"Token type: {token_type}")  # YvSpecialTokenType.MULTIMODAL
    >>> 
    >>> # Serialize
    >>> config = special.to_dict()
    >>> restored = YvSpecialTokens.from_dict(config)

Dependencies:
    - dataclasses: Dataclass support for YvSpecialTokens
    - enum: Enum support for YvSpecialTokenType
    - typing: Type hints for annotations

Note:
    The module uses dataclass __post_init__ to build internal mappings
    automatically after initialization. Custom tokens can be added via
    the add_tokens() method.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class YvSpecialTokenType(Enum):
    """Enumeration of special token categories for classification.
    
    This enum defines the five categories of special tokens used in
    the Yv architecture, enabling type-based filtering and queries.
    
    Attributes:
        STANDARD: Basic tokens for sequence processing (BOS, EOS, PAD, etc.)
        MULTIMODAL: Tokens for multimodal content (image, audio, video, document)
        REASONING: Tokens for reasoning tasks (think, answer, verify)
        AGENT: Tokens for agent interactions (tool, action, observation)
        CONTROL: Tokens for structural control (separator, mask, class)
    
    Example:
        >>> token_type = YvSpecialTokenType.MULTIMODAL
        >>> print(token_type.value)  # "multimodal"
    """
    STANDARD = "standard"
    MULTIMODAL = "multimodal"
    REASONING = "reasoning"
    AGENT = "agent"
    CONTROL = "control"


@dataclass
class YvSpecialTokens:
    """Special tokens configuration for Yv tokenizer.
    
    This dataclass defines all special tokens used in the Yv model,
    organized into logical categories. It provides bidirectional mappings
    between tokens and their IDs, along with category-based retrieval methods.
    
    Token Categories:
        - **Standard**: BOS, EOS, PAD, UNK, MASK, SEP, CLS
        - **Multimodal**: image, audio, video, document
        - **Reasoning**: think, answer, verify, reasoning, reflection
        - **Agent**: tool, action, observation
    
    Attributes:
        bos_token (str): Beginning-of-sequence token. Default: "<s>".
        eos_token (str): End-of-sequence token. Default: "</s>".
        pad_token (str): Padding token for batch alignment. Default: "<pad>".
        unk_token (str): Unknown token for out-of-vocabulary items. Default: "<unk>".
        mask_token (str): Mask token for masked language modeling. Default: "<mask>".
        sep_token (str): Separator token between sequences. Default: "<sep>".
        cls_token (str): Classification token for sentence-level tasks. Default: "<cls>".
        image_token (str): Image modality marker. Default: "<image>".
        audio_token (str): Audio modality marker. Default: "<audio>".
        video_token (str): Video modality marker. Default: "<video>".
        document_token (str): Document modality marker. Default: "<document>".
        think_token (str): Thinking block start marker. Default: "<think".
        think_end_token (str): Thinking block end marker. Default: "</think".
        answer_token (str): Answer block start marker. Default: "<answer>".
        answer_end_token (str): Answer block end marker. Default: "</answer>".
        verify_token (str): Verification block start marker. Default: "<verify>".
        verify_end_token (str): Verification block end marker. Default: "</verify>".
        tool_token (str): Tool invocation start marker. Default: "<tool>".
        tool_end_token (str): Tool invocation end marker. Default: "</tool>".
        action_token (str): Action execution start marker. Default: "<action>".
        action_end_token (str): Action execution end marker. Default: "</action>".
        observation_token (str): Observation start marker. Default: "<observation>".
        observation_end_token (str): Observation end marker. Default: "</observation>".
        reasoning_start (str): Reasoning section start marker. Default: "<reasoning>".
        reasoning_end (str): Reasoning section end marker. Default: "</reasoning>".
        reflection_start (str): Reflection section start marker. Default: "<reflection>".
        reflection_end (str): Reflection section end marker. Default: "</reflection>".
        additional_special_tokens (List[str]): Extra special tokens. Default: [].
    
    Example:
        >>> special = YvSpecialTokens()
        >>> print(special.vocab_size)  # Number of special tokens
        >>> print(special.get_multimodal_tokens())  # ['<image>', '<audio>', ...]
    
    Note:
        Internal mappings (_token_to_id, _id_to_token) are built automatically
        in __post_init__. Use token_to_id() and id_to_token() for lookups.
    """

    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    mask_token: str = "<mask>"
    sep_token: str = "<sep>"
    cls_token: str = "<cls>"

    image_token: str = "<image>"
    audio_token: str = "<audio>"
    video_token: str = "<video>"
    document_token: str = "<document>"

    think_token: str = "<think"
    think_end_token: str = "</think"
    answer_token: str = "<answer>"
    answer_end_token: str = "</answer>"
    verify_token: str = "<verify>"
    verify_end_token: str = "</verify>"

    tool_token: str = "<tool>"
    tool_end_token: str = "</tool>"
    action_token: str = "<action>"
    action_end_token: str = "</action>"
    observation_token: str = "<observation>"
    observation_end_token: str = "</observation>"

    reasoning_start: str = "<reasoning>"
    reasoning_end: str = "</reasoning>"
    reflection_start: str = "<reflection>"
    reflection_end: str = "</reflection>"

    additional_special_tokens: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize internal mappings after dataclass construction.
        
        This method is automatically called after the dataclass is initialized.
        It creates the bidirectional token-to-ID and ID-to-token mappings
        used for efficient lookups.
        
        Side Effects:
            Initializes _token_to_id and _id_to_token dictionaries.
        """
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._build_mappings()

    def _build_mappings(self):
        """Build bidirectional token-to-ID mappings.
        
        Iterates through all tokens (including additional_special_tokens)
        and creates mappings for efficient lookup operations.
        
        Side Effects:
            Populates _token_to_id and _id_to_token dictionaries.
        """
        all_tokens = self.get_all_tokens()
        for idx, token in enumerate(all_tokens):
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token

    def get_all_tokens(self) -> List[str]:
        """Get all special tokens as a flat list.
        
        Returns all defined special tokens in a consistent order,
        including any additional_special_tokens that were added.
        
        Returns:
            List[str]: Complete list of all special token strings.
        
        Note:
            The order is significant for ID assignment. PAD token is
            always first (ID 0) for padding convenience.
        """
        return [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.mask_token,
            self.sep_token,
            self.cls_token,
            self.image_token,
            self.audio_token,
            self.video_token,
            self.document_token,
            self.think_token,
            self.think_end_token,
            self.answer_token,
            self.answer_end_token,
            self.verify_token,
            self.verify_end_token,
            self.tool_token,
            self.tool_end_token,
            self.action_token,
            self.action_end_token,
            self.observation_token,
            self.observation_end_token,
            self.reasoning_start,
            self.reasoning_end,
            self.reflection_start,
            self.reflection_end,
        ] + self.additional_special_tokens

    def get_standard_tokens(self) -> List[str]:
        """Get standard processing tokens.
        
        Returns the basic tokens used for sequence processing,
        including boundary markers, padding, and special markers.
        
        Returns:
            List[str]: List of 7 standard tokens (BOS, EOS, PAD, UNK,
                MASK, SEP, CLS).
        """
        return [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.unk_token,
            self.mask_token,
            self.sep_token,
            self.cls_token,
        ]

    def get_multimodal_tokens(self) -> List[str]:
        """Get multimodal content tokens.
        
        Returns tokens used to mark different modalities in the input,
        enabling the model to distinguish between text and other content.
        
        Returns:
            List[str]: List of 4 multimodal tokens (image, audio,
                video, document).
        """
        return [
            self.image_token,
            self.audio_token,
            self.video_token,
            self.document_token,
        ]

    def get_reasoning_tokens(self) -> List[str]:
        """Get reasoning task tokens.
        
        Returns tokens used to structure reasoning processes, including
        thinking blocks, answer generation, and verification sections.
        
        Returns:
            List[str]: List of 10 reasoning tokens (think, answer,
                verify, reasoning, reflection with their end markers).
        """
        return [
            self.think_token,
            self.think_end_token,
            self.answer_token,
            self.answer_end_token,
            self.verify_token,
            self.verify_end_token,
            self.reasoning_start,
            self.reasoning_end,
            self.reflection_start,
            self.reflection_end,
        ]

    def get_agent_tokens(self) -> List[str]:
        """Get agent interaction tokens.
        
        Returns tokens used for agent-based interactions, including
        tool invocations, action executions, and observations.
        
        Returns:
            List[str]: List of 6 agent tokens (tool, action, observation
                with their end markers).
        """
        return [
            self.tool_token,
            self.tool_end_token,
            self.action_token,
            self.action_end_token,
            self.observation_token,
            self.observation_end_token,
        ]

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token string to its ID.
        
        Looks up the integer ID for a given token string. Returns None
        if the token is not a registered special token.
        
        Args:
            token (str): Token string to look up.
        
        Returns:
            Optional[int]: Token ID if found, None otherwise.
        
        Example:
            >>> special = YvSpecialTokens()
            >>> special.token_to_id("<s>")  # Returns BOS token ID
            >>> special.token_to_id("hello")  # Returns None
        """
        return self._token_to_id.get(token)

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert a token ID to its string representation.
        
        Looks up the token string for a given integer ID. Returns None
        if the ID is not a valid special token ID.
        
        Args:
            token_id (int): Token ID to look up.
        
        Returns:
            Optional[str]: Token string if found, None otherwise.
        
        Example:
            >>> special = YvSpecialTokens()
            >>> special.id_to_token(0)  # Returns "<pad>"
        """
        return self._id_to_token.get(token_id)

    def is_special_token(self, token: str) -> bool:
        """Check if a token is a registered special token.
        
        Args:
            token (str): Token string to check.
        
        Returns:
            bool: True if token is a special token, False otherwise.
        
        Example:
            >>> special = YvSpecialTokens()
            >>> special.is_special_token("<s>")  # True
            >>> special.is_special_token("hello")  # False
        """
        return token in self._token_to_id

    def get_token_type(self, token: str) -> Optional[YvSpecialTokenType]:
        """Get the category type of a special token.
        
        Determines which category (STANDARD, MULTIMODAL, REASONING, AGENT,
        or CONTROL) a token belongs to.
        
        Args:
            token (str): Token string to categorize.
        
        Returns:
            Optional[YvSpecialTokenType]: Token type enum value if
                the token is a special token, None otherwise.
        
        Example:
            >>> special = YvSpecialTokens()
            >>> special.get_token_type("<image>")
            YvSpecialTokenType.MULTIMODAL
        """
        if token in self.get_standard_tokens():
            return YvSpecialTokenType.STANDARD
        elif token in self.get_multimodal_tokens():
            return YvSpecialTokenType.MULTIMODAL
        elif token in self.get_reasoning_tokens():
            return YvSpecialTokenType.REASONING
        elif token in self.get_agent_tokens():
            return YvSpecialTokenType.AGENT
        elif token in [self.sep_token, self.mask_token, self.cls_token]:
            return YvSpecialTokenType.CONTROL
        return None

    @property
    def vocab_size(self) -> int:
        """Get the total number of special tokens.
        
        Returns:
            int: Number of registered special tokens including
                additional_special_tokens.
        """
        return len(self._token_to_id)

    def add_tokens(self, tokens: List[str]) -> int:
        """Add new special tokens to the vocabulary.
        
        Adds new tokens to the additional_special_tokens list and updates
        the internal mappings. Tokens that already exist are skipped.
        
        Args:
            tokens (List[str]): List of token strings to add.
        
        Returns:
            int: Number of tokens actually added (excluding duplicates).
        
        Example:
            >>> special = YvSpecialTokens()
            >>> added = special.add_tokens(["<custom1>", "<custom2>"])
            >>> print(f"Added {added} new tokens")
        """
        added = 0
        for token in tokens:
            if token not in self._token_to_id:
                new_id = len(self._token_to_id)
                self._token_to_id[token] = new_id
                self._id_to_token[new_id] = token
                self.additional_special_tokens.append(token)
                added += 1
        return added

    def to_dict(self) -> Dict[str, any]:
        """Serialize special tokens configuration to dictionary.
        
        Converts all token definitions to a JSON-serializable dictionary,
        suitable for saving to configuration files.
        
        Returns:
            Dict[str, any]: Dictionary with all token definitions.
        
        Example:
            >>> special = YvSpecialTokens()
            >>> config = special.to_dict()
            >>> print(config['bos_token'])  # "<s>"
        """
        return {
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "mask_token": self.mask_token,
            "sep_token": self.sep_token,
            "cls_token": self.cls_token,
            "image_token": self.image_token,
            "audio_token": self.audio_token,
            "video_token": self.video_token,
            "document_token": self.document_token,
            "think_token": self.think_token,
            "think_end_token": self.think_end_token,
            "answer_token": self.answer_token,
            "answer_end_token": self.answer_end_token,
            "verify_token": self.verify_token,
            "verify_end_token": self.verify_end_token,
            "tool_token": self.tool_token,
            "tool_end_token": self.tool_end_token,
            "action_token": self.action_token,
            "action_end_token": self.action_end_token,
            "observation_token": self.observation_token,
            "observation_end_token": self.observation_end_token,
            "reasoning_start": self.reasoning_start,
            "reasoning_end": self.reasoning_end,
            "reflection_start": self.reflection_start,
            "reflection_end": self.reflection_end,
            "additional_special_tokens": self.additional_special_tokens,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, any]) -> "YvSpecialTokens":
        """Deserialize special tokens configuration from dictionary.
        
        Reconstructs a YvSpecialTokens instance from a dictionary
        configuration, typically loaded from a JSON file.
        
        Args:
            config (Dict[str, any]): Dictionary containing token definitions.
                Keys should match dataclass field names.
        
        Returns:
            YvSpecialTokens: Reconstructed special tokens instance.
        
        Example:
            >>> config = {"bos_token": "<BOS>", "eos_token": "<EOS>"}
            >>> special = YvSpecialTokens.from_dict(config)
        """
        return cls(**config)
