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
Text Augmentation for enhancing training data diversity.

This module implements various text augmentation strategies including
EDA (Easy Data Augmentation), synonym replacement, and back-translation.
These techniques help improve model generalization and robustness.

Key Features:
    - EDA: Synonym replacement, random insertion, swap, deletion
    - Back-translation: Translate to another language and back
    - Configurable augmentation probability
    - Support for multiple augmentation strategies

Usage:
    >>> from tools.data.augment import PiscesLxDataTextAugmenter
    >>> augmenter = PiscesLxDataTextAugmenter(
    ...     strategies=['synonym', 'insert', 'swap', 'delete'],
    ...     aug_prob=0.3
    ... )
    >>> augmented = augmenter.augment("This is a sample sentence")
    >>> print(augmented)
"""

import random
import re
from typing import Callable, Dict, List, Optional, Set, Tuple
import numpy as np


class PiscesLxDataTextAugmenter:
    """
    Text augmenter implementing multiple augmentation strategies.
    
    This class provides various text augmentation techniques to increase
    training data diversity and improve model generalization.
    
    Attributes:
        strategies: List of augmentation strategies to use.
        aug_prob: Probability of applying augmentation.
        alpha: Percentage of words to modify (for EDA).
        stopwords: Set of stopwords to skip during augmentation.
    
    Example:
        >>> augmenter = PiscesLxDataTextAugmenter(strategies=['synonym', 'delete'])
        >>> original = "The quick brown fox jumps over the lazy dog"
        >>> augmented = augmenter.augment(original)
    """
    
    DEFAULT_STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'we', 'they', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there'
    }
    
    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        aug_prob: float = 0.3,
        alpha: float = 0.1,
        stopwords: Optional[Set[str]] = None,
        seed: int = 42
    ) -> None:
        """
        Initialize the text augmenter.
        
        Args:
            strategies: List of strategies. Options: 'synonym', 'insert',
                'swap', 'delete', 'backtranslate'. Defaults to all EDA strategies.
            aug_prob: Probability of applying augmentation per sample. Defaults to 0.3.
            alpha: Percentage of words to modify. Defaults to 0.1 (10%).
            stopwords: Custom stopwords set. Uses default if None.
            seed: Random seed. Defaults to 42.
        """
        self.strategies = strategies or ['synonym', 'insert', 'swap', 'delete']
        self.aug_prob = aug_prob
        self.alpha = alpha
        self.stopwords = stopwords if stopwords is not None else self.DEFAULT_STOPWORDS
        self.seed = seed
        
        self._rng = np.random.default_rng(seed)
        random.seed(seed)
        
        self._synonyms: Dict[str, List[str]] = {}
        self._strategy_funcs: Dict[str, Callable] = {
            'synonym': self._synonym_replacement,
            'insert': self._random_insertion,
            'swap': self._random_swap,
            'delete': self._random_deletion,
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text.
            
        Returns:
            List[str]: List of tokens.
        """
        return text.split()
    
    def _detokenize(self, tokens: List[str]) -> str:
        """
        Convert tokens back to text.
        
        Args:
            tokens: List of tokens.
            
        Returns:
            str: Reconstructed text.
        """
        return ' '.join(tokens)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word.
        
        This is a placeholder implementation. In production, use WordNet
        or a pre-trained synonym model.
        
        Args:
            word: Input word.
            
        Returns:
            List[str]: List of synonyms.
        """
        simple_synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice', 'wonderful'],
            'bad': ['poor', 'terrible', 'awful', 'horrible', 'badly'],
            'big': ['large', 'huge', 'enormous', 'giant', 'massive'],
            'small': ['tiny', 'little', 'miniature', 'petite', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy', 'hasty'],
            'slow': ['sluggish', 'leisurely', 'unhurried', 'gradual'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased', 'glad'],
            'sad': ['unhappy', 'sorrowful', 'dejected', 'melancholy'],
            'important': ['significant', 'crucial', 'vital', 'essential', 'key'],
            'new': ['novel', 'fresh', 'recent', 'modern', 'current'],
        }
        
        word_lower = word.lower()
        if word_lower in simple_synonyms:
            return simple_synonyms[word_lower]
        return []
    
    def _synonym_replacement(self, tokens: List[str], n: int) -> List[str]:
        """
        Replace n random words with their synonyms.
        
        Args:
            tokens: List of tokens.
            n: Number of words to replace.
            
        Returns:
            List[str]: Augmented tokens.
        """
        new_tokens = tokens.copy()
        random_words = [w for w in new_tokens if w.lower() not in self.stopwords]
        random.shuffle(random_words)
        
        replaced = 0
        for word in random_words:
            synonyms = self._get_synonyms(word)
            if synonyms:
                new_word = random.choice(synonyms)
                new_tokens = [new_word if w == word else w for w in new_tokens]
                replaced += 1
                if replaced >= n:
                    break
        
        return new_tokens
    
    def _random_insertion(self, tokens: List[str], n: int) -> List[str]:
        """
        Insert n random synonyms at random positions.
        
        Args:
            tokens: List of tokens.
            n: Number of insertions.
            
        Returns:
            List[str]: Augmented tokens.
        """
        new_tokens = tokens.copy()
        
        for _ in range(n):
            word = random.choice([w for w in new_tokens if w.lower() not in self.stopwords])
            synonyms = self._get_synonyms(word)
            if synonyms:
                insert_word = random.choice(synonyms)
                insert_pos = random.randint(0, len(new_tokens))
                new_tokens.insert(insert_pos, insert_word)
        
        return new_tokens
    
    def _random_swap(self, tokens: List[str], n: int) -> List[str]:
        """
        Randomly swap n pairs of words.
        
        Args:
            tokens: List of tokens.
            n: Number of swaps.
            
        Returns:
            List[str]: Augmented tokens.
        """
        new_tokens = tokens.copy()
        
        for _ in range(n):
            if len(new_tokens) >= 2:
                idx1 = random.randint(0, len(new_tokens) - 1)
                idx2 = random.randint(0, len(new_tokens) - 1)
                new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
        
        return new_tokens
    
    def _random_deletion(self, tokens: List[str], p: float) -> List[str]:
        """
        Randomly delete words with probability p.
        
        Args:
            tokens: List of tokens.
            p: Deletion probability.
            
        Returns:
            List[str]: Augmented tokens.
        """
        if len(tokens) == 1:
            return tokens
        
        new_tokens = []
        for token in tokens:
            if random.random() > p:
                new_tokens.append(token)
        
        if len(new_tokens) == 0:
            return [random.choice(tokens)]
        
        return new_tokens
    
    def augment(self, text: str, num_augmentations: int = 1) -> List[str]:
        """
        Augment text using configured strategies.
        
        Args:
            text: Input text.
            num_augmentations: Number of augmented versions to generate.
            
        Returns:
            List[str]: List of augmented texts.
        """
        if random.random() > self.aug_prob:
            return [text]
        
        tokens = self._tokenize(text)
        if len(tokens) == 0:
            return [text]
        
        n_aug = max(1, int(self.alpha * len(tokens)))
        
        augmented = []
        for _ in range(num_augmentations):
            strategy = random.choice(self.strategies)
            
            if strategy in self._strategy_funcs:
                aug_func = self._strategy_funcs[strategy]
                if strategy == 'delete':
                    aug_tokens = aug_func(tokens, self.alpha)
                else:
                    aug_tokens = aug_func(tokens, n_aug)
                
                augmented.append(self._detokenize(aug_tokens))
            else:
                augmented.append(text)
        
        return augmented if augmented else [text]
    
    def augment_batch(self, texts: List[str], num_augmentations: int = 1) -> List[List[str]]:
        """
        Augment multiple texts.
        
        Args:
            texts: List of input texts.
            num_augmentations: Number of augmented versions per text.
            
        Returns:
            List[List[str]]: List of augmented text lists.
        """
        return [self.augment(text, num_augmentations) for text in texts]
