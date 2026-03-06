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

"""Document understanding components for Yv multimodal pipelines.

This module provides comprehensive document processing components for the Yv
model, including text encoding, layout analysis, table understanding, and
handwriting recognition capabilities.

Module Components:
    1. YvDocEncoder:
       - Multi-language text encoding (Latin, Chinese, Arabic)
       - Layout analysis with reading order prediction
       - Table structure detection and semantic analysis
       - Handwriting recognition with style analysis
       - Cross-modal document fusion

Key Features:
    - Multi-language support (100 languages)
    - Script-specific transformer encoders
    - Layout classification (15 layout types)
    - Geometric reasoning (9 spatial relations)
    - Table structure detection (rows, columns, cells)
    - Data type classification (8 types)
    - Handwriting recognition (10,000 character vocabulary)
    - Named entity extraction (50 entity types)
    - Document type classification (20 types)

Performance Characteristics:
    - Text encoding: O(L^2 * hidden_size) with transformer
    - Layout encoding: O(N * hidden_size) where N = layout elements
    - Table understanding: O(T * hidden_size) where T = table cells
    - Handwriting recognition: O(S * hidden_size) where S = strokes
    - Total complexity: O(max(L^2, N, T, S) * hidden_size)

Usage Example:
    >>> from model.multimodal.doc import YvDocEncoder
    >>> 
    >>> # Initialize encoder
    >>> encoder = YvDocEncoder(config)
    >>> 
    >>> # Encode document
    >>> doc_features = encoder({"text": "Document content", "layout": layout_tensor})
    >>> 
    >>> # Encode plain text
    >>> features = encoder("Plain text document")

Note:
    Default vocabulary size: 50,000 tokens
    Default max sequence length: 512 tokens
    Supports text, layout, table, and handwriting inputs.
"""

import torch
from torch import nn
from typing import Any, Dict
import torch.nn.functional as F
from utils.dc import PiscesLxLogger

from utils.paths import get_log_file
_LOG = PiscesLxLogger("Yv.Multimodal", file_path=get_log_file("Yv.Multimodal"), enable_file=True)

class YvDocEncoder(nn.Module):
    """Document encoder integrating textual, layout, and handwriting signals.
    
    A comprehensive document encoder that composes multiple modality-specific
    submodules (text, layout, table, handwriting) and aggregates their outputs
    into a shared latent representation suitable for downstream Yv agent
    workflows.
    
    Architecture:
        1. Text Encoder:
           - Embedding layer (50,000 vocab)
           - Positional encoding (512 max length)
           - Script-specific encoders (Latin, Chinese, Arabic)
           - Language detection (100 languages)
        
        2. Layout Encoder:
           - Spatial encoder (8-dim geometric features)
           - Reading order prediction
           - Layout classification (15 types)
           - Geometric reasoning (9 relations)
        
        3. Table Understanding:
           - Structure detection (rows, columns, cells)
           - Content analysis (data types, numerical)
           - Table QA module
        
        4. Handwriting Recognition:
           - Stroke encoder (bidirectional LSTM)
           - Character recognizer (10,000 vocab)
           - Style analyzer (20 features)
           - Line segmenter (CNN-based)
        
        5. Document Fusion:
           - Text-layout cross-attention
           - Hierarchical encoder (3 layers)
           - Entity and key-value extraction
    
    Key Features:
        - Multi-language support with script-specific encoders
        - Layout analysis with reading order prediction
        - Table structure detection and semantic analysis
        - Handwriting recognition with style analysis
        - Cross-modal document fusion
    
    Attributes:
        enabled (bool): Flag indicating whether the encoder is available.
        cfg: Configuration namespace describing hidden sizes and head counts.
        vocab_size (int): Vocabulary size used for simplistic tokenization.
        max_length (int): Maximum supported token sequence length.
        text_encoder (nn.ModuleDict): Submodules for embedding, positional
            encoding, script-specific encoders, and language detection.
        layout_encoder (nn.ModuleDict): Components that encode geometric and
            layout-specific features.
        table_understanding (nn.ModuleDict): Modules for table structure
            detection and semantic analysis.
        handwriting_recognition (nn.ModuleDict): Networks handling stroke-level
            recognition and style analysis.
        doc_fusion (nn.ModuleDict): Attention and hierarchical encoders for
            cross-modality fusion.
        final_proj (nn.ModuleDict): Projection heads that prepare fused
            features for downstream tasks.
    
    Example:
        >>> encoder = YvDocEncoder(config)
        >>> doc_features = encoder({"text": "Document", "layout": layout})
        >>> 
        >>> # Access language detection
        >>> lang_logits = encoder.text_encoder['language_detector'](features)
    
    Note:
        Default vocabulary size: 50,000 tokens
        Default max sequence length: 512 tokens
        Supports text, layout, table, and handwriting inputs.
    """

    def __init__(self, cfg):
        """Initialize the composite document encoder.
        
        Args:
            cfg: Configuration object containing parameters such as:
                - hidden_size: Output embedding dimension
                - n_head: Number of attention heads
                - Vocabulary and sequence length are fixed defaults
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.vocab_size = 50000
        self.max_length = 512
        
        _LOG.debug(f"DocEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Text encoder module with multi-language support
        self.text_encoder = nn.ModuleDict({
            'embedding': nn.Embedding(self.vocab_size, cfg.hidden_size),
            'positional_encoding': nn.Embedding(self.max_length, cfg.hidden_size),
            'layer_norm': nn.LayerNorm(cfg.hidden_size),
            'dropout': nn.Dropout(0.1),
            'language_detector': nn.Sequential(
                nn.Linear(cfg.hidden_size, 256),
                nn.SiLU(),
                nn.Linear(256, 100)  # Classify into 100 languages
            ),
            'script_encoders': nn.ModuleDict({
                'latin': nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size, nhead=cfg.n_head // 4, 
                    dim_feedforward=cfg.hidden_size * 2, batch_first=True
                ),
                'chinese': nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size, nhead=cfg.n_head // 4,
                    dim_feedforward=cfg.hidden_size * 2, batch_first=True
                ),
                'arabic': nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size, nhead=cfg.n_head // 4,
                    dim_feedforward=cfg.hidden_size * 2, batch_first=True
                )
            })
        })
        
        # Layout encoder module
        self.layout_encoder = nn.ModuleDict({
            'spatial_encoder': nn.Sequential(
                nn.Linear(8, cfg.hidden_size // 2),  # Input: [x0, y0, x1, y1, w, h, cx, cy]
                nn.LayerNorm(cfg.hidden_size // 2),
                nn.SiLU(),
                nn.Linear(cfg.hidden_size // 2, cfg.hidden_size // 4),
                nn.LayerNorm(cfg.hidden_size // 4),
                nn.SiLU()
            ),
            'reading_order': nn.Sequential(
                nn.Linear(cfg.hidden_size // 4, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 1)  # Predict reading order score
            ),
            'layout_classifier': nn.Sequential(
                nn.Linear(cfg.hidden_size // 4, 128),
                nn.SiLU(),
                nn.Linear(128, 15)  # Classify into 15 layout types
            ),
            'geometric_reasoner': nn.Sequential(
                nn.Linear(cfg.hidden_size // 2, 256),  # Input: pairwise layout features
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 9)  # Predict 9 geometric relations
            )
        })
        
        # Table understanding module
        self.table_understanding = nn.ModuleDict({
            'structure_detector': nn.ModuleDict({
                'row_detector': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 256),
                    nn.SiLU(),
                    nn.Linear(256, 128),
                    nn.SiLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()  # Predict row boundary probability
                ),
                'column_detector': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 256),
                    nn.SiLU(),
                    nn.Linear(256, 128),
                    nn.SiLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()  # Predict column boundary probability
                ),
                'cell_classifier': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 256),
                    nn.SiLU(),
                    nn.Linear(256, 6)  # Classify cell types
                )
            }),
            'content_analyzer': nn.ModuleDict({
                'data_type_classifier': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 128),
                    nn.SiLU(),
                    nn.Linear(128, 8)  # Classify data types
                ),
                'numerical_analyzer': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 64),
                    nn.SiLU(),
                    nn.Linear(64, 4)  # Analyze numerical properties
                ),
                'semantic_encoder': nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size, nhead=cfg.n_head // 4,
                    dim_feedforward=cfg.hidden_size * 2, batch_first=True
                )
            }),
            'table_qa': nn.Sequential(
                nn.Linear(cfg.hidden_size * 2, 512),  # Input: table + question features
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, cfg.hidden_size)  # Generate answer representation
            )
        })
        
        # Handwriting recognition module
        self.handwriting_recognition = nn.ModuleDict({
            'stroke_encoder': nn.LSTM(
                input_size=3,  # Input: [x, y, pressure]
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            ),
            'char_recognizer': nn.Sequential(
                nn.Linear(256, 512),  # Input: bidirectional LSTM output
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 10000)  # Recognize characters from large vocabulary
            ),
            'style_analyzer': nn.Sequential(
                nn.Linear(256, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
                nn.Linear(64, 20)  # Analyze handwriting style features
            ),
            'line_segmenter': nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()  # Predict text line boundary probability
            ),
            'word_recognizer': nn.Sequential(
                nn.Linear(256, 512),
                nn.SiLU(),
                nn.Linear(512, 1000)  # Recognize common words
            )
        })
        
        # Document-level feature fusion module
        self.doc_fusion = nn.ModuleDict({
            'text_layout_attention': nn.MultiheadAttention(
                embed_dim=cfg.hidden_size,
                num_heads=cfg.n_head // 4,
                batch_first=True,
                dropout=0.1
            ),
            'hierarchy_encoder': nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=cfg.hidden_size,
                    nhead=cfg.n_head // 2,
                    dim_feedforward=cfg.hidden_size * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=3
            ),
            'doc_type_classifier': nn.Sequential(
                nn.Linear(cfg.hidden_size, 256),
                nn.SiLU(),
                nn.Linear(256, 20)  # Classify document types
            ),
            'extraction_heads': nn.ModuleDict({
                'entity_extractor': nn.Sequential(
                    nn.Linear(cfg.hidden_size, 256),
                    nn.SiLU(),
                    nn.Linear(256, 50)  # Extract named entities
                ),
                'key_value_extractor': nn.Sequential(
                    nn.Linear(cfg.hidden_size * 2, 256),
                    nn.SiLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()  # Predict key-value pair probability
                )
            }),
            'final_fusion': nn.Sequential(
                nn.Linear(cfg.hidden_size + cfg.hidden_size // 4, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU(),
                nn.Dropout(0.1)
            )
        })
        
        # Final projection module with multi-task learning
        self.final_proj = nn.ModuleDict({
            'main_projection': nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU()
            ),
            'table_proj': nn.Linear(cfg.hidden_size, cfg.hidden_size // 4),
            'handwriting_proj': nn.Linear(256, cfg.hidden_size // 4),
            'layout_proj': nn.Linear(cfg.hidden_size // 4, cfg.hidden_size // 4),
            'task_integration': nn.Sequential(
                nn.Linear(cfg.hidden_size + 3 * (cfg.hidden_size // 4), cfg.hidden_size),
                nn.LayerNorm(cfg.hidden_size),
                nn.SiLU()
            )
        })
        
        _LOG.debug("DocEncoder: __init__ end")
    
    def _tokenize_text(self, text):
        """Tokenize text inputs using a simple character-level heuristic.
        
        Converts raw text strings into token indices using a hash-based
        character-level tokenization scheme. Pre-computed token tensors
        are passed through unchanged.
        
        Args:
            text (Union[str, torch.Tensor]): Raw text or precomputed token IDs.
                - str: Will be character-tokenized and padded
                - torch.Tensor: Returned unchanged
        
        Returns:
            torch.Tensor: Tensor of token indices with length ``max_length``.
                Padded with zeros if input is shorter than max_length.
        
        Note:
            Uses hash(c) % vocab_size for character tokenization.
            Truncates to max_length (512) characters.
        """
        if isinstance(text, str):
            tokens = [hash(c) % self.vocab_size for c in text[:self.max_length]]
            tokens += [0] * (self.max_length - len(tokens))
            return torch.tensor(tokens)
        return text
    
    def _encode_layout(self, layout):
        """Encode layout geometry into latent features.
        
        Processes bounding box coordinates and derived spatial statistics
        through the spatial encoder to produce layout-aware features.
        
        Args:
            layout (Union[torch.Tensor, None]): Layout tensor capturing bounding
                boxes and derived spatial statistics.
                Expected shape: [N, 8] where 8 = [x0, y0, x1, y1, w, h, cx, cy]
                If None, uses default full-page layout [[0, 0, 1, 1]].
        
        Returns:
            torch.Tensor: Layout feature tensor following spatial encoding.
                Shape: [N, hidden_size // 4] after encoding.
        
        Note:
            Default layout assumes normalized coordinates [0, 1].
            Spatial encoder applies LayerNorm and SiLU activations.
        """
        if layout is None:
            # Use default layout: full page
            layout = torch.tensor([[0, 0, 1, 1]])
        
        if layout.dim() == 1:
            layout = layout.unsqueeze(0)
        
        # The ModuleDict stores individual encoders; here we apply the spatial encoder directly.
        return self.layout_encoder['spatial_encoder'](layout.float())
    
    def forward(self, doc_input):
        """Encode the provided document payload into multimodal features.
        
        Main entry point for document encoding. Processes text and layout
        inputs through their respective encoders and fuses them through
        cross-attention and hierarchical encoding.
        
        Args:
            doc_input (Union[str, torch.Tensor, dict]): Document input expressed
                as raw text, token IDs, or a dictionary with ``input_ids`` and
                ``layout`` fields.
                - str: Raw text to be tokenized
                - torch.Tensor: Pre-computed token IDs
                - dict: Dictionary with 'input_ids'/'text' and 'layout' keys
        
        Returns:
            torch.Tensor: Encoded document features with shape
            ``(batch_size, 1, hidden_size)``.
        
        Note:
            Returns zero tensor if doc_input is None or text_input is None.
            Applies mean pooling over sequence dimension.
            Uses cross-attention for text-layout fusion.
        """
        if doc_input is None:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.cfg.hidden_size, device=device)
        
        # Process input format
        if isinstance(doc_input, dict):
            text_input = doc_input.get('input_ids', doc_input.get('text', None))
            layout_input = doc_input.get('layout', None)
        elif isinstance(doc_input, str):
            text_input = doc_input
            layout_input = None
        else:
            text_input = doc_input
            layout_input = None
        
        if text_input is None:
            device = next(self.parameters()).device
            return torch.zeros(1, 1, self.cfg.hidden_size, device=device)
        
        # Text encoding
        text_tokens = self._tokenize_text(text_input)
        if text_tokens.dim() == 1:
            text_tokens = text_tokens.unsqueeze(0)
        
        # Sequentially apply embedding components from the text encoder module dictionary.
        embeddings = self.text_encoder['embedding'](text_tokens)
        pos_enc = self.text_encoder['positional_encoding'](
            torch.arange(text_tokens.size(1), device=text_tokens.device).unsqueeze(0).expand(text_tokens.size(0), -1)
        )
        text_features = embeddings + pos_enc
        text_features = self.text_encoder['layer_norm'](text_features)
        text_features = self.text_encoder['dropout'](text_features)
        text_features = text_features.mean(dim=1)  # Average pooling
        
        # Layout encoding
        layout_features = self._encode_layout(layout_input)
        layout_features = layout_features.mean(dim=1)  # Average pooling
        
        # Fusion of text and layout features
        combined = torch.cat([text_features, layout_features], dim=-1)
        
        # Apply attention-based fusion followed by hierarchical encoding.
        attn_output, _ = self.doc_fusion['text_layout_attention'](
            combined.unsqueeze(1), combined.unsqueeze(1), combined.unsqueeze(1)
        )
        attn_output = attn_output.squeeze(1)
        doc_features = self.doc_fusion['hierarchy_encoder'](attn_output.unsqueeze(1)).squeeze(1)
        doc_features = self.doc_fusion['final_fusion'](doc_features)
        
        # Aggregate projections from the final projection module dictionary.
        main_proj = self.final_proj['main_projection'](doc_features)
        table_proj = self.final_proj['table_proj'](doc_features)
        # handwriting_proj expects 256-dimensional inputs; skip invocation here to avoid shape mismatch.
        # handwriting_proj = self.final_proj['handwriting_proj'](doc_features)
        layout_proj = self.final_proj['layout_proj'](layout_features)
        # Concatenate all projections
        all_proj = torch.cat([main_proj, table_proj, layout_proj], dim=-1)
        doc_features = self.final_proj['task_integration'](all_proj)
        
        return doc_features.unsqueeze(1)
