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

# Paper: Huang et al., "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking", ACM Multimedia 2022, arXiv:2204.08387
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

    def __init__(self, cfg, device=None, dtype=None):
        """Initialize the composite document encoder.

        Args:
            cfg: Configuration object containing parameters such as:
                - hidden_size: Output embedding dimension
                - n_head: Number of attention heads
                - Vocabulary and sequence length are fixed defaults
            device: Target device for the encoder's parameters. Accepted
                for API compatibility with the rest of the Yv model
                stack; submodules are constructed on CPU and the parent
                model performs a single .to(device, dtype) after this
                __init__ returns.
            dtype: Target dtype for the encoder's parameters. Same
                caveat as ``device``.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        # Recorded for downstream inspection. Submodules are built on
        # CPU; the top-level YvModel moves the whole tree in one .to().
        self._init_device = device
        self._init_dtype = dtype
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
                num_layers=3,
                enable_nested_tensor=False
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
    
    def forward(self, document_input: Dict[str, Any]) -> Dict[str, Any]:
        """Encode the provided document payload into multimodal features.

        Main entry point for document encoding. Processes text, layout, table,
        and handwriting inputs through all submodules defined in the encoder.
        Returns a dictionary containing the fused document representation
        along with auxiliary outputs from each pipeline stage.

        Args:
            document_input (Dict[str, Any]): Document expressed as a dictionary
                with the following optional keys:
                - 'text' (str): Raw text content.
                - 'layout' (torch.Tensor, optional): Bounding-box geometry
                  tensor of shape ``[N, 8]``. Defaults to full-page layout.
                - 'tables' (Union[torch.Tensor, Dict], optional): Table cell
                  features or a dict with 'cells'/'features' key.
                - 'handwriting' (Union[torch.Tensor, Dict], optional): Stroke
                  data ``[num_strokes, seq_len, 3]`` or a dict with
                  'strokes'/'image' keys.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - ``features``: Fused document tensor ``(batch, 1, hidden_size)``.
                - ``text_features``: Raw text representation
                  ``(batch, hidden_size)``.
                - ``language``: Language detection results
                  (logits, lang_id, script).
                - ``layout``: Layout analysis results
                  (features, reading_order, type_logits, geometric_relations).
                - ``tables``: Table understanding output (or None).
                - ``handwriting``: Handwriting recognition output (or None).
                - ``doc_type``: Document-type classification logits
                  ``(batch, 20)``.
                - ``entities``: Named-entity logits ``(batch, 50)``.
                - ``key_value``: Key-value pair probabilities ``(batch, 1)``.

        Note:
            Returns a zero-feature dict if ``document_input`` is ``None`` or
            contains no text. All input tensors are moved to the model device.
        """
        device = next(self.parameters()).device

        if document_input is None:
            return {
                'features': torch.zeros(1, 1, self.cfg.hidden_size, device=device),
                'text_features': torch.zeros(1, self.cfg.hidden_size, device=device),
                'language': None,
                'layout': None,
                'tables': None,
                'handwriting': None,
                'doc_type': None,
                'entities': None,
                'key_value': None,
            }

        # ---- extract inputs -------------------------------------------------
        text = document_input.get('text', '')
        layout_input = document_input.get('layout', None)
        tables_input = document_input.get('tables', None)
        handwriting_input = document_input.get('handwriting', None)

        if not text:
            return {
                'features': torch.zeros(1, 1, self.cfg.hidden_size, device=device),
                'text_features': torch.zeros(1, self.cfg.hidden_size, device=device),
                'language': None,
                'layout': None,
                'tables': None,
                'handwriting': None,
                'doc_type': None,
                'entities': None,
                'key_value': None,
            }

        # =====================================================================
        #  TEXT ENCODING
        # =====================================================================
        text_tokens = self._tokenize_text(text)
        if text_tokens.dim() == 1:
            text_tokens = text_tokens.unsqueeze(0)
        text_tokens = text_tokens.to(device)

        # Embedding + positional encoding.
        embeddings = self.text_encoder['embedding'](text_tokens)
        positions = torch.arange(text_tokens.size(1), device=device)
        pos_enc = self.text_encoder['positional_encoding'](
            positions.unsqueeze(0).expand(text_tokens.size(0), -1)
        )
        text_seq = embeddings + pos_enc                              # (B, L, H)
        text_seq = self.text_encoder['layer_norm'](text_seq)
        text_seq = self.text_encoder['dropout'](text_seq)

        # ---- language detection (1/2: before script encoding) ---------------
        lang_logits = self.text_encoder['language_detector'](
            text_seq.mean(dim=1)
        )                                                            # (B, 100)
        lang_id = int(lang_logits.argmax(dim=-1)[0].item())

        # ---- script-specific encoding --------------------------------------
        if lang_id >= 80:
            text_seq = self.text_encoder['script_encoders']['arabic'](text_seq)
            script = 'arabic'
        elif lang_id >= 50:
            text_seq = self.text_encoder['script_encoders']['chinese'](text_seq)
            script = 'chinese'
        else:
            text_seq = self.text_encoder['script_encoders']['latin'](text_seq)
            script = 'latin'

        text_features = text_seq.mean(dim=1)                         # (B, H)

        # =====================================================================
        #  LAYOUT ENCODING
        # =====================================================================
        layout_embed = self._encode_layout(layout_input).to(device)  # (N, H//4)

        # ---- reading order & layout type classification ---------------------
        reading_order = self.layout_encoder['reading_order'](
            layout_embed
        )                                                            # (N, 1)
        layout_type_logits = self.layout_encoder['layout_classifier'](
            layout_embed
        )                                                            # (N, 15)

        # ---- geometric reasoning (pairwise relations) -----------------------
        n_elements = layout_embed.size(0)
        if n_elements > 1:
            idx_i, idx_j = torch.triu_indices(
                n_elements, n_elements, offset=1, device=device
            )
            pairwise_input = torch.cat(
                [layout_embed[idx_i], layout_embed[idx_j]], dim=-1
            )                                                        # (pairs, H//2)
            geometric_relations = self.layout_encoder['geometric_reasoner'](
                pairwise_input
            )                                                        # (pairs, 9)
        else:
            geometric_relations = None

        layout_features = layout_embed.mean(dim=0, keepdim=True)     # (1, H//4)

        # =====================================================================
        #  TABLE UNDERSTANDING
        # =====================================================================
        table_output = None
        if tables_input is not None:
            if isinstance(tables_input, dict):
                table_cells = tables_input.get(
                    'cells', tables_input.get('features', None)
                )
            else:
                # Assume raw tensor of cell features.
                table_cells = tables_input

            if table_cells is not None:
                if table_cells.dim() == 1:
                    table_cells = table_cells.unsqueeze(0)
                table_cells = table_cells.float().to(device)

                # Project to hidden_size if necessary.
                cell_dim = table_cells.size(-1)
                if cell_dim != self.cfg.hidden_size:
                    proj = nn.Linear(
                        cell_dim, self.cfg.hidden_size, device=device,
                        dtype=table_cells.dtype,
                    )
                    table_cells = proj(table_cells)

                # Structure detection.
                row_scores = self.table_understanding['structure_detector'][
                    'row_detector'
                ](table_cells)                                       # (T, 1)
                col_scores = self.table_understanding['structure_detector'][
                    'column_detector'
                ](table_cells)                                       # (T, 1)
                cell_types = self.table_understanding['structure_detector'][
                    'cell_classifier'
                ](table_cells)                                       # (T, 6)

                # Content analysis.
                data_types = self.table_understanding['content_analyzer'][
                    'data_type_classifier'
                ](table_cells)                                       # (T, 8)
                numerical_props = self.table_understanding[
                    'content_analyzer'
                ]['numerical_analyzer'](table_cells)                 # (T, 4)
                semantic_cells = self.table_understanding[
                    'content_analyzer'
                ]['semantic_encoder'](table_cells)                   # (T, H)

                # Table QA: pool table cells and concat with text.
                table_repr = semantic_cells.mean(
                    dim=0, keepdim=True
                )                                                    # (1, H)
                table_text = torch.cat(
                    [table_repr, text_features], dim=-1
                )                                                    # (1, 2H)
                table_qa_out = self.table_understanding['table_qa'](
                    table_text
                )                                                    # (1, H)

                table_output = {
                    'row_scores': row_scores,
                    'col_scores': col_scores,
                    'cell_types': cell_types,
                    'data_types': data_types,
                    'numerical_props': numerical_props,
                    'qa_repr': table_qa_out,
                }

        # =====================================================================
        #  HANDWRITING RECOGNITION
        # =====================================================================
        hw_output = None
        hw_style_input = None  # 256-dim tensor for handwriting_proj
        if handwriting_input is not None:
            if isinstance(handwriting_input, dict):
                strokes = handwriting_input.get('strokes', None)
                hw_image = handwriting_input.get('image', None)
            else:
                strokes = handwriting_input
                hw_image = None

            hw_output = {}

            if strokes is not None:
                if strokes.dim() == 2:
                    strokes = strokes.unsqueeze(0)                  # (1, S, 3)
                strokes = strokes.float().to(device)

                # Stroke encoder (bidirectional LSTM).
                stroke_hidden, _ = self.handwriting_recognition[
                    'stroke_encoder'
                ](strokes)                                           # (1, S, 256)

                stroke_repr = stroke_hidden.mean(dim=1)              # (1, 256)

                # Character / word recognition.
                char_logits = self.handwriting_recognition[
                    'char_recognizer'
                ](stroke_repr)                                       # (1, 10000)
                word_logits = self.handwriting_recognition[
                    'word_recognizer'
                ](stroke_repr)                                       # (1, 1000)

                # Style analysis.
                style_out = self.handwriting_recognition[
                    'style_analyzer'
                ](stroke_repr)                                       # (1, 20)

                # Store the 256-dim stroke_repr for handwriting_proj.
                # (style_analyzer consumes 256-dim, so we pass the same
                #  intermediate to handwriting_proj.)
                hw_style_input = stroke_repr                         # (1, 256)

                hw_output.update({
                    'char_logits': char_logits,
                    'word_logits': word_logits,
                    'style_features': style_out,
                })

            if hw_image is not None:
                if hw_image.dim() == 2:
                    hw_image = hw_image.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
                hw_image = hw_image.float().to(device)
                line_boundaries = self.handwriting_recognition[
                    'line_segmenter'
                ](hw_image)                                          # (1, 1, H, W)
                hw_output['line_boundaries'] = line_boundaries

        # =====================================================================
        #  DOCUMENT FUSION
        # =====================================================================
        # Self-attention over text features (dimension = hidden_size).
        attn_output, _ = self.doc_fusion['text_layout_attention'](
            text_features.unsqueeze(1),
            text_features.unsqueeze(1),
            text_features.unsqueeze(1),
        )
        text_context = attn_output.squeeze(1)                        # (B, H)

        # Hierarchical encoding over text context.
        hierarchical = self.doc_fusion['hierarchy_encoder'](
            text_context.unsqueeze(1)
        ).squeeze(1)                                                 # (B, H)

        # Fuse with layout features (final_fusion expects H + H//4 input).
        fusion_input = torch.cat(
            [hierarchical, layout_features.expand(text_features.size(0), -1)],
            dim=-1,
        )                                                            # (B, H + H//4)
        doc_features = self.doc_fusion['final_fusion'](fusion_input) # (B, H)

        # ---- doc-type classification & extraction heads ---------------------
        doc_type_logits = self.doc_fusion['doc_type_classifier'](
            doc_features
        )                                                            # (B, 20)

        entity_logits = self.doc_fusion['extraction_heads'][
            'entity_extractor'
        ](doc_features)                                              # (B, 50)

        kv_concat = torch.cat([doc_features, doc_features], dim=-1) # (B, 2H)
        kv_probs = self.doc_fusion['extraction_heads'][
            'key_value_extractor'
        ](kv_concat)                                                 # (B, 1)

        # =====================================================================
        #  FINAL PROJECTION (multi-task)
        # =====================================================================
        main_proj = self.final_proj['main_projection'](doc_features) # (B, H)
        table_proj = self.final_proj['table_proj'](doc_features)     # (B, H//4)
        layout_proj = self.final_proj['layout_proj'](
            layout_features.squeeze(0)
        )                                                            # (B, H//4)

        # Handwriting projection uses the 256-dim stroke representation.
        if hw_style_input is not None:
            handwriting_proj = self.final_proj['handwriting_proj'](
                hw_style_input
            )                                                        # (1, H//4)
            # Broadcast to match batch size if needed.
            if handwriting_proj.size(0) != main_proj.size(0):
                handwriting_proj = handwriting_proj.expand(
                    main_proj.size(0), -1
                )
        else:
            handwriting_proj = torch.zeros(
                main_proj.size(0), self.cfg.hidden_size // 4,
                device=device,
            )

        all_proj = torch.cat(
            [main_proj, table_proj, handwriting_proj, layout_proj], dim=-1
        )                                                            # (B, H + 3*H//4)
        fused_features = self.final_proj['task_integration'](
            all_proj
        )                                                            # (B, H)

        # =====================================================================
        #  ASSEMBLE RESULT
        # =====================================================================
        return {
            'features': fused_features.unsqueeze(1),
            'text_features': text_features,
            'language': {
                'logits': lang_logits,
                'lang_id': lang_id,
                'script': script,
            },
            'layout': {
                'features': layout_features,
                'reading_order': reading_order,
                'layout_type_logits': layout_type_logits,
                'geometric_relations': geometric_relations,
            },
            'tables': table_output,
            'handwriting': hw_output,
            'doc_type': doc_type_logits,
            'entities': entity_logits,
            'key_value': kv_probs,
        }
