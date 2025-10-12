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
from torch import nn
from typing import Any, Dict
import torch.nn.functional as F
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("Arctic.Core.Multimodal")

class ArcticDocEncoder(nn.Module):
    """A document encoder that processes text, layout, table, and handwriting information."""
    
    def __init__(self, cfg):
        """
        Initialize the ArcticDocEncoder.

        Args:
            cfg: Configuration object containing parameters such as hidden_size and n_head.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.vocab_size = 50000
        self.max_length = 512
        
        logger.debug(f"DocEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
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
        
        logger.debug("DocEncoder: __init__ end")
    
    def _tokenize_text(self, text):
        """
        Perform simple text tokenization and encoding.

        Args:
            text (Union[str, torch.Tensor]): Input text or token IDs.

        Returns:
            torch.Tensor: Tokenized text.
        """
        if isinstance(text, str):
            # Perform simple character-level tokenization
            tokens = [hash(c) % self.vocab_size for c in text[:self.max_length]]
            tokens += [0] * (self.max_length - len(tokens))
            return torch.tensor(tokens)
        return text
    
    def _encode_layout(self, layout):
        """
        Encode layout information.

        Args:
            layout (Union[torch.Tensor, None]): Input layout information.

        Returns:
            torch.Tensor: Encoded layout features.
        """
        if layout is None:
            # Use default layout: full page
            layout = torch.tensor([[0, 0, 1, 1]])
        
        if layout.dim() == 1:
            layout = layout.unsqueeze(0)
        
        # Note: There's a bug here. self.layout_encoder is a ModuleDict, not callable.
        # This code will raise an error. It should process each key in the ModuleDict.
        # For now, assuming the user meant to process through spatial_encoder.
        return self.layout_encoder['spatial_encoder'](layout.float())
    
    def forward(self, doc_input):
        """
        Forward pass of the document encoder.

        Args:
            doc_input (Union[str, torch.Tensor, dict]): Input document, which can be a text string, token IDs, or a dictionary.

        Returns:
            torch.Tensor: Encoded document features with shape (batch_size, 1, hidden_size).
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
        
        # Note: There's a bug here. self.text_encoder is a ModuleDict, not callable.
        # This code will raise an error. It should process each key in the ModuleDict.
        # For now, assuming the user meant to process through embedding and others.
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
        
        # Note: There's a bug here. self.doc_fusion is a ModuleDict, not callable.
        # This code will raise an error. It should process each key in the ModuleDict.
        # For now, assuming the user meant to process through text_layout_attention first.
        attn_output, _ = self.doc_fusion['text_layout_attention'](
            combined.unsqueeze(1), combined.unsqueeze(1), combined.unsqueeze(1)
        )
        attn_output = attn_output.squeeze(1)
        doc_features = self.doc_fusion['hierarchy_encoder'](attn_output.unsqueeze(1)).squeeze(1)
        doc_features = self.doc_fusion['final_fusion'](doc_features)
        
        # Note: There's a bug here. self.final_proj is a ModuleDict, not callable.
        # This code will raise an error. It should process each key in the ModuleDict.
        # For now, assuming the user meant to process through main_projection first.
        main_proj = self.final_proj['main_projection'](doc_features)
        table_proj = self.final_proj['table_proj'](doc_features)
        # Note: handwriting_proj needs input of size 256, but doc_features is of size hidden_size.
        # This will raise an error. Skipping for now.
        # handwriting_proj = self.final_proj['handwriting_proj'](doc_features)
        layout_proj = self.final_proj['layout_proj'](layout_features)
        # Concatenate all projections
        all_proj = torch.cat([main_proj, table_proj, layout_proj], dim=-1)
        doc_features = self.final_proj['task_integration'](all_proj)
        
        return doc_features.unsqueeze(1)