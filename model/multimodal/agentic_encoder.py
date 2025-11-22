#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

"""Compatibility module that bridges legacy Arctic agent encoder pipelines.

The shim keeps historical integrations operational while newer flows migrate to
PiscesAgent-backed implementations. It exposes the original encoder interface,
yet dispatches all multimodal processing to :class:`ArcticAgentic`, allowing
callers that still depend on ``ArcticAgenticEncoder`` to reuse their existing
data marshaling logic without modification.
"""

import torch
from torch import nn
from .agentic import ArcticAgentic
from typing import Any, Dict, List, Tuple
from .types import ArcticAgenticObservation

class ArcticAgenticEncoder(nn.Module):
    """Wrapper that preserves the legacy Arctic agent encoder interface.

    The encoder delegates multimodal encoding duties to :class:`ArcticAgentic` so
    that older integration points can continue to rely on the pre-PiscesAgent
    contract.

    Attributes:
        enabled (bool): Flag indicating whether the encoder is active.
        cfg: Configuration namespace with model hyperparameters.
        pisces_agentic (ArcticAgentic): Underlying agent implementation handling
            observation processing and reasoning.
    """

    def __init__(self, cfg: Any) -> None:
        """Instantiate the compatibility encoder.

        Args:
            cfg (Any): Configuration object providing at least ``vocab_size`` and
                ``hidden_size`` attributes. Downstream components are expected to
                populate additional encoder submodules (for example ``obs_text_encoder``)
                referenced within this class.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.pisces_agentic = ArcticAgentic(cfg)
        
    def forward(self, agent_input):
        """Delegate a raw observation to the underlying Pisces agent.

        Args:
            agent_input: Observation payload accepted by
                :meth:`ArcticAgentic.process_observation`.

        Returns:
            torch.Tensor: Encoded feature tensor produced by the delegated agent.
        """
        return self.pisces_agentic.process_observation(agent_input)
    
    def encode_observation(self, observation):
        """Convert a single multimodal observation into a hidden embedding.

        Args:
            observation (Dict[str, Any]): Dictionary describing the observation.
                The ``modality`` key specifies the encoder pathway
                (``"text"``, ``"image"``, ``"audio"``, or ``"tool_result"``) and
                ``content`` carries the raw payload.

        Returns:
            torch.Tensor: Encoded observation tensor with shape ``[1, 1, hidden]``.

        Raises:
            KeyError: If the observation dict is missing the ``modality`` field.
        """
        if observation['modality'] == "text":
            # Normalize textual content into hashed token identifiers.
            tokens = observation['content']
            if isinstance(tokens, str):
                # Derive a deterministic token ID by hashing the string input.
                tokens = torch.tensor([hash(tokens) % self.cfg.vocab_size])
            return self.obs_text_encoder(tokens)
        
        elif observation['modality'] == "image":
            # Route image payloads through the vision encoder pathway.
            return self.obs_image_encoder(observation['content'])
        
        elif observation['modality'] == "audio":
            # Transform audio inputs using the audio encoder stack.
            return self.obs_audio_encoder(observation['content'])
        
        elif observation['modality'] == "tool_result":
            # Serialize tool outputs and reuse the textual encoder branch.
            result_str = str(observation['content'])
            tokens = torch.tensor([hash(result_str) % self.cfg.vocab_size])
            return self.obs_text_encoder(tokens)
        
        else:
            # Provide a zero tensor when the modality is not supported.
            return torch.zeros(1, 1, self.cfg.hidden_size)
    
    def encode_memory(self, memory_data):
        """Aggregate encoded memories for observations, actions, and reflections.

        Args:
            memory_data (Dict[str, List]): Container with optional ``observations``,
                ``actions``, and ``reflections`` sequences.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Encoded memory tensors
            for observations, actions, and reflections respectively. Each tensor has
            shape ``[1, 1, hidden]`` when no historical entries exist.
        """
        # Encode observation history into a recurrent memory representation.
        obs_features = []
        for obs in memory_data.get('observations', []):
            # Encode each stored observation independently.
            obs_feat = self.encode_observation(obs)
            obs_features.append(obs_feat)
        
        if obs_features:
            obs_tensor = torch.stack(obs_features, dim=1)
            obs_memory, _ = self.memory_encoder['obs_memory'](obs_tensor)
        else:
            obs_memory = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Transform action history into latent memory features.
        action_features = []
        for action in memory_data.get('actions', []):
            action_str = str(action)
            tokens = torch.tensor([hash(action_str) % self.cfg.vocab_size])
            action_feat = self.obs_text_encoder(tokens)
            action_features.append(action_feat)
        
        if action_features:
            action_tensor = torch.stack(action_features, dim=1)
            action_memory, _ = self.memory_encoder['action_memory'](action_tensor)
        else:
            action_memory = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Encode textual reflections to capture agent reasoning traces.
        reflection_features = []
        for reflection in memory_data.get('reflections', []):
            tokens = torch.tensor([hash(str(reflection)) % self.cfg.vocab_size])
            ref_feat = self.obs_text_encoder(tokens)
            reflection_features.append(ref_feat)
        
        if reflection_features:
            ref_tensor = torch.stack(reflection_features, dim=1)
            reflection_memory, _ = self.memory_encoder['reflection_memory'](ref_tensor)
        else:
            reflection_memory = torch.zeros(1, 1, self.cfg.hidden_size)
        
        return obs_memory, action_memory, reflection_memory
    
    def forward(self, agent_input):
        """Produce a comprehensive agent embedding from multimodal context.

        Args:
            agent_input (Dict[str, Any]): Dictionary that may include ``observations``,
                ``actions``, ``reflections``, ``current_state``, and ``task_context``.

        Returns:
            torch.Tensor: Aggregated feature tensor after attention fusion and
            final projection. The output has shape ``[1, 1, hidden]``.
        """
        # Encode current observation batch for downstream fusion.
        obs_features = []
        for obs in agent_input.get('observations', []):
            obs_feat = self.encode_observation(obs)
            obs_features.append(obs_feat)
        
        # Collect historical memory inputs before fusion.
        memory_data = {
            'observations': agent_input.get('observations', []),
            'actions': agent_input.get('actions', []),
            'reflections': agent_input.get('reflections', [])
        }
        obs_memory, action_memory, reflection_memory = self.encode_memory(memory_data)
        
        # Derive the latent representation of the current agent state if available.
        if 'current_state' in agent_input:
            state_feat = self.state_encoder(agent_input['current_state'])
        else:
            state_feat = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Embed the task context description for condition-aware actions.
        if 'task_context' in agent_input:
            task_tokens = torch.tensor([hash(str(agent_input['task_context'])) % self.cfg.vocab_size])
            task_feat = self.obs_text_encoder(task_tokens)
        else:
            task_feat = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Concatenate the most recent memories and contextual features.
        combined_features = torch.cat([
            obs_memory[:, -1:],  # Latest observation
            action_memory[:, -1:],  # Latest action
            reflection_memory[:, -1:],  # Latest reflection
            state_feat,
            task_feat
        ], dim=1)
        
        # Apply self-attention to integrate cross-modal signals.
        attended_features, _ = self.agent_attention(combined_features, combined_features, combined_features)
        
        # Generate action logits, parameters, and confidence estimates.
        action_logits = self.action_type_head(attended_features.mean(dim=1))
        action_params = self.action_param_head(attended_features.mean(dim=1))
        confidence = torch.sigmoid(self.confidence_head(attended_features.mean(dim=1)))
        
        # Form the final agent embedding prior to projection.
        all_features = torch.cat([
            attended_features.mean(dim=1),
            action_params,
            confidence
        ], dim=-1)
        
        return self.final_proj(all_features).unsqueeze(1)
