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
from torch import nn
from .agent import ArcticAgent
from typing import Any, Dict, Union
from .types import AgentObservation

class ArcticAgentEncoder(nn.Module):
    """
    Legacy agent encoder maintained for backward compatibility. This encoder is now replaced by PiscesAgent 
    for comprehensive agent functionality.
    """
    def __init__(self, cfg):
        """
        Initialize the ArcticAgentEncoder.

        Args:
            cfg: Configuration object containing necessary parameters such as vocab_size and hidden_size.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.pisces_agent = ArcticAgent(cfg)
        
    def forward(self, agent_input):
        """
        Perform a forward pass using PiscesAgent.

        Args:
            agent_input: Input data to be processed by the PiscesAgent.

        Returns:
            The result of processing the input through the PiscesAgent.
        """
        return self.pisces_agent.process_observation(agent_input)
    
    def encode_observation(self, observation):
        """
        Encode multimodal observations.

        Args:
            observation (dict): A dictionary containing the observation data with a 'modality' key indicating 
                                the type of observation ('text', 'image', 'audio', 'tool_result', etc.).

        Returns:
            torch.Tensor: Encoded observation tensor of shape (1, 1, cfg.hidden_size).
        """
        if observation['modality'] == "text":
            # Extract tokens from text observation
            tokens = observation['content']
            if isinstance(tokens, str):
                # Convert string input to token tensor using hash function
                tokens = torch.tensor([hash(tokens) % self.cfg.vocab_size])
            return self.obs_text_encoder(tokens)
        
        elif observation['modality'] == "image":
            # Encode image observation
            return self.obs_image_encoder(observation['content'])
        
        elif observation['modality'] == "audio":
            # Encode audio observation
            return self.obs_audio_encoder(observation['content'])
        
        elif observation['modality'] == "tool_result":
            # Convert tool result to string and encode as text
            result_str = str(observation['content'])
            tokens = torch.tensor([hash(result_str) % self.cfg.vocab_size])
            return self.obs_text_encoder(tokens)
        
        else:
            # Return zero tensor for unsupported modalities
            return torch.zeros(1, 1, self.cfg.hidden_size)
    
    def encode_memory(self, memory_data):
        """
        Encode agent memory, including observations, actions, and reflections.

        Args:
            memory_data (dict): A dictionary containing memory data with keys 'observations', 'actions', 
                               and 'reflections'. Each key maps to a list of corresponding data.

        Returns:
            tuple: A tuple containing three tensors representing encoded observation memory, 
                   action memory, and reflection memory, each of shape (1, 1, cfg.hidden_size).
        """
        # Encode observations memory
        obs_features = []
        for obs in memory_data.get('observations', []):
            obs_feat = self.encode_observation(obs)
            obs_features.append(obs_feat)
        
        if obs_features:
            obs_tensor = torch.stack(obs_features, dim=1)
            obs_memory, _ = self.memory_encoder['obs_memory'](obs_tensor)
        else:
            obs_memory = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Encode actions memory
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
        
        # Encode reflections memory
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
        """
        Perform a forward pass of the comprehensive AgentEncoder.

        Args:
            agent_input (dict): Dictionary containing:
                - observations: List of agent observations
                - actions: List of agent actions
                - reflections: List of agent reflections
                - current_state: Current agent state tensor
                - task_context: Current task description

        Returns:
            torch.Tensor: Comprehensive agent features including state, memory, and predictions, 
                          with shape (1, 1, ...).
        """
        # Encode observations
        obs_features = []
        for obs in agent_input.get('observations', []):
            obs_feat = self.encode_observation(obs)
            obs_features.append(obs_feat)
        
        # Encode memory
        memory_data = {
            'observations': agent_input.get('observations', []),
            'actions': agent_input.get('actions', []),
            'reflections': agent_input.get('reflections', [])
        }
        obs_memory, action_memory, reflection_memory = self.encode_memory(memory_data)
        
        # Encode current state
        if 'current_state' in agent_input:
            state_feat = self.state_encoder(agent_input['current_state'])
        else:
            state_feat = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Encode task context
        if 'task_context' in agent_input:
            task_tokens = torch.tensor([hash(str(agent_input['task_context'])) % self.cfg.vocab_size])
            task_feat = self.obs_text_encoder(task_tokens)
        else:
            task_feat = torch.zeros(1, 1, self.cfg.hidden_size)
        
        # Combine all features using concatenation
        combined_features = torch.cat([
            obs_memory[:, -1:],  # Latest observation
            action_memory[:, -1:],  # Latest action
            reflection_memory[:, -1:],  # Latest reflection
            state_feat,
            task_feat
        ], dim=1)
        
        # Apply cross-modal attention
        attended_features, _ = self.agent_attention(combined_features, combined_features, combined_features)
        
        # Predict action type and parameters
        action_logits = self.action_type_head(attended_features.mean(dim=1))
        action_params = self.action_param_head(attended_features.mean(dim=1))
        confidence = torch.sigmoid(self.confidence_head(attended_features.mean(dim=1)))
        
        # Perform final comprehensive encoding
        all_features = torch.cat([
            attended_features.mean(dim=1),
            action_params,
            confidence
        ], dim=-1)
        
        return self.final_proj(all_features).unsqueeze(1)