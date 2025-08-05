#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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
import uuid
import json
import torch
import numpy as np
from torch import nn
from PIL import Image
from enum import Enum
from datetime import datetime
import torch.nn.functional as F
from dataclasses import dataclass
from utils.log import DEBUG, ERROR
from einops import rearrange, repeat
from transformers import ASTFeatureExtractor
from typing import Optional, Tuple, List, Dict, Any, Callable, Union

class VisionEncoder(nn.Module):
    """
    Unified Vision Encoder with NaViT native resolution support and SigLIP-style architecture.
    Supports arbitrary image resolutions and aspect ratios with efficient patch processing.
    """
    def __init__(self, cfg):
        """
        Initialize the VisionEncoder with NaViT-style native resolution support.

        Args:
            cfg: Configuration object containing parameters such as hidden size, etc.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.patch_size = 14
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.n_head
        self.num_layers = cfg.n_layer
        
        DEBUG(f"VisionEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Image preprocessing: register mean and std for normalization (ImageNet stats)
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Patch embedding for any resolution (NaViT-style)
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # 2D positional embedding with native resolution support (up to 1024x1024)
        max_patches_h = 1024 // self.patch_size
        max_patches_w = 1024 // self.patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches_h * max_patches_w, self.hidden_size))
        
        # Classification token (SigLIP-style)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))
        
        # Enhanced Transformer encoder with pre-norm architecture
        self.transformer = nn.ModuleDict({
            'layers': nn.ModuleList([
                nn.ModuleDict({
                    'norm1': nn.LayerNorm(self.hidden_size),
                    'attn': nn.MultiheadAttention(
                        embed_dim=self.hidden_size,
                        num_heads=self.num_heads,
                        batch_first=True
                    ),
                    'norm2': nn.LayerNorm(self.hidden_size),
                    'mlp': nn.Sequential(
                        nn.Linear(self.hidden_size, 4 * self.hidden_size),
                        nn.GELU(),
                        nn.Linear(4 * self.hidden_size, self.hidden_size)
                    )
                }) for _ in range(self.num_layers)
            ]),
            'norm': nn.LayerNorm(self.hidden_size)
        })
        
        # Final projection layer
        self.proj = nn.Linear(self.hidden_size, cfg.hidden_size)
        
        # Object detection and coordinate marking
        self.num_classes = 1000  # Common object classes
        self.num_anchors = 9  # Anchor boxes per location
        
        # Detection head for bounding box regression and classification
        self.detection_head = nn.ModuleDict({
            'bbox_regressor': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 4 * self.num_anchors)  # [x, y, w, h] for each anchor
            ),
            'classifier': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_classes * self.num_anchors)
            ),
            'objectness': nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_anchors)  # Objectness score
            )
        })
        
        # Coordinate marker for precise location
        self.coordinate_marker = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [x_center, y_center] normalized coordinates
        )
        
        DEBUG("VisionEncoder: __init__ end")
    
    def process_image(self, image_path, target_size=None):
        """
        Process an image from the given path with native resolution support.
        
        Args:
            image_path (str): Path to the image file.
            target_size (tuple, optional): Target size (H, W) for resizing. If None, use native resolution.
            
        Returns:
            torch.Tensor: Processed image tensor, or None if an error occurs.
        """
        DEBUG(f"Processing image: {image_path}")
        try:
            # Read images using PIL and convert them into tensors
            image = Image.open(image_path).convert('RGB')
            
            # Native resolution processing - no forced resizing
            if target_size is not None:
                image = image.resize(target_size, Image.LANCZOS)
            
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            image = (image - self.mean) / self.std
            return image
        except Exception as e:
            ERROR(f"Image processing error: {e}")
            return None

    def interpolate_pos_encoding(self, pos_embed, h, w):
        """Interpolate positional encoding for arbitrary image sizes (NaViT-style)."""
        npatch = h * w
        N = pos_embed.shape[1] - 1  # Remove cls token
        
        if npatch == N:
            return pos_embed
        
        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]
        
        dim = self.hidden_size
        w0 = w
        h0 = h
        
        # Reshape and interpolate
        sqrt_N = int(math.sqrt(N))
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode='bicubic',
            align_corners=False
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, h0 * w0, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values):
        """
        Forward pass of the unified VisionEncoder with NaViT native resolution support and SigLIP-style processing.
        
        Args:
            pixel_values (torch.Tensor): Input image pixel values of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Encoded image features of shape (B, 1, hidden_size).
        """
        if pixel_values is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=self.proj.weight.device)
        
        # Image preprocessing: normalize pixel values (SigLIP-style)
        x = (pixel_values - self.mean) / self.std
        
        # Native resolution processing (NaViT-style)
        B, C, H, W = x.shape
        patch_size = self.patch_size
        
        # Dynamic patch embedding for any resolution
        x = self.patch_embed(x)  # (B, hidden_size, H//patch_size, W//patch_size)
        h_patches, w_patches = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # Add classification token (SigLIP-style)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Dynamic position embedding interpolation (NaViT-style)
        pos_embed = self.interpolate_pos_encoding(self.pos_embed, h_patches, w_patches)
        x = x + pos_embed
        
        # Enhanced Transformer encoder (SigLIP + NaViT fusion)
        for layer in self.transformer['layers']:
            # Pre-norm self-attention
            x_norm = layer['norm1'](x)
            attn_out = layer['attn'](x_norm, x_norm, x_norm)[0]
            x = x + attn_out
            
            # Pre-norm MLP
            mlp_out = layer['mlp'](layer['norm2'](x))
            x = x + mlp_out
        
        # Final normalization
        x = self.transformer['norm'](x)
        
        # Split classification token and patch features
        cls_token = x[:, :1]  # (B, 1, hidden_size)
        patch_features = x[:, 1:]  # (B, num_patches, hidden_size)
        
        # Global average pooling (SigLIP-style) and projection
        pooled_features = patch_features.mean(dim=1)  # Average over all patches
        x = self.proj(pooled_features)
        
        # Coordinate marking and object detection
        
        # Detection outputs
        detection_results = {
            'features': x.unsqueeze(1),  # (B, 1, hidden_size) for downstream tasks
            'bbox_coords': None,
            'object_classes': None,
            'confidence_scores': None,
            'coordinate_markers': None
        }
        
        if self.training or hasattr(self, '_enable_detection'):
            # Generate detection predictions
            batch_size, num_patches, _ = patch_features.shape
            h_patches = int(math.sqrt(num_patches))
            w_patches = h_patches
            
            # Spatial reshape for detection
            patch_features_2d = patch_features.view(batch_size, h_patches, w_patches, -1)
            
            # Detection head forward pass
            bbox_pred = self.detection_head['bbox_regressor'](patch_features)
            class_pred = self.detection_head['classifier'](patch_features)
            objectness_pred = self.detection_head['objectness'](patch_features)
            
            # Coordinate markers for precise location
            coord_markers = self.coordinate_marker(patch_features)
            
            # Process detection results
            detection_results.update({
                'bbox_coords': bbox_pred.view(batch_size, num_patches, self.num_anchors, 4),
                'object_classes': class_pred.view(batch_size, num_patches, self.num_anchors, self.num_classes),
                'confidence_scores': torch.sigmoid(objectness_pred),
                'coordinate_markers': coord_markers.view(batch_size, num_patches, 2),
                'spatial_shape': (h_patches, w_patches)
            })
        
        return detection_results
    
    def convert_patch_to_image_coords(self, patch_coords, image_size, patch_size=14):
        """
        将patch坐标转换为实际图像坐标
        
        Args:
            patch_coords: [x_patch, y_patch] patch网格坐标
            image_size: (H, W) 原始图像尺寸
            patch_size: patch大小，默认14
            
        Returns:
            [x_image, y_image] 实际图像坐标
        """
        h, w = image_size
        h_patches = h // patch_size
        w_patches = w // patch_size
        
        x_patch, y_patch = patch_coords
        x_image = (x_patch + 0.5) * patch_size
        y_image = (y_patch + 0.5) * patch_size
        
        return [min(x_image, w-1), min(y_image, h-1)]
    
    def get_object_locations(self, detection_results, image_size, confidence_threshold=0.5):
        """
        获取检测到的物体位置坐标
        
        Args:
            detection_results: VisionEncoder的输出
            image_size: (H, W) 原始图像尺寸
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Dict]: 包含坐标和类别的物体列表
        """
        if not detection_results.get('bbox_coords'):
            return []
        
        bbox_coords = detection_results['bbox_coords']
        object_classes = detection_results['object_classes']
        confidence_scores = detection_results['confidence_scores']
        
        objects = []
        batch_size, num_patches, num_anchors, _ = bbox_coords.shape
        h_patches = int(math.sqrt(num_patches))
        
        for b in range(batch_size):
            for p in range(num_patches):
                patch_y = p // h_patches
                patch_x = p % h_patches
                
                for a in range(num_anchors):
                    confidence = confidence_scores[b, p, a].item()
                    if confidence > confidence_threshold:
                        # 获取边界框坐标 [dx, dy, dw, dh]
                        bbox = bbox_coords[b, p, a]
                        
                        # 转换为实际坐标
                        center_x = (patch_x + bbox[0].item()) * self.patch_size
                        center_y = (patch_y + bbox[1].item()) * self.patch_size
                        width = bbox[2].item() * image_size[1]
                        height = bbox[3].item() * image_size[0]
                        
                        # 获取类别
                        class_probs = torch.softmax(object_classes[b, p, a], dim=0)
                        class_id = torch.argmax(class_probs).item()
                        class_name = f"class_{class_id}"  # 实际应用中应映射到类别名称
                        
                        objects.append({
                            'coordinates': [center_x, center_y],
                            'bbox': [center_x - width/2, center_y - height/2, 
                                    center_x + width/2, center_y + height/2],
                            'class': class_name,
                            'confidence': confidence,
                            'patch_coords': [patch_x, patch_y]
                        })
        
        return objects
    
    def enable_detection(self, enable=True):
        """启用/禁用坐标标记功能"""
        self._enable_detection = enable

class AudioEncoder(nn.Module):
    """
    An audio encoder module that processes audio data and encodes it using a convolutional network.
    """
    def __init__(self, cfg):
        """
        Initialize the AudioEncoder.

        Args:
            cfg: Configuration object containing parameters such as hidden size.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        DEBUG(f"AudioEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        self.processor = ASTFeatureExtractor()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=8, stride=8)
        
        self.proj = nn.Sequential(
            nn.Linear(64 * 128, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size)
        )
        
        DEBUG("AudioEncoder: __init__ end")
    
    def process_audio(self, audio_path):
        """
        Process audio from the given path.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            torch.Tensor: Processed audio tensor, or None if an error occurs.
        """
        DEBUG(f"Processing audio: {audio_path}")
        try:
            audio = self.processor(audio=audio_path, return_tensors="pt")
            return audio['input_values'][0]
        except Exception as e:
            ERROR(f"Audio processing error: {e}")
            return None
    
    def forward(self, audio_input):
        """
        Forward pass of the AudioEncoder.

        Args:
            audio_input (dict): Dictionary containing audio input values.

        Returns:
            torch.Tensor: Encoded audio features.
        """
        if audio_input is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=audio_input.device)
        
        x = self.conv1(audio_input['input_values'].unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x.unsqueeze(1)

class DocEncoder(nn.Module):
    """
    A document encoder module that processes document input and projects it to a hidden dimension.
    """
    def __init__(self, cfg):
        """
        Initialize the DocEncoder.

        Args:
            cfg: Configuration object containing parameters such as hidden size.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        DEBUG(f"DocEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        self.doc_proj = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size)
        )
        
        DEBUG("DocEncoder: __init__ end")
    
    def forward(self, doc_input):
        """
        Forward pass of the DocEncoder.

        Args:
            doc_input (dict): Dictionary containing document input IDs.

        Returns:
            torch.Tensor: Encoded document features.
        """
        x = self.doc_proj(doc_input['input_ids'])
        return x.unsqueeze(1)

class VideoEncoder(nn.Module):
    """
    A video encoder module that processes video frames using a vision encoder and performs temporal pooling.
    """
    def __init__(self, cfg):
        """
        Initialize the VideoEncoder.

        Args:
            cfg: Configuration object containing parameters such as hidden size.
        """
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        DEBUG(f"VideoEncoder: __init__ start ({'enabled' if self.enabled else 'disabled'})")
        
        # Simple video encoding using frame-based approach
        self.frame_encoder = VisionEncoder(cfg)
        self.temporal_proj = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size)
        )
        
        # Enhanced temporal modeling
        self.temporal_conv = nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3, padding=1)
        self.temporal_attention = nn.MultiheadAttention(cfg.hidden_size, cfg.n_head, batch_first=True)
        
        DEBUG("VideoEncoder: __init__ end")
    
    def forward(self, video_frames):
        """
        Forward pass of the VideoEncoder.

        Args:
            video_frames (torch.Tensor): Input video frames.

        Returns:
            torch.Tensor: Encoded video features.
        """
        if video_frames is None:
            return torch.zeros(1, 1, self.cfg.hidden_size, device=self.cfg.device)
        
        # Process each frame with vision encoder
        batch_size, num_frames, channels, height, width = video_frames.shape
        frames_flat = video_frames.view(-1, channels, height, width)
        frame_features = self.frame_encoder(frames_flat)
        
        # Reshape back to temporal sequence
        frame_features = frame_features.view(batch_size, num_frames, -1, self.cfg.hidden_size)
        
        # Temporal pooling (simple average)
        video_features = frame_features.mean(dim=2)  # Average across spatial dimensions
        video_features = self.temporal_proj(video_features.mean(dim=1))  # Average across frames
        
        return video_features.unsqueeze(1)

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentAction:
    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""

@dataclass
class AgentObservation:
    modality: str  # "text", "image", "audio", "tool_result"
    content: Any
    metadata: Dict[str, Any]

@dataclass
class AgentMemory:
    observations: List[AgentObservation]
    actions: List[AgentAction]
    reflections: List[str]
    
    def add_observation(self, observation: AgentObservation):
        self.observations.append(observation)
    
    def add_action(self, action: AgentAction):
        self.actions.append(action)
    
    def add_reflection(self, reflection: str):
        self.reflections.append(reflection)
    
    def get_recent_context(self, k: int = 5) -> Dict[str, List]:
        return {
            "recent_observations": self.observations[-k:],
            "recent_actions": self.actions[-k:],
            "recent_reflections": self.reflections[-k:]
        }

class MCPMessageType(Enum):
    """MCP message types for agent communication"""
    OBSERVATION = "observation"
    ACTION = "action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    CAPABILITY_REGISTER = "capability_register"
    HEARTBEAT = "heartbeat"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"

@dataclass
class MCPMessage:
    """Standardized MCP message format"""
    message_type: str
    agent_id: str
    payload: Dict[str, Any]
    timestamp: str
    correlation_id: str = ""
    priority: str = "normal"

class MCPToolRegistry:
    """Registry for managing MCP tools and capabilities"""
    
    def __init__(self, agent_id: str, message_handler: Callable):
        self.agent_id = agent_id
        self.message_handler = message_handler
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, Dict[str, Any]] = {}
    
    async def register_capability(self, name: str, description: str, 
                                  parameters: Dict[str, Any], handler: Callable):
        """Register a new capability"""
        self.capabilities[name] = {
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
    
    async def handle_tool_call(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool call"""
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            return await tool["handler"](**kwargs)
        else:
            raise ValueError(f"Tool {tool_name} not found")

class TreeSearchReasoner:
    """Tree search reasoning module for advanced planning"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = 5
        self.max_width = 3
    
    async def search(self, problem: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform tree search for complex reasoning"""
        # Simplified tree search implementation
        return [{"solution": "tree_search_result", "confidence": 0.8}]

class PiscesReasoner(nn.Module):
    """Pisces L1 reasoning module"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden_size,
                nhead=cfg.n_head,
                dim_feedforward=cfg.hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(2)
        ])
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Forward pass for reasoning"""
        if torch.is_tensor(input_ids):
            x = input_ids
        else:
            x = torch.randn(1, 1, self.cfg.hidden_size)
        
        for layer in self.reasoning_layers:
            x = layer(x)
        
        return {"thinking_logits": x}

class MCPProtocol:
    """MCP protocol implementation for agent communication"""
    
    @staticmethod
    def create_message(
        message_type: MCPMessageType,
        agent_id: str,
        payload: Dict[str, Any],
        correlation_id: str = ""
    ) -> MCPMessage:
        return MCPMessage(
            message_type=message_type.value,
            agent_id=agent_id,
            payload=payload,
            timestamp=datetime.utcnow().isoformat(),
            correlation_id=correlation_id or str(uuid.uuid4())
        )

class PiscesAgent(nn.Module):
    """
    Native Pisces L1 Agent with integrated reasoning, perception, and action capabilities.
    Fully migrated from agent.py to multimodal.py for unified architecture.
    
    Features:
    1. Unified multimodal perception (text, image, audio)
    2. Advanced reasoning with CoT and self-reflection
    3. Tool use and environment interaction
    4. Persistent memory and context management
    5. End-to-end trainable architecture
    6. Full MCP protocol support
    """
    
    def __init__(self, cfg, tokenizer=None, model=None, agent_id: str = None):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        import weakref
        self._model_ref = None  # weak reference placeholder
        self.agent_id = agent_id or str(uuid.uuid4())
        
        # Use provided model or create new one
        if model is not None:
            # Store a weak reference (callable) to avoid registering model as a submodule.
            self._base_model_ref = weakref.ref(model)
            self._model_ref = self._base_model_ref
        else:
            # Stand-alone agent: no linked PiscesModel to avoid cycles.
            self._base_model_ref = None
            self._model_ref = None
            self._reasoner = PiscesReasoner(cfg)
            self._vision_encoder = VisionEncoder(cfg)
            self._audio_encoder = AudioEncoder(cfg)
        
        self.tree_reasoner = TreeSearchReasoner(None, tokenizer) if tokenizer else None

        # MCP Agent infrastructure
        self.memory = AgentMemory([], [], [])
        self.mcp_tools = MCPToolRegistry(self.agent_id, self._handle_mcp_message)
        self.state = AgentState.IDLE
        self.mcp_peers: Dict[str, Dict[str, Any]] = {}
        self.mcp_capabilities: Dict[str, Dict[str, Any]] = {}
        
        # Coordinate marking support
        self._coordinate_detection_enabled = True
        
        # Action prediction heads (now MCP-aware)
        self.action_type_head = nn.Linear(cfg.hidden_size, 10)  # 10 action types
        self.action_param_head = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.confidence_head = nn.Linear(cfg.hidden_size, 1)
        
        # MCP message handlers
        self.mcp_handlers = {
            MCPMessageType.OBSERVATION.value: self._handle_observation,
            MCPMessageType.ACTION.value: self._handle_action,
            MCPMessageType.TOOL_CALL.value: self._handle_tool_call,
            MCPMessageType.TOOL_RESULT.value: self._handle_tool_result,
            MCPMessageType.CAPABILITY_REGISTER.value: self._handle_capability_register,
            MCPMessageType.SYNC_REQUEST.value: self._handle_sync_request,
            MCPMessageType.SYNC_RESPONSE.value: self._handle_sync_response,
        }

    @property
    def base_model(self):
        return self._base_model_ref() if self._base_model_ref else None

    @property
    def reasoner(self):
        if self._base_model_ref:
            return self._base_model_ref().reasoner
        return self._reasoner

    @property
    def vision_encoder(self):
        if self._base_model_ref:
            return self._base_model_ref().vision
        return self._vision_encoder

    @property
    def audio_encoder(self):
        if self._base_model_ref:
            return self._base_model_ref().audio
        return self._audio_encoder

    async def register_capability(self, name: str, description: str, 
                                  parameters: Dict[str, Any], handler: Callable):
        """Register a capability via MCP protocol"""
        await self.mcp_tools.register_capability(name, description, parameters, handler)

    async def _handle_mcp_message(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP messages"""
        handler = self.mcp_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            return MCPProtocol.create_message(
                MCPMessageType.STATE_UPDATE,
                self.agent_id,
                {"error": f"Unknown message type: {message.message_type}"}
            )

    async def _handle_observation(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP observation messages"""
        observation_data = message.payload
        observation = AgentObservation(
            modality=observation_data["modality"],
            content=observation_data["content"],
            metadata=observation_data.get("metadata", {})
        )
        
        self.memory.add_observation(observation)
        obs_embedding = self.process_observation(observation)
        
        return MCPProtocol.create_message(
            MCPMessageType.OBSERVATION,
            self.agent_id,
            {
                "status": "processed",
                "embedding_shape": list(obs_embedding.shape) if torch.is_tensor(obs_embedding) else None,
                "observation_id": str(uuid.uuid4())
            }
        )

    async def _handle_action(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP action messages"""
        action_data = message.payload
        context = {
            "observation": action_data.get("observation"),
            "available_tools": list(self.mcp_capabilities.keys())
        }
        
        action = await self.plan_action(context)
        self.memory.add_action(action)
        
        return MCPProtocol.create_message(
            MCPMessageType.ACTION,
            self.agent_id,
            {
                "action": asdict(action),
                "agent_state": self.state.value
            }
        )

    async def _handle_tool_call(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP tool call messages"""
        tool_data = message.payload
        tool_name = tool_data["tool_name"]
        parameters = tool_data["parameters"]
        
        if tool_name in self.mcp_capabilities:
            handler = self.mcp_capabilities[tool_name]["handler"]
            result = await handler(**parameters)
            
            return MCPProtocol.create_message(
                MCPMessageType.TOOL_RESULT,
                self.agent_id,
                {
                    "tool_name": tool_name,
                    "result": result,
                    "success": True
                }
            )
        else:
            return MCPProtocol.create_message(
                MCPMessageType.TOOL_RESULT,
                self.agent_id,
                {
                    "tool_name": tool_name,
                    "error": f"Tool {tool_name} not found",
                    "success": False
                }
            )

    async def _handle_tool_result(self, message: MCPMessage) -> MCPMessage:
        """Handle MCP tool result messages"""
        result_data = message.payload
        
        observation = AgentObservation(
            modality="tool_result",
            content=result_data,
            metadata={"source": message.agent_id}
        )
        
        self.memory.add_observation(observation)
        
        return MCPProtocol.create_message(
            MCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "tool_result_processed", "result_id": str(uuid.uuid4())}
        )

    async def _handle_capability_register(self, message: MCPMessage) -> MCPMessage:
        """Handle capability registration from other agents"""
        capability_data = message.payload
        capability_name = capability_data["capability"]
        
        if message.agent_id != self.agent_id:
            self.mcp_capabilities[f"{message.agent_id}.{capability_name}"] = capability_data
        
        return MCPProtocol.create_message(
            MCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "capability_registered", "capability": capability_name}
        )

    async def _handle_sync_request(self, message: MCPMessage) -> MCPMessage:
        """Handle synchronization requests"""
        sync_type = message.payload.get("type")
        
        if sync_type == "capability_discovery":
            return MCPProtocol.create_message(
                MCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "capabilities",
                    "capabilities": list(self.mcp_capabilities.keys())
                }
            )
        elif sync_type == "state_sync":
            return MCPProtocol.create_message(
                MCPMessageType.SYNC_RESPONSE,
                self.agent_id,
                {
                    "type": "state",
                    "state": self.state.value,
                    "memory_summary": self._summarize_memory()
                }
            )
        
        return MCPProtocol.create_message(
            MCPMessageType.SYNC_RESPONSE,
            self.agent_id,
            {"error": "Unknown sync type"}
        )

    async def _handle_sync_response(self, message: MCPMessage) -> MCPMessage:
        """Handle synchronization responses"""
        response_data = message.payload
        
        if response_data.get("type") == "capabilities":
            self.mcp_peers[message.agent_id] = {
                "capabilities": response_data.get("capabilities", [])
            }
        
        return MCPProtocol.create_message(
            MCPMessageType.STATE_UPDATE,
            self.agent_id,
            {"status": "sync_completed", "peer_id": message.agent_id}
        )

    def process_observation(self, observation: AgentObservation) -> torch.Tensor:
        """Process multimodal observations into unified representation"""
        if observation.modality == "text":
            if hasattr(self, 'tokenizer') and self.tokenizer:
                tokens = self.tokenizer.encode(str(observation.content), return_tensors="pt")
                if self.base_model:
                    return self.base_model.embed_tokens(tokens)
                else:
                    return torch.randn(1, tokens.size(1), self.cfg.hidden_size)
            else:
                return torch.zeros(1, 1, self.cfg.hidden_size)
        
        elif observation.modality == "image":
            if isinstance(observation.content, str):  # image path
                image_tensor = self.vision_encoder.process_image(observation.content)
                if image_tensor is not None:
                    image_tensor = image_tensor.unsqueeze(0)
                    return self.vision_encoder(image_tensor)
            elif torch.is_tensor(observation.content):
                return self.vision_encoder(observation.content)
        
        elif observation.modality == "audio":
            if isinstance(observation.content, str):  # audio path
                audio_tensor = self.audio_encoder.process_audio(observation.content)
                if audio_tensor is not None:
                    return self.audio_encoder(audio_tensor)
            elif torch.is_tensor(observation.content):
                return self.audio_encoder(observation.content)
        
        elif observation.modality == "tool_result":
            # Convert tool results to embedding
            import json
            result_str = json.dumps(observation.content)
            if hasattr(self, 'tokenizer') and self.tokenizer:
                tokens = self.tokenizer.encode(result_str, return_tensors="pt")
                if self.base_model:
                    return self.base_model.embed_tokens(tokens)
                else:
                    return torch.randn(1, tokens.size(1), self.cfg.hidden_size)
        
        # Fallback to zero tensor
        return torch.zeros(1, 1, self.cfg.hidden_size)

    async def plan_action(self, context: Dict[str, Any]) -> AgentAction:
        """Generate action based on current context and reasoning"""
        # Get recent context from memory
        memory_context = self.memory.get_recent_context(k=3)
        
        # Combine current observation with memory
        combined_input = self._prepare_reasoning_input(context, memory_context)
        
        # Use reasoner for deep thinking
        with torch.no_grad():
            if self.base_model:
                reasoning_output = self.reasoner(combined_input)
                
                # Predict action type and parameters
                action_logits = self.action_type_head(reasoning_output["thinking_logits"][:, -1])
                action_type_idx = torch.argmax(action_logits, dim=-1).item()
            else:
                # Stand-alone mode
                action_logits = self.action_type_head(torch.randn(1, self.cfg.hidden_size))
                action_type_idx = torch.argmax(action_logits, dim=-1).item()
            
            action_types = [
                "respond", "use_tool", "ask_clarification", "reflect", 
                "search_memory", "plan_next", "wait", "verify", 
                "correct_action", "explore"
            ]
            action_type = action_types[action_type_idx]
            
            # Predict confidence
            if self.base_model:
                confidence = torch.sigmoid(self.confidence_head(reasoning_output["thinking_logits"][:, -1])).item()
            else:
                confidence = torch.sigmoid(self.confidence_head(torch.randn(1, self.cfg.hidden_size))).item()
            
            # Generate action parameters
            if self.base_model:
                param_embedding = self.action_param_head(reasoning_output["thinking_logits"][:, -1])
            else:
                param_embedding = self.action_param_head(torch.randn(1, self.cfg.hidden_size))
            action_params = self._decode_action_params(param_embedding, action_type)
            
            return AgentAction(
                action_type=action_type,
                parameters=action_params,
                confidence=confidence,
                reasoning="Generated via PiscesAgent reasoning"
            )

    def _prepare_reasoning_input(self, context: Dict[str, Any], memory_context: Dict[str, List]) -> Dict[str, Any]:
        """Prepare input for the reasoner"""
        return {
            "context": context,
            "memory": memory_context,
            "agent_state": self.state.value
        }

    def _decode_action_params(self, param_embedding: torch.Tensor, action_type: str) -> Dict[str, Any]:
        """Decode action parameters from embedding"""
        return {
            "embedding": param_embedding.detach().cpu().numpy().tolist(),
            "decoded_from": action_type,
            "confidence": torch.sigmoid(self.confidence_head(param_embedding)).item()
        }

    def _summarize_memory(self) -> Dict[str, int]:
        """Summarize memory for state sync"""
        return {
            "observations": len(self.memory.observations),
            "actions": len(self.memory.actions),
            "reflections": len(self.memory.reflections)
        }

    def detect_objects(self, image_input: Union[str, torch.Tensor, np.ndarray]) -> Dict[str, Any]:
        """
        Detect objects in image and return their coordinates.
        
        Args:
            image_input: Image path string, tensor, or numpy array
            
        Returns:
            Dict containing detected objects with coordinates:
            {
                "objects": [
                    {
                        "class": str,
                        "confidence": float,
                        "coordinates": [x_center, y_center],
                        "bbox": [x_min, y_min, x_max, y_max]
                    }
                ],
                "image_size": [width, height],
                "num_objects": int
            }
        """
        if not self._coordinate_detection_enabled:
            return {"objects": [], "image_size": [0, 0], "num_objects": 0}
        
        try:
            # Process image through vision encoder
            if isinstance(image_input, str):
                image_tensor = self.vision_encoder.process_image(image_input)
                if image_tensor is None:
                    return {"objects": [], "image_size": [0, 0], "num_objects": 0}
                image_tensor = image_tensor.unsqueeze(0)
            elif isinstance(image_input, np.ndarray):
                image_tensor = torch.from_numpy(image_input).float()
                if len(image_tensor.shape) == 3:
                    image_tensor = image_tensor.unsqueeze(0)
            else:
                image_tensor = image_input
            
            # Get detection results from vision encoder
            with torch.no_grad():
                detection_results = self.vision_encoder(image_tensor)
            
            if "detection_results" not in detection_results:
                return {"objects": [], "image_size": [0, 0], "num_objects": 0}
            
            results = detection_results["detection_results"]
            objects = []
            
            # Process bounding boxes and coordinates
            if "boxes" in results and "labels" in results:
                boxes = results["boxes"].cpu().numpy()
                labels = results["labels"].cpu().numpy()
                scores = results.get("scores", torch.ones(len(boxes))).cpu().numpy()
                
                # Convert to image coordinates
                img_coords = self.vision_encoder.convert_patch_to_image_coords(boxes)
                
                for i, (box, label, score) in enumerate(zip(img_coords, labels, scores)):
                    if score > 0.5:  # Confidence threshold
                        x_min, y_min, x_max, y_max = box
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        
                        objects.append({
                            "class": f"class_{label}",
                            "confidence": float(score),
                            "coordinates": [float(x_center), float(y_center)],
                            "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)]
                        })
            
            # Get image dimensions
            if isinstance(image_input, str):
                from PIL import Image
                with Image.open(image_input) as img:
                    width, height = img.size
            else:
                # Assume square image for tensor inputs
                width = height = 224
            
            return {
                "objects": objects,
                "image_size": [width, height],
                "num_objects": len(objects)
            }
            
        except Exception as e:
            return {"objects": [], "image_size": [0, 0], "num_objects": 0, "error": str(e)}

    def get_coordinates(self, image_input: Union[str, torch.Tensor, np.ndarray], 
                         target_object: str = None) -> List[List[float]]:
        """
        Get coordinates of detected objects or specific target.
        
        Args:
            image_input: Image to analyze
            target_object: Optional target object name to filter results
            
        Returns:
            List of [x, y] coordinates for detected objects
        """
        detection_results = self.detect_objects(image_input)
        
        coordinates = []
        for obj in detection_results.get("objects", []):
            if target_object is None or target_object.lower() in obj["class"].lower():
                coordinates.append(obj["coordinates"])
        
        return coordinates

    def point_to_object(self, image_input: Union[str, torch.Tensor, np.ndarray], 
                       object_description: str) -> Dict[str, Any]:
        """
        Point to a specific object in the image based on description.
        
        Args:
            image_input: Image to analyze
            object_description: Description of the object to find
            
        Returns:
            Dict with object information and pointing coordinates
        """
        detection_results = self.detect_objects(image_input)
        
        # Simple matching based on description
        best_match = None
        highest_confidence = 0
        
        for obj in detection_results.get("objects", []):
            # Simple keyword matching for now
            obj_class = obj["class"].lower()
            description_lower = object_description.lower()
            
            if any(keyword in obj_class or obj_class in keyword 
                   for keyword in description_lower.split()):
                if obj["confidence"] > highest_confidence:
                    best_match = obj
                    highest_confidence = obj["confidence"]
        
        if best_match:
            return {
                "found": True,
                "object": best_match,
                "point_coordinates": best_match["coordinates"],
                "message": f"Found {object_description} at coordinates {best_match['coordinates']}"
            }
        else:
            return {
                "found": False,
                "point_coordinates": None,
                "message": f"Could not find {object_description} in the image"
            }

    def enable_coordinate_detection(self, enabled: bool = True):
        """Enable or disable coordinate detection functionality"""
        self._coordinate_detection_enabled = enabled
        if hasattr(self.vision_encoder, 'enable_detection'):
            self.vision_encoder.enable_detection(enabled)

class AgentEncoder(nn.Module):
    """
    Legacy Agent encoder - now replaced by PiscesAgent for comprehensive agent functionality.
    Maintained for backward compatibility.
    """
    def __init__(self, cfg):
        super().__init__()
        self.enabled = True
        self.cfg = cfg
        self.pisces_agent = PiscesAgent(cfg)
        
    def forward(self, agent_input):
        """Forward pass using PiscesAgent"""
        return self.pisces_agent.process_observation(agent_input)
    
    def encode_observation(self, observation):
        """Encode multimodal observations like PiscesAgent"""
        if observation['modality'] == "text":
            tokens = observation['content']
            if isinstance(tokens, str):
                # Handle string input
                tokens = torch.tensor([hash(tokens) % self.cfg.vocab_size])
            return self.obs_text_encoder(tokens)
        
        elif observation['modality'] == "image":
            return self.obs_image_encoder(observation['content'])
        
        elif observation['modality'] == "audio":
            return self.obs_audio_encoder(observation['content'])
        
        elif observation['modality'] == "tool_result":
            # Encode tool results as text
            result_str = str(observation['content'])
            tokens = torch.tensor([hash(result_str) % self.cfg.vocab_size])
            return self.obs_text_encoder(tokens)
        
        else:
            return torch.zeros(1, 1, self.cfg.hidden_size)
    
    def encode_memory(self, memory_data):
        """Encode agent memory like PiscesAgent memory system"""
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
        Forward pass of the comprehensive AgentEncoder.
        
        Args:
            agent_input (dict): Dictionary containing:
                - observations: List of agent observations
                - actions: List of agent actions
                - reflections: List of agent reflections
                - current_state: Current agent state tensor
                - task_context: Current task description
        
        Returns:
            torch.Tensor: Comprehensive agent features including state, memory, and predictions.
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
        
        # Combine all features using attention
        combined_features = torch.cat([
            obs_memory[:, -1:],  # Latest observation
            action_memory[:, -1:],  # Latest action
            reflection_memory[:, -1:],  # Latest reflection
            state_feat,
            task_feat
        ], dim=1)
        
        # Apply cross-modal attention
        attended_features, _ = self.agent_attention(combined_features, combined_features, combined_features)
        
        # Predict action type and parameters (like PiscesAgent)
        action_logits = self.action_type_head(attended_features.mean(dim=1))
        action_params = self.action_param_head(attended_features.mean(dim=1))
        confidence = torch.sigmoid(self.confidence_head(attended_features.mean(dim=1)))
        
        # Final comprehensive encoding
        all_features = torch.cat([
            attended_features.mean(dim=1),
            action_params,
            confidence
        ], dim=-1)
        
        return self.final_proj(all_features).unsqueeze(1)

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention module for enhanced multimodal fusion.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.n_head
        self.hidden_size = cfg.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        B, T_k, _ = key.shape
        
        # Project to Q, K, V
        Q = self.q_proj(self.norm1(query)).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(self.norm2(key)).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.hidden_size)
        
        return self.o_proj(out)

class DynamicModalFusion(nn.Module):
    """
    Dynamic modal fusion module with learnable modality weights including Agent as a modality.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_modalities = 6  # text, image, audio, video, document, agent
        self.hidden_size = cfg.hidden_size
        
        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(self.num_modalities))
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_modalities, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Sigmoid()
        )
        
        # Cross-modal attention layers
        self.text_image_attn = CrossModalAttention(cfg)
        self.text_audio_attn = CrossModalAttention(cfg)
        self.image_audio_attn = CrossModalAttention(cfg)
        self.text_agent_attn = CrossModalAttention(cfg)
        self.image_agent_attn = CrossModalAttention(cfg)
        self.audio_agent_attn = CrossModalAttention(cfg)
        
        self.norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, modal_features):
        """
        Forward pass of dynamic modal fusion including Agent modality.
        
        Args:
            modal_features: Dict containing features from different modalities
            
        Returns:
            Fused multimodal representation
        """
        # Normalize modality weights
        weights = F.softmax(self.modality_weights, dim=0)
        
        # Extract features
        text_feat = modal_features.get('text', torch.zeros(1, 1, self.hidden_size))
        image_feat = modal_features.get('image', torch.zeros(1, 1, self.hidden_size))
        audio_feat = modal_features.get('audio', torch.zeros(1, 1, self.hidden_size))
        video_feat = modal_features.get('video', torch.zeros(1, 1, self.hidden_size))
        doc_feat = modal_features.get('document', torch.zeros(1, 1, self.hidden_size))
        agent_feat = modal_features.get('agent', torch.zeros(1, 1, self.hidden_size))
        
        # Cross-modal interactions including Agent
        text_img = self.text_image_attn(text_feat, image_feat, image_feat)
        text_aud = self.text_audio_attn(text_feat, audio_feat, audio_feat)
        text_agent = self.text_agent_attn(text_feat, agent_feat, agent_feat)
        img_aud = self.image_audio_attn(image_feat, audio_feat, audio_feat)
        img_agent = self.image_agent_attn(image_feat, agent_feat, agent_feat)
        aud_agent = self.audio_agent_attn(audio_feat, agent_feat, agent_feat)
        
        # Weighted fusion with all modalities
        fused_features = [
            text_feat * weights[0],
            image_feat * weights[1],
            audio_feat * weights[2],
            video_feat * weights[3],
            doc_feat * weights[4],
            agent_feat * weights[5],
            text_img * 0.2,  # Interaction terms with reduced weight
            text_aud * 0.2,
            text_agent * 0.2,
            img_aud * 0.2,
            img_agent * 0.2,
            aud_agent * 0.2
        ]
        
        # Concatenate and gate
        concatenated = torch.cat(fused_features, dim=-1)
        gate = self.fusion_gate(concatenated.mean(dim=1))
        
        # Final fusion
        output = self.norm(concatenated.mean(dim=1)) * gate
        return output.unsqueeze(1)

# Export all multimodal components including the new agent system
__all__ = [
    # Vision encoders
    'VisionEncoder',
    'ImageProcessor',
    
    # Audio encoders  
    'AudioEncoder',
    'AudioProcessor',
    
    # Video encoders
    'VideoEncoder',
    'FrameEncoder',
    
    # Agent system (fully migrated from agent.py)
    'PiscesAgent',
    'AgentState',
    'AgentAction',
    'AgentObservation', 
    'AgentMemory',
    'MCPMessageType',
    'MCPMessage',
    'MCPProtocol',
    'MCPToolRegistry',
    'TreeSearchReasoner',
    'PiscesReasoner',
    'AgentEncoder',  # Legacy wrapper maintained for compatibility
    
    # Cross-modal components
    'CrossModalAttention',
    'DynamicModalFusion'
]