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

import math
import torch
from torch import nn
from .moe import MoELayer
import torch.nn.functional as F
from .config import PiscesConfig
from utils.log import  DEBUG, ERROR
from .reasoner import PiscesReasoner
from model.moe_dynamic import DynamicMoELayer
from model.yarn_rope import YaRNRotaryEmbedding
from .multimodal import VisionEncoder, AudioEncoder, DocEncoder, VideoEncoder, AgentEncoder, DynamicModalFusion, CrossModalAttention

def pisces_init_weights(m):
    """
    Initialize weights for PyTorch modules.

    Args:
        m (torch.nn.Module): PyTorch module to initialize weights for.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)

class RMSNorm(nn.Module):
    """
    RMS normalization layer. Normalizes the input using Root Mean Square normalization.
    """
    def __init__(self, dim, eps=1e-6):
        """
        Initialize the RMSNorm layer.

        Args:
            dim (int): Dimension of the input tensor.
            eps (float, optional): Small value to avoid division by zero. Defaults to 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        """
        Forward pass of the RMSNorm layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding module. Adds rotary positional embeddings to the input.
    """
    def __init__(self, dim, max_seq_len=8192, base=1e6, device=None, dtype=None):
        """
        Initialize the RotaryEmbedding layer.

        Args:
            dim (int): Dimension of the input tensor.
            max_seq_len (int, optional): Maximum sequence length. Defaults to 8192.
            base (float, optional): Base value for frequency calculation. Defaults to 1e6.
            device (torch.device, optional): Device to place the tensors on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to None.
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, x, seq_len):
        """
        Forward pass of the RotaryEmbedding layer.

        Args:
            x (torch.Tensor): Input tensor.
            seq_len (int): Current sequence length.

        Returns:
            torch.Tensor: Tensor with rotary positional embeddings applied.
        """
        cos, sin = self.cos[:seq_len], self.sin[:seq_len]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class Attention(nn.Module):
    """
    Multi-head attention module with grouped-query attention using flash_attn.
    """
    def __init__(self, cfg, device=None, dtype=None):
        """
        Initialize the Attention layer.

        Args:
            cfg (PiscesConfig): Configuration object.
            device (torch.device, optional): Device to place the tensors on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to None.
        """
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.hidden_size // cfg.n_head
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.head_dim, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_head * self.head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.hidden_size, bias=False, device=device, dtype=dtype)
        self.rope = YaRNRotaryEmbedding(self.head_dim, cfg.max_position_embeddings, cfg.rope_theta, scale=32, device=device)
        # Attention dropout
        self.attn_dropout = nn.Dropout(getattr(cfg, 'attention_dropout', 0.0))
        self.apply(pisces_init_weights)

    def forward(self, x, mask, past_key_values=None, use_cache=False):
        """
        Forward pass of the Attention layer with KV-Cache support.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Attention mask.
            past_key_values (tuple, optional): Cached key and value tensors from previous steps.
            use_cache (bool, optional): Whether to return cached key and value tensors. Defaults to False.

        Returns:
            torch.Tensor: Output tensor after attention operation.
            tuple: Cached key and value tensors (if use_cache=True).
        """
        # Prefer xformers' memory_efficient_attention; fallback to PyTorch SDPA (>=2.0)
        try:
            from xformers.ops import memory_efficient_attention  # type: ignore
            _use_xformers = True
        except ImportError:
            _use_xformers = False

        b, t, _ = x.shape
        # Project to QKV and reshape to (B, n_head, T, head_dim)
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Rotary positional embeddings
        q, k = self.rope(q, t), self.rope(k, t)

        # Handle KV-Cache
        if past_key_values is not None:
            past_k, past_v = past_key_values
            # Concatenate cached keys and values
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
            
        # Update sequence length after cache
        seq_len = k.size(-2)

        # Broadcast KV heads if using GQA
        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Ensure dtype consistency for xformers (all must match)
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        if _use_xformers:
            # xformers expects shape (B, n_head, T, head_dim)
            out = memory_efficient_attention(q, k, v)  # (B, n_head, T, head_dim)
            out = out.transpose(1, 2).contiguous().view(b, t, -1)
        else:
            # Fallback: torch scaled_dot_product_attention expects (B*n_head, T, head_dim)
            import torch.nn.functional as F
            q_ = q.reshape(b * self.n_head, t, self.head_dim)
            k_ = k.reshape(b * self.n_head, seq_len, self.head_dim)
            v_ = v.reshape(b * self.n_head, seq_len, self.head_dim)
            
            # Create causal mask for the full sequence length
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
            out_ = F.scaled_dot_product_attention(
                q_, k_, v_, 
                attn_mask=~causal_mask[-t:, :], 
                dropout_p=self.attn_dropout.p if self.training else 0.0, 
                is_causal=False
            )
            out = out_.reshape(b, self.n_head, t, self.head_dim).transpose(1, 2).contiguous().view(b, t, -1)

        # Apply attention dropout
        out = self.attn_dropout(out)
        out = self.o_proj(out)
        
        if use_cache:
            # Return cached key and value tensors (before broadcasting)
            k_cache = k[:, :self.n_kv_head] if self.n_kv_head != self.n_head else k
            v_cache = v[:, :self.n_kv_head] if self.n_kv_head != self.n_head else v
            return out, (k_cache, v_cache)
        
        return out

class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MoE MLP, featuring Pre-Norm and residual scaling.
    """
    def __init__(self, cfg, device=None, dtype=None, quantization_config=None):
        """
        Initialize the TransformerBlock.

        Args:
            cfg (PiscesConfig): Configuration object.
            device (torch.device, optional): Device to place the tensors on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to None.
        """
        super().__init__()
        self.attn = Attention(cfg, device=device, dtype=dtype)
        self.mlp = DynamicMoELayer(cfg, device=device, dtype=dtype)
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.norm2 = RMSNorm(cfg.hidden_size)
        # Pre-Norm layers for stability
        self.pre_norm1 = RMSNorm(cfg.hidden_size)
        self.pre_norm2 = RMSNorm(cfg.hidden_size)
        # Residual scaling for deep networks
        self.residual_scale = nn.Parameter(torch.ones(1) * (2.0 * cfg.n_layer) ** -0.5)
        # Dropout for residual connections
        self.residual_dropout = nn.Dropout(getattr(cfg, 'residual_dropout_p', 0.1))
        # Gradient checkpointing flag
        self.use_checkpoint = getattr(cfg, 'use_gradient_checkpointing', True)
        # store quant config
        self.quantization_config = quantization_config

        # Apply 4-bit quantization to this block only (saves memory)
        if self.quantization_config is not None:
            try:
                import bitsandbytes as bnb

                def convert_linear_to_4bit(module):
                    for name, child in module.named_children():
                        if isinstance(child, nn.Linear):
                            new_mod = bnb.nn.Linear4bit(
                                child.in_features,
                                child.out_features,
                                bias=child.bias is not None,
                                quant_type=getattr(self.quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                                compute_dtype=getattr(self.quantization_config, 'bnb_4bit_compute_dtype', torch.bfloat16),
                                compress_statistics=getattr(self.quantization_config, 'bnb_4bit_use_double_quant', True),
                            )
                            setattr(module, name, new_mod)
                        else:
                            convert_linear_to_4bit(child)
                convert_linear_to_4bit(self)
            except Exception as e:
                ERROR(f"Block 4bit quantization failed: {e}")

    def forward(self, x, mask, past_key_values=None, use_cache=False):
        """
        Forward pass of the TransformerBlock with Pre-Norm, residual scaling, gradient checkpointing, and KV-Cache.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Attention mask.
            past_key_values (tuple, optional): Cached key and value tensors from previous steps.
            use_cache (bool, optional): Whether to return cached key and value tensors. Defaults to False.

        Returns:
            tuple: Output tensor, auxiliary loss, and cached key and value tensors (if use_cache=True).
        """
        import torch.utils.checkpoint as cp
        
        def _forward_pass(x_input, attn_past_key_values=None):
            # Pre-Norm attention with residual scaling and dropout
            residual = x_input
            x_norm = self.pre_norm1(x_input)
            
            if use_cache:
                attn_out, attn_cache = self.attn(x_norm, mask, past_key_values=attn_past_key_values, use_cache=True)
            else:
                attn_out = self.attn(x_norm, mask, past_key_values=attn_past_key_values, use_cache=False)
                attn_cache = None
                
            x_out = residual + self.residual_dropout(self.residual_scale * attn_out)
            
            # Post-norm for stability (dual norm approach)
            x_out = self.norm1(x_out)
            
            # Pre-Norm MLP with residual scaling and dropout
            residual = x_out
            x_norm = self.pre_norm2(x_out)
            mlp_out, aux_loss = self.mlp(x_norm)
            x_out = residual + self.residual_dropout(self.residual_scale * mlp_out)
            
            # Post-norm for stability
            x_out = self.norm2(x_out)
            return x_out, aux_loss, attn_cache
        
        # Apply gradient checkpointing if enabled
        if past_key_values is not None:
            attn_past_key_values = past_key_values
        else:
            attn_past_key_values = None
            
        if self.use_checkpoint and self.training:
            x_out, aux_loss, attn_cache = cp.checkpoint(_forward_pass, x, attn_past_key_values, use_reentrant=False)
        else:
            x_out, aux_loss, attn_cache = _forward_pass(x, attn_past_key_values)
        
        if use_cache:
            return x_out, aux_loss, attn_cache
        return x_out, aux_loss

class PiscesModel(nn.Module):
    """Main Pisces L1 multimodal model (Aurora architecture) with enhanced stability mechanisms.
    
    Features:
    - Pre-Norm architecture for improved training stability
    - Residual scaling for deep networks
    - Gradient checkpointing for memory efficiency
    - Attention dropout and residual dropout
    - Dual normalization approach (Pre-Norm + Post-Norm)
    """

    # Exclude the nested PiscesAgent from PyTorch module traversal to avoid
    # infinite recursion when calling `.to()`, `.cuda()`, `.state_dict()`, etc.
    def named_children(self):
        for name, module in super().named_children():
            if name == "agent":
                continue
            yield name, module
    """
    Pisces L1 multimodal MoE model (oneflow style) with stability enhancements.
    """
    def __init__(self, cfg, device=None, dtype=None, quantization_config=None, lora_config=None):
        """
        Initialize the PiscesModel.

        Args:
            cfg (PiscesConfig): Configuration object.
            device (torch.device, optional): Device to place the tensors on. Defaults to None.
            dtype (torch.dtype, optional): Data type of the tensors. Defaults to None.
            quantization_config (object, optional): Configuration for quantization. Defaults to None.
            lora_config (object, optional): Configuration for LoRA. Defaults to None.
        """
        super().__init__()
        DEBUG("PiscesModel: __init__ start")
        self.cfg = cfg
        self.quantization_config = quantization_config
        self.lora_config = lora_config
        
        DEBUG("PiscesModel: initializing embedding...")
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=dtype)
        DEBUG(f"PiscesModel: initializing {cfg.n_layer} transformer layers...")
        self.layers = nn.ModuleList([])
        for i in range(cfg.n_layer):
            if (i % 4 == 0) or (i == cfg.n_layer-1):
                DEBUG(f"PiscesModel: initializing TransformerBlock {i+1}/{cfg.n_layer}")
            self.layers.append(TransformerBlock(cfg, device=device, dtype=dtype, quantization_config=self.quantization_config))
        DEBUG("PiscesModel: initializing norm...")
        self.norm = RMSNorm(cfg.hidden_size)
        DEBUG("PiscesModel: initializing multimodal encoders...")
        # Use unified VisionEncoder with NaViT native resolution support
        self.vision = VisionEncoder(cfg)
        self.video = VideoEncoder(cfg)
        self.audio = AudioEncoder(cfg)
        self.doc = DocEncoder(cfg)
        
        # Agent encoder for behavior/policy modality - now using unified AgentEncoder
        self.agent_encoder = AgentEncoder(cfg)
        
        self.modal_fusion = DynamicModalFusion(cfg)
        DEBUG("PiscesModel: initializing output heads...")
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=device, dtype=dtype)
        self.task_head = nn.Linear(cfg.hidden_size, cfg.task_classes, device=device, dtype=dtype)
        self.eval_head = nn.Linear(cfg.hidden_size, cfg.eval_dims, device=device, dtype=dtype)
        
        DEBUG("PiscesModel: initializing reasoner...")
        self.reasoner = PiscesReasoner(cfg)
        
        DEBUG("PiscesModel: initializing agent...")
        from .multimodal import PiscesAgent
        self.agent = PiscesAgent(cfg, model=self)
        
        # Skipped global weight initialization to avoid lengthy CPU-bound stall.
        # self.apply(pisces_init_weights)
        DEBUG("PiscesModel: skipped duplicate global weight initialization")
        
        if lora_config is not None:
            try:
                from peft import get_peft_model
                self = get_peft_model(self, lora_config)
                DEBUG("PiscesModel: LoRA adapters injected (peft)")
            except Exception as e:
                ERROR(f"LoRA injection failed: {e}")
        total_params = sum(p.numel() for p in self.parameters())
        DEBUG(f"PiscesModel: total parameters = {total_params/1e6:.2f}M")
        DEBUG("PiscesModel: __init__ end")

    def set_gradient_checkpointing(self, enabled: bool = True):
        """
        Enable or disable gradient checkpointing for memory efficiency.
        
        Args:
            enabled (bool): Whether to enable gradient checkpointing. Defaults to True.
        """
        for layer in self.layers:
            layer.use_checkpoint = enabled
        
    def resize_token_embeddings(self, new_num_tokens):
        """
        Resizes token embeddings and associated heads to accommodate a new vocabulary size.

        Args:
            new_num_tokens (int): New vocabulary size.
        """
        # 1. Resize main token embedding
        old_embed = self.embed
        new_embed = nn.Embedding(new_num_tokens, self.cfg.hidden_size, device=old_embed.weight.device, dtype=old_embed.weight.dtype)
        
        # Copy old weights
        num_to_copy = min(old_embed.num_embeddings, new_num_tokens)
        new_embed.weight.data[:num_to_copy, :] = old_embed.weight.data[:num_to_copy, :]
        self.embed = new_embed

        # 2. Resize LM head
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(self.cfg.hidden_size, new_num_tokens, bias=False, device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        new_lm_head.weight.data[:num_to_copy, :] = old_lm_head.weight.data[:num_to_copy, :]
        self.lm_head = new_lm_head

        # 3. Resize reasoner's thinking head
        self.reasoner.resize_vocab(new_num_tokens)
        
        # 4. Update config
        self.cfg.vocab_size = new_num_tokens
        # Note: The 'RIGHT' function is not defined, assuming it's a logging function
        try:
            from utils.log import RIGHT
            RIGHT(f"Resized token embeddings to {new_num_tokens}. Remember to update special token IDs in the reasoner.")
        except ImportError:
            pass

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True, **kwargs):
        """
        Prepare inputs for text generation with KV-Cache support, compatible with PEF/Transformers generation interface.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.
            past_key_values (list, optional): Cached key and value tensors from previous steps.
            use_cache (bool, optional): Whether to use and return KV-Cache. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary containing model inputs for generation with KV-Cache support.
        """
        model_inputs = {"input_ids": input_ids}
        
        # Add attention_mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        model_inputs["attention_mask"] = attention_mask
        
        # Add position_ids if not provided
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        model_inputs["position_ids"] = position_ids
        
        # Add KV-Cache support
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            
        model_inputs["use_cache"] = use_cache
        
        # Include other kwargs
        model_inputs.update(kwargs)
        return model_inputs

    def forward(self, input_ids, images=None, audio=None, video=None, docs=None, labels=None, agent_mode=False, task=None, max_steps=None, agent_obs=None, agent_embed=None, past_key_values=None, use_cache=False):
        """
        Forward pass of the PiscesModel with full multimodal support including Agent and KV-Cache.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len].
            images (torch.Tensor, optional): Input images [batch_size, channels, height, width]. Defaults to None.
            audio (torch.Tensor, optional): Input audio [batch_size, channels, time_steps]. Defaults to None.
            video (torch.Tensor, optional): Input video [batch_size, channels, frames, height, width]. Defaults to None.
            docs (torch.Tensor, optional): Input documents [batch_size, seq_len, hidden_size]. Defaults to None.
            labels (torch.Tensor, optional): Ground truth labels. Defaults to None.
            agent_mode (bool, optional): Enable agent mode. Defaults to False.
            task (str, optional): Task description for agent mode. Defaults to None.
            max_steps (int, optional): Maximum steps for agent execution. Defaults to None.
            agent_obs (torch.Tensor, optional): Agent observations [batch_size, seq_len, hidden_size]. Defaults to None.
            agent_embed (torch.Tensor, optional): Pre-computed agent embeddings [batch_size, seq_len, hidden_size]. Defaults to None.
            past_key_values (list, optional): List of cached key and value tensors for each layer. Defaults to None.
            use_cache (bool, optional): Whether to return cached key and value tensors. Defaults to False.

        Returns:
            dict: Dictionary containing model outputs and optionally cached key and value tensors.
        """
        import torch.utils.checkpoint as cp
        import torch
        
        # Agent mode: delegate to agent
        if agent_mode:
            return self.agent.run(
                input_ids=input_ids,
                images=images,
                audio=audio,
                docs=docs,
                task=task,
                max_steps=max_steps
            )
            
        b, t = input_ids.shape
        
        # Process text embeddings
        text_emb = self.embed(input_ids)
        
        # Process multimodal features
        modal_features = {}
        modal_features['text'] = text_emb
        
        if images is not None:
            modal_features['image'] = self.vision(images)
        if audio is not None:
            modal_features['audio'] = self.audio(audio)
        if video is not None:
            modal_features['video'] = self.video(video)
        if docs is not None:
            modal_features['doc'] = self.doc(docs)
            
        # Process Agent modality if provided
        if agent_embed is not None:
            # Prepare comprehensive agent input for AgentEncoder
            agent_input = {
                'observations': agent_embed.get('observations', []),
                'actions': agent_embed.get('actions', []),
                'reflections': agent_embed.get('reflections', []),
                'current_state': agent_embed.get('current_state', None),
                'task_context': agent_embed.get('task_context', None)
            }
            modal_features['agent'] = self.agent_encoder(agent_input)
            
        if agent_obs is not None:
            # Prepare comprehensive agent observation input
            agent_obs_input = {
                'observations': agent_obs.get('observations', []),
                'actions': agent_obs.get('actions', []),
                'reflections': agent_obs.get('reflections', []),
                'current_state': agent_obs.get('current_state', None),
                'task_context': agent_obs.get('task_context', None)
            }
            agent_feat = self.agent_encoder(agent_obs_input)
            modal_features['agent'] = agent_feat

        # Enhanced multimodal fusion
        if len(modal_features) > 1:
            # Use dynamic fusion for multiple modalities
            fused_features = self.modal_fusion(modal_features)
            x = torch.cat([fused_features, text_emb], dim=1)
        else:
            # Single modality, use text only
            x = text_emb
            
        t = x.shape[1]
        
        # Original sequence length for LM loss calculation
        lm_seq_len = x.shape[1]

        mask = torch.full((t, t), float('-inf'), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        total_aux_loss = 0.0
        
        chunk_size = min(getattr(self.cfg, 'max_position_embeddings', 2048), 8192)
        outputs = []
        if use_cache and past_key_values is not None:
            # Dynamically quantize KV-Cache: automatically select precision based on sequence length
            seq_len = x.shape[1]
            if seq_len > 1024:  # Use 4-bit quantization for long sequences
                cache_dtype = torch.float16
                cache_quant_bits = 4
            elif seq_len > 512:  # Use 8-bit quantization for medium sequences
                cache_dtype = torch.float16  
                cache_quant_bits = 8
            else:  # Use full precision for short sequences
                cache_dtype = torch.float32
                cache_quant_bits = 16
        else:
            cache_dtype = torch.float32
            cache_quant_bits = 16
            
        autocast_ctx = torch.amp.autocast("cuda", dtype=cache_dtype)
        with autocast_ctx:
            # Initialize KV-Cache storage
            next_cache = [] if use_cache else None
            
            for i in range(0, x.shape[1], chunk_size):
                x_chunk = x[:, i:i+chunk_size, ...]
                mask_chunk = mask[i:i+chunk_size, i:i+chunk_size]
                
                def block_fn(xc, msk, layer_past_key_values=None):
                    """
                    Helper function for gradient checkpointing. Applies all transformer layers with KV-Cache support.

                    Args:
                        xc (torch.Tensor): Chunk of the input tensor.
                        msk (torch.Tensor): Chunk of the attention mask.
                        layer_past_key_values (list, optional): List of KV-Cache for each transformer layer.

                    Returns:
                        tuple: Output tensor, accumulated auxiliary loss, and updated KV-Cache.
                    """
                    h = xc
                    aux = 0.0
                    new_caches = []
                    
                    for layer_idx, layer in enumerate(self.layers):
                        past_kv = layer_past_key_values[layer_idx] if layer_past_key_values is not None else None
                        
                        # Dynamically dequantize past_key_value
                        if past_kv is not None and cache_quant_bits < 16:
                            past_kv = tuple(
                                tensor.to(cache_dtype) if tensor is not None else None 
                                for tensor in past_kv
                            )
                        
                        if use_cache:
                            h, aux_loss, cache = layer(h, msk, past_key_values=past_kv, use_cache=True)
                            # Dynamically quantize the new KV-Cache
                            if cache is not None and cache_quant_bits < 16:
                                cache = tuple(
                                    tensor.to(torch.float16) if tensor is not None else None
                                    for tensor in cache
                                )
                            new_caches.append(cache)
                        else:
                            h, aux_loss = layer(h, msk, past_key_values=past_kv, use_cache=False)
                            
                        aux = aux + aux_loss if aux_loss is not None else aux
                    
                    if use_cache:
                        return h, aux, new_caches
                    return h, aux, None
                
                if use_cache:
                    h_chunk, aux_chunk, cache_chunk = cp.checkpoint(block_fn, x_chunk, mask_chunk, past_key_values, use_reentrant=False)
                    if next_cache is not None:
                        next_cache.extend(cache_chunk)
                else:
                    h_chunk, aux_chunk, _ = cp.checkpoint(block_fn, x_chunk, mask_chunk, past_key_values, use_reentrant=False)
                    
                outputs.append(h_chunk)
                total_aux_loss = total_aux_loss + aux_chunk
            if outputs:
                x = torch.cat(outputs, dim=1)
            
            if x.shape[1] == 0:
                # Handle empty sequences gracefully to prevent indexing errors in heads.
                return {
                    "logits": self.lm_head(x),
                    "loss": torch.tensor(0.0, device=x.device, requires_grad=True),
                    "task_logits": torch.zeros(x.shape[0], self.cfg.task_classes, device=x.device),
                    "eval_score": torch.zeros(x.shape[0], self.cfg.eval_dims, device=x.device),
                    "aux_loss": total_aux_loss,
                    "reasoner_out": {"loss": torch.tensor(0.0, device=x.device, requires_grad=True)}
                }

            x = self.norm(x)
            
            # Main model outputs
            logits = self.lm_head(x)
            
            # Reasoner outputs - align input_ids length with x
            reasoner_input_ids = input_ids[:, :x.shape[1]] if input_ids.shape[1] > x.shape[1] else input_ids
            reasoner_labels = labels[:, :x.shape[1]] if labels is not None and labels.shape[1] > x.shape[1] else labels
            reasoner_out = self.reasoner(x, reasoner_input_ids, reasoner_labels)

            loss = None
            if labels is not None:
                # Standard language modeling loss
                lm_loss = F.cross_entropy(
                    logits[:, :lm_seq_len, :].reshape(-1, logits.size(-1)), 
                    labels.view(-1)
                )
                
                # Combine with reasoner loss
                reasoner_loss = reasoner_out.get("loss", torch.tensor(0.0, device=x.device))
                loss = lm_loss + reasoner_loss

            task_logits = self.task_head(x[:, 0])
            eval_score = self.eval_head(x.mean(1))

        result = {
            "logits": logits,
            "loss": loss,
            "task_logits": task_logits,
            "eval_score": eval_score,
            "aux_loss": total_aux_loss,
            "reasoner_out": reasoner_out
        }
        
        if use_cache:
            result["past_key_values"] = next_cache
            
        return result