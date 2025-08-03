#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces L1.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0/
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
from model.vision_native import NativeSiglipVisionEncoder
from .multimodal import VisionEncoder, AudioEncoder, DocEncoder

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
    Multi-head attention module with grouped-query attention.
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
        self.apply(pisces_init_weights)

    def forward(self, x, mask):
        """
        Forward pass of the Attention layer.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Output tensor after attention operation.
        """
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, t), self.rope(k, t)
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    """
    Transformer block with attention and MoE MLP.
    """
    def __init__(self, cfg, device=None, dtype=None):
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

    def forward(self, x, mask):
        """
        Forward pass of the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Attention mask.

        Returns:
            tuple: Output tensor and auxiliary loss.
        """
        x = x + self.attn(self.norm1(x), mask)
        mlp_out, aux_loss = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x, aux_loss

class PiscesModel(nn.Module):
    """
    Pisces L1 multimodal MoE model (oneflow style).
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
        
        if quantization_config is not None:
            try:
                import bitsandbytes as bnb
                def convert_linear_to_4bit(module):
                    """
                    Convert all Linear layers in a module to 4-bit Linear layers.

                    Args:
                        module (torch.nn.Module): Module to convert.
                    """
                    for name, child in module.named_children():
                        if isinstance(child, nn.Linear):
                            new_mod = bnb.nn.Linear4bit(
                                child.in_features, child.out_features, bias=child.bias is not None,
                                quant_type=getattr(quantization_config, 'bnb_4bit_quant_type', 'nf4'),
                                compute_dtype=getattr(quantization_config, 'bnb_4bit_compute_dtype', torch.bfloat16),
                                compress_statistics=getattr(quantization_config, 'bnb_4bit_use_double_quant', True)
                            )
                            setattr(module, name, new_mod)
                        else:
                            convert_linear_to_4bit(child)
                convert_linear_to_4bit(self)
                DEBUG("PiscesModel: All Linear layers converted to 4bit (bitsandbytes)")
            except Exception as e:
                ERROR(f"4bit quantization failed: {e}")
        DEBUG("PiscesModel: initializing embedding...")
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=dtype)
        DEBUG(f"PiscesModel: initializing {cfg.n_layer} transformer layers...")
        self.layers = nn.ModuleList([])
        for i in range(cfg.n_layer):
            if (i % 4 == 0) or (i == cfg.n_layer-1):
                DEBUG(f"PiscesModel: initializing TransformerBlock {i+1}/{cfg.n_layer}")
            self.layers.append(TransformerBlock(cfg, device=device, dtype=dtype))
        DEBUG("PiscesModel: initializing norm...")
        self.norm = RMSNorm(cfg.hidden_size)
        DEBUG("PiscesModel: initializing multimodal encoders...")
        self.vision = VisionEncoder(cfg)
        self.audio = AudioEncoder(cfg)
        self.doc = DocEncoder(cfg)
        DEBUG("PiscesModel: initializing output heads...")
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=device, dtype=dtype)
        self.task_head = nn.Linear(cfg.hidden_size, cfg.task_classes, device=device, dtype=dtype)
        self.eval_head = nn.Linear(cfg.hidden_size, cfg.eval_dims, device=device, dtype=dtype)
        
        DEBUG("PiscesModel: initializing reasoner...")
        self.reasoner = PiscesReasoner(cfg)

        self.apply(pisces_init_weights)
        
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

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        """
        Prepare inputs for text generation, compatible with PEF/Transformers generation interface.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            position_ids (torch.Tensor, optional): Position IDs. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary containing model inputs for generation.
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
        
        # Include other kwargs
        model_inputs.update(kwargs)
        return model_inputs

    def forward(self, input_ids, images=None, audio=None, docs=None, labels=None):
        """
        Forward pass of the PiscesModel.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            images (torch.Tensor, optional): Input images. Defaults to None.
            audio (torch.Tensor, optional): Input audio. Defaults to None.
            docs (torch.Tensor, optional): Input documents. Defaults to None.
            labels (torch.Tensor, optional): Ground truth labels. Defaults to None.

        Returns:
            dict: Dictionary containing model outputs.
        """
        import torch.utils.checkpoint as cp
        import torch
        b, t = input_ids.shape
        x = self.embed(input_ids)
        if images is not None:
            x = torch.cat([self.vision(images), x], dim=1)
            t += 1
        if audio is not None:
            x = torch.cat([self.audio(audio), x], dim=1)
            t += 1
        if docs is not None:
            x = torch.cat([self.doc(docs), x], dim=1)
            t += 1
        
        # Original sequence length for LM loss calculation
        lm_seq_len = x.shape[1]

        mask = torch.full((t, t), float('-inf'), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        total_aux_loss = 0.0
        
        chunk_size = min(getattr(self.cfg, 'max_position_embeddings', 2048), 8192)
        outputs = []
        
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)
        with autocast_ctx:
            for i in range(0, x.shape[1], chunk_size):
                x_chunk = x[:, i:i+chunk_size, ...]
                mask_chunk = mask[i:i+chunk_size, i:i+chunk_size]
                def block_fn(xc, msk):
                    """
                    Helper function for checkpointing, applies all transformer layers.

                    Args:
                        xc (torch.Tensor): Input tensor chunk.
                        msk (torch.Tensor): Attention mask chunk.

                    Returns:
                        tuple: Output tensor and accumulated auxiliary loss.
                    """
                    h = xc
                    aux = 0.0
                    for layer in self.layers:
                        h, aux_loss = layer(h, msk)
                        aux = aux + aux_loss if aux_loss is not None else aux
                    return h, aux
                h_chunk, aux_chunk = cp.checkpoint(block_fn, x_chunk, mask_chunk, use_reentrant=False)
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

        return {
            "logits": logits,
            "loss": loss,
            "task_logits": task_logits,
            "eval_score": eval_score,
            "aux_loss": total_aux_loss,
            "reasoner_out": reasoner_out
        }