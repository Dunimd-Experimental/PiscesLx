#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import math
import torch
from torch import nn
from .moe import MoELayer
import torch.nn.functional as F
from .config import PiscesConfig
from .multimodal import VisionEncoder, AudioEncoder, DocEncoder

def pisces_init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)

class RMSNorm(nn.Module):
    """RMS normalization layer"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding"""
    def __init__(self, dim, max_seq_len=8192, base=1e6, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())
    def forward(self, x, seq_len):
        cos, sin = self.cos[:seq_len], self.sin[:seq_len]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

class Attention(nn.Module):
    """Multi-head attention with grouped-query attention"""
    def __init__(self, cfg, device=None, dtype=None):
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
        self.rope = RotaryEmbedding(self.head_dim, cfg.max_position_embeddings, cfg.rope_theta, device=device, dtype=dtype)
        self.apply(pisces_init_weights)
    def forward(self, x, mask):
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
    """Transformer block with attention and MoE MLP"""
    def __init__(self, cfg, device=None, dtype=None):
        super().__init__()
        self.attn = Attention(cfg, device=device, dtype=dtype)
        self.mlp = MoELayer(cfg, device=device, dtype=dtype)
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.norm2 = RMSNorm(cfg.hidden_size)
    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        mlp_out, aux_loss = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x, aux_loss

class PiscesModel(nn.Module):
    """Pisces L1 multimodal MoE model (oneflow style)"""
    def __init__(self, cfg, device=None, dtype=None, quantization_config=None, lora_config=None):
        super().__init__()
        print("🟧\tPiscesModel: __init__ start")
        self.cfg = cfg
        self.quantization_config = quantization_config
        self.lora_config = lora_config
        
        if quantization_config is not None:
            try:
                import bitsandbytes as bnb
                def convert_linear_to_4bit(module):
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
                print("🟧\tPiscesModel: All Linear layers converted to 4bit (bitsandbytes)")
            except Exception as e:
                print(f"❌\t4bit quantization failed: {e}")
        print("🟧\tPiscesModel: initializing embedding...")
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=dtype)
        print(f"🟧\tPiscesModel: initializing {cfg.n_layer} transformer layers...")
        self.layers = nn.ModuleList([])
        for i in range(cfg.n_layer):
            if (i % 4 == 0) or (i == cfg.n_layer-1):
                print(f"🟧\tPiscesModel: initializing TransformerBlock {i+1}/{cfg.n_layer}")
            self.layers.append(TransformerBlock(cfg, device=device, dtype=dtype))
        print("🟧\tPiscesModel: initializing norm...")
        self.norm = RMSNorm(cfg.hidden_size)
        print("🟧\tPiscesModel: initializing multimodal encoders...")
        self.vision = VisionEncoder(cfg)
        self.audio = AudioEncoder(cfg)
        self.doc = DocEncoder(cfg)
        print("🟧\tPiscesModel: initializing output heads...")
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=device, dtype=dtype)
        self.task_head = nn.Linear(cfg.hidden_size, cfg.task_classes, device=device, dtype=dtype)
        self.eval_head = nn.Linear(cfg.hidden_size, cfg.eval_dims, device=device, dtype=dtype)
        self.apply(pisces_init_weights)
        
        if lora_config is not None:
            try:
                from peft import get_peft_model
                self = get_peft_model(self, lora_config)
                print("🟧\tPiscesModel: LoRA adapters injected (peft)")
            except Exception as e:
                print(f"❌\tLoRA injection failed: {e}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🟧\tPiscesModel: total parameters = {total_params/1e6:.2f}M")
        print("🟧\tPiscesModel: __init__ end")

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Compatible with PEF/Transformers generation interface, can be extended as needed in practice"""
        return {"input_ids": input_ids, **kwargs}

    def forward(self, input_ids, images=None, audio=None, docs=None, labels=None):
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
        mask = torch.full((t, t), float('-inf'), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)
        total_aux_loss = 0.0
        
        chunk_size = min(getattr(self.cfg, 'max_position_embeddings', 2048), 512)
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
                    h = xc
                    aux = 0.0
                    for layer in self.layers:
                        h, aux_loss = layer(h, msk)
                        aux = aux + aux_loss if aux_loss is not None else aux
                    return h, aux
                h_chunk, aux_chunk = cp.checkpoint(block_fn, x_chunk, mask_chunk, use_reentrant=False)
                outputs.append(h_chunk)
                total_aux_loss = total_aux_loss + aux_chunk
            x = torch.cat(outputs, dim=1)
            x = self.norm(x)
            logits = self.lm_head(x)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            task_logits = self.task_head(x[:, 0])
            eval_score = self.eval_head(x.mean(1))
        return logits, loss, task_logits, eval_score, total_aux_loss