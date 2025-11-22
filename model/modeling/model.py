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

"""
Main model implementation for Arctic.

This module implements the complete Arctic model architecture including
transformer layers, multimodal encoders, reasoning components, and generation
capabilities with support for hybrid attention-Mamba blocks.
"""

import torch
from torch import nn
import torch.nn.functional as F
from .norms import ArcticRMSNorm
from ..config import ArcticConfig
from .hybrid import ArcticHybridBlock
from utils.log.core import PiscesLxCoreLog
from .blocks import ArcticTransformerBlock
from .cache import ArcticUnifiedCacheManager
from typing import Optional, Tuple, Dict, Any
from ..multimodal.reasoner.enhancer import ArcticMultiModalReasoningEnhancer
from ..speculative_decoder import ArcticAdaptiveSpeculativeDecoder, ArcticSpeculativeConfig
from ..multimodal import (
    ArcticUnifiedReasoner,
    ArcticVisionEncoder,
    ArcticAudioEncoder,
    ArcticDocEncoder,
    ArcticVideoEncoder,
    ArcticAgenticEncoder,
    ArcticDynamicModalFusion
)

logger = PiscesLxCoreLog("Arctic.Core.Modeling.Model", file_path="logs/ArcticCore.log")

class ArcticModel(nn.Module):
    """
    Main Arctic model implementation.

    Implements a transformer-based language model with support for multimodal
    inputs, hybrid attention-Mamba blocks, reasoning capabilities, and
    speculative decoding.
    """

    def named_children(self):
        """
        Override named_children to exclude agentic module.

        Yields:
            tuple: (name, module) pairs for all child modules except 'agentic'.
        """
        for name, module in super().named_children():
            if name == "agentic":
                continue
            yield name, module

    def __init__(self, cfg, device=None, dtype=None, quantization_config=None, lora_config=None):
        """
        Initialize the Arctic model.

        Args:
            cfg: Configuration object containing model hyperparameters.
            device (torch.device, optional): Device to place the model on.
            dtype (torch.dtype, optional): Data type for model parameters.
            quantization_config (object, optional): Configuration for model quantization.
            lora_config (object, optional): Configuration for LoRA adapters.
        """
        super().__init__()
        logger.debug("ArcticModel: __init__ start")
        self.cfg = cfg
        self.config = cfg

        # Set default attributes if not present
        if not hasattr(self.config, 'num_layers'):
            setattr(self.config, 'num_layers', getattr(self.config, 'n_layer', 0))
        if not hasattr(self.config, 'num_heads'):
            setattr(self.config, 'num_heads', getattr(self.config, 'n_head', 0))
        if not hasattr(self.config, 'n_kv_head'):
            setattr(
                self.config,
                'n_kv_head',
                getattr(self.config, 'n_kv_head', getattr(self.config, 'n_head', 0))
            )

        # Enable H2O attention for very long sequences
        if getattr(self.config, 'max_position_embeddings', 0) >= 1_048_576 and not hasattr(self.config, 'use_h2o_attention'):
            setattr(self.config, 'use_h2o_attention', True)

        self.quantization_config = quantization_config
        self.lora_config = lora_config

        # Initialize cache manager
        cache_config = getattr(cfg, 'cache_config', {
            "enabled": True,
            "kv_cache_max_size": 2048,
            "h2o_cache_max_size": 1024,
            "generation_cache_max_size": 512,
            "speculative_cache_max_size": 256,
            "quantization_enabled": True,
            "dynamic_quantization": True,
            "cache_eviction_policy": "lru"
        })
        self.cache_manager = ArcticUnifiedCacheManager(cache_config)

        # Initialize token embedding
        logger.debug("ArcticModel: initializing embedding...")
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, device=device, dtype=dtype)

        # Initialize transformer layers
        logger.debug(f"ArcticModel: initializing {cfg.n_layer} transformer layers...")
        self.layers = nn.ModuleList([])
        for i in range(cfg.n_layer):
            if (i % 4 == 0) or (i == cfg.n_layer - 1):
                logger.debug(f"ArcticModel: initializing TransformerBlock {i+1}/{cfg.n_layer}")

            # Select block type based on configuration
            use_hybrid = getattr(cfg, 'use_mamba3', False)
            if use_hybrid:
                # Check if this layer should use Mamba-3
                mamba3_layers = getattr(cfg, 'mamba3_layers', [])
                if not mamba3_layers or i in mamba3_layers:
                    logger.debug(f"ArcticModel: using ArcticHybridBlock for layer {i+1}")
                    block = ArcticHybridBlock(
                        cfg,
                        device=device,
                        dtype=dtype,
                        quantization_config=self.quantization_config
                    )
                else:
                    block = ArcticTransformerBlock(
                        cfg,
                        device=device,
                        dtype=dtype,
                        quantization_config=self.quantization_config
                    )
            else:
                block = ArcticTransformerBlock(
                    cfg,
                    device=device,
                    dtype=dtype,
                    quantization_config=self.quantization_config
                )

            # Set cache manager and layer index
            block.cache_manager = self.cache_manager
            block.layer_idx = i
            self.layers.append(block)

        # Initialize final normalization layer
        logger.debug("ArcticModel: initializing norm...")
        self.norm = ArcticRMSNorm(cfg.hidden_size)

        # Initialize multimodal encoders
        logger.debug("ArcticModel: initializing multimodal encoders...")
        self.vision = ArcticVisionEncoder(cfg)
        self.video = ArcticVideoEncoder(cfg)
        self.audio = ArcticAudioEncoder(cfg)
        self.doc = ArcticDocEncoder(cfg)

        self.agent_encoder = ArcticAgenticEncoder(cfg)
        self.modal_fusion = ArcticDynamicModalFusion(cfg)

        # Unregister agent_encoder to avoid traversing non-core infrastructure during .to()
        try:
            if 'agent_encoder' in self._modules:
                del self._modules['agent_encoder']
        except Exception:
            pass

        # Initialize output heads
        logger.debug("ArcticModel: initializing output heads...")
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device=device, dtype=dtype)
        self.task_head = nn.Linear(cfg.hidden_size, cfg.task_classes, device=device, dtype=dtype)
        self.eval_head = nn.Linear(cfg.hidden_size, cfg.eval_dims, device=device, dtype=dtype)

        self.modal_token_count = getattr(cfg, 'modal_token_count', 8)
        self.fusion_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False, device=device, dtype=dtype)

        # Initialize reasoning components
        logger.debug("ArcticModel: initializing reasoner...")
        self.reasoner = ArcticUnifiedReasoner(cfg)
        self.reasoner.initialize_reasoning_tokens(None)

        logger.debug("ArcticModel: initializing multi-modal reasoning enhancer...")
        self.mm_reasoning_enhancer = ArcticMultiModalReasoningEnhancer(cfg)

        # Initialize agentic module
        logger.debug("ArcticModel: initializing agentic...")
        try:
            from ..multimodal import ArcticAgentic
        except ImportError:
            ArcticAgentic = None
        self.agentic = ArcticAgentic(cfg, model=self)

        # Unregister agentic to avoid circular traversals during .to()/._apply()
        # It holds weak references and includes non-training infrastructure
        try:
            if 'agentic' in self._modules:
                del self._modules['agentic']
        except Exception:
            pass

        # Initialize speculative decoder
        logger.debug("ArcticModel: initializing speculative decoder...")
        self.speculative_config = ArcticSpeculativeConfig(
            num_candidates=getattr(cfg, 'speculative_candidates', 4),
            draft_length=getattr(cfg, 'speculative_draft_length', 5),
            acceptance_threshold=getattr(cfg, 'speculative_acceptance_threshold', 0.8),
            temperature=getattr(cfg, 'speculative_temperature', 0.7),
            top_k=getattr(cfg, 'speculative_top_k', 50),
            top_p=getattr(cfg, 'speculative_top_p', 0.9)
        )
        self.speculative_decoder = ArcticAdaptiveSpeculativeDecoder(self.speculative_config, self, None)
        logger.debug("ArcticModel: speculative decoder initialized")

        # Unregister non-core modules to avoid .to() recursion
        try:
            for k in ("mm_reasoning_enhancer", "speculative_decoder"):
                if k in self._modules:
                    del self._modules[k]
        except Exception:
            pass

        # Apply LoRA adapters if configured
        if lora_config is not None:
            try:
                from peft import get_peft_model
                self = get_peft_model(self, lora_config)
                logger.debug("ArcticModel: LoRA adapters injected (peft)")
            except Exception as e:
                logger.error(f"LoRA injection failed: {e}")

        total_params = sum(p.numel() for p in self.parameters())
        logger.debug(f"ArcticModel: total parameters = {total_params/1e6:.2f}M")
        logger.debug("ArcticModel: __init__ end")

    def set_gradient_checkpointing(self, enabled: bool = True):
        """
        Enable or disable gradient checkpointing for all layers.

        Args:
            enabled (bool): Whether to enable gradient checkpointing. Defaults to True.
        """
        for layer in self.layers:
            layer.use_checkpoint = enabled

    def to(self, device=None, dtype=None, non_blocking=False):
        """
        Move model to specified device and dtype.

        Customized implementation to avoid descending into auxiliary components
        that may form back-references. Only moves core trainable submodules.

        Args:
            device (torch.device, optional): Target device.
            dtype (torch.dtype, optional): Target data type.
            non_blocking (bool): Whether to use non-blocking transfer.

        Returns:
            nn.Module: Self for method chaining.
        """
        if device is None and dtype is None:
            return super().to(device, dtype, non_blocking)

        modules = [
            self.embed,
            self.layers,
            self.norm,
            self.lm_head,
            self.task_head,
            self.eval_head,
            self.fusion_proj,
        ]

        for m in modules:
            if isinstance(m, nn.ModuleList):
                for sub in m:
                    sub.to(device=device, dtype=dtype, non_blocking=non_blocking)
            else:
                m.to(device=device, dtype=dtype, non_blocking=non_blocking)

        return self

    def resize_token_embeddings(self, new_num_tokens):
        """
        Resize token embeddings and language model head.

        Copies existing weights when possible and updates vocabulary size
        across all relevant components.

        Args:
            new_num_tokens (int): New vocabulary size.
        """
        # Resize embedding layer
        old_embed = self.embed
        new_embed = nn.Embedding(
            new_num_tokens,
            self.cfg.hidden_size,
            device=old_embed.weight.device,
            dtype=old_embed.weight.dtype
        )
        num_to_copy = min(old_embed.num_embeddings, new_num_tokens)
        new_embed.weight.data[:num_to_copy, :] = old_embed.weight.data[:num_to_copy, :]
        self.embed = new_embed

        # Resize language model head
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(
            self.cfg.hidden_size,
            new_num_tokens,
            bias=False,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype
        )
        new_lm_head.weight.data[:num_to_copy, :] = old_lm_head.weight.data[:num_to_copy, :]
        self.lm_head = new_lm_head

        # Update vocabulary size across components
        if hasattr(self.reasoner, 'resize_vocab'):
            self.reasoner.resize_vocab(new_num_tokens)
        self.cfg.vocab_size = new_num_tokens

        # Re-initialize special reasoning tokens if supported
        try:
            self.reasoner.initialize_reasoning_tokens(None)
        except Exception:
            pass

        try:
            logger.info(
                f"Resized token embeddings to {new_num_tokens}. "
                f"Remember to update special token IDs in the reasoner."
            )
        except ImportError:
            pass

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        **kwargs
    ):
        """
        Prepare inputs for generation.

        Creates a dictionary of model inputs with default values for missing
        attention mask and position IDs.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask tensor.
            position_ids (torch.Tensor, optional): Position IDs tensor.
            past_key_values (tuple, optional): Cached key/value pairs.
            use_cache (bool): Whether to use caching. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary of model inputs.
        """
        model_inputs = {"input_ids": input_ids}

        # Create default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.long,
                device=input_ids.device
            )
        model_inputs["attention_mask"] = attention_mask

        # Create position IDs from attention mask if not provided
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        model_inputs["position_ids"] = position_ids

        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        model_inputs["use_cache"] = use_cache
        model_inputs.update(kwargs)
        return model_inputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_speculative: bool = True,
        mode: str = 'auto',
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate text using the model.

        Supports two routing modes: 'fast' for standard generation and 'thinking'
        for speculative decoding with adjusted sampling parameters.

        Args:
            input_ids (torch.Tensor): Input token IDs tensor.
            attention_mask (Optional[torch.Tensor]): Optional attention mask.
            max_length (int): Maximum generation length. Defaults to 100.
            temperature (float): Sampling temperature. Defaults to 0.7.
            top_k (int): Top-k sampling parameter. Defaults to 50.
            top_p (float): Top-p (nucleus) sampling parameter. Defaults to 0.9.
            use_speculative (bool): Whether to use speculative decoding. Defaults to True.
            mode (str): Generation mode ('auto', 'fast', or 'thinking'). Defaults to 'auto'.
            **kwargs: Additional generation parameters.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Generated token IDs and statistics dictionary.
        """
        # Determine routing mode
        routing = 'fast'
        if mode == 'thinking':
            routing = 'thinking'
        elif mode == 'auto':
            seq_len = input_ids.shape[1]
            # Use thinking mode for long sequences or high sampling diversity
            if seq_len > 256 or top_k >= 50 or top_p >= 0.9:
                routing = 'thinking'
        else:
            routing = 'fast'

        # Adjust parameters based on routing mode
        use_speculative_final = use_speculative
        temperature_final = temperature
        top_k_final = top_k
        top_p_final = top_p

        if routing == 'thinking':
            # Thinking mode: use speculative decoding with adjusted parameters
            use_speculative_final = True
            temperature_final = max(0.6, temperature * 0.9)
            top_k_final = max(50, top_k)
            top_p_final = max(0.9, top_p)

        # Generate using speculative decoder or standard generation
        if use_speculative_final and hasattr(self, 'speculative_decoder'):
            self.speculative_config.temperature = temperature_final
            self.speculative_config.top_k = top_k_final
            self.speculative_config.top_p = top_p_final
            out_ids, stats = self.speculative_decoder.speculative_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                cache_manager=self.cache_manager if hasattr(self, 'cache_manager') else None,
                **kwargs
            )
            stats['routing'] = routing
            return out_ids, stats
        else:
            out_ids, stats = self._standard_generate(
                input_ids,
                attention_mask,
                max_length,
                temperature_final,
                top_k_final,
                top_p_final,
                **kwargs
            )
            stats['routing'] = routing
            return out_ids, stats

    def _standard_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Standard autoregressive generation without speculative decoding.

        Generates tokens one at a time using top-k and top-p sampling with
        optional adaptive MoE routing temperature adjustment.

        Args:
            input_ids (torch.Tensor): Input token IDs tensor.
            attention_mask (Optional[torch.Tensor]): Optional attention mask.
            max_length (int): Maximum generation length. Defaults to 100.
            temperature (float): Sampling temperature. Defaults to 0.7.
            top_k (int): Top-k sampling parameter. Defaults to 50.
            top_p (float): Top-p (nucleus) sampling parameter. Defaults to 0.9.
            **kwargs: Additional generation parameters including:
                - adaptive_moe: Configuration for adaptive MoE routing
                - min_new_tokens: Minimum tokens to generate before allowing EOS

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: Generated token IDs and statistics dictionary.
        """
        device = input_ids.device
        generated_ids = input_ids.clone()
        stats = {
            'total_draft_tokens': 0,
            'accepted_tokens': 0,
            'rejected_tokens': 0,
            'draft_acceptance_rate': 0.0,
            'speedup': 1.0,
            'method': 'standard'
        }

        # Optional adaptive MoE routing temperature adjustment
        adaptive_cfg = kwargs.pop('adaptive_moe', None)
        adaptive_enabled = bool(adaptive_cfg and adaptive_cfg.get('enabled', False))
        adaptive_step = float(adaptive_cfg.get('temp_step', 0.03)) if adaptive_enabled else 0.0
        adaptive_interval = int(adaptive_cfg.get('interval', 16)) if adaptive_enabled else 0
        adaptive_cap = float(adaptive_cfg.get('temp_cap', 1.30)) if adaptive_enabled else 0.0

        # Enforce minimum number of new tokens before honoring EOS
        min_new_tokens = int(kwargs.pop('min_new_tokens', 0)) if 'min_new_tokens' in kwargs else 0
        new_tokens_generated = 0

        def _bump_gate_temperature(_model, delta: float, cap: float):
            """
            Increase gate temperature for MoE routing to encourage load balance.

            Args:
                _model: Model to update.
                delta (float): Temperature increment.
                cap (float): Maximum temperature cap.
            """
            try:
                for m in _model.modules():
                    if hasattr(m, 'temperature'):
                        try:
                            if isinstance(m.temperature, torch.Tensor):
                                cur = m.temperature.detach().float().mean().item()
                                newv = min(cap, cur + delta)
                                m.temperature.fill_(newv)
                            else:
                                cur = float(getattr(m, 'temperature', 1.0))
                                newv = min(cap, cur + delta)
                                setattr(m, 'temperature', newv)
                        except Exception:
                            continue
            except Exception:
                pass

        with torch.no_grad():
            for step_idx in range(max_length - input_ids.shape[1]):
                # Prepare inputs for this generation step
                model_inputs = self.prepare_inputs_for_generation(
                    generated_ids,
                    attention_mask,
                    **kwargs
                )

                # Remove attention_mask to avoid shape mismatch with fused tokens
                model_inputs.pop("attention_mask", None)
                model_inputs.pop('adaptive_moe', None)

                # Forward pass
                outputs = self(**model_inputs)
                logits = outputs.get('logits', outputs) if isinstance(outputs, dict) else outputs

                # Apply temperature scaling
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_logits,
                        min(top_k, next_token_logits.size(-1))
                    )
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(-1, top_k_indices, top_k_logits)

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1,
                        sorted_indices,
                        sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                new_tokens_generated += 1

                # Update attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            device=device,
                            dtype=attention_mask.dtype
                        )
                    ], dim=-1)

                # Check for EOS token
                if next_token.item() == getattr(self.cfg, 'eos_token_id', 2) and new_tokens_generated >= min_new_tokens:
                    break

                # Periodically adjust MoE routing temperature for load balance
                if adaptive_enabled and adaptive_interval > 0 and ((step_idx + 1) % adaptive_interval == 0):
                    _bump_gate_temperature(self, adaptive_step, adaptive_cap)

        return generated_ids, stats

    def forward(
        self,
        input_ids,
        images=None,
        audio=None,
        video=None,
        docs=None,
        labels=None,
        agent_mode=False,
        task=None,
        max_steps=None,
        agent_obs=None,
        agent_embed=None,
        past_key_values=None,
        use_cache=False,
        attention_mask=None,
        position_ids=None,
        **kwargs
    ):
        """
        Forward pass through the model.

        Processes multimodal inputs (text, images, audio, video, documents, agentic),
        fuses them if multiple modalities are present, passes through transformer
        layers, and computes outputs including logits, loss, and reasoning results.

        Args:
            input_ids (torch.Tensor): Input token IDs tensor.
            images (torch.Tensor, optional): Image inputs.
            audio (torch.Tensor, optional): Audio inputs.
            video (torch.Tensor, optional): Video inputs.
            docs (torch.Tensor, optional): Document inputs.
            labels (torch.Tensor, optional): Target labels for loss computation.
            agent_mode (bool): Whether to run in agentic mode. Defaults to False.
            task (str, optional): Task identifier.
            max_steps (int, optional): Maximum steps for agentic mode.
            agent_obs (dict, optional): Agent observations.
            agent_embed (dict, optional): Agent embeddings.
            past_key_values (tuple, optional): Cached key/value pairs.
            use_cache (bool): Whether to use and update cache. Defaults to False.
            attention_mask (torch.Tensor, optional): Attention mask.
            position_ids (torch.Tensor, optional): Position IDs.
            **kwargs: Additional forward parameters.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - logits: Language model logits
                - loss: Total loss (if labels provided)
                - task_logits: Task classification logits
                - eval_score: Evaluation scores
                - aux_loss: Auxiliary loss from MoE layers
                - reasoner_out: Reasoning module outputs
                - tool_trigger: Tool usage trigger information
                - tool_intent: Tool intent information
                - past_key_values: Updated cache (if use_cache=True)
                - cache_stats: Cache statistics (if available)
        """
        import torch.utils.checkpoint as cp

        # Agentic mode: delegate to agentic module
        if agent_mode:
            return self.agentic.run(
                input_ids=input_ids,
                images=images,
                audio=audio,
                docs=docs,
                task=task,
                max_steps=max_steps
            )

        # Encode text inputs
        b, t = input_ids.shape
        text_emb = self.embed(input_ids)
        modal_features = {'text': text_emb}

        # Encode multimodal inputs
        if images is not None:
            img_out = self.vision(images)
            modal_features['image'] = (
                img_out['features'] if isinstance(img_out, dict) and 'features' in img_out else img_out
            )

        if audio is not None:
            aud_out = self.audio(audio)
            modal_features['audio'] = (
                aud_out['features'] if isinstance(aud_out, dict) and 'features' in aud_out else aud_out
            )

        if video is not None:
            vid_out = self.video(video)
            modal_features['video'] = (
                vid_out['features'] if isinstance(vid_out, dict) and 'features' in vid_out else vid_out
            )

        if docs is not None:
            doc_out = self.doc(docs)
            modal_features['doc'] = (
                doc_out['features'] if isinstance(doc_out, dict) and 'features' in doc_out else doc_out
            )

        # Encode agentic inputs
        if agent_embed is not None:
            agent_input = {
                'observations': agent_embed.get('observations', []),
                'actions': agent_embed.get('actions', []),
                'reflections': agent_embed.get('reflections', []),
                'current_state': agent_embed.get('current_state', None),
                'task_context': agent_embed.get('task_context', None)
            }
            modal_features['agentic'] = self.agent_encoder(agent_input)

        if agent_obs is not None:
            agent_obs_input = {
                'observations': agent_obs.get('observations', []),
                'actions': agent_obs.get('actions', []),
                'reflections': agent_obs.get('reflections', []),
                'current_state': agent_obs.get('current_state', None),
                'task_context': agent_obs.get('task_context', None)
            }
            agent_feat = self.agent_encoder(agent_obs_input)
            modal_features['agentic'] = agent_feat

        # Fuse multimodal features if multiple modalities present
        if len(modal_features) > 1:
            fused_features = self.modal_fusion(modal_features)
            if fused_features is None:
                x = text_emb
            elif fused_features.dim() == 3:
                # 3D features: concatenate directly
                if fused_features.dtype != text_emb.dtype:
                    fused_features = fused_features.to(text_emb.dtype)
                if fused_features.device != text_emb.device:
                    fused_features = fused_features.to(text_emb.device)
                x = torch.cat([fused_features, text_emb], dim=1)
            elif fused_features.dim() == 2:
                # 2D features: project and expand to tokens
                B, H = fused_features.shape
                ff = fused_features.to(device=text_emb.device, dtype=text_emb.dtype)
                proj = self.fusion_proj(ff)
                tokens = proj.unsqueeze(1).expand(B, self.modal_token_count, H).contiguous()
                x = torch.cat([tokens, text_emb], dim=1)
            else:
                x = text_emb
        else:
            x = text_emb

        # Prepare attention mask (causal mask)
        t = x.shape[1]
        lm_seq_len = x.shape[1]
        mask = torch.full((t, t), float('-inf'), device=x.device, dtype=x.dtype)
        mask = torch.triu(mask, diagonal=1)

        total_aux_loss = 0.0
        chunk_size = min(getattr(self.cfg, 'max_position_embeddings', 2048), 8192)
        outputs = []

        # Determine cache quantization based on sequence length
        if use_cache:
            seq_len = x.shape[1]
            if seq_len > 1024:
                cache_dtype = torch.float16
                cache_quant_bits = 4
            elif seq_len > 512:
                cache_dtype = torch.float16
                cache_quant_bits = 8
            else:
                cache_dtype = torch.float32
                cache_quant_bits = 16
        else:
            cache_dtype = torch.float32
            cache_quant_bits = 16

        # Process sequence in chunks with mixed precision
        autocast_ctx = torch.amp.autocast("cuda", dtype=cache_dtype)
        with autocast_ctx:
            next_cache = [] if use_cache else None

            for i in range(0, x.shape[1], chunk_size):
                x_chunk = x[:, i:i+chunk_size, ...]
                mask_chunk = mask[i:i+chunk_size, i:i+chunk_size]

                def block_fn(xc, msk, layer_past_key_values=None):
                    """
                    Process a chunk through all transformer layers.

                    Args:
                        xc (torch.Tensor): Input chunk tensor.
                        msk (torch.Tensor): Attention mask for chunk.
                        layer_past_key_values (list, optional): Past key/value pairs per layer.

                    Returns:
                        tuple: (hidden_states, aux_loss, new_caches) or (hidden_states, aux_loss, None).
                    """
                    h = xc
                    aux = 0.0
                    new_caches = []
                    seq_len = xc.shape[1]

                    for layer_idx, layer in enumerate(self.layers):
                        # Determine if Mamba-3 should be used for this layer
                        use_mamba3_for_layer = False
                        if getattr(self.cfg, 'use_mamba3', False):
                            threshold = getattr(self.cfg, 'mamba3_sequence_threshold', 8192)
                            mamba3_layers = getattr(self.cfg, 'mamba3_layers', [])

                            # Check if this layer should use Mamba-3 based on configuration and sequence length
                            if (not mamba3_layers or layer_idx in mamba3_layers) and seq_len >= threshold:
                                use_mamba3_for_layer = True

                        # Retrieve cached key/value pairs
                        past_kv = self.cache_manager.get_kv_cache(
                            layer_idx,
                            layer_past_key_values[layer_idx] if layer_past_key_values is not None else None
                        )

                        # Convert cache to quantized dtype if needed
                        if past_kv is not None and cache_quant_bits < 16:
                            past_kv = tuple(
                                tensor.to(cache_dtype) if tensor is not None else None
                                for tensor in past_kv
                            )

                        if use_cache:
                            # Pass sequence length information to hybrid blocks
                            if hasattr(layer, 'set_sequence_length'):
                                layer.set_sequence_length(seq_len)

                            h, aux_loss, cache = layer(h, msk, past_key_values=past_kv, use_cache=True)

                            if cache is not None:
                                key_states, value_states = cache
                                updated = self.cache_manager.update_kv_cache(
                                    layer_idx,
                                    key_states,
                                    value_states,
                                    i + xc.shape[1],
                                    use_h2o=getattr(self.cfg, 'use_h2o_attention', False)
                                )
                                cache = updated

                                # Convert updated cache to quantized dtype if needed
                                if cache_quant_bits < 16:
                                    cache = tuple(
                                        tensor.to(torch.float16) if tensor is not None else None
                                        for tensor in cache
                                    )
                            new_caches.append(cache)
                        else:
                            # Pass sequence length information to hybrid blocks
                            if hasattr(layer, 'set_sequence_length'):
                                layer.set_sequence_length(seq_len)

                            h, aux_loss = layer(h, msk, past_key_values=past_kv, use_cache=False)

                        aux = aux + (aux_loss if aux_loss is not None else 0.0)

                    if use_cache:
                        return h, aux, new_caches
                    return h, aux, None

                # Process chunk with or without checkpointing
                if use_cache:
                    h_chunk, aux_chunk, cache_chunk = block_fn(x_chunk, mask_chunk, past_key_values)
                    if next_cache is not None and cache_chunk is not None:
                        next_cache.extend(cache_chunk)
                else:
                    h_chunk, aux_chunk, _ = cp.checkpoint(
                        block_fn,
                        x_chunk,
                        mask_chunk,
                        None,
                        use_reentrant=False
                    )

                outputs.append(h_chunk)
                total_aux_loss = total_aux_loss + aux_chunk
            # Concatenate chunk outputs
            if outputs:
                x = torch.cat(outputs, dim=1)

            # Handle empty sequence case
            if x.shape[1] == 0:
                return {
                    "logits": self.lm_head(x),
                    "loss": torch.tensor(0.0, device=x.device, requires_grad=True),
                    "task_logits": torch.zeros(x.shape[0], self.cfg.task_classes, device=x.device),
                    "eval_score": torch.zeros(x.shape[0], self.cfg.eval_dims, device=x.device),
                    "aux_loss": total_aux_loss,
                    "reasoner_out": {"loss": torch.tensor(0.0, device=x.device, requires_grad=True)}
                }

            # Final normalization and output projection
            x = self.norm(x)
            logits = self.lm_head(x)

            # Prepare inputs for reasoner (truncate if needed)
            reasoner_input_ids = (
                input_ids[:, :x.shape[1]] if input_ids.shape[1] > x.shape[1] else input_ids
            )
            reasoner_labels = (
                labels[:, :x.shape[1]]
                if labels is not None and labels.shape[1] > x.shape[1]
                else labels
            )

            # Run reasoner
            reasoner_out = self.reasoner(x, reasoner_input_ids, reasoner_labels)

            # Compute loss if labels provided
            loss = None
            if labels is not None:
                lm_loss = F.cross_entropy(
                    logits[:, :lm_seq_len, :].reshape(-1, logits.size(-1)),
                    labels.view(-1)
                )
                reasoner_loss = reasoner_out.get("loss", torch.tensor(0.0, device=x.device))
                loss = lm_loss + reasoner_loss

            # Determine tool usage trigger based on reasoner outputs
            tool_trigger = None
            try:
                unc = reasoner_out.get("uncertainty_scores", None)
                fac = reasoner_out.get("fact_consistency", None)
                unc_val = unc.mean().item() if unc is not None and unc.numel() > 0 else 0.0
                fac_val = fac.mean().item() if fac is not None and fac.numel() > 0 else 1.0
                unc_th = getattr(self.cfg, 'tool_uncertainty_threshold', 0.7)
                fac_th = getattr(self.cfg, 'tool_fact_consistency_threshold', 0.6)
                should_tool = (unc_val >= unc_th) or (fac_val <= fac_th)
                tool_trigger = {
                    'should_tool': bool(should_tool),
                    'uncertainty': float(unc_val),
                    'fact_consistency': float(fac_val),
                    'suggested_tools': ['search'] if should_tool else []
                }
            except Exception:
                tool_trigger = None

            # Compute task and evaluation outputs
            task_logits = self.task_head(x[:, 0])
            eval_score = self.eval_head(x.mean(1))

        # Generate tool intent if tool usage is triggered
        tool_intent = None
        try:
            if tool_trigger is not None and tool_trigger.get('should_tool', False):
                tool_intent = {
                    'type': 'tool_intent',
                    'version': '1.0',
                    'triggers': {
                        'uncertainty': tool_trigger.get('uncertainty', 0.0),
                        'fact_consistency': tool_trigger.get('fact_consistency', 1.0),
                        'thresholds': {
                            'uncertainty': getattr(self.cfg, 'tool_uncertainty_threshold', 0.7),
                            'fact_consistency': getattr(self.cfg, 'tool_fact_consistency_threshold', 0.6)
                        }
                    },
                    'suggested_tools': tool_trigger.get('suggested_tools', []),
                    'confidence': float(min(1.0, max(0.0, tool_trigger.get('uncertainty', 0.0)))),
                    'reason': 'High uncertainty or low fact consistency detected by reasoner',
                }
        except Exception:
            tool_intent = None

        # Build result dictionary
        result = {
            "logits": logits,
            "loss": loss,
            "task_logits": task_logits,
            "eval_score": eval_score,
            "aux_loss": total_aux_loss,
            "reasoner_out": reasoner_out,
            "tool_trigger": tool_trigger,
            "tool_intent": tool_intent
        }

        if use_cache:
            result["past_key_values"] = next_cache

        # Add cache statistics if available
        if hasattr(self, 'cache_manager') and self.cache_manager is not None:
            try:
                result["cache_stats"] = self.cache_manager.get_cache_stats()
            except Exception:
                pass

        return result
        