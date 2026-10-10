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

import math
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PiscesLxVRAMProfile:
    """Hardware VRAM profile."""
    total_gb: float = 0.0
    free_gb: float = 0.0
    device_name: str = "unknown"
    cuda_capability: Tuple[int, int] = (0, 0)
    is_ampere_or_newer: bool = False
    is_hopper_or_newer: bool = False
    num_gpus: int = 1

    @property
    def safe_budget_gb(self) -> float:
        """Budget that leaves 10% headroom."""
        return self.total_gb * 0.9
@dataclass
class PiscesLxVRAMOptConfig:
    """Output of VRAM optimization planning.

    This is what gets applied to the model config.
    """
    # Tier 0 (no speed impact)
    use_flash_attention: bool = True
    use_mla: bool = True
    sdpa_prefer_flash: bool = True

    # Tier 1 (negligible speed impact)
    cache_quantization: bool = True
    use_mixed_precision_cache: bool = True
    use_fp4: bool = False
    kv_cache_block_size: int = 512
    use_partial_rope: bool = True

    # Tier 2 (small speed impact)
    use_gradient_checkpointing: bool = True
    adaptive_recomputation: bool = True
    selective_checkpointing: bool = True

    # Tier 3 (noticeable speed impact, only for extreme cases)
    cpu_offload_optimizer: bool = False
    cpu_offload_weights: bool = False
    activation_quantization: bool = False
    enable_teraio: bool = False

    # MoE-specific
    dynamic_expert_loading: bool = True
    max_experts_on_gpu: int = 4

    # ZeRO
    zero_stage: int = 3

    # Recommended training params (auto-detected)
    recommended_batch_size: int = 1
    recommended_grad_accum: int = 4
    recommended_seq_len: int = 2048

    def apply_to(self, cfg) -> None:
        """Apply these optimizations to a YvConfig object."""
        # Tier 0
        cfg.use_flash_attention = self.use_flash_attention
        cfg.use_mla = self.use_mla
        cfg.sdpa_prefer_flash = self.sdpa_prefer_flash

        # Tier 1
        cfg.cache_quantization = self.cache_quantization
        cfg.use_mixed_precision_cache = self.use_mixed_precision_cache
        cfg.use_fp4 = self.use_fp4
        cfg.kv_cache_block_size = self.kv_cache_block_size
        cfg.use_partial_rope = self.use_partial_rope

        # Tier 2
        cfg.use_gradient_checkpointing = self.use_gradient_checkpointing
        cfg.adaptive_recomputation = self.adaptive_recomputation
        cfg.vram_selective_checkpointing = self.selective_checkpointing

        # Tier 3
        cfg.cpu_offload_optimizer = self.cpu_offload_optimizer
        cfg.cpu_offload_weights = self.cpu_offload_weights
        cfg.activation_quantization = self.activation_quantization
        cfg.enable_teraio = self.enable_teraio

        # MoE
        cfg.vram_dynamic_expert_loading = self.dynamic_expert_loading
        cfg.vram_max_experts_on_gpu = self.max_experts_on_gpu

        # ZeRO
        cfg.vram_zero_stage = self.zero_stage

        # Resolve conflicting settings
        if self.use_fp4:
            cfg.coat_enabled = True
            cfg.vram_fp4_training = False  # conflict: use unified fp4 flag

        if self.cpu_offload_optimizer or self.cpu_offload_weights:
            pass  # User explicitly requested offloading
class PiscesLxVRAMMonitor:
    """Runtime VRAM monitor that dynamically adjusts optimizations.

    Monitors GPU VRAM usage during training and can trigger adjustments
    when memory pressure is detected.
    """

    def __init__(self, cfg, check_interval: int = 50):
        self.cfg = cfg
        self.check_interval = check_interval
        self.step = 0
        self.high_pressure_count = 0
        self.low_pressure_count = 0
        self.current_tier = 0

        self._profile = profile_gpu()

    def check_and_adjust(self) -> bool:
        """Check VRAM and adjust settings if needed.

        Returns:
            True if adjustments were made.
        """
        if not torch.cuda.is_available():
            return False

        self.step += 1
        if self.step % self.check_interval != 0:
            return False

        try:
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            usage_ratio = allocated / max(total, 1)

            if usage_ratio > 0.92:
                self.high_pressure_count += 1
                self.low_pressure_count = 0
            elif usage_ratio < 0.70:
                self.low_pressure_count += 1
                self.high_pressure_count = 0
            else:
                self.high_pressure_count = max(0, self.high_pressure_count - 1)
                self.low_pressure_count = max(0, self.low_pressure_count - 1)

            # High pressure for 3+ checks — escalate
            if self.high_pressure_count >= 3:
                return self._escalate()

            # Low pressure for 5+ checks — de-escalate
            if self.low_pressure_count >= 5:
                return self._deescalate()

        except Exception:
            pass

        return False

    def _escalate(self) -> bool:
        """Increase optimization tier. Returns True if changed."""
        if self.current_tier >= 3:
            return False
        self.current_tier += 1
        self._apply_tier(self.current_tier)
        return True

    def _deescalate(self) -> bool:
        """Decrease optimization tier. Returns True if changed."""
        if self.current_tier <= 0:
            return False
        self.current_tier -= 1
        self._apply_tier(self.current_tier)
        return True

    def _apply_tier(self, tier: int):
        """Apply a specific optimization tier to the running config."""
        cfg = self.cfg

        if tier >= 3:
            cfg.activation_quantization = True
            cfg.cpu_offload_optimizer = True
            cfg.vram_offload_activations = True
        elif tier >= 2:
            cfg.vram_selective_checkpointing = True
            cfg.adaptive_recomputation = True
            cfg.vram_max_experts_on_gpu = 2
        elif tier >= 1:
            cfg.kv_cache_block_size = max(128, cfg.kv_cache_block_size // 2)
            cfg.vram_max_experts_on_gpu = max(2, getattr(cfg, 'vram_max_experts_on_gpu', 4) // 2)
        else:
            cfg.kv_cache_block_size = 512
            cfg.vram_max_experts_on_gpu = 4
            cfg.cpu_offload_optimizer = False
            cfg.activation_quantization = False

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory stats."""
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "cached_gb": 0, "total_gb": 0}

        try:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "usage_pct": round(allocated / total * 100, 1),
                "tier": self.current_tier,
            }
        except Exception:
            return {"allocated_gb": 0, "cached_gb": 0, "total_gb": 0}


# Globally shared VRAM monitor
_monitor: Optional[PiscesLxVRAMMonitor] = None

class PiscesLxVramController:
    """Operations formerly exposed as module-level functions of vram_controller.

    Standard §6.2 requires functionality be exposed through a class, so each became a static method here.
"""

    @staticmethod
    def profile_gpu() -> PiscesLxVRAMProfile:
        """Profile available GPU hardware."""
        profile = PiscesLxVRAMProfile()
        if not torch.cuda.is_available():
            profile.total_gb = 0
            profile.free_gb = 0
            profile.device_name = "cpu"
            return profile

        try:
            device_count = torch.cuda.device_count()
            profile.num_gpus = device_count
            props = torch.cuda.get_device_properties(0)
            profile.total_gb = props.total_memory / 1e9
            profile.device_name = props.name
            profile.cuda_capability = (props.major, props.minor)
            profile.is_ampere_or_newer = props.major >= 8
            profile.is_hopper_or_newer = props.major >= 9

            free_mem, _ = torch.cuda.mem_get_info(0)
            profile.free_gb = free_mem / 1e9
        except Exception:
            profile.total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            profile.device_name = "cuda"

        return profile

    @staticmethod
    def estimate_model_vram(
        num_params_b: float,
        dtype_bytes: float = 2.0,
        optimizer_mult: float = 2.0,
        activation_mult: float = 1.5,
    ) -> Dict[str, float]:
        """Estimate VRAM for a model.

        Args:
            num_params_b: Number of parameters in billions.
            dtype_bytes: Bytes per parameter (2 for BF16, 0.5 for FP4).
            optimizer_mult: Optimizer state multiplier (2 for Adam).
            activation_mult: Activation memory multiplier.

        Returns:
            Dict with 'weights', 'optimizer', 'gradients', 'activations', 'total' in GB.
        """
        weights_gb = num_params_b * dtype_bytes
        optimizer_gb = weights_gb * optimizer_mult
        gradients_gb = weights_gb * 1.0
        activations_gb = weights_gb * activation_mult

        return {
            "weights": round(weights_gb, 2),
            "optimizer": round(optimizer_gb, 2),
            "gradients": round(gradients_gb, 2),
            "activations": round(activations_gb, 2),
            "total": round(weights_gb + optimizer_gb + gradients_gb + activations_gb, 2),
        }

    @staticmethod
    def _estimate_activation_gb(cfg, batch_size: int = 1, seq_len: int = 2048) -> float:
        """Estimate activation memory per training step in GB."""
        hidden = getattr(cfg, 'hidden_size', 640)
        n_layer = getattr(cfg, 'n_layer', 16)
        n_head = getattr(cfg, 'n_head', 10)
        intermediate = getattr(cfg, 'intermediate_size', hidden * 4)

        # Per-token activation per layer (self-attention + FFN)
        attn_act = batch_size * seq_len * hidden * 4  # QKV output + O projection
        ffn_act = batch_size * seq_len * intermediate * 3  # gate, up, down
        per_layer = (attn_act + ffn_act) * 2  # 2 bytes (BF16)

        total = per_layer * n_layer
        return total / (1024**3)

    @staticmethod
    def optimize_for_vram(
        cfg,
        profile: Optional[PiscesLxVRAMProfile] = None,
        num_params_b: Optional[float] = None,
        batch_size: int = 1,
        seq_len: int = 2048,
    ) -> PiscesLxVRAMOptConfig:
        """Automatically select optimal VRAM settings.

        Given a model config and hardware profile, selects the best set of
        optimizations that maximize VRAM savings while minimizing speed loss.

        Args:
            cfg: YvConfig object (or anything with model size attrs).
            profile: GPU hardware profile. Auto-detected if None.
            num_params_b: Model size in billions. Estimated from cfg if None.
            batch_size: Training batch size per GPU.
            seq_len: Training sequence length.

        Returns:
            PiscesLxVRAMOptConfig with optimized settings.
        """
        profile = profile or PiscesLxVramController.profile_gpu()
        opt = PiscesLxVRAMOptConfig()

        if num_params_b is None:
            num_params_b = PiscesLxVramController._estimate_params_b(cfg)

        total_gb = profile.safe_budget_gb
        act_gb = PiscesLxVramController._estimate_activation_gb(cfg, batch_size, seq_len)

        def estimate(dtype_bytes: float, opt_mult: float = 2.0, checkpoint: bool = False) -> float:
            w = num_params_b * dtype_bytes  # weights
            o = w * opt_mult               # optimizer states
            g = w                          # gradients
            a = act_gb * (0.3 if checkpoint else 1.0)  # activations
            return w + o + g + a

        bf16_total = estimate(2.0, opt_mult=2.0)
        bf16_ckpt = estimate(2.0, opt_mult=2.0, checkpoint=True)
        fp4_total = estimate(0.5, opt_mult=4.0)  # FP4 master + FP32 optimizer
        ink_total = estimate(2.0, opt_mult=0.5)  # BF16 weights + 8-bit states (~1x)
        ink_ckpt = estimate(2.0, opt_mult=0.5, checkpoint=True)

        # Tier selection based on actual budget
        if bf16_total <= total_gb:
            opt.use_gradient_checkpointing = False
            opt.use_fp4 = False
        elif ink_total <= total_gb:
            opt.use_gradient_checkpointing = True
            opt.adaptive_recomputation = True
        elif ink_ckpt <= total_gb:
            opt.use_gradient_checkpointing = True
            opt.adaptive_recomputation = True
            opt.selective_checkpointing = True
        elif fp4_total <= total_gb:
            opt.use_fp4 = True
            opt.use_gradient_checkpointing = True
            opt.selective_checkpointing = True
            opt.adaptive_recomputation = True
        else:
            opt.use_fp4 = True
            opt.use_gradient_checkpointing = True
            opt.selective_checkpointing = True
            opt.adaptive_recomputation = True
            opt.cpu_offload_optimizer = True
            opt.activation_quantization = True
            opt.max_experts_on_gpu = 2
            opt.kv_cache_block_size = 256
            opt.enable_teraio = True

        opt.use_flash_attention = profile.is_ampere_or_newer
        opt.use_mla = True
        opt.cache_quantization = True

        est_act_per_sample = PiscesLxVramController._estimate_activation_gb(cfg, batch_size=1, seq_len=seq_len)
        act_scale = 0.3 if opt.use_gradient_checkpointing else 1.0
        w = num_params_b * (0.5 if opt.use_fp4 else 2.0)
        o = w * (0.5 if opt.use_gradient_checkpointing else 2.0)
        g = w
        other = w + o + g
        avail_for_act = max(1.0, total_gb * 0.85 - other)
        max_bs = max(1, int(avail_for_act / max(0.1, est_act_per_sample * act_scale)))
        opt.recommended_batch_size = min(max_bs, 64)
        if opt.recommended_batch_size >= 8:
            opt.recommended_grad_accum = 1
        elif opt.recommended_batch_size >= 4:
            opt.recommended_grad_accum = 2
        else:
            opt.recommended_grad_accum = max(2, 16 // max(1, opt.recommended_batch_size))
        return opt

    @staticmethod
    def _estimate_params_b(cfg) -> float:
        """Estimate parameter count in billions from config."""
        hidden = getattr(cfg, 'hidden_size', 2048)
        n_layer = getattr(cfg, 'n_layer', 24)
        vocab = getattr(cfg, 'vocab_size', 151646)
        num_experts = getattr(cfg, 'moe_num_experts', 64)
        top_k = getattr(cfg, 'moe_top_k', 2)
        intermediate = getattr(cfg, 'intermediate_size', 5632)

        # Embedding
        emb_params = vocab * hidden

        # Per-layer
        attn_params = 4 * hidden * hidden  # Q, K, V, O
        ffn_params = 3 * hidden * intermediate  # gate, up, down

        # MoE scaling (only active experts count for inference memory)
        moe_scale = top_k / num_experts
        total_ffn = ffn_params * num_experts * moe_scale

        per_layer = attn_params + total_ffn
        total = emb_params + per_layer * n_layer

        return total / 1e9

    @staticmethod
    def get_vram_monitor(cfg=None) -> PiscesLxVRAMMonitor:
        """Get or create the shared VRAM monitor."""
        global _monitor
        if _monitor is None and cfg is not None:
            _monitor = PiscesLxVRAMMonitor(cfg)
        return _monitor

    @staticmethod
    def auto_optimize(cfg, force: bool = False) -> bool:
        """Auto-optimize VRAM settings for a model config.

        Call this after creating a YvConfig but before model init.

        Args:
            cfg: YvConfig instance.
            force: If True, re-optimize even if already optimized.

        Returns:
            True if optimizations were applied.
        """
        if getattr(cfg, '_vram_optimized', False) and not force:
            return False

        opt = PiscesLxVramController.optimize_for_vram(cfg)
        opt.apply_to(cfg)
        cfg._vram_optimized = True

        # Initialize monitor
        PiscesLxVramController.get_vram_monitor(cfg)

        return True
