#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnTA Bridge — Connects EnTA orchestration to PiscesL1 training ops.

This is the glue layer between the Encre-based EnTA agent and the PiscesL1
model training pipeline. It translates EnTA's task execution results into
actual training updates for the 7B student model.

Key connections:
    - EnTATrainer → Bridge → SubconsciousTrainer (0.5B head RL)
    - EnTATrainer → Bridge → opss.train.* (SFT, DPO, GRPO for 7B core)
    - Teacher data → Bridge → training dataset formatting
    - Sandbox reward → Bridge → normalized reward signal

All components are optional. If a component is None, the bridge skips it.
This allows running EnTA in stages: first validate the pipeline, then
add model components one by one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("encre.enta.bridge")


@dataclass
class TrainingSample:
    """A single training sample formatted for student model consumption.

    Produced by the bridge from raw teacher output, ready to be fed into
    the PiscesL1 training pipeline.

    Args:
        input_text: The prompt/input for the student.
        target_text: The expected output (teacher's reasoning + answer).
        reward: Associated reward signal.
        skill_area: Which skill this sample targets.
        metadata: Additional metadata for training ops.
    """
    input_text: str
    target_text: str
    reward: float = 0.0
    skill_area: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class BridgeConfig:
    """Configuration for the EnTA-PiscesL1 bridge.

    Args:
        subconscious_enabled: Whether to train the 0.5B subconscious head.
        sft_enabled: Whether to perform SFT on the 7B core.
        subconscious_update_interval: Steps between subconscious updates.
        batch_size: Training batch size for SFT.
    """
    subconscious_enabled: bool = True
    sft_enabled: bool = True
    subconscious_update_interval: int = 50
    batch_size: int = 8


class EnTABridge:
    """Bridge between EnTA orchestration and PiscesL1 model training.

    Translates EnTA's training loop outputs into actual model updates.
    All training components (subconscious trainer, SFT ops, model) are
    optional — the bridge gracefully degrades when they are None.
    """

    def __init__(
        self,
        config: BridgeConfig,
        student_model: Any = None,
        subconscious_trainer: Any = None,
        training_ops: Optional[Dict[str, Any]] = None,
    ):
        self.cfg = config
        self.model = student_model
        self.subconscious = subconscious_trainer
        self.training_ops = training_ops or {}

        self.sample_buffer: List[TrainingSample] = []
        self.total_samples_processed = 0
        self.total_reward_accumulated = 0.0

        # Validate — log warnings for missing but configured components
        if config.subconscious_enabled and subconscious_trainer is None:
            logger.warning("Bridge: subconscious_enabled=True but trainer is None (will skip)")
        if config.sft_enabled and not self.training_ops.get("sft"):
            logger.warning("Bridge: sft_enabled=True but no SFT op registered (will buffer)")

        logger.info(
            f"Bridge initialized: subconscious={config.subconscious_enabled}, "
            f"sft={config.sft_enabled}, "
            f"model={'loaded' if student_model else 'None'}"
        )

    def feed_teacher_output(
        self,
        teacher_response: str,
        task_description: str,
        reward: float,
        skill_area: str = "general",
        hidden_states: Any = None,
    ) -> TrainingSample:
        """Process teacher output into a training sample and route to training.

        Args:
            teacher_response: Raw response from teacher model.
            task_description: The original task prompt.
            reward: Execution reward (0.0–1.0).
            skill_area: Which skill area this belongs to.
            hidden_states: 7B hidden states (optional, for subconscious).

        Returns:
            The formatted TrainingSample.
        """
        sample = TrainingSample(
            input_text=task_description,
            target_text=teacher_response,
            reward=reward,
            skill_area=skill_area,
        )
        self.sample_buffer.append(sample)
        self.total_samples_processed += 1
        self.total_reward_accumulated += reward

        # Trigger subconscious update at interval
        # hidden_states must be a torch.Tensor (from actual 7B forward pass)
        if (
            self.cfg.subconscious_enabled
            and self.subconscious is not None
            and hidden_states is not None
            and self.total_samples_processed % self.cfg.subconscious_update_interval == 0
        ):
            import torch
            if isinstance(hidden_states, torch.Tensor):
                self._update_subconscious(hidden_states, reward)

        # Trigger SFT batch at interval
        if self.cfg.sft_enabled and len(self.sample_buffer) >= self.cfg.batch_size:
            self._flush_sft_batch()

        return sample

    def _update_subconscious(self, hidden_states: Any, reward: float):
        """Update the 0.5B subconscious head via SubconsciousTrainer.

        Args:
            hidden_states: 7B hidden states for addressing.
            reward: Quality signal for RL update.
        """
        if self.subconscious is None:
            logger.debug("Subconscious update skipped (trainer is None)")
            return
        try:
            metrics = self.subconscious.step(
                hidden_states=hidden_states,
                quality_score=reward,
                task_metadata={"step": self.total_samples_processed},
            )
            logger.debug(f"Subconscious update at step {self.total_samples_processed}: {metrics}")
        except Exception as e:
            logger.error(f"Subconscious update failed: {e}")

    def _flush_sft_batch(self):
        """Flush buffered samples as an SFT training batch."""
        if not self.sample_buffer:
            return

        batch = self.sample_buffer[:self.cfg.batch_size]
        self.sample_buffer = self.sample_buffer[self.cfg.batch_size:]

        sft_op = self.training_ops.get("sft")
        if sft_op is None:
            logger.debug(f"SFT batch skipped (no op registered): {len(batch)} samples buffered")
            # Re-buffer for later when model is available
            self.sample_buffer = batch + self.sample_buffer
            self.sample_buffer = self.sample_buffer[:self.cfg.batch_size * 4]  # cap buffer
            return

        try:
            sft_op.execute(self.model, batch)
            logger.debug(f"SFT batch flushed: {len(batch)} samples")
        except Exception as e:
            logger.error(f"SFT batch failed: {e}")

    def feed_roundtable_output(
        self,
        consensus: Dict[str, Any],
        task: Any,
        reward: float,
    ) -> TrainingSample:
        """Process round-table consensus into a training sample.

        Args:
            consensus: Round-table consensus dict.
            task: Original task.
            reward: Execution reward.

        Returns:
            Formatted TrainingSample.
        """
        synthesis = consensus.get("synthesis", "")
        sample = self.feed_teacher_output(
            teacher_response=synthesis,
            task_description=task.description,
            reward=reward,
            skill_area=task.skill_area.value,
        )

        # Optional DPO for high-quality round-table samples
        if (
            self.cfg.subconscious_enabled
            and reward > 0.7
            and self.training_ops.get("dpo") is not None
        ):
            try:
                self.training_ops["dpo"].execute(self.model, [sample])
            except Exception as e:
                logger.error(f"DPO update failed: {e}")

        return sample

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "total_samples": self.total_samples_processed,
            "avg_reward": round(self.total_reward_accumulated / max(1, self.total_samples_processed), 4),
            "buffer_size": len(self.sample_buffer),
            "subconscious": self.subconscious is not None,
            "model": self.model is not None,
            "sft_op": self.training_ops.get("sft") is not None,
        }
