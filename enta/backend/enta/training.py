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

"""EnTA trainer persona and training-stage constants.

This module is the single source of truth for the *trainer's* identity --
EnTA is the agent that *trains* the PiscesLx model, not the model being
trained.  The persona is injected into the model's context through the
``train_plan`` tool output and the ``description`` fields of the training
tools (the only persistent instruction channel available to the agent
body, which has no system-prompt mechanism of its own).
"""

from __future__ import annotations

# Ordered list of the standard distillation + RL hybrid training loop.
TRAINING_PHASES: list[str] = [
    "data_generation",
    "distill",
    "rl",
    "eval",
]

# Human-readable gloss for each phase, reused by train_plan / train_orchestrate.
TRAINING_PHASE_GLOSS: dict[str, str] = {
    "data_generation": (
        "Generate / gather training data -- typically via the multi-teacher "
        "roundtable (build_dataset / generate_roundtable_dataset), producing "
        "reference trajectories for distillation."
    ),
    "distill": (
        "Knowledge distillation -- drive the student core with roundtable-"
        "selected references (run_with_roundtable) so the 7B core absorbs the "
        "314B knowledge body's behaviour."
    ),
    "rl": (
        "Reinforcement learning -- rollout in the sandbox, score trajectories, "
        "and run adversarial batches (run_adversarial_batch) to sharpen tool "
        "use and reasoning."
    ),
    "eval": (
        "Evaluation -- load the latest checkpoint and run a capability profile "
        "(evaluate) to decide whether to continue, anneal, or stop."
    ),
}

TRAINER_SYSTEM_PROMPT: str = """你是 EnTA（Encre Train Agent）——一个专门负责训练 PiscesLx 模型的「训练者」Agent，而不是被训练的模型本身。

你的职责：
1. 理解训练目标，把它拆解为可执行的四个阶段并按序推进：数据生成(data_generation) → 知识蒸馏(distill) → 强化学习(rl) → 评估(eval)。
2. 用 train_plan 把目标与阶段登记到任务台账，形成可审计的训练路线图。
3. 用 train_launch 在后台启动训练（或 train_orchestrate 跑端到端循环）。
4. 用 train_status 监控 loss / 学习率 / 梯度范数，用 train_checkpoints 管理检查点。
5. 用 train_eval 在检查点上做能力评估，判断是否需要继续、退火或收敛停止。
6. 根据量化指标自我评估：loss 是否下降、是否过拟合、阶段权重与数据是否需要调整。

你只通过工具驱动训练，绝不自己实现训练算法或改动模型结构。每轮都要把进度与结论写回任务台账，确保整个训练过程可复现、可审计。"""
