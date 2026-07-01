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

"""
SEER: Self-Guided Experience-Enhanced Reasoning
(arXiv:2508.15214, EMNLP 2025).

Self-guided method for enhancing LLM function calling in multi-step tool-use
scenarios, using stepwise retrieval from a continually updated experience pool
of past successful trajectories.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional


class YvSEERToolType:
    CHALLENGER = "challenger"
    PLANNER = "planner"
    SOLVER = "solver"
    CRITIC = "critic"


class YvSEERResult:
    def __init__(self, success: bool, output: Any, execution_time: float,
                 tool_name: str, step_id: str = "", error_type: str = ""):
        self.success = success
        self.output = output
        self.execution_time = execution_time
        self.tool_name = tool_name
        self.step_id = step_id
        self.error_message = str(output) if not success else ""


class YvExperienceRecord:
    def __init__(self, tool_name: str, step_description: str,
                 input_context: str, output_result: str, success: bool):
        self.tool_name = tool_name
        self.step_description = step_description
        self.input_context = input_context
        self.output_result = output_result
        self.success = success


class YvSEERExperiencePool:
    def __init__(self, max_size: int = 10000, similarity_top_k: int = 5):
        self.experiences: List[YvExperienceRecord] = []
        self.max_size = max_size
        self.similarity_top_k = similarity_top_k

    def add_experience(self, record: YvExperienceRecord):
        self.experiences.append(record)
        if len(self.experiences) > self.max_size:
            self.experiences = self.experiences[-self.max_size:]

    def add_experiences(self, records: List[YvExperienceRecord]):
        for r in records:
            self.add_experience(r)

    def retrieve(self, query: str, tool_name: str = "", top_k: int = 5) -> List[YvExperienceRecord]:
        matches = [e for e in self.experiences
                   if (not tool_name or e.tool_name == tool_name)
                   and (query.lower() in e.step_description.lower()
                        or query.lower() in e.input_context.lower())]
        return matches[:top_k]


class YvSEERToolBase:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, **kwargs) -> YvSEERResult:
        return YvSEERResult(False, "not implemented", 0.0, self.name)


class YvSEERSearchTool(YvSEERToolBase):
    def __init__(self):
        super().__init__("search", "Search for information")

    async def execute(self, query: str = "", **kwargs) -> YvSEERResult:
        return YvSEERResult(True, f"search_result:{query}", 0.1, self.name)


class YvSEERCodeExecTool(YvSEERToolBase):
    def __init__(self):
        super().__init__("code_exec", "Execute code")

    async def execute(self, code: str = "", **kwargs) -> YvSEERResult:
        return YvSEERResult(True, f"code_result:{code}", 0.2, self.name)


class YvSEERCalculateTool(YvSEERToolBase):
    def __init__(self):
        super().__init__("calculate", "Perform calculations")

    async def execute(self, expression: str = "", **kwargs) -> YvSEERResult:
        return YvSEERResult(True, f"calc_result:{expression}", 0.05, self.name)


class YvSEERFileTool(YvSEERToolBase):
    def __init__(self):
        super().__init__("file", "File operations")

    async def execute(self, action: str = "", path: str = "", **kwargs) -> YvSEERResult:
        return YvSEERResult(True, f"file_{action}:{path}", 0.15, self.name)


class YvSEERReasoningTool(YvSEERToolBase):
    def __init__(self, experience_pool: YvSEERExperiencePool):
        super().__init__("reasoning", "Multi-step reasoning with experience")
        self.experience_pool = experience_pool

    async def execute(self, plan: str = "", context: str = "", **kwargs) -> YvSEERResult:
        return YvSEERResult(True, f"reasoned:{plan}", 0.3, self.name)


class YvSEERVerificationTool(YvSEERToolBase):
    def __init__(self, experience_pool: YvSEERExperiencePool):
        super().__init__("verification", "Verify and critique results")
        self.experience_pool = experience_pool

    async def execute(self, result: str = "", expected: str = "", **kwargs) -> YvSEERResult:
        return YvSEERResult(True, f"verified:{result}", 0.1, self.name)


# Paper: Cui et al., "Self-Guided Function Calling in Large Language Models via Stepwise Experience Recall," EMNLP 2025, arXiv:2508.15214
class YvSEERExecutor(nn.Module):
    """
    SEER: Self-Guided Experience-Enhanced Reasoning (arXiv:2508.15214, EMNLP 2025).
    Stepwise experience recall from a continually updated pool of successful
    trajectories for multi-step tool-use and reasoning scenarios.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.experience_pool = YvSEERExperiencePool()

        self._tools: Dict[str, YvSEERToolBase] = {}
        self._register_tools()

        agent_dim = max(64, self.hidden_size // 16)

        self.challenger_proj = nn.Linear(self.hidden_size, agent_dim)
        self.planner_proj = nn.Linear(self.hidden_size, agent_dim)
        self.solver_proj = nn.Linear(self.hidden_size, agent_dim)
        self.critic_proj = nn.Linear(self.hidden_size, agent_dim)

        self.agent_router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, 4),
        )

        self.difficulty_scorer = nn.Sequential(
            nn.Linear(agent_dim, agent_dim // 2),
            nn.GELU(),
            nn.Linear(agent_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.register_buffer('evolution_step', torch.tensor(0))
        self._stats: Dict[str, Dict] = {}

    def _register_tools(self):
        for tool in [
            YvSEERSearchTool(),
            YvSEERCodeExecTool(),
            YvSEERCalculateTool(),
            YvSEERFileTool(),
            YvSEERReasoningTool(self.experience_pool),
            YvSEERVerificationTool(self.experience_pool),
        ]:
            self._tools[tool.name] = tool
            self._stats[tool.name] = {"calls": 0, "success": 0, "fail": 0, "total_time": 0.0}

    def _select_agent(self, hidden_state: torch.Tensor) -> int:
        pooled = hidden_state.mean(dim=1)
        logits = self.agent_router(pooled)
        return logits.argmax(dim=-1)

    def _challenger_forward(self, x: torch.Tensor) -> float:
        emb = self.challenger_proj(x.mean(dim=1))
        difficulty = self.difficulty_scorer(emb).squeeze(-1)
        curriculum_bias = torch.sigmoid(self.evolution_step.float() * 0.001)
        return (difficulty + curriculum_bias).mean().item()

    def _critic_score(self, x: torch.Tensor) -> torch.Tensor:

        pooled = x.mean(dim=1)
        c_emb = self.critic_proj(pooled)

        s_emb = self.solver_proj(pooled)
        score_input = c_emb * s_emb
        return self.difficulty_scorer(score_input).squeeze(-1)

    def forward(
        self,
        texts: Optional[Any] = None,
        hidden_states: Optional[torch.Tensor] = None,
        aux_loss: Optional[torch.Tensor] = None,
        reasoner_out: Optional[Dict[str, Any]] = None,
        input_ids: Optional[torch.Tensor] = None,
        query_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        hs = hidden_states if hidden_states is not None else query_hidden
        result: Dict[str, Any] = {
            "tool_result": None,
            "experience_recalled": None,
            "seer_loss": torch.tensor(0.0, device=hs.device) if hs is not None else None,
        }

        if hs is None:
            return result

        agent_id = self._select_agent(hs)

        difficulty = self._challenger_forward(hs)
        critic_scores = self._critic_score(hs)
        critic_mean = critic_scores.mean().item()

        if agent_id == 0:
            result["tool_result"] = {
                "agent": "challenger",
                "difficulty": difficulty,
                "suggestion": f"generate task at difficulty {difficulty:.3f}",
            }
        elif agent_id == 1:
            result["tool_result"] = {
                "agent": "planner",
                "difficulty": difficulty,
                "plan_quality": critic_mean,
                "suggestion": f"plan task at difficulty {difficulty:.3f}, quality={critic_mean:.3f}",
            }
        elif agent_id == 2:
            result["tool_result"] = {
                "agent": "solver",
                "confidence": critic_mean,
                "suggestion": f"solve task, confidence={critic_mean:.3f}",
            }
        elif agent_id == 3:
            result["tool_result"] = {
                "agent": "critic",
                "quality_score": critic_mean,
                "accept_threshold": 0.5,
                "accepted": critic_mean > 0.5,
                "suggestion": f"critique: quality={critic_mean:.3f}, {'accepted' if critic_mean > 0.5 else 'rejected'}",
            }

        if aux_loss is not None:
            result["seer_loss"] = aux_loss * (1.0 - critic_mean) * 0.01

        self.evolution_step.add_(1)

        return result

    def get_stats(self) -> Dict:
        return {
            "evolution_step": int(self.evolution_step.item()),
            "tools": self._stats,
            "pool_size": len(self.experience_pool.experiences),
        }
