#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

from encre.swarm.mailbox import EncreMailbox, MailboxMessage
from encre.swarm.teammate import EncreTeammate, TeammateHandle
from encre.swarm.manager import EncreSwarmManager, SwarmProgress
from encre.swarm.planner import EncreTaskPlanner, TaskTree, TaskNode
from encre.swarm.roles import (
    AgentRole, RoleRegistry,
    ROLE_ARCHITECT, ROLE_CODER, ROLE_REVIEWER, ROLE_TESTER,
    ROLE_RESEARCHER, ROLE_DEBUGGER, ROLE_GENERAL,
)
from encre.swarm.orchestrator import EncreOrchestrator, OrchestrationEvent
from encre.swarm.blackboard import EncreBlackboard, BlackboardEntry
from encre.swarm.consensus import EncreConsensus, Proposal, Vote, ConsensusResult
from encre.swarm.session import EncreSwarmSession, SwarmEvent, SwarmResult

__all__ = [
    "EncreTeammate",
    "TeammateHandle",
    "EncreMailbox",
    "MailboxMessage",
    "EncreSwarmManager",
    "SwarmProgress",
    "EncreTaskPlanner",
    "TaskTree",
    "TaskNode",
    "AgentRole",
    "RoleRegistry",
    "ROLE_ARCHITECT",
    "ROLE_CODER",
    "ROLE_REVIEWER",
    "ROLE_TESTER",
    "ROLE_RESEARCHER",
    "ROLE_DEBUGGER",
    "ROLE_GENERAL",
    "EncreOrchestrator",
    "OrchestrationEvent",
    "EncreBlackboard",
    "BlackboardEntry",
    "EncreConsensus",
    "Proposal",
    "Vote",
    "ConsensusResult",
    "EncreSwarmSession",
    "SwarmEvent",
    "SwarmResult",
]
