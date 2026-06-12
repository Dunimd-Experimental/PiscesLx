#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from encre.agent import EncreAgent
from encre.goal import EncreGoalRunner, EncreGoalLoop, GoalDefinition, GoalResult, GoalStatus, GoalEvent
from encre.scheduler import EncreScheduler, ScheduledJob, CronSchedule, ScheduleType, JobState
from encre.recovery import ErrorRecoveryEngine, RetryableExecutor, RecoveryState, RecoveryDecision, RecoveryAction, ErrorCategory
from encre.backend import create_backend
from encre.backends.base import BaseBackend
from encre.backends.registry import BackendRegistry, ModelInfo
from encre.backends.retry import RetryConfig, retry_with_backoff, DEFAULT_RETRY_CONFIG
from encre.backends.openai import OpenAIBackend
from encre.backends.anthropic import AnthropicBackend
from encre.backends.ollama import OllamaBackend
from encre.backends.deepseek import DeepSeekBackend
from encre.backends.google import GoogleBackend
from encre.backends.groq import GroqBackend
from encre.backends.local import LocalBackend
from encre.backends.bedrock import BedrockBackend
from encre.backends.openai_compatible import OpenAICompatibleBackend
from encre.crypto import encrypt, decrypt, encrypt_bytes, decrypt_bytes, ensure_keyfile
from encre.rollback import EncreRollbackGit, CommitEntry
from encre.config import EncreConfig, ModelConfig, get_data_dir, SubAgentConfig
from encre.rules.loader import RulesLoader
from encre.loop import EncreLoop
from encre.memdir.system import EncreMemorySystem, MemoryHeader, EntrypointResult
from encre.memdir.semantic import SemanticMemorySearch, SearchResult, WorkingMemory, MemoryConsolidator, ConsolidationAction
from encre.profile import EncreProfileSystem, UserProfile
from encre.soul.system import EncreSoulSystem, SoulFiles
from encre.safety import EncreSafetyEngine, BashAnalysis, DangerLevel, analyze_bash_command
from encre.autosafety import EncreAutoSafetyClassifier, AutoDecision, ClassificationResult, UserDecisionRecord
from encre.sandbox.container import EncreContainerSandbox
from encre.sandbox.types import SandboxConfig, SandboxResult
from encre.spec import SpecDocument, SpecSection, SpecStatus, EncreSpecEngine
from encre.evolution.config import EvolutionConfig
from encre.evolution.learner import EncreEvolutionLearner, SuccessRecord, ErrorRecord
from encre.evolution.optimizer import EncreStrategyOptimizer, ToolStrategy
from encre.evolution.reflex import EncreReflexLoop, ReflexResult
from encre.evolution.meta import EncreMetaCognition, CapabilityProfile
from encre.session import EncreSession, SessionCheckpoint, BranchMeta
from encre.utils.idgen import BranchIDGenerator
from encre.telemetry import EncreTelemetry, ToolCallRecord, TurnRecord, RetryRecord
from encre.logging_config import setup_logging, get_logger
from encre.native import (
    read_file as native_read_file,
    write_file as native_write_file,
    grep as native_grep,
    glob_pattern as native_glob,
    count_tokens as native_count_tokens,
    compute_diff as native_compute_diff,
    apply_diff as native_apply_diff,
    sandbox_execute as native_sandbox_execute,
    search_codebase as native_search_codebase,
)
from encre.tools.base import EncreTool
from encre.tools.registry import ToolRegistry
from encre.tools.discovery import ToolDiscovery, BASE_TOOLS
from encre.tools.mcp import EncreMCPTool
from encre.tools.mcp_manager import (
    MCPManager,
    MCPServerSpec,
    bootstrap_mcp_servers,
    default_mcp_config_path,
)
from encre.git.repo import EncreGitRepo, GitState
from encre.git.diff import EncreGitDiff, GitDiffResult
from encre.lsp.client import EncreLSPClient
from encre.lsp.manager import EncreLSPManager
from encre.lsp.protocol import (
    Position,
    Range as LSPRange,
    Location as LSPLocation,
    Diagnostic as LSPDiagnostic,
    HoverResult,
    LSPState,
)

from encre.tools.builtin import (
    EncreFileReadTool,
    EncreFileWriteTool,
    EncreFileEditTool,
    EncreApplyPatchTool,
    EncreBashTool,
    EncreBashOutputTool,
    EncreBashKillTool,
    EncreBashListTool,
    EncreGrepTool,
    EncreGlobTool,
    EncreWebFetchTool,
    EncreWebSearchTool,
    EncreTodoTool,
    EncreTaskCreateTool,
    EncreTaskGetTool,
    EncreTaskListTool,
    EncreTaskUpdateTool,
    EncreTaskStopTool,
    EncreTaskOutputTool,
    EncreCronCreateTool,
    EncreCronDeleteTool,
    EncreCronListTool,
    EncreAgentTool,
    EncreFindToolTool,
    EncreLSPTool,
    EncreBrowserTool,
    EncreDatabaseTool,
    EncreDockerTool,
    EncreGitTool,
    EncreRESTTool,
    EncrePDFTool,
    EncreSpreadsheetTool,
    EncreImageTool,
    EncreDeployTool,
    EncreDesktopTool,
)
from encre.tools.builtin.notebook import EncreNotebookTool
from encre.computer.browser import EncreBrowserSession, BrowserState, BrowserViewport
from encre.computer.desktop import EncreDesktopSession, DesktopScreenState, DesktopLocateResult
from encre.swarm.teammate import EncreTeammate, TeammateHandle
from encre.swarm.mailbox import EncreMailbox, MailboxMessage
from encre.swarm.manager import EncreSwarmManager, SwarmProgress
from encre.swarm.planner import EncreTaskPlanner, TaskTree, TaskNode
from encre.swarm.roles import AgentRole, RoleRegistry
from encre.swarm.orchestrator import EncreOrchestrator, OrchestrationEvent
from encre.swarm.blackboard import EncreBlackboard, BlackboardEntry
from encre.swarm.consensus import EncreConsensus, Proposal, Vote, ConsensusResult
from encre.swarm.session import EncreSwarmSession, SwarmEvent, SwarmResult
from encre.task.manager import EncreTaskManager
from encre.task.executor import EncreTaskExecutor
from encre.task.types import EncreTask
from encre.prompts.base import EncreBasePrompt, EncrePromptTemplate
from encre.prompts.system import EncrePromptBuilder
from encre.prompts.coding import EncreCodingPrompt
from encre.prompts.general import EncreGeneralPrompt
from encre.prompts.research import EncreResearchPrompt
from encre.prompts.data import EncreDataPrompt
from encre.ssrf import EncreSSRFGuard
from encre.ratelimit import EncreRateLimiter, RateLimitResult
from encre.notebook.session import EncreNotebookSession
from encre.codebase.indexer import EncreCodeIndex, ModuleInfo
from encre.feedback.learner import EncreFeedbackLearner, CorrectionRecord
from encre.backends.failover import FailoverBackend, BackendHealth
from encre.backends.router import RouterBackend, CostTracker, TaskCategory
from encre.backends.catalog import (
    PROVIDERS as MODEL_PROVIDERS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    get_provider,
    get_model,
    default_output_tokens,
    catalog_payload,
)
from encre.learning import LearningEngine, SkillGenerator, MemoryConsolidator as LearningConsolidator
from encre.skills.types import BundledSkillDefinition
from encre.skills.registry import EncreSkillRegistry
from encre.skills.bundled import create_bundled_skills
from encre.utils.types import (
    AdaptiveThinking,
    BackendError,
    BackendEvent,
    BackendFinish,
    BackendText,
    BackendToolCall,
    BackendToolCallDelta,
    DisabledThinking,
    EnabledThinking,
    Finish,
    FinishReason,
    PermissionAllow,
    PermissionAsk,
    PermissionBehavior,
    PermissionDecision,
    PermissionDeny,
    PermissionMode,
    PermissionRequest,
    PlanUpdate,
    TaskStatus,
    TaskType,
    TextDelta,
    ThinkingConfig,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolProgress,
    ToolResult,
    create_backend_error,
    create_backend_finish,
    create_backend_text,
    create_backend_tool_call,
    create_backend_tool_call_delta,
    create_finish,
    create_permission_request,
    create_text_delta,
    create_thinking_delta,
    create_tool_call_delta,
    create_tool_call_end,
    create_tool_call_start,
    create_tool_progress,
    create_tool_result,
)

__all__ = [
    "encrypt", "decrypt", "encrypt_bytes", "decrypt_bytes", "ensure_keyfile",
    "EncreRollbackGit", "CommitEntry",
    "RulesLoader",
    "EncreAgent",
    "EncreGoalRunner", "EncreGoalLoop", "GoalDefinition", "GoalResult", "GoalStatus", "GoalEvent",
    "EncreScheduler", "ScheduledJob", "CronSchedule", "ScheduleType", "JobState",
    "ErrorRecoveryEngine", "RetryableExecutor", "RecoveryState", "RecoveryDecision",
    "RecoveryAction", "ErrorCategory",
    "EncreLoop",
    "EncreSession", "SessionCheckpoint", "BranchMeta",
    "BranchIDGenerator",
    "EncreTelemetry", "ToolCallRecord", "TurnRecord", "RetryRecord",
    "setup_logging", "get_logger",
    "RetryConfig", "retry_with_backoff", "DEFAULT_RETRY_CONFIG",
    "native_read_file", "native_write_file", "native_grep", "native_glob",
    "native_count_tokens", "native_compute_diff", "native_apply_diff",
    "native_sandbox_execute", "native_search_codebase",
    "EncreSafetyEngine", "BashAnalysis", "DangerLevel", "analyze_bash_command",
    "EncreAutoSafetyClassifier", "AutoDecision", "ClassificationResult", "UserDecisionRecord",
    "EncreConfig", "ModelConfig", "get_data_dir",
    "EncreTool", "ToolRegistry", "ToolDiscovery", "BASE_TOOLS",
    "EncreMCPTool", "MCPManager", "MCPServerSpec", "bootstrap_mcp_servers", "default_mcp_config_path",
    "EncreMemorySystem", "MemoryHeader", "EntrypointResult",
    "SemanticMemorySearch", "SearchResult", "WorkingMemory", "MemoryConsolidator", "ConsolidationAction",
    "EncreProfileSystem", "UserProfile",
    "EncreContainerSandbox", "SandboxConfig", "SandboxResult",
    "EncreFileReadTool", "EncreFileWriteTool", "EncreFileEditTool", "EncreApplyPatchTool",
    "EncreBashTool", "EncreBashOutputTool", "EncreBashKillTool", "EncreBashListTool",
    "EncreGrepTool", "EncreGlobTool", "EncreWebFetchTool", "EncreWebSearchTool",
    "EncreTodoTool", "EncreTaskCreateTool", "EncreTaskGetTool", "EncreTaskListTool",
    "EncreTaskUpdateTool", "EncreTaskStopTool", "EncreTaskOutputTool",
    "EncreCronCreateTool", "EncreCronDeleteTool", "EncreCronListTool",
    "EncreAgentTool", "EncreFindToolTool",
    "EncreNotebookTool", "EncreDatabaseTool", "EncreDockerTool", "EncreGitTool",
    "EncreRESTTool", "EncrePDFTool", "EncreSpreadsheetTool", "EncreImageTool",
    "EncreDeployTool", "EncreDesktopTool",
    "EncreNotebookSession",
    "EncreSSRFGuard", "EncreRateLimiter", "RateLimitResult",
    "EncreCodeIndex", "ModuleInfo",
    "EncreFeedbackLearner", "CorrectionRecord",
    "EncreTaskManager", "EncreTaskExecutor", "EncreTask",
    "EncreBasePrompt", "EncrePromptTemplate",
    "EncrePromptBuilder", "EncreCodingPrompt", "EncreGeneralPrompt",
    "EncreResearchPrompt", "EncreDataPrompt",
    "EncreLSPTool", "EncreBrowserTool",
    "EncreBrowserSession", "BrowserState", "BrowserViewport",
    "EncreDesktopSession", "DesktopScreenState", "DesktopLocateResult",
    "EncreEvolutionLearner", "EvolutionConfig", "SuccessRecord", "ErrorRecord",
    "EncreStrategyOptimizer", "ToolStrategy",
    "EncreReflexLoop", "ReflexResult",
    "EncreMetaCognition", "CapabilityProfile",
    "EncreTeammate", "TeammateHandle",
    "EncreMailbox", "MailboxMessage",
    "EncreSwarmManager", "SwarmProgress",
    "EncreTaskPlanner", "TaskTree", "TaskNode",
    "AgentRole", "RoleRegistry",
    "EncreOrchestrator", "OrchestrationEvent",
    "EncreBlackboard", "BlackboardEntry",
    "EncreConsensus", "Proposal", "Vote", "ConsensusResult",
    "EncreSwarmSession", "SwarmEvent", "SwarmResult",
    "EncreGitRepo", "GitState",
    "EncreGitDiff", "GitDiffResult",
    "EncreLSPClient", "EncreLSPManager",
    "Position", "LSPRange", "LSPLocation", "LSPDiagnostic", "HoverResult", "LSPState",
    "EncreSkillRegistry", "BundledSkillDefinition", "create_bundled_skills",
    "SpecDocument", "SpecSection", "SpecStatus", "EncreSpecEngine",
    "BaseBackend", "BackendRegistry", "ModelInfo",
    "OpenAIBackend", "AnthropicBackend", "OllamaBackend",
    "DeepSeekBackend", "GoogleBackend", "GroqBackend",
    "LocalBackend", "BedrockBackend", "OpenAICompatibleBackend",
    "FailoverBackend", "BackendHealth",
    "RouterBackend", "CostTracker", "TaskCategory",
    "create_backend",
    "LearningEngine", "SkillGenerator", "LearningConsolidator",
    "TextDelta", "ThinkingDelta",
    "ToolCallStart", "ToolCallDelta", "ToolCallEnd",
    "ToolProgress", "ToolResult",
    "PermissionRequest", "PlanUpdate",
    "Finish", "FinishReason",
    "PermissionMode", "PermissionBehavior",
    "PermissionAllow", "PermissionDeny", "PermissionAsk", "PermissionDecision",
    "TaskType", "TaskStatus",
    "ThinkingConfig", "AdaptiveThinking", "EnabledThinking", "DisabledThinking",
    "BackendText", "BackendToolCall", "BackendToolCallDelta",
    "BackendFinish", "BackendError", "BackendEvent",
    "create_text_delta", "create_thinking_delta",
    "create_tool_call_start", "create_tool_call_delta", "create_tool_call_end",
    "create_tool_progress", "create_tool_result",
    "create_permission_request", "create_finish",
    "create_backend_text", "create_backend_tool_call",
    "create_backend_tool_call_delta", "create_backend_finish", "create_backend_error",
    "MODEL_PROVIDERS", "DEFAULT_MAX_OUTPUT_TOKENS",
    "get_provider", "get_model", "default_output_tokens", "catalog_payload",
]
