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
Agent benchmark evaluation suite for PiscesLx.

This module provides comprehensive agent capability evaluation including:
- AgentBench: Multi-task agent evaluation
- WebShop: E-commerce web navigation
- WebArena: Web-based task completion
- OSWorld: Operating system interaction
- SWE-bench: Software engineering tasks
- ToolBench: Tool usage evaluation
"""

import os
import json
import time
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
import threading

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir

_LOG = PiscesLxLogger("PiscesLx.Tools.Benchmark.Agent", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)


@dataclass
class PiscesLxToolsAgentConfig:
    """Configuration for agent benchmark evaluation."""
    
    model_path: str = ".pisceslx/ckpt"
    output_dir: str = ".pisceslx/benchmark/agent"
    
    max_steps: int = 30
    max_retries: int = 3
    timeout: int = 300
    
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    device: str = "cuda"
    
    agent_benchmarks: List[str] = field(default_factory=lambda: [
        "agentbench", "webshop", "webarena", "osworld", "swebench", "toolbench"
    ])
    
    save_results: bool = True
    verbose: bool = True


class PiscesLxToolsAgentMetrics:
    """Metrics for agent evaluation."""
    
    @staticmethod
    def success_rate(results: List[Dict]) -> float:
        """Calculate task success rate."""
        if not results:
            return 0.0
        successful = sum(1 for r in results if r.get("success", False))
        return successful / len(results)
    
    @staticmethod
    def tool_usage_accuracy(results: List[Dict]) -> float:
        """Calculate tool usage accuracy."""
        if not results:
            return 0.0
        correct = 0
        total = 0
        for r in results:
            tool_calls = r.get("tool_calls", [])
            expected = r.get("expected_tools", [])
            if tool_calls and expected:
                total += 1
                if tool_calls == expected:
                    correct += 1
        return correct / max(1, total)
    
    @staticmethod
    def task_completion_rate(results: List[Dict]) -> float:
        """Calculate task completion rate."""
        if not results:
            return 0.0
        completed = sum(1 for r in results if r.get("completed", False))
        return completed / len(results)
    
    @staticmethod
    def reasoning_quality(results: List[Dict]) -> float:
        """Calculate reasoning quality score."""
        if not results:
            return 0.0
        scores = []
        for r in results:
            if "reasoning_steps" in r and "expected_steps" in r:
                if r["expected_steps"] > 0:
                    score = min(1.0, r["reasoning_steps"] / r["expected_steps"])
                    scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0
    
    @staticmethod
    def step_efficiency(results: List[Dict]) -> float:
        """Calculate step efficiency (lower is better)."""
        if not results:
            return 0.0
        efficiencies = []
        for r in results:
            if "steps_taken" in r and "optimal_steps" in r:
                if r["optimal_steps"] > 0:
                    efficiency = r["optimal_steps"] / max(1, r["steps_taken"])
                    efficiencies.append(min(1.0, efficiency))
        return sum(efficiencies) / len(efficiencies) if efficiencies else 0.0


class PiscesLxToolsAgentEvaluator:
    """Agent benchmark evaluator for PiscesLx model."""
    
    def __init__(
        self,
        config: PiscesLxToolsAgentConfig,
        model: nn.Module,
        tokenizer: Any,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
        self.metrics = PiscesLxToolsAgentMetrics()
        
        _LOG.info("PiscesLxToolsAgentEvaluator initialized")
    
    def evaluate_agentbench(self) -> Dict[str, float]:
        """Evaluate on AgentBench - Multi-task agent evaluation."""
        _LOG.info("Evaluating AgentBench...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("THUDM/AgentBench", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load AgentBench: {e}")
            return {"success_rate": 0.0, "total": 0}
        
        results = []
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="AgentBench"):
            try:
                task = item.get("task", "")
                instruction = item.get("instruction", "")
                expected_output = item.get("expected_output", "")
                tools = item.get("tools", [])
                
                if not task or not instruction:
                    continue
                
                prompt = self._build_agent_prompt(task, instruction, tools)
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=4096,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=1024,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                success = self._evaluate_agent_response(generated, expected_output)
                
                results.append({
                    "task": task,
                    "success": success,
                    "generated": generated,
                    "expected": expected_output,
                })
                
            except Exception as e:
                _LOG.debug(f"AgentBench sample error: {e}")
                continue
        
        success_rate = self.metrics.success_rate(results)
        
        self.results["agentbench"] = {
            "success_rate": success_rate,
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
        }
        
        _LOG.info(f"AgentBench Success Rate: {success_rate:.4f}")
        return self.results["agentbench"]
    
    def evaluate_webshop(self) -> Dict[str, float]:
        """Evaluate on WebShop - E-commerce web navigation."""
        _LOG.info("Evaluating WebShop...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("princeton-nlp/WebShop", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load WebShop: {e}")
            return {"success_rate": 0.0, "total": 0}
        
        results = []
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="WebShop"):
            try:
                instruction = item.get("instruction", "")
                product_id = item.get("product_id", "")
                expected_actions = item.get("actions", [])
                
                if not instruction:
                    continue
                
                prompt = self._build_webshop_prompt(instruction)
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=4096,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=512,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                actions = self._parse_webshop_actions(generated)
                success = self._evaluate_webshop_actions(actions, expected_actions)
                
                results.append({
                    "instruction": instruction,
                    "success": success,
                    "actions": actions,
                    "expected": expected_actions,
                })
                
            except Exception as e:
                _LOG.debug(f"WebShop sample error: {e}")
                continue
        
        success_rate = self.metrics.success_rate(results)
        
        self.results["webshop"] = {
            "success_rate": success_rate,
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
        }
        
        _LOG.info(f"WebShop Success Rate: {success_rate:.4f}")
        return self.results["webshop"]
    
    def evaluate_webarena(self) -> Dict[str, float]:
        """Evaluate on WebArena - Web-based task completion."""
        _LOG.info("Evaluating WebArena...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("webarena/webarena", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load WebArena: {e}")
            return {"success_rate": 0.0, "total": 0}
        
        results = []
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="WebArena"):
            try:
                task = item.get("task", "")
                intent = item.get("intent", "")
                expected = item.get("expected", "")
                
                if not task or not intent:
                    continue
                
                prompt = self._build_webarena_prompt(task, intent)
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=4096,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=1024,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                success = self._evaluate_webarena_response(generated, expected)
                
                results.append({
                    "task": task,
                    "success": success,
                    "generated": generated,
                    "expected": expected,
                })
                
            except Exception as e:
                _LOG.debug(f"WebArena sample error: {e}")
                continue
        
        success_rate = self.metrics.success_rate(results)
        
        self.results["webarena"] = {
            "success_rate": success_rate,
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
        }
        
        _LOG.info(f"WebArena Success Rate: {success_rate:.4f}")
        return self.results["webarena"]
    
    def evaluate_osworld(self) -> Dict[str, float]:
        """Evaluate on OSWorld - Operating system interaction."""
        _LOG.info("Evaluating OSWorld...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("xlang-ai/OSWorld", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load OSWorld: {e}")
            return {"success_rate": 0.0, "total": 0}
        
        results = []
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="OSWorld"):
            try:
                instruction = item.get("instruction", "")
                expected_commands = item.get("commands", [])
                platform = item.get("platform", "linux")
                
                if not instruction:
                    continue
                
                prompt = self._build_osworld_prompt(instruction, platform)
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=4096,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=512,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                commands = self._parse_shell_commands(generated)
                success = self._evaluate_commands(commands, expected_commands)
                
                results.append({
                    "instruction": instruction,
                    "success": success,
                    "commands": commands,
                    "expected": expected_commands,
                })
                
            except Exception as e:
                _LOG.debug(f"OSWorld sample error: {e}")
                continue
        
        success_rate = self.metrics.success_rate(results)
        
        self.results["osworld"] = {
            "success_rate": success_rate,
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
        }
        
        _LOG.info(f"OSWorld Success Rate: {success_rate:.4f}")
        return self.results["osworld"]
    
    def evaluate_swebench(self) -> Dict[str, float]:
        """Evaluate on SWE-bench - Software engineering tasks."""
        _LOG.info("Evaluating SWE-bench...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load SWE-bench: {e}")
            return {"success_rate": 0.0, "total": 0}
        
        results = []
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="SWE-bench"):
            try:
                problem = item.get("problem_statement", "")
                repo = item.get("repo", "")
                base_commit = item.get("base_commit", "")
                patch = item.get("patch", "")
                
                if not problem:
                    continue
                
                prompt = self._build_swebench_prompt(problem, repo)
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=8192,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=2048,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                generated_patch = self._extract_patch(generated)
                success = self._evaluate_patch(generated_patch, patch)
                
                results.append({
                    "problem": problem[:100],
                    "success": success,
                    "generated_patch": generated_patch,
                    "expected_patch": patch,
                })
                
            except Exception as e:
                _LOG.debug(f"SWE-bench sample error: {e}")
                continue
        
        success_rate = self.metrics.success_rate(results)
        
        self.results["swebench"] = {
            "success_rate": success_rate,
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
        }
        
        _LOG.info(f"SWE-bench Success Rate: {success_rate:.4f}")
        return self.results["swebench"]
    
    def evaluate_toolbench(self) -> Dict[str, float]:
        """Evaluate on ToolBench - Tool usage evaluation."""
        _LOG.info("Evaluating ToolBench...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("openbmb/ToolBench", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load ToolBench: {e}")
            return {"success_rate": 0.0, "total": 0}
        
        results = []
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="ToolBench"):
            try:
                query = item.get("query", "")
                available_tools = item.get("available_tools", [])
                expected_tool_calls = item.get("expected_tool_calls", [])
                
                if not query:
                    continue
                
                prompt = self._build_toolbench_prompt(query, available_tools)
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=4096,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=encoding["input_ids"],
                        attention_mask=encoding["attention_mask"],
                        max_new_tokens=512,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        do_sample=self.config.do_sample,
                    )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                tool_calls = self._parse_tool_calls(generated)
                success = self._evaluate_tool_calls(tool_calls, expected_tool_calls)
                
                results.append({
                    "query": query,
                    "success": success,
                    "tool_calls": tool_calls,
                    "expected": expected_tool_calls,
                })
                
            except Exception as e:
                _LOG.debug(f"ToolBench sample error: {e}")
                continue
        
        success_rate = self.metrics.success_rate(results)
        tool_accuracy = self.metrics.tool_usage_accuracy(results)
        
        self.results["toolbench"] = {
            "success_rate": success_rate,
            "tool_accuracy": tool_accuracy,
            "total": len(results),
            "successful": sum(1 for r in results if r["success"]),
        }
        
        _LOG.info(f"ToolBench Success Rate: {success_rate:.4f}")
        return self.results["toolbench"]
    
    def _build_agent_prompt(self, task: str, instruction: str, tools: List[str]) -> str:
        """Build prompt for agent task."""
        prompt = f"""You are an AI agent. Complete the following task.

Task: {task}

Instruction: {instruction}

"""
        if tools:
            prompt += "Available Tools:\n"
            for tool in tools:
                prompt += f"- {tool}\n"
            prompt += "\n"
        
        prompt += "Provide your solution step by step. Use the format:\n"
        prompt += "Thought: [your reasoning]\n"
        prompt += "Action: [action to take]\n"
        prompt += "Observation: [result of action]\n"
        prompt += "...\n"
        prompt += "Final Answer: [your final answer]\n\n"
        prompt += "Response:"
        
        return prompt
    
    def _build_webshop_prompt(self, instruction: str) -> str:
        """Build prompt for WebShop task."""
        return f"""You are shopping on an e-commerce website. Complete the following task.

Task: {instruction}

Available actions:
- search[query]
- click[element]
- buy[item]

Provide your actions in order, one per line.

Actions:"""
    
    def _build_webarena_prompt(self, task: str, intent: str) -> str:
        """Build prompt for WebArena task."""
        return f"""You are navigating a website. Complete the following task.

Task: {task}
Intent: {intent}

Available actions:
- goto[url]
- click[element]
- type[text]
- scroll[direction]

Provide your actions in order.

Actions:"""
    
    def _build_osworld_prompt(self, instruction: str, platform: str) -> str:
        """Build prompt for OSWorld task."""
        return f"""You are using a {platform} operating system. Complete the following task.

Task: {instruction}

Provide shell commands to complete the task. One command per line.

Commands:"""
    
    def _build_swebench_prompt(self, problem: str, repo: str) -> str:
        """Build prompt for SWE-bench task."""
        return f"""You are a software engineer. Fix the following bug.

Repository: {repo}

Problem:
{problem}

Provide a git diff patch to fix the issue.

Patch:"""
    
    def _build_toolbench_prompt(self, query: str, tools: List[str]) -> str:
        """Build prompt for ToolBench task."""
        prompt = f"""You have access to the following tools. Use them to complete the task.

Query: {query}

Available Tools:
"""
        for tool in tools:
            prompt += f"- {tool}\n"
        
        prompt += """
Use tools in this format:
tool_name[arg1, arg2, ...]

Your response:"""
        
        return prompt
    
    def _evaluate_agent_response(self, generated: str, expected: str) -> bool:
        """Evaluate agent response against expected output."""
        generated_lower = generated.lower()
        expected_lower = expected.lower()
        
        if expected_lower in generated_lower:
            return True
        
        expected_keywords = set(expected_lower.split())
        generated_keywords = set(generated_lower.split())
        overlap = len(expected_keywords & generated_keywords)
        
        return overlap >= len(expected_keywords) * 0.5
    
    def _parse_webshop_actions(self, text: str) -> List[str]:
        """Parse WebShop actions from text."""
        actions = []
        patterns = [
            r'search\[(.*?)\]',
            r'click\[(.*?)\]',
            r'buy\[(.*?)\]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            actions.extend(matches)
        
        return actions
    
    def _evaluate_webshop_actions(self, actions: List[str], expected: List[str]) -> bool:
        """Evaluate WebShop actions."""
        if not expected:
            return len(actions) > 0
        
        correct = 0
        for exp in expected:
            for act in actions:
                if exp.lower() in act.lower() or act.lower() in exp.lower():
                    correct += 1
                    break
        
        return correct >= len(expected) * 0.7
    
    def _evaluate_webarena_response(self, generated: str, expected: str) -> bool:
        """Evaluate WebArena response."""
        generated_lower = generated.lower()
        expected_lower = expected.lower()
        
        return expected_lower in generated_lower or generated_lower in expected_lower
    
    def _parse_shell_commands(self, text: str) -> List[str]:
        """Parse shell commands from text."""
        commands = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('$ '):
                    line = line[2:]
                commands.append(line)
        
        return commands
    
    def _evaluate_commands(self, commands: List[str], expected: List[str]) -> bool:
        """Evaluate shell commands."""
        if not expected:
            return len(commands) > 0
        
        correct = 0
        for exp in expected:
            for cmd in commands:
                if exp.lower() in cmd.lower():
                    correct += 1
                    break
        
        return correct >= len(expected) * 0.7
    
    def _extract_patch(self, text: str) -> str:
        """Extract git patch from text."""
        patch_patterns = [
            r'```diff\n(.*?)```',
            r'```\n(.*?)```',
            r'diff --git.*?(?=diff --git|$)',
        ]
        
        for pattern in patch_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        
        return text
    
    def _evaluate_patch(self, generated: str, expected: str) -> bool:
        """Evaluate generated patch."""
        if not expected:
            return len(generated) > 0
        
        generated_lines = set(generated.strip().split('\n'))
        expected_lines = set(expected.strip().split('\n'))
        
        overlap = len(generated_lines & expected_lines)
        
        return overlap >= len(expected_lines) * 0.5
    
    def _parse_tool_calls(self, text: str) -> List[Dict]:
        """Parse tool calls from text."""
        tool_calls = []
        pattern = r'(\w+)\[([^\]]*)\]'
        
        matches = re.findall(pattern, text)
        for match in matches:
            tool_name = match[0]
            args = [arg.strip() for arg in match[1].split(',')]
            tool_calls.append({
                "tool": tool_name,
                "args": args,
            })
        
        return tool_calls
    
    def _evaluate_tool_calls(self, generated: List[Dict], expected: List[Dict]) -> bool:
        """Evaluate tool calls."""
        if not expected:
            return len(generated) > 0
        
        correct = 0
        for exp in expected:
            exp_tool = exp.get("tool", "").lower()
            for gen in generated:
                gen_tool = gen.get("tool", "").lower()
                if exp_tool == gen_tool:
                    correct += 1
                    break
        
        return correct >= len(expected) * 0.7
    
    def run_all_agent_benchmarks(self) -> Dict[str, Any]:
        """Run all agent benchmarks."""
        _LOG.info("Running all agent benchmarks...")
        
        benchmarks_map = {
            "agentbench": self.evaluate_agentbench,
            "webshop": self.evaluate_webshop,
            "webarena": self.evaluate_webarena,
            "osworld": self.evaluate_osworld,
            "swebench": self.evaluate_swebench,
            "toolbench": self.evaluate_toolbench,
        }
        
        for benchmark in self.config.agent_benchmarks:
            if benchmark in benchmarks_map:
                try:
                    benchmarks_map[benchmark]()
                except Exception as e:
                    _LOG.error(f"Agent benchmark {benchmark} failed: {e}")
        
        self.results["agent_summary"] = self._generate_agent_summary()
        
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _generate_agent_summary(self) -> Dict[str, float]:
        """Generate agent benchmark summary."""
        summary = {}
        for name, result in self.results.items():
            if name == "agent_summary":
                continue
            if isinstance(result, dict) and "success_rate" in result:
                summary[name] = result["success_rate"]
        
        if summary:
            summary["average"] = sum(summary.values()) / len(summary)
        
        return summary
    
    def _save_results(self) -> None:
        """Save results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"agent_benchmark_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Agent benchmark results saved to {output_path}")


class PiscesLxToolsAgentBenchmarkRunner:
    """Runner for agent benchmarks."""
    
    def __init__(
        self,
        config: PiscesLxToolsAgentConfig,
        model: nn.Module,
        tokenizer: Any,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.results = {}
        
        _LOG.info("PiscesLxToolsAgentBenchmarkRunner initialized")
    
    def run_all(self) -> Dict[str, Any]:
        """Run all agent benchmarks."""
        _LOG.info("Running all agent benchmarks...")
        
        evaluator = PiscesLxToolsAgentEvaluator(
            self.config, self.model, self.tokenizer
        )
        self.results = evaluator.run_all_agent_benchmarks()
        
        return self.results
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("PiscesLx Agent Benchmark Results")
        print("=" * 60)
        
        if "agent_summary" in self.results:
            summary = self.results["agent_summary"]
            for benchmark, score in summary.items():
                if benchmark != "average":
                    print(f"  {benchmark:20s}: {score:.4f}")
            if "average" in summary:
                print(f"  {'Average':20s}: {summary['average']:.4f}")
        
        print("=" * 60 + "\n")


def create_agent_evaluator(
    config: PiscesLxToolsAgentConfig,
    model: nn.Module,
    tokenizer: Any,
) -> PiscesLxToolsAgentBenchmarkRunner:
    """Factory function to create agent benchmark runner."""
    return PiscesLxToolsAgentBenchmarkRunner(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
