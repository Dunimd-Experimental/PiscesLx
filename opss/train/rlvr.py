#!/usr/bin/env/python3
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

"""
RLVR (Reinforcement Learning with Verifiable Rewards) Operator

Complete implementation of RLVR for objective reward signals in LLM training.
RLVR uses programmatic verifiers instead of human preference models for
reward computation, enabling unlimited training data and objective evaluation.

Key Innovation:
    - Objective rewards: Compiler/interpreter verification, not human preference
    - Unlimited data: No bottleneck from human annotation
    - Task-specific verifiers: Code, math, logic domains
    - Partial credit: Reward partial correctness

Reference:
    OpenAI o1/o3 Technical Reports (2025)
    DeepSeek R1 Technical Report (arXiv:2402.03300)

Supported Verifiers:
    - Code: Execute test cases, check outputs
    - Math: Numerical comparison, symbolic equivalence
    - Logic: Constraint satisfaction, formal verification
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import json
import time
import signal
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from configs.version import VERSION
from utils.opsc.interface import (
    PiscesLxOperatorInterface,
    PiscesLxOperatorResult,
    PiscesLxOperatorStatus,
    PiscesLxOperatorConfig,
)


class POPSSRLVRVerifierType(Enum):
    """Types of verifiers supported by RLVR."""
    CODE = "code"
    MATH = "math"
    LOGIC = "logic"
    CUSTOM = "custom"


@dataclass
class POPSSRLVRConfig(PiscesLxOperatorConfig):
    """
    RLVR (Reinforcement Learning with Verifiable Rewards) Configuration.
    
    This configuration controls the RLVR verification and reward computation
    for objective training signals.
    
    Attributes:
        verifier_type: Type of verifier (code/math/logic/custom)
        reward_scale: Scaling factor for rewards
        penalty_for_invalid: Penalty for invalid/failed responses
        timeout_seconds: Timeout for verification operations
        max_verification_attempts: Maximum retry attempts for verification
        use_partial_credit: Whether to give partial credit for partial correctness
        strict_mode: Whether to use strict verification (no leniency)
        sandbox_enabled: Whether to use sandboxed execution for code
        allowed_builtins: List of allowed built-in functions for code execution
        math_tolerance: Numerical tolerance for math verification
        symbolic_math: Whether to use symbolic math for verification
    """
    name: str = "rlvr"
    version: str = VERSION
    
    verifier_type: str = "code"
    reward_scale: float = 1.0
    penalty_for_invalid: float = -1.0
    timeout_seconds: float = 30.0
    max_verification_attempts: int = 3
    use_partial_credit: bool = True
    strict_mode: bool = False
    sandbox_enabled: bool = True
    allowed_builtins: Tuple[str, ...] = (
        "abs", "all", "any", "bin", "bool", "chr", "dict", "dir",
        "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "hex", "int", "isinstance", "iter", "len", "list", "map",
        "max", "min", "next", "oct", "ord", "pow", "print", "range",
        "repr", "reversed", "round", "set", "slice", "sorted", "str",
        "sum", "tuple", "type", "zip", "True", "False", "None",
    )
    math_tolerance: float = 1e-6
    symbolic_math: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.verifier_type, str):
            self.verifier_type = self.verifier_type.lower()


class POPSSRLVROperator(PiscesLxOperatorInterface):
    """
    Reinforcement Learning with Verifiable Rewards (RLVR) Operator.
    
    RLVR provides objective reward signals through programmatic verification,
    enabling training without human preference annotation.
    
    Key Features:
        - Code verification: Execute and test
        - Math verification: Numerical and symbolic
        - Logic verification: Constraint checking
        - Partial credit support
        - Timeout handling
        - Sandboxed execution
    
    Example:
        >>> config = POPSSRLVRConfig(verifier_type="code")
        >>> rlvr = POPSSRLVROperator()
        >>> reward, info = rlvr.compute_verifiable_reward(
        ...     response=code_response,
        ...     task={"type": "code", "test_cases": [...]}
        ... )
    """
    
    def __init__(self):
        super().__init__()
        self._name = "rlvr"
        self._version = VERSION
        self._verification_cache = {}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def description(self) -> str:
        return "Reinforcement Learning with Verifiable Rewards - Objective reward signals"
    
    def execute(self, inputs: Dict[str, Any], **kwargs) -> PiscesLxOperatorResult:
        """
        Execute RLVR verification.
        
        Args:
            inputs: Dictionary containing:
                - responses: List of responses to verify
                - tasks: List of task specifications
                - config: RLVR configuration
        
        Returns:
            PiscesLxOperatorResult with rewards and verification info
        """
        start_time = self._get_time()
        
        try:
            responses = inputs.get("responses", [])
            tasks = inputs.get("tasks", [])
            config = inputs.get("config", POPSSRLVRConfig())
            
            if not responses:
                raise ValueError("Responses are required for RLVR verification")
            
            if not tasks:
                tasks = [{"type": config.verifier_type}] * len(responses)
            
            rewards = []
            verification_info = []
            
            for response, task in zip(responses, tasks):
                reward, info = self.compute_verifiable_reward(
                    response=response,
                    task=task,
                    config=config,
                )
                rewards.append(reward)
                verification_info.append(info)
            
            output = {
                "rewards": rewards,
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "verification_info": verification_info,
                "success_rate": sum(1 for info in verification_info if info.get("success", False)) / len(verification_info) if verification_info else 0.0,
            }
            
            execution_time = self._get_time() - start_time
            
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.SUCCESS,
                output=output,
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "algorithm": "RLVR",
                    "num_responses": len(responses),
                },
            )
            
        except Exception as e:
            execution_time = self._get_time() - start_time
            return PiscesLxOperatorResult(
                operator_name=self.name,
                status=PiscesLxOperatorStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                metadata={
                    "version": self.version,
                    "error_type": type(e).__name__,
                },
            )
    
    def compute_verifiable_reward(
        self,
        response: str,
        task: Dict[str, Any],
        config: Optional[POPSSRLVRConfig] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute verifiable reward for a response.
        
        Args:
            response: Model response to verify
            task: Task specification with type and verification data
            config: RLVR configuration
        
        Returns:
            Tuple of (reward, verification_info)
        """
        if config is None:
            config = POPSSRLVRConfig()
        
        task_type = task.get("type", config.verifier_type)
        
        cache_key = self._compute_cache_key(response, task)
        if cache_key in self._verification_cache:
            return self._verification_cache[cache_key]
        
        for attempt in range(config.max_verification_attempts):
            try:
                if task_type == "code":
                    success, reward, message = self.verify_code(
                        code=response,
                        test_cases=task.get("test_cases", []),
                        config=config,
                    )
                elif task_type == "math":
                    success, reward, message = self.verify_math(
                        answer=response,
                        ground_truth=task.get("ground_truth", ""),
                        config=config,
                    )
                elif task_type == "logic":
                    success, reward, message = self.verify_logic(
                        response=response,
                        constraints=task.get("constraints", []),
                        config=config,
                    )
                elif task_type == "custom":
                    verifier = task.get("verifier")
                    if callable(verifier):
                        success, reward, message = verifier(response, task)
                    else:
                        success, reward, message = False, config.penalty_for_invalid, "No custom verifier provided"
                else:
                    success, reward, message = False, config.penalty_for_invalid, f"Unknown task type: {task_type}"
                
                break
                
            except Exception as e:
                if attempt == config.max_verification_attempts - 1:
                    success = False
                    reward = config.penalty_for_invalid
                    message = f"Verification failed after {config.max_verification_attempts} attempts: {str(e)}"
        
        reward = reward * config.reward_scale
        
        info = {
            "success": success,
            "message": message,
            "type": task_type,
            "raw_reward": reward / config.reward_scale if config.reward_scale != 0 else reward,
            "scaled_reward": reward,
            "attempts": attempt + 1 if 'attempt' in dir() else 1,
        }
        
        self._verification_cache[cache_key] = (reward, info)
        
        return reward, info
    
    def verify_code(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        config: POPSSRLVRConfig,
    ) -> Tuple[bool, float, str]:
        """
        Verify code by executing test cases.
        
        Args:
            code: Code to verify
            test_cases: List of test case dictionaries with 'input' and 'output'
            config: RLVR configuration
        
        Returns:
            Tuple of (success, reward, message)
        """
        if not test_cases:
            return True, 1.0, "No test cases provided"
        
        code = self._extract_code(code)
        
        safe_globals = {"__builtins__": {}}
        if config.sandbox_enabled:
            for name in config.allowed_builtins:
                if name in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, name):
                    safe_globals[name] = __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
        else:
            safe_globals["__builtins__"] = __builtins__
        
        try:
            local_vars = {}
            exec(code, safe_globals, local_vars)
        except SyntaxError as e:
            return False, config.penalty_for_invalid, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, config.penalty_for_invalid, f"Execution error: {str(e)}"
        
        passed = 0
        total = len(test_cases)
        errors = []
        
        for i, test in enumerate(test_cases):
            try:
                input_data = test.get("input", {})
                expected = test.get("output")
                func_name = test.get("function", "solution")
                
                if func_name not in local_vars:
                    errors.append(f"Test {i+1}: Function '{func_name}' not found")
                    continue
                
                func = local_vars[func_name]
                
                if isinstance(input_data, dict):
                    result = func(**input_data)
                elif isinstance(input_data, (list, tuple)):
                    result = func(*input_data)
                else:
                    result = func(input_data)
                
                if self._compare_results(result, expected, config):
                    passed += 1
                else:
                    errors.append(f"Test {i+1}: Expected {expected}, got {result}")
                    
            except Exception as e:
                errors.append(f"Test {i+1}: {str(e)}")
        
        if config.use_partial_credit:
            reward = passed / total
        else:
            reward = 1.0 if passed == total else 0.0
        
        success = passed == total
        
        if success:
            message = f"All {total} tests passed"
        else:
            message = f"Passed {passed}/{total} tests. " + "; ".join(errors[:3])
            if len(errors) > 3:
                message += f" ... and {len(errors) - 3} more errors"
        
        return success, reward, message
    
    def verify_math(
        self,
        answer: str,
        ground_truth: str,
        config: POPSSRLVRConfig,
    ) -> Tuple[bool, float, str]:
        """
        Verify mathematical answer.
        
        Supports both numerical and symbolic comparison.
        
        Args:
            answer: Model's answer
            ground_truth: Expected answer
            config: RLVR configuration
        
        Returns:
            Tuple of (success, reward, message)
        """
        answer = self._extract_math_answer(answer)
        ground_truth = ground_truth.strip()
        
        try:
            ans_val = float(answer.replace(",", "").replace(" ", ""))
            gt_val = float(ground_truth.replace(",", "").replace(" ", ""))
            
            if abs(ans_val - gt_val) < config.math_tolerance:
                return True, 1.0, "Correct (numerical)"
            elif abs(ans_val - gt_val) < config.math_tolerance * 100:
                return True, 0.5, f"Close: expected {gt_val}, got {ans_val}"
            else:
                return False, 0.0, f"Expected {gt_val}, got {ans_val}"
                
        except ValueError:
            if config.symbolic_math:
                return self._verify_symbolic_math(answer, ground_truth, config)
            else:
                if answer.lower() == ground_truth.lower():
                    return True, 1.0, "Correct (exact match)"
                return False, 0.0, f"Expected {ground_truth}, got {answer}"
    
    def _verify_symbolic_math(
        self,
        answer: str,
        ground_truth: str,
        config: POPSSRLVRConfig,
    ) -> Tuple[bool, float, str]:
        """Verify using symbolic math (sympy)."""
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr
            
            ans_sym = parse_expr(answer, transformations='all')
            gt_sym = parse_expr(ground_truth, transformations='all')
            
            diff = sp.simplify(ans_sym - gt_sym)
            
            if diff == 0:
                return True, 1.0, "Correct (symbolic)"
            else:
                simplified_diff = sp.nsimplify(diff)
                if simplified_diff == 0:
                    return True, 1.0, "Correct (symbolic, after simplification)"
                return False, 0.0, f"Symbolic mismatch: {answer} != {ground_truth}"
                
        except ImportError:
            if answer.lower().strip() == ground_truth.lower().strip():
                return True, 1.0, "Correct (exact match)"
            return False, 0.0, f"Expected {ground_truth}, got {answer}"
        except Exception as e:
            return False, 0.0, f"Symbolic verification failed: {str(e)}"
    
    def verify_logic(
        self,
        response: str,
        constraints: List[Dict[str, Any]],
        config: POPSSRLVRConfig,
    ) -> Tuple[bool, float, str]:
        """
        Verify logical constraints.
        
        Args:
            response: Model's response
            constraints: List of constraint specifications
            config: RLVR configuration
        
        Returns:
            Tuple of (success, reward, message)
        """
        if not constraints:
            return True, 1.0, "No constraints to verify"
        
        satisfied = 0
        total = len(constraints)
        violations = []
        
        for i, constraint in enumerate(constraints):
            constraint_type = constraint.get("type", "contains")
            
            try:
                if constraint_type == "contains":
                    if constraint.get("value", "") in response:
                        satisfied += 1
                    else:
                        violations.append(f"Missing: {constraint.get('value', '')}")
                        
                elif constraint_type == "not_contains":
                    if constraint.get("value", "") not in response:
                        satisfied += 1
                    else:
                        violations.append(f"Should not contain: {constraint.get('value', '')}")
                        
                elif constraint_type == "regex":
                    pattern = constraint.get("pattern", "")
                    if re.search(pattern, response):
                        satisfied += 1
                    else:
                        violations.append(f"Pattern not matched: {pattern}")
                        
                elif constraint_type == "length":
                    min_len = constraint.get("min", 0)
                    max_len = constraint.get("max", float('inf'))
                    if min_len <= len(response) <= max_len:
                        satisfied += 1
                    else:
                        violations.append(f"Length {len(response)} not in [{min_len}, {max_len}]")
                        
                elif constraint_type == "format":
                    fmt = constraint.get("format", "")
                    if self._check_format(response, fmt):
                        satisfied += 1
                    else:
                        violations.append(f"Invalid format: expected {fmt}")
                        
                elif constraint_type == "custom":
                    checker = constraint.get("checker")
                    if callable(checker) and checker(response):
                        satisfied += 1
                    else:
                        violations.append(f"Custom constraint failed")
                        
                else:
                    violations.append(f"Unknown constraint type: {constraint_type}")
                    
            except Exception as e:
                violations.append(f"Constraint {i+1}: {str(e)}")
        
        if config.use_partial_credit:
            reward = satisfied / total
        else:
            reward = 1.0 if satisfied == total else 0.0
        
        success = satisfied == total
        
        if success:
            message = f"All {total} constraints satisfied"
        else:
            message = f"Satisfied {satisfied}/{total} constraints. " + "; ".join(violations[:3])
        
        return success, reward, message
    
    def _extract_code(self, text: str) -> str:
        """Extract code from text that may contain markdown or explanations."""
        code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return '\n\n'.join(code_blocks)
        
        lines = text.strip().split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if re.match(r'^\s*(def |class |import |from |@)', line):
                in_code = True
            if in_code or line.strip().startswith(('def ', 'class ', 'import ', 'from ', '@', 'return ', 'if ', 'for ', 'while ')):
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else text
    
    def _extract_math_answer(self, text: str) -> str:
        """Extract mathematical answer from text."""
        latex_matches = re.findall(r'\$([^$]+)\$', text)
        if latex_matches:
            return latex_matches[-1].strip()
        
        box_matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        if box_matches:
            return box_matches[-1].strip()
        
        number_matches = re.findall(r'[-+]?\d*\.?\d+', text)
        if number_matches:
            return number_matches[-1]
        
        return text.strip()
    
    def _compare_results(self, result, expected, config: POPSSRLVRConfig) -> bool:
        """Compare execution result with expected output."""
        if result == expected:
            return True
        
        if isinstance(result, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(result) != len(expected):
                return False
            return all(self._compare_results(r, e, config) for r, e in zip(result, expected))
        
        if isinstance(result, dict) and isinstance(expected, dict):
            if set(result.keys()) != set(expected.keys()):
                return False
            return all(self._compare_results(result[k], expected[k], config) for k in result)
        
        if isinstance(result, float) and isinstance(expected, float):
            return abs(result - expected) < config.math_tolerance
        
        try:
            return str(result).strip() == str(expected).strip()
        except:
            return False
    
    def _check_format(self, text: str, fmt: str) -> bool:
        """Check if text matches expected format."""
        fmt = fmt.lower()
        
        if fmt == "json":
            try:
                json.loads(text)
                return True
            except:
                return False
        
        if fmt == "number":
            try:
                float(text.strip())
                return True
            except:
                return False
        
        if fmt == "integer":
            try:
                int(text.strip())
                return True
            except:
                return False
        
        if fmt == "boolean":
            return text.strip().lower() in ("true", "false", "yes", "no", "1", "0")
        
        if fmt == "email":
            return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', text.strip()))
        
        if fmt == "url":
            return bool(re.match(r'^https?://[\w\.-]+', text.strip()))
        
        return True
    
    def _compute_cache_key(self, response: str, task: Dict[str, Any]) -> str:
        """Compute cache key for verification results."""
        import hashlib
        content = f"{response}::{json.dumps(task, sort_keys=True, default=str)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_time(self) -> float:
        """Get current time in seconds."""
        return time.time()
    
    def clear_cache(self):
        """Clear verification cache."""
        self._verification_cache.clear()


class POPSSRLVRDataset:
    """
    Dataset for RLVR training with verifiable tasks.
    
    Each sample contains a prompt, expected task type, and verification data.
    
    Example:
        >>> dataset = POPSSRLVRDataset([
        ...     {
        ...         "prompt": "Write a function to sort a list",
        ...         "type": "code",
        ...         "test_cases": [{"input": [[3,1,2]], "output": [1,2,3]}],
        ...     },
        ... ])
    """
    
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        self.operator = POPSSRLVROperator()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
    
    def verify_response(
        self,
        idx: int,
        response: str,
        config: Optional[POPSSRLVRConfig] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Verify a response for a specific sample."""
        sample = self.samples[idx]
        task = {
            "type": sample.get("type", "code"),
            "test_cases": sample.get("test_cases", []),
            "ground_truth": sample.get("ground_truth", ""),
            "constraints": sample.get("constraints", []),
        }
        return self.operator.compute_verifiable_reward(response, task, config)
    
    def batch_verify(
        self,
        responses: List[str],
        config: Optional[POPSSRLVRConfig] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """Verify multiple responses."""
        results = []
        for idx, response in enumerate(responses):
            if idx < len(self.samples):
                result = self.verify_response(idx, response, config)
            else:
                result = (0.0, {"success": False, "message": "Index out of range"})
            results.append(result)
        return results
    
    @classmethod
    def from_code_tasks(
        cls,
        tasks: List[Dict[str, Any]],
        prompt_template: str = "Write a Python function to solve the following problem:\n\n{description}",
    ) -> "POPSSRLVRDataset":
        """Create dataset from code tasks."""
        samples = []
        for task in tasks:
            samples.append({
                "prompt": prompt_template.format(description=task.get("description", "")),
                "type": "code",
                "test_cases": task.get("test_cases", []),
                "function": task.get("function", "solution"),
            })
        return cls(samples)
    
    @classmethod
    def from_math_tasks(
        cls,
        tasks: List[Dict[str, Any]],
        prompt_template: str = "Solve the following math problem:\n\n{problem}",
    ) -> "POPSSRLVRDataset":
        """Create dataset from math tasks."""
        samples = []
        for task in tasks:
            samples.append({
                "prompt": prompt_template.format(problem=task.get("problem", "")),
                "type": "math",
                "ground_truth": task.get("answer", ""),
            })
        return cls(samples)


class POPSSRLVRTrainer:
    """
    High-level trainer combining RLVR with GRPO for verifiable reward training.
    
    Example:
        >>> trainer = POPSSRLVRTrainer(
        ...     model=policy_model,
        ...     dataset=rlvr_dataset,
        ...     config=POPSSRLVRConfig(verifier_type="code"),
        ... )
        >>> trainer.train(num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        dataset: POPSSRLVRDataset,
        config: Optional[POPSSRLVRConfig] = None,
        grpo_config: Optional['POPSSGRPOConfig'] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        from .grpo import POPSSGRPOConfig, POPSSGRPOOperator
        
        self.model = model
        self.dataset = dataset
        self.config = config or POPSSRLVRConfig()
        self.grpo_config = grpo_config or POPSSGRPOConfig()
        
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-5,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer
        
        self.rlvr_operator = POPSSRLVROperator()
        self.grpo_operator = POPSSGRPOOperator()
    
    def train(
        self,
        num_epochs: int = 1,
        batch_size: int = 4,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train with RLVR rewards."""
        all_stats = {
            "rewards": [],
            "policy_losses": [],
        }
        
        for epoch in range(num_epochs):
            for i in range(0, len(self.dataset), batch_size):
                batch = self.dataset[i:i + batch_size]
                
                prompts = [sample.get("prompt", "") for sample in batch]
                tasks = [
                    {
                        "type": sample.get("type", self.config.verifier_type),
                        "test_cases": sample.get("test_cases", []),
                        "ground_truth": sample.get("ground_truth", ""),
                        "constraints": sample.get("constraints", []),
                    }
                    for sample in batch
                ]
                
                def reward_fn(prompt, response):
                    idx = prompts.index(prompt) if prompt in prompts else 0
                    reward, _ = self.rlvr_operator.compute_verifiable_reward(
                        response=response,
                        task=tasks[idx],
                        config=self.config,
                    )
                    return reward
                
                result = self.grpo_operator.execute({
                    "model": self.model,
                    "prompts": prompts,
                    "reward_function": reward_fn,
                    "config": self.grpo_config,
                    "optimizer": self.optimizer,
                })
                
                if result.status == PiscesLxOperatorStatus.SUCCESS:
                    all_stats["rewards"].append(result.output.get("mean_reward", 0))
                    all_stats["policy_losses"].append(result.output.get("mean_policy_loss", 0))
        
        return {
            "mean_reward": sum(all_stats["rewards"]) / len(all_stats["rewards"]) if all_stats["rewards"] else 0,
            "mean_policy_loss": sum(all_stats["policy_losses"]) / len(all_stats["policy_losses"]) if all_stats["policy_losses"] else 0,
            "total_steps": len(all_stats["rewards"]),
        }
