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

"""
Inference Pipeline Operator
Implements end-to-end inference pipeline management.
"""

import torch
from typing import Dict, Any, Callable, Optional, List, Union
from pathlib import Path
import time
from datetime import datetime
import json
import yaml

# OPSC operator system integration
from utils.opsc.base import PiscesLxTransformOperator
from utils.opsc.interface import PiscesLxOperatorConfig
from utils.opsc.registry import PiscesLxOperatorRegistrar
from utils.dc import PiscesLxLogger

from .core import PiscesLxInferenceEngine
from .config import InferenceConfig


from utils.paths import get_log_file
_LOG = PiscesLxLogger("PiscesLx.Tools.Infer", file_path=get_log_file("PiscesLx.Tools.Infer"), enable_file=True)


@PiscesLxOperatorRegistrar()
class InferencePipelineOperator(PiscesLxTransformOperator):
    """
    Inference Pipeline Operator.
    Manages complete end-to-end inference pipeline.
    """
    
    def __init__(self, config: Optional[Union[PiscesLxOperatorConfig, InferenceConfig, str, Dict[str, Any]]] = None):
        op_config = config if isinstance(config, PiscesLxOperatorConfig) else None
        super().__init__(op_config)
        self.infer_config = self._normalize_infer_config(config)
        self.inferencer = PiscesLxInferenceEngine(self.infer_config)
        self.callbacks = []
        self.request_queue = []

    def _normalize_infer_config(self, config: Optional[Union[PiscesLxOperatorConfig, InferenceConfig, str, Dict[str, Any]]]) -> InferenceConfig:
        if isinstance(config, InferenceConfig):
            return config
        if isinstance(config, str):
            if config.lower().endswith(".json"):
                return InferenceConfig.load_from_json(config)
            if config.lower().endswith((".yaml", ".yml")):
                with open(config, "r", encoding="utf-8") as f:
                    return InferenceConfig.from_dict(yaml.safe_load(f) or {})
            raise ValueError(f"Unsupported config file format: {config}")
        if isinstance(config, dict):
            return InferenceConfig.from_dict(config)
        if isinstance(config, PiscesLxOperatorConfig):
            params = getattr(config, "parameters", {}) or {}
            cfg = params.get("inference_config", None)
            if isinstance(cfg, InferenceConfig):
                return cfg
            if isinstance(cfg, str):
                if cfg.lower().endswith(".json"):
                    return InferenceConfig.load_from_json(cfg)
                if cfg.lower().endswith((".yaml", ".yml")):
                    with open(cfg, "r", encoding="utf-8") as f:
                        return InferenceConfig.from_dict(yaml.safe_load(f) or {})
                raise ValueError(f"Unsupported config file format: {cfg}")
            if isinstance(cfg, dict):
                return InferenceConfig.from_dict(cfg)
        return InferenceConfig()

    def transform(self, data: Any) -> Any:
        if isinstance(data, dict):
            prompt = data.get("prompt", None)
            prompts = data.get("prompts", None)
            kwargs = data.get("generation", None) or {}
            if prompts is not None:
                return self.batch_inference(prompts, **kwargs)
            if prompt is not None:
                return self.batch_inference([prompt], **kwargs)[0]
            raise ValueError("Missing 'prompt' or 'prompts' in transform input")
        if isinstance(data, str):
            return self.batch_inference([data])[0]
        if isinstance(data, list):
            return self.batch_inference(data)
        raise TypeError(f"Unsupported input type for inference pipeline: {type(data).__name__}")
        
    def add_callback(self, callback: Callable):
        """Add inference callback function."""
        self.callbacks.append(callback)
        _LOG.info(f"Inference callback added: {callback.__name__}")
    
    def remove_callback(self, callback: Callable):
        """Remove inference callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            _LOG.info(f"Inference callback removed: {callback.__name__}")
    
    def execute_callbacks(self, stage: str, **kwargs):
        """Execute all callback functions."""
        for callback in self.callbacks:
            try:
                callback(stage=stage, **kwargs)
            except Exception as e:
                _LOG.error(f"Callback {callback.__name__} failed: {e}")
    
    def preprocess_inputs(self, inputs: Union[str, List[str]]) -> List[str]:
        """
        Preprocess inputs.
        
        Args:
            inputs: Raw input text.
            
        Returns:
            List of preprocessed inputs.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Execute preprocessing callbacks
        self.execute_callbacks('preprocessing_start', inputs=inputs)
        
        # Basic preprocessing
        processed_inputs = []
        for inp in inputs:
            # Clean and normalize text
            cleaned_input = inp.strip()
            if cleaned_input:
                processed_inputs.append(cleaned_input)
        
        self.execute_callbacks('preprocessing_end', processed_inputs=processed_inputs)
        return processed_inputs
    
    def postprocess_outputs(self, outputs: List[str], 
                          original_inputs: List[str]) -> Union[str, List[str]]:
        """
        Postprocess outputs.
        
        Args:
            outputs: Raw output list.
            original_inputs: Original input list.
            
        Returns:
            Postprocessed outputs.
        """
        self.execute_callbacks('postprocessing_start', outputs=outputs)
        
        # Basic postprocessing
        processed_outputs = []
        for output in outputs:
            # Clean output text
            cleaned_output = output.strip()
            processed_outputs.append(cleaned_output)
        
        self.execute_callbacks('postprocessing_end', processed_outputs=processed_outputs)
        
        # Maintain input-output consistency
        if len(original_inputs) == 1 and len(processed_outputs) == 1:
            return processed_outputs[0]
        return processed_outputs
    
    def generate_stream(self, prompt: str, **kwargs):
        """
        Streaming generation interface.
        
        Args:
            prompt: Input prompt.
            **kwargs: Generation parameters.
            
        Yields:
            Generated text chunks.
        """
        self.execute_callbacks('stream_start', prompt=prompt)
        
        try:
            # Use core inference operator for streaming generation
            if hasattr(self.inferencer, '_stream_generate'):
                for chunk in self.inferencer._stream_generate(prompt, **kwargs):
                    self.execute_callbacks('stream_chunk', chunk=chunk)
                    yield chunk
            else:
                # Simulate streaming generation
                full_output = self.inferencer.generate(prompt, **kwargs)
                for i in range(0, len(full_output), 10):  # Return 10 characters at a time
                    chunk = full_output[i:i+10]
                    self.execute_callbacks('stream_chunk', chunk=chunk)
                    yield chunk
                    
        except Exception as e:
            self.execute_callbacks('stream_error', error=str(e))
            raise
        finally:
            self.execute_callbacks('stream_end')
    
    def batch_inference(self, prompts: List[str], 
                       batch_size: Optional[int] = None,
                       **kwargs) -> List[str]:
        """
        Batch inference.
        
        Args:
            prompts: List of input prompts.
            batch_size: Batch processing size.
            **kwargs: Inference parameters.
            
        Returns:
            List of inference results.
        """
        if batch_size is None:
            batch_size = self.infer_config.batch_size
        
        _LOG.info(f"Starting batch inference with {len(prompts)} prompts, batch_size={batch_size}")
        self.execute_callbacks('batch_start', prompts=prompts, batch_size=batch_size)
        
        results = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_index = i // batch_size + 1
            
            _LOG.info(f"Processing batch {batch_index}/{total_batches}")
            self.execute_callbacks('batch_processing', 
                                 batch_index=batch_index,
                                 batch_prompts=batch_prompts)
            
            try:
                # Execute batch inference
                batch_results = self.inferencer.batch_generate(batch_prompts, **kwargs)
                results.extend(batch_results)
                
                self.execute_callbacks('batch_completed', 
                                     batch_index=batch_index,
                                     batch_results=batch_results)
                
            except Exception as e:
                _LOG.error(f"Batch {batch_index} failed: {e}")
                # Add empty results for failed batch
                results.extend([""] * len(batch_prompts))
                self.execute_callbacks('batch_failed', 
                                     batch_index=batch_index,
                                     error=str(e))
        
        self.execute_callbacks('batch_end', results=results)
        _LOG.info("Batch inference completed")
        return results
    
    def async_inference(self, prompts: List[str], 
                       callback: Optional[Callable] = None,
                       **kwargs):
        """
        Asynchronous inference.
        
        Args:
            prompts: List of input prompts.
            callback: Completion callback function.
            **kwargs: Inference parameters.
        """
        import asyncio
        import threading
        
        def inference_worker():
            try:
                results = self.batch_inference(prompts, **kwargs)
                if callback:
                    callback(results)
            except Exception as e:
                _LOG.error(f"Async inference failed: {e}")
                if callback:
                    callback(None, error=str(e))
        
        # Execute inference in new thread
        thread = threading.Thread(target=inference_worker)
        thread.daemon = True
        thread.start()
        
        _LOG.info("Async inference started")
        return thread


@PiscesLxOperatorRegistrar()
class PromptEngineeringOperator(PiscesLxTransformOperator):
    """
    Prompt Engineering Operator.
    Implements advanced prompt optimization and template management.
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None):
        super().__init__(config)
        self.prompt_templates = {}
        self.optimization_strategies = {}

    def transform(self, data: Any) -> Any:
        if isinstance(data, dict):
            action = data.get("action") or data.get("mode") or "optimize"
            if action == "render":
                template_name = data.get("template_name")
                variables = data.get("variables", {}) or {}
                if not template_name:
                    raise ValueError("Missing 'template_name' for render action")
                return self.render_template(template_name, **variables)
            if action == "analyze":
                prompt = data.get("prompt", "")
                return self.analyze_prompt_quality(prompt)
            prompt = data.get("prompt", "")
            strategy = data.get("strategy", "length")
            return self.optimize_prompt(prompt, strategy=strategy)
        if isinstance(data, str):
            return self.optimize_prompt(data, strategy="length")
        raise TypeError(f"Unsupported input type for prompt engineering: {type(data).__name__}")
    
    def add_template(self, name: str, template: str, 
                    variables: Optional[List[str]] = None):
        """
        Add prompt template.
        
        Args:
            name: Template name.
            template: Template string.
            variables: Variable list.
        """
        self.prompt_templates[name] = {
            'template': template,
            'variables': variables or []
        }
        _LOG.info(f"Prompt template '{name}' added")
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render prompt template.
        
        Args:
            template_name: Template name.
            **kwargs: Template variables.
            
        Returns:
            Rendered prompt text.
        """
        if template_name not in self.prompt_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template_info = self.prompt_templates[template_name]
        template = template_info['template']
        
        # Validate required variables
        for var in template_info['variables']:
            if var not in kwargs:
                raise ValueError(f"Missing required variable: {var}")
        
        # Render template
        try:
            rendered_prompt = template.format(**kwargs)
            return rendered_prompt
        except KeyError as e:
            raise ValueError(f"Invalid template variable: {e}")
    
    def optimize_prompt(self, prompt: str, strategy: str = "length") -> str:
        """
        Optimize prompt text.
        
        Args:
            prompt: Original prompt.
            strategy: Optimization strategy.
            
        Returns:
            Optimized prompt.
        """
        if strategy == "length":
            # Length optimization: remove redundant content
            sentences = prompt.split('.')
            if len(sentences) > 3:
                # Keep first two and last sentence
                optimized = '.'.join(sentences[:2] + sentences[-1:])
                return optimized.strip()
        elif strategy == "clarity":
            # Clarity optimization: simplify expression
            # More complex NLP processing can be integrated here
            return prompt.replace('\n', ' ').strip()
        
        return prompt
    
    def analyze_prompt_quality(self, prompt: str) -> Dict[str, Any]:
        """
        Analyze prompt quality.
        
        Args:
            prompt: Prompt text.
            
        Returns:
            Quality analysis results.
        """
        analysis = {
            'length': len(prompt),
            'word_count': len(prompt.split()),
            'sentence_count': len([s for s in prompt.split('.') if s.strip()]),
            'complexity_score': self._calculate_complexity(prompt),
            'clarity_score': self._calculate_clarity(prompt),
            'recommendations': []
        }
        
        # Generate recommendations
        if analysis['length'] > 500:
            analysis['recommendations'].append("Prompt is quite long, consider shortening")
        if analysis['complexity_score'] > 0.8:
            analysis['recommendations'].append("High complexity detected, consider simplifying")
        if analysis['clarity_score'] < 0.5:
            analysis['recommendations'].append("Low clarity detected, consider rephrasing")
        
        return analysis
    
    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity."""
        # Simplified complexity calculation
        words = prompt.split()
        if not words:
            return 0.0
        
        # Based on vocabulary diversity and sentence structure
        unique_words = len(set(words))
        total_words = len(words)
        complexity = unique_words / total_words
        
        # Consider sentence length variation
        sentences = [s.strip() for s in prompt.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            complexity += min(avg_sentence_length / 20.0, 0.5)
        
        return min(complexity, 1.0)
    
    def _calculate_clarity(self, prompt: str) -> float:
        """Calculate prompt clarity."""
        # Simplified clarity calculation
        if not prompt.strip():
            return 0.0
        
        # Based on punctuation usage and sentence structure
        clarity = 0.5  # Base score
        
        # Check punctuation
        punctuation_count = sum(prompt.count(p) for p in '.!?;:')
        if punctuation_count > 0:
            clarity += 0.3
        
        # Check question words
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        question_count = sum(1 for word in question_words if word in prompt.lower())
        clarity += min(question_count * 0.1, 0.2)
        
        return min(clarity, 1.0)


@PiscesLxOperatorRegistrar()
class ResultProcessorOperator(PiscesLxTransformOperator):
    """
    Result Processor Operator.
    Implements post-processing and formatting of inference results.
    """
    
    def __init__(self, config: Optional[PiscesLxOperatorConfig] = None):
        super().__init__(config)
        self.processors = []

    def transform(self, data: Any) -> Any:
        if isinstance(data, dict):
            results = data.get("results")
            if results is None:
                results = data.get("data")
            if results is None:
                raise ValueError("Missing 'results' in transform input")
            return self.process_results(results)
        return self.process_results(data)
    
    def add_processor(self, processor: Callable[[str], str]):
        """Add result processor."""
        self.processors.append(processor)
        _LOG.info(f"Result processor added: {processor.__name__}")
    
    def process_results(self, results: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Process inference results.
        
        Args:
            results: Raw results.
            
        Returns:
            Processed results.
        """
        is_single = isinstance(results, str)
        if is_single:
            results = [results]
        
        processed_results = []
        for result in results:
            processed_result = result
            # Apply all processors sequentially
            for processor in self.processors:
                try:
                    processed_result = processor(processed_result)
                except Exception as e:
                    _LOG.warning(f"Processor {processor.__name__} failed: {e}")
            processed_results.append(processed_result)
        
        return processed_results[0] if is_single else processed_results
    
    def filter_results(self, results: List[str], 
                      filter_func: Callable[[str], bool]) -> List[str]:
        """
        Filter results.
        
        Args:
            results: Results list.
            filter_func: Filter function.
            
        Returns:
            Filtered results.
        """
        return [result for result in results if filter_func(result)]
    
    def format_results(self, results: List[str], 
                      format_type: str = "json") -> str:
        """
        Format results.
        
        Args:
            results: Results list.
            format_type: Format type.
            
        Returns:
            Formatted result string.
        """
        if format_type.lower() == "json":
            return json.dumps(results, indent=2, ensure_ascii=False)
        elif format_type.lower() == "csv":
            import csv
            from io import StringIO
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(["result"])
            for result in results:
                writer.writerow([result])
            return output.getvalue()
        elif format_type.lower() == "markdown":
            md_content = "# Inference Results\n\n"
            for i, result in enumerate(results, 1):
                md_content += f"## Result {i}\n\n{result}\n\n"
            return md_content
        else:
            return "\n---\n".join(results)


