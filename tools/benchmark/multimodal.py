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
Multimodal benchmark evaluation suite for PiscesLx.

This module provides comprehensive multimodal benchmark evaluation including:
- Vision: VQAv2, GQA, TextVQA, ChartQA, MMMU, RefCOCO
- Audio: LibriSpeech, Common Voice, FLEURS
- Video: MVBench, ActivityNet QA, MSRVTT QA
"""

import os
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict
import threading

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file, get_work_dir

_LOG = PiscesLxLogger("PiscesLx.Tools.Benchmark.Multimodal", file_path=get_log_file("PiscesLx.Tools.Benchmark"), enable_file=True)


@dataclass
class PiscesLxToolsMultimodalConfig:
    """Configuration for multimodal benchmark evaluation."""
    
    model_path: str = ".pisceslx/ckpt"
    output_dir: str = ".pisceslx/benchmark/multimodal"
    
    batch_size: int = 1
    max_seq_length: int = 4096
    max_generation_length: int = 512
    
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    
    device: str = "cuda"
    use_bf16: bool = True
    
    image_size: int = 336
    audio_sample_rate: int = 16000
    video_num_frames: int = 8
    
    vision_benchmarks: List[str] = field(default_factory=lambda: [
        "vqav2", "gqa", "textvqa", "chartqa", "mmmu", "refcoco"
    ])
    
    audio_benchmarks: List[str] = field(default_factory=lambda: [
        "librispeech", "common_voice", "fleurs"
    ])
    
    video_benchmarks: List[str] = field(default_factory=lambda: [
        "mvbench", "activitynet_qa", "msrvtt_qa"
    ])
    
    save_results: bool = True
    verbose: bool = True


class PiscesLxToolsVisionEvaluator:
    """Vision benchmark evaluator for PiscesLx multimodal model."""
    
    def __init__(
        self,
        config: PiscesLxToolsMultimodalConfig,
        model: nn.Module,
        tokenizer: Any,
        image_processor: Any = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.results = {}
        self.lock = threading.Lock()
        
        _LOG.info("PiscesLxToolsVisionEvaluator initialized")
    
    def evaluate_vqav2(self) -> Dict[str, float]:
        """Evaluate on VQAv2 - Visual Question Answering v2."""
        _LOG.info("Evaluating VQAv2...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("Multimodal-Fatima/VQAv2", split="validation")
        except Exception as e:
            _LOG.warning(f"Failed to load VQAv2: {e}, trying alternative...")
            try:
                dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation")
            except Exception as e2:
                _LOG.error(f"Failed to load VQAv2: {e2}")
                return {"accuracy": 0.0, "total": 0}
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="VQAv2"):
            try:
                image = item.get("image")
                question = item.get("question", "")
                answers = item.get("answers", [])
                
                if image is None or not answers:
                    continue
                
                if self.image_processor:
                    image_tensor = self.image_processor(image, return_tensors="pt")
                    pixel_values = image_tensor.get("pixel_values").to(self.config.device)
                else:
                    pixel_values = None
                
                prompt = f"Question: {question}\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    if pixel_values is not None:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            pixel_values=pixel_values,
                            max_new_tokens=32,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=32,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()
                
                total += 1
                
                for ans in answers:
                    if isinstance(ans, dict):
                        ans_text = ans.get("answer", "").lower()
                    else:
                        ans_text = str(ans).lower()
                    if ans_text in generated or generated in ans_text:
                        correct += 1
                        break
                        
            except Exception as e:
                _LOG.debug(f"VQAv2 sample error: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["vqav2"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"VQAv2 Accuracy: {accuracy:.4f}")
        return self.results["vqav2"]
    
    def evaluate_gqa(self) -> Dict[str, float]:
        """Evaluate on GQA - GQA dataset for visual reasoning."""
        _LOG.info("Evaluating GQA...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("lmms-lab/GQA", split="testdev_balanced")
        except Exception as e:
            _LOG.warning(f"Failed to load GQA: {e}, trying alternative...")
            try:
                dataset = load_dataset("gqa", split="testdev_balanced")
            except Exception as e2:
                _LOG.error(f"Failed to load GQA: {e2}")
                return {"accuracy": 0.0, "total": 0}
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="GQA"):
            try:
                image = item.get("image")
                question = item.get("question", "")
                answer = item.get("answer", "")
                
                if image is None or not answer:
                    continue
                
                if self.image_processor:
                    image_tensor = self.image_processor(image, return_tensors="pt")
                    pixel_values = image_tensor.get("pixel_values").to(self.config.device)
                else:
                    pixel_values = None
                
                prompt = f"Question: {question}\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    if pixel_values is not None:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            pixel_values=pixel_values,
                            max_new_tokens=32,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=32,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()
                
                total += 1
                
                if answer.lower() in generated or generated in answer.lower():
                    correct += 1
                    
            except Exception as e:
                _LOG.debug(f"GQA sample error: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["gqa"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"GQA Accuracy: {accuracy:.4f}")
        return self.results["gqa"]
    
    def evaluate_textvqa(self) -> Dict[str, float]:
        """Evaluate on TextVQA - Text-based Visual Question Answering."""
        _LOG.info("Evaluating TextVQA...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("lmms-lab/textvqa", split="validation")
        except Exception as e:
            _LOG.warning(f"Failed to load TextVQA: {e}")
            return {"accuracy": 0.0, "total": 0}
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="TextVQA"):
            try:
                image = item.get("image")
                question = item.get("question", "")
                answers = item.get("answers", [])
                
                if image is None or not answers:
                    continue
                
                if self.image_processor:
                    image_tensor = self.image_processor(image, return_tensors="pt")
                    pixel_values = image_tensor.get("pixel_values").to(self.config.device)
                else:
                    pixel_values = None
                
                prompt = f"Question: {question}\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    if pixel_values is not None:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            pixel_values=pixel_values,
                            max_new_tokens=32,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=32,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()
                
                total += 1
                
                for ans in answers:
                    ans_text = str(ans).lower()
                    if ans_text in generated or generated in ans_text:
                        correct += 1
                        break
                        
            except Exception as e:
                _LOG.debug(f"TextVQA sample error: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["textvqa"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"TextVQA Accuracy: {accuracy:.4f}")
        return self.results["textvqa"]
    
    def evaluate_chartqa(self) -> Dict[str, float]:
        """Evaluate on ChartQA - Chart Question Answering."""
        _LOG.info("Evaluating ChartQA...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("lmms-lab/ChartQA", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load ChartQA: {e}")
            return {"accuracy": 0.0, "total": 0}
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="ChartQA"):
            try:
                image = item.get("image")
                question = item.get("question", "")
                answer = item.get("answer", "")
                
                if image is None or not answer:
                    continue
                
                if self.image_processor:
                    image_tensor = self.image_processor(image, return_tensors="pt")
                    pixel_values = image_tensor.get("pixel_values").to(self.config.device)
                else:
                    pixel_values = None
                
                prompt = f"Question: {question}\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    if pixel_values is not None:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            pixel_values=pixel_values,
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()
                
                total += 1
                
                if answer.lower() in generated or generated in answer.lower():
                    correct += 1
                    
            except Exception as e:
                _LOG.debug(f"ChartQA sample error: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["chartqa"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"ChartQA Accuracy: {accuracy:.4f}")
        return self.results["chartqa"]
    
    def evaluate_mmmu(self) -> Dict[str, float]:
        """Evaluate on MMMU - Massive Multi-discipline Multimodal Understanding."""
        _LOG.info("Evaluating MMMU...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("MMMU/MMMU", split="validation")
        except Exception as e:
            _LOG.warning(f"Failed to load MMMU: {e}")
            return {"accuracy": 0.0, "total": 0}
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="MMMU"):
            try:
                images = item.get("images", [])
                question = item.get("question", "")
                choices = item.get("choices", [])
                answer = item.get("answer", "")
                
                if not question or not choices:
                    continue
                
                prompt = f"Question: {question}\n\nOptions:\n"
                for i, choice in enumerate(choices):
                    prompt += f"{chr(65 + i)}. {choice}\n"
                prompt += "\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                pixel_values = None
                if images and self.image_processor:
                    first_image = images[0] if isinstance(images, list) else images
                    if first_image:
                        image_tensor = self.image_processor(first_image, return_tensors="pt")
                        pixel_values = image_tensor.get("pixel_values").to(self.config.device)
                
                with torch.no_grad():
                    if pixel_values is not None:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            pixel_values=pixel_values,
                            max_new_tokens=10,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=10,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().upper()
                
                total += 1
                
                if generated and generated[0] == answer.upper():
                    correct += 1
                    
            except Exception as e:
                _LOG.debug(f"MMMU sample error: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["mmmu"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"MMMU Accuracy: {accuracy:.4f}")
        return self.results["mmmu"]
    
    def evaluate_refcoco(self) -> Dict[str, float]:
        """Evaluate on RefCOCO - Referring Expression Comprehension."""
        _LOG.info("Evaluating RefCOCO...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("lmms-lab/RefCOCO", split="validation")
        except Exception as e:
            _LOG.warning(f"Failed to load RefCOCO: {e}")
            return {"accuracy": 0.0, "total": 0}
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="RefCOCO"):
            try:
                image = item.get("image")
                caption = item.get("caption", "")
                bbox = item.get("bbox", [])
                
                if image is None or not bbox:
                    continue
                
                if self.image_processor:
                    image_tensor = self.image_processor(image, return_tensors="pt")
                    pixel_values = image_tensor.get("pixel_values").to(self.config.device)
                else:
                    pixel_values = None
                
                prompt = f"Locate the object in the image: {caption}\nBounding box (x, y, width, height):"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    if pixel_values is not None:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            pixel_values=pixel_values,
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                total += 1
                
                pred_bbox = self._parse_bbox(generated)
                if pred_bbox and len(bbox) == 4:
                    iou = self._compute_iou(pred_bbox, bbox)
                    if iou >= 0.5:
                        correct += 1
                    
            except Exception as e:
                _LOG.debug(f"RefCOCO sample error: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["refcoco"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"RefCOCO Accuracy: {accuracy:.4f}")
        return self.results["refcoco"]
    
    def _parse_bbox(self, text: str) -> Optional[List[float]]:
        """Parse bounding box from text."""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if len(numbers) >= 4:
            return [float(n) for n in numbers[:4]]
        return None
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute Intersection over Union for bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1e-6)
    
    def run_all_vision_benchmarks(self) -> Dict[str, Any]:
        """Run all vision benchmarks."""
        _LOG.info("Running all vision benchmarks...")
        
        benchmarks_map = {
            "vqav2": self.evaluate_vqav2,
            "gqa": self.evaluate_gqa,
            "textvqa": self.evaluate_textvqa,
            "chartqa": self.evaluate_chartqa,
            "mmmu": self.evaluate_mmmu,
            "refcoco": self.evaluate_refcoco,
        }
        
        for benchmark in self.config.vision_benchmarks:
            if benchmark in benchmarks_map:
                try:
                    benchmarks_map[benchmark]()
                except Exception as e:
                    _LOG.error(f"Vision benchmark {benchmark} failed: {e}")
        
        self.results["vision_summary"] = self._generate_vision_summary()
        
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _generate_vision_summary(self) -> Dict[str, float]:
        """Generate vision benchmark summary."""
        summary = {}
        for name, result in self.results.items():
            if name == "vision_summary":
                continue
            if isinstance(result, dict) and "accuracy" in result:
                summary[name] = result["accuracy"]
        
        if summary:
            summary["average"] = sum(summary.values()) / len(summary)
        
        return summary
    
    def _save_results(self) -> None:
        """Save results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"vision_benchmark_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Vision benchmark results saved to {output_path}")


class PiscesLxToolsAudioEvaluator:
    """Audio benchmark evaluator for PiscesLx multimodal model."""
    
    def __init__(
        self,
        config: PiscesLxToolsMultimodalConfig,
        model: nn.Module,
        tokenizer: Any,
        audio_processor: Any = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.results = {}
        
        _LOG.info("PiscesLxToolsAudioEvaluator initialized")
    
    def evaluate_librispeech(self) -> Dict[str, float]:
        """Evaluate on LibriSpeech - Automatic Speech Recognition."""
        _LOG.info("Evaluating LibriSpeech...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("librispeech_asr", "clean", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load LibriSpeech: {e}")
            return {"wer": 1.0, "total": 0}
        
        total_wer = 0.0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="LibriSpeech"):
            try:
                audio = item.get("audio")
                text = item.get("text", "")
                
                if audio is None or not text:
                    continue
                
                if self.audio_processor:
                    audio_input = self.audio_processor(
                        audio["array"],
                        sampling_rate=audio["sampling_rate"],
                        return_tensors="pt"
                    ).to(self.config.device)
                else:
                    audio_input = None
                
                with torch.no_grad():
                    if audio_input is not None:
                        outputs = self.model.generate(
                            input_features=audio_input.get("input_features"),
                            max_new_tokens=256,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        continue
                
                generated = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip().lower()
                
                wer = self._compute_wer(text.lower(), generated)
                total_wer += wer
                total += 1
                
            except Exception as e:
                _LOG.debug(f"LibriSpeech sample error: {e}")
                continue
        
        avg_wer = total_wer / max(1, total)
        
        self.results["librispeech"] = {
            "wer": avg_wer,
            "total": total,
        }
        
        _LOG.info(f"LibriSpeech WER: {avg_wer:.4f}")
        return self.results["librispeech"]
    
    def evaluate_common_voice(self) -> Dict[str, float]:
        """Evaluate on Common Voice - Multilingual ASR."""
        _LOG.info("Evaluating Common Voice...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="test", trust_remote_code=True)
        except Exception as e:
            _LOG.warning(f"Failed to load Common Voice: {e}")
            return {"wer": 1.0, "total": 0}
        
        total_wer = 0.0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="Common Voice"):
            try:
                audio = item.get("audio")
                text = item.get("sentence", "")
                
                if audio is None or not text:
                    continue
                
                if self.audio_processor:
                    audio_input = self.audio_processor(
                        audio["array"],
                        sampling_rate=audio["sampling_rate"],
                        return_tensors="pt"
                    ).to(self.config.device)
                else:
                    audio_input = None
                
                with torch.no_grad():
                    if audio_input is not None:
                        outputs = self.model.generate(
                            input_features=audio_input.get("input_features"),
                            max_new_tokens=256,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        continue
                
                generated = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip().lower()
                
                wer = self._compute_wer(text.lower(), generated)
                total_wer += wer
                total += 1
                
            except Exception as e:
                _LOG.debug(f"Common Voice sample error: {e}")
                continue
        
        avg_wer = total_wer / max(1, total)
        
        self.results["common_voice"] = {
            "wer": avg_wer,
            "total": total,
        }
        
        _LOG.info(f"Common Voice WER: {avg_wer:.4f}")
        return self.results["common_voice"]
    
    def evaluate_fleurs(self) -> Dict[str, float]:
        """Evaluate on FLEURS - Multilingual speech recognition."""
        _LOG.info("Evaluating FLEURS...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("google/fleurs", "en_us", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load FLEURS: {e}")
            return {"wer": 1.0, "total": 0}
        
        total_wer = 0.0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="FLEURS"):
            try:
                audio = item.get("audio")
                text = item.get("transcription", "")
                
                if audio is None or not text:
                    continue
                
                if self.audio_processor:
                    audio_input = self.audio_processor(
                        audio["array"],
                        sampling_rate=audio["sampling_rate"],
                        return_tensors="pt"
                    ).to(self.config.device)
                else:
                    audio_input = None
                
                with torch.no_grad():
                    if audio_input is not None:
                        outputs = self.model.generate(
                            input_features=audio_input.get("input_features"),
                            max_new_tokens=256,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        continue
                
                generated = self.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                ).strip().lower()
                
                wer = self._compute_wer(text.lower(), generated)
                total_wer += wer
                total += 1
                
            except Exception as e:
                _LOG.debug(f"FLEURS sample error: {e}")
                continue
        
        avg_wer = total_wer / max(1, total)
        
        self.results["fleurs"] = {
            "wer": avg_wer,
            "total": total,
        }
        
        _LOG.info(f"FLEURS WER: {avg_wer:.4f}")
        return self.results["fleurs"]
    
    def _compute_wer(self, reference: str, hypothesis: str) -> float:
        """Compute Word Error Rate."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,
                        d[i][j - 1] + 1,
                        d[i - 1][j - 1] + 1
                    )
        
        return d[len(ref_words)][len(hyp_words)] / len(ref_words)
    
    def run_all_audio_benchmarks(self) -> Dict[str, Any]:
        """Run all audio benchmarks."""
        _LOG.info("Running all audio benchmarks...")
        
        benchmarks_map = {
            "librispeech": self.evaluate_librispeech,
            "common_voice": self.evaluate_common_voice,
            "fleurs": self.evaluate_fleurs,
        }
        
        for benchmark in self.config.audio_benchmarks:
            if benchmark in benchmarks_map:
                try:
                    benchmarks_map[benchmark]()
                except Exception as e:
                    _LOG.error(f"Audio benchmark {benchmark} failed: {e}")
        
        self.results["audio_summary"] = self._generate_audio_summary()
        
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _generate_audio_summary(self) -> Dict[str, float]:
        """Generate audio benchmark summary."""
        summary = {}
        for name, result in self.results.items():
            if name == "audio_summary":
                continue
            if isinstance(result, dict) and "wer" in result:
                summary[name] = result["wer"]
        
        if summary:
            summary["average_wer"] = sum(summary.values()) / len(summary)
        
        return summary
    
    def _save_results(self) -> None:
        """Save results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"audio_benchmark_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Audio benchmark results saved to {output_path}")


class PiscesLxToolsVideoEvaluator:
    """Video benchmark evaluator for PiscesLx multimodal model."""
    
    def __init__(
        self,
        config: PiscesLxToolsMultimodalConfig,
        model: nn.Module,
        tokenizer: Any,
        video_processor: Any = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.results = {}
        
        _LOG.info("PiscesLxToolsVideoEvaluator initialized")
    
    def evaluate_mvbench(self) -> Dict[str, float]:
        """Evaluate on MVBench - Multi-task Video Understanding."""
        _LOG.info("Evaluating MVBench...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("OpenGVLab/MVBench", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load MVBench: {e}")
            return {"accuracy": 0.0, "total": 0}
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="MVBench"):
            try:
                video = item.get("video")
                question = item.get("question", "")
                answer = item.get("answer", "")
                
                if video is None or not question:
                    continue
                
                if self.video_processor:
                    video_input = self.video_processor(video, return_tensors="pt")
                    pixel_values = video_input.get("pixel_values").to(self.config.device)
                else:
                    pixel_values = None
                
                prompt = f"Question: {question}\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    if pixel_values is not None:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            pixel_values=pixel_values,
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()
                
                total += 1
                
                if answer.lower() in generated or generated in answer.lower():
                    correct += 1
                    
            except Exception as e:
                _LOG.debug(f"MVBench sample error: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["mvbench"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"MVBench Accuracy: {accuracy:.4f}")
        return self.results["mvbench"]
    
    def evaluate_activitynet_qa(self) -> Dict[str, float]:
        """Evaluate on ActivityNet QA - Video Question Answering."""
        _LOG.info("Evaluating ActivityNet QA...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("lmms-lab/ActivityNetQA", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load ActivityNet QA: {e}")
            return {"accuracy": 0.0, "total": 0}
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="ActivityNet QA"):
            try:
                video = item.get("video")
                question = item.get("question", "")
                answer = item.get("answer", "")
                
                if video is None or not question:
                    continue
                
                if self.video_processor:
                    video_input = self.video_processor(video, return_tensors="pt")
                    pixel_values = video_input.get("pixel_values").to(self.config.device)
                else:
                    pixel_values = None
                
                prompt = f"Question: {question}\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    if pixel_values is not None:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            pixel_values=pixel_values,
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()
                
                total += 1
                
                if answer.lower() in generated or generated in answer.lower():
                    correct += 1
                    
            except Exception as e:
                _LOG.debug(f"ActivityNet QA sample error: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["activitynet_qa"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"ActivityNet QA Accuracy: {accuracy:.4f}")
        return self.results["activitynet_qa"]
    
    def evaluate_msrvtt_qa(self) -> Dict[str, float]:
        """Evaluate on MSRVTT QA - Video Question Answering."""
        _LOG.info("Evaluating MSRVTT QA...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("lmms-lab/MSRVTT-QA", split="test")
        except Exception as e:
            _LOG.warning(f"Failed to load MSRVTT QA: {e}")
            return {"accuracy": 0.0, "total": 0}
        
        correct = 0
        total = 0
        
        self.model.eval()
        
        for item in tqdm(dataset, desc="MSRVTT QA"):
            try:
                video = item.get("video")
                question = item.get("question", "")
                answer = item.get("answer", "")
                
                if video is None or not question:
                    continue
                
                if self.video_processor:
                    video_input = self.video_processor(video, return_tensors="pt")
                    pixel_values = video_input.get("pixel_values").to(self.config.device)
                else:
                    pixel_values = None
                
                prompt = f"Question: {question}\nAnswer:"
                
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)
                
                with torch.no_grad():
                    if pixel_values is not None:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            pixel_values=pixel_values,
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                    else:
                        outputs = self.model.generate(
                            input_ids=encoding["input_ids"],
                            attention_mask=encoding["attention_mask"],
                            max_new_tokens=64,
                            temperature=self.config.temperature,
                            do_sample=self.config.do_sample,
                        )
                
                generated = self.tokenizer.decode(
                    outputs[0][encoding["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()
                
                total += 1
                
                if answer.lower() in generated or generated in answer.lower():
                    correct += 1
                    
            except Exception as e:
                _LOG.debug(f"MSRVTT QA sample error: {e}")
                continue
        
        accuracy = correct / max(1, total)
        
        self.results["msrvtt_qa"] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        
        _LOG.info(f"MSRVTT QA Accuracy: {accuracy:.4f}")
        return self.results["msrvtt_qa"]
    
    def run_all_video_benchmarks(self) -> Dict[str, Any]:
        """Run all video benchmarks."""
        _LOG.info("Running all video benchmarks...")
        
        benchmarks_map = {
            "mvbench": self.evaluate_mvbench,
            "activitynet_qa": self.evaluate_activitynet_qa,
            "msrvtt_qa": self.evaluate_msrvtt_qa,
        }
        
        for benchmark in self.config.video_benchmarks:
            if benchmark in benchmarks_map:
                try:
                    benchmarks_map[benchmark]()
                except Exception as e:
                    _LOG.error(f"Video benchmark {benchmark} failed: {e}")
        
        self.results["video_summary"] = self._generate_video_summary()
        
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _generate_video_summary(self) -> Dict[str, float]:
        """Generate video benchmark summary."""
        summary = {}
        for name, result in self.results.items():
            if name == "video_summary":
                continue
            if isinstance(result, dict) and "accuracy" in result:
                summary[name] = result["accuracy"]
        
        if summary:
            summary["average"] = sum(summary.values()) / len(summary)
        
        return summary
    
    def _save_results(self) -> None:
        """Save results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"video_benchmark_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Video benchmark results saved to {output_path}")


class PiscesLxToolsMultimodalBenchmarkRunner:
    """Runner for all multimodal benchmarks."""
    
    def __init__(
        self,
        config: PiscesLxToolsMultimodalConfig,
        model: nn.Module,
        tokenizer: Any,
        image_processor: Any = None,
        audio_processor: Any = None,
        video_processor: Any = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.video_processor = video_processor
        self.results = {}
        
        _LOG.info("PiscesLxToolsMultimodalBenchmarkRunner initialized")
    
    def run_all(self) -> Dict[str, Any]:
        """Run all multimodal benchmarks."""
        _LOG.info("Running all multimodal benchmarks...")
        
        if self.config.vision_benchmarks:
            vision_evaluator = PiscesLxToolsVisionEvaluator(
                self.config, self.model, self.tokenizer, self.image_processor
            )
            self.results["vision"] = vision_evaluator.run_all_vision_benchmarks()
        
        if self.config.audio_benchmarks:
            audio_evaluator = PiscesLxToolsAudioEvaluator(
                self.config, self.model, self.tokenizer, self.audio_processor
            )
            self.results["audio"] = audio_evaluator.run_all_audio_benchmarks()
        
        if self.config.video_benchmarks:
            video_evaluator = PiscesLxToolsVideoEvaluator(
                self.config, self.model, self.tokenizer, self.video_processor
            )
            self.results["video"] = video_evaluator.run_all_video_benchmarks()
        
        self.results["multimodal_summary"] = self._generate_multimodal_summary()
        
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _generate_multimodal_summary(self) -> Dict[str, Any]:
        """Generate overall multimodal summary."""
        summary = {
            "vision": {},
            "audio": {},
            "video": {},
        }
        
        if "vision" in self.results:
            vision_summary = self.results["vision"].get("vision_summary", {})
            summary["vision"] = vision_summary
        
        if "audio" in self.results:
            audio_summary = self.results["audio"].get("audio_summary", {})
            summary["audio"] = audio_summary
        
        if "video" in self.results:
            video_summary = self.results["video"].get("video_summary", {})
            summary["video"] = video_summary
        
        return summary
    
    def _save_results(self) -> None:
        """Save all results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"multimodal_benchmark_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        _LOG.info(f"Multimodal benchmark results saved to {output_path}")
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("PiscesLx Multimodal Benchmark Results")
        print("=" * 60)
        
        if "vision" in self.results:
            print("\n[Vision Benchmarks]")
            vision_summary = self.results["vision"].get("vision_summary", {})
            for benchmark, score in vision_summary.items():
                if benchmark != "average":
                    print(f"  {benchmark:20s}: {score:.4f}")
            if "average" in vision_summary:
                print(f"  {'Average':20s}: {vision_summary['average']:.4f}")
        
        if "audio" in self.results:
            print("\n[Audio Benchmarks]")
            audio_summary = self.results["audio"].get("audio_summary", {})
            for benchmark, score in audio_summary.items():
                if benchmark != "average_wer":
                    print(f"  {benchmark:20s}: WER {score:.4f}")
            if "average_wer" in audio_summary:
                print(f"  {'Average WER':20s}: {audio_summary['average_wer']:.4f}")
        
        if "video" in self.results:
            print("\n[Video Benchmarks]")
            video_summary = self.results["video"].get("video_summary", {})
            for benchmark, score in video_summary.items():
                if benchmark != "average":
                    print(f"  {benchmark:20s}: {score:.4f}")
            if "average" in video_summary:
                print(f"  {'Average':20s}: {video_summary['average']:.4f}")
        
        print("=" * 60 + "\n")


def create_multimodal_evaluator(
    config: PiscesLxToolsMultimodalConfig,
    model: nn.Module,
    tokenizer: Any,
    image_processor: Any = None,
    audio_processor: Any = None,
    video_processor: Any = None,
) -> PiscesLxToolsMultimodalBenchmarkRunner:
    """Factory function to create multimodal benchmark runner."""
    return PiscesLxToolsMultimodalBenchmarkRunner(
        config=config,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )
