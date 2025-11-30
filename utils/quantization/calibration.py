#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import json
from datasets import load_dataset
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None
# Use dms_core logging exclusively
import dms_core
PiscesLxCoreLog = dms_core.log.get_logger
from utils.error import PiscesLxCoreValidationError, PiscesLxCoreIOError

logger = PiscesLxCoreLog("PiscesLx.Core.Quantization.Calibration")

class CalibrationDataLoader:
    """Enhanced calibration data loader supporting multiple dataset formats."""
    
    SUPPORTED_DATASETS = {
        "wikitext": ["wikitext-2-v1", "wikitext-103-v1"],
        "c4": ["en", "realnewslike", "webtextlike"],
        "openwebtext": ["plain_text"],
        "pile": ["plain_text"],
        "custom": []
    }
    
    def __init__(self, dataset_name: str = "wikitext", num_samples: int = 128,
                 dataset_config: Optional[str] = None, 
                 custom_data_path: Optional[Union[str, Path]] = None,
                 text_column: str = "text"):
        self.dataset_name = dataset_name.lower()
        self.num_samples = num_samples
        self.dataset_config = dataset_config
        self.custom_data_path = Path(custom_data_path) if custom_data_path else None
        self.text_column = text_column
        self.data = None
        
        # Validate dataset name
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            logger.warning(f"Unknown dataset {dataset_name}, using wikitext as fallback")
            self.dataset_name = "wikitext"
    
    def load_data(self) -> List[Dict[str, Any]]:
        """Load calibration data from various dataset sources."""
        try:
            logger.info(f"Loading calibration data from {self.dataset_name}")
            
            if self.dataset_name == "custom" and self.custom_data_path:
                self.data = self._load_custom_data()
            elif self.dataset_name in ["wikitext", "c4", "openwebtext", "pile"]:
                self.data = self._load_huggingface_dataset()
            else:
                self.data = self._generate_sample_data()
            
            logger.info(f"Loaded {len(self.data)} calibration samples")
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")
            # Fallback to sample data
            logger.info("Falling back to sample data")
            self.data = self._generate_sample_data()
            return self.data
    
    def _load_huggingface_dataset(self) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace datasets."""
        try:
            # Determine dataset configuration
            if self.dataset_config:
                dataset_config = self.dataset_config
            else:
                dataset_config = self.SUPPORTED_DATASETS[self.dataset_name][0]
            
            # Load dataset
            if self.dataset_name == "wikitext":
                dataset = load_dataset("wikitext", dataset_config, split="train")
            elif self.dataset_name == "c4":
                dataset = load_dataset("c4", dataset_config, split="train", streaming=True)
                # For streaming datasets, take first N samples
                dataset = dataset.take(self.num_samples * 2)  # Take extra for filtering
            elif self.dataset_name == "openwebtext":
                dataset = load_dataset("openwebtext", split="train", streaming=True)
                dataset = dataset.take(self.num_samples * 2)
            elif self.dataset_name == "pile":
                dataset = load_dataset("pile", dataset_config, split="train", streaming=True)
                dataset = dataset.take(self.num_samples * 2)
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
            # Convert to list and filter valid samples
            data_list = []
            for item in dataset:
                text = item.get(self.text_column, "")
                if self._is_valid_text(text):
                    data_list.append({
                        "text": text,
                        "input_ids": None,  # Will be tokenized later
                        "metadata": {
                            "source": self.dataset_name,
                            "length": len(text.split()),
                            "char_length": len(text)
                        }
                    })
                    
                    if len(data_list) >= self.num_samples:
                        break
            
            return data_list
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            return self._generate_sample_data()
    
    def _load_custom_data(self) -> List[Dict[str, Any]]:
        """Load custom data from local files."""
        try:
            if not self.custom_data_path.exists():
                raise FileNotFoundError(f"Custom data file not found: {self.custom_data_path}")
            
            data_list = []
            
            if self.custom_data_path.suffix == '.json':
                with open(self.custom_data_path, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
                    
                if isinstance(custom_data, list):
                    for item in custom_data[:self.num_samples]:
                        text = item.get(self.text_column, str(item))
                        if self._is_valid_text(text):
                            data_list.append({
                                "text": text,
                                "input_ids": None,
                                "metadata": {
                                    "source": "custom_json",
                                    "length": len(text.split()),
                                    "char_length": len(text)
                                }
                            })
                            
            elif self.custom_data_path.suffix == '.txt':
                with open(self.custom_data_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines[:self.num_samples]:
                    text = line.strip()
                    if self._is_valid_text(text):
                        data_list.append({
                            "text": text,
                            "input_ids": None,
                            "metadata": {
                                "source": "custom_txt",
                                "length": len(text.split()),
                                "char_length": len(text)
                            }
                        })
            
            return data_list
            
        except Exception as e:
            logger.error(f"Failed to load custom data: {e}")
            return self._generate_sample_data()
    
    def _is_valid_text(self, text: str) -> bool:
        """Check if text is valid for calibration."""
        if not text or not isinstance(text, str):
            return False
        
        # Remove very short or very long texts
        word_count = len(text.split())
        if word_count < 5 or word_count > 1000:
            return False
        
        # Check for excessive special characters or numbers
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:  # More than 30% special characters
            return False
        
        return True
    
    def _generate_sample_data(self) -> List[Dict[str, Any]]:
        """Generate enhanced sample calibration data."""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog in the peaceful garden.",
            "Machine learning algorithms are transforming how we process and understand data.",
            "Quantization techniques reduce model size while maintaining reasonable accuracy.",
            "Large language models require substantial computational resources for training.",
            "Efficient inference is crucial for deploying artificial intelligence in production.",
            "Neural networks consist of interconnected layers that process information.",
            "Deep learning has revolutionized computer vision and natural language processing.",
            "Optimization algorithms help models converge to better solutions during training.",
            "Transformer architectures have become the standard for sequence modeling tasks.",
            "Parallel processing enables faster training of complex machine learning models."
        ]
        
        # Repeat and shuffle samples to reach desired number
        extended_texts = []
        while len(extended_texts) < self.num_samples:
            extended_texts.extend(sample_texts)
        
        extended_texts = extended_texts[:self.num_samples]
        
        return [{"text": text, "input_ids": None, "metadata": {
            "source": "generated",
            "length": len(text.split()),
            "char_length": len(text)
        }} for text in extended_texts]
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        if self.data is None:
            return {"status": "not_loaded"}
        
        total_samples = len(self.data)
        if total_samples == 0:
            return {"status": "empty"}
        
        # Calculate statistics
        text_lengths = [item["metadata"]["length"] for item in self.data if item.get("metadata")]
        char_lengths = [item["metadata"]["char_length"] for item in self.data if item.get("metadata")]
        
        sources = list(set(item["metadata"]["source"] for item in self.data if item.get("metadata")))
        
        return {
            "status": "loaded",
            "total_samples": total_samples,
            "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "avg_char_length": sum(char_lengths) / len(char_lengths) if char_lengths else 0,
            "sources": sources,
            "dataset_name": self.dataset_name
        }

class CalibrationProcessor:
    """Process calibration data for different quantization methods."""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.processed_data = None
    
    def process_for_static_quantization(self, raw_data: List[Dict[str, Any]], 
                                      max_length: int = 512) -> torch.Tensor:
        """Process data for static quantization calibration."""
        try:
            logger.info("Processing data for static quantization")
            
            if self.tokenizer is None:
                logger.warning("No tokenizer provided, using dummy tokenization")
                return self._dummy_tokenize(raw_data, max_length)
            
            # Tokenize the data
            texts = [item["text"] for item in raw_data]
            
            # Use the tokenizer to process the texts
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            return encoded["input_ids"]
            
        except Exception as e:
            logger.error(f"Failed to process data for static quantization: {e}")
            raise PiscesLxCoreValidationError(f"Failed to process data: {e}")
    
    def process_for_gptq(self, raw_data: List[Dict[str, Any]], 
                        max_length: int = 512) -> List[torch.Tensor]:
        """Process data for GPTQ quantization calibration."""
        try:
            logger.info("Processing data for GPTQ quantization")
            
            if self.tokenizer is None:
                logger.warning("No tokenizer provided, using dummy tokenization")
                return self._dummy_tokenize_for_gptq(raw_data, max_length)
            
            # Tokenize the data for GPTQ
            texts = [item["text"] for item in raw_data]
            
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # GPTQ expects a list of input tensors
            input_ids = encoded["input_ids"]
            return [input_ids[i:i+1] for i in range(input_ids.size(0))]
            
        except Exception as e:
            logger.error(f"Failed to process data for GPTQ: {e}")
            raise PiscesLxCoreValidationError(f"Failed to process data for GPTQ: {e}")
    
    def process_for_awq(self, raw_data: List[Dict[str, Any]], 
                       max_length: int = 512) -> List[torch.Tensor]:
        """Process data for AWQ quantization calibration."""
        try:
            logger.info("Processing data for AWQ quantization")
            
            # AWQ uses similar processing to GPTQ
            return self.process_for_gptq(raw_data, max_length)
            
        except Exception as e:
            logger.error(f"Failed to process data for AWQ: {e}")
            raise PiscesLxCoreValidationError(f"Failed to process data for AWQ: {e}")
    
    def _dummy_tokenize(self, raw_data: List[Dict[str, Any]], max_length: int) -> torch.Tensor:
        """Dummy tokenization for testing."""
        # Create dummy token IDs
        batch_size = len(raw_data)
        return torch.randint(0, 1000, (batch_size, max_length))
    
    def _dummy_tokenize_for_gptq(self, raw_data: List[Dict[str, Any]], max_length: int) -> List[torch.Tensor]:
        """Dummy tokenization for GPTQ."""
        dummy_tensor = self._dummy_tokenize(raw_data, max_length)
        return [dummy_tensor[i:i+1] for i in range(dummy_tensor.size(0))]

class CalibrationMetrics:
    """Calculate calibration quality metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_distribution_stats(self, data: torch.Tensor) -> Dict[str, float]:
        """Calculate distribution statistics for calibration data."""
        try:
            stats = {
                'mean': float(data.float().mean()),
                'std': float(data.float().std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'median': float(data.float().median()),
                'q25': float(torch.quantile(data.float(), 0.25)),
                'q75': float(torch.quantile(data.float(), 0.75))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate distribution stats: {e}")
            return {}
    
    def calculate_outlier_ratio(self, data: torch.Tensor, threshold: float = 3.0) -> float:
        """Calculate the ratio of outliers in the data."""
        try:
            mean = data.float().mean()
            std = data.float().std()
            
            # Calculate Z-scores
            z_scores = torch.abs((data.float() - mean) / std)
            
            # Count outliers (Z-score > threshold)
            outliers = (z_scores > threshold).sum()
            total_elements = data.numel()
            
            outlier_ratio = float(outliers) / float(total_elements)
            
            return outlier_ratio
            
        except Exception as e:
            logger.error(f"Failed to calculate outlier ratio: {e}")
            return 0.0
    
    def assess_calibration_quality(self, original_data: torch.Tensor, 
                                   processed_data: torch.Tensor) -> Dict[str, float]:
        """Assess the quality of calibration data processing."""
        try:
            original_stats = self.calculate_distribution_stats(original_data)
            processed_stats = self.calculate_distribution_stats(processed_data)
            
            # Calculate information retention metrics
            mean_retention = processed_stats['mean'] / original_stats['mean'] if original_stats['mean'] != 0 else 1.0
            variance_retention = processed_stats['std'] / original_stats['std'] if original_stats['std'] != 0 else 1.0
            
            quality_metrics = {
                'mean_retention': mean_retention,
                'variance_retention': variance_retention,
                'original_outlier_ratio': self.calculate_outlier_ratio(original_data),
                'processed_outlier_ratio': self.calculate_outlier_ratio(processed_data)
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Failed to assess calibration quality: {e}")
            return {}

class CalibrationManager:
    """Main calibration manager that orchestrates the calibration process."""
    
    def __init__(self, dataset_name: str = "wikitext", num_samples: int = 128):
        self.data_loader = CalibrationDataLoader(dataset_name, num_samples)
        self.processor = CalibrationProcessor()
        self.metrics = CalibrationMetrics()
        self.calibration_data = None
    
    def prepare_calibration_data(self, method: str = "static", 
                                tokenizer=None, max_length: int = 512) -> Any:
        """
        Prepare calibration data for the specified quantization method.
        
        Args:
            method: Quantization method (static, gptq, awq)
            tokenizer: Tokenizer to use for processing
            max_length: Maximum sequence length
            
        Returns:
            Processed calibration data
        """
        try:
            logger.info(f"Preparing calibration data for {method} quantization")
            
            # Load raw calibration data
            raw_data = self.data_loader.load_data()
            
            # Update tokenizer
            self.processor.tokenizer = tokenizer
            
            # Process data based on method
            if method == "static":
                processed_data = self.processor.process_for_static_quantization(
                    raw_data, max_length
                )
            elif method == "gptq":
                processed_data = self.processor.process_for_gptq(raw_data, max_length)
            elif method == "awq":
                processed_data = self.processor.process_for_awq(raw_data, max_length)
            else:
                logger.warning(f"Unknown method {method}, using static processing")
                processed_data = self.processor.process_for_static_quantization(
                    raw_data, max_length
                )
            
            # Calculate and log calibration metrics
            if isinstance(processed_data, torch.Tensor):
                stats = self.metrics.calculate_distribution_stats(processed_data)
                outlier_ratio = self.metrics.calculate_outlier_ratio(processed_data)
                
                logger.info("Calibration data statistics calculated",
                           mean=f"{stats.get('mean', 0):.3f}",
                           std=f"{stats.get('std', 0):.3f}",
                           min_val=f"{stats.get('min', 0):.3f}",
                           max_val=f"{stats.get('max', 0):.3f}",
                           outlier_ratio=f"{outlier_ratio:.3f}")
            
            self.calibration_data = processed_data
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to prepare calibration data: {e}")
            raise PiscesLxCoreIOError(f"Failed to prepare calibration data: {e}")
    
    def get_calibration_data(self) -> Any:
        """Get the prepared calibration data."""
        return self.calibration_data
    
    def validate_calibration_data(self, data: Any) -> bool:
        """Validate that calibration data is suitable for quantization."""
        try:
            if data is None:
                logger.error("Calibration data is None")
                return False
            
            if isinstance(data, torch.Tensor):
                if data.numel() == 0:
                    logger.error("Calibration data tensor is empty")
                    return False
                
                if torch.isnan(data).any() or torch.isinf(data).any():
                    logger.error("Calibration data contains NaN or Inf values")
                    return False
                
                # Check for reasonable value range
                if data.max() > 1e6 or data.min() < -1e6:
                    logger.warning("Calibration data has extreme values")
            
            logger.info("Calibration data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Calibration data validation failed: {e}")
            return False
    
    def save_calibration_data(self, filepath: str):
        """Save calibration data to file."""
        try:
            if self.calibration_data is None:
                logger.warning("No calibration data to save")
                return
            
            torch.save(self.calibration_data, filepath)
            logger.info(f"Calibration data saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration data: {e}")
            raise PiscesLxCoreIOError(f"Failed to save calibration data: {e}")
    
    def load_calibration_data(self, filepath: str):
        """Load calibration data from file."""
        try:
            self.calibration_data = torch.load(filepath)
            logger.info(f"Calibration data loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load calibration data: {e}")
            raise PiscesLxCoreIOError(f"Failed to load calibration data: {e}")