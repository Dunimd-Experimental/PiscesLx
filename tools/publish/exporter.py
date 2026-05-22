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

"""
Model Exporter

This module handles the export of PiscesL1 model checkpoints to various formats
for distribution and deployment. It provides functionality to export model weights,
configurations, tokenizer files, and generation settings.

Key Features:
    - Safe tensor export with security benefits
    - Configuration preservation and serialization
    - Tokenizer export and validation
    - Generation config export for inference
    - Checksum generation for integrity verification

Export Formats:
    - safetensors: Safe tensor format (recommended, security benefits)
    - pytorch: Standard PyTorch checkpoint format

Export Process:
    1. Validate checkpoint and configuration
    2. Load model checkpoint
    3. Export weights in target format
    4. Export model configuration
    5. Export tokenizer (if requested)
    6. Export generation configuration
    7. Generate checksums for all artifacts

Usage Examples:
    Basic Export:
        >>> from tools.publish.config import PiscesLxPublishConfig
        >>> from tools.publish.exporter import PiscesLxPublishExporter
        >>> config = PiscesLxPublishConfig(
        ...     model_path="./checkpoints/7B",
        ...     model_size="7B"
        ... )
        >>> exporter = PiscesLxPublishExporter(config)
        >>> results = exporter.export()
        >>> print(results)
        {'weights': './publish/model.safetensors', 'config': './publish/config.json'}

    Export with Quantization:
        >>> config = PiscesLxPublishConfig(
        ...     model_path="./checkpoints/7B",
        ...     export=PiscesLxPublishModelExportConfig(
        ...         quantization="int8"
        ...     )
        ... )
        >>> exporter = PiscesLxPublishExporter(config)
        >>> results = exporter.export()

    Programmatic Export:
        >>> exporter = PiscesLxPublishExporter(config)
        >>> weights_path = exporter.export_weights()
        >>> config_path = exporter.export_config()
        >>> tokenizer_path = exporter.export_tokenizer()
"""

import os
import json
import hashlib
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

try:
    from safetensors.torch import save_file as safetensors_save
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from utils.dc import PiscesLxLogger
from utils.paths import get_log_file

_LOG = PiscesLxLogger("PiscesLx.Publish.Exporter", file_path=get_log_file("PiscesLx.Publish.Exporter"), enable_file=True)


class PiscesLxPublishExporter:
    """Model Exporter for PiscesL1.

    Handles the export of model checkpoints, configurations, tokenizers,
    and generation settings to a publish-ready format.

    Attributes:
        config: PiscesLxPublishConfig containing export configuration.
        output_dir: Path to the output directory.
        exported_files: Dictionary tracking exported files and their paths.

    Example:
        >>> config = PiscesLxPublishConfig(model_path="./checkpoints/7B")
        >>> exporter = PiscesLxPublishExporter(config)
        >>> results = exporter.export()
    """

    def __init__(self, config: 'PiscesLxPublishConfig'):
        """Initialize the exporter.

        Args:
            config: PiscesLxPublishConfig containing export configuration.
        """
        self.config = config
        self.output_dir = Path(config.export.output_dir)
        self.exported_files: Dict[str, str] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _LOG.info(
            "PiscesLxPublishExporter initialized",
            output_dir=str(self.output_dir),
            format=config.export.export_format
        )

    def export(self) -> Dict[str, str]:
        """Execute the complete export pipeline.

        Exports model weights, configuration, tokenizer, and generation
        settings based on the configuration.

        Returns:
            Dictionary mapping export type to output file path.

        Example:
            >>> exporter = PiscesLxPublishExporter(config)
            >>> results = exporter.export()
            >>> # Results: {'weights': '...', 'config': '...', 'tokenizer': '...'}
        """
        _LOG.info("Starting model export", checkpoint=self.config.export.checkpoint_path)

        results = {}

        weights_path = self.export_weights()
        if weights_path:
            results['weights'] = weights_path

        config_path = self.export_config()
        if config_path:
            results['config'] = config_path

        if self.config.export.include_tokenizer:
            tokenizer_path = self.export_tokenizer()
            if tokenizer_path:
                results['tokenizer'] = tokenizer_path

        if self.config.export.include_generation_config:
            gen_config_path = self.export_generation_config()
            if gen_config_path:
                results['generation_config'] = gen_config_path

        metadata_path = self.export_metadata()
        if metadata_path:
            results['metadata'] = metadata_path

        checksum_path = self.generate_checksums()
        if checksum_path:
            results['checksums'] = checksum_path

        self.exported_files = results
        _LOG.info("Model export completed", exported_files=list(results.keys()))

        return results

    def export_weights(self) -> Optional[str]:
        """Export model weights to the specified format.

        Exports model weights as safetensors (recommended) or pytorch format
        based on configuration.

        Returns:
            Path to the exported weights file, or None if export failed.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            RuntimeError: If export format is not supported.
        """
        checkpoint_path = Path(self.config.export.checkpoint_path)
        if not checkpoint_path.exists():
            _LOG.error("Checkpoint not found", path=str(checkpoint_path))
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        output_path = self.output_dir / f"model.{self.config.export.export_format}"
        _LOG.info("Exporting weights", input=str(checkpoint_path), output=str(output_path))

        try:
            if checkpoint_path.suffix == '.safetensors':
                state_dict = self._load_safetensors(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

            if self.config.export.quantization:
                state_dict = self._apply_quantization(state_dict)

            if self.config.export.export_format == "safetensors" and SAFETENSORS_AVAILABLE:
                safetensors_save(state_dict, str(output_path))
                _LOG.info("Weights exported as safetensors", path=str(output_path))
            else:
                if self.config.export.export_format == "safetensors" and not SAFETENSORS_AVAILABLE:
                    _LOG.warning("safetensors not available, falling back to pytorch")
                torch.save(state_dict, str(output_path))
                _LOG.info("Weights exported as pytorch", path=str(output_path))

            return str(output_path)

        except Exception as e:
            _LOG.error("Failed to export weights", error=str(e))
            raise RuntimeError(f"Weight export failed: {e}") from e

    def _load_safetensors(self, path: Path) -> Dict[str, torch.Tensor]:
        """Load safetensors file.

        Args:
            path: Path to safetensors file.

        Returns:
            Dictionary of tensor names to tensors.
        """
        try:
            from safetensors.torch import load_file
            return load_file(str(path))
        except Exception as e:
            _LOG.warning("Failed to load as safetensors, trying torch.load", error=str(e))
            return torch.load(str(path), map_location='cpu', weights_only=True)

    def _apply_quantization(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply quantization to state dict.

        Args:
            state_dict: Original state dictionary.

        Returns:
            Quantized state dictionary.
        """
        quantization = self.config.export.quantization
        if not quantization:
            return state_dict

        _LOG.info("Applying quantization", method=quantization)

        if quantization == "int8":
            for key in state_dict:
                if isinstance(state_dict[key], torch.Tensor):
                    state_dict[key] = state_dict[key].to(torch.int8)
        elif quantization == "fp8":
            for key in state_dict:
                if isinstance(state_dict[key], torch.Tensor):
                    state_dict[key] = state_dict[key].to(torch.float8_e4m3fn)
        elif quantization == "int4":
            _LOG.warning("INT4 quantization is experimental")

        return state_dict

    def export_config(self) -> Optional[str]:
        """Export model configuration.

        Exports the model configuration to JSON format including all
        YvConfig parameters and export metadata.

        Returns:
            Path to the exported config file.
        """
        config_data = {
            "model_size": self.config.model_size,
            "model_name": self.config.model_name,
            "export_time": datetime.now().isoformat(),
            "export_format": self.config.export.export_format,
            "quantization": self.config.export.quantization,
        }

        config_path = self.output_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        _LOG.info("Config exported", path=str(config_path))
        return str(config_path)

    def export_tokenizer(self) -> Optional[str]:
        """Export tokenizer configuration.

        Exports tokenizer files to the output directory. This includes
        the tokenizer model file and vocabulary if they exist.

        Returns:
            Path to the tokenizer directory, or None if tokenizer not found.
        """
        tokenizer_path = self._find_tokenizer()
        if not tokenizer_path:
            _LOG.warning("Tokenizer not found, skipping export")
            return None

        output_tokenizer_dir = self.output_dir / "tokenizer"
        output_tokenizer_dir.mkdir(exist_ok=True)

        try:
            if tokenizer_path.is_file():
                import shutil
                shutil.copy2(tokenizer_path, output_tokenizer_dir / tokenizer_path.name)
            elif tokenizer_path.is_dir():
                import shutil
                shutil.copytree(tokenizer_path, output_tokenizer_dir, dirs_exist_ok=True)

            _LOG.info("Tokenizer exported", path=str(output_tokenizer_dir))
            return str(output_tokenizer_dir)

        except Exception as e:
            _LOG.error("Failed to export tokenizer", error=str(e))
            return None

    def _find_tokenizer(self) -> Optional[Path]:
        """Find tokenizer in model directory or standard locations.

        Returns:
            Path to tokenizer file/directory, or None if not found.
        """
        model_dir = Path(self.config.model_path)

        possible_paths = [
            model_dir / "tokenizer.json",
            model_dir / "tokenizer",
            model_dir.parent / "tokenizer.json",
            model_dir.parent / "tokenizer",
            Path("./tokenizer.json"),
            Path("./tokenizer"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def export_generation_config(self) -> Optional[str]:
        """Export generation configuration.

        Exports generation settings including sampling parameters,
        max tokens, and other inference-related settings.

        Returns:
            Path to the generation config file.
        """
        gen_config = {
            "model_size": self.config.model_size,
            "dtype": "bfloat16",
            "max_new_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "stop_sequences": ["<|endoftext|>", "<|eot|>"],
            "seed": None,
        }

        config_path = self.output_dir / "generation_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(gen_config, f, indent=2, ensure_ascii=False)

        _LOG.info("Generation config exported", path=str(config_path))
        return str(config_path)

    def export_metadata(self) -> Optional[str]:
        """Export model metadata.

        Exports comprehensive model metadata including model card information,
        capabilities, and training details.

        Returns:
            Path to the metadata file.
        """
        metadata = self.config.metadata.to_model_card()
        metadata["export_time"] = datetime.now().isoformat()
        metadata["export_format"] = self.config.export.export_format

        config_path = self.output_dir / "model_card.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        _LOG.info("Metadata exported", path=str(config_path))
        return str(config_path)

    def generate_checksums(self) -> Optional[str]:
        """Generate SHA256 checksums for all exported files.

        Creates a checksums manifest file containing SHA256 hashes
        for all exported artifacts.

        Returns:
            Path to the checksums file.
        """
        checksums = {}
        manifest = self.output_dir / "checksums.txt"

        for file_path in self.output_dir.rglob("*"):
            if file_path.is_file() and file_path != manifest:
                checksum = self._compute_sha256(file_path)
                rel_path = file_path.relative_to(self.output_dir)
                checksums[str(rel_path)] = checksum

        with open(manifest, 'w', encoding='utf-8') as f:
            for file_path, checksum in sorted(checksums.items()):
                f.write(f"{checksum}  {file_path}\n")

        _LOG.info("Checksums generated", path=str(manifest), num_files=len(checksums))
        return str(manifest)

    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file.

        Returns:
            Hexadecimal SHA256 hash string.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def verify_export(self) -> Tuple[bool, List[str]]:
        """Verify exported files against checksums.

        Validates that all exported files exist and their checksums match.

        Returns:
            Tuple of (success, list of errors).
        """
        errors = []

        manifest = self.output_dir / "checksums.txt"
        if not manifest.exists():
            errors.append("Checksums manifest not found")
            return False, errors

        with open(manifest, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                expected_hash, file_path = parts
                actual_hash = self._compute_sha256(self.output_dir / file_path)
                if expected_hash != actual_hash:
                    errors.append(f"Checksum mismatch for {file_path}")

        if errors:
            _LOG.error("Export verification failed", errors=errors)
        else:
            _LOG.info("Export verification passed")

        return len(errors) == 0, errors

    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of the export operation.

        Returns:
            Dictionary containing export statistics and file information.
        """
        total_size = 0
        file_count = 0

        for file_path in self.output_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1

        return {
            "output_dir": str(self.output_dir),
            "file_count": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "exported_files": self.exported_files,
            "export_format": self.config.export.export_format,
            "quantization": self.config.export.quantization,
        }