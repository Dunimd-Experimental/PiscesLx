#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import importlib.util
from types import ModuleType
from utils import RIGHT, DEBUG, ERROR
from utils.hooks import PiscesLxCoreHookBus
from .profiler import PiscesLxToolsProfiler
from .config import PiscesLxToolsTrainConfig
from .quant_export import PiscesLxToolsQuantExporter
from .pref_align import PiscesLxToolsPreferenceTrainer

class PiscesLxToolsTrainOrchestrator:
    """
    Orchestrates training workflows for Pisces L1, including:
    - Standard supervised training (legacy-compatible)
    - Quantization and export (via unified interface)
    - Human preference alignment (SFT/DPO/PPO)

    This class preserves original behavior by delegating to legacy tools/train.py
    for standard training, ensuring backward compatibility.
    """

    def __init__(self, args):
        """
        Initialize the training orchestrator.

        Args:
            args: Command line arguments or configuration object.
        """
        self.args = args
        # Create a configuration object from the provided arguments
        self.cfg = PiscesLxToolsTrainConfig.from_args(args)
        # Initialize the hook bus for event handling
        self.hooks = PiscesLxCoreHookBus()
        # Initialize the profiler for performance monitoring
        self.profiler = PiscesLxToolsProfiler()

    def run(self, args) -> None:
        """
        Run the training workflow based on the specified mode.

        Args:
            args: Command line arguments or configuration object.
        """
        # Get the training mode from the configuration, default to "standard"
        mode = self.cfg.get("train.mode", default="standard")
        RIGHT(f"Train orchestrator mode: {mode}")
        if mode == "standard":
            self.run_standard_training()
        elif mode == "quant_export":
            self.run_quant_and_export()
        elif mode == "preference":
            self.run_preference_alignment()
        else:
            ERROR(f"Unknown train.mode: {mode}")

    def run_standard_training(self) -> None:
        """Run standard supervised training via the class-based runner.

        Behavior remains identical because the runner delegates to the
        legacy implementation internally during this migration phase.
        """
        from .runner import PiscesLxToolsTrainRunner
        # Create a runner instance with the provided arguments and configuration
        runner = PiscesLxToolsTrainRunner(self.args, hooks=self.hooks, profiler=self.profiler, cfg=self.cfg)
        # Start the training process
        runner.train()

    def run_quant_and_export(self) -> None:
        """
        Run quantization and export process.

        Quantizes the model and exports it in specified formats if provided.
        """
        # Create a quantization exporter instance
        qe = PiscesLxToolsQuantExporter(self.cfg, self.hooks, self.profiler)
        # Get the quantization bits from args or config, default to 4
        bits = getattr(self.args, "quant_bits", None) or self.cfg.get("quant.bits", default=4)
        # Get the checkpoint path from args or config
        ckpt = getattr(self.args, "ckpt", None) or self.cfg.get("quant.ckpt", default="")
        # Get the save path from args or config
        save = getattr(self.args, "save", None) or self.cfg.get("quant.save", default="")
        # Check if checkpoint and save paths are provided
        if not ckpt or not save:
            ERROR("quant_export requires --ckpt and --save or corresponding config keys")
            raise SystemExit(1)
        # Perform quantization
        qe.quantize(ckpt, bits, save)
        # Get the export formats from config
        export_formats = self.cfg.get("export.formats", default=[])
        if export_formats:
            # Export the quantized model in specified formats
            qe.export(save, export_formats)

    def run_preference_alignment(self) -> None:
        """
        Run human preference alignment training.

        Supports different preference alignment methods like SFT, DPO, and PPO.
        """
        # Create a preference trainer instance
        pa = PiscesLxToolsPreferenceTrainer(self.cfg, self.hooks, self.profiler, args=self.args)
        # Get the preference alignment type from config, default to "sft"
        pref_type = self.cfg.get("train.pref.type", default="sft")
        RIGHT(f"Running preference alignment: {pref_type}")
        if pref_type == "sft":
            pa.run_sft(self.cfg)
        elif pref_type == "dpo":
            pa.run_dpo(self.cfg)
        elif pref_type == "ppo":
            pa.run_ppo(self.cfg)
        else:
            ERROR(f"Unknown train.pref.type: {pref_type}")
            raise SystemExit(1)

    def _load_legacy_train_module(self) -> ModuleType | None:
        """
        Load the legacy tools/train.py as a separate module to avoid name collision
        with the new package tools/train/.

        Returns:
            The loaded module if successful, None otherwise.
        """
        # Get the root directory path
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        # Construct the path to the legacy train.py file
        legacy_path = os.path.join(root, "tools", "train.py")
        if not os.path.exists(legacy_path):
            return None
        # Create a module spec from the legacy file
        spec = importlib.util.spec_from_file_location("tools.train_legacy", legacy_path)
        if spec is None or spec.loader is None:
            return None
        # Create a module object from the spec
        mod = importlib.util.module_from_spec(spec)
        # Add the module to sys.modules
        sys.modules[spec.name] = mod
        # Execute the module
        spec.loader.exec_module(mod)
        return mod
