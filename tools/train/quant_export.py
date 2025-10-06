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

from typing import List
from utils import PiscesLxCoreLog as LOG
RIGHT = LOG.info; ERROR = LOG.error; DEBUG = LOG.debug
from utils import PiscesLxCoreQuantizationFacade

class PiscesLxToolsQuantExporter:
    """Unified quantization and export facade for train pipeline.

    This class bridges to the legacy tools/quantize.py implementation to keep
    behavior identical while providing a stable class-based API.
    """

    def __init__(self, cfg, hooks, profiler):
        """Initialize the PiscesLxToolsQuantExporter instance.

        Args:
            cfg: Configuration object used for quantization and export.
            hooks: Hooks to be used during the process.
            profiler: Profiler object for performance measurement.
        """
        self.cfg = cfg
        self.hooks = hooks
        self.profiler = profiler

    def prepare(self, model, cfg) -> None:
        """Placeholder method for future needs; keeps API for compatibility.

        Args:
            model: The model to be prepared.
            cfg: Configuration object.
        """
        pass

    def quantize(self, ckpt_path: str, bits: int, save_path: str) -> None:
        """Quantize the model from the given checkpoint and save the result.

        Args:
            ckpt_path (str): Path to the checkpoint file.
            bits (int): Number of bits for quantization.
            save_path (str): Path to save the quantized model.

        Raises:
            SystemExit: If ckpt_path or save_path is not provided.
        """
        if not ckpt_path or not save_path:
            ERROR("quantize requires ckpt_path and save_path")
            raise SystemExit(1)
        # Resolve bits: config-first, then CLI-provided argument
        resolved_bits = None
        try:
            cfg_bits = self.cfg.get('training_config.quant_bits', default=None)
        except Exception:
            cfg_bits = None
        if bits is not None:
            resolved_bits = bits
        elif cfg_bits is not None:
            resolved_bits = cfg_bits
        else:
            resolved_bits = 4

        try:
            resolved_bits = int(resolved_bits)
        except Exception:
            ERROR("quant_bits must be integer (4 or 8)")
            raise SystemExit(1)
        if resolved_bits not in (4, 8):
            ERROR("quant_bits must be one of {4, 8}")
            raise SystemExit(1)

        RIGHT(f"Quantizing model: bits={resolved_bits}, ckpt={ckpt_path} -> {save_path}")
        RIGHT(f"Quantizing checkpoint {ckpt_path} -> {save_path} (bits={resolved_bits})")

        # Use unified utils.quantization to perform quantization
        # Attempt to infer model_size from cfg if available
        model_size = None
        try:
            model_size = self.cfg.get('model.size', default=None)
        except Exception:
            model_size = None
        
        # Use the new facade class instead of the function
        quantizer = PiscesLxCoreQuantizationFacade()
        quantizer.quantize_checkpoint(ckpt_path, save_path, resolved_bits, model_size=model_size)

    def export(self, save_path: str, formats: List[str]) -> None:
        """Export the given weights (quantized or not) to requested formats.

        Currently supports: 'safetensors'. Other formats will be ignored with a log.

        Args:
            save_path (str): Path to the model weights (state_dict or checkpoint with 'model' key).
            formats (List[str]): List of export formats (e.g., ['safetensors']).
        """
        import os
        import torch
        from utils import PiscesLxCoreLog as LOG
        RIGHT = LOG.info; ERROR = LOG.error; DEBUG = LOG.debug

        if not formats:
            return
        if not save_path or not os.path.exists(save_path):
            ERROR(f"export: save_path not found: {save_path}")
            raise SystemExit(1)

        want_safetensors = any(f.lower() == 'safetensors' for f in formats)
        other_formats = [f for f in formats if f.lower() != 'safetensors']
        if other_formats:
            RIGHT(f"Export formats not yet supported and will be skipped: {other_formats}")

        if not want_safetensors:
            return

        try:
            from safetensors.torch import save_file as save_safetensors
        except Exception as e:
            ERROR(f"safetensors not available: {e}. Please install 'safetensors' package.")
            raise SystemExit(1)

        RIGHT(f"Exporting to safetensors from: {save_path}")
        obj = torch.load(save_path, map_location='cpu')
        # Accept either raw state_dict or checkpoint with 'model'
        state_dict = obj['model'] if isinstance(obj, dict) and 'model' in obj else obj
        if not isinstance(state_dict, dict):
            ERROR("export: state_dict must be a dict of tensors")
            raise SystemExit(1)

        # Filter to tensors only; drop non-tensor entries to satisfy safetensors
        tensor_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor):
                tensor_dict[k] = v.detach().cpu()
        if not tensor_dict:
            ERROR("export: no tensor parameters found to export")
            raise SystemExit(1)

        out_path = os.path.splitext(save_path)[0] + '.safetensors'
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        save_safetensors(tensor_dict, out_path)
        RIGHT(f"Safetensors export complete: {out_path}")
