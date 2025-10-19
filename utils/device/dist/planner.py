#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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
import torch
from .env import PiscesLxCoreClusterEnv
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Literal
from utils.log.core import PiscesLxCoreLog

logger = PiscesLxCoreLog("PiscesLx.Utils.Device.Dist.Planner")

@dataclass
class PiscesLxCoreDistConfig:
    """Configuration class for distributed training and inference of PiscesLx core.

    Attributes:
        phase (Literal["train", "infer"]): Execution phase, either "train" or "infer". Defaults to "train".
        dp_size (int): Data parallel size. Defaults to 1.
        tp_size (int): Tensor parallel size. Defaults to 1.
        pp_size (int): Pipeline parallel size. Defaults to 1.
        ep_size (int): Expert parallel size. Defaults to 1.
        zero_stage (int): ZeRO optimization stage. Defaults to 0.
        dtype (str): Data type, options are "auto", "fp16", "bf16", or "fp32". Defaults to "auto".
        amp (Optional[bool]): Automatic Mixed Precision flag. Defaults to None.
    """
    phase: Literal["train", "infer"] = "train"
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    zero_stage: int = 0
    dtype: str = "auto"
    amp: Optional[bool] = None

    @classmethod
    def from_args(cls, args: Any, phase: Literal["train", "infer"] = "train") -> "PiscesLxCoreDistConfig":
        """Create a PiscesLxCoreDistConfig instance from the given arguments.

        Args:
            args (Any): Argument object containing distributed configuration parameters.
            phase (Literal["train", "infer"]): Execution phase, either "train" or "infer". Defaults to "train".

        Returns:
            PiscesLxCoreDistConfig: A new instance of PiscesLxCoreDistConfig.
        """
        def _get(name: str, default: Any) -> Any:
            """Get the attribute value from the args object.
            
            Args:
                name (str): The name of the attribute.
                default (Any): The default value if the attribute does not exist.
                
            Returns:
                Any: The value of the attribute or the default value.
            """
            return getattr(args, name, default)
        return cls(
            phase=phase,
            dp_size=int(_get("dp_size", 1)),
            tp_size=int(_get("tp_size", 1)),
            pp_size=int(_get("pp_size", 1)),
            ep_size=int(_get("ep_size", 1)),
            zero_stage=int(_get("zero_stage", 0)),
            dtype=str(_get("dtype", "auto")).lower(),
            amp=_get("amp", None),
        )

@dataclass
class PiscesLxCoreDistPlan:
    """Plan class for distributed execution of PiscesLx core.

    Attributes:
        device (str): Device to use for computation.
        local_rank (int): Local rank of the process.
        rank (int): Global rank of the process.
        world_size (int): Total number of processes.
        master_addr (Optional[str]): Address of the master node.
        master_port (Optional[int]): Port of the master node.
        node_rank (int): Rank of the current node.
        nnodes (int): Total number of nodes.
        dp_size (int): Data parallel size.
        tp_size (int): Tensor parallel size.
        pp_size (int): Pipeline parallel size.
        ep_size (int): Expert parallel size.
        zero_stage (int): ZeRO optimization stage.
        dtype (str): Data type.
        amp (bool): Automatic Mixed Precision flag.
    """
    device: str
    local_rank: int
    rank: int
    world_size: int
    # Cluster
    master_addr: Optional[str]
    master_port: Optional[int]
    node_rank: int
    nnodes: int
    dp_size: int
    tp_size: int
    pp_size: int
    ep_size: int
    zero_stage: int
    dtype: str
    amp: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert the instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the instance.
        """
        return asdict(self)

class PiscesLxCoreDistPlanner:
    @staticmethod
    def _detect_env() -> Dict[str, int]:
        """Detect distributed environment variables.

        Returns:
            Dict[str, int]: A dictionary containing LOCAL_RANK, RANK, and WORLD_SIZE.
        """
        def _env_i(name: str, default: int) -> int:
            """Get the environment variable value as an integer.
            
            Args:
                name (str): The name of the environment variable.
                default (int): The default value if the environment variable does not exist or is not a valid integer.
                
            Returns:
                int: The value of the environment variable or the default value.
            """
            try:
                return int(os.environ.get(name, default))
            except Exception:
                return default
        return {
            "LOCAL_RANK": _env_i("LOCAL_RANK", -1),
            "RANK": _env_i("RANK", 0),
            "WORLD_SIZE": _env_i("WORLD_SIZE", 1),
        }

    @staticmethod
    def _pick_device(local_rank: int) -> str:
        """Pick the appropriate device based on the local rank and available resources.

        Args:
            local_rank (int): Local rank of the process.

        Returns:
            str: Device string, e.g., "cpu", "cuda:0".
        """
        if torch is None:
            return "cpu"
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            if local_rank is not None and local_rank >= 0:
                return f"cuda:{local_rank}"
            # Default to the first device explicitly to avoid implicit device selection
            return "cuda:0"
        return "cpu"

    @staticmethod
    def _resolve_dtype(dtype: str) -> str:
        """Resolve the data type based on the input and available resources.

        Args:
            dtype (str): Input data type, can be "auto", "fp16", "bf16", or "fp32".

        Returns:
            str: Resolved data type.
        """
        d = (dtype or "auto").lower()
        if d != "auto":
            return d
        if torch is None:
            return "fp32"
        # Prefer bf16 when available, then fp16 on CUDA, otherwise fp32
        if getattr(torch, "cuda", None) and torch.cuda.is_available():
            # Use bf16 if Ampere+ architecture and bf16 is supported, otherwise use fp16
            try:
                if hasattr(torch, "bfloat16"):
                    return "bf16"
            except Exception as e:
                # Log bfloat16 detection failures for debugging purposes
                logger.debug("bfloat16 detection failed, fallback to fp16: %s", e)
            return "fp16"
        return "fp32"

    @staticmethod
    def _resolve_amp(dtype: str, amp: Optional[bool]) -> bool:
        """Resolve the Automatic Mixed Precision flag.

        Args:
            dtype (str): Data type.
            amp (Optional[bool]): Input AMP flag.

        Returns:
            bool: Resolved AMP flag.
        """
        if amp is not None:
            return bool(amp)
        return dtype in ("fp16", "bf16")

    @staticmethod
    def _fit_parallel_sizes(dp: int, tp: int, pp: int, world_size: int) -> tuple[int, int, int]:
        """Ensure the product of dp, tp, and pp does not exceed world_size by reducing them in the order of dp -> tp -> pp.

        Args:
            dp (int): Data parallel size.
            tp (int): Tensor parallel size.
            pp (int): Pipeline parallel size.
            world_size (int): Total number of processes.

        Returns:
            tuple[int, int, int]: Adjusted parallel sizes.
        """
        dp = max(1, int(dp))
        tp = max(1, int(tp))
        pp = max(1, int(pp))
        ws = max(1, int(world_size))

        def prod(a: int, b: int, c: int) -> int:
            """Calculate the product of three integers.
            
            Args:
                a (int): The first integer.
                b (int): The second integer.
                c (int): The third integer.
                
            Returns:
                int: The product of a, b, and c.
            """
            return a * b * c

        if prod(dp, tp, pp) <= ws:
            return dp, tp, pp

        # Reduce dp first, then tp, and finally pp
        while prod(dp, tp, pp) > ws and dp > 1:
            dp //= 2
        while prod(dp, tp, pp) > ws and tp > 1:
            tp //= 2
        while prod(dp, tp, pp) > ws and pp > 1:
            pp //= 2

        # Final adjustment
        if prod(dp, tp, pp) > ws:
            # Fall back to pure data parallel as a safe default
            dp = ws
            tp = 1
            pp = 1

        return dp, tp, pp

    @classmethod
    def plan(cls, cfg: PiscesLxCoreDistConfig) -> PiscesLxCoreDistPlan:
        """Generate a distributed execution plan based on the given configuration.

        Args:
            cfg (PiscesLxCoreDistConfig): Distributed configuration.

        Returns:
            PiscesLxCoreDistPlan: A distributed execution plan.
        """
        cluster = PiscesLxCoreClusterEnv.detect()
        world_size = max(1, cluster.world_size)
        local_rank = cluster.local_rank
        rank = cluster.rank

        # If user didn't specify sizes, provide sensible defaults based on GPU count
        dp, tp, pp, ep = cfg.dp_size, cfg.tp_size, cfg.pp_size, cfg.ep_size
        if torch is not None and (dp, tp, pp, ep) == (1, 1, 1, 1):
            try:
                ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except Exception:
                ngpu = 0
            # Default split guided by GPU count; prefer DP first, then TP, then PP
            if ngpu >= 8:
                dp, tp, pp = 4, 2, 1
            elif ngpu >= 4:
                dp, tp, pp = 2, 2, 1
            elif ngpu >= 2:
                dp, tp, pp = 2, 1, 1

        # Ensure the product of parallel sizes does not exceed world size
        dp, tp, pp = cls._fit_parallel_sizes(dp, tp, pp, world_size)

        dtype = cls._resolve_dtype(cfg.dtype)
        amp = cls._resolve_amp(dtype, cfg.amp)
        device = cls._pick_device(local_rank)

        return PiscesLxCoreDistPlan(
            device=device,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            master_addr=cluster.master_addr,
            master_port=cluster.master_port,
            node_rank=cluster.node_rank,
            nnodes=cluster.nnodes,
            dp_size=int(dp),
            tp_size=int(tp),
            pp_size=int(pp),
            ep_size=int(ep),
            zero_stage=int(cfg.zero_stage),
            dtype=dtype,
            amp=amp,
        )

    @classmethod
    def plan_train(cls, cfg: PiscesLxCoreDistConfig) -> PiscesLxCoreDistPlan:
        """Generate a distributed execution plan for training.

        Args:
            cfg (PiscesLxCoreDistConfig): Distributed configuration.

        Returns:
            PiscesLxCoreDistPlan: A distributed execution plan for training.
        """
        cfg.phase = "train"
        return cls.plan(cfg)

    @classmethod
    def plan_infer(cls, cfg: PiscesLxCoreDistConfig) -> PiscesLxCoreDistPlan:
        """Generate a distributed execution plan for inference.

        Args:
            cfg (PiscesLxCoreDistConfig): Distributed configuration.

        Returns:
            PiscesLxCoreDistPlan: A distributed execution plan for inference.
        """
        cfg.phase = "infer"
        return cls.plan(cfg)
