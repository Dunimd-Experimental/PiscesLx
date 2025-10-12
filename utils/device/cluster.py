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

from utils.log.core import PiscesLxCoreLog
from typing import Any, Dict, Literal
from .facade import PiscesLxCoreDeviceFacade

logger = PiscesLxCoreLog("PiscesLx.Utils.Device.Cluster")

class PiscesLxCoreDeviceUnifiedPlanner:
    @staticmethod
    def plan_from_args(args: Any, phase: Literal["train", "infer"] = "train") -> Dict[str, Any]:
        """Generates a unified device plan based on the provided arguments and operation phase.

        This function first retrieves per-host device recommendations and generates a cluster/distributed plan.
        Then it merges device and distributed configurations, respecting explicit argument overrides.
        Finally, it logs a concise summary of the generated plan.

        Args:
            args (Any): Arguments containing device and distribution configurations.
            phase (Literal["train", "infer"], optional): Phase of operation, either "train" or "infer". Defaults to "train".

        Returns:
            Dict[str, Any]: A unified device plan containing device and distribution configurations.
        """

        
        # Retrieve per-host device recommendations using facade
        dev = PiscesLxCoreDeviceFacade(args=args)
        dev_cfg: Dict[str, Any] = dev._auto_setup()

        # Generate cluster/distributed plan from configuration
        from utils.device.dist import PiscesLxCoreDistConfig, PiscesLxCoreDistPlanner
        dist_cfg = PiscesLxCoreDistConfig.from_args(args, phase=phase)
        dist_plan = PiscesLxCoreDistPlanner.plan(dist_cfg)
        plan = dist_plan.to_dict()

        # Extract explicitly specified parameters from arguments
        explicit_dtype = getattr(args, "dtype", None)
        explicit_amp = getattr(args, "amp", None)
        explicit_batch = getattr(args, "batch_size", None)

        # Determine data type for computation
        if explicit_dtype:
            plan["dtype"] = str(explicit_dtype).lower()
        elif str(dist_cfg.dtype).lower() == "auto":
            plan["dtype"] = str(dev_cfg.get("dtype", plan.get("dtype", "fp32"))).lower()

        # Determine whether to use automatic mixed precision
        if explicit_amp is not None:
            plan["amp"] = bool(explicit_amp)
        else:
            plan["amp"] = bool(dev_cfg.get("mixed_precision", plan.get("amp", False)))

        # Add convenience fields for external access
        plan["device_type"] = dev_cfg.get("device_type")
        plan["gpu_ids"] = list(dev_cfg.get("gpu_ids", []) or [])
        
        # Set batch size based on explicit value or device configuration
        if explicit_batch is not None:
            plan["batch_size"] = int(explicit_batch)
        elif "batch_size" in dev_cfg:
            plan["batch_size"] = int(dev_cfg["batch_size"])

        # Include strategy information if available
        if "strategy" in dev_cfg:
            plan["strategy"] = dev_cfg["strategy"]

        # Log a concise summary of the generated plan
        try:
            logger.info(
                "DEVICE",
                {
                    "message": "Plan ready",
                    "phase": phase,
                    "device_type": plan.get("device_type"),
                    "gpu_ids": plan.get("gpu_ids"),
                    "dtype": plan.get("dtype"),
                    "amp": plan.get("amp"),
                    "batch_size": plan.get("batch_size"),
                    "world_size": plan.get("world_size"),
                    "rank": plan.get("rank"),
                    "local_rank": plan.get("local_rank"),
                    "dp": plan.get("dp_size"),
                    "tp": plan.get("tp_size"),
                    "pp": plan.get("pp_size"),
                    "ep": plan.get("ep_size"),
                    "zero_stage": plan.get("zero_stage"),
                    "strategy": plan.get("strategy"),
                },
            )
        except Exception as e:
            logger.debug("failed to log device plan summary", event="device.plan.log_error", phase=phase, error=str(e))

        return plan

    # --- Convenience helpers ---
    @staticmethod
    def plan_train(args: Any) -> Dict[str, Any]:
        """Generates a unified device plan for the training phase.

        Args:
            args (Any): Arguments containing device and distribution configurations.

        Returns:
            Dict[str, Any]: A unified device plan for training.
        """
        return PiscesLxCoreDeviceUnifiedPlanner.plan_from_args(args, phase="train")

    @staticmethod
    def plan_infer(args: Any) -> Dict[str, Any]:
        """Generates a unified device plan for the inference phase.

        Args:
            args (Any): Arguments containing device and distribution configurations.

        Returns:
            Dict[str, Any]: A unified device plan for inference.
        """
        return PiscesLxCoreDeviceUnifiedPlanner.plan_from_args(args, phase="infer")

    @staticmethod
    def init_pg() -> None:
        """Initializes the process group for distributed computing.

        Raises:
            Exception: If the process group initialization fails.
        """

        from utils.device import PiscesLxCoreProcessGroupManager
        
        # Attempt to initialize the process group
        try:
            PiscesLxCoreProcessGroupManager.init()
            logger.info("PG", {"message": "Process group initialized"})
        except Exception as e:
            logger.error("PG", {"message": "Process group init failed", "error": str(e)})
            raise

    @staticmethod
    def finalize_pg() -> None:
        """Finalizes and cleans up the process group.

        Raises:
            Exception: If the process group finalization fails.
        """

        from utils.device import PiscesLxCoreProcessGroupManager
        
        # Attempt to finalize the process group
        try:
            PiscesLxCoreProcessGroupManager.finalize()
            logger.info("PG", {"message": "Process group finalized"})
        except Exception as e:
            logger.error("PG", {"message": "Process group finalize failed", "error": str(e)})
            raise

    @staticmethod
    def wrap_model_for_train(model: Any, plan: Dict[str, Any]) -> Any:
        """Wraps the model for training according to the device plan.

        Args:
            model (Any): The model to be wrapped.
            plan (Dict[str, Any]): The device plan containing parallelization configurations.

        Returns:
            Any: The wrapped model.
        """
        logger = PiscesLxCoreLog()
        from utils.device import PiscesLxCoreModelParallelizer
        
        # Convert the dictionary plan to a lightweight object with the attributes expected by the wrapper
        class _P:  # pragma: no cover
            def __init__(self, d: Dict[str, Any]) -> None:
                self.__dict__.update(d)
        
        # Wrap the model for training using the parallelizer
        wrapped = PiscesLxCoreModelParallelizer.wrap_for_train(model, _P(plan))
        
        # Log the wrapping mode used
        try:
            mode = (
                "DDP" if getattr(wrapped, "__class__", None).__name__ == "DistributedDataParallel" else
                "DataParallel" if getattr(wrapped, "__class__", None).__name__ == "DataParallel" else
                "SingleProcess"
            )
            logger.info("WRAP", {"message": "Model wrapped for train", "mode": mode})
        except Exception as e:
            logger.debug("failed to log model wrap mode for train", event="device.wrap.train_log_error", error=str(e))
            
        return wrapped

    @staticmethod
    def wrap_model_for_infer(model: Any, plan: Dict[str, Any]) -> Any:
        """Wraps the model for inference according to the device plan.

        Args:
            model (Any): The model to be wrapped.
            plan (Dict[str, Any]): The device plan containing parallelization configurations.

        Returns:
            Any: The wrapped model.
        """
        logger = PiscesLxCoreLog()
        from utils.device import PiscesLxCoreModelParallelizer
        
        # Convert the dictionary plan to a lightweight object with the attributes expected by the wrapper
        class _P:  # pragma: no cover
            def __init__(self, d: Dict[str, Any]) -> None:
                self.__dict__.update(d)
                
        # Wrap the model for inference using the parallelizer
        wrapped = PiscesLxCoreModelParallelizer.wrap_for_infer(model, _P(plan))
        
        # Log the wrapping mode used
        try:
            mode = (
                "DDP" if getattr(wrapped, "__class__", None).__name__ == "DistributedDataParallel" else
                "DataParallel" if getattr(wrapped, "__class__", None).__name__ == "DataParallel" else
                "SingleProcess"
            )
            logger.info("WRAP", {"message": "Model wrapped for infer", "mode": mode})
        except Exception as e:
            logger.debug("failed to log model wrap mode for infer", event="device.wrap.infer_log_error", error=str(e))
            
        return wrapped
