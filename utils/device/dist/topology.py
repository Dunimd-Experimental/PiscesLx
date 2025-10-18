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

import torch
from typing import Optional
from dataclasses import dataclass

@dataclass
class PiscesLxCoreTopologySuggestion:
    """Represents a suggested parallel topology for the PiscesLx core.

    This data class encapsulates the suggested sizes for data parallelism (dp_size),
    tensor parallelism (tp_size), and pipeline parallelism (pp_size) based on 
    the available GPU resources.

    Attributes:
        dp_size (int): The degree of data parallelism.
        tp_size (int): The degree of tensor parallelism.
        pp_size (int): The degree of pipeline parallelism.
    """
    dp_size: int
    tp_size: int
    pp_size: int


class PiscesLxCoreTopologyOptimizer:
    """Optimizer class for suggesting optimal parallel topologies based on available GPU resources."""
    
    def suggest_topology(self, total_gpus: Optional[int] = None) -> PiscesLxCoreTopologySuggestion:
        """Suggests an appropriate parallel topology based on the number of available GPUs.

        This method applies a heuristic strategy to recommend a conservative 
        parallel configuration. If the number of GPUs is not specified, it attempts 
        to automatically detect the count of available CUDA devices.

        Args:
            total_gpus (Optional[int]): The total number of available GPUs. 
                If not provided, the function will attempt to detect the number 
                of CUDA-capable devices. Defaults to None.

        Returns:
            PiscesLxCoreTopologySuggestion: A data class containing the recommended 
                parallel configuration (dp_size, tp_size, pp_size).
        """
        # Automatically detect the number of available GPUs if not specified
        if total_gpus is None:
            try:
                total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except Exception:
                total_gpus = 0

        # Suggest topology based on the number of available GPUs
        if total_gpus >= 8:
            return PiscesLxCoreTopologySuggestion(dp_size=2, tp_size=2, pp_size=2)
        elif total_gpus >= 4:
            return PiscesLxCoreTopologySuggestion(dp_size=2, tp_size=2, pp_size=1)
        elif total_gpus >= 2:
            return PiscesLxCoreTopologySuggestion(dp_size=2, tp_size=1, pp_size=1)
        else:
            return PiscesLxCoreTopologySuggestion(dp_size=1, tp_size=1, pp_size=1)

# Legacy function for backward compatibility
def suggest_topology(total_gpus: Optional[int] = None) -> PiscesLxCoreTopologySuggestion:
    """Suggests an appropriate parallel topology based on the number of available GPUs.

    This function applies a heuristic strategy to recommend a conservative 
    parallel configuration. If the number of GPUs is not specified, it attempts 
    to automatically detect the count of available CUDA devices.

    Args:
        total_gpus (Optional[int]): The total number of available GPUs. 
            If not provided, the function will attempt to detect the number 
            of CUDA-capable devices. Defaults to None.

    Returns:
        PiscesLxCoreTopologySuggestion: A data class containing the recommended 
            parallel configuration (dp_size, tp_size, pp_size).
    """
    optimizer = PiscesLxCoreTopologyOptimizer()
    return optimizer.suggest_topology(total_gpus)
