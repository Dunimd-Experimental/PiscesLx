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

from .orchestrator import PiscesLxToolsTrainOrchestrator

def train(args):
    """
    Serve as a compatibility bridge for manage.py to call the new orchestrator.

    This function maintains the original public entrypoint `tools.train.train(args)` unchanged,
    while delegating the actual training task to the class-based implementation.

    Args:
        args: The input arguments required for the training process.

    Returns:
        None
    """
    import warnings as _warnings
    _warnings.warn(
        "tools.train.train(args) is deprecated and will be removed in a future version. "
        "Please use class-based facade via Runner/Orchestrator: PiscesLxToolsTrainImpl().set_context(...); .train(args).",
        DeprecationWarning,
        stacklevel=2,
    )
    # Initialize the orchestrator instance with the provided arguments
    orchestrator = PiscesLxToolsTrainOrchestrator(args)
    # Call the run method of the orchestrator to start the training process
    orchestrator.run(args)
