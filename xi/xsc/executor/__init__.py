#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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
Xi Executor Module

Command execution layer for Xi Studio.
"""

from .executor import XiExecutor
from .argv import build_argv, build_argv_from_schema
from .stream import stream_output, get_output_stream
from .process import control_process, generate_run_id, update_status_on_exit

from .execute import execute_command

__all__ = [
    "XiExecutor",
    "build_argv",
    "build_argv_from_schema",
    "stream_output",
    "get_output_stream",
    "control_process",
    "generate_run_id",
    "update_status_on_exit",
    "execute_command",
]
