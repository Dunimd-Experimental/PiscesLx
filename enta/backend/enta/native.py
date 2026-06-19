#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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



"""Native acceleration bridge 鈥?all model-facing operations are Rust-only.

All functions are re-exported directly from the compiled ``enta._native``
Rust extension.  If the extension is not installed, Python's normal
``ModuleNotFoundError`` applies 鈥?build it with::

    cd native && cargo build --release
    # copy target/release/_native.{dll,so} 鈫?backend/enta/_native.pyd
"""

from enta._native import (
    Bm25Index,
    apply_diff,
    build_content_length_header,
    build_lsp_request,
    compute_diff,
    cosine_similarity,
    count_tokens,
    execute_shell,
    glob,
    grep,
    landlock_abi_version,
    landlock_available,
    landlock_full_sandbox,
    landlock_restrict_network,
    landlock_restrict_read_only,
    landlock_workspace_sandbox,
    parse_diagnostics,
    parse_lsp_message,
    permission_check,
    permission_get_policies,
    permission_record_decision,
    permission_set_policies,
    read_file,
    sandbox_execute,
    sandbox_read_file,
    sandbox_write_file,
    search_codebase,
    simd_contains,
    simd_find_all,
    simd_memmem,
    text_similarity,
    write_file,
)

# Backward-compatible alias
glob_pattern = glob


__all__ = [
    "Bm25Index",
    "apply_diff",
    "build_content_length_header",
    "build_lsp_request",
    "compute_diff",
    "cosine_similarity",
    "count_tokens",
    "execute_shell",
    "glob",
    "glob_pattern",
    "grep",
    "landlock_abi_version",
    "landlock_available",
    "landlock_full_sandbox",
    "landlock_restrict_network",
    "landlock_restrict_read_only",
    "landlock_workspace_sandbox",
    "parse_diagnostics",
    "parse_lsp_message",
    "permission_check",
    "permission_get_policies",
    "permission_record_decision",
    "permission_set_policies",
    "read_file",
    "sandbox_execute",
    "sandbox_read_file",
    "sandbox_write_file",
    "search_codebase",
    "simd_contains",
    "simd_find_all",
    "simd_memmem",
    "text_similarity",
    "write_file",
]
