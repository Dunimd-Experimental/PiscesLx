#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

from __future__ import annotations
"""
Standalone indexing worker — runs in a separate process so the main server
is not blocked by CPU-intensive codebase scanning.

Usage:
    python -m codebase.index_worker --ws-id <id> --ws-path <path> --data-dir <dir>

Writes progress to ``<data-dir>/iwork/<ws-id>/index_progress.json``.
"""

import argparse
import json
import os
import sys
import time


def _progress_path(data_dir: str, ws_id: str) -> str:
    return os.path.join(data_dir, "iwork", ws_id, "index_progress.json")


def _metadata_path(data_dir: str, ws_id: str) -> str:
    return os.path.join(data_dir, "iwork", ws_id, "index_metadata.json")


def _write_progress(filepath: str, progress: int, status: str, files: int = 0) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"progress": progress, "status": status, "files": files}, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encre codebase indexing worker")
    parser.add_argument("--ws-id", required=True)
    parser.add_argument("--ws-path", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    # Ensure the project root is on sys.path for imports
    _self_dir = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_self_dir)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from codebase.indexer import EncreCodeIndex  # noqa: E402

    progress_file = _progress_path(args.data_dir, args.ws_id)
    meta_file = _metadata_path(args.data_dir, args.ws_id)

    total_files = 0

    def progress_cb(rel_path: str, total: int) -> None:
        nonlocal total_files
        total_files = total
        # Map file index (0..total) to percentage (0..99)
        pct = min(99, int(total * 100 / max(total, 1))) if total > 0 else 0
        _write_progress(progress_file, pct, "indexing")

    try:
        _write_progress(progress_file, 0, "indexing")
        idx = EncreCodeIndex(args.ws_path)
        idx.scan(progress_cb=progress_cb)
        file_count = len(idx._modules)

        # Save metadata
        os.makedirs(os.path.dirname(meta_file), exist_ok=True)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"files": file_count, "indexed_at": time.time()}, f)

        _write_progress(progress_file, 100, "ready", files=file_count)
    except Exception as e:
        _write_progress(progress_file, 0, f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
