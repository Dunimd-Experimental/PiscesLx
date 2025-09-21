#!/usr/bin/env/python3

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
import shutil
from typing import Dict, Any
from utils import get_cache_manager, RIGHT

def cache_stats() -> Dict[str, Any]:
    """
    Return current cache statistics for the entire .pisceslx root directory and its cache subdirectory.

    Returns:
        Dict[str, Any]: A dictionary containing cache statistics for the .pisceslx root and cache subdirectory.
                        Each entry includes base directory path, total file count, total size in bytes, 
                        and total size in megabytes.
    """
    import os
    from pathlib import Path
    cm = get_cache_manager()
    # Convert the base cache directory to a Path object
    cache_dir = Path(str(cm.base_cache_dir))
    # Get the parent directory of the cache directory, which is the .pisceslx root directory
    root_dir = cache_dir.parent

    def _walk_stats(root: Path) -> Dict[str, Any]:
        """
        Calculate cache statistics for a given directory.

        Args:
            root (Path): The root directory to calculate statistics for.

        Returns:
            Dict[str, Any]: A dictionary containing base directory path, total file count, 
                            total size in bytes, and total size in megabytes.
        """
        total_files = 0
        total_size = 0
        # Traverse the directory tree
        for r, dnames, fnames in os.walk(str(root)):
            for fn in fnames:
                fp = os.path.join(r, fn)
                try:
                    total_files += 1
                    total_size += os.path.getsize(fp)
                except Exception:
                    # Skip files that cannot be accessed
                    pass
        return {
            "base_dir": str(root),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }

    return {
        "pisces_root": _walk_stats(root_dir),
        "cache_subdir": _walk_stats(cache_dir),
    }


def clear_dataset_cache() -> None:
    """
    Clear only the dataset-related cache directory.
    Re-create the directory for future use after clearing.
    """
    cm = get_cache_manager()
    ds_dir = cm.get_cache_dir("dataset")
    if os.path.exists(ds_dir):
        # Remove the dataset cache directory if it exists
        shutil.rmtree(ds_dir, ignore_errors=True)
    # Re-create the directory for future use
    cm.get_cache_dir("dataset")
    RIGHT("Dataset cache cleared.")


def clear_all_cache() -> None:
    """
    Clear ALL cache under the base cache directory.
    Re-create the base cache directory after clearing.
    """
    cm = get_cache_manager()
    base = cm.base_cache_dir
    if os.path.exists(base):
        # Remove the base cache directory if it exists
        shutil.rmtree(base, ignore_errors=True)
    # Re-create the base cache directory
    base.mkdir(parents=True, exist_ok=True)
    RIGHT("All cache cleared.")


def clear_downloads_cache() -> None:
    """
    Clear downloaded datasets and download caches under .pisceslx/cache/.

    - data downloaded by data/download.py: /.pisceslx/cache/data_cache
    - downloads caches: /.pisceslx/cache/datatemp
    """
    cm = get_cache_manager()
    base = cm.base_cache_dir  # /.pisceslx/cache
    data_dir = base / "data_cache"
    datatemp_dir = base / "datatemp"
    # Iterate over the data cache and download temp directories
    for p in [data_dir, datatemp_dir]:
        if p.exists():
            # Remove the directory if it exists
            shutil.rmtree(p, ignore_errors=True)
    RIGHT("Downloads and caches cleared.")


__all__ = [
    "cache_stats",
    "clear_dataset_cache",
    "clear_all_cache",
    "clear_downloads_cache",
]
