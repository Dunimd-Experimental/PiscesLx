#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
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

from pathlib import Path
from typing import Optional
from utils.fs.core import PiscesLxCoreFS
from utils.log.core import PiscesLxCoreLog
from utils.cache.enhanced import PiscesLxCoreEnhancedCacheManager
from utils.config.manager import PiscesLxCoreConfigManagerFacade as _ConfigFacade

logger = PiscesLxCoreLog("PiscesLx.Utils.Cache.Facade")

class PiscesLxCoreCacheManagerFacade:
    """Lightweight facade exposing filesystem cache directories.
    
    - Provides a stable API for directory access used across tools and data modules
    - Keeps compatibility with pre-existing code expecting `.base_cache_dir` and `.get_cache_dir(subdir)`
    - Internally, resolves base cache dir via `utils.fs.core.PiscesLxCoreFS`
    - Optionally exposes the enhanced cache instance for advanced use
    """

    _instance: Optional["PiscesLxCoreCacheManagerFacade"] = None

    def __init__(self) -> None:
        self._fs = PiscesLxCoreFS()
        self._base_cache_dir: Optional[Path] = None
        self._enhanced_cache_mgr = PiscesLxCoreEnhancedCacheManager.get_instance()

    @property
    def base_cache_dir(self) -> Path:
        if self._base_cache_dir is None:
            base = self._fs.cache_dir()
            try:
                base.mkdir(parents=True, exist_ok=True)
            except Exception:
                # Best-effort; downstream will attempt mkdir again if needed
                pass
            self._base_cache_dir = base
        return self._base_cache_dir

    def get_cache_dir(self, subdir: str) -> Path:
        p = self.base_cache_dir / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Optional: expose the enhanced cache for value caching if needed
    def get_default_cache(self):
        return self._enhanced_cache_mgr.get_default_cache()

    @classmethod
    def get_instance(cls) -> "PiscesLxCoreCacheManagerFacade":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_cache_manager() -> PiscesLxCoreCacheManagerFacade:
    """Public accessor expected by multiple modules (tools/data).
    Returns a singleton facade with directory helpers.
    """
    return PiscesLxCoreCacheManagerFacade.get_instance()

# Re-export for backward compatibility with tools/__init__.py import path
PiscesLxCoreConfigManagerFacade = _ConfigFacade

__all__ = [
    "get_cache_manager",
    "PiscesLxCoreCacheManagerFacade",
    "PiscesLxCoreConfigManagerFacade",
]
