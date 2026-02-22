#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

import os
import time
from typing import Optional


class POPSSFileLock:
    def __init__(self, lock_path: str, timeout_s: float = 10.0, poll_interval_s: float = 0.05):
        self._lock_path = str(lock_path)
        self._timeout_s = float(timeout_s)
        self._poll_interval_s = float(poll_interval_s)
        self._fh = None

    def __enter__(self):
        start = time.time()
        os.makedirs(os.path.dirname(self._lock_path) or ".", exist_ok=True)
        self._fh = open(self._lock_path, "a+b")

        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if (time.time() - start) >= self._timeout_s:
                        raise TimeoutError(f"lock_timeout:{self._lock_path}")
                    time.sleep(self._poll_interval_s)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if (time.time() - start) >= self._timeout_s:
                        raise TimeoutError(f"lock_timeout:{self._lock_path}")
                    time.sleep(self._poll_interval_s)

        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fh is None:
                return False
            if os.name == "nt":
                import msvcrt

                try:
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                import fcntl

                try:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
        finally:
            try:
                if self._fh is not None:
                    self._fh.close()
            finally:
                self._fh = None
        return False

