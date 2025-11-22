#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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
import logging
import logging.handlers

def _ensure_dir(path: str) -> None:
    """Ensure the parent directory of the given file path exists.

    Args:
        path (str): The file path for which the parent directory needs to be created if it doesn't exist.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass

def build_rotating_file_handler(
    file_path: str,
    rotate_when: str = "size",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Handler:
    """Build a rotating file handler based on the specified rotation type.

    Args:
        file_path (str): Path to the log file.
        rotate_when (str, optional): Rotation type, either "size" or "time". Defaults to "size".
        max_bytes (int, optional): Maximum size of the log file before rotation (in bytes). 
                                   Only used when rotate_when is "size". Defaults to 10MB.
        backup_count (int, optional): Number of backup log files to keep. Defaults to 5.

    Returns:
        logging.Handler: An instance of a rotating file handler.

    Raises:
        ValueError: If rotate_when is neither "size" nor "time".
    """
    _ensure_dir(file_path)
    if rotate_when == "time":
        return logging.handlers.TimedRotatingFileHandler(
            file_path, when="D", interval=1, backupCount=backup_count, encoding="utf-8"
        )
    elif rotate_when == "size":
        return logging.handlers.RotatingFileHandler(
            file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
    else:
        raise ValueError("rotate_when must be either 'size' or 'time'")
