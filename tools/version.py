#!/usr/bin/env/python3

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

import sys
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
logger = PiscesLxCoreLog("pisceslx.data.download")

def show_version():
    """
    Display the current version information and the corresponding update log.
    If an error occurs during the process, it will print an error message and exit the program.
    """
    try:
        # Obtain the project root directory by navigating up two levels from the current file.
        project_root = Path(__file__).parent.parent
        
        # Get current version from configs/version.py
        from configs.version import CVERSION as current_version
        
        # Display basic version info (removed UL changelog functionality)
        print("")
        print("🟢\tPiscesL1 - Arctic Architecture")
        print(f"🟢\tVersion: {current_version}")
        print("🟢\tProject: PiscesLx Series by Dunimd Project Group")
        print("")
        
    except Exception as e:
        logger.error(f"Failed to display version information: {e}")
        sys.exit(1)

def _display_version_changelog(project_root: Path, current_version: str):
    """
    Display the changelog for the specified version.

    Args:
        project_root (Path): Path to the project root directory.
        current_version (str): Current version string in the format of "major.minor.patch".
    """
    # UL functionality removed - changelog display disabled
    logger.debug("UL changelog functionality removed")
    return
