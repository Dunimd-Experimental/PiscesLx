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

import sys
from pathlib import Path
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager, PiscesLxCoreUL
logger = PiscesLxCoreLog("pisceslx.data.download")

def show_version():
    """
    Display the current version information and the corresponding update log.
    If an error occurs during the process, it will print an error message and exit the program.
    """
    try:
        # Obtain the project root directory by navigating up two levels from the current file.
        project_root = Path(__file__).parent.parent
        
        # Retrieve the current version of the project.
        current_version = PiscesLxCoreUL.get_current_version(project_root)
        if not current_version:
            logger.error("Could not determine current version")
            sys.exit(1)
        
        # Display the version information and changelog.
        PiscesLxCoreUL.display_version_changelog(project_root, current_version)
        
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
    # Construct the path to the UL directory.
    ul_dir = project_root / "UL"
    
    # Skip changelog display if the UL directory does not exist.
    if not ul_dir.exists():
        logger.debug("UL directory not found, skipping changelog display")
        return
    
    try:
        # Convert the version string to the UL filename format (e.g., "1.0.0150" -> "100150.UL").
        version_parts = current_version.split('.')
        if len(version_parts) != 3:
            logger.debug(f"Invalid version format: {current_version}")
            return
        
        # Format the version components into the UL filename: major + minor (2 digits) + patch (3 digits).
        try:
            major = int(version_parts[0])
            minor = int(version_parts[1])
            patch = int(version_parts[2])
            ul_filename = f"{major:01d}{minor:02d}{patch:03d}.UL"
        except ValueError:
            logger.debug(f"Could not convert version to UL filename: {current_version}")
            return
        
        # Construct the path to the UL file.
        ul_file_path = ul_dir / ul_filename
        
        # Skip changelog display if the UL file does not exist.
        if not ul_file_path.exists():
            logger.debug(f"\nNo changelog found for version {current_version}")
            return
        
        # Read the content of the UL file.
        with open(ul_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract changelog items that start with '-'.
        changelog_lines = []
        lines = content.split('\n')
        in_changelog = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('###'):
                in_changelog = True
                continue
            elif in_changelog and line.startswith('-'):
                changelog_lines.append(line)
            elif in_changelog and line and not line.startswith('-'):
                # End of the changelog section.
                break
        
        # Print the header of the changelog.
        logger.debug(f"Changelog for Version {current_version}")
        
        if changelog_lines:
            for item in changelog_lines:
                # Clean up the changelog item and add an emoji prefix.
                clean_item = item.lstrip('- ').strip()
                print(f"  🔸 {clean_item}")
        else:
            logger.debug("No changelog items found")
        
    except Exception as e:
        logger.debug(f"Error reading changelog: {e}")
        logger.error(f"\nFailed to read changelog for version {current_version}")