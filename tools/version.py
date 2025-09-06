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
from utils.log import ERROR
from utils.ul import display_version_changelog, get_current_version

def show_version():
    """
    Display current version information and corresponding update log.
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent
        
        # Get current version
        current_version = get_current_version(project_root)
        if not current_version:
            ERROR("Could not determine current version")
            sys.exit(1)
        
        # Display version information and changelog
        display_version_changelog(project_root, current_version)
        
    except Exception as e:
        ERROR(f"Failed to display version information: {e}")
        sys.exit(1)

def _display_version_changelog(project_root: Path, current_version: str):
    """
    Display the changelog for the current version.
    
    Args:
        project_root (Path): Project root directory path
        current_version (str): Current version string
    """
    ul_dir = project_root / "UL"
    
    if not ul_dir.exists():
        DEBUG("UL directory not found, skipping changelog display")
        return
    
    try:
        # Convert version to UL filename format (e.g., "1.0.0150" -> "100150.UL")
        version_parts = current_version.split('.')
        if len(version_parts) != 3:
            DEBUG(f"Invalid version format: {current_version}")
            return
        
        # Format: major + minor (2 digits) + patch (3 digits)
        try:
            major = int(version_parts[0])
            minor = int(version_parts[1])
            patch = int(version_parts[2])
            ul_filename = f"{major:01d}{minor:02d}{patch:03d}.UL"
        except ValueError:
            DEBUG(f"Could not convert version to UL filename: {current_version}")
            return
        
        ul_file_path = ul_dir / ul_filename
        
        if not ul_file_path.exists():
            DEBUG(f"\nNo changelog found for version {current_version}")
            return
        
        # Read and parse the update log
        with open(ul_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract changelog items (lines starting with '-')
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
                # End of changelog section
                break
        
        # Display changelog
        DEBUG(f"Changelog for Version {current_version}")
        
        if changelog_lines:
            for item in changelog_lines:
                # Clean up the changelog item and add emoji
                clean_item = item.lstrip('- ').strip()
                print(f"  🔸 {clean_item}")
        else:
            DEBUG("No changelog items found")

        
    except Exception as e:
        DEBUG(f"Error reading changelog: {e}")
        ERROR(f"\nFailed to read changelog for version {current_version}")