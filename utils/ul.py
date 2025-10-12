#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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

import re
from pathlib import Path
from typing import Optional, List
from utils.log.core import PiscesLxCoreLog

# 模块级统一日志
logger = PiscesLxCoreLog("PiscesLx.Utils.UL")

def get_current_version(project_root: Path) -> str:
    """
    Retrieves the current version from the configs/version.py file.

    Args:
        project_root (Path): The root path of the project.

    Returns:
        str: The current version if found; otherwise, returns "Unknown".
    """
    version_file = project_root / "configs" / "version.py"
    
    # Return "Unknown" if the version file does not exist
    if not version_file.exists():
        return "Unknown"
    
    try:
        content = version_file.read_text(encoding='utf-8')
        # Use regular expression to extract the VERSION value
        version_match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
        if version_match:
            return version_match.group(1)
    except Exception as log_e:
        logger.debug("UL_VERSION_PARSE_FAILED", error=str(log_e))
    
    return "Unknown"

def version_to_ul_filename(version: str) -> str:
    """
    Converts a version string to a UL filename.

    Args:
        version (str): The version string (e.g., '0.0.0150').

    Returns:
        str: The corresponding UL filename (e.g., '000150.UL').
    """
    # Remove dots from the version string
    version_clean = version.replace('.', '')
    return f"{version_clean}.UL"

def parse_ul_file(ul_file_path: Path) -> List[str]:
    """
    Parses a UL file and extracts changelog entries.

    Args:
        ul_file_path (Path): The path to the UL file.

    Returns:
        List[str]: A list of changelog entries. Returns an empty list if the file does not exist or an error occurs.
    """
    # Return an empty list if the file does not exist
    if not ul_file_path.exists():
        return []
    
    try:
        content = ul_file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        changelog_entries = []
        in_changelog = False
        
        for line in lines:
            # Remove leading and trailing whitespace
            line = line.strip()
            
            # Skip empty lines and comment lines
            if not line or line.startswith('!'):
                continue
            
            # Mark the start of the changelog section
            if line.startswith('###'):
                in_changelog = True
                continue
            
            # Extract changelog entries
            if in_changelog and line.startswith('-'):
                # Remove the leading '-' and clean up the entry
                entry = line[1:].strip()
                if entry:
                    changelog_entries.append(entry)
        
        return changelog_entries
    
    except Exception as log_e:
        logger.debug("UL_PARSE_FILE_FAILED", error=str(log_e))
        return []

def display_version_changelog(project_root: Path, current_version: str):
    """
    Displays version information and changelog in a specified format.

    Args:
        project_root (Path): The root path of the project.
        current_version (str): The current version to display.
    """
    print("")
    print("🟢\tPiscesL1 - Arctic Architecture")
    print(f"🟢\tVersion: {current_version}")
    print("🟢\tProject: PiscesLx Series by Dunimd Project Group")
    print("")
    print(f"🔵\tChangelog for Version {current_version}")
    
    # Generate UL filename and get file path
    ul_filename = version_to_ul_filename(current_version)
    ul_file_path = project_root / "UL" / ul_filename
    
    changelog_entries = parse_ul_file(ul_file_path)
    
    if changelog_entries:
        for entry in changelog_entries:
            print(f"  🔸 {entry}")
    else:
        print("  🔸 No changelog available for this version")

def get_all_versions(project_root: Path) -> List[str]:
    """
    Retrieves all available versions from the UL directory.

    Args:
        project_root (Path): The root path of the project.

    Returns:
        List[str]: A list of version strings sorted in descending order. Returns an empty list if the UL directory does not exist or an error occurs.
    """
    ul_dir = project_root / "UL"
    
    # Return an empty list if the UL directory does not exist
    if not ul_dir.exists():
        return []
    
    versions = []
    
    try:
        for ul_file in ul_dir.glob("*.UL"):
            # Get the filename without the extension
            filename = ul_file.stem
            
            # Parse the filename to extract version components
            if len(filename) >= 6:
                try:
                    major = int(filename[0])
                    minor = int(filename[1:3])
                    patch = int(filename[3:])
                    version = f"{major}.{minor}.{patch:04d}"
                    versions.append(version)
                except ValueError:
                    continue
        
        # Sort versions in descending order (newest first)
        versions.sort(key=lambda x: [int(part) for part in x.split('.')], reverse=True)
        
    except Exception as log_e:
        logger.debug("UL_VERSION_SCAN_FAILED", error=str(log_e))
    
    return versions

def display_all_versions(project_root: Path):
    """
    Displays all available versions and their corresponding changelogs.

    Args:
        project_root (Path): The root path of the project.
    """
    versions = get_all_versions(project_root)
    
    # Print a message if no version history is found
    if not versions:
        print("🔴\tNo version history found")
        return
    
    print("")
    print("🟢\tPiscesL1 - Arctic Architecture")
    print("🟢\tAll Version History")
    print("🟢\tProject: PiscesLx Series by Dunimd Project Group")
    print("")
    
    for version in versions:
        print(f"🔵\tVersion {version}")
        
        # Generate UL filename and get file path
        ul_filename = version_to_ul_filename(version)
        ul_file_path = project_root / "UL" / ul_filename
        
        changelog_entries = parse_ul_file(ul_file_path)
        
        if changelog_entries:
            for entry in changelog_entries:
                print(f"  🔸 {entry}")
        else:
            print("  🔸 No changelog available")
        
        print()  # Add an empty line between versions

def display_specific_version(project_root: Path, target_version: str):
    """
    Displays the changelog for a specific version.

    Args:
        project_root (Path): The root path of the project.
        target_version (str): The target version to display.
    """
    # Validate the version format
    if not target_version or not target_version.replace('.', '').isdigit():
        print(f"\n🔴\tInvalid version format: {target_version}")
        print("\n🔵\tExpected format: x.x.xxxx (e.g., 1.0.0150)")
        return
    
    # Generate UL filename and get file path
    ul_filename = version_to_ul_filename(target_version)
    ul_file_path = project_root / "UL" / ul_filename
    
    # Check if the version file exists
    if not ul_file_path.exists():
        print(f"\n🔴\tVersion {target_version} not found")
        
        # Display available versions
        available_versions = get_all_versions(project_root)
        if available_versions:
            print("\n🔵\tAvailable versions:")
            for version in available_versions[:5]:  # Show the first 5 versions
                print(f"  🔸 {version}")
            if len(available_versions) > 5:
                print(f"  🔸 ... and {len(available_versions) - 5} more versions")
        return
    
    # Display the changelog for the specific version
    display_version_changelog(project_root, target_version)

def display_update_log(project_root: Path):
    """
    Displays the update log after a successful update.

    Args:
        project_root (Path): The root path of the project.
    """
    current_version = get_current_version(project_root)
    display_version_changelog(project_root, current_version)

class PiscesLxCoreUL:
    """UL utilities wrapped in an OOP style, providing access to module functions."""
 
    @staticmethod
    def get_current_version(project_root: Path) -> str:
        """
        Retrieves the current version from the configs/version.py file.

        Args:
            project_root (Path): The root path of the project.

        Returns:
            str: The current version if found; otherwise, returns "Unknown".
        """
        return get_current_version(project_root)
 
    @staticmethod
    def version_to_ul_filename(version: str) -> str:
        """
        Converts a version string to a UL filename.

        Args:
            version (str): The version string (e.g., '0.0.0150').

        Returns:
            str: The corresponding UL filename (e.g., '000150.UL').
        """
        return version_to_ul_filename(version)
 
    @staticmethod
    def parse_ul_file(ul_file_path: Path) -> List[str]:
        """
        Parses a UL file and extracts changelog entries.

        Args:
            ul_file_path (Path): The path to the UL file.

        Returns:
            List[str]: A list of changelog entries. Returns an empty list if the file does not exist or an error occurs.
        """
        return parse_ul_file(ul_file_path)
 
    @classmethod
    def display_version_changelog(cls, project_root: Path, current_version: str) -> None:
        """
        Displays version information and changelog in a specified format.

        Args:
            project_root (Path): The root path of the project.
            current_version (str): The current version to display.
        """
        return display_version_changelog(project_root, current_version)
 
    @classmethod
    def get_all_versions(cls, project_root: Path) -> List[str]:
        """
        Retrieves all available versions from the UL directory.

        Args:
            project_root (Path): The root path of the project.

        Returns:
            List[str]: A list of version strings sorted in descending order. Returns an empty list if the UL directory does not exist or an error occurs.
        """
        return get_all_versions(project_root)
 
    @classmethod
    def display_all_versions(cls, project_root: Path) -> None:
        """
        Displays all available versions and their corresponding changelogs.

        Args:
            project_root (Path): The root path of the project.
        """
        return display_all_versions(project_root)
 
    @classmethod
    def display_specific_version(cls, project_root: Path, target_version: str) -> None:
        """
        Displays the changelog for a specific version.

        Args:
            project_root (Path): The root path of the project.
            target_version (str): The target version to display.
        """
        return display_specific_version(project_root, target_version)
 
    @classmethod
    def display_update_log(cls, project_root: Path) -> None:
        """
        Displays the update log after a successful update.

        Args:
            project_root (Path): The root path of the project.
        """
        return display_update_log(project_root)