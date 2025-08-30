#!/usr/bin/env python3

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

import re
from pathlib import Path
from typing import Optional, List

def get_current_version(project_root: Path) -> str:
    """Get current version from configs/version.py"""
    version_file = project_root / "configs" / "version.py"
    
    if not version_file.exists():
        return "Unknown"
    
    try:
        content = version_file.read_text(encoding='utf-8')
        # Extract VERSION using regex
        version_match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
        if version_match:
            return version_match.group(1)
    except Exception:
        pass
    
    return "Unknown"

def version_to_ul_filename(version: str) -> str:
    """Convert version format to UL filename (e.g., '1.0.0150' -> '100150.UL')"""
    # Remove dots and convert to UL filename
    version_clean = version.replace('.', '')
    return f"{version_clean}.UL"

def parse_ul_file(ul_file_path: Path) -> List[str]:
    """Parse UL file and extract changelog entries"""
    if not ul_file_path.exists():
        return []
    
    try:
        content = ul_file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        changelog_entries = []
        in_changelog = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('!'):
                continue
            
            # Start of changelog section
            if line.startswith('###'):
                in_changelog = True
                continue
            
            # Changelog entry
            if in_changelog and line.startswith('-'):
                # Remove the leading '-' and clean up
                entry = line[1:].strip()
                if entry:
                    changelog_entries.append(entry)
        
        return changelog_entries
    
    except Exception:
        return []

def display_version_changelog(project_root: Path, current_version: str):
    """Display version information and changelog with specified format"""
    print("")
    print("✅\tPisces L1 - Arctic Architecture")
    print(f"✅\tVersion: {current_version}")
    print("✅\tProject: PiscesLx Series by Dunimd Project Group")
    print("")
    print(f"🟧\tChangelog for Version {current_version}")
    
    # Get UL file and parse changelog
    ul_filename = version_to_ul_filename(current_version)
    ul_file_path = project_root / "UL" / ul_filename
    
    changelog_entries = parse_ul_file(ul_file_path)
    
    if changelog_entries:
        for entry in changelog_entries:
            print(f"  🔸 {entry}")
    else:
        print("  🔸 No changelog available for this version")

def get_all_versions(project_root: Path) -> List[str]:
    """Get all available versions from UL directory"""
    ul_dir = project_root / "UL"
    
    if not ul_dir.exists():
        return []
    
    versions = []
    
    try:
        for ul_file in ul_dir.glob("*.UL"):
            # Convert UL filename back to version format
            filename = ul_file.stem  # Remove .UL extension
            
            # Parse filename (e.g., "100150" -> "1.0.0150")
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
        
    except Exception:
        pass
    
    return versions


def display_all_versions(project_root: Path):
    """Display all available versions and their changelogs"""
    versions = get_all_versions(project_root)
    
    if not versions:
        print("\n❌\tNo version history found")
        return
    
    print("\n✅\tPisces L1 - Arctic Architecture")
    print("✅\tAll Version History")
    print("✅\tProject: PiscesLx Series by Dunimd Project Group")
    print()
    
    for version in versions:
        print(f"🟧\tVersion {version}")
        
        # Get UL file and parse changelog
        ul_filename = version_to_ul_filename(version)
        ul_file_path = project_root / "UL" / ul_filename
        
        changelog_entries = parse_ul_file(ul_file_path)
        
        if changelog_entries:
            for entry in changelog_entries:
                print(f"  🔸 {entry}")
        else:
            print("  🔸 No changelog available")
        
        print()  # Empty line between versions


def display_specific_version(project_root: Path, target_version: str):
    """Display changelog for a specific version"""
    # Validate version format
    if not target_version or not target_version.replace('.', '').isdigit():
        print(f"\n❌\tInvalid version format: {target_version}")
        print("\n🟧\tExpected format: x.x.xxxx (e.g., 1.0.0150)")
        return
    
    # Check if version exists
    ul_filename = version_to_ul_filename(target_version)
    ul_file_path = project_root / "UL" / ul_filename
    
    if not ul_file_path.exists():
        print(f"\n❌\tVersion {target_version} not found")
        
        # Show available versions
        available_versions = get_all_versions(project_root)
        if available_versions:
            print("\n🟧\tAvailable versions:")
            for version in available_versions[:5]:  # Show first 5
                print(f"  🔸 {version}")
            if len(available_versions) > 5:
                print(f"  🔸 ... and {len(available_versions) - 5} more versions")
        return
    
    # Display the specific version
    display_version_changelog(project_root, target_version)


def display_update_log(project_root: Path):
    """Display update log after successful update"""
    current_version = get_current_version(project_root)
    display_version_changelog(project_root, current_version)