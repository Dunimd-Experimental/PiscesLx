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

import sys
import argparse
from pathlib import Path
from utils.log import ERROR
from utils.ul import display_all_versions, display_specific_version

def show_changelog(args=None):
    """
    Display changelog information for all versions or specific version.
    
    Args:
        args: Command line arguments (optional)
    """
    try:
        # Get project root directory
        project_root = Path(__file__).parent.parent
        
        # Parse command line arguments if provided
        if args and hasattr(args, 'all') and args.all:
            # Show all versions
            display_all_versions(project_root)
        elif args and hasattr(args, 'version') and args.version:
            # Show specific version
            display_specific_version(project_root, args.version)
        else:
            # Default: show all versions
            display_all_versions(project_root)
        
    except Exception as e:
        ERROR(f"Failed to display changelog information: {e}")
        sys.exit(1)


def parse_changelog_args():
    """
    Parse command line arguments for changelog command.
    """
    parser = argparse.ArgumentParser(
        description='Display changelog information for all or specific versions',
        prog='python manage.py changelog'
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--all', '-a',
        action='store_true',
        help='Show all available versions and their changelogs (default behavior)'
    )
    group.add_argument(
        '--version', '-v',
        type=str,
        metavar='VERSION',
        help='Show changelog for a specific version (e.g., 1.0.0150)'
    )
    
    return parser