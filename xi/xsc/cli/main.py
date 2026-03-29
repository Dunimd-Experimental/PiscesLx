#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Xi.
# The Xi project belongs to the Dunimd Team.
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

"""
Command-line interface for Xi Studio.

Usage:
    xis                     Start Xi Studio (backend + frontend)
    xis --port PORT         Specify backend port (default: 3140)
    xis --frontend-port PORT Specify frontend port (default: 3000)
    xis --help              Show this help message
"""

import sys
import argparse

from ..launcher import XiLauncher


def main():
    """
    Main entry point for Xi Studio CLI.
    
    Parses command line arguments and starts the launcher.
    """
    parser = argparse.ArgumentParser(
        prog="xis",
        description="Xi Studio - Flagship LLM Workstation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    xis                     Start Xi Studio with default ports
    xis --port 8080         Start with custom backend port
    xis --frontend-port 8081 Start with custom frontend port
    xis --port 8080 --frontend-port 8081 Start with both custom ports
        """
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=3140,
        help="Backend API port (default: 3140)"
    )
    
    parser.add_argument(
        "--frontend-port", "-f",
        type=int,
        default=3000,
        help="Frontend port (default: 3000)"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Xi Studio 1.0.0"
    )
    
    args = parser.parse_args()
    
    launcher = XiLauncher(
        xi_port=args.port,
        frontend_port=args.frontend_port
    )
    
    exit_code = launcher.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
