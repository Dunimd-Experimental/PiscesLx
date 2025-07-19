#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import argparse
from pathlib import Path


ROOT = os.path.abspath(os.path.dirname(__file__))

COMMANDS = [
    'setup',      # Environment setup
    'train',      # Train model
    'infer',      # Inference
    'check',      # Check GPU/deps
    'monitor',    # System monitor
    'download',   # Download datasets
    'arrow',      # Arrow/JSON conversion
    'help',       # Show help for commands
]

def main():
    parser = argparse.ArgumentParser(description="Pisces L1 Management Tool (manage.py)")
    parser.add_argument('command', nargs='?', choices=COMMANDS, help="Command to execute")
    parser.add_argument('--ckpt', default='', help='Checkpoint file (for infer)')
    parser.add_argument('--prompt', default='Hello, please introduce yourself', help='Prompt (for infer)')
    parser.add_argument('--image', default='', help='Image path (for infer)')
    parser.add_argument('--max_samples', type=int, default=50000, help='Max samples per dataset (for download)')
    parser.add_argument('--json_dir', default='', help='[arrow] Directory containing .json files to merge into one .arrow')
    parser.add_argument('--arrow_out', default='', help='[arrow] Output .arrow file path')
    parser.add_argument('--arrow_in', default='', help='[arrow] Input .arrow file path to convert to .json')
    parser.add_argument('--json_out', default='', help='[arrow] Output .json file path (single file)')
    args, extra = parser.parse_known_args()
    if args.command is None or args.command == 'help':
        from tools.help import help
        help()
    elif args.command == 'train':
        from tools.train import add_train_args, train
        parser = add_train_args(parser)
        args = parser.parse_args()
        train(args)
    elif args.command == 'infer':
        from tools.infer import infer
        infer(args)
    elif args.command == 'check':
        from tools.check import check
        check(args)
    elif args.command == 'monitor':
        from tools.monitor import monitor
        monitor()
    elif args.command == 'download':
        from data.download import download_datasets
        download_datasets(args.max_samples)
    elif args.command == 'arrow':
        from tools.arrow import arrow
        arrow(args)
    elif args.command == 'setup':
        from tools.setup import setup
        setup(args)
    elif args.command == 'help':
        from tools.help import help
        help()
    else:
        print(f"❌\tUnknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 