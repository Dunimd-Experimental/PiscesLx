#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei
#
# This file is part of Pisces L1.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).
# You may not use this file except in compliance with the License.
# Commercial use is strictly prohibited.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import pyarrow as pa
from utils.log import RIGHT, DEBUG, ERROR
from datasets import Dataset, load_from_disk

def arrow(args):
    """
    Convert between JSON files and Arrow format datasets.

    Args:
        args (object): An object containing command-line arguments with the following attributes:
            - json_dir (str): Directory path containing JSON files.
            - arrow_out (str): Output path for the Arrow format dataset.
            - arrow_in (str): Input path for the Arrow format dataset.
            - json_out (str): Output path for the JSON file.

    Returns:
        None: The function saves the converted data to the specified path and returns None.
    """
    if args.json_dir and args.arrow_out:
        # Get all JSON files in the specified directory
        json_files = [os.path.join(args.json_dir, f) for f in os.listdir(args.json_dir) if f.endswith('.json')]
        if not json_files:
            ERROR(f"No .json files found in {args.json_dir}")
            return
        all_data = []
        # Load data from each JSON file
        for jf in json_files:
            with open(jf, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        all_data.append(json.loads(line))
                    except Exception as e:
                        ERROR(f"Error parsing {jf}: {e}")
        if not all_data:
            ERROR("No data loaded from json files.")
            return
        # Convert data list to Dataset and save to disk
        ds = Dataset.from_list(all_data)
        ds.save_to_disk(args.arrow_out)
        RIGHT(f"Saved {len(ds)} samples to {args.arrow_out}")
        return
    elif args.arrow_in and args.json_out:
        # Check if the input Arrow file exists
        if not os.path.exists(args.arrow_in):
            ERROR(f"Arrow file not found: {args.arrow_in}")
            return
        # Load the Arrow dataset from disk
        ds = load_from_disk(args.arrow_in)
        with open(args.json_out, 'w', encoding='utf-8') as f:
            for item in ds:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        RIGHT(f"Saved {len(ds)} samples to {args.json_out}")
        return
    else:
        ERROR("Please specify either --json_dir + --arrow_out or --arrow_in + --json_out")
        DEBUG("For example:")
        print("\tpython manage.py arrow --json_dir ./jsons --arrow_out ./out.arrow")
        print("\tpython manage.py arrow --arrow_in ./in.arrow --json_out ./out.json")