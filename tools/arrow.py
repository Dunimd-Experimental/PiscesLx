#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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

import os
import json
import pyarrow as pa
from utils.log import RIGHT, DEBUG, ERROR
from datasets import Dataset, load_from_disk

def arrow(args):
    """
    Convert between JSON files and Arrow format datasets.

    Args:
        args (object): An object containing command-line arguments
    """
    
    def convert_single_arrow_file(arrow_file, json_out):
        """Convert a single .arrow file to JSON."""
        try:
            # Open the arrow file using memory mapping
            with pa.memory_map(arrow_file, 'rb') as source:
                # Create a RecordBatchStreamReader to read the arrow data
                reader = pa.RecordBatchStreamReader(source)
                # Read all data into a table
                table = reader.read_all()
            
            # Convert the arrow table to a dataset
            dataset = Dataset.from_arrow(table)
            
            # Write each item in the dataset to the JSON output file
            with open(json_out, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            RIGHT(f"Successfully converted {len(dataset)} records to {json_out}")
            return True
            
        except Exception as e:
            ERROR(f"Failed to convert single Arrow file: {e}")
            return False
    
    def convert_arrow_directory(arrow_dir, json_out):
        """Convert an Arrow dataset directory to JSON."""
        try:
            # Load the dataset from the specified directory
            dataset = load_from_disk(arrow_dir)
            
            # Write each item in the dataset to the JSON output file
            with open(json_out, 'w', encoding='utf-8') as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            RIGHT(f"Successfully converted {len(dataset)} records to {json_out}")
            return True
            
        except Exception as e:
            ERROR(f"Failed to convert Arrow directory: {e}")
            return False
    
    # Convert JSON directory to Arrow format
    if args.json_dir and args.arrow_out:
        # Get all JSON files in the specified directory
        json_files = [os.path.join(args.json_dir, f) for f in os.listdir(args.json_dir) if f.endswith('.json')]
        if not json_files:
            ERROR(f"No .json files found in {args.json_dir}")
            return
        
        all_data = []
        # Read and parse each JSON file
        for jf in json_files:
            with open(jf, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_data.append(json.loads(line))
                        except Exception as e:
                            ERROR(f"Error parsing {jf}: {e}")
        
        if not all_data:
            ERROR("No data loaded from JSON files")
            return
        
        # Convert the list of data to a dataset
        ds = Dataset.from_list(all_data)
        # Save the dataset to disk in Arrow format
        ds.save_to_disk(args.arrow_out)
        RIGHT(f"Saved {len(ds)} samples to {args.arrow_out}")
        return
    
    # Convert Arrow file or directory to JSON
    elif args.arrow_in and args.json_out:
        if not os.path.exists(args.arrow_in):
            ERROR(f"Input file/directory does not exist: {args.arrow_in}")
            return
        
        if os.path.isfile(args.arrow_in) and args.arrow_in.endswith('.arrow'):
            # Process a single .arrow file
            convert_single_arrow_file(args.arrow_in, args.json_out)
        elif os.path.isdir(args.arrow_in):
            # Process an Arrow dataset directory
            convert_arrow_directory(args.arrow_in, args.json_out)
        else:
            ERROR(f"Unsupported input type: {args.arrow_in}")
        return
    
    else:
        ERROR("Please specify one of the following parameter combinations:")
        ERROR("  --json_dir directory --arrow_out output.arrow")
        ERROR("  --arrow_in file_or_directory --json_out output.json")
        DEBUG("Examples:")
        print("\tpython manage.py arrow --json_dir ./jsons --arrow_out ./dataset")
        print("\tpython manage.py arrow --arrow_in ./data_cache/Chinese1 --json_out ./chinese.json")
        print("\tpython manage.py arrow --arrow_in data-00000-of-00002.arrow --json_out ./output.json")