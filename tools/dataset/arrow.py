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

import os
import gc
import json
import glob
import pyarrow as pa
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

def arrow(progress_cb=None):
    """Convert Arrow format datasets to JSON format files.

    Args:
        progress_cb (callable | None): Optional callback function in the form of progress_cb(level:str, message:str).
            Suggested values for level: "debug", "info", "success", "error".
    """
    def emit(level: str, message: str):
        # Output only through the UI callback; if no callback is available, fall back to print.
        if callable(progress_cb):
            try:
                progress_cb(level, message)
                return
            except Exception:
                pass
        # Fallback mechanism
        try:
            print(f"[{level}] {message}")
        except Exception:
            pass
    
    def find_arrow_datasets(root_dir):
        """Recursively find all Arrow dataset directories.

        Args:
            root_dir (str): The root directory to start the search from.

        Returns:
            list: A list of paths to directories containing Arrow datasets.
        """
        datasets = []
        for root, dirs, files in os.walk(root_dir):
            dataset_info_path = os.path.join(root, "dataset_info.json")
            state_json_path = os.path.join(root, "state.json")
            dataset_dict_path = os.path.join(root, "dataset_dict.json")
            
            if any([os.path.exists(p) for p in [dataset_info_path, state_json_path, dataset_dict_path]]):
                datasets.append(root)
        return datasets

    def get_next_filename(base_dir, prefix="train"):
        """Get the next available filename in the format train*.json.

        Args:
            base_dir (str): The directory where the files are located.
            prefix (str, optional): The prefix of the filename. Defaults to "train".

        Returns:
            str: The next available filename.
        """
        existing = glob.glob(os.path.join(base_dir, f"{prefix}*.json"))
        if not existing:
            return os.path.join(base_dir, f"{prefix}1.json")
        
        numbers = []
        for f in existing:
            basename = os.path.basename(f)
            if basename.startswith(prefix) and basename.endswith('.json'):
                try:
                    num = int(basename[len(prefix):-5])
                    numbers.append(num)
                except ValueError:
                    continue
        
        next_num = max(numbers) + 1 if numbers else 1
        return os.path.join(base_dir, f"{prefix}{next_num}.json")

    # Start of the main logic
    current_dir = os.getcwd()
    processed_count = 0
    skipped_count = 0

    # Find all Arrow dataset directories
    dataset_dirs = find_arrow_datasets(current_dir)

    if not dataset_dirs:
        emit("error", "No Arrow datasets found.")
    else:
        emit("info", f"Found {len(dataset_dirs)} dataset directories.")

    for dataset_dir in dataset_dirs:
        emit("info", f"Processing: {dataset_dir}")
        
        dataset = None
        
        # Try load_from_disk first
        try:
            dataset = load_from_disk(dataset_dir)
            emit("success", "Successfully loaded using load_from_disk.")
        except Exception as e:
            emit("error", f"load_from_disk failed: {e}")
        
        # If load_from_disk fails, try load_dataset
        if dataset is None:
            try:
                dataset = load_dataset(dataset_dir)
                emit("success", "Successfully loaded using load_dataset.")
            except Exception as e:
                emit("error", f"load_dataset failed: {e}")
                continue
        
        if dataset is None:
            emit("error", "Unable to load dataset, skipping.")
            continue
        
        # Process DatasetDict or single Dataset
        if isinstance(dataset, DatasetDict):
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                
                if len(split_data) == 0:
                    emit("error", f"Skipping empty dataset: {split_name}")
                    continue
                
                # Check the size of the dataset
                total_records = len(split_data)
                
                # Process in small batches regardless of size to prevent memory overflow
                batch_size = min(5000, max(1000, total_records // 10))
                
                emit("info", f"Dataset size: {total_records} records, batch size: {batch_size}")
                
                output_file = get_next_filename(current_dir, "train")
                
                # Delete the existing file first
                if os.path.exists(output_file):
                    os.remove(output_file)
                
                processed = 0
                first_batch = True
                
                while processed < total_records:
                    end_idx = min(processed + batch_size, total_records)
                    
                    try:
                        batch = split_data.select(range(processed, end_idx))
                        
                        # Process directly without converting to pandas to reduce memory usage
                        records = []
                        for item in batch:
                            # Clean encoding issues
                            cleaned_item = {}
                            for key, value in item.items():
                                if isinstance(value, str):
                                    cleaned_item[key] = value.encode('utf-8', 'ignore').decode('utf-8')
                                else:
                                    cleaned_item[key] = value
                            records.append(cleaned_item)
                        
                        # Write directly to JSON to avoid pandas memory overhead
                        mode = 'w' if first_batch else 'a'
                        with open(output_file, mode, encoding='utf-8') as f:
                            for record in records:
                                json.dump(record, f, ensure_ascii=False)
                                f.write('\n')
                        
                        processed = end_idx
                        first_batch = False
                        
                        emit("debug", f"Processed: {processed}/{total_records} ({processed/total_records*100:.1f}%)")
                        
                        # Clean up memory immediately
                        del records, batch
                        gc.collect()
                        
                    except Exception as e:
                        emit("error", f"Error processing batch: {e}")
                        break
                
                processed_count += 1
                emit("success", f"Saved successfully: {output_file}")
                
        elif isinstance(dataset, Dataset):
            if len(dataset) == 0:
                emit("error", "Skipping empty dataset.")
                continue
            
            total_records = len(dataset)
            
            # Process in small batches regardless of size
            batch_size = min(5000, max(1000, total_records // 10))
            
            emit("info", f"Dataset size: {total_records} records, batch size: {batch_size}")
            
            output_file = get_next_filename(current_dir, "train")
            
            # Delete the existing file first
            if os.path.exists(output_file):
                os.remove(output_file)
            
            processed = 0
            first_batch = True
            
            while processed < total_records:
                end_idx = min(processed + batch_size, total_records)
                
                try:
                    batch = dataset.select(range(processed, end_idx))
                    
                    # Process directly without converting to pandas
                    records = []
                    for item in batch:
                        # Clean encoding issues
                        cleaned_item = {}
                        for key, value in item.items():
                            if isinstance(value, str):
                                cleaned_item[key] = value.encode('utf-8', 'ignore').decode('utf-8')
                            else:
                                cleaned_item[key] = value
                        records.append(cleaned_item)
                    
                    # Write directly to JSON
                    mode = 'w' if first_batch else 'a'
                    with open(output_file, mode, encoding='utf-8') as f:
                        for record in records:
                            json.dump(record, f, ensure_ascii=False)
                            f.write('\n')
                    
                    processed = end_idx
                    first_batch = False
                    
                    emit("debug", f"Processed: {processed}/{total_records} ({processed/total_records*100:.1f}%)")
                    
                    # Clean up memory immediately
                    del records, batch
                    gc.collect()
                    
                except Exception as e:
                    emit("error", f"Error processing batch: {e}")
                    break
            
            processed_count += 1
            emit("success", f"Saved successfully: {output_file}")
