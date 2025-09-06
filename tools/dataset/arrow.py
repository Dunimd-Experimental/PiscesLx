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

import os
import gc
import json
import glob
import pyarrow as pa
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

def arrow(progress_cb=None):
    """Advanced Arrow format datasets to JSON conversion with intelligent processing and error recovery.

    Args:
        progress_cb (callable | None): Optional callback function in the form of progress_cb(level:str, message:str).
            Suggested values for level: "debug", "info", "success", "error", "warning".
    """
    def emit(level: str, message: str, **kwargs):
        """Enhanced emission with context and error details."""
        # Add timestamp and context
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        enriched_message = f"[{timestamp}] {message}"
        
        # Output through callback with enhanced context
        if callable(progress_cb):
            try:
                progress_cb(level, enriched_message)
                return
            except Exception as callback_error:
                # Log callback errors separately
                try:
                    print(f"[callback_error] {str(callback_error)}")
                except:
                    pass
        
        # Fallback mechanism with error handling
        try:
            print(f"[{level}] {enriched_message}")
        except Exception as print_error:
            # Final fallback - write to stderr if possible
            import sys
            try:
                sys.stderr.write(f"EMERGENCY: {str(print_error)}\n")
            except:
                pass
    
    def find_arrow_datasets(root_dir):
        """Advanced Arrow dataset discovery with comprehensive validation and metadata extraction.

        Args:
            root_dir (str): The root directory to start the search from.

        Returns:
            list: A list of validated paths to directories containing Arrow datasets.
        """
        if not os.path.exists(root_dir):
            emit("error", f"Root directory does not exist: {root_dir}")
            return []
            
        datasets = []
        validation_cache = {}
        
        try:
            for root, dirs, files in os.walk(root_dir):
                # Skip hidden directories and common cache directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('__')]
                
                # Enhanced detection with multiple indicators
                indicators = [
                    "dataset_info.json",
                    "state.json", 
                    "dataset_dict.json",
                    "features.json",
                    "dataset.arrow"
                ]
                
                # Quick validation cache
                cache_key = root
                if cache_key in validation_cache:
                    if validation_cache[cache_key]:
                        datasets.append(root)
                    continue
                
                # Check for dataset indicators
                found_indicators = [ind for ind in indicators if os.path.exists(os.path.join(root, ind))]
                
                if found_indicators:
                    # Enhanced validation
                    try:
                        # Check for actual Arrow files
                        arrow_files = [f for f in files if f.endswith('.arrow') or f.endswith('.parquet')]
                        
                        # Validate dataset integrity
                        is_valid = False
                        for indicator in found_indicators:
                            indicator_path = os.path.join(root, indicator)
                            try:
                                if os.path.getsize(indicator_path) > 0:  # Non-empty indicator file
                                    is_valid = True
                                    break
                            except (OSError, IOError):
                                continue
                        
                        if is_valid:
                            # Additional metadata collection
                            dataset_info = {
                                'path': root,
                                'indicators': found_indicators,
                                'arrow_files': len(arrow_files),
                                'total_files': len(files),
                                'subdirs': len(dirs)
                            }
                            
                            validation_cache[cache_key] = True
                            datasets.append(root)
                            emit("debug", f"Valid dataset found: {root} ({dataset_info})")
                        else:
                            validation_cache[cache_key] = False
                            
                    except Exception as validation_error:
                        emit("warning", f"Validation error for {root}: {str(validation_error)}")
                        validation_cache[cache_key] = False
                        
        except Exception as discovery_error:
            emit("error", f"Dataset discovery failed: {str(discovery_error)}")
            
        # Remove duplicates and sort by modification time (newest first)
        try:
            datasets = list(set(datasets))
            datasets.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        except Exception:
            pass  # Fallback to unsorted
            
        emit("info", f"Found {len(datasets)} validated Arrow datasets")
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
        
        # Advanced processing with memory optimization and error recovery
        processing_stats = {
            'total_datasets': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_records': 0,
            'errors': []
        }
        
        def process_dataset_with_recovery(dataset_obj, dataset_name="dataset"):
            """Process individual dataset with comprehensive error handling."""
            try:
                if len(dataset_obj) == 0:
                    emit("warning", f"Skipping empty dataset: {dataset_name}")
                    processing_stats['failed_conversions'] += 1
                    return
                
                total_records = len(dataset_obj)
                processing_stats['total_records'] += total_records
                
                # Intelligent batch sizing based on available memory
                import psutil
                available_memory = psutil.virtual_memory().available
                base_batch_size = min(5000, max(1000, total_records // 20))
                
                # Adjust batch size based on memory pressure
                memory_factor = max(0.1, min(1.0, available_memory / (1024**3)))  # GB-based scaling
                batch_size = max(100, int(base_batch_size * memory_factor))
                
                emit("info", f"Processing {dataset_name}: {total_records} records, batch size: {batch_size}, memory: {available_memory/1024**3:.1f}GB")
                
                output_file = get_next_filename(current_dir, dataset_name)
                
                # Safe file handling with backup
                backup_file = output_file + ".backup"
                if os.path.exists(output_file):
                    try:
                        import shutil
                        shutil.copy2(output_file, backup_file)
                        os.remove(output_file)
                    except Exception as backup_error:
                        emit("warning", f"Backup failed: {backup_error}")
                
                processed = 0
                first_batch = True
                conversion_errors = []
                
                # Enhanced data processing with type safety
                def safe_encode_value(value):
                    """Safely encode various data types."""
                    if value is None:
                        return None
                    elif isinstance(value, str):
                        return value.encode('utf-8', 'ignore').decode('utf-8')
                    elif isinstance(value, (int, float, bool)):
                        return value
                    elif isinstance(value, (list, dict)):
                        try:
                            return json.loads(json.dumps(value, ensure_ascii=False))
                        except:
                            return str(value)
                    else:
                        return str(value)
                
                while processed < total_records:
                    end_idx = min(processed + batch_size, total_records)
                    
                    try:
                        batch = dataset_obj.select(range(processed, end_idx))
                        
                        # Enhanced record processing
                        records = []
                        for item in batch:
                            cleaned_item = {}
                            for key, value in item.items():
                                cleaned_item[key] = safe_encode_value(value)
                            records.append(cleaned_item)
                        
                        # Atomic file writing with error recovery
                        temp_file = output_file + ".tmp"
                        mode = 'w' if first_batch else 'a'
                        
                        try:
                            with open(temp_file, mode, encoding='utf-8') as f:
                                for record in records:
                                    json.dump(record, f, ensure_ascii=False)
                                    f.write('\n')
                            
                            # Atomic move
                            if first_batch and os.path.exists(output_file):
                                os.remove(output_file)
                            os.rename(temp_file, output_file)
                            
                        except Exception as file_error:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                            raise file_error
                        
                        processed = end_idx
                        first_batch = False
                        
                        progress_percent = (processed / max(total_records, 1)) * 100
                        emit("debug", f"Progress: {processed}/{total_records} ({progress_percent:.1f}%)")
                        
                        # Aggressive memory cleanup
                        del records, batch
                        gc.collect()
                        
                        # Memory pressure check
                        if psutil.virtual_memory().percent > 85:
                            emit("warning", "High memory usage detected, reducing batch size")
                            batch_size = max(50, batch_size // 2)
                        
                    except Exception as batch_error:
                        conversion_errors.append(str(batch_error))
                        emit("error", f"Batch processing failed: {batch_error}")
                        
                        # Recovery attempt with smaller batch
                        if batch_size > 100:
                            batch_size = max(50, batch_size // 2)
                            emit("info", f"Retrying with reduced batch size: {batch_size}")
                            continue
                        else:
                            break
                
                if processed == total_records:
                    processing_stats['successful_conversions'] += 1
                    emit("success", f"Successfully converted {dataset_name}: {output_file}")
                    
                    # Cleanup backup if successful
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                        
                    return True
                else:
                    processing_stats['failed_conversions'] += 1
                    processing_stats['errors'].extend(conversion_errors)
                    
                    # Restore from backup on failure
                    if os.path.exists(backup_file):
                        try:
                            if os.path.exists(output_file):
                                os.remove(output_file)
                            os.rename(backup_file, output_file)
                            emit("info", "Restored from backup due to conversion failure")
                        except Exception as restore_error:
                            emit("error", f"Backup restoration failed: {restore_error}")
                    
                    return False
                    
            except Exception as processing_error:
                processing_stats['failed_conversions'] += 1
                processing_stats['errors'].append(str(processing_error))
                emit("error", f"Dataset processing failed: {processing_error}")
                return False
        
        # Process DatasetDict or single Dataset with enhanced handling
        if isinstance(dataset, DatasetDict):
            processing_stats['total_datasets'] = len(dataset.keys())
            
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                process_dataset_with_recovery(split_data, f"{split_name}")
                
        elif isinstance(dataset, Dataset):
            processing_stats['total_datasets'] = 1
            process_dataset_with_recovery(dataset, "dataset")
            
        # Final statistics and cleanup
        emit("info", f"Conversion completed. Stats: {processing_stats}")
        
        # Memory cleanup
        gc.collect()
        
        return processing_stats
