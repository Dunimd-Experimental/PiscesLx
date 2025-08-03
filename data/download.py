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
import gc
import glob
import shutil
import requests
import math
import multiprocessing
from tqdm import tqdm
from .clean import DatasetCleaner
from datasets import load_from_disk
from modelscope.msdatasets import MsDataset

# Get the root directory of the current script
ROOT = os.path.dirname(__file__)
# Define the data cache directory path
DATA = os.path.join(ROOT, "..", "data_cache")
# Create the data cache directory if it doesn't exist
os.makedirs(DATA, exist_ok=True)

def save(ds, name):
    """
    Save dataset to local cache.

    Args:
        ds: The dataset to be saved.
        name (str): The name used to save the dataset.

    Returns:
        bool: True if the dataset is saved successfully, False otherwise.
    """
    try:
        # Generate the save path for the dataset
        save_path = os.path.join(DATA, name)
        print(f"✅\tSaving {name} to {save_path}...")
        # Save the dataset to the specified path
        ds.save_to_disk(save_path)
        print(f"✅\t{name} saved to {save_path}")
        return True
    except Exception as e:
        print(f"❌\tFailed to save {name}: {e}")
        return False

def download_datasets(max_samples_per_dataset=50000, post_download_clean=True):
    """
    Download all datasets with size control.

    Args:
        max_samples_per_dataset (int, optional): The maximum number of samples per dataset. Defaults to 50000.
        post_download_clean (bool, optional): Whether to perform post-download cleaning. Defaults to True.
    """
    print("✅\tStarting ModelScope dataset download...")
    
    # Core datasets for Pisces L1 training
    datasets = [
        # Chinese
        ("baicai003/Llama3-Chinese-dataset", "Chinese1", "Chinese1"),
        ("liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT", "Chinese2", "Chinese2"),

        #Math
        ("swift/MetaMathQA", "Math1", "Math1"),
        ("AI-MO/NuminaMath-CoT", "Math2", "Math2"),
        ("AI-ModelScope/NuminaMath-CoT", "Math3", "Math3"),
        ("xpengx/EleutherAI-proof-pile-2", "Math4", "Math4"),

        #Code
        ("HuggingFaceH4/CodeAlpaca_20K", "Code1", "Code1"),
        ("jablonkagroup/codeparrot_github-code-chemistry-python", "Code2", "Code2"),
        ("jablonkagroup/codeparrot_github-code-chemistry-python", "Code3", "Code3"),
        
        #Web
        ("AI-ModelScope/webvid-10M", "Web1", "Web1"),
        ("prithivMLmods/OpenWeb888K", "Web2", "Web2"),
        ("OmniData/Pile-OpenWebText2", "Web3", "Web3"),

        #Audio
        ("OmniData/Clotho", "Audio1", "Audio1"),
        ("modelscope/Libri2Mix_8k", "Audio2", "Audio2"),
        ("lmms-lab/AudioSetCaps_350k_converted", "Audio3", "Audio3"),

        #Image
        ("modelscope/coco_captions_small_slice", "Image1", "Image1"),
        ("FreedomIntelligence/ShareGPT-4o-Image", "Image2", "Image2"),

        #Other
        ("swift/VQAv2", "VQAv2", "VQAv2"),
        ("OmniData/FinQA", "FinQA", "FinQA"),
        ("swift/DocVQA", "DocVQ1A", "DocVQA1"),
        ("modelscope/ceval-exam", "Exam", "Exam"),
        ("AI-ModelScope/LAION-SG", "SG1", "SG1"),
        ("HuggingFaceH4/ultrachat_200k", "Chat1", "Chat1"),
        ("OpenDataLab/PubLayNet", "Publaynet1", "PubLayNet1"),
        ("krisfu/delicate_medical_r1_data", "Medical1", "Medical1"),
    ]
    
    # Check downloaded datasets
    downloaded = set()
    for _, save_name, _ in datasets:
        dataset_path = os.path.join(DATA, save_name)
        if os.path.exists(dataset_path):
            downloaded.add(save_name)
    
    # Filter datasets to be downloaded
    to_download = [(d, s, desc) for d, s, desc in datasets if s not in downloaded]
    total = len(datasets)
    downloaded_count = len(downloaded)
    
    if not to_download:
        print(f"✅\tAll {total} datasets already downloaded")
        return
    
    print(f"🟧\tDetected {total} total datasets, {downloaded_count} downloaded, {len(to_download)} need download")
    success_count = 0
    successfully_downloaded = set()

    if to_download:
        # Smart core allocation: CPU-1 for <8 cores, max 8 for >=8 cores
        cpu_cores = multiprocessing.cpu_count()
        workers = max(1, cpu_cores - 1) if cpu_cores < 8 else min(cpu_cores, 8)
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(_download_worker, to_download)
        for save_name in results:
            if save_name:
                success_count += 1
                successfully_downloaded.add(save_name)
    else:
        print(f"✅\tAll {total} datasets already downloaded")
            
    # Perform unified cleaning after all datasets are downloaded using auto_clean with multiprocessing
    if post_download_clean and successfully_downloaded:
        print(f"\n🟧\tStarting unified cleaning for all {len(successfully_downloaded)} downloaded datasets...")
        try:
            DatasetCleaner.auto_clean(
                input_dir=DATA,
                output_dir=DATA,  # Overwrite all datasets in-place
                min_length=1,
                text_field=None,
                workers=None  # Enable multiprocessing support
            )
            print(f"✅\tUnified cleaning completed for all datasets")
        except Exception as e:
            print(f"❌\tUnified cleaning failed: {e}")
            try:
                DatasetCleaner.auto_clean(
                    input_dir=DATA,
                    output_dir=DATA,
                    min_length=1,
                    text_field=None
                )
                print(f"✅\tUnified cleaning completed in fallback mode")
            except Exception as e2:
                print(f"❌\tUnified cleaning in fallback mode failed: {e2}")
    
        print("🟧\tCleaning up system cache...")

        modelscope_ds_cache = os.path.expanduser("~/.cache/modelscope/hub/datasets")
        cache_dirs = [
            os.path.join(DATA, ".cache"),
            os.path.join(DATA, "tmp"),
            os.path.join(DATA, "temp"),
            os.path.join(DATA, "cache"),
            os.path.join(DATA, "downloads"),
            modelscope_ds_cache
        ]
        # Remove all available cache directories
        for dir_path in cache_dirs:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"✅\tRemoved cache directory: {dir_path}")
        
        gc.collect()
        print("✅\tSystem garbage collection completed")
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✅\tCUDA memory cache cleared")
        except ImportError:
            pass
        
        print(f"✅\tDownload completed! Success: {success_count}/{len(datasets)}")
        
        # Generate model.txt with successfully downloaded datasets
        if successfully_downloaded:
            model_file = os.path.join(DATA, "model.txt")
            try:
                with open(model_file, 'w', encoding='utf-8') as f:
                    for dataset_name in sorted(successfully_downloaded):
                        f.write(f"{dataset_name}\n")
                print(f"✅\tGenerated model.txt with {len(successfully_downloaded)} datasets: {model_file}")
            except Exception as e:
                print(f"❌\tFailed to generate model.txt: {e}")

def _download_worker(args):
    """
    Worker function for multiprocessing dataset downloads.
    Must be at module level to be picklable by multiprocessing.
    """
    dataset_name, save_name, description = args
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"✅\tDownloading {description} ({dataset_name})... (Attempt {attempt+1}/{max_retries})")

            split_tried = None
            ds = None
            for split in ['train', 'validation', 'test']:
                try:
                    try:
                        ds = MsDataset.load(dataset_name, split=split, trust_remote_code=False)
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "trust_remote_code" in error_msg or "remote code" in error_msg:
                            ds = MsDataset.load(dataset_name, split=split, trust_remote_code=True)
                            print(f"🟧\tEnabling remote code trust mode to load {dataset_name}")
                        elif "oss2" in error_msg or "oss" in error_msg:
                            print(f"🟧\tOSS dependency issue detected, please ensure oss2 is installed")
                            raise
                        else:
                            raise
                    split_tried = split
                    print(f"✅\tUsing split '{split}' for {dataset_name}")
                    break
                except Exception as e:
                    last_split_error = str(e)
                    if "dataset generation" in str(e).lower():
                        print(f"🟧\tDataset generation error, will retry...")
                        continue
            if ds is None:
                print(f"❌\tNo available split (tried train/validation/test). Last error: {last_split_error}")
                if attempt < max_retries - 1:
                    print(f"🟧\tRetrying in 5 seconds...")
                    import time
                    time.sleep(5)
                    continue
                return None
            if hasattr(ds, 'to_hf_dataset'):
                ds = ds.to_hf_dataset()
            original_size = len(ds)
            print(f"✅\tDataset loaded successfully, original samples: {original_size:,}")
            if original_size > 0:
                print(f"✅\tKeeping all {original_size:,} samples...")
                ds = ds
            if save(ds, save_name):
                print(f"✅\tClearing cache after {save_name}...")
                if 'ds' in locals():
                    del ds
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
                return save_name
            else:
                print(f"🟧\tSave failed for {dataset_name}, retrying...")
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"❌\tAttempt {attempt+1} failed: {e}. Retrying...")
            else:
                print(f"❌\tAttempt {attempt+1} failed: {e}. No retries left.")
    return None

def optimize_datasets(max_keep=None):
    """
    Clean the dataset directly in-place without creating any temporary directories or suffix files.

    Args:
        max_keep (int, optional): Deprecated, no longer limit the number of datasets.
    """
    from datasets import load_from_disk, Dataset
    import pandas as pd
    
    for raw_dir in glob.glob("data_cache/*"):
        # Skip non-directory files
        if not os.path.isdir(raw_dir):
            continue
            
        try:
            # Load the original dataset directly
            print(f"🟧\tProcessing {raw_dir}...")
            ds = load_from_disk(raw_dir)
            original_len = len(ds)
            
            if original_len == 0:
                print(f"🟧\t{raw_dir} - Original dataset is empty, skipping")
                continue
            
            # Convert to pandas DataFrame for processing
            df = ds.to_pandas()
            
            # Automatically detect the text field
            text_field = None
            possible_fields = ['text', 'content', 'instruction', 'input', 'prompt', 'question', 'sentence', 'passage', 'document']
            for field in possible_fields:
                if field in df.columns:
                    text_field = field
                    break
            
            if not text_field:
                # If no standard field is found, use the first string column
                string_cols = df.select_dtypes(include=['object']).columns
                if len(string_cols) > 0:
                    text_field = string_cols[0]
                    print(f"🟧\tUsing string column '{text_field}' as the text field")
                else:
                    print(f"🟧\t{raw_dir} - No text field found, skipping")
                    continue
            
            # Clean text data
            import re
            def clean_text_simple(text):
                if not isinstance(text, str):
                    return ""
                text = str(text).strip()
                if not text:
                    return ""
                
                # Only remove control characters and standardize spaces
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            
            # Apply cleaning
            df[text_field] = df[text_field].apply(clean_text_simple)
            
            # Filter empty text using standard (minimum 1 character)
            mask = df[text_field].astype(str).str.strip().str.len() >= 1
            df_cleaned = df[mask]
            
            if len(df_cleaned) == 0:
                print(f"🟧\t{raw_dir} - No valid data after cleaning, skipping")
                continue
            
            # No longer limit the data volume, keep all cleaned data
            # If len(df_cleaned) > 0, keep all
                
            # Save the cleaned data directly back to the original directory
            new_ds = Dataset.from_pandas(df_cleaned, preserve_index=False)
            new_ds.save_to_disk(raw_dir)
            
            print(f"✅\t{raw_dir} | In-place cleaning completed: {len(df_cleaned)}/{original_len} records")
            
        except Exception as e:
            print(f"❌\t{raw_dir} - Processing failed: {e}")
            continue