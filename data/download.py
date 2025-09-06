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
import glob
import math
import shutil
import requests
import logging
import warnings
from tqdm import tqdm
import multiprocessing
from .clean import DatasetCleaner
from datasets import load_from_disk
from utils.log import RIGHT, DEBUG, ERROR

# Configure ModelScope cache to use separate directory from data_cache
MODELSCOPE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "modelscope")
os.environ['MODELSCOPE_CACHE'] = MODELSCOPE_CACHE_DIR
os.environ['MODELSCOPE_HUB_CACHE'] = MODELSCOPE_CACHE_DIR
os.environ['MODELSCOPE_DATASETS_CACHE'] = os.path.join(MODELSCOPE_CACHE_DIR, "datasets")
RIGHT(f"ModelScope cache configured to: {MODELSCOPE_CACHE_DIR}")

# Handle modelscope import gracefully
try:
    from modelscope.msdatasets import MsDataset
    MODELSCOPE_AVAILABLE = True
except ImportError as e:
    DEBUG(f"Modelscope not available, falling back to datasets: {e}")
    MODELSCOPE_AVAILABLE = False

logging.getLogger('modelscope').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='modelscope')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def detect_available_splits(dataset_name, trust_remote_code=True):
    """
    Detect available splits of the dataset.
    """
    available_splits = []
    split_candidates = [
        'train', 'train_full', 'train_all', 
        'validation', 'valid', 'dev',
        'test', 'eval', 'test_all'
    ]
    
    for split in split_candidates:
        try:
            if MODELSCOPE_AVAILABLE:
                MsDataset.load(dataset_name, split=split, trust_remote_code=trust_remote_code)
            else:
                from datasets import load_dataset
                load_dataset(dataset_name, split=split, trust_remote_code=True)
            available_splits.append(split)
        except Exception as e:
            DEBUG(f"Split {split} not available for {dataset_name}: {str(e)[:200]}...")
            continue
    
    # If no splits are found, try loading without specifying a split
    if not available_splits:
        try:
            if MODELSCOPE_AVAILABLE:
                MsDataset.load(dataset_name, trust_remote_code=trust_remote_code)
            else:
                from datasets import load_dataset
                load_dataset(dataset_name, trust_remote_code=True)
            available_splits.append('default')
        except Exception as e:
            DEBUG(f"No default split found for {dataset_name}: {str(e)[:200]}...")
    
    return available_splits

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
        RIGHT(f"Saving {name} to {save_path}...")
        # Save the dataset to the specified path
        ds.save_to_disk(save_path)
        RIGHT(f"{name} saved to {save_path}")
        return True
    except Exception as e:
        ERROR(f"Failed to save {name}: {e}")
        return False

def download_datasets(max_samples_per_dataset=50000, post_download_clean=True):
    """
    Download all datasets with size control.

    Args:
        max_samples_per_dataset (int, optional): The maximum number of samples per dataset. Defaults to 50000.
        post_download_clean (bool, optional): Whether to perform post-download cleaning. Defaults to True.
    """
    RIGHT("Starting ModelScope dataset download...")
    
    # Core datasets for Pisces L1 training
    datasets = [
        # Chinese
        ("baicai003/Llama3-Chinese-dataset", "Chinese1", "Chinese1"),
        ("liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT", "Chinese2", "Chinese2"),
        ("AI-ModelScope/OpenOrca-Chinese", "Chinese3", "Chinese3"),

        # English
        ("YorickHe/CoT", "English1", "English1"),
        ("DAMO_ConvAI/EnDoc2BotDialogue", "English2", "English2"),
        ("Intelligent-Internet/wikipedia_en", "English3", "English3"),

        # Math
        ("swift/MetaMathQA", "Math1", "Math1"),
        ("AI-MO/NuminaMath-CoT", "Math2", "Math2"),
        ("AI-ModelScope/NuminaMath-CoT", "Math3", "Math3"),
        ("xpengx/EleutherAI-proof-pile-2", "Math4", "Math4"),
        ("tastelikefeet/competition_math", "Math5", "Math5"),

        # Code
        ("HuggingFaceH4/CodeAlpaca_20K", "Code1", "Code1"),
        ("jablonkagroup/codeparrot_github-code-chemistry-python", "Code2", "Code2"),
        ("jablonkagroup/codeparrot_github-code-chemistry-python", "Code3", "Code3"),
        ("codefuse-ai/CodeExercise-Python-27k", "Code4", "Code4"),

        # Web
        ("AI-ModelScope/webvid-10M", "Web1", "Web1"),
        ("prithivMLmods/OpenWeb888K", "Web2", "Web2"),
        ("OmniData/Pile-OpenWebText2", "Web3", "Web3"),

        # Audio
        ("OmniData/Clotho", "Audio1", "Audio1"),
        ("modelscope/Libri2Mix_8k", "Audio2", "Audio2"),
        ("lmms-lab/AudioSetCaps_350k_converted", "Audio3", "Audio3"),

        # Image
        ("modelscope/coco_captions_small_slice", "Image1", "Image1"),
        ("FreedomIntelligence/ShareGPT-4o-Image", "Image2", "Image2"),

        # Agent
        ("iic/MSAgent-Bench", "Agent1", "Agent1"),
        ("iic/MSAgent-MultiRole", "Agent2", "Agent2"),
        ("AI-ModelScope/orca-agentinstruct-1M-v1", "Agent3", "Agent3"),
        ("iic/MSAgent-Pro", "Agent4", "Agent4"),
        ("AI-ModelScope/ms_agent_for_agentfabric", "Agent5", "Agent5"),

        # Other
        ("swift/VQAv2", "VQAv2", "VQAv2"),
        ("OmniData/FinQA", "FinQA", "FinQA"),
        ("swift/DocVQA", "DocVQ1A", "DocVQA1"),
        ("modelscope/ceval-exam", "Exam", "Exam"),
        ("AI-ModelScope/LAION-SG", "SG1", "SG1"),
        ("HuggingFaceH4/ultrachat_200k", "Chat1", "Chat1"),
        ("OpenDataLab/PubLayNet", "Publaynet1", "PubLayNet1"),
        ("krisfu/delicate_medical_r1_data", "Medical1", "Medical1"),
        ("BJQW14B/bs_challenge_financial_14b_dataset", "Financial1", "Financial1"),
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
        RIGHT(f"All {total} datasets already downloaded")
        return
    
    DEBUG(f"Detected {total} total datasets, {downloaded_count} downloaded, {len(to_download)} need download")
    success_count = 0
    successfully_downloaded = set()

    if to_download:
        # Smart core allocation: CPU-1 for <8 cores, max 8 for >=8 cores
        cpu_cores = multiprocessing.cpu_count()
        workers = max(1, cpu_cores - 1) if cpu_cores < 8 else min(cpu_cores, 8)
        with multiprocessing.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap_unordered(_download_worker, to_download), total=len(to_download), desc="Downloading datasets"))
            for save_name in results:
                if save_name:
                    success_count += 1
                    successfully_downloaded.add(save_name)
    else:
        RIGHT(f"All {total} datasets already downloaded")
            
    # Perform unified cleaning after all datasets are downloaded using auto_clean with multiprocessing
    if post_download_clean and successfully_downloaded:
        DEBUG(f"\nStarting unified cleaning for all {len(successfully_downloaded)} downloaded datasets...")
        try:
            DatasetCleaner.auto_clean(
                input_dir=DATA,
                output_dir=DATA,  # Overwrite all datasets in-place
                min_length=1,
                text_field=None,
                workers=None  # Enable multiprocessing support
            )
            RIGHT(f"Unified cleaning completed for all datasets")
        except Exception as e:
            ERROR(f"Unified cleaning failed: {e}")
            try:
                DatasetCleaner.auto_clean(
                    input_dir=DATA,
                    output_dir=DATA,
                    min_length=1,
                    text_field=None
                )
                RIGHT(f"Unified cleaning completed in fallback mode")
            except Exception as e2:
                ERROR(f"Unified cleaning in fallback mode failed: {e2}")
    
        DEBUG("Cleaning up system cache...")

        # Use the configured ModelScope cache directory
        cache_dirs = [
            os.path.join(DATA, ".cache"),
            os.path.join(DATA, "tmp"),
            os.path.join(DATA, "temp"),
            os.path.join(DATA, "cache"),
            os.path.join(DATA, "downloads"),
            MODELSCOPE_CACHE_DIR,  # Use the configured ModelScope cache
            os.path.join(MODELSCOPE_CACHE_DIR, "datasets"),
            os.path.join(MODELSCOPE_CACHE_DIR, "hub")
        ]
        # Remove all available cache directories
        for dir_path in cache_dirs:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                RIGHT(f"Removed cache directory: {dir_path}")
        
        gc.collect()
        RIGHT("System garbage collection completed")
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                RIGHT("CUDA memory cache cleared")
        except ImportError:
            pass
        
        RIGHT(f"Download completed! Success: {success_count}/{len(datasets)}")
        
        # Generate model.txt with successfully downloaded datasets
        if successfully_downloaded:
            model_file = os.path.join(DATA, "model.txt")
            try:
                with open(model_file, 'w', encoding='utf-8') as f:
                    for dataset_name in sorted(successfully_downloaded):
                        f.write(f"{dataset_name}\n")
                RIGHT(f"Generated model.txt with {len(successfully_downloaded)} datasets")
            except Exception as e:
                ERROR(f"Failed to generate model.txt: {e}")

def _download_worker(args):
    """
    Worker function for multiprocessing dataset downloads.
    Must be at module level to be picklable by multiprocessing.
    
    Args:
        args: 包含 (dataset_name, save_name, description) 的元�?
    """
    dataset_name, save_name, description = args
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            RIGHT(f"Downloading {description} ({dataset_name})... (Attempt {attempt+1}/{max_retries})")

            ds = None
            last_error = None
            
            # Try different loading methods with compatibility fixes
            load_methods = [
                # Method 1: Direct load without specifying a split
                {'kwargs': {'trust_remote_code': True}, 'desc': 'direct load with trust_remote_code=True'},
                # Method 2: Try different splits
                {'kwargs': {'split': 'train', 'trust_remote_code': True}, 'desc': 'with split=train'},
                {'kwargs': {'split': 'validation', 'trust_remote_code': True}, 'desc': 'with split=validation'},
                {'kwargs': {'split': 'test', 'trust_remote_code': True}, 'desc': 'with split=test'},
                # Method 3: Try using the default split
                {'kwargs': {'split': 'default', 'trust_remote_code': True}, 'desc': 'with split=default'}
            ]
            
            for method in load_methods:
                try:
                    if MODELSCOPE_AVAILABLE:
                        # Use try-except to handle DatasetFormations error
                        ds = MsDataset.load(dataset_name, **method['kwargs'])
                        if ds is not None and len(ds) > 0:
                            break
                    else:
                        # Fallback to datasets library
                        from datasets import load_dataset
                        split = method['kwargs'].get('split')
                        if split and split != 'default':
                            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
                        else:
                            ds = load_dataset(dataset_name, trust_remote_code=True)
                        if ds is not None and len(ds) > 0:
                            break
                            
                except ValueError as e:
                    if "DatasetFormations" in str(e) or "not a valid DatasetFormations" in str(e) or "cannot import" in str(e):
                        # Handle DatasetFormations enum error by trying simpler approach
                        try:
                            from datasets import load_dataset
                            split = method['kwargs'].get('split')
                            if split and split != 'default':
                                ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
                            else:
                                ds = load_dataset(dataset_name, trust_remote_code=True)
                            if ds is not None and len(ds) > 0:
                                break
                        except Exception as e2:
                            last_error = str(e2)
                            continue
                    else:
                        last_error = str(e)
                        continue
                except Exception as e:
                    # Try datasets library as fallback for any other error
                    try:
                        from datasets import load_dataset
                        split = method['kwargs'].get('split')
                        if split and split != 'default':
                            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
                        else:
                            ds = load_dataset(dataset_name, trust_remote_code=True)
                        if ds is not None and len(ds) > 0:
                            break
                    except Exception as e2:
                        last_error = str(e2)
                        continue
                
            if ds is not None and len(ds) > 0:
                RIGHT(f"Successfully loaded {dataset_name}")
            else:
                # Final fallback: try using datasets library directly
                try:
                    from datasets import load_dataset
                    ds = load_dataset(dataset_name, trust_remote_code=True)
                    if ds is not None and len(ds) > 0:
                        RIGHT(f"Successfully loaded {dataset_name} via fallback")
                    else:
                        ERROR(f"Failed to load {dataset_name} after all attempts")
                except Exception as e:
                    ERROR(f"Failed to load {dataset_name} after all attempts")
                    last_error = str(e)
            
            if ds is None or (hasattr(ds, '__len__') and len(ds) == 0):
                ERROR(f"Failed to load dataset {dataset_name} after trying all methods. Last error: {last_error}")
                if attempt < max_retries - 1:
                    DEBUG(f"Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
                    continue
                return None
                
            # Convert to Hugging Face format if needed
            if hasattr(ds, 'to_hf_dataset'):
                ds = ds.to_hf_dataset()
            # Handle different dataset formats
            elif hasattr(ds, 'data') and hasattr(ds, 'info'):
                # Already in HF format
                pass
            elif hasattr(ds, '__iter__') and not hasattr(ds, 'save_to_disk'):
                # Convert from modelscope format to HF format
                try:
                    from datasets import Dataset
                    if hasattr(ds, 'to_pandas'):
                        ds = Dataset.from_pandas(ds.to_pandas())
                    else:
                        # Assume it's already compatible
                        pass
                except Exception:
                    pass
            
            original_size = len(ds)
            RIGHT(f"Dataset loaded successfully, samples: {original_size:,}")
            
            if save(ds, save_name):
                RIGHT(f"Successfully saved {save_name}")
                # Clean up memory
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
                DEBUG(f"Save failed for {dataset_name}, retrying...")
                
        except Exception as e:
            if attempt < max_retries - 1:
                ERROR(f"Attempt {attempt+1} failed: {e}. Retrying...")
                import time
                time.sleep(5)  # Add delay to avoid frequent retries
            else:
                ERROR(f"All {max_retries} attempts failed for {dataset_name}. Last error: {e}")
    return None


def optimize_datasets(max_keep=None):
    """
    Optimize all downloaded datasets.
    
    Args:
        max_keep (int, optional): Deprecated, no longer limit the number of datasets.
    """
    from datasets import load_from_disk, Dataset
    import pandas as pd
    import glob
    import os
    
    for raw_dir in glob.glob("data_cache/*"):
        # Skip non-directory files
        if not os.path.isdir(raw_dir):
            continue
            
        try:
            # Load the original dataset directly
            DEBUG(f"Processing {raw_dir}...")
            ds = load_from_disk(raw_dir)
            original_len = len(ds)
            
            if original_len == 0:
                DEBUG(f"{raw_dir} - Original dataset is empty, skipping")
                continue
            
            # Convert to pandas DataFrame for processing
            df = ds.to_pandas()
            
            # Automatically detect the text field
            text_field = None
            from .__init__ import TEXT_FIELD_KEYS
            for field in TEXT_FIELD_KEYS:
                if field in df.columns:
                    text_field = field
                    break
            
            if not text_field:
                # If no standard field is found, use the first string column
                string_cols = df.select_dtypes(include=['object']).columns
                if len(string_cols) > 0:
                    text_field = string_cols[0]
                    DEBUG(f"Using string column '{text_field}' as the text field")
                else:
                    DEBUG(f"{raw_dir} - No text field found, skipping")
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
            
            # Filter empty text using the standard (at least 1 character)
            mask = df[text_field].astype(str).str.strip().str.len() >= 1
            df_cleaned = df[mask]
            
            if len(df_cleaned) == 0:
                DEBUG(f"{raw_dir} - No valid data after cleaning, skipping")
                continue
            
            # No longer limit the data volume, keep all cleaned data
            # If len(df_cleaned) > 0, keep all
                
            # Save the cleaned data back to the original directory directly
            new_ds = Dataset.from_pandas(df_cleaned, preserve_index=False)
            new_ds.save_to_disk(raw_dir)
            
            RIGHT(f"{raw_dir} | In-place cleaning completed: {len(df_cleaned)}/{original_len} records")
            
        except Exception as e:
            ERROR(f"{raw_dir} - Processing failed: {e}")
            continue