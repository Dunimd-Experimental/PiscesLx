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
import logging
import warnings
import multiprocessing
from .clean import DatasetCleaner
from utils.log import RIGHT, DEBUG, ERROR
from modelscope.msdatasets import MsDataset

warnings.filterwarnings("ignore", message=".*trust_remote_code.*")
warnings.filterwarnings("ignore", message=".*Will invoke codes.*")
warnings.filterwarnings("ignore", message=".*Repo.*not exists.*")

logging.getLogger('modelscope').setLevel(logging.ERROR)

# Get the root directory of the current script
ROOT = os.path.dirname(__file__)
# Define the data cache directory path
DATA = os.path.join(ROOT, "..", "data_cache")
# Create the data cache directory if it doesn't exist
os.makedirs(DATA, exist_ok=True)

def save(ds, name):
    """
    Save the dataset to the local cache.

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
        ERROR(f"Failed to save {name}: {str(e)}")
        return False

def download_datasets(max_samples_per_dataset=50000, post_download_clean=True, silent=True):
    """
    Download all datasets with size control.

    Args:
        max_samples_per_dataset (int, optional): The maximum number of samples per dataset. Defaults to 50000.
        post_download_clean (bool, optional): Whether to perform post-download cleaning. Defaults to True.
        silent (bool, optional): Whether to suppress warnings like trust_remote_code. Defaults to True.
    """
    if silent:
        warnings.filterwarnings("ignore", category=UserWarning, message=".*trust_remote_code.*")
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*trust_remote_code.*")
    
    # Core datasets for Pisces L1 training - stable working datasets
    datasets = [
        # Chinese
        ("baicai003/Llama3-Chinese-dataset", "Chinese1", "Chinese1"),
        ("liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT", "Chinese2", "Chinese2"),
        ("AI-ModelScope/OpenOrca-Chinese", "Chinese3", "Chinese3"),
        ("AI-ModelScope/ultrachat_200k", "Chinese4", "Chinese4"),
        ("gxlzgdmds/baidu_baike", "Chinese5", "Chinese5"),

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
    
    downloaded = {s for _, s, _ in datasets if os.path.exists(os.path.join(DATA, s))}
    to_download = [(d, s, desc) for d, s, desc in datasets if s not in downloaded]
    total, need = len(datasets), len(to_download)
    
    if not to_download:
        RIGHT(f"All {total} datasets already downloaded")
        return
    
    DEBUG(f"Total {total}, downloaded {total - need}, need {need}")
    success_count = 0
    successfully_downloaded = set()

    workers = max(1, multiprocessing.cpu_count() - 1) if multiprocessing.cpu_count() < 8 else min(multiprocessing.cpu_count(), 8)
    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.imap_unordered(_download_worker, to_download)
        for save_name in results:
            if save_name:
                success_count += 1
                successfully_downloaded.add(save_name)
            
    # Perform unified cleaning after all datasets are downloaded using auto_clean with multiprocessing
    if post_download_clean and successfully_downloaded:
        DEBUG(f"Starting unified cleaning for all {len(successfully_downloaded)} downloaded datasets...")
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
            ERROR(f"Unified cleaning failed: {str(e)}")
            try:
                DatasetCleaner.auto_clean(
                    input_dir=DATA,
                    output_dir=DATA,
                    min_length=1,
                    text_field=None
                )
                RIGHT(f"Unified cleaning completed in fallback mode")
            except Exception as e2:
                ERROR(f"Unified cleaning in fallback mode failed: {str(e2)}")
    
        DEBUG("Cleaning up system cache...")

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
                RIGHT(f"Removed cache directory: {dir_path}")
        
        gc.collect()
        RIGHT("System garbage collection completed")
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                RIGHT("CUDA memory cache cleared")
        except ImportError:
            print(f"Downloading {description} ({dataset_name})... ", end='')
        
        print(f"Download completed! Success: {success_count}/{len(datasets)}")
        
        # Generate model.txt with successfully downloaded datasets
        if successfully_downloaded:
            model_file = os.path.join(DATA, "model.txt")
            try:
                with open(model_file, 'w', encoding='utf-8') as f:
                    for dataset_name in sorted(successfully_downloaded):
                        f.write(f"{dataset_name}\n")
                RIGHT(f"Generated model.txt with {len(successfully_downloaded)} datasets: {model_file}")
            except Exception as e:
                ERROR(f"Failed to generate model.txt: {str(e)}")

def _load_with_hf_fallback(dataset_name):
    """Fallback to Hugging Face datasets when ModelScope fails."""
    try:
        from datasets import load_dataset
        
        # Extract dataset name from ModelScope format
        hf_name = dataset_name.split('/')[-1]
        
        # Try common configurations
        try:
            ds = load_dataset(hf_name, split='train', streaming=False)
            return ds
        except:
            try:
                ds = load_dataset(dataset_name, split='train', streaming=False)
                return ds
            except:
                ds = load_dataset(hf_name, streaming=False)
                return ds[0] if isinstance(ds, tuple) else ds
    except Exception as e:
        raise RuntimeError(f"Hugging Face fallback failed: {e}")


def _download_worker(args):
    """
    Worker function for multiprocessing dataset downloads.
    Must be at module level to be picklable by multiprocessing.
    """
    dataset_name, save_name, description = args
    max_retries = 3
    
    # Skip problematic datasets with metadata issues
    problematic_datasets = [
        "gxlzgdmds/baidu_baike",  # repo card metadata error
        "tastelikefeet/competition_math",  # metadata issues
    ]
    
    if dataset_name in problematic_datasets:
        ERROR(f"Skipping problematic dataset {dataset_name} due to metadata issues")
        return None
    
    for attempt in range(max_retries):
        try:
            pass

            ds = None
            last_error = None
            
            from tqdm import tqdm
            
            # Robust loading strategies with error handling and smart skip
            loading_strategies = [
                # Strategy 1: Default load with trust_remote_code=True to avoid warnings
                lambda: MsDataset.load(dataset_name, trust_remote_code=True, disable_tqdm=True),
                # Strategy 2: Fallback without parameters
                lambda: MsDataset.load(dataset_name, disable_tqdm=True),
                # Strategy 3: Hugging Face fallback
                lambda: _load_with_hf_fallback(dataset_name),
            ]
            
            with tqdm(total=len(loading_strategies), desc=f"{description}", unit="step", leave=False, ncols=80) as pbar:
                for i, load_func in enumerate(loading_strategies):
                    try:
                        ds = load_func()
                        if ds is not None and len(ds) > 0:
                            pbar.update(1)
                            break
                        else:
                            pbar.update(1)
                    except Exception as e:
                        pbar.update(1)
                        last_error = e
                        error_msg = str(e).lower()
                        
                        # Smart skip for known non-critical and data parsing errors
                        skip_patterns = [
                            "datasetformations", "csvconfig", "trust_remote_code", 
                            "unexpected keyword", "split", "namespace", "不存在的数据集",
                            "repository not found", "404", "permission denied",
                            "jsondecode", "parse", "schema", "format", "encoding",
                            "invalid", "corrupt", "malformed", "syntax", "utf-8",
                            "eof", "inside string", "tokenizing", "csv", "row"
                        ]
                        if any(pattern in error_msg for pattern in skip_patterns):
                            continue
            
            if ds is None or len(ds) == 0:
                # Try Hugging Face datasets as fallback
                try:
                    from datasets import load_dataset
                    hf_name = dataset_name.split('/')[-1]
                    ds = load_dataset(hf_name, split='train', streaming=False, disable_tqdm=True)
                    print(f"✅ {dataset_name} (HF fallback)")
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(skip_error in error_msg for skip_error in [
                        "not found", "does not exist", "no such dataset", "404"
                    ]):
                        ERROR(f"Dataset {dataset_name} does not exist in Hugging Face either, skipping...")
                        return None
                    
                    ERROR(f"Failed to load dataset {dataset_name} from all sources. Last error: {str(last_error or e)}")
                    if attempt < max_retries - 1:
                        DEBUG("Retrying in 3 seconds...")
                        import time
                        time.sleep(3)
                        continue
                    return None
            
            # Convert to HF dataset format
            if hasattr(ds, 'to_hf_dataset'):
                ds = ds.to_hf_dataset()
            
            original_size = len(ds)
            if original_size == 0:
                ERROR(f"Dataset {dataset_name} is empty, skipping...")
                return None
                
            if save(ds, save_name):
                model_txt_path = os.path.join('data_cache', 'model.txt')
                os.makedirs('data_cache', exist_ok=True)
                with open(model_txt_path, 'a', encoding='utf-8') as f:
                    f.write(f"{save_name}\n")
                return save_name
            else:
                return None
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Skip retry for non-existent datasets
            if any(skip_error in error_msg for skip_error in [
                "不存在的数据集", "not found", "does not exist", "no such dataset",
                "repository not found", "404", "permission denied"
            ]):
                ERROR(f"Dataset {dataset_name} does not exist or is not accessible, skipping...")
                return None
                
            if attempt < max_retries - 1:
                ERROR(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
            else:
                ERROR(f"Attempt {attempt + 1} failed: {str(e)}. No retries left.")
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
            
            # Filter empty text using standard (minimum 1 character)
            mask = df[text_field].astype(str).str.strip().str.len() >= 1
            df_cleaned = df[mask]
            
            if len(df_cleaned) == 0:
                DEBUG(f"{raw_dir} - No valid data after cleaning, skipping")
                continue
            
            # No longer limit the data volume, keep all cleaned data
            # If len(df_cleaned) > 0, keep all
                
            # Save the cleaned data directly back to the original directory
            new_ds = Dataset.from_pandas(df_cleaned, preserve_index=False)
            new_ds.save_to_disk(raw_dir)
            
            RIGHT(f"{raw_dir} | In-place cleaning completed: {len(df_cleaned)}/{original_len} records")
            
        except Exception as e:
            ERROR(f"{raw_dir} - Processing failed: {str(e)}")
            continue