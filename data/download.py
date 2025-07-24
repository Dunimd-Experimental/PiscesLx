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
import shutil
import shutil, glob
from clean import DatasetCleaner
from datasets import load_from_disk


try:
    from modelscope.msdatasets import MsDataset
except ImportError as e:
    print("❌\tCurrent modelscope version requires datasets >=2.14.7. Please upgrade datasets to enable ModelScope dataset download."); MsDataset = None
except Exception as e:
    print(f"❌\tModelScope import error: {e}"); MsDataset = None


from clean import DatasetCleaner

ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, "..", "data_cache")
os.makedirs(DATA, exist_ok=True)


def save(ds, name):
    """Save dataset to local cache"""
    try:
        save_path = os.path.join(DATA, name)
        print(f"✅\tSaving {name} to {save_path}...")
        ds.save_to_disk(save_path)
        print(f"✅\t{name} saved to {save_path}")
        return True
    except Exception as e:
        print(f"❌\tFailed to save {name}: {e}")
        return False


def download_datasets(max_samples_per_dataset=50000):
    """Download all datasets with size control"""
    print("✅\tStarting ModelScope dataset download...")
    print(f"\tMax samples per dataset: {max_samples_per_dataset:,}")
    
    # Core datasets for Pisces L1 training
    datasets = [
        ("open-r1/OpenR1-Math-220k", "Mat1h", "Math1"),
        ("zhuangxialie/Llama3-Chinese-Dataset", "Chinese1", "Chinese1"),
        ("liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT", "Chinese2", "Chinese2"),
        ("prithivMLmods/OpenWeb888K", "Web1", "Web1"),
        ("swift/wikipedia", "Wikipedia1", "20220301.zh")
        ("FreedomIntelligence/ShareGPT-4o-Image", "Image1", "Image1"),
        ("HuggingFaceH4/ultrachat_200k", "Chat1", "Chat1"),
        ("HuggingFaceH4/CodeAlpaca_20K", "Code1", "Code1"),
        ("jablonkagroup/codeparrot_github-code-chemistry-python", "Code2", "Code2"),
        ("modelscope/coco_captions_small_slice", "Image1", "Image1"),
        ("AI-ModelScope/LAION-SG", "SG1", "SG1"),
        ("lmms-lab/AudioSetCaps_350k_converted", "Audio1", "Audio1"),
        ("modelscope/Libri2Mix_8k", "Audio2", "Audio2"),
        ("OmniData/Clotho", "Audio3", "Audio3"),
        ("swift/DocVQA", "DocVQ1A", "DocVQA1"),
        ("OpenDataLab/PubLayNet", "Publaynet1", "PubLayNet1"),
        ("swift/VQAv2", "VQAv2", "VQAv2"),
    ]
    
    success_count = 0
    for dataset_name, save_name, description in datasets:
        if MsDataset is None:
            print(f"❌\tMsDataset unavailable. Skipping {dataset_name}. Please upgrade modelscope>=1.28.0 and datasets>=2.14.7.")
            continue
        max_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                print(f"✅\tDownloading {description} ({dataset_name})... (Attempt {attempt+1}/{max_retries})")
                split_tried = None
                ds = None
                for split in ['train', 'validation', 'test']:
                    try:
                        ds = MsDataset.load(dataset_name, split=split, trust_remote_code=True)
                        split_tried = split
                        print(f"✅\tUsing split '{split}' for {dataset_name}")
                        break
                    except Exception as e:
                        last_split_error = e
                if ds is None:
                    print(f"❌\tNo available split (tried train/validation/test). Last error: {last_split_error}")
                    break  # No need to retry if no splits available
                if hasattr(ds, 'to_hf_dataset'):
                    ds = ds.to_hf_dataset()
                original_size = len(ds)
                print(f"✅\tDataset loaded successfully, original samples: {original_size:,}")
                if original_size > max_samples_per_dataset:
                    print(f"✅\tLimiting to {max_samples_per_dataset:,} samples...")
                    ds = ds.select(range(min(max_samples_per_dataset, original_size)))
                if save(ds, save_name):
                    success_count += 1
                    success = True
                    break
                else:
                    print(f"🟧\tSave failed for {dataset_name}, retrying...")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"❌\tAttempt {attempt+1} failed: {e}. Retrying...")
                else:
                    print(f"❌\tAttempt {attempt+1} failed: {e}. No retries left.")
        if not success:
            print(f"❌\tFailed to download {dataset_name} after {max_retries} attempts")
            
    model_txt_path = os.path.join(DATA, "model.txt")
    with open(model_txt_path, "w", encoding="utf-8") as f:
        for _, save_name, _ in datasets:
            f.write(f"{save_name}\n")
    print(f"✅\tUpdated {model_txt_path}")
    print(f"✅\tDownload completed! Success: {success_count}/{len(datasets)}")
    print(f"🔄 Starting automatic dataset cleaning...")
    try:
        DatasetCleaner.auto_clean(input_dir=DATA)
        print(f"✅ Automatic cleaning completed successfully")
    except Exception as e:
        print(f"❌ Error during dataset cleaning: {e}")
    print(f"✅\tDownload and cleaning completed! Success: {success_count}/{len(datasets)}")


def optimize_datasets(max_keep=10000):
    """
    1. Clean and truncate datasets immediately after downloading.
    2. Output to the data_clean directory, retaining Chinese and general text.
    3. Overwrite model.txt to point to the cleaned directory.
    """
    for raw_dir in glob.glob("data_cache/*"):
        if not os.path.isdir(raw_dir) or raw_dir.endswith("_clean"):
            continue
        clean_dir = raw_dir + "_clean"
        if not os.path.exists(clean_dir):
            DatasetCleaner.auto_clean(
                input_dir=raw_dir,
                output_dir=clean_dir,
                min_length=10,
                keep_pattern=r'[\u4e00-\u9fff\d\w，。！？]'
            )
        # Keep only the first max_keep entries
        ds = load_from_disk(clean_dir)
        if len(ds) > max_keep:
            ds = ds.select(range(max_keep))
            ds.save_to_disk(clean_dir)
        print(f"✅ {raw_dir} -> {clean_dir} | {len(ds)} entries")
    # Rewrite model.txt to point to the cleaned directories
    with open("data_cache/model.txt", "w") as f:
        for d in glob.glob("data_cache/*_clean"):
            f.write(os.path.basename(d) + "\n")
