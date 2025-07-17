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
try:
    from modelscope.msdatasets import MsDataset
except ImportError as e:
    print("❌ Current modelscope version requires datasets >=2.14.7. Please upgrade datasets to enable ModelScope dataset download."); MsDataset = None
except Exception as e:
    print(f"❌ ModelScope import error: {e}"); MsDataset = None


ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, "..", "data_cache")
os.makedirs(DATA, exist_ok=True)


def save(ds, name):
    """Save dataset to local cache"""
    try:
        save_path = os.path.join(DATA, name)
        print(f"✅ Saving {name} to {save_path}...")
        ds.save_to_disk(save_path)
        print(f"✅ {name} saved to {save_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save {name}: {e}")
        return False


def download_datasets(max_samples_per_dataset=50000):
    """Download all datasets with size control"""
    print("✅ Starting ModelScope dataset download...")
    print(f"Max samples per dataset: {max_samples_per_dataset:,}")
    
    # Core datasets for Pisces L1 training
    datasets = [
        ("AI-ModelScope/TinyStories", "tiny_stories", "High-quality text stories"),
        ("AI-ModelScope/alpaca-gpt4-data-zh", "alpaca_zh", "Chinese instruction data"),
        ("AI-ModelScope/alpaca-gpt4-data-en", "alpaca_en", "English instruction data"),
        ("AI-ModelScope/ShareGPT-Chinese-English-90k", "sharegpt", "ShareGPT dialogue data"),
        ("AI-ModelScope/ultrachat_200k", "ultrachat", "UltraChat dialogue data"),
        ("zacbi2023/coco2017_caption", "coco_caption_2017", "COCO 2017 Captions"),
        ("AI-ModelScope/LLaVA-Instruct-150K", "llava_instruct", "LLaVA instruction data"),
        ("thomas/MMC", "mmc4", "Multimodal C4 dataset"),
        ("lmms-lab/AudioSetCaps_220k_converted", "audioset_caps_220k", "Audio caption data (220k)"),
        ("AI-ModelScope/ESC-50", "esc50", "Environmental sound classification"),
        ("AI-ModelScope/DocVQA", "docvqa", "Document Visual Question Answering"),
        ("AI-ModelScope/RVL-CDIP", "rvl_cdip", "Document image classification"),
        # ("AI-ModelScope/webvid-10M", "webvid", "Web video dataset"),
        # ("AI-ModelScope/ActivityNet-QA", "activitynet_qa", "Video question answering dataset"),
        # ("AI-ModelScope/YouCook2", "youcook2", "Video cooking dataset with captions"),
        ("AI-ModelScope/DocBank", "docbank", "Document layout analysis dataset"),
        ("AI-ModelScope/FUNSD", "funsd", "Form understanding in noisy scanned documents"),
        ("AI-ModelScope/OCR-Receipt-Dataset", "ocr_receipt", "Receipt OCR dataset"),
        ("AI-ModelScope/TableBank", "tablebank", "Table structure recognition dataset"),
        ("AI-ModelScope/TVQA", "tvqa", "TVQA video-language dataset"),
        ("AI-ModelScope/VoxCeleb1", "voxceleb1", "Speaker recognition audio dataset"),
        ("cutedataset/imagenet-1k", "imagenet1k", "ImageNet 1K classification dataset"),
        # ("Data-Juicer/the-pile-philpaper-refined-by-data-juicer", "the_pile", "The Pile (large-scale high-quality English text)"),
        # ("swift/RedPajama-Data-1T", "redpajama", "RedPajama (1T tokens)"),
        # ("swift/chinese-c4", "c4", "Colossal Clean Crawled Corpus (C4)"),
        ("Intelligent-Internet/wikipedia_en", "wikipedia_en", "Wikipedia English"),
        ("OmniData/Pile-BookCorpus2", "bookcorpus", "BookCorpus English books"),
        # ("AI-ModelScope/LAION-SG", "laion_400m", "LAION-400M image-text pairs"),
        # ("wxzhuyeah/laion5b_parquet", "laion_5b", "LAION-5B image-text pairs"),
        # ("swift/moondream2-coyo-5M-captions", "coyo_700m", "COYO-700M image-text pairs"),
        ("lmms-lab/LLaVA-ReCap-CC3M", "cc3m", "Conceptual Captions 3M"),
        ("lmms-lab/LLaVA-ReCap-CC12M", "cc12m", "Conceptual Captions 12M"),
        ("OmniData/Clotho", "clotho", "Clotho audio captioning"),
        ("pkufool/LibriSpeech", "librispeech", "LibriSpeech English audiobooks"),
        ("OpenDataLab/PubLayNet", "publaynet", "PubLayNet scientific paper layout"),
        ("OmniData/CORD-19", "cord", "CORD receipt OCR"),
        # ("OmniData/HowTo100M", "howto100m", "HowTo100M video-subtitle pairs"),
        # ("OpenDataLab/MSR-VTT", "msrvtt", "MSR-VTT video retrieval"),
        # ("cucl2a/VGGSound4AVQA_small", "vggsound", "VGGSound video audio events"),
        ("lmms-lab/MMBench", "mmbench", "MMBench multimodal reasoning benchmark"),
        ("lmms-lab/MMVet", "mmvet", "MMVet multimodal reasoning benchmark"),
        ("AI-ModelScope/MMMU", "mmmu", "MMMU multimodal understanding"),
        ("AI-ModelScope/ScienceQA", "scienceqa", "ScienceQA science image-text QA"),
        ("AI-ModelScope/modelscope/R1-Distill-Math-Test", "mathvista", "MathVista math visual reasoning"),
        ("AI-ModelScope/M3IT", "m3it", "M3IT multimodal instruction tuning"),
    ]
    
    success_count = 0
    for dataset_name, save_name, description in datasets:
        if MsDataset is None:
            print(f"❌ MsDataset unavailable. Skipping {dataset_name}. Please upgrade modelscope>=1.28.0 and datasets>=2.14.7.")
            continue
        try:
            print(f"✅ Downloading {description} ({dataset_name})...")
            split_tried = None
            ds = None
            for split in ['train', 'validation', 'test']:
                try:
                    ds = MsDataset.load(dataset_name, split=split)
                    split_tried = split
                    print(f"✅ Using split '{split}' for {dataset_name}")
                    break
                except Exception as e:
                    last_split_error = e
            if ds is None:
                print(f"❌ Failed to download {dataset_name}: No available split (tried train/validation/test). Last error: {last_split_error}")
                continue
            if hasattr(ds, 'to_hf_dataset'):
                ds = ds.to_hf_dataset()
            original_size = len(ds)
            print(f"✅ Dataset loaded successfully, original samples: {original_size:,}")
            # Limit dataset size if needed
            if original_size > max_samples_per_dataset:
                print(f"✅ Limiting to {max_samples_per_dataset:,} samples...")
                ds = ds.select(range(min(max_samples_per_dataset, original_size)))
            if save(ds, save_name):
                success_count += 1
        except Exception as e:
            print(f"❌ Failed to download {dataset_name}: {e}")
    
    print(f"✅ Download completed! Success: {success_count}/{len(datasets)}")
    return success_count