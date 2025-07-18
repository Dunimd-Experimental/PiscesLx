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
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from model.tokenizer import get_tokenizer
from model.multimodal import VisionEncoder, AudioEncoder, DocEncoder


class PiscesDataset(Dataset):
    """Pisces dataset with multimodal support (text, image, audio, doc, video)"""
    def __init__(self, subset="tiny", split="train", config=None):
        # Try loading from local cache first
        cache_path = os.path.join("data_cache", subset)
        
        try:
            if os.path.exists(cache_path):
                print(f"✅\tLoading dataset from local cache: {cache_path}")
                self.ds = load_from_disk(cache_path)
                if split == "train" and "train" in self.ds:
                    self.ds = self.ds["train"]
                elif split == "test" and "test" in self.ds:
                    self.ds = self.ds["test"]
                print(f"✅\tLocal dataset loaded successfully: {len(self.ds)} samples")
            else:
                print(f"❌\tLocal cache not found, trying online download: {subset}")
                if MsDataset is None:
                    print("❌\tMsDataset unavailable. Cannot load ModelScope dataset online. Please upgrade modelscope>=1.28.0 and datasets>=2.14.7, or use only local datasets.")
                    self.ds = [{"text": f"Hello world {i}", "id": i} for i in range(100)]
                else:
                    try:
                        msds = MsDataset.load(subset, split=split)
                        if hasattr(msds, 'to_hf_dataset'):
                            self.ds = msds.to_hf_dataset()
                        else:
                            self.ds = msds
                        print(f"✅\tOnline dataset loaded successfully: {len(self.ds)} samples")
                    except Exception as e:
                        print(f"❌\tMsDataset.load failed: {e}")
                        print("❌\tCould not load ModelScope dataset online. Falling back to local test data.")
                        self.ds = [{"text": f"Hello world {i}", "id": i} for i in range(100)]
        except Exception as e:
            print(f"❌\tDataset loading failed: {e}")
            print("❌\tCreating test dataset...")
            # Create simple test dataset
            self.ds = [{"text": f"Hello world {i}", "id": i} for i in range(100)]
        
        self.tokenizer = get_tokenizer()
        self.config = config
        # Initialize preprocessors
        self.vision_encoder = VisionEncoder(config) if config else None
        self.audio_encoder = AudioEncoder(config) if config else None
        self.doc_encoder = DocEncoder(config) if config else None
        # self.video_encoder = VideoEncoder(config) if (config and VideoEncoder is not None) else None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        # Text
        text = item.get("text", "")
        input_ids = self.tokenizer.encode(text, return_tensors="pt")[0]

        # Image
        pixel_values = None
        if "image" in item and self.vision_encoder and self.vision_encoder.enabled:
            try:
                pixel_values = self.vision_encoder.process_image(item["image"])
                print(f"🟧\tImage processed successfully: {item['image']}")
            except Exception as e:
                print(f"❌\tImage processing error: {e}")

        # Audio
        audio_input = None
        if "audio" in item and self.audio_encoder and self.audio_encoder.enabled:
            try:
                audio_input = self.audio_encoder.process_audio(item["audio"])
                print(f"🟧\tAudio processed successfully: {item['audio']}")
            except Exception as e:
                print(f" Audio processing error: {e}")

        # Document
        doc_input = None
        if "doc" in item and self.doc_encoder and self.doc_encoder.enabled:
            try:
                doc_input = self.doc_encoder.process_doc(item["doc"])
                print(f"🟧\tDoc processed successfully: {item['doc']}")
            except Exception as e:
                print(f"❌\tDoc processing error: {e}")

        # Video
        # video_frames = None
        # if "video" in item and self.video_encoder and getattr(self.video_encoder, 'enabled', True):
        #     try:
        #         video_frames = self.video_encoder.process_video(item["video"])
        #         print(f"🟧\tVideo processed successfully: {item['video']}")
        #     except Exception as e:
        #         print(f"❌\tVideo processing error: {e}")

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "pixel_values": pixel_values,
            "audio_input": audio_input if audio_input is not None else {'input_values': None},
            "doc_input": doc_input,
            # "video_frames": video_frames
        }