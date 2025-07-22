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

IMAGE_KEYS = ["image", "img_path", "image_path", "picture", "pic"]
AUDIO_KEYS = ["audio", "audio_path", "wav", "sound"]
DOC_KEYS = ["doc", "document", "doc_path", "pdf"]
VIDEO_KEYS = ["video", "video_path", "mp4", "avi"]

def _get_first_valid(item, keys):
    for k in keys:
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k]
    return None

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
                
                # Filter out empty/invalid samples
                original_size = len(self.ds)
                print(f"✅\tLocal dataset loaded successfully: {original_size} samples")
                if original_size > 0:
                    print("✅\tFiltering dataset to remove samples with no valid content...")
                    self.ds = self.ds.filter(lambda example: self._get_text(example).strip() != "")
                    filtered_size = len(self.ds)
                    print(f"✅\tFiltered out {original_size - filtered_size} samples. {filtered_size}/{original_size} samples remain ({(filtered_size/original_size)*100:.2f}%).")
            else:
                print(f"❌\tLocal cache not found, trying online download: {subset}")
                if "MsDataset" not in globals() or MsDataset is None:
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

    def _get_text(self, item):
        # Common keys for text content in various datasets
        possible_keys = [
            "text", "content", "sentence", "paragraph", "body", "article", "summary", "desc", "description", "title",
            "instruction", "input", "output", "response", "target", "answer", "question", "reasoning", "explanation",
            "conversations", "turns", "messages", "dialogue", "history", "utterance",
            "problem", "solution", "proof", "rationale", "choices", "options",
            "prompt", "completion", "code", "canonical_solution", "test", "reference_solution", "nl", "pl",
            "caption", "image_caption", "audio_caption", "video_caption", "label",
        ]
        
        # Look for a direct key match
        for key in possible_keys:
            if isinstance(item.get(key), str) and item[key].strip():
                return item[key]

        # Handle conversational formats (e.g., ShareGPT)
        if 'conversations' in item and isinstance(item['conversations'], list) and item['conversations']:
            full_text = []
            for turn in item['conversations']:
                if isinstance(turn, dict) and 'from' in turn and 'value' in turn:
                    full_text.append(f"{turn['from']}: {turn['value']}")
            if full_text:
                return "\n".join(full_text)

        # Handle instruction-following formats
        if 'instruction' in item and 'input' in item:
            return f"{item['instruction']}\n{item['input']}"
        
        # Fallback: concatenate all string values
        all_strings = [v for v in item.values() if isinstance(v, str)]
        if all_strings:
            return " ".join(all_strings)
            
        return ""

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        # Text
        text = self._get_text(item)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")[0]

        # Image
        pixel_values = None
        image_path = _get_first_valid(item, IMAGE_KEYS)
        if image_path and self.vision_encoder and self.vision_encoder.enabled:
            try:
                pixel_values = self.vision_encoder.process_image(image_path)
                print(f"🟧\tImage processed successfully: {image_path}")
            except Exception as e:
                print(f"❌\tImage processing error: {e}")

        # Audio
        audio_input = None
        audio_path = _get_first_valid(item, AUDIO_KEYS)
        if audio_path and self.audio_encoder and self.audio_encoder.enabled:
            try:
                audio_input = self.audio_encoder.process_audio(audio_path)
                print(f"🟧\tAudio processed successfully: {audio_path}")
            except Exception as e:
                print(f"❌\tAudio processing error: {e}")

        # Document
        doc_input = None
        doc_path = _get_first_valid(item, DOC_KEYS)
        if doc_path and self.doc_encoder and self.doc_encoder.enabled:
            try:
                doc_input = self.doc_encoder.process_doc(doc_path)
                print(f"🟧\tDoc processed successfully: {doc_path}")
            except Exception as e:
                print(f"❌\tDoc processing error: {e}")

        # Video
        video_frames = None
        if hasattr(self, "video_encoder") and self.video_encoder and getattr(self.video_encoder, 'enabled', True):
            video_path = _get_first_valid(item, VIDEO_KEYS)
            if video_path:
                try:
                    video_frames = self.video_encoder.process_video(video_path)
                    print(f"🟧\tVideo processed successfully: {video_path}")
                except Exception as e:
                    print(f"❌\tVideo processing error: {e}")

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "pixel_values": pixel_values,
            "audio_input": audio_input if audio_input is not None else {'input_values': None},
            "doc_input": doc_input,
            "video_frames": video_frames
        }