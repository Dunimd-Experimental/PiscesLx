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
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from utils.progress import progress_bar
from model.tokenizer import get_tokenizer
from utils.log import RIGHT, DEBUG, ERROR
from model.multimodal import VisionEncoder, AudioEncoder, DocEncoder, VideoEncoder

# Keys used to identify image data in the dataset
IMAGE_KEYS = ["image", "img_path", "image_path", "picture", "pic"]
# Keys used to identify audio data in the dataset
AUDIO_KEYS = ["audio", "audio_path", "wav", "sound"]
# Keys used to identify document data in the dataset
DOC_KEYS = ["doc", "document", "doc_path", "pdf"]
# Keys used to identify video data in the dataset
VIDEO_KEYS = ["video", "video_path", "mp4", "avi", "mov", "mkv"]

def _get_first_valid(item: dict, keys: list) -> str:
    """
    Retrieve the value corresponding to the first valid key from the given dictionary.

    Args:
        item (dict): The dictionary to search within.
        keys (list): A list of keys to search for.

    Returns:
        str: The value of the first valid key, or None if no valid key is found.
    """
    for k in keys:
        if k in item and isinstance(item[k], str) and item[k].strip():
            return item[k]
    return None

class PiscesDataset(Dataset):
    """
    Pisces dataset with multimodal support for text, image, audio, document, and video data.
    """
    def __init__(self, subset: str = "tiny", split: str = "train", config=None):
        """
        Initialize the PiscesDataset.

        Args:
            subset (str, optional): Name of the dataset subset. Defaults to "tiny".
            split (str, optional): Dataset split type, either "train" or "test". Defaults to "train".
            config: Configuration object. Defaults to None.
        """
        # First attempt to load dataset from local cache
        cache_path = os.path.join("data_cache", subset)
        
        try:
            if os.path.exists(cache_path):
                RIGHT(f"Loading dataset from local cache: {cache_path}")
                self.ds = load_from_disk(cache_path)
                if split == "train" and "train" in self.ds:
                    self.ds = self.ds["train"]
                elif split == "test" and "test" in self.ds:
                    self.ds = self.ds["test"]
                
                # Filter out empty or invalid samples with improved cleaning
                original_size = len(self.ds)
                RIGHT(f"Local dataset loaded successfully: {original_size} samples")
                if original_size > 0:
                    RIGHT("Filtering dataset to remove samples with no valid content...")
                    
                    # Import cleaning utilities
                    from .clean import StreamCleaner
                    cleaner = StreamCleaner(min_len=1, max_len=2048)  # Very permissive
                    
                    def has_valid_content(example: dict) -> bool:
                        """
                        Check if an example has valid content after cleaning.

                        Args:
                            example (dict): A single example from the dataset.

                        Returns:
                            bool: True if the example has valid content, False otherwise.
                        """
                        text = self._get_text(example)
                        if text is None:
                            return False
                        
                        # Apply gentle cleaning
                        cleaned_text = cleaner.clean_text(text)
                        
                        # Accept any non-empty content after cleaning
                        return cleaned_text is not None and len(cleaned_text.strip()) > 0
                    
                    self.ds = self.ds.filter(has_valid_content)
                    filtered_size = len(self.ds)
                    if filtered_size == 0:
                        DEBUG(f"Warning: All {original_size} samples were filtered out. Applying emergency cleaning...")
                        # Emergency mode: use original dataset with basic validation
                        def emergency_filter(example: dict) -> bool:
                            """
                            Perform basic validation on an example.

                            Args:
                                example (dict): A single example from the dataset.

                            Returns:
                                bool: True if the example has non-empty text, False otherwise.
                            """
                            text = self._get_text(example)
                            return text is not None and len(str(text).strip()) > 0
                        
                        self.ds = self.ds.filter(emergency_filter)
                        filtered_size = len(self.ds)
                        RIGHT(f"Emergency filter: {filtered_size}/{original_size} samples remain")
                    else:
                        RIGHT(f"Filtered out {original_size - filtered_size} samples. {filtered_size}/{original_size} samples remain ({(filtered_size/original_size)*100:.2f}%).")
            else:
                ERROR(f"Local cache not found, trying online download: {subset}")
                if "MsDataset" not in globals() or "MsDataset" not in locals():
                    ERROR("MsDataset unavailable. Cannot load ModelScope dataset online. Please upgrade modelscope>=1.28.0 and datasets>=2.14.7, or use only local datasets.")
                    self.ds = [{"text": f"Hello world {i}", "id": i} for i in range(100)]
                else:
                    try:
                        msds = MsDataset.load(subset, split=split)
                        if hasattr(msds, 'to_hf_dataset'):
                            self.ds = msds.to_hf_dataset()
                        else:
                            self.ds = msds
                        RIGHT(f"Online dataset loaded successfully: {len(self.ds)} samples")
                    except Exception as e:
                        ERROR(f"MsDataset.load failed: {str(e)}")
                        ERROR("Could not load ModelScope dataset online. Falling back to local test data.")
                        self.ds = [{"text": f"Hello world {i}", "id": i} for i in range(100)]
        except Exception as e:
            ERROR(f"Dataset loading failed: {str(e)}")
            ERROR("Creating test dataset...")
            # Create simple test dataset
            self.ds = [{"text": f"Hello world {i}", "id": i} for i in range(100)]
        
        self.tokenizer = get_tokenizer()
        self.config = config
        # Initialize preprocessors
        self.vision_encoder = VisionEncoder(config) if config else None
        self.audio_encoder = AudioEncoder(config) if config else None
        self.doc_encoder = DocEncoder(config) if config else None
        self.video_encoder = VideoEncoder(config) if config else None
        
        # Validate token IDs across the entire dataset
        self._validate_token_ids()
    
    def _validate_token_ids(self):
        """
        Validate whether the token IDs in the dataset are within the valid range.
        """
        vocab_size = len(self.tokenizer)
        invalid_samples = []
        
        # Limit validation to the first 1000 samples for performance reasons
        max_samples = min(1000, len(self.ds))
        
        # Use a unified progress bar
        for i in progress_bar(range(max_samples), desc="🟧\tValidating token IDs"):
            item = self.ds[i]
            text = self._get_text(item)
            if text.strip():
                try:
                    input_ids = self.tokenizer.encode(text, return_tensors="pt")[0]
                    max_token_id = input_ids.max().item() if len(input_ids) > 0 else 0
                    min_token_id = input_ids.min().item() if len(input_ids) > 0 else 0
                        
                    if max_token_id >= vocab_size or min_token_id < 0:
                        invalid_samples.append(i)
                            
                except Exception as e:
                    invalid_samples.append(i)
            
        RIGHT("Token validation complete!")
        
        # Print token ID range and vocabulary size for debugging
        DEBUG(f"Vocab size: {vocab_size}")
        if len(self.ds) > 0:
            sample_text = self._get_text(self.ds[0])
            if sample_text:
                sample_ids = self.tokenizer.encode(sample_text, return_tensors="pt")[0]
                DEBUG(f"Token ID range: {sample_ids.min().item()} - {sample_ids.max().item()}")
        
        if invalid_samples:
            DEBUG(f"Found {len(invalid_samples)} invalid samples in validation set")
        else:
            RIGHT("All token IDs validated successfully")
            
        if len(self.ds) > max_samples:
            DEBUG(f"Validated {max_samples}/{len(self.ds)} samples (performance optimization)")

    def _get_text(self, item: dict) -> str:
        """
        Extract text content from a dataset item.

        Args:
            item (dict): A single item from the dataset.

        Returns:
            str: The extracted text content, or an empty string if no text is found.
        """
        # Import text field keys from the data module
        from .__init__ import TEXT_FIELD_KEYS
        
        # Debug: Print the structure of the first few samples
        if hasattr(self, '_debug_count'):
            self._debug_count = getattr(self, '_debug_count', 0) + 1
            if self._debug_count <= 3:
                DEBUG(f"Debug item structure: {type(item)}, keys: {list(item.keys()) if isinstance(item, dict) else 'N/A'}")
        
        # Look for a direct key match
        for key in TEXT_FIELD_KEYS:
            if isinstance(item.get(key), str) and item[key].strip():
                return item[key]

        # Handle formats specific to the Chinese1 dataset
        if isinstance(item, dict):
            # Handle Chinese dataset formats
            for chinese_key in ['chinese', 'content', 'text_cn', 'cn_text', '中文']:
                if isinstance(item.get(chinese_key), str) and item[chinese_key].strip():
                    return item[chinese_key]

        # Handle conversational formats (e.g., ShareGPT)
        if 'conversations' in item and isinstance(item['conversations'], list) and item['conversations']:
            full_text = []
            for turn in item['conversations']:
                if isinstance(turn, dict):
                    # Handle various conversation formats
                    content = None
                    if 'value' in turn and turn['value'] and str(turn['value']).strip():
                        content = str(turn['value']).strip()
                    elif 'content' in turn and turn['content'] and str(turn['content']).strip():
                        content = str(turn['content']).strip()
                    elif 'text' in turn and turn['text'] and str(turn['text']).strip():
                        content = str(turn['text']).strip()
                    
                    if content:
                        # Add role prefix if available
                        role = turn.get('from', turn.get('role', ''))
                        if role:
                            full_text.append(f"{role}: {content}")
                        else:
                            full_text.append(content)
            if full_text:
                return "\n".join(full_text)

        # Handle instruction-following formats
        if 'instruction' in item and 'input' in item:
            instruction = str(item['instruction']).strip() if item['instruction'] else ""
            input_text = str(item['input']).strip() if item['input'] else ""
            if instruction and input_text:
                return f"{instruction}\n{input_text}"
            elif instruction:
                return instruction
            elif input_text:
                return input_text

        # Handle formats specific to Chinese datasets
        if 'prompt' in item and 'response' in item:
            prompt = str(item['prompt']).strip() if item['prompt'] else ""
            response = str(item['response']).strip() if item['response'] else ""
            if prompt and response:
                return f"{prompt}\n{response}"
            elif prompt:
                return prompt
            elif response:
                return response

        # Handle nested dictionary formats
        for key, value in item.items():
            if isinstance(value, dict):
                # Recursively extract text from nested dictionaries
                nested_text = self._get_text(value)
                if nested_text.strip():
                    return nested_text
            elif isinstance(value, str) and value.strip():
                # Use any non-empty string field as a fallback
                return value.strip()

        # Handle list formats (like the Chinese1 dataset)
        if isinstance(item, list) and item:
            # If the item is a list, try to extract text from the first element
            for sub_item in item:
                if isinstance(sub_item, dict):
                    text = self._get_text(sub_item)
                    if text.strip():
                        return text
                elif isinstance(sub_item, str) and sub_item.strip():
                    return sub_item.strip()

        # Final fallback: Concatenate all non-empty string values recursively
        def extract_all_strings(obj) -> list:
            """
            Recursively extract all non-empty string values from an object.

            Args:
                obj: The object to extract strings from.

            Returns:
                list: A list of non-empty string values.
            """
            strings = []
            if isinstance(obj, dict):
                for v in obj.values():
                    strings.extend(extract_all_strings(v))
            elif isinstance(obj, list):
                for v in obj:
                    strings.extend(extract_all_strings(v))
            elif isinstance(obj, str) and obj.strip():
                strings.append(obj.strip())
            return strings

        all_strings = extract_all_strings(item)
        if all_strings:
            return " ".join(all_strings)
            
        return ""

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the processed sample data.
        """
        item = self.ds[idx]

        # Text processing
        text = self._get_text(item)
        input_ids = self.tokenizer.encode(text, return_tensors="pt")[0]
        
        # Ensure token IDs are within the vocabulary bounds
        vocab_size = len(self.tokenizer)
        max_token_id = input_ids.max().item() if len(input_ids) > 0 else 0
        if max_token_id >= vocab_size:
            DEBUG(f"Token ID {max_token_id} exceeds vocab size {vocab_size}, clamping...")
            input_ids = torch.clamp(input_ids, max=vocab_size - 1)
        
        # Ensure no negative indices
        input_ids = torch.clamp(input_ids, min=0)

        # Image processing
        pixel_values = None
        image_path = _get_first_valid(item, IMAGE_KEYS)
        if image_path and self.vision_encoder and self.vision_encoder.enabled:
            try:
                pixel_values = self.vision_encoder.process_image(image_path)
                DEBUG(f"Image processed successfully: {image_path}")
            except Exception as e:
                ERROR(f"Image processing error: {str(e)}")

        # Audio processing
        audio_input = None
        audio_path = _get_first_valid(item, AUDIO_KEYS)
        if audio_path and self.audio_encoder and self.audio_encoder.enabled:
            try:
                audio_input = self.audio_encoder.process_audio(audio_path)
                DEBUG(f"Audio processed successfully: {audio_path}")
            except Exception as e:
                ERROR(f"Audio processing error: {str(e)}")

        # Document processing
        doc_input = None
        doc_path = _get_first_valid(item, DOC_KEYS)
        if doc_path and self.doc_encoder and self.doc_encoder.enabled:
            try:
                doc_input = self.doc_encoder.process_doc(doc_path)
                DEBUG(f"Doc processed successfully: {doc_path}")
            except Exception as e:
                ERROR(f"Doc processing error: {str(e)}")

        # Video processing with memory optimization warnings
        video_frames = None
        video_path = _get_first_valid(item, VIDEO_KEYS)
        if video_path and self.video_encoder and self.video_encoder.enabled:
            try:
                if torch.cuda.is_available():
                    # Check available memory before processing video
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    free = total - reserved
                    
                    if free < 2.0:  # Less than 2GB free
                        DEBUG(f"Low GPU memory detected ({free:.1f}GB free). Video processing may fail.")
                        DEBUG("Consider using --memory_efficient flag for training.")
                
                video_frames = self.video_encoder.process_video(video_path)
                RIGHT(f"Video processed successfully: {video_path}")
                
                # Memory cleanup after video processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                ERROR(f"Video processing error: {str(e)}")
                video_frames = None

        # Reasoning correctness label (for training the reflection head)
        correct_label = item.get("correct", 1)  # Default to 1 (correct) if not specified
        correct = torch.tensor(correct_label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "pixel_values": pixel_values,
            "audio_input": audio_input if audio_input is not None else {'input_values': None},
            "doc_input": doc_input,
            "video_frames": video_frames,
            "correct": correct
        }