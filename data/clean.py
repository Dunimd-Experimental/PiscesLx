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
import re
import gc
import json
import pandas as pd
import multiprocessing as mp
from utils.progress import progress_bar
from utils.log import RIGHT, DEBUG, ERROR
from typing import Dict, Callable, List, Optional, Tuple
from datasets import load_from_disk, Dataset, concatenate_datasets

# ========== Plug-in Rule System ==========
# Define a dictionary of text cleaning rules
RULES: Dict[str, Callable[[str], str]] = {
    # Remove control characters from the text
    "ctrl_chars": lambda x: re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', x),
    # Normalize whitespace in the text
    "whitespace": lambda x: re.sub(r'\s+', ' ', x).strip(),
    # Remove URLs from the text
    "urls":       lambda x: re.sub(r'http\S+|www\S+', '', x),
    # Remove email addresses from the text
    "emails":     lambda x: re.sub(r'\S+@\S+', '', x),
    # Remove HTML tags from the text
    "html_tags":  lambda x: re.sub(r'<[^>]*?>', '', x),
    # Remove emojis from the text (keep Chinese characters, word characters, whitespace and some punctuation)
    "emoji":      lambda x: re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:]', '', x),
    # Remove special characters from the text (keep word characters, whitespace, Chinese characters and some punctuation)
    "special_chars": lambda x: re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()""''-]', '', x),
    # Replace consecutive punctuation marks with a single period
    "extra_punct": lambda x: re.sub(r'[.,!?;:]{2,}', '.', x),
    # Remove text that consists only of digits
    "digits_only": lambda x: x if not x.strip().isdigit() else "",
    # Remove single-character text
    "single_chars": lambda x: x if len(x.strip()) > 1 else "",
}

# ========== Automatic Discovery of Multimodal Fields ==========
# Define a dictionary for automatic discovery of multimodal fields
AUTO_FIELDS = {
    # Possible field names for image data
    "image": ["image", "img_path", "image_path", "picture", "pic", "img"],
    # Possible field names for audio data
    "audio": ["audio", "audio_path", "wav", "sound", "mp3"],
    # Possible field names for document data
    "doc":   ["doc", "document", "doc_path", "pdf", "file_path"],
    # Possible field names for video data
    "video": ["video", "video_path", "mp4", "avi", "mov", "mkv"],
}

class StreamCleaner:
    """
    A memory-safe cleaner that supports streaming processing.
    It can clean text and multimedia files.
    """
    
    def __init__(self, rules=None, min_len=10, max_len=1024):
        """
        Initialize the StreamCleaner.

        Args:
            rules (list, optional): List of cleaning rules. If None, use all rules in RULES.
            min_len (int, optional): Minimum length of valid text. Defaults to 10.
            max_len (int, optional): Maximum length of valid text. Defaults to 1024.
        """
        self.rules = rules or list(RULES.values())
        self.min_len, self.max_len = min_len, max_len
        
    def clean_text(self, text: str) -> str:
        """
        Clean the input text using the defined rules.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text if its length is within the specified range, otherwise an empty string.
        """
        if not isinstance(text, str):
            return ""
        for rule in self.rules:
            text = rule(text)
        return text if self.min_len <= len(text) <= self.max_len else ""
    
    def clean_media(self, path: str, media_type: str) -> Optional[str]:
        """
        Clean and validate multimedia files.

        Args:
            path (str): Path to the multimedia file.
            media_type (str): Type of the multimedia file (e.g., "image", "audio", "video", "doc").

        Returns:
            Optional[str]: Path to the validated file if valid, None otherwise.
        """
        if not path or not isinstance(path, str):
            return None
            
        try:
            if media_type == "image":
                return MediaCleaner.clean_image(path)
            elif media_type == "audio":
                return MediaCleaner.clean_audio(path)
            elif media_type == "video":
                return MediaCleaner.clean_video(path)
            elif media_type == "doc":
                return MediaCleaner.clean_document(path)
            return path
        except Exception:
            return None

class MediaCleaner:
    """
    A general-purpose media cleaning utility class that supports cleaning and validation 
    of images, audio, documents, and videos.
    """
    
    @staticmethod
    def clean_image(image_path, min_size=(224, 224)):
        """
        Clean and validate image files.
        
        Args:
            image_path (str): Path to the image file.
            min_size (tuple): Minimum size requirement (width, height).
            
        Returns:
            str: Path to the validated image if valid, None otherwise.
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if img.size < min_size:
                    return None
                return image_path
        except Exception:
            return None

    @staticmethod
    def clean_audio(audio_path, min_duration=1.0, max_duration=30.0):
        """
        Clean and validate audio files.
        
        Args:
            audio_path (str): Path to the audio file.
            min_duration (float): Minimum duration in seconds.
            max_duration (float): Maximum duration in seconds.
            
        Returns:
            str: Path to the validated audio if valid, None otherwise.
        """
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            if duration < min_duration or duration > max_duration:
                return None
            return audio_path
        except Exception:
            return None

    @staticmethod
    def clean_video(video_path, min_duration=3, min_frames=8):
        """
        Clean and validate video files.
        
        Args:
            video_path (str): Path to the video file.
            min_duration (int): Minimum duration in seconds.
            min_frames (int): Minimum number of frames.
            
        Returns:
            str: Path to the validated video if valid, None otherwise.
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            if duration < min_duration or frame_count < min_frames:
                return None
                
            return video_path
        except Exception:
            return None

    @staticmethod
    def clean_document(doc_path, max_pages=50):
        """
        Clean and validate document files.
        
        Args:
            doc_path (str): Path to the document file (e.g., PDF, DOCX).
            max_pages (int): Maximum number of pages allowed.
            
        Returns:
            str: Path to the validated document if valid, None otherwise.
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(doc_path)
            if len(doc) > max_pages:
                return None
            doc.close()
            return doc_path
        except Exception:
            return None

class DatasetCleaner:
    @staticmethod
    def fast_clean(data_dir: str, max_len: int = 256):
        """
        Perform 30-second emergency data cleaning on JSON files in the specified directory.

        Args:
            data_dir (str): Directory containing JSON files to be cleaned.
            max_len (int, optional): Maximum length of valid text. Defaults to 256.
        """
        DEBUG("Performing emergency data cleaning...")
        
        def emergency_filter(batch):
            """
            Filter out overlong or empty samples from the batch.

            Args:
                batch (dict): A batch of data with a 'text' key.

            Returns:
                dict: Filtered batch of data.
            """
            mask = [len(str(x)) < max_len and len(str(x).strip()) > 0 for x in batch['text']]
            return {k: [v[i] for i,m in enumerate(mask) if m] for k,v in batch.items()}
        
        # Quickly clean cached data
        cache_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        for file in cache_files:
            file_path = os.path.join(data_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Apply emergency filter
                if isinstance(data, dict) and 'text' in data:
                    filtered = emergency_filter(data)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered, f, ensure_ascii=False, indent=2)
                
                RIGHT(f"Cleaning completed: {file}")
            except Exception as e:
                DEBUG(f"Skipping file {file}: {e}")
        
        RIGHT("Emergency data cleaning completed!")
    
    @staticmethod
    def clean_text(text, min_length=1):
        """
        Clean text data using basic text cleaning techniques.

        Args:
            text (str): Text to be cleaned.
            min_length (int): Minimum length of the cleaned text. If the length is less than this value, None will be returned. Default is 1.

        Returns:
            str: Cleaned text. Returns None if the text is invalid or the length after cleaning is less than min_length.
        """
        if not text or not isinstance(text, str):
            return None
            
        try:
            # Basic text cleaning instead of datacleaner
            import re
            # Remove control characters and normalize whitespace
            cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            
            if len(cleaned_text) >= min_length:
                return cleaned_text
            else:
                return None
        except Exception:
            return None

    @staticmethod
    def process_dataset(input_path, output_path, text_field='text', **clean_kwargs):
        """
        Process a single dataset and clean the text field in the dataset.

        Args:
            input_path (str): Path to the input dataset.
            output_path (str): Path to save the cleaned dataset.
            text_field (str): Name of the text field to be cleaned in the dataset. Default is 'text'.
            **clean_kwargs: Additional arguments passed to the clean_text method.

        Returns:
            tuple: A tuple containing the number of samples retained after cleaning and the total number of original samples.
        """
        return DatasetCleaner.process_dataset_core(
            input_path, output_path, text_field=text_field, **clean_kwargs
        )

    @staticmethod
    def process_dataset_core(input_path, output_path, text_field='text', **clean_kwargs):
        """
        Process datasets using datacleaner and dedicated text cleaning methods, supporting Arrow, JSON, and JSONL formats.
        
        Args:
            input_path (str): Path to the input dataset (supports .arrow directory, .json, .jsonl files).
            output_path (str): Path to save the cleaned dataset.
            text_field (str): Name of the text field to be cleaned. Default is 'text'. If None, auto-detect.
            **clean_kwargs: Additional arguments passed to cleaning methods.

        Returns:
            tuple: Tuple of (retained_samples, total_samples)
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Dataset path does not exist: {input_path}")

        # Create the parent directory of the output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load data based on file format
        original_size = 0
        if os.path.isdir(input_path):
            # Arrow format directory
            dataset = load_from_disk(input_path)
            df = dataset.to_pandas()
            original_size = len(dataset)
        elif input_path.endswith('.json'):
            # JSON format, excluding system files
            if os.path.basename(input_path) in ['dataset_info.json', 'state.json']:
                raise ValueError(f"Skipping system file: {input_path}")
            df = pd.read_json(input_path)
            original_size = len(df)
        elif input_path.endswith('.jsonl'):
            # JSONL format
            df = pd.read_json(input_path, lines=True)
            original_size = len(df)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        # Auto-detect text field if specified field not found
        from . import TEXT_FIELD_KEYS
        
        if text_field not in df.columns:
            detected_field = None
            for field in TEXT_FIELD_KEYS:
                if field in df.columns:
                    detected_field = field
                    break
            
            if detected_field:
                DEBUG(f"Text field '{text_field}' not found, using '{detected_field}' instead")
                text_field = detected_field
            else:
                # If no text field found, use the first string column
                string_cols = df.select_dtypes(include=['object']).columns
                if len(string_cols) > 0:
                    text_field = string_cols[0]
                    DEBUG(f"No standard text field found, using first string column '{text_field}'")
                else:
                    raise ValueError(f"No text field found in dataset. Available columns: {list(df.columns)}")

        # Handle special formats like conversations, messages, and code
        if text_field in ['conversations', 'messages', 'conversation', 'code']:
            def extract_text_from_dialogue(dialogue_list):
                """
                Perform universal text extraction for conversations, messages, and code formats.

                Args:
                    dialogue_list (list|str|dict): A list, string, or dictionary representing dialogue data.

                Returns:
                    str: Extracted text joined by newlines, or an empty string if no text is extracted.
                """
                # Handle None/NaN values
                if dialogue_list is None:
                    return ""
                
                # Handle pandas Series/Array-like NaN
                try:
                    if pd.isna(dialogue_list):
                        return ""
                except (ValueError, TypeError):
                    # Handle array-like NaN checks gracefully
                    if hasattr(dialogue_list, '__iter__') and not isinstance(dialogue_list, (str, dict, list)):
                        dialogue_list = str(dialogue_list)
                
                # Handle string input (fallback)
                if isinstance(dialogue_list, str):
                    # Try to parse as JSON first
                    try:
                        import json
                        parsed = json.loads(dialogue_list)
                        if isinstance(parsed, list):
                            dialogue_list = parsed
                        else:
                            return str(parsed).strip()
                    except json.JSONDecodeError:
                        # Handle escaped JSON strings
                        try:
                            dialogue_list = json.loads(dialogue_list.replace('\\"', '"'))
                        except:
                            return dialogue_list.strip()
                    except:
                        return dialogue_list.strip()
                
                if not isinstance(dialogue_list, list):
                    return str(dialogue_list).strip()
                
                full_text = []
                for turn in dialogue_list:
                    if turn is None:
                        continue
                    try:
                        if pd.isna(turn):
                            continue
                    except (ValueError, TypeError):
                        # Skip if turn is problematic
                        continue
                    
                    # Handle dict format conversation turns
                    if isinstance(turn, dict):
                        content = None
                        
                        # Check standard conversation formats
                        conversation_formats = [
                            (['human', 'user', 'Human', 'User'], ['assistant', 'bot', 'Assistant', 'Bot']),
                            (['input', 'prompt'], ['output', 'response']),
                            (['question'], ['answer']),
                            (['instruction'], ['output']),
                            (['text'], ['response']),
                            (['content'], ['response']),
                            (['from', 'role'], ['value', 'content'])  # Add OpenAI format
                        ]
                        
                        # Special handling for code datasets
                        if 'code' in turn and turn['code']:
                            code_content = str(turn['code']).strip()
                            if code_content:
                                full_text.append(code_content)
                                continue
                        
                        # Special handling for Chinese1 conversation format
                        # Handle nested conversation structure and Chinese-specific formats
                        extracted_content = []
                        for key, value in turn.items():
                            if isinstance(value, list):
                                nested_text = extract_text_from_dialogue(value)
                                if nested_text.strip():
                                    extracted_content.append(nested_text)
                            elif isinstance(value, str) and value.strip():
                                # Skip keys that are likely metadata
                                if key.lower() not in ['id', 'idx', 'index', 'role', 'from']:
                                    extracted_content.append(value.strip())
                        
                        if extracted_content:
                            full_text.extend(extracted_content)
                            continue
                        
                        # Standard conversation extraction
                        content = None
                        for human_keys, assistant_keys in conversation_formats:
                            for hk in human_keys:
                                if hk in turn and turn[hk] and str(turn[hk]).strip():
                                    content = str(turn[hk]).strip()
                                    break
                            if content:
                                break
                                
                            for ak in assistant_keys:
                                if ak in turn and turn[ak] and str(turn[ak]).strip():
                                    content = str(turn[ak]).strip()
                                    break
                            if content:
                                break
                        
                        # Fallback: extract any non-empty key
                        if not content:
                            for key, value in turn.items():
                                if value and str(value).strip() and key.lower() not in ['id', 'idx', 'index']:
                                    content = str(value).strip()
                                    break
                        
                        if content:
                            full_text.append(content)
                    
                    # Handle string format conversation turns
                    elif isinstance(turn, str):
                        content = turn.strip()
                        if content:
                            full_text.append(content)
                    
                    # Handle nested list format conversations
                    elif isinstance(turn, list):
                        nested_text = extract_text_from_dialogue(turn)
                        if nested_text.strip():
                            full_text.append(nested_text)
                    
                    # Handle other types (bool, numbers, etc)
                    else:
                        text = str(turn).strip()
                        if text and text.lower() not in ['none', 'null', '']:
                            full_text.append(text)
                
                return "\n".join(full_text) if full_text else ""
            
            # Create a temporary text column from conversations/messages
            temp_text_field = 'extracted_text'
            df[temp_text_field] = df[text_field].apply(extract_text_from_dialogue)
            
            text_field = temp_text_field

        # Handle structured data issues (missing values, outliers, etc.)
        # Note: datacleaner package is deprecated/unavailable, using basic pandas operations
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Basic data cleaning for numeric columns
        for col in numeric_cols:
            if col != text_field:
                # Remove outliers using IQR method for numeric columns
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Basic cleaning for categorical columns
        for col in categorical_cols:
            if col != text_field:
                # Skip non-string/object columns to avoid array comparison issues
                if df[col].dtype == 'object' or str(df[col].dtype).startswith('string'):
                    try:
                        # Remove empty strings and None values
                        df[col] = df[col].replace('', None)
                        # Trim whitespace
                        df[col] = df[col].str.strip() if hasattr(df[col], 'str') else df[col]
                    except (ValueError, TypeError):
                        # Handle cases where replace fails due to array-like data
                        mask = df[col].apply(lambda x: isinstance(x, str) and x.strip() == '')
                        df.loc[mask, col] = None
        
        # Dedicated text cleaning
        import re
        def clean_text_content(text):
            """
            Clean text content by removing control characters and normalizing whitespace.

            Args:
                text (str): The text to be cleaned.

            Returns:
                str: The cleaned text, or an empty string if the input is not a valid string.
            """
            if not isinstance(text, str):
                return ""
            text = str(text).strip()
            if not text:
                return ""
            
            # Remove control characters
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        df[text_field] = df[text_field].apply(clean_text_content)
        
        # Filter short text
        min_length = clean_kwargs.get('min_length', 1)
        
        mask = df[text_field].astype(str).str.strip().str.len() >= min_length
        df = df[mask]
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Convert back to Dataset
        cleaned_dataset = Dataset.from_pandas(df, preserve_index=False)
        cleaned_dataset.save_to_disk(output_path)
        
        return len(df), original_size

    @staticmethod
    def clean_video(video_path, min_duration=3):
        """
        Clean and validate video file.
        
        Args:
            video_path (str): Path to the video file
            min_duration (int): Minimum video duration in seconds
            
        Returns:
            str: Validated video path or None if invalid
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            if duration < min_duration:
                return None
                
            return video_path
        except Exception:
            return None

    @staticmethod
    def find_field(item: dict, candidates: List[str]) -> Optional[str]:
        """
        Find the first existing multimodal field in the item based on the candidate field names.

        Args:
            item (dict): A dictionary representing a data item.
            candidates (List[str]): List of candidate field names.

        Returns:
            Optional[str]: The value of the first existing field, or None if none is found.
        """
        return next((item[k] for k in candidates if k in item and item[k]), None)

    @staticmethod
    def worker(args: Tuple[str, List[Callable], int, int]) -> Optional[Dataset]:
        """
        Multiprocessing worker function to clean a single dataset.

        Args:
            args (Tuple[str, List[Callable], int, int]): A tuple containing dataset path, cleaning rules, 
                                                         minimum text length, and maximum text length.

        Returns:
            Optional[Dataset]: The cleaned dataset if successful, None otherwise.
        """
        ds_path, rules, min_len, max_len = args
        
        try:
            ds = load_from_disk(ds_path)
            cleaner = StreamCleaner(rules, min_len, max_len)
            
            # Streaming chunk processing, trigger GC every 10k records
            chunk_size = 10000
            cleaned_chunks = []
            
            for start in range(0, len(ds), chunk_size):
                end = min(start + chunk_size, len(ds))
                chunk = ds.select(range(start, end))
                df = chunk.to_pandas()
                
                # Text cleaning
                if 'text' in df.columns:
                    df['text'] = df['text'].apply(cleaner.clean_text)
                    df = df[df['text'].str.strip() != '']
                
                # Automatic detection and cleaning of multimodal fields
                for media_type, field_names in AUTO_FIELDS.items():
                    col = next((c for c in field_names if c in df.columns), None)
                    if col and col in df.columns:
                        df[col] = df[col].apply(lambda x: cleaner.clean_media(str(x), media_type) if pd.notna(x) else None)
                
                if len(df) > 0:
                    cleaned_chunks.append(Dataset.from_pandas(df))
                
                # Force memory recycling
                del df, chunk
                gc.collect()
            
            if cleaned_chunks:
                merged = concatenate_datasets(cleaned_chunks)
                return merged
            return None
            
        except Exception as e:
            ERROR(f"Failed to process dataset {ds_path}: {str(e)}")
            return None

    @staticmethod
    def merge_and_clean(
        input_dir="data_cache",
        output_dir=None,
        min_len=1,
        max_len=1024,
        workers=None,
        rules=None
    ) -> Optional[Dataset]:
        """
        Perform one-click merge and clean on datasets in the input directory, supporting multiprocessing streaming processing.

        Args:
            input_dir (str, optional): Directory containing datasets to be cleaned. Defaults to "data_cache".
            output_dir (str, optional): Directory to save the merged and cleaned dataset. Defaults to None.
            min_len (int, optional): Minimum length of valid text. Defaults to 1.
            max_len (int, optional): Maximum length of valid text. Defaults to 1024.
            workers (int, optional): Number of worker processes. Defaults to the minimum of 4 and CPU count.
            rules (List[Callable], optional): List of cleaning rules. Defaults to all rules in RULES.

        Returns:
            Optional[Dataset]: The merged and cleaned dataset if successful, None otherwise.
        """
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        workers = workers or min(4, mp.cpu_count())
        rules = rules or list(RULES.values())
        
        # Find datasets to be cleaned
        raw_paths = [
            os.path.join(input_dir, d)
            for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        ]
        
        if not raw_paths:
            DEBUG("No datasets to be cleaned found")
            return None
        
        DEBUG(f"Found {len(raw_paths)} datasets, starting {workers} processes for cleaning...")
        
        # Multiprocessing cleaning
        with mp.Pool(workers) as pool:
            results = list(
                pool.imap(
                    DatasetCleaner.worker,
                    [(p, rules, min_len, max_len) for p in raw_paths]
                )
            )
        
        # Merge valid results
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return None
        
        RIGHT(f"Merging {len(valid_results)} cleaned datasets...")
        merged = concatenate_datasets(valid_results)
        
        # Remove source column to avoid metadata
        if "source" in merged.column_names:
            merged = merged.remove_columns(["source"])
        
        RIGHT(f"Cleaning and merging completed! Total {len(merged)} records → {output_dir}")
        return merged

    @staticmethod
    def is_download_complete(dataset_path):
        """
        Check if dataset download is complete by looking for download completion markers.
        
        Args:
            dataset_path (str): Path to the dataset directory

        Returns:
            bool: True if download is complete, False otherwise
        """
        # Check for common download completion indicators
        completion_markers = [
            '.download_complete',
            '.finished',
            'download_status.txt',
            'completed.flag'
        ]
        
        # Check for completion markers
        for marker in completion_markers:
            if os.path.exists(os.path.join(dataset_path, marker)):
                return True

        # Check if dataset has expected structure
        try:
            dataset = load_from_disk(dataset_path)
            # Check if dataset has data
            if len(dataset) > 0:
                # Check if all files are present and accessible
                for sample in dataset.take(1):
                    pass
                return True
        except Exception:
            return False
        
        return False

    @staticmethod
    def _process_single_dataset_wrapper(args):
        """
        Wrapper for multiprocessing - unpack arguments and call _process_single_dataset.

        Args:
            args (tuple): A tuple containing dataset name, input path, output path, media fields, and cleaning arguments.

        Returns:
            tuple: A tuple of (cleaned_count, total_count) returned by _process_single_dataset.
        """
        dataset_name, input_path, output_path, media_fields, clean_kwargs = args
        return DatasetCleaner._process_single_dataset(
            dataset_name, input_path, output_path, media_fields, **clean_kwargs
        )

    @staticmethod
    def _process_single_dataset(dataset_name, input_path, output_path, media_fields=None, **clean_kwargs):
        """
        Process a single dataset - extracted from auto_clean for multiprocessing support.
        
        Returns:
            tuple: (cleaned_count, total_count)
        """
        try:
            # Validate dataset exists and is accessible
            if not os.path.exists(input_path):
                DEBUG(f"Dataset path does not exist: {input_path}")
                return (0, 0)
            
            # Check if dataset is empty
            try:
                dataset = load_from_disk(input_path)
                if len(dataset) == 0:
                    DEBUG(f"Dataset {dataset_name} is empty, skipping processing")
                    return (0, 0)
                DEBUG(f"Starting to process dataset: {dataset_name} (Total {len(dataset)} records)")
            except Exception as e:
                ERROR(f"Failed to load dataset {dataset_name}: {str(e)}")
                return (0, 0)
            
            # Process the dataset and get the cleaning results
            if media_fields:
                cleaned_count, total_count = DatasetCleaner.process_multimodal_dataset(
                    input_path, output_path, media_fields=media_fields, **clean_kwargs
                )
            else:
                cleaned_count, total_count = DatasetCleaner.process_dataset(
                    input_path, output_path, **clean_kwargs
                )
            
            return (cleaned_count, total_count)
            
        except Exception as e:
            ERROR(f"Error cleaning {dataset_name}: {str(e)}")
            import traceback
            ERROR(f"Detailed error information: {traceback.format_exc()}")
            return (0, 0)

    @staticmethod
    def fast_clean(
        input_dir="data_cache",
        output_dir=None,
        min_len=1,
        max_len=1024,
        workers=None,
        enable_multiprocessing=True
    ):
        """
        Fast cleaning mode - complete all cleaning steps in one click.
        
        Args:
            input_dir: Input dataset directory.
            output_dir: Output directory for cleaned datasets (None to avoid creation).
            min_len: Minimum text length.
            max_len: Maximum text length.
            workers: Number of processes, default is the number of CPU cores.
            enable_multiprocessing: Whether to enable multiprocessing.
        """
        try:
            if enable_multiprocessing:
                return DatasetCleaner.merge_and_clean(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    min_len=min_len,
                    max_len=max_len,
                    workers=workers
                )
            else:
                # Backward-compatible single-process mode
                return DatasetCleaner.auto_clean(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    min_length=min_len
                )
        except Exception as e:
            ERROR(f"Fast cleaning failed: {str(e)}")
            return None

    @staticmethod
    def auto_clean(input_dir, output_dir='data_clean', media_fields=None, workers=None, **clean_kwargs):
        """
        Automatically clean all datasets in the specified directory.
        
        Only processes datasets that are completely downloaded.

        Args:
            input_dir (str): Input directory containing multiple datasets.
            output_dir (str): Output directory for the cleaned datasets. Default is 'data_clean'.
            media_fields (dict, optional): Dict mapping field names to cleaning functions for multimodal data.
                           Example: {'image_path': MediaCleaner.clean_image, 'audio_path': MediaCleaner.clean_audio}
            workers (int, optional): Number of worker processes for parallel processing. Default is CPU count.
            **clean_kwargs: Additional arguments passed to the clean_text method.

        Returns:
            bool: Returns True after cleaning is completed.
        """
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        datasets_to_clean = []
        
        for dataset_name in os.listdir(input_dir):
            input_path = os.path.join(input_dir, dataset_name)
            if os.path.isdir(input_path) and not dataset_name.endswith('_clean'):
                
                # Check if download is complete
                if not DatasetCleaner.is_download_complete(input_path):
                    DEBUG(f"Dataset {dataset_name} download not complete, skipping...")
                    continue
                
                output_path = os.path.join(output_dir, f"{dataset_name}_clean")
                if not os.path.exists(output_path):
                    datasets_to_clean.append((dataset_name, input_path, output_path))
                else:
                    RIGHT(f"Cleaned dataset already exists: {output_path}, skipping processing")
        
        # Process datasets that are ready
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        workers = workers or min(4, mp.cpu_count())
        
        if workers <= 1 or len(datasets_to_clean) <= 1:
            # Single process mode
            for dataset_name, input_path, output_path in datasets_to_clean:
                DatasetCleaner._process_single_dataset(
                    dataset_name, input_path, output_path, media_fields, **clean_kwargs
                )
        else:
            # Multi-process mode
            DEBUG(f"Using {workers} processes to clean {len(datasets_to_clean)} datasets...")
            
            # Prepare arguments for parallel processing
            process_args = [
                (dataset_name, input_path, output_path, media_fields, clean_kwargs)
                for dataset_name, input_path, output_path in datasets_to_clean
            ]
            
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks
                future_to_dataset = {
                    executor.submit(DatasetCleaner._process_single_dataset_wrapper, args): args[0]
                    for args in process_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_dataset):
                    dataset_name = future_to_dataset[future]
                    try:
                        cleaned_count, total_count = future.result()
                        if cleaned_count == 0:
                            DEBUG(f"Warning: No valid data left after cleaning {dataset_name} (Original {total_count} samples)")
                        else:
                            RIGHT(f"Cleaning completed: {dataset_name} -> {dataset_name}_clean | Samples retained: {cleaned_count}/{total_count}")
                    except Exception as e:
                        ERROR(f"Error cleaning {dataset_name}: {str(e)}")
        
        return True