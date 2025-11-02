#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Callable, Dict, Optional, Any, List

# Regular expression rules for text cleaning, mapping rule names to corresponding cleaning functions.
RULES: Dict[str, Callable[[str], str]] = {
    # Remove control characters from the input string.
    "ctrl_chars": lambda x: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", x),
    # Replace multiple whitespace characters with a single space and trim leading and trailing whitespace.
    "whitespace": lambda x: re.sub(r"\s+", " ", x).strip(),
    # Remove URLs starting with 'http' or 'www' from the input string.
    "urls":       lambda x: re.sub(r"http\S+|www\S+", "", x),
    # Remove email addresses from the input string.
    "emails":     lambda x: re.sub(r"\S+@\S+", "", x),
    # Remove HTML tags from the input string.
    "html_tags":  lambda x: re.sub(r"<[^>]*?>", "", x),
    # Remove non-Chinese characters, non-word characters, and non-whitespace characters except common punctuation from the input string.
    "emoji":      lambda x: re.sub(r"[^\u4e00-\u9fff\w\s.,!?;:]", "", x),
    # Remove special characters except common punctuation, Chinese characters, word characters, and whitespace from the input string.
    "special_chars": lambda x: re.sub(r"[^\w\s\u4e00-\u9fff.,!?;:()\"'-]", "", x),
    # Replace consecutive punctuation marks with a single period.
    "extra_punct": lambda x: re.sub(r"[.,!?;:]{2,}", ".", x),
    # Return an empty string if the input string contains only digits, otherwise return the original string.
    "digits_only": lambda x: x if not str(x).strip().isdigit() else "",
    # Return an empty string if the input string has only one character after trimming, otherwise return the original string.
    "single_chars": lambda x: x if len(str(x).strip()) > 1 else "",
}

# Auto-detection fields for different media types, mapping media types to candidate column names.
AUTO_FIELDS = {
    "image": ["image", "img_path", "image_path", "picture", "pic", "img"],
    "audio": ["audio", "audio_path", "wav", "sound", "mp3"],
    "doc":   ["doc", "document", "doc_path", "pdf", "file_path"],
    "video": ["video", "video_path", "mp4", "avi", "mov", "mkv"],
}

class PiscesLxToolsDataStreamCleaner:
    """
    A class for cleaning text and media data, supporting text cleaning with specified rules and media quality control.
    """

    def __init__(self, rules: Optional[List[Callable[[str], str]]] = None, min_len: int = 5, max_len: int = 512):
        """
        Initialize the StreamCleaner instance.

        Args:
            rules (Optional[List[Callable[[str], str]]]): A list of text cleaning functions. If None, use the default RULES.
            min_len (int): The minimum length of the cleaned text. Defaults to 5.
            max_len (int): The maximum length of the cleaned text. Defaults to 512.
        """
        self.rules = rules or list(RULES.values())
        self.min_len, self.max_len = min_len, max_len

    def clean_text(self, text: Any) -> str:
        """
        Clean the input text using the specified rules and filter by length.

        Args:
            text (Any): The input text to be cleaned.

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
        Clean media files based on their type.

        Args:
            path (str): The path to the media file.
            media_type (str): The type of the media file, e.g., 'image', 'audio', 'video', 'doc'.

        Returns:
            Optional[str]: The cleaned media path if successful, otherwise None.
        """
        if not path or not isinstance(path, str):
            return None
        try:
            from .media import MediaCleaner
            if media_type == "image":
                return MediaCleaner.clean_image_with_quality(path)
            if media_type == "audio":
                return MediaCleaner.clean_audio_with_quality(path)
            if media_type == "video":
                return MediaCleaner.clean_video_with_quality(path)
            if media_type == "doc":
                return MediaCleaner.clean_document_with_quality(path)
            return path
        except Exception as e:
            pass
            return None

    @staticmethod
    def get_media_quality_score(media_path: str, media_type: str) -> float:
        """
        Calculate the quality score of a media file based on its type.

        Args:
            media_path (str): The path to the media file.
            media_type (str): The type of the media file, e.g., 'image', 'audio', 'video', 'doc'.

        Returns:
            float: The quality score of the media file. Returns 0.5 if an error occurs.
        """
        try:
            from .media import MediaCleaner
            if media_type == "image":
                from PIL import Image
                with Image.open(media_path) as img:
                    return MediaCleaner._calculate_image_quality(img)
            if media_type == "audio":
                return MediaCleaner._calculate_audio_quality(media_path)
            if media_type == "video":
                import cv2
                cap = cv2.VideoCapture(media_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                score = MediaCleaner._calculate_video_quality(cap, total)
                cap.release()
                return score
            if media_type == "doc":
                import fitz
                doc = fitz.open(media_path)
                score = MediaCleaner._calculate_document_quality(doc)
                doc.close()
                return score
            return 0.5
        except Exception as e:
            pass
            return 0.5

    @staticmethod
    def find_multimodal_fields_from_dataset(hf_dataset) -> Dict[str, str]:
        """
        Automatically discover multimodal fields from a Hugging Face dataset based on column names.

        Args:
            hf_dataset: A Hugging Face dataset object.

        Returns:
            Dict[str, str]: A mapping of column names to media types.
        """
        mapping: Dict[str, str] = {}
        try:
            cols = list(hf_dataset.column_names)
            for media_type, candidates in AUTO_FIELDS.items():
                for c in candidates:
                    if c in cols:
                        mapping[c] = media_type
                        break
        except Exception:
            pass
        return mapping
