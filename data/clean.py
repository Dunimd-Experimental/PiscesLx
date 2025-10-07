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
import re
import gc
import json
import math
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import Counter
from utils import PiscesLxCoreLog, PiscesLxCoreCacheManagerFacade
from typing import Dict, Callable, List, Optional, Tuple, Any
from datasets import load_from_disk, Dataset, concatenate_datasets

# ========== Plug-in Rule System ==========
# Initialize logger for this module
_log = PiscesLxCoreLog("PiscesLx.DataClean")

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
    
    def __init__(self, rules=None, min_len=5, max_len=512):
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
        Clean and validate multimedia files with enhanced quality checks.

        Args:
            path (str): Path to the multimedia file.
            media_type (str): Type of the multimedia file (e.g., "image", "audio", "video", "doc").

        Returns:
            Optional[str]: Path to the validated file if valid, None otherwise.
        """
        if not path or not isinstance(path, str):
            return None
            
        try:
            # Enhanced media validation with quality scoring
            if media_type == "image":
                return MediaCleaner.clean_image_with_quality(path)
            elif media_type == "audio":
                return MediaCleaner.clean_audio_with_quality(path)
            elif media_type == "video":
                return MediaCleaner.clean_video_with_quality(path)
            elif media_type == "doc":
                return MediaCleaner.clean_document_with_quality(path)
            return path
        except Exception as e:
            _log.debug(f"Media cleaning failed for {path}: {str(e)}")
            return None

    @staticmethod
    def get_media_quality_score(media_path: str, media_type: str) -> float:
        """
        Get quality score for a media file.
        
        Args:
            media_path (str): Path to the media file
            media_type (str): Type of media (image, audio, video, doc)
            
        Returns:
            float: Quality score between 0 and 1
        """
        try:
            if media_type == "image":
                from PIL import Image
                with Image.open(media_path) as img:
                    return MediaCleaner._calculate_image_quality(img)
            elif media_type == "audio":
                return MediaCleaner._calculate_audio_quality(media_path)
            elif media_type == "video":
                import cv2
                cap = cv2.VideoCapture(media_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                quality = MediaCleaner._calculate_video_quality(cap, total_frames)
                cap.release()
                return quality
            elif media_type == "doc":
                import fitz
                doc = fitz.open(media_path)
                quality = MediaCleaner._calculate_document_quality(doc)
                doc.close()
                return quality
            else:
                return 0.5  # Default score for unknown types
        except Exception as e:
            _log.debug(f"Quality score calculation failed for {media_path}: {str(e)}")
            return 0.5  # Default medium quality

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
    def clean_image_with_quality(image_path, min_size=(224, 224), min_quality_score=0.6):
        """
        Clean and validate image files with advanced quality scoring.
        
        Args:
            image_path (str): Path to the image file.
            min_size (tuple): Minimum size requirement (width, height).
            min_quality_score (float): Minimum quality score (0-1).
            
        Returns:
            str: Path to the validated image if valid, None otherwise.
        """
        try:
            from PIL import Image
            import numpy as np
            
            with Image.open(image_path) as img:
                # Basic validation
                img = img.convert("RGB")
                if img.size < min_size:
                    return None
                
                # Advanced quality scoring
                quality_score = MediaCleaner._calculate_image_quality(img)
                if quality_score < min_quality_score:
                    return None
                
                return image_path
        except Exception as e:
            _logger.debug(f"Image quality cleaning failed for {image_path}: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_image_quality(img):
        """
        Calculate comprehensive image quality score.
        
        Args:
            img (PIL.Image): Image object
            
        Returns:
            float: Quality score (0-1)
        """
        try:
            import numpy as np
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Resolution score
            resolution_score = min((img.width * img.height) / (1024 * 1024), 1.0)  # Normalize to 1MP
            
            # Sharpness score (using Laplacian variance)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
            sharpness_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000, 1.0)
            
            # Contrast score
            contrast_score = min(gray.std() / 128, 1.0)
            
            # Brightness score (avoid over/under exposure)
            mean_brightness = gray.mean()
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            
            # Combine scores
            quality_score = (
                resolution_score * 0.3 +
                sharpness_score * 0.3 +
                contrast_score * 0.2 +
                brightness_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5  # Default medium quality

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
    def clean_audio_with_quality(audio_path, min_duration=1.0, max_duration=30.0, min_quality_score=0.5):
        """
        Clean and validate audio files with advanced quality scoring.
        
        Args:
            audio_path (str): Path to the audio file.
            min_duration (float): Minimum duration in seconds.
            max_duration (float): Maximum duration in seconds.
            min_quality_score (float): Minimum quality score (0-1).
            
        Returns:
            str: Path to the validated audio if valid, None otherwise.
        """
        try:
            import librosa
            import numpy as np
            
            # Basic duration check
            y, sr = librosa.load(audio_path, sr=None, duration=max_duration + 1)
            duration = len(y) / sr
            
            if duration < min_duration or duration > max_duration:
                return None
            
            # Advanced quality scoring
            quality_score = MediaCleaner._calculate_audio_quality(y, sr)
            if quality_score < min_quality_score:
                return None
            
            return audio_path
        except Exception as e:
            _logger.debug(f"Audio quality cleaning failed for {audio_path}: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_audio_quality(y, sr):
        """
        Calculate comprehensive audio quality score.
        
        Args:
            y (np.array): Audio signal
            sr (int): Sample rate
            
        Returns:
            float: Quality score (0-1)
        """
        try:
            import numpy as np
            
            # Duration score
            duration = len(y) / sr
            duration_score = min(duration / 10.0, 1.0)  # Normalize to 10 seconds
            
            # Dynamic range score
            dynamic_range = np.max(y) - np.min(y)
            dynamic_score = min(dynamic_range / 0.5, 1.0)  # Normalize to 0.5 range
            
            # RMS energy score (avoid silence)
            rms_energy = np.sqrt(np.mean(y**2))
            energy_score = min(rms_energy / 0.1, 1.0)  # Normalize to 0.1 RMS
            
            # Zero crossing rate (avoid excessive noise)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y)[0])
            noise_score = 1.0 - min(zcr / 0.1, 1.0)  # Penalize high ZCR
            
            # Combine scores
            quality_score = (
                duration_score * 0.3 +
                dynamic_score * 0.3 +
                energy_score * 0.2 +
                noise_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5  # Default medium quality

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
    def clean_video_with_quality(video_path, min_duration=3, min_frames=8, min_quality_score=0.5):
        """
        Clean and validate video files with advanced quality scoring.
        
        Args:
            video_path (str): Path to the video file.
            min_duration (int): Minimum duration in seconds.
            min_frames (int): Minimum number of frames.
            min_quality_score (float): Minimum quality score (0-1).
            
        Returns:
            str: Path to the validated video if valid, None otherwise.
        """
        try:
            import cv2
            import numpy as np
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Basic validation
            if duration < min_duration or frame_count < min_frames:
                cap.release()
                return None
            
            # Advanced quality scoring (sample frames)
            quality_score = MediaCleaner._calculate_video_quality(cap, frame_count)
            cap.release()
            
            if quality_score < min_quality_score:
                return None
                
            return video_path
        except Exception as e:
            _logger.debug(f"Video quality cleaning failed for {video_path}: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_video_quality(cap, total_frames):
        """
        Calculate comprehensive video quality score by sampling frames.
        
        Args:
            cap (cv2.VideoCapture): Video capture object
            total_frames (int): Total number of frames
            
        Returns:
            float: Quality score (0-1)
        """
        try:
            import numpy as np
            
            # Sample frames for quality assessment
            sample_indices = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)
            frame_scores = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert frame to PIL Image for quality calculation
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    frame_score = MediaCleaner._calculate_image_quality(pil_frame)
                    frame_scores.append(frame_score)
            
            if not frame_scores:
                return 0.0
            
            # Consistency score (low variance is good)
            avg_score = np.mean(frame_scores)
            consistency_score = 1.0 - min(np.std(frame_scores) / avg_score if avg_score > 0 else 0, 0.5)
            
            # Duration score
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            duration_score = min(duration / 30.0, 1.0)  # Normalize to 30 seconds
            
            # Combine scores
            quality_score = (
                avg_score * 0.5 +
                consistency_score * 0.3 +
                duration_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5  # Default medium quality

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

    @staticmethod
    def clean_document_with_quality(doc_path, max_pages=50, min_quality_score=0.3):
        """
        Clean and validate document files with advanced quality scoring.
        
        Args:
            doc_path (str): Path to the document file.
            max_pages (int): Maximum allowed pages.
            min_quality_score (float): Minimum quality score (0-1).
            
        Returns:
            str: Path to the validated document if valid, None otherwise.
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(doc_path)
            page_count = len(doc)
            
            # Basic validation
            if page_count > max_pages:
                doc.close()
                return None
            
            # Advanced quality scoring
            quality_score = MediaCleaner._calculate_document_quality(doc)
            doc.close()
            
            if quality_score < min_quality_score:
                return None
                
            return doc_path
        except Exception as e:
            _logger.debug(f"Document quality cleaning failed for {doc_path}: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_document_quality(doc):
        """
        Calculate comprehensive document quality score.
        
        Args:
            doc (fitz.Document): PyMuPDF document object
            
        Returns:
            float: Quality score (0-1)
        """
        try:
            page_count = len(doc)
            
            if page_count == 0:
                return 0.0
            
            # Page count score (normalize to reasonable range)
            page_score = min(page_count / 10.0, 1.0)  # 10 pages = perfect score
            
            # Text content analysis
            total_text_length = 0
            valid_pages = 0
            
            for page_num in range(min(page_count, 5)):  # Sample first 5 pages
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text and len(text.strip()) > 10:  # At least 10 characters
                    total_text_length += len(text.strip())
                    valid_pages += 1
            
            # Content density score
            avg_text_length = total_text_length / max(valid_pages, 1)
            content_score = min(avg_text_length / 1000.0, 1.0)  # 1000 chars = perfect score
            
            # Structure score (check for basic document structure)
            structure_score = valid_pages / min(page_count, 5)
            
            # Combine scores
            quality_score = (
                page_score * 0.2 +
                content_score * 0.6 +
                structure_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5  # Default medium quality

class DataQualityController:
    """
    Advanced data quality controller with intelligent filtering and dynamic thresholds.
    """
    
    def __init__(self, quality_threshold=0.7, diversity_threshold=0.5, min_samples_per_domain=100):
        """
        Initialize quality controller with configurable thresholds.
        
        Args:
            quality_threshold (float): Minimum quality score for data acceptance
            diversity_threshold (float): Minimum diversity score for dataset balance
            min_samples_per_domain (int): Minimum samples required per domain
        """
        self.quality_threshold = quality_threshold
        self.diversity_threshold = diversity_threshold
        self.min_samples_per_domain = min_samples_per_domain
        self.quality_stats = {}
        self.domain_weights = {}
        
    def analyze_dataset_quality(self, dataset_path):
        """
        Comprehensive dataset quality analysis with multi-dimensional metrics.
        
        Args:
            dataset_path (str): Path to dataset
            
        Returns:
            dict: Comprehensive quality analysis results
        """
        try:
            if not os.path.exists(dataset_path):
                return {"error": "Dataset path does not exist"}
            
            # Load dataset
            if os.path.isdir(dataset_path):
                # Arrow format directory
                dataset = load_from_disk(dataset_path)
                df = dataset.to_pandas()
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
            elif dataset_path.endswith('.jsonl'):
                df = pd.read_json(dataset_path, lines=True)
            elif dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            else:
                return {"error": "Unsupported file format. Supported formats: .arrow (directory), .json, .jsonl, .csv, .parquet"}
            
            total_samples = len(df)
            if total_samples == 0:
                return {"error": "Empty dataset"}
            
            # Text field detection
            text_field = None
            for field in TEXT_FIELD_KEYS:
                if field in df.columns:
                    text_field = field
                    break
            
            if not text_field:
                # Use first string column
                string_cols = df.select_dtypes(include=['object']).columns
                if len(string_cols) > 0:
                    text_field = string_cols[0]
                else:
                    return {"error": "No text field found"}
            
            # Quality analysis
            text_data = df[text_field].astype(str)
            
            # Basic statistics
            text_lengths = text_data.str.len()
            quality_scores = text_data.apply(DatasetCleaner.calculate_text_quality_score)
            
            # Advanced metrics
            char_diversity = text_data.apply(lambda x: len(set(x.lower())) / len(x) if x else 0)
            word_counts = text_data.apply(lambda x: len(re.findall(r'\b\w+\b', x.lower())))
            sentence_counts = text_data.apply(lambda x: len(re.split(r'[.!?]+', x)))
            
            # Domain detection (simplified)
            domain_keywords = {
                'code': ['function', 'class', 'def', 'import', 'return', '{', '}', ';'],
                'math': ['equation', 'formula', 'calculate', 'solve', 'math', 'algebra'],
                'science': ['experiment', 'theory', 'research', 'study', 'analysis'],
                'medical': ['patient', 'treatment', 'diagnosis', 'symptom', 'medicine'],
                'finance': ['investment', 'market', 'stock', 'trading', 'financial']
            }
            
            domain_scores = {}
            for domain, keywords in domain_keywords.items():
                keyword_matches = text_data.apply(
                    lambda x: sum(1 for keyword in keywords if keyword.lower() in x.lower()) / len(keywords)
                )
                domain_scores[domain] = keyword_matches.mean()
            
            # Quality classification
            high_quality = (quality_scores >= self.quality_threshold).sum()
            medium_quality = ((quality_scores >= 0.5) & (quality_scores < self.quality_threshold)).sum()
            low_quality = (quality_scores < 0.5).sum()
            
            stats = {
                'total_samples': total_samples,
                'text_field': text_field,
                'avg_text_length': text_lengths.mean(),
                'median_text_length': text_lengths.median(),
                'std_text_length': text_lengths.std(),
                'avg_quality_score': quality_scores.mean(),
                'median_quality_score': quality_scores.median(),
                'quality_score_std': quality_scores.std(),
                'avg_char_diversity': char_diversity.mean(),
                'avg_word_count': word_counts.mean(),
                'avg_sentence_count': sentence_counts.mean(),
                'high_quality_samples': high_quality,
                'medium_quality_samples': medium_quality,
                'low_quality_samples': low_quality,
                'quality_distribution': {
                    'high': high_quality / total_samples,
                    'medium': medium_quality / total_samples,
                    'low': low_quality / total_samples
                },
                'domain_scores': domain_scores,
                'recommended_action': self._get_recommendation(quality_scores.mean(), high_quality / total_samples)
            }
            
            self.quality_stats[dataset_path] = stats
            return stats
            
        except Exception as e:
            _logger.error(f"Quality analysis failed for {dataset_path}: {str(e)}")
            return {"error": str(e)}
    
    def _get_recommendation(self, avg_quality, high_quality_ratio):
        """
        Generate recommendations based on quality metrics.
        
        Args:
            avg_quality (float): Average quality score
            high_quality_ratio (float): Ratio of high quality samples
            
        Returns:
            str: Recommendation message
        """
        if avg_quality >= 0.8 and high_quality_ratio >= 0.7:
            return "Excellent quality - suitable for training"
        elif avg_quality >= 0.6 and high_quality_ratio >= 0.5:
            return "Good quality - minor filtering recommended"
        elif avg_quality >= 0.4 and high_quality_ratio >= 0.3:
            return "Moderate quality - significant filtering required"
        else:
            return "Poor quality - major cleanup or replacement needed"
    
    def optimize_data_mixing(self, dataset_paths, target_distribution=None):
        """
        Optimize dataset mixing ratios for balanced training data.
        
        Args:
            dataset_paths (list): List of dataset paths
            target_distribution (dict): Target domain distribution
            
        Returns:
            dict: Optimized mixing ratios and recommendations
        """
        if target_distribution is None:
            target_distribution = {
                'general': 0.4,
                'code': 0.2,
                'math': 0.15,
                'science': 0.1,
                'medical': 0.08,
                'finance': 0.07
            }
        
        # Analyze each dataset
        dataset_stats = {}
        for path in dataset_paths:
            stats = self.analyze_dataset_quality(path)
            if 'error' not in stats:
                dataset_stats[path] = stats
        
        if not dataset_stats:
            return {"error": "No valid datasets for analysis"}
        
        # Calculate optimal mixing ratios
        total_samples = sum(stats['total_samples'] for stats in dataset_stats.values())
        mixing_ratios = {}
        
        for path, stats in dataset_stats.items():
            base_ratio = stats['total_samples'] / total_samples
            
            # Quality adjustment
            quality_factor = stats['quality_distribution']['high']
            if stats['avg_quality_score'] < self.quality_threshold:
                quality_factor *= 0.8
            
            # Domain alignment
            domain_alignment = 0.0
            for domain, score in stats['domain_scores'].items():
                if domain in target_distribution:
                    domain_alignment += score * target_distribution[domain]
            
            # Final ratio
            adjusted_ratio = base_ratio * quality_factor * (0.5 + domain_alignment)
            mixing_ratios[path] = {
                'original_ratio': base_ratio,
                'quality_adjusted_ratio': base_ratio * quality_factor,
                'final_ratio': adjusted_ratio,
                'sample_count': int(adjusted_ratio * 50000),  # Assume 50k total samples
                'quality_score': stats['avg_quality_score'],
                'domain_scores': stats['domain_scores']
            }
        
        # Normalize ratios
        total_adjusted = sum(ratio['final_ratio'] for ratio in mixing_ratios.values())
        for path in mixing_ratios:
            mixing_ratios[path]['normalized_ratio'] = mixing_ratios[path]['final_ratio'] / total_adjusted
        
        return {
            'mixing_ratios': mixing_ratios,
            'total_samples': 50000,
            'recommendations': self._generate_mixing_recommendations(mixing_ratios)
        }
    
    def _generate_mixing_recommendations(self, mixing_ratios):
        """
        Generate recommendations for dataset mixing.
        
        Args:
            mixing_ratios (dict): Calculated mixing ratios
            
        Returns:
            list: List of recommendation strings
        """
        recommendations = []
        
        # Sort by final ratio
        sorted_ratios = sorted(mixing_ratios.items(), key=lambda x: x[1]['final_ratio'], reverse=True)
        
        for path, ratio in sorted_ratios[:3]:  # Top 3 datasets
            if ratio['final_ratio'] > 0.1:
                recommendations.append(f"Primary dataset: {os.path.basename(path)} ({ratio['normalized_ratio']:.1%})")
            elif ratio['final_ratio'] > 0.05:
                recommendations.append(f"Secondary dataset: {os.path.basename(path)} ({ratio['normalized_ratio']:.1%})")
        
        # Quality warnings
        low_quality_datasets = [
            path for path, ratio in mixing_ratios.items() 
            if ratio['quality_score'] < self.quality_threshold
        ]
        
        if low_quality_datasets:
            recommendations.append(f"Warning: {len(low_quality_datasets)} datasets have low quality scores")
        
        return recommendations


class DatasetCleaner:
    @staticmethod
    def fast_clean(data_dir: str, max_len: int = 256):
        """
        Perform 30-second emergency data cleaning on dataset files in the specified directory.

        Args:
            data_dir (str): Directory containing JSON files to be cleaned.
            max_len (int, optional): Maximum length of valid text. Defaults to 256.
        """
        _logger.debug("Performing emergency data cleaning...")
        
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
        cache_files = [f for f in os.listdir(data_dir) if f.endswith(('.json', '.jsonl', '.csv', '.parquet'))]
        for file in cache_files:
            file_path = os.path.join(data_dir, file)
            try:
                # Load data based on file extension
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif file_path.endswith('.jsonl'):
                    data = []
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    data.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    data = df.to_dict('records')
                elif file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                    data = df.to_dict('records')
                
                # Apply emergency filter
                if isinstance(data, dict) and 'text' in data:
                    filtered = emergency_filter(data)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered, f, ensure_ascii=False, indent=2)
                
                _logger.success(f"Cleaning completed: {file}")
            except Exception as e:
                _logger.debug(f"Skipping file {file}: {e}")
        
        _logger.success("Emergency data cleaning completed!")
    
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
    def calculate_text_quality_score(text):
        """
        Calculate comprehensive text quality score based on multiple metrics.
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            float: Quality score between 0 and 1
        """
        if not text or not isinstance(text, str):
            return 0.0
        
        text = text.strip()
        if not text:
            return 0.0
        
        try:
            # Length score
            length_score = min(len(text) / 1000, 1.0)  # Normalize to 1000 chars max
            
            # Character diversity score
            unique_chars = len(set(text.lower()))
            char_diversity = min(unique_chars / 26, 1.0)  # Normalize to alphabet size
            
            # Word diversity score
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words = len(set(words))
            word_diversity = min(unique_words / len(words) if words else 0, 1.0)
            
            # Sentence structure score
            sentences = re.split(r'[.!?]+', text)
            valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
            structure_score = min(len(valid_sentences) / len(sentences) if sentences else 0, 1.0)
            
            # Punctuation balance score
            punct_count = len(re.findall(r'[.!?,:;]', text))
            punct_score = min(punct_count / (len(text) / 100), 1.0)  # Normalize punctuation density
            
            # Repetition penalty
            word_counts = Counter(words)
            if word_counts:
                most_common_count = word_counts.most_common(1)[0][1]
                repetition_penalty = 1.0 - min(most_common_count / len(words), 0.5)
            else:
                repetition_penalty = 1.0
            
            # Combine scores with weights
            quality_score = (
                length_score * 0.2 +
                char_diversity * 0.15 +
                word_diversity * 0.25 +
                structure_score * 0.25 +
                punct_score * 0.1 +
                repetition_penalty * 0.05
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5  # Default medium quality on error

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
        elif input_path.endswith('.csv'):
            # CSV format
            df = pd.read_csv(input_path)
            original_size = len(df)
        elif input_path.endswith('.parquet'):
            # Parquet format
            df = pd.read_parquet(input_path)
            original_size = len(df)
        else:
            raise ValueError(f"Unsupported file format: {input_path}. Supported formats: .arrow (directory), .json, .jsonl, .csv, .parquet")
        
        # Auto-detect text field if specified field not found
        from . import TEXT_FIELD_KEYS
        
        if text_field not in df.columns:
            detected_field = None
            for field in TEXT_FIELD_KEYS:
                if field in df.columns:
                    detected_field = field
                    break
            
            if detected_field:
                _logger.debug(f"Text field '{text_field}' not found, using '{detected_field}' instead")
                text_field = detected_field
            else:
                # If no text field found, use the first string column
                string_cols = df.select_dtypes(include=['object']).columns
                if len(string_cols) > 0:
                    text_field = string_cols[0]
                    _logger.debug(f"No standard text field found, using first string column '{text_field}'")
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

        # Advanced data cleaning with multiple strategies
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.ensemble import IsolationForest
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Advanced outlier detection and handling
        for col in numeric_cols:
            if col != text_field and len(df[col].dropna()) > 10:
                try:
                    # Robust outlier detection using Isolation Forest
                    valid_data = df[col].dropna().values.reshape(-1, 1)
                    if len(valid_data) > 0:
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outliers = iso_forest.fit_predict(valid_data)
                        
                        # Get outlier indices and replace with median
                        outlier_indices = np.where(outliers == -1)[0]
                        median_value = np.median(valid_data)
                        
                        # Create mask for outliers in original dataframe
                        col_values = df[col].values
                        outlier_mask = np.isin(np.arange(len(col_values)), outlier_indices)
                        df.loc[outlier_mask, col] = median_value
                    
                    # Robust scaling for normalization
                    scaler = RobustScaler()
                    valid_mask = df[col].notna()
                    if valid_mask.sum() > 0:
                        df.loc[valid_mask, col] = scaler.fit_transform(
                            df.loc[valid_mask, col].values.reshape(-1, 1)
                        ).flatten()
                        
                except Exception as e:
                    # Fallback to IQR method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Advanced categorical cleaning with encoding and validation
        for col in categorical_cols:
            if col != text_field:
                try:
                    # Convert to string and handle complex types
                    df[col] = df[col].astype(str)
                    
                    # Advanced cleaning pipeline
                    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else str(x).strip())
                    df[col] = df[col].replace(['None', 'nan', 'NaN', 'null', 'NULL'], None)
                    
                    # Frequency-based filtering for rare categories
                    value_counts = df[col].value_counts()
                    rare_categories = value_counts[value_counts < 3].index
                    df[col] = df[col].replace(rare_categories, 'OTHER')
                    
                    # Handle empty strings
                    df[col] = df[col].replace('', None)
                    
                except Exception:
                    # Basic fallback cleaning
                    df[col] = df[col].astype(str).str.strip()
                    df[col] = df[col].replace('', None)
        
        # Advanced text cleaning with NLP processing
        import re
        try:
            import spacy
            SPACY_AVAILABLE = True
        except ImportError:
            SPACY_AVAILABLE = False
        
        try:
            from textblob import TextBlob
            TEXTBLOB_AVAILABLE = True
        except ImportError:
            TEXTBLOB_AVAILABLE = False
        
        def clean_text_content(text):
            """
            Advanced text cleaning with NLP processing and quality enhancement.
            
            Args:
                text (str): The text to be cleaned.

            Returns:
                str: The cleaned and enhanced text.
            """
            if not isinstance(text, str):
                return ""
            text = str(text).strip()
            if not text:
                return ""
            
            try:
                # Load spaCy model for advanced cleaning
                nlp = None
                if SPACY_AVAILABLE:
                    try:
                        nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        try:
                            nlp = spacy.load("en_core_web_md")
                        except OSError:
                            pass  # Fallback to basic regex
                
                # Multi-stage cleaning pipeline
                
                # Stage 1: Character-level cleaning
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # Control chars
                text = re.sub(r'[^\w\s\.,!?;:\-\(\)\[\]\{\}"\'@#$%&*+=<>/?\\|`~]', '', text)  # Special chars
                
                # Stage 2: Structural cleaning
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'([.!?])\1+', r'\1', text)  # Remove repeated punctuation
                text = re.sub(r'(.)\1{3,}', r'\1\1', text)  # Limit character repetition
                
                # Stage 3: NLP-enhanced cleaning
                if nlp:
                    doc = nlp(text)
                    
                    # Remove sentences with low quality (too short or malformed)
                    quality_sentences = []
                    for sent in doc.sents:
                        sent_text = sent.text.strip()
                        if len(sent_text.split()) >= 3:  # Minimum 3 words per sentence
                            # Check for reasonable sentence structure
                            has_verb = any(token.pos_ == "VERB" for token in sent)
                            has_noun = any(token.pos_ in ["NOUN", "PROPN"] for token in sent)
                            if has_verb and has_noun:
                                quality_sentences.append(sent_text)
                    
                    text = ' '.join(quality_sentences) if quality_sentences else text
                
                # Stage 4: Quality enhancement
                # Correct common spelling issues
                if TEXTBLOB_AVAILABLE:
                    try:
                        blob = TextBlob(text)
                        text = str(blob.correct())
                    except:
                        pass  # Skip if TextBlob fails
                
                # Final normalization
                text = text.strip()
                text = re.sub(r'\s+', ' ', text)  # Final whitespace cleanup
                
                return text
                
            except Exception:
                # Ultimate fallback cleaning
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
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
    def enhanced_multimodal_cleaning_work(
        dataset_path, 
        output_path, 
        text_field='text',
        quality_threshold=0.5,
        enable_quality_scoring=True,
        chunk_size=2000,
        num_workers=None,
        **kwargs
    ):
        """
        Enhanced multimodal dataset cleaning with advanced quality scoring and comprehensive validation.
        
        Args:
            dataset_path (str): Path to input dataset
            output_path (str): Path to save cleaned dataset
            text_field (str): Name of text field
            quality_threshold (float): Minimum quality score for acceptance
            enable_quality_scoring (bool): Enable advanced quality scoring
            chunk_size (int): Processing chunk size
            num_workers (int): Number of parallel workers
            **kwargs: Additional cleaning parameters
            
        Returns:
            dict: Cleaning statistics and quality metrics
        """
        try:
            # Initialize quality controller
            quality_controller = DataQualityController(
                quality_threshold=quality_threshold,
                diversity_threshold=0.3
            )
            
            # Analyze dataset quality first
            quality_analysis = quality_controller.analyze_dataset_quality(dataset_path)
            
            if "error" in quality_analysis:
                _logger.error(f"Quality analysis failed: {quality_analysis['error']}")
                return {"error": quality_analysis["error"]}
            
            _logger.debug(f"Dataset quality analysis: {quality_analysis}")
            
            # Load dataset
            if os.path.isdir(dataset_path):
                # Arrow format directory
                dataset = load_from_disk(dataset_path)
            elif dataset_path.endswith('.json'):
                df = pd.read_json(dataset_path)
                dataset = Dataset.from_pandas(df)
            elif dataset_path.endswith('.jsonl'):
                df = pd.read_json(dataset_path, lines=True)
                dataset = Dataset.from_pandas(df)
            elif dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
                dataset = Dataset.from_pandas(df)
            elif dataset_path.endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
                dataset = Dataset.from_pandas(df)
            else:
                return {"error": "Unsupported file format. Supported formats: .arrow (directory), .json, .jsonl, .csv, .parquet"}
            
            total_samples = len(dataset)
            
            # Find multimodal fields
            multimodal_fields = StreamCleaner.find_multimodal_fields(dataset)
            
            # Process in chunks with quality scoring
            cleaned_data = []
            quality_scores = []
            media_quality_scores = []
            
            # Set up multiprocessing
            if num_workers is None:
                num_workers = min(os.cpu_count(), 4)
            
            for i in range(0, total_samples, chunk_size):
                chunk = dataset.select(range(i, min(i + chunk_size, total_samples)))
                
                # Clean chunk
                cleaned_chunk, chunk_quality_scores, chunk_media_scores = DatasetCleaner._process_chunk_with_quality(
                    chunk, 
                    text_field, 
                    multimodal_fields,
                    enable_quality_scoring,
                    **kwargs
                )
                
                cleaned_data.extend(cleaned_chunk)
                quality_scores.extend(chunk_quality_scores)
                media_quality_scores.extend(chunk_media_scores)
                
                # Memory management
                gc.collect()
                
                _logger.debug(f"Processed chunk {i//chunk_size + 1}/{(total_samples-1)//chunk_size + 1}")
            
            # Filter by quality threshold
            if enable_quality_scoring:
                valid_indices = [
                    i for i, score in enumerate(quality_scores) 
                    if score >= quality_threshold
                ]
                cleaned_data = [cleaned_data[i] for i in valid_indices]
                quality_scores = [quality_scores[i] for i in valid_indices]
                media_quality_scores = [media_quality_scores[i] for i in valid_indices]
            
            # Create cleaned dataset
            if cleaned_data:
                cleaned_df = pd.DataFrame(cleaned_data)
                cleaned_dataset = Dataset.from_pandas(cleaned_df)
                cleaned_dataset.save_to_disk(output_path)
                
                # Generate comprehensive statistics
                stats = {
                    'original_samples': total_samples,
                    'cleaned_samples': len(cleaned_data),
                    'retention_rate': len(cleaned_data) / total_samples if total_samples > 0 else 0,
                    'avg_text_quality': np.mean(quality_scores) if quality_scores else 0,
                    'avg_media_quality': np.mean(media_quality_scores) if media_quality_scores else 0,
                    'quality_distribution': {
                        'high': sum(1 for s in quality_scores if s >= 0.8) / len(quality_scores) if quality_scores else 0,
                        'medium': sum(1 for s in quality_scores if 0.5 <= s < 0.8) / len(quality_scores) if quality_scores else 0,
                        'low': sum(1 for s in quality_scores if s < 0.5) / len(quality_scores) if quality_scores else 0,
                    },
                    'multimodal_fields_found': list(multimodal_fields.keys()),
                    'quality_analysis': quality_analysis
                }
                
                _logger.success(f"Enhanced multimodal cleaning completed: {stats['cleaned_samples']}/{stats['original_samples']} samples retained")
                return stats
            else:
                WARNING("No samples passed quality filtering")
                return {"error": "All samples failed quality filtering"}
                
        except Exception as e:
            _logger.error(f"Enhanced multimodal cleaning failed: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def _process_chunk_with_quality(chunk, text_field, multimodal_fields, enable_quality_scoring, **kwargs):
        """
        Process a chunk of data with comprehensive quality scoring.
        
        Args:
            chunk: Dataset chunk
            text_field (str): Text field name
            multimodal_fields (dict): Multimodal field information
            enable_quality_scoring (bool): Enable quality scoring
            **kwargs: Cleaning parameters
            
        Returns:
            tuple: (cleaned_data, text_quality_scores, media_quality_scores)
        """
        cleaned_data = []
        text_quality_scores = []
        media_quality_scores = []
        
        for sample in chunk:
            try:
                cleaned_sample = dict(sample)
                
                # Clean text field
                if text_field in cleaned_sample:
                    original_text = cleaned_sample[text_field]
                    
                    # Extract text from complex formats
                    if isinstance(original_text, (list, dict)):
                        extracted_text = DatasetCleaner._extract_text_from_complex_format(original_text)
                        cleaned_sample[text_field] = extracted_text
                    
                    # Calculate text quality
                    if enable_quality_scoring:
                        text_quality = DatasetCleaner.calculate_text_quality_score(cleaned_sample[text_field])
                        cleaned_sample['text_quality_score'] = text_quality
                        text_quality_scores.append(text_quality)
                    else:
                        text_quality_scores.append(1.0)  # Default score
                
                # Clean multimodal fields
                media_score_sum = 0
                media_count = 0
                
                for field_name, field_type in multimodal_fields.items():
                    if field_name in cleaned_sample and cleaned_sample[field_name]:
                        media_path = cleaned_sample[field_name]
                        
                        # Clean media with quality scoring
                        cleaned_path = StreamCleaner.clean_media(media_path, field_type)
                        
                        if cleaned_path:
                            cleaned_sample[field_name] = cleaned_path
                            
                            # Get quality score if available
                            if enable_quality_scoring and hasattr(StreamCleaner, 'get_media_quality_score'):
                                media_quality = StreamCleaner.get_media_quality_score(cleaned_path, field_type)
                                cleaned_sample[f'{field_name}_quality_score'] = media_quality
                                media_score_sum += media_quality
                                media_count += 1
                        else:
                            # Media failed cleaning, remove from sample
                            del cleaned_sample[field_name]
                
                # Calculate average media quality
                avg_media_quality = media_score_sum / media_count if media_count > 0 else 1.0
                media_quality_scores.append(avg_media_quality)
                
                cleaned_data.append(cleaned_sample)
                
            except Exception as e:
                _logger.debug(f"Sample processing failed: {str(e)}")
                continue
        
        return cleaned_data, text_quality_scores, media_quality_scores
    
    @staticmethod
    def _extract_text_from_complex_format(data):
        """
        Extract text from complex formats (conversations, nested structures).
        
        Args:
            data: Complex data structure
            
        Returns:
            str: Extracted text
        """
        if isinstance(data, str):
            return data.strip()
        elif isinstance(data, list):
            texts = []
            for item in data:
                if isinstance(item, dict):
                    # Extract from conversation formats
                    for key in ['content', 'text', 'value', 'human', 'assistant', 'user', 'bot']:
                        if key in item and item[key]:
                            texts.append(str(item[key]).strip())
                            break
                elif isinstance(item, str):
                    texts.append(item.strip())
            return ' '.join(texts)
        elif isinstance(data, dict):
            texts = []
            for key in ['content', 'text', 'value', 'human', 'assistant', 'user', 'bot']:
                if key in data and data[key]:
                    texts.append(str(data[key]).strip())
            return ' '.join(texts)
        else:
            return str(data).strip()

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
            _logger.error(f"Failed to process dataset {ds_path}: {str(e)}")
            return None

    @staticmethod
    def merge_and_clean(
        input_dir=None,
        output_dir=None,
        min_len=1,
        max_len=1024,
        workers=None,
        rules=None
    ) -> Optional[Dataset]:
        """
        Perform one-click merge and clean on datasets in the input directory, supporting multiprocessing streaming processing.

        Args:
            input_dir (str, optional): Directory containing datasets to be cleaned. Defaults to cache manager data directory.
            output_dir (str, optional): Directory to save the merged and cleaned dataset. Defaults to None.
            min_len (int, optional): Minimum length of valid text. Defaults to 1.
            max_len (int, optional): Maximum length of valid text. Defaults to 1024.
            workers (int, optional): Number of worker processes. Defaults to the minimum of 4 and CPU count.
            rules (List[Callable], optional): List of cleaning rules. Defaults to all rules in RULES.

        Returns:
            Optional[Dataset]: The merged and cleaned dataset if successful, None otherwise.
        """
        
        # Use cache manager to get data directory if input_dir is not provided
        if input_dir is None:
            cache_manager = PiscesLxCoreCacheManagerFacade.get_instance()
            input_dir = cache_manager.get_cache_dir("data_cache")
        
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
            _logger.debug("No datasets to be cleaned found")
            return None
        
        _logger.debug(f"Found {len(raw_paths)} datasets, starting {workers} processes for cleaning...")
        
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
        
        _logger.success(f"Merging {len(valid_results)} cleaned datasets...")
        merged = concatenate_datasets(valid_results)
        
        # Remove source column to avoid metadata
        if "source" in merged.column_names:
            merged = merged.remove_columns(["source"])
        
        _logger.success(f"Cleaning and merging completed! Total {len(merged)} records saved to {output_dir}")
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
                _logger.debug(f"Dataset path does not exist: {input_path}")
                return (0, 0)
            
            # Check if dataset is empty
            try:
                dataset = load_from_disk(input_path)
                if len(dataset) == 0:
                    _logger.debug(f"Dataset {dataset_name} is empty, skipping processing")
                    return (0, 0)
                _logger.debug(f"Starting to process dataset: {dataset_name} (Total {len(dataset)} records)")
            except Exception as e:
                _logger.error(f"Failed to load dataset {dataset_name}: {str(e)}")
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
        input_dir=None,
        output_dir=None,
        min_len=1,
        max_len=1024,
        workers=None,
        enable_multiprocessing=True
    ):
        """
        Fast cleaning mode - complete all cleaning steps in one click.
        
        Args:
            input_dir: Input dataset directory. Defaults to cache manager data directory.
            output_dir: Output directory for cleaned datasets (None to avoid creation).
            min_len: Minimum text length.
            max_len: Maximum text length.
            workers: Number of processes, default is the number of CPU cores.
            enable_multiprocessing: Whether to enable multiprocessing.
        """
        try:
            # Use cache manager to get data directory if input_dir is not provided
            if input_dir is None:
                cache_manager = PiscesLxCoreCacheManagerFacade.get_instance()
                input_dir = cache_manager.get_cache_dir("data_cache")
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
                    _logger.debug(f"Dataset {dataset_name} download not complete, skipping...")
                    continue
                
                output_path = os.path.join(output_dir, f"{dataset_name}_clean")
                if not os.path.exists(output_path):
                    datasets_to_clean.append((dataset_name, input_path, output_path))
                else:
                    _logger.success(f"Cleaned dataset already exists: {output_path}, skipping processing")
        
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
            _logger.debug(f"Using {workers} processes to clean {len(datasets_to_clean)} datasets...")
            
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
                            _logger.debug(f"Warning: No valid data left after cleaning {dataset_name} (Original {total_count} samples)")
                        else:
                            _logger.success(f"Cleaning completed: {dataset_name} -> {dataset_name}_clean | Samples retained: {cleaned_count}/{total_count}")
                    except Exception as e:
                        _logger.error(f"Error cleaning {dataset_name}: {str(e)}")
        
        return True