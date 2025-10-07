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

import re
from typing import Callable, Dict, Optional, Any, List


# logs removed

# 基础文本清洗规则
RULES: Dict[str, Callable[[str], str]] = {
    "ctrl_chars": lambda x: re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]", "", x),
    "whitespace": lambda x: re.sub(r"\\s+", " ", x).strip(),
    "urls":       lambda x: re.sub(r"http\\S+|www\\S+", "", x),
    "emails":     lambda x: re.sub(r"\\S+@\\S+", "", x),
    "html_tags":  lambda x: re.sub(r"<[^>]*?>", "", x),
    "emoji":      lambda x: re.sub(r"[^\\u4e00-\\u9fff\\w\\s.,!?;:]", "", x),
    "special_chars": lambda x: re.sub(r"[^\\w\\s\\u4e00-\\u9fff.,!?;:()\"'-]", "", x),
    "extra_punct": lambda x: re.sub(r"[.,!?;:]{2,}", ".", x),
    "digits_only": lambda x: x if not str(x).strip().isdigit() else "",
    "single_chars": lambda x: x if len(str(x).strip()) > 1 else "",
}

# 多模态字段候选
AUTO_FIELDS = {
    "image": ["image", "img_path", "image_path", "picture", "pic", "img"],
    "audio": ["audio", "audio_path", "wav", "sound", "mp3"],
    "doc":   ["doc", "document", "doc_path", "pdf", "file_path"],
    "video": ["video", "video_path", "mp4", "avi", "mov", "mkv"],
}


class StreamCleaner:
    def __init__(self, rules: Optional[List[Callable[[str], str]]] = None, min_len: int = 5, max_len: int = 512):
        self.rules = rules or list(RULES.values())
        self.min_len, self.max_len = min_len, max_len

    def clean_text(self, text: Any) -> str:
        if not isinstance(text, str):
            return ""
        for rule in self.rules:
            text = rule(text)
        return text if self.min_len <= len(text) <= self.max_len else ""

    def clean_media(self, path: str, media_type: str) -> Optional[str]:
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
        # 根据列名自动发现多模态字段，返回 {列名: 媒体类型}
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