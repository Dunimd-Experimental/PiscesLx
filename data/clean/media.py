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

from typing import Optional
from utils import PiscesLxCoreLog

_log = PiscesLxCoreLog("PiscesLx.DataClean.Media")


class MediaCleaner:
    @staticmethod
    def clean_image(image_path: str, min_size=(224, 224)) -> Optional[str]:
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                    return None
            return image_path
        except Exception:
            return None

    @staticmethod
    def clean_image_with_quality(image_path: str, min_size=(224, 224), min_quality_score: float = 0.6) -> Optional[str]:
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                    return None
                score = MediaCleaner._calculate_image_quality(img)
                if score < min_quality_score:
                    return None
            return image_path
        except Exception as e:
            _log.debug(f"Image quality cleaning failed for {image_path}: {e}")
            return None

    @staticmethod
    def _calculate_image_quality(img) -> float:
        try:
            import numpy as np
            import cv2
            arr = np.array(img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
            # 分辨率（相对 1MP 归一化）
            resolution_score = min((img.width * img.height) / (1024 * 1024), 1.0)
            # 锐度（拉普拉斯方差）
            sharpness_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000, 1.0)
            # 对比度
            contrast_score = min(gray.std() / 128, 1.0)
            # 亮度均衡
            mean_brightness = gray.mean()
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            score = (
                resolution_score * 0.3 +
                sharpness_score * 0.3 +
                contrast_score * 0.2 +
                brightness_score * 0.2
            )
            return max(0.0, min(1.0, float(score)))
        except Exception:
            return 0.5

    @staticmethod
    def clean_audio(audio_path: str, min_duration: float = 1.0, max_duration: float = 30.0) -> Optional[str]:
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            dur = len(y) / float(sr)
            if dur < min_duration or dur > max_duration:
                return None
            return audio_path
        except Exception:
            return None

    @staticmethod
    def clean_audio_with_quality(audio_path: str, min_duration: float = 1.0, max_duration: float = 30.0,
                                 min_quality_score: float = 0.5) -> Optional[str]:
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None, duration=max_duration + 1)
            dur = len(y) / float(sr)
            if dur < min_duration or dur > max_duration:
                return None
            score = MediaCleaner._calculate_audio_quality(y, sr)
            if score < min_quality_score:
                return None
            return audio_path
        except Exception as e:
            _log.debug(f"Audio quality cleaning failed for {audio_path}: {e}")
            return None

    @staticmethod
    def _calculate_audio_quality(y, sr) -> float:
        try:
            import numpy as np
            import librosa
            dur = len(y) / float(sr)
            duration_score = min(dur / 10.0, 1.0)
            dynamic_range = float(np.max(y) - np.min(y))
            dynamic_score = min(dynamic_range / 0.5, 1.0)
            rms_energy = float(np.sqrt(np.mean(y ** 2)))
            energy_score = min(rms_energy / 0.1, 1.0)
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
            noise_score = 1.0 - min(zcr / 0.1, 1.0)
            score = (
                duration_score * 0.3 +
                dynamic_score * 0.3 +
                energy_score * 0.2 +
                noise_score * 0.2
            )
            return max(0.0, min(1.0, float(score)))
        except Exception:
            return 0.5

    @staticmethod
    def clean_video(video_path: str, min_duration: int = 3, min_frames: int = 8) -> Optional[str]:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            duration = frames / fps if fps and fps > 0 else 0
            if duration < min_duration or frames < min_frames:
                return None
            return video_path
        except Exception:
            return None

    @staticmethod
    def clean_video_with_quality(video_path: str, min_duration: int = 3, min_frames: int = 8,
                                 min_quality_score: float = 0.5) -> Optional[str]:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (frames / fps if fps and fps > 0 else 0) < min_duration or frames < min_frames:
                cap.release()
                return None
            score = MediaCleaner._calculate_video_quality(cap, frames)
            cap.release()
            if score < min_quality_score:
                return None
            return video_path
        except Exception as e:
            _log.debug(f"Video quality cleaning failed for {video_path}: {e}")
            return None

    @staticmethod
    def _calculate_video_quality(cap, total_frames: int) -> float:
        try:
            import numpy as np
            import cv2
            from PIL import Image
            idxs = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)
            scores = []
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                scores.append(MediaCleaner._calculate_image_quality(pil_img))
            if not scores:
                return 0.0
            import numpy as _np
            avg = float(sum(scores) / len(scores))
            var = float(_np.std(scores))
            consistency = 1.0 - min(var / (avg if avg > 0 else 1.0), 0.5)
            # 取更强的平均+一致性
            score = avg * 0.7 + consistency * 0.3
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    @staticmethod
    def clean_document(doc_path: str, max_pages: int = 50) -> Optional[str]:
        try:
            import fitz
            doc = fitz.open(doc_path)
            ok = len(doc) <= max_pages
            doc.close()
            return doc_path if ok else None
        except Exception:
            return None

    @staticmethod
    def clean_document_with_quality(doc_path: str, max_pages: int = 50, min_quality_score: float = 0.3) -> Optional[str]:
        try:
            import fitz
            doc = fitz.open(doc_path)
            if len(doc) > max_pages:
                doc.close()
                return None
            score = MediaCleaner._calculate_document_quality(doc)
            doc.close()
            if score < min_quality_score:
                return None
            return doc_path
        except Exception as e:
            _log.debug(f"Document quality cleaning failed for {doc_path}: {e}")
            return None

    @staticmethod
    def _calculate_document_quality(doc) -> float:
        try:
            page_count = len(doc)
            if page_count == 0:
                return 0.0
            page_score = min(page_count / 10.0, 1.0)
            total_text = 0
            valid_pages = 0
            for i in range(min(page_count, 5)):
                page = doc.load_page(i)
                text = page.get_text() or ""
                t = text.strip()
                if len(t) > 10:
                    total_text += len(t)
                    valid_pages += 1
            avg_len = total_text / max(valid_pages, 1)
            content_score = min(avg_len / 1000.0, 1.0)
            structure_score = valid_pages / min(page_count, 5)
            score = page_score * 0.2 + content_score * 0.6 + structure_score * 0.2
            return max(0.0, min(1.0, float(score)))
        except Exception:
            return 0.5