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
import gc
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from datasets import load_from_disk, Dataset, concatenate_datasets
from utils import PiscesLxCoreLog, PiscesLxCoreCacheManagerFacade

from .rules import StreamCleaner, AUTO_FIELDS
from .quality import calculate_text_quality_score

_log = PiscesLxCoreLog("PiscesLx.DataClean.Pipeline")


class DatasetCleaner:
    @staticmethod
    def process_dataset(input_path: str, output_path: str, text_field: str = "text", **clean_kwargs: Any) -> Tuple[int, int]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Dataset path does not exist: {input_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        original_size = 0
        if os.path.isdir(input_path):
            dataset = load_from_disk(input_path)
            df = dataset.to_pandas()
            original_size = len(dataset)
        elif input_path.endswith(".json"):
            if os.path.basename(input_path) in ["dataset_info.json", "state.json"]:
                raise ValueError(f"Skipping system file: {input_path}")
            df = pd.read_json(input_path)
            original_size = len(df)
        elif input_path.endswith(".jsonl"):
            df = pd.read_json(input_path, lines=True)
            original_size = len(df)
        elif input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
            original_size = len(df)
        elif input_path.endswith(".parquet"):
            df = pd.read_parquet(input_path)
            original_size = len(df)
        else:
            raise ValueError("Unsupported file format. Supported: .arrow dir, .json, .jsonl, .csv, .parquet")

        # 自动探测文本列
        if text_field not in df.columns:
            detected = None
            # 使用统一的TEXT_FIELD_KEYS进行字段探测
            from .. import TEXT_FIELD_KEYS
            for k in TEXT_FIELD_KEYS:
                if k in df.columns:
                    detected = k
                    break
            if detected:
                _log.debug(f"Text field '{text_field}' not found, using '{detected}'")
                text_field = detected
            else:
                string_cols = df.select_dtypes(include=["object"]).columns
                if len(string_cols) > 0:
                    text_field = string_cols[0]
                    _log.debug(f"No standard text field found, using first string column '{text_field}'")
                else:
                    raise ValueError(f"No text field found. Columns: {list(df.columns)}")

        # 复杂结构抽取（对话/代码）
        if text_field in ["conversations", "messages", "conversation", "code"]:
            df[text_field] = df[text_field].apply(DatasetCleaner._extract_text_from_complex_format)

        # 文本清洗（加强版）
        def _clean_text_content(text):
            if not isinstance(text, str):
                return ""
            import re
            t = str(text).strip()
            if not t:
                return ""
            try:
                t = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]", "", t)
                t = re.sub(r"[^\\w\\s\\.,!?;:\\-\\(\\)\\[\\]\\{\\}\"'@#$%&*+=<>/?\\\\|`~]", "", t)
                t = re.sub(r"\\s+", " ", t)
                t = re.sub(r"([.!?])\\1+", r"\\1", t)
                t = re.sub(r"(.)\\1{3,}", r"\\1\\1", t)
                t = t.strip()
                return t
            except Exception:
                t = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]", "", t)
                t = re.sub(r"\\s+", " ", t).strip()
                return t

        df[text_field] = df[text_field].apply(_clean_text_content)

        # 过滤短文本
        min_length = int(clean_kwargs.get("min_length", 1))
        mask = df[text_field].astype(str).str.strip().str.len() >= min_length
        df = df[mask]

        df = df.dropna(how="all")
        cleaned = Dataset.from_pandas(df, preserve_index=False)
        cleaned.save_to_disk(output_path)
        return len(df), original_size

    @staticmethod
    def _extract_text_from_complex_format(data):
        if isinstance(data, str):
            return data.strip()
        elif isinstance(data, list):
            texts: List[str] = []
            for item in data:
                if isinstance(item, dict):
                    for key in ["content", "text", "value", "human", "assistant", "user", "bot", "output", "response"]:
                        if key in item and item[key]:
                            texts.append(str(item[key]).strip())
                            break
                elif isinstance(item, str):
                    texts.append(item.strip())
            return " ".join(texts)
        elif isinstance(data, dict):
            texts: List[str] = []
            for key in ["content", "text", "value", "human", "assistant", "user", "bot", "output", "response"]:
                if key in data and data[key]:
                    texts.append(str(data[key]).strip())
            return " ".join(texts)
        else:
            return str(data).strip()

    @staticmethod
    def _process_chunk_with_quality(chunk, text_field: str, multimodal_fields: Dict[str, str],
                                    enable_quality_scoring: bool, **kwargs: Any):
        cleaned_rows: List[Dict[str, Any]] = []
        text_scores: List[float] = []
        media_scores: List[float] = []
        for sample in chunk:
            try:
                row = dict(sample)
                # 文本
                if text_field in row:
                    v = row[text_field]
                    if isinstance(v, (list, dict)):
                        v = DatasetCleaner._extract_text_from_complex_format(v)
                    row[text_field] = v
                    if enable_quality_scoring:
                        s = calculate_text_quality_score(v)
                        row["text_quality_score"] = s
                        text_scores.append(s)
                    else:
                        text_scores.append(1.0)
                # 多模态
                sum_q = 0.0
                cnt_q = 0
                for col, mtype in multimodal_fields.items():
                    if col in row and row[col]:
                        cleaned_path = StreamCleaner().clean_media(str(row[col]), mtype)
                        if cleaned_path:
                            row[col] = cleaned_path
                            if enable_quality_scoring:
                                q = StreamCleaner.get_media_quality_score(cleaned_path, mtype)
                                row[f"{col}_quality_score"] = q
                                sum_q += q
                                cnt_q += 1
                        else:
                            row.pop(col, None)
                media_scores.append((sum_q / cnt_q) if cnt_q else 1.0)
                cleaned_rows.append(row)
            except Exception as e:
                _log.debug(f"sample fail: {e}")
                continue
        return cleaned_rows, text_scores, media_scores

    @staticmethod
    def merge_and_clean(input_dir: Optional[str] = None, output_dir: Optional[str] = None,
                        min_len: int = 1, max_len: int = 1024, workers: Optional[int] = None, rules=None):
        cache = PiscesLxCoreCacheManagerFacade.get_instance()
        input_dir = input_dir or cache.get_cache_dir("data_cache")
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        import multiprocessing as mp
        workers = workers or min(4, mp.cpu_count())
        rules = rules  # 保留参数位，后续扩展
        raw_paths = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        if not raw_paths:
            _log.debug("No datasets to be cleaned found")
            return None

        from multiprocessing import Pool
        def _worker(ds_path):
            try:
                ds = load_from_disk(ds_path)
                cleaner = StreamCleaner(min_len=min_len, max_len=max_len)
                chunk = []
                step = 10000
                for s in range(0, len(ds), step):
                    sub = ds.select(range(s, min(s + step, len(ds))))
                    df = sub.to_pandas()
                    if "text" in df.columns:
                        df["text"] = df["text"].apply(cleaner.clean_text)
                        df = df[df["text"].astype(str).str.strip() != ""]
                    # 多模态简单处理
                    for mtype, cands in AUTO_FIELDS.items():
                        col = next((c for c in cands if c in df.columns), None)
                        if col:
                            df[col] = df[col].apply(lambda x: cleaner.clean_media(str(x), mtype) if pd.notna(x) else None)
                    if len(df) > 0:
                        chunk.append(Dataset.from_pandas(df))
                    del df, sub
                    gc.collect()
                return concatenate_datasets(chunk) if chunk else None
            except Exception as e:
                _log.error(f"worker fail: {ds_path} {e}")
                return None

        with Pool(processes=workers) as pool:
            results = list(pool.imap(_worker, raw_paths))
        valid = [r for r in results if r is not None]
        if not valid:
            return None
        merged = concatenate_datasets(valid)
        if "source" in merged.column_names:
            merged = merged.remove_columns(["source"])
        _log.success(f"Merged cleaned datasets: {len(merged)} rows")
        return merged

    @staticmethod
    def is_download_complete(dataset_path: str) -> bool:
        markers = [".download_complete", ".finished", "download_status.txt", "completed.flag"]
        for m in markers:
            if os.path.exists(os.path.join(dataset_path, m)):
                return True
        try:
            ds = load_from_disk(dataset_path)
            if len(ds) > 0:
                for _ in ds.take(1):
                    pass
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def auto_clean(input_dir: str, output_dir: str = "data_clean", media_fields: Optional[Dict[str, str]] = None,
                   workers: Optional[int] = None, **clean_kwargs: Any) -> bool:
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        todo: List[Tuple[str, str, str]] = []
        for name in os.listdir(input_dir):
            in_p = os.path.join(input_dir, name)
            if os.path.isdir(in_p) and not name.endswith("_clean"):
                if not DatasetCleaner.is_download_complete(in_p):
                    _log.debug(f"Dataset {name} download not complete, skip")
                    continue
                out_p = os.path.join(output_dir, f"{name}_clean")
                if not os.path.exists(out_p):
                    todo.append((name, in_p, out_p))
                else:
                    _log.success(f"Cleaned exists: {out_p}, skip")

        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        workers = workers or min(4, mp.cpu_count())
        if workers <= 1 or len(todo) <= 1:
            for name, in_p, out_p in todo:
                DatasetCleaner._process_single_dataset(name, in_p, out_p, media_fields, **clean_kwargs)
        else:
            _log.debug(f"Using {workers} processes to clean {len(todo)} datasets...")
            args = [(n, i, o, media_fields, clean_kwargs) for (n, i, o) in todo]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                fut = {ex.submit(DatasetCleaner._process_single_dataset_wrapper, a): a[0] for a in args}
                for f in as_completed(fut):
                    name = fut[f]
                    try:
                        cleaned, total = f.result()
                        if cleaned == 0:
                            _log.debug(f"No valid left after cleaning {name} (orig {total})")
                        else:
                            _log.success(f"Clean OK: {name} -> {name}_clean | {cleaned}/{total}")
                    except Exception as e:
                        _log.error(f"Error cleaning {name}: {e}")
        return True

    @staticmethod
    def _process_single_dataset_wrapper(args):
        n, i, o, m, kw = args
        return DatasetCleaner._process_single_dataset(n, i, o, m, **kw)

    @staticmethod
    def _process_single_dataset(dataset_name, input_path, output_path, media_fields=None, **clean_kwargs):
        try:
            if not os.path.exists(input_path):
                _log.debug(f"Dataset not exists: {input_path}")
                return (0, 0)
            try:
                dataset = load_from_disk(input_path)
                if len(dataset) == 0:
                    _log.debug(f"Dataset {dataset_name} empty, skip")
                    return (0, 0)
                _log.debug(f"Process dataset: {dataset_name} ({len(dataset)} rows)")
            except Exception as e:
                _log.error(f"Load fail {dataset_name}: {e}")
                return (0, 0)
            if media_fields:
                cleaned, total = DatasetCleaner.process_multimodal_dataset(input_path, output_path, media_fields=media_fields, **clean_kwargs)
            else:
                cleaned, total = DatasetCleaner.process_dataset(input_path, output_path, **clean_kwargs)
            return (cleaned, total)
        except Exception as e:
            _log.error(f"Error cleaning {dataset_name}: {e}")
            return (0, 0)

    @staticmethod
    def process_multimodal_dataset(input_path: str, output_path: str, text_field: str = "text",
                                   media_fields: Optional[Dict[str, str]] = None,
                                   quality_threshold: float = 0.5, enable_quality_scoring: bool = True,
                                   chunk_size: int = 2000, **kwargs: Any):
        if not os.path.exists(input_path):
            return (0, 0)
        dataset = load_from_disk(input_path) if os.path.isdir(input_path) else None
        if dataset is None:
            return (0, 0)
        total = len(dataset)
        multimodal_fields = media_fields or StreamCleaner.find_multimodal_fields_from_dataset(dataset)
        cleaned_rows: List[Dict[str, Any]] = []
        text_scores: List[float] = []
        media_scores: List[float] = []
        for i in range(0, total, chunk_size):
            chunk = dataset.select(range(i, min(i + chunk_size, total)))
            rows, ts, ms = DatasetCleaner._process_chunk_with_quality(chunk, text_field, multimodal_fields, enable_quality_scoring, **kwargs)
            cleaned_rows.extend(rows)
            text_scores.extend(ts)
            media_scores.extend(ms)
            gc.collect()
            _log.debug(f"Processed chunk {i // chunk_size + 1}/{(total - 1) // chunk_size + 1}")
        if enable_quality_scoring:
            idx = [i for i, s in enumerate(text_scores) if s >= quality_threshold]
            cleaned_rows = [cleaned_rows[i] for i in idx]
        if cleaned_rows:
            df = pd.DataFrame(cleaned_rows)
            ds = Dataset.from_pandas(df)
            ds.save_to_disk(output_path)
            return (len(cleaned_rows), total)
        return (0, total)