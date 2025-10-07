import os
import gc
import shutil
import multiprocessing
from typing import Iterable, Tuple, Optional, Set

from tqdm import tqdm
from datasets import load_from_disk, Dataset  # type: ignore

from utils import PiscesLxCoreLog
from .caches import get_data_dir, get_temp_dir, MODELSCOPE_CACHE_DIR
from .config import load_config
from .sources import load_any, to_hf_if_needed
from data.clean import DatasetCleaner  # reuse existing cleaner

_log = PiscesLxCoreLog("PiscesLx.DataDownload")

DATA = get_data_dir()
DATATEMP_DIR = get_temp_dir()

def _save(ds, name: str) -> bool:
    try:
        save_path = os.path.join(DATA, name)
        _log.success(f"Saving {name} to {save_path}...")
        ds.save_to_disk(save_path)
        _log.success(f"{name} saved to {save_path}")
        return True
    except Exception as e:
        _log.error(f"Failed to save {name}: {e}")
        return False

def _norm_sources(srcs):
    if not srcs:
        return ["modelscope", "huggingface"]
    norm = []
    for s in srcs:
        s_lower = (s or "").strip().lower()
        if s_lower in ("hf", "huggingface"):
            norm.append("huggingface")
        elif s_lower in ("ms", "modelscope"):
            norm.append("modelscope")
    seen = set()
    out = []
    for s in norm:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out or ["modelscope", "huggingface"]

def _load_with_methods(dataset_name: str, preferred_sources=None):
    methods = [
        ({}, "direct"),
        ({"split": "train"}, "split=train"),
        ({"split": "validation"}, "split=validation"),
        ({"split": "test"}, "split=test"),
        ({"split": "default"}, "split=default"),
    ]
    last_err: Optional[str] = None
    for kwargs, _ in methods:
        try:
            ds = load_any(dataset_name, kwargs, preferred_sources=preferred_sources)
            if ds is not None and len(ds) > 0:
                return ds
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"Failed to load {dataset_name}: {last_err}")

def _download_worker(args: Tuple[str, str, str, list]) -> Optional[str]:
    dataset_name, save_name, description, preferred_sources = args
    max_retries = 3
    for attempt in range(max_retries):
        try:
            _log.success(f"Downloading {description} ({dataset_name})... (Attempt {attempt+1}/{max_retries})")
            ds = _load_with_methods(dataset_name, preferred_sources=preferred_sources)
            ds = to_hf_if_needed(ds)
            if hasattr(ds, "__len__"):
                _log.success(f"Dataset loaded successfully, samples: {len(ds):,}")
            if _save(ds, save_name):
                # Memory cleanup
                del ds
                gc.collect()
                try:
                    import torch  # type: ignore
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return save_name
            else:
                _log.debug(f"Save failed for {dataset_name}, retrying...")
        except Exception as e:
            if attempt < max_retries - 1:
                _log.error(f"Attempt {attempt+1} failed: {e}. Retrying...")
                import time
                time.sleep(5)
            else:
                _log.error(f"All {max_retries} attempts failed for {dataset_name}. Last error: {e}")
    return None

def _cleanup_caches():
    cache_dirs = [
        os.path.join(DATA, ".cache"),
        os.path.join(DATA, "tmp"),
        os.path.join(DATA, "temp"),
        os.path.join(DATA, "cache"),
        os.path.join(DATA, "downloads"),
        str(DATATEMP_DIR),
        os.path.join(MODELSCOPE_CACHE_DIR, "datasets"),
        os.path.join(MODELSCOPE_CACHE_DIR, "hub"),
        os.path.join(os.path.dirname(__file__), "..", "modelscope"),
    ]
    for dir_path in cache_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                _log.success(f"Removed cache directory: {dir_path}")
            except Exception as e:
                _log.debug(f"Skip removing {dir_path}: {e}")

def download_datasets(config_path: str = "configs/model.json"):
    cfg = load_config(config_path)

    # Skip already downloaded
    downloaded: Set[str] = set()
    for item in cfg.datasets:
        p = os.path.join(DATA, item.save)
        if os.path.exists(p):
            downloaded.add(item.save)

    # build preferred_sources for each dataset
    def _item_sources(d):
        if getattr(d, "source", None):
            return [d.source]
        if getattr(d, "source_preference", None):
            return d.source_preference
        return cfg.source_preference
    to_download = [(d.name, d.save, d.desc, _norm_sources(_item_sources(d))) for d in cfg.datasets if d.save not in downloaded]
    total = len(cfg.datasets)
    if not to_download:
        _log.success(f"All {total} datasets already downloaded")
        return

    _log.success("Starting ModelScope dataset download...")
    _log.debug(f"Detected {total} total datasets, {len(downloaded)} downloaded, {len(to_download)} need download")

    cpu_cores = multiprocessing.cpu_count()
    workers = max(1, cpu_cores - 1) if cpu_cores < 8 else min(cpu_cores, 8)

    success_count = 0
    successfully_downloaded: Set[str] = set()
    with multiprocessing.Pool(processes=workers) as pool:
        results = list(tqdm(pool.imap_unordered(_download_worker, to_download), total=len(to_download), desc="Downloading datasets"))
        for save_name in results:
            if save_name:
                success_count += 1
                successfully_downloaded.add(save_name)

    if cfg.post_download_clean and successfully_downloaded:
        _log.debug(f"Starting unified cleaning for all {len(successfully_downloaded)} downloaded datasets...")
        try:
            DatasetCleaner.auto_clean(
                input_dir=DATA,
                output_dir=DATA,
                min_length=1,
                text_field=None,
                workers=None
            )
            _log.success("Unified cleaning completed for all datasets")
        except Exception as e:
            _log.error(f"Unified cleaning failed: {e}")
            try:
                DatasetCleaner.auto_clean(
                    input_dir=DATA,
                    output_dir=DATA,
                    min_length=1,
                    text_field=None
                )
                _log.success("Unified cleaning completed in fallback mode")
            except Exception as e2:
                _log.error(f"Unified cleaning in fallback mode failed: {e2}")

        _cleanup_caches()
        gc.collect()
        _log.success("System garbage collection completed")

        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _log.success("CUDA memory cache cleared")
        except Exception:
            pass

        _log.success(f"Download completed! Success: {success_count}/{len(cfg.datasets)}")

        # Generate model.txt
        if successfully_downloaded:
            model_file = os.path.join(DATA, "model.txt")
            try:
                with open(model_file, "w", encoding="utf-8") as f:
                    for name in sorted(successfully_downloaded):
                        f.write(f"{name}\n")
                _log.success(f"Generated model.txt with {len(successfully_downloaded)} datasets")
            except Exception as e:
                _log.error(f"Failed to generate model.txt: {e}")

def optimize_datasets(max_keep=None):
    """
    Keep behavior consistent with original optimize_datasets: in-place clean.
    """
    data_cache_dir = DATA
    for entry in os.listdir(data_cache_dir):
        raw_dir = os.path.join(data_cache_dir, entry)
        if not os.path.isdir(raw_dir):
            continue
        try:
            _log.debug(f"Processing {raw_dir}...")
            ds = load_from_disk(raw_dir)
            original_len = len(ds)
            if original_len == 0:
                _log.debug(f"{raw_dir} - Original dataset is empty, skipping")
                continue

            df = ds.to_pandas()

            # Detect text field
            text_field = None
            from data.__init__ import TEXT_FIELD_KEYS  # original location
            for field in TEXT_FIELD_KEYS:
                if field in df.columns:
                    text_field = field
                    break
            if not text_field:
                string_cols = df.select_dtypes(include=["object"]).columns
                if len(string_cols) > 0:
                    text_field = string_cols[0]
                    _log.debug(f"Using string column '{text_field}' as the text field")
                else:
                    _log.debug(f"{raw_dir} - No text field found, skipping")
                    continue

            # Clean
            import re
            def clean_text_simple(text):
                if not isinstance(text, str):
                    return ""
                text = str(text).strip()
                if not text:
                    return ""
                text = re.sub(r"[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f-\\x9f]", "", text)
                text = re.sub(r"\\s+", " ", text).strip()
                return text

            df[text_field] = df[text_field].apply(clean_text_simple)
            mask = df[text_field].astype(str).str.strip().str.len() >= 1
            df_cleaned = df[mask]
            if len(df_cleaned) == 0:
                _log.debug(f"{raw_dir} - No valid data after cleaning, skipping")
                continue

            new_ds = Dataset.from_pandas(df_cleaned, preserve_index=False)
            new_ds.save_to_disk(raw_dir)
            _log.success(f"{raw_dir} | In-place cleaning completed: {len(df_cleaned)}/{original_len} records")
        except Exception as e:
            _log.error(f"{raw_dir} - Processing failed: {e}")
            continue