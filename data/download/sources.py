from typing import Any, Dict, List
from .caches import MODELSCOPE_AVAILABLE

def load_from_modelscope(dataset_name: str, kwargs: Dict[str, Any] | None = None):
    kwargs = kwargs or {}
    if not MODELSCOPE_AVAILABLE:
        raise RuntimeError("ModelScope not available in current environment")
    from modelscope.msdatasets import MsDataset
    ds = MsDataset.load(dataset_name, **kwargs)
    if ds is None or (hasattr(ds, "__len__") and len(ds) == 0):
        raise RuntimeError(f"ModelScope returned empty dataset for {dataset_name}")
    return ds

def load_from_huggingface(dataset_name: str, kwargs: Dict[str, Any] | None = None):
    kwargs = kwargs or {}
    from datasets import load_dataset
    split = kwargs.get("split")
    if split and split != "default":
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name)
    if ds is None or (hasattr(ds, "__len__") and len(ds) == 0):
        raise RuntimeError(f"HuggingFace returned empty dataset for {dataset_name}")
    return ds

def _norm_sources(srcs: List[str] | None) -> List[str]:
    if not srcs:
        return ["modelscope", "huggingface"]
    norm = []
    for s in srcs:
        s_lower = (s or "").strip().lower()
        if s_lower in ("hf", "huggingface"):
            norm.append("huggingface")
        elif s_lower in ("ms", "modelscope"):
            norm.append("modelscope")
    # de-duplicate but keep order
    seen = set()
    out = []
    for s in norm:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out or ["modelscope", "huggingface"]

def load_any(dataset_name: str, kwargs: Dict[str, Any] | None = None, preferred_sources: List[str] | None = None):
    """
    Load dataset according to preferred_sources order. Supported values: "modelscope", "huggingface".
    Falls back to default order if not provided.
    """
    kwargs = kwargs or {}
    order = _norm_sources(preferred_sources)
    ds = None
    last_error = None

    for src in order:
        if src == "modelscope":
            try:
                ds = load_from_modelscope(dataset_name, kwargs)
                return ds
            except Exception as e:
                last_error = str(e)
        elif src == "huggingface":
            try:
                ds = load_from_huggingface(dataset_name, kwargs)
                return ds
            except Exception as e:
                last_error = str(e)
    # if all sources failed, raise

    if ds is None:
        raise RuntimeError(f"Failed to load dataset {dataset_name}. Last error: {last_error}")

def to_hf_if_needed(ds):
    """
    Convert various dataset formats into HuggingFace Dataset if possible.
    """
    if hasattr(ds, "to_hf_dataset"):
        return ds.to_hf_dataset()
    if hasattr(ds, "data") and hasattr(ds, "info"):
        return ds
    if hasattr(ds, "__iter__") and not hasattr(ds, "save_to_disk"):
        try:
            from datasets import Dataset
            if hasattr(ds, "to_pandas"):
                return Dataset.from_pandas(ds.to_pandas())
        except Exception:
            pass
    return ds