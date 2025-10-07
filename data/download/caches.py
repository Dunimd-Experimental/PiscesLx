import os
import logging
import warnings
from utils import PiscesLxCoreLog, PiscesLxCoreCacheManagerFacade

_log = PiscesLxCoreLog("PiscesLx.DataDownload")

# Cache init
_cache_manager = PiscesLxCoreCacheManagerFacade.get_instance()
DATATEMP_DIR = _cache_manager.get_cache_dir("datatemp")
DATA_DIR = str(_cache_manager.get_cache_dir("data_cache"))

# Env for ModelScope
os.environ["MODELSCOPE_CACHE"] = str(DATATEMP_DIR)
os.environ["MODELSCOPE_HUB_CACHE"] = str(DATATEMP_DIR)
os.environ["MODELSCOPE_DATASETS_CACHE"] = str(DATATEMP_DIR / "datasets")
MODELSCOPE_CACHE_DIR = str(DATATEMP_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging preferences for modelscope
logging.getLogger("modelscope").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="modelscope")

# Probe modelscope availability with diagnostics
try:
    from modelscope.msdatasets import MsDataset  # noqa: F401
    MODELSCOPE_AVAILABLE = True
except ImportError as e:
    try:
        import importlib.metadata as _meta
        _ds_ver = _meta.version("datasets")
    except Exception:
        _ds_ver = "unknown"
    try:
        import importlib.metadata as _meta
        _ms_ver = _meta.version("modelscope")
    except Exception:
        _ms_ver = "not installed"
    MODELSCOPE_AVAILABLE = False
    _log.debug(f"Modelscope not available, fallback to HuggingFace datasets: {e}")
    _log.debug(f"Installed versions => datasets: {_ds_ver}, modelscope: {_ms_ver} (expected: datasets<3.0, modelscope>=1.28.0)")

def get_data_dir() -> str:
    return DATA_DIR

def get_temp_dir():
    return DATATEMP_DIR

__all__ = ["MODELSCOPE_AVAILABLE", "MODELSCOPE_CACHE_DIR", "get_data_dir", "get_temp_dir"]