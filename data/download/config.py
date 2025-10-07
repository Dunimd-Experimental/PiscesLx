import json
from dataclasses import dataclass
from typing import List, Optional

from utils import PiscesLxCoreLog

_log = PiscesLxCoreLog("PiscesLx.DataDownload")

@dataclass
class DatasetItem:
    name: str
    save: str
    desc: str
    # Optional: force a single source ("huggingface" or "modelscope")
    source: Optional[str] = None
    # Optional: preferred source order for this item
    source_preference: Optional[list[str]] = None

@dataclass
class DownloadConfig:
    max_samples_per_dataset: int = 50000
    post_download_clean: bool = True
    source_preference: list[str] = None
    datasets: List[DatasetItem] = None

def load_config(path: str = "configs/model.json") -> DownloadConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    defaults = raw.get("defaults", {})
    datasets_raw = raw.get("datasets", [])

    items: List[DatasetItem] = []
    for d in datasets_raw:
        items.append(
            DatasetItem(
                name=d["name"],
                save=d["save"],
                desc=d.get("desc", d["save"]),
                source=d.get("source"),
                source_preference=d.get("source_preference")
            )
        )

    cfg = DownloadConfig(
        max_samples_per_dataset=defaults.get("max_samples_per_dataset", 50000),
        post_download_clean=defaults.get("post_download_clean", True),
        source_preference=defaults.get("source_preference", ["modelscope", "huggingface"]),
        datasets=items
    )
    _log.debug(f"Loaded download config: {len(items)} datasets")
    return cfg