from dataclasses import dataclass
from dataclasses import asdict  # re-export convenience
from pathlib import Path
from typing import Dict, List, Optional, Any

# Five-modality dataset presets (adjust according to your EvalScope backend support)
MODALITY_DATASETS: Dict[str, List[str]] = {
    "text": ["mmlu", "ceval", "gsm8k", "arc", "hellaswag", "truthfulqa", "math", "humaneval"],
    "image": ["mmbench", "textvqa"],
    "audio": ["librispeech_asr"],
    "video": ["mvbench"],
    "doc": ["docvqa"],
}


@dataclass
class PiscesLxToolsBenchmarkConfig:
    """Benchmark configuration data class"""
    model_path: str
    model_name: Optional[str] = None
    datasets: List[str] = None
    metrics: List[str] = None
    batch_size: int = 8
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"
    output_dir: str = "benchmark_results"
    use_cache: bool = True
    save_predictions: bool = True
    debug: bool = False

    # New fields for modality and service evaluation
    modality: str = "text"  # one of: text, image, audio, video, doc
    eval_type: str = "LOCAL"  # or "SERVICE"
    api_url: Optional[str] = None
    generation_config: Optional[Dict[str, Any]] = None
    eval_batch_size: Optional[int] = None
    timeout: Optional[int] = None  # milliseconds

    def __post_init__(self):
        if self.datasets is None:
            if self.modality in MODALITY_DATASETS:
                self.datasets = MODALITY_DATASETS[self.modality]
            else:
                self.datasets = ["mmlu", "ceval", "gsm8k", "arc", "hellaswag"]
        if self.metrics is None:
            self.metrics = ["accuracy", "f1", "precision", "recall"]
        if self.model_name is None:
            self.model_name = Path(self.model_path).name
