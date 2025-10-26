import os
import json
from typing import List, Dict, Any, Optional

from utils.log.core import PiscesLxCoreLog

from .config import MODALITY_DATASETS
from .runner import run_single_benchmark, compare_multiple_models


class PiscesLxToolsBenchmarkOrchestrator:
    """Orchestrator to integrate benchmark with manage.py command.

    It adapts manage.py arguments to the new tools.benchmark package APIs.
    """

    def __init__(self, args):
        self.args = args
        self.logger = PiscesLxCoreLog("pisceslx.tools.benchmark.orch").bind(cmd="benchmark")

    def _parse_generation_config(self) -> Optional[Dict[str, Any]]:
        # Allow JSON via env var PISCES_BENCHMARK_GENERATION_CONFIG
        cfg = os.getenv("PISCES_BENCHMARK_GENERATION_CONFIG", "").strip()
        if not cfg:
            return None
        try:
            return json.loads(cfg)
        except Exception as e:
            self.logger.error("Invalid PISCES_BENCHMARK_GENERATION_CONFIG", error=str(e))
            return None

    def _service_overrides(self) -> Dict[str, Any]:
        # Optional service-eval overrides via environment variables
        eval_type = os.getenv("PISCES_BENCHMARK_EVAL_TYPE", "LOCAL").upper()
        api_url = os.getenv("PISCES_BENCHMARK_API_URL")
        eval_batch_size = os.getenv("PISCES_BENCHMARK_EVAL_BATCH_SIZE")
        timeout = os.getenv("PISCES_BENCHMARK_TIMEOUT_MS")
        gen_cfg = self._parse_generation_config()
        out: Dict[str, Any] = {
            "eval_type": eval_type,
            "api_url": api_url,
            "generation_config": gen_cfg,
        }
        if eval_batch_size and eval_batch_size.isdigit():
            out["eval_batch_size"] = int(eval_batch_size)
        if timeout and timeout.isdigit():
            out["timeout"] = int(timeout)
        return out

    def _resolve_datasets(self) -> List[str]:
        # Priority: --benchmark (comma/space separated) -> MODALITY env -> default
        # Accept env PISCES_BENCHMARK_MODALITY to select preset
        if self.args.benchmark:
            txt = str(self.args.benchmark).strip()
            if "," in txt:
                return [x.strip() for x in txt.split(",") if x.strip()]
            elif " " in txt:
                return [x.strip() for x in txt.split(" ") if x.strip()]
            else:
                return [txt]
        modality = os.getenv("PISCES_BENCHMARK_MODALITY", "text").lower()
        return MODALITY_DATASETS.get(modality, MODALITY_DATASETS["text"])  # default text preset

    def _resolve_model_paths(self) -> List[str]:
        # If --model provided: single path; If comma-separated list: compare mode
        if not self.args.model:
            return []
        txt = str(self.args.model).strip()
        if "," in txt:
            return [x.strip() for x in txt.split(",") if x.strip()]
        return [txt]

    def run_tests(self, _args=None):
        """Run a minimal self-test using defaults and tiny batch."""
        self.logger.info("Running benchmark self-tests", event="benchmark.selftest.start")
        datasets = self._resolve_datasets()[:1]
        models = self._resolve_model_paths()
        if not models:
            self.logger.error("Selftest requires --model path")
            return
        model_path = models[0]
        overrides = self._service_overrides()
        result = run_single_benchmark(
            model_path=model_path,
            datasets=datasets,
            model_name=None,
            metrics=["accuracy"],
            batch_size=2,
            max_length=min(getattr(self.args, "seq_len", 512) or 512, 1024),
            temperature=0.7,
            top_p=0.9,
            output_dir="benchmark_results",
            device="auto",
            debug=True,
            use_cache=True,
            save_predictions=False,
            modality=os.getenv("PISCES_BENCHMARK_MODALITY", "text"),
            **overrides,
        )
        print(json.dumps(result.get("summary", {}), indent=2))
        self.logger.success("Self-tests completed", event="benchmark.selftest.done")

    def run(self, _args=None):
        datasets = self._resolve_datasets()
        models = self._resolve_model_paths()
        overrides = self._service_overrides()

        if not models:
            self.logger.error("Benchmark requires --model path(s)")
            return

        if len(models) == 1:
            model_path = models[0]
            result = run_single_benchmark(
                model_path=model_path,
                datasets=datasets,
                model_name=None,
                metrics=["accuracy"],
                batch_size=getattr(self.args, "seq_len", 512) and getattr(self.args, "seq_len", 512) // 64 or 8,
                max_length=getattr(self.args, "seq_len", 2048) or 2048,
                temperature=0.7,
                top_p=0.9,
                output_dir="benchmark_results",
                device="auto",
                debug=bool(getattr(self.args, "perf", False)),
                use_cache=True,
                save_predictions=True,
                modality=os.getenv("PISCES_BENCHMARK_MODALITY", "text"),
                **overrides,
            )
            print(json.dumps(result.get("summary", {}), indent=2))
        else:
            model_configs = []
            for p in models:
                model_configs.append({
                    "model_path": p,
                    "model_name": None,
                    "datasets": datasets,
                    "metrics": ["accuracy"],
                    "output_dir": "benchmark_results",
                    "device": "auto",
                    "debug": bool(getattr(self.args, "perf", False)),
                    "batch_size": getattr(self.args, "seq_len", 512) and getattr(self.args, "seq_len", 512) // 64 or 8,
                    "max_length": getattr(self.args, "seq_len", 2048) or 2048,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "use_cache": True,
                    "save_predictions": False,
                    "modality": os.getenv("PISCES_BENCHMARK_MODALITY", "text"),
                    **overrides,
                })
            results = compare_multiple_models(model_configs)
            print(json.dumps(results.get("comparison_summary", {}), indent=2))
