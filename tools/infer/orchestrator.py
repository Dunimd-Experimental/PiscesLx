#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd project team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional
from utils import PiscesLxCoreLog, PiscesLxCoreConfigManager
from utils import PiscesLxCoreConfigManager, PiscesLxCoreCheckpointManager
from utils import PiscesLxCoreObservabilityFacade, PiscesLxCoreMetricsRegistry
from utils import PiscesLxCoreHookBus, PiscesLxCoreDeviceFacade, PiscesLxCoreEnhancedCacheManager

logger = PiscesLxCoreLog("PiscesLx.Tools.Infer.Orchestrator")

# Reuse the training profiler to avoid duplication
try:
    from tools.train.profiler import PiscesLxToolsProfiler
except Exception:
    import time
    from typing import Dict, Optional
    
    class PiscesLxToolsProfiler:  # fallback with basic timing
        """Fallback profiler with basic timing functionality when training profiler is unavailable."""
        
        def __init__(self):
            self._phase_timers: Dict[str, float] = {}
            self._phase_results: Dict[str, float] = {}
            self._active = False
        
        def start(self, phase_name: str = "infer", **kwargs) -> None:
            """Start profiling for a specific phase with optional metadata."""
            self._active = True
            self._phase_timers[phase_name] = time.perf_counter()
            logger.debug(f"Profiler started for phase: {phase_name}")
        
        def stop(self, phase_name: str = "infer", **kwargs) -> Optional[float]:
            """Stop profiling and return elapsed time for the specified phase."""
            if not self._active or phase_name not in self._phase_timers:
                logger.debug(f"Profiler stop called for inactive phase: {phase_name}")
                return None
            
            elapsed = time.perf_counter() - self._phase_timers[phase_name]
            self._phase_results[phase_name] = elapsed
            self._active = False
            
            logger.debug(f"Profiler stopped for phase: {phase_name}, elapsed: {elapsed:.3f}s")
            return elapsed

try:
    from .config import PiscesLxToolsInferConfig
except Exception:
    class PiscesLxToolsInferConfig:  # minimal facade
        def __init__(self, data: dict):
            self.data = data
        @classmethod
        def from_args(cls, args: Any) -> "PiscesLxToolsInferConfig":
            d = {}
            if getattr(args, 'infer_mode', None):
                d.setdefault('infer', {})
                d['infer']['mode'] = args.infer_mode
            return cls(d)
        def get(self, key: str, default: Optional[Any] = None) -> Any:
            cur = self.data
            for part in key.split('.'):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur

class PiscesLxToolsInferOrchestrator:
    """Orchestrates inference workflows for PiscesL1.

    Modes (phase-1):
    - standard: native Pisces inference (behavior preserved)
    - vllm: high-performance inference via VLLM (auto-fallback if unavailable)
    
    Enhanced with utils device management, caching, observability, and performance optimization.
    """

    def __init__(self, args: Any) -> None:
        self.args = args
        self.hooks = PiscesLxCoreHookBus()
        self.profiler = PiscesLxToolsProfiler()
        self.cfg = PiscesLxToolsInferConfig.from_args(args)
        # Speculative stats rolling window and policy thresholds (inline closures)
        self._spec_window = []
        self._policy = {
            "spec_accept_floor": self.cfg.get("infer.spec_accept_floor", 0.4),
            "spec_speedup_floor": self.cfg.get("infer.spec_speedup_floor", 1.2),
            "spec_window_size": self.cfg.get("infer.spec_window_size", 10)
        }
        def _summ():
            if not self._spec_window:
                return None
            n = len(self._spec_window)
            avg_accept = sum(x["accept"] for x in self._spec_window) / n
            avg_speedup = sum(x["speedup"] for x in self._spec_window) / n
            avg_time = sum(x["time_ms"] for x in self._spec_window) / n
            avg_draft_ms = sum(x.get("draft_ms", 0.0) for x in self._spec_window) / n
            avg_verify_ms = sum(x.get("verify_ms", 0.0) for x in self._spec_window) / n
            return {
                "avg_accept": round(avg_accept, 4),
                "avg_speedup": round(avg_speedup, 4),
                "avg_time_ms": round(avg_time, 2),
                "avg_draft_time_ms": round(avg_draft_ms, 2),
                "avg_verify_time_ms": round(avg_verify_ms, 2),
                "window_len": n
            }
        def _cb(stats: dict):
            try:
                accept_rate = float(stats.get("draft_acceptance_rate", 0.0))
                speedup = float(stats.get("speedup", 1.0))
                iter_accept = stats.get("iter_accept", [])
                total_ms = float(stats.get("total_time_ms", 0.0))
                draft_ms = float(stats.get("total_draft_time_ms", 0.0))
                verify_ms = float(stats.get("total_verify_time_ms", 0.0))
                # push into window
                self._spec_window.append({
                    "accept": accept_rate,
                    "speedup": speedup,
                    "iters": len(iter_accept),
                    "time_ms": total_ms,
                    "draft_ms": draft_ms,
                    "verify_ms": verify_ms
                })
                # trim window
                win_size = int(self._policy.get("spec_window_size", 10))
                if len(self._spec_window) > win_size:
                    self._spec_window = self._spec_window[-win_size:]
                # summarize and emit
                summary = _summ()
                if summary is not None:
                    self.hooks.emit("infer.spec.stats", **summary)
                # policy suggestion
                if summary and (summary["avg_accept"] < float(self._policy["spec_accept_floor"]) or summary["avg_speedup"] < float(self._policy["spec_speedup_floor"])):
                    suggestion = {
                        "adjust_draft_length": -1,
                        "adjust_num_candidates": -1,
                        "reason": "Low acceptance or speedup observed",
                        "avg_accept": summary["avg_accept"],
                        "avg_speedup": summary["avg_speedup"]
                    }
                    self.hooks.emit("infer.policy.suggest", **suggestion)
            except Exception:
                pass
        # bind closures to instance for external use
        self._summarize_spec_window = _summ
        self.get_on_stats_callback = lambda: _cb
        
        # Initialize utils-enhanced components for inference optimization
        self._init_utils_components()

    def _init_utils_components(self):
        """Initialize utils-enhanced components for inference optimization and monitoring."""
        # Initialize device facade for optimal device selection
        self.device_facade = PiscesLxCoreDeviceFacade(self.args)
        
        # Initialize cache manager for model and inference result caching
        self.cache_manager = PiscesLxCoreEnhancedCacheManager.get_instance()
        
        # Initialize observability facade for inference performance monitoring
        self.observability = PiscesLxCoreObservabilityFacade()
        
        # Initialize metrics registry for inference-specific metrics
        self.metrics_registry = PiscesLxCoreMetricsRegistry()
        
        # Initialize checkpoint manager for model loading optimization
        self.checkpoint_manager = PiscesLxCoreCheckpointManager()
        
        # Initialize configuration manager for dynamic inference config
        self.config_manager = PiscesLxCoreConfigManager()
        
        # Emit orchestrator initialization event
        device_config = self.device_facade.setup_devices(mode="inference") if hasattr(self.device_facade, 'setup_devices') else {}
        self.hooks.emit("infer.orchestrator.init",
                       config=self.cfg.dump_effective() if hasattr(self.cfg, 'dump_effective') else {},
                       device_config=device_config,
                       cache_enabled=self.cache_manager is not None,
                       observability_enabled=self.observability is not None)

    def run(self, args: Any) -> None:
        mode = self.cfg.get('infer.mode', default=(getattr(args, 'infer_mode', None) or 'standard'))
        logger.success(f"Infer orchestrator mode: {mode}")
        
        # Emit inference start event with enhanced context
        inference_context = {
            "mode": mode,
            "device_config": self.device_facade.setup_devices(mode="inference") if hasattr(self.device_facade, 'setup_devices') else {},
            "cache_enabled": self.cache_manager is not None,
            "observability_enabled": self.observability is not None
        }
        self.hooks.emit("infer.start", **inference_context)
        
        try:
            if mode in ('standard', 'vllm'):
                # Expose speculative stats callback via hooks for cooperation
                try:
                    self.hooks.emit("infer.spec.callback", callback=self.get_on_stats_callback())
                except Exception:
                    pass
                self.run_standard_infer()
            else:
                logger.error(f"Unknown infer.mode: {mode}")
                self.hooks.emit("infer.error", error=f"Unknown infer.mode: {mode}")
                raise SystemExit(1)
                
            # Emit inference completion event
            self.hooks.emit("infer.complete", mode=mode)
            
        except Exception as e:
            # Emit inference failure event with detailed error context
            error_context = {
                "error": str(e),
                "mode": mode,
                "device_config": inference_context.get("device_config", {})
            }
            self.hooks.emit("infer.failure", **error_context)
            logger.error(f"Inference failed in mode {mode}: {e}")
            raise

    def run_standard_infer(self) -> None:
        """Run native/vLLM inference via the class-based runner.
        
        Enhanced with utils device optimization, model caching, and performance monitoring.
        """
        from .runner import PiscesLxToolsInferRunner
        
        # Optimize device configuration for inference
        device_config = self.device_facade.setup_devices(mode="inference")
        if device_config.get('device_type') == 'cpu':
            logger.success("Inference on CPU - performance may be limited")
        
        # Setup intelligent model caching
        cache_hit = False
        if self.cache_manager:
            model_key = f"infer_model_{self.cfg.get('model.name', 'unknown')}_{hash(str(self.args))}"
            cached_model = self.cache_manager.get(model_key)
            if cached_model:
                cache_hit = True
                logger.debug(f"Model cache hit: Using cached model configuration")
        
        # Initialize inference performance monitoring
        if self.observability:
            inference_baseline = self.observability.collect_system_metrics()
            self.metrics_registry.set_baseline("infer.baseline", inference_baseline)
            logger.debug(f"Inference baseline established: GPU memory {inference_baseline.get('gpu_memory_used', 'N/A')}MB")
        
        # Validate model checkpoint before inference
        model_path = self.cfg.get('infer.model_path') or getattr(self.args, 'model_path', None)
        if model_path and self.checkpoint_manager:
            checkpoint_valid = self.checkpoint_manager.validate_checkpoint(model_path)
            if not checkpoint_valid:
                logger.error(f"Invalid model checkpoint: {model_path}")
                self.hooks.emit("infer.checkpoint.error", error="Invalid model checkpoint", path=model_path)
                raise SystemExit(1)
            logger.debug(f"Model checkpoint validated: {model_path}")
        
        # Emit inference standard start event with enhanced context
        inference_context = {
            "device_config": device_config,
            "cache_hit": cache_hit,
            "model_path": model_path,
            "batch_size": self.cfg.get("infer.batch_size", 1),
            "max_tokens": self.cfg.get("infer.max_tokens", 512)
        }
        self.hooks.emit("infer.standard.start", **inference_context)
        
        # Create runner with enhanced utils integration
        runner = PiscesLxToolsInferRunner(
            self.args, 
            hooks=self.hooks, 
            profiler=self.profiler, 
            cfg=self.cfg,
            cache_manager=self.cache_manager,
            device_manager=self.device_facade,
            observability=self.observability,
            on_stats_cb=self.get_on_stats_callback() if hasattr(self, "get_on_stats_callback") else None
        )
        
        # Execute inference with comprehensive monitoring
        inference_success = False
        try:
            # Pre-inference optimization
            if self.observability:
                pre_inference_metrics = self.observability.collect_system_metrics()
                self.hooks.emit("infer.pre.optimization", metrics=pre_inference_metrics)
            
            # Run inference
            runner.infer()
            inference_success = True
            # Emit speculative summary if any stats were recorded
            try:
                summary = self._summarize_spec_window()
                if summary is not None:
                    self.hooks.emit("infer.spec.summary", **summary)
            except Exception:
                pass
            
            # Post-inference analysis and caching
            if self.observability:
                post_inference_metrics = self.observability.collect_system_metrics()
                performance_analysis = self.metrics_registry.calculate_delta("infer.baseline", post_inference_metrics)
                
                # Cache model if performance is optimal
                if self.cache_manager and not cache_hit and performance_analysis.get('efficiency_score', 0) > 0.8:
                    model_cache_data = {
                        "model_config": self.cfg.dump_effective() if hasattr(self.cfg, 'dump_effective') else {},
                        "device_config": device_config,
                        "performance_metrics": post_inference_metrics,
                        "efficiency_score": performance_analysis.get('efficiency_score', 0),
                        "timestamp": self.config_manager.get_current_timestamp()
                    }
                    self.cache_manager.set(model_key, model_cache_data, ttl=3600.0)  # 1小时TTL
                    logger.debug(f"Model configuration cached with efficiency score: {performance_analysis.get('efficiency_score', 0)}")
                
                self.hooks.emit("infer.performance.analysis", analysis=performance_analysis)
                
        except Exception as e:
            # Enhanced error handling for inference failures
            error_context = {
                "error": str(e),
                "inference_phase": "standard_inference",
                "cache_hit": cache_hit,
                "device_config": device_config,
                "model_path": model_path
            }
            self.hooks.emit("infer.standard.error", **error_context)
            
            # Attempt recovery with alternative device configuration
            if self.device_facade and 'cuda' in str(e).lower():
                logger.debug("Attempting CPU fallback due to CUDA error...")
                # Implementation for device fallback would go here
            
            raise
        
        # Emit inference standard completion event with results
        completion_context = {
            "success": inference_success,
            "cache_utilized": cache_hit,
            "device_optimized": device_config.get('optimization_applied', False),
            "performance_monitored": self.observability is not None
        }
        self.hooks.emit("infer.standard.complete", **completion_context)

