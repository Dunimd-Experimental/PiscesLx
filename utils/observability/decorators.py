#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
# The PiscesL1 project belongs to the Dunimd Team.
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

import os
import uuid
import time
import json
import hashlib
from functools import wraps
from utils.log.core import PiscesLxCoreLog
from utils.cache.core import get_default_cache
from utils.hooks.bus import get_global_hook_bus
from .metrics import PiscesLxCoreMetricsRegistry
from .service import PiscesLxCoreObservabilityService
from .otel_helper import start_span, is_enabled as _otel_enabled
from typing import Any, Callable, Optional, Iterable, Iterator, Generator

class PiscesLxCoreDecorators:
    def __init__(self) -> None:
        """Initialize the PiscesLxCoreDecorators class and create a logger instance."""
        self.logger = PiscesLxCoreLog()

    @staticmethod
    def observe_tool_request(provider: str = "unknown", model: str = "unknown", route: str = "unknown", component: str = "tools") -> Callable:
        """Decorator for tool/service request functions to emit standard observability events.

        This decorator measures latency and emits events to the global HookBus:
        - tools.request.success on success
        - tools.request.error on exception

        With this in place, utils/observability/service.py listeners will update:
        - counters: llm_requests, llm_errors
        - histogram: llm_latency_ms (ms)

        Args:
            provider (str): Tool/LLM provider label.
            model (str): Model label.
            route (str): Logical route/endpoint label.
            component (str): Component label (default: tools).

        Returns:
            Callable: Decorator that wraps the function and emits observability events.
        """
        def _decorator(func: Callable) -> Callable:
            @wraps(func)
            def _wrapped(*args, **kwargs):
                bus = get_global_hook_bus()
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    end = time.time()
                    try:
                        bus.emit(
                            "tools.request.success",
                            provider=provider,
                            model=model,
                            route=route,
                            component=component,
                            start_time=start,
                            end_time=end,
                            latency_ms=(end - start) * 1000.0,
                        )
                    except Exception as log_e:
                        PiscesLxCoreLog().debug("OBS_EMIT_SUCCESS_FAILED", event="OBS_EMIT_SUCCESS_FAILED", func=func.__name__, error=str(log_e))
                    return result
                except Exception as e:
                    end = time.time()
                    try:
                        bus.emit(
                            "tools.request.error",
                            provider=provider,
                            model=model,
                            route=route,
                            component=component,
                            start_time=start,
                            end_time=end,
                            latency_ms=(end - start) * 1000.0,
                            error=str(e),
                        )
                    except Exception as log_e:
                        PiscesLxCoreLog().debug("OBS_EMIT_ERROR_FAILED", event="OBS_EMIT_ERROR_FAILED", func=func.__name__, error=str(log_e))
                    raise
            return _wrapped
        return _decorator

    @staticmethod
    def log_span(event_name: str) -> Callable:
        """Create a decorator to log the start, end, and error events of a function, and optionally create an OTel span.

        Args:
            event_name (str): The name of the event to be logged.

        Returns:
            Callable: A decorator function.
        """
        def _decorator(func: Callable) -> Callable:
            """Decorate a function to add logging and tracing functionality.

            Args:
                func (Callable): The function to be decorated.

            Returns:
                Callable: A wrapped function with logging and tracing.
            """
            @wraps(func)
            def _wrapped(*args, **kwargs):
                """The wrapped function that adds logging and tracing functionality.

                Args:
                    *args: Positional arguments passed to the original function.
                    **kwargs: Keyword arguments passed to the original function.

                Returns:
                    Any: The result of the original function.

                Raises:
                    Exception: Raises any exception thrown by the original function after logging the error.
                """
                obs = PiscesLxCoreObservabilityService.instance()
                log = obs.get_logger()
                bus = get_global_hook_bus()

                # Extract trace context information from kwargs or environment variables
                parent_span_id = kwargs.pop("parent_span_id", None)
                incoming_tp = kwargs.pop("traceparent", None) or os.environ.get("TRACEPARENT")
                if incoming_tp:
                    tp_trace_id, tp_parent_span = PiscesLxCoreDecorators._parse_traceparent(incoming_tp)
                else:
                    tp_trace_id, tp_parent_span = None, None
                trace_id = kwargs.pop("trace_id", None) or tp_trace_id or uuid.uuid4().hex[:16]
                span_id = uuid.uuid4().hex[:16]
                parent_span_id = parent_span_id or tp_parent_span

                # Record the start time
                t0 = time.time()

                # Prepare fields for logging
                fields = {"func": func.__name__, "trace_id": trace_id, "span_id": span_id}
                if parent_span_id:
                    fields["parent_span_id"] = parent_span_id

                # Build a new traceparent
                new_tp = PiscesLxCoreDecorators._make_traceparent(trace_id, span_id)

                # Log the start of the event
                log.info(f"{event_name}.start", **fields)
                try:
                    bus.emit(f"{event_name}.start", **fields)
                except Exception as e:
                    try:
                        log.debug(
                            event="hooks.emit.error",
                            phase="start",
                            func=func.__name__,
                            error=str(e),
                            error_class=type(e).__name__)
                    except Exception as log_e:
                        PiscesLxCoreLog().debug("HOOKS_EMIT_START_LOG_ERROR", event="HOOKS_EMIT_START_LOG_ERROR", event_name=event_name, error=str(log_e))

                try:
                    # Create an optional OTel span
                    span_attrs = {"trace_id": trace_id, "span_id": span_id, "func": func.__name__}
                    with start_span(event_name, kind="internal", attributes=span_attrs):
                        result = func(*args, **kwargs)

                    # Calculate the duration and log the end of the event
                    duration = int((time.time() - t0) * 1000)
                    end_fields = {"func": func.__name__, "duration_ms": duration, "trace_id": trace_id, "span_id": span_id}
                    if parent_span_id:
                        end_fields["parent_span_id"] = parent_span_id
                    end_fields["traceparent"] = new_tp
                    log.info(f"{event_name}.end", **end_fields)
                    try:
                        bus.emit(f"{event_name}.end", **end_fields)
                    except Exception as e:
                        try:
                            log.debug(
                                f"{event_name}.end.emit_error",
                                event="hooks.emit.error",
                                phase="end",
                                func=func.__name__,
                                error=str(e),
                                error_class=type(e).__name__,
                            )
                        except Exception as log_e:
                            PiscesLxCoreLog().debug("HOOKS_EMIT_END_LOG_ERROR", event="HOOKS_EMIT_END_LOG_ERROR", event_name=event_name, error=str(log_e))
                    return result
                except Exception as e:
                    # Calculate the duration and log the error
                    duration = int((time.time() - t0) * 1000)
                    err_fields = {"func": func.__name__, "duration_ms": duration, "error": str(e), "trace_id": trace_id, "span_id": span_id}
                    if parent_span_id:
                        err_fields["parent_span_id"] = parent_span_id
                    err_fields["traceparent"] = new_tp

                    # Add error classification if available
                    try:
                        err_fields["error_class"] = type(e).__name__
                        code = getattr(e, "error_code", None)
                        if code:
                            err_fields["error_code"] = code
                    except Exception as log_e:
                        PiscesLxCoreLog().debug("ERROR_CLASSIFICATION_FAILED", event="ERROR_CLASSIFICATION_FAILED", event_name=event_name, error=str(log_e))

                    log.error(f"{event_name}.error", **err_fields)
                    try:
                        bus.emit(f"{event_name}.error", **err_fields)
                    except Exception as e2:
                        try:
                            log.debug(
                                f"{event_name}.error.emit_error",
                                event="hooks.emit.error",
                                phase="error",
                                func=func.__name__,
                                error=str(e2),
                                error_class=type(e2).__name__,
                            )
                        except Exception as log_e:
                            PiscesLxCoreLog().debug("HOOKS_EMIT_ERROR_LOG_ERROR", event="HOOKS_EMIT_ERROR_LOG_ERROR", event_name=event_name, error=str(log_e))
                    raise
                finally:
                    log.debug(f"{event_name}.finally", message="No extra cleanup required, kept for symmetry and future extension")

            return _wrapped
        return _decorator

    @staticmethod
    def observe_llm_stream(provider: str = "unknown", model: str = "unknown", task: str = "inference",
                           chunk_token_fn: Optional[Callable[[Any], int]] = None,
                           chunk_text_fn: Optional[Callable[[Any], str]] = None) -> Callable:
        """Create a decorator for streaming LLM generation functions.
        This decorator counts output tokens incrementally, records TTFT (time to first token) and total latency,
        and maintains a tokens/sec gauge during streaming.

        Args:
            provider (str, optional): The LLM provider. Defaults to "unknown".
            model (str, optional): The LLM model. Defaults to "unknown".
            task (str, optional): The task type. Defaults to "inference".
            chunk_token_fn (Callable[[Any], int], optional): Function to count tokens in a chunk. Defaults to None.
            chunk_text_fn (Callable[[Any], str], optional): Function to extract text from a chunk. Defaults to None.

        Returns:
            Callable: A decorator function.
        """
        def _decorator(func: Callable) -> Callable:
            """Decorate a streaming LLM generation function to add observability metrics.

            Args:
                func (Callable): The function to be decorated.

            Returns:
                Callable: A wrapped function with observability metrics.
            """
            @wraps(func)
            def _wrapped(*args, **kwargs):
                """The wrapped function that adds observability metrics to a streaming LLM generation function.

                Args:
                    *args: Positional arguments passed to the original function.
                    **kwargs: Keyword arguments passed to the original function.

                Returns:
                    Iterator[Any]: An iterator that yields chunks from the original function with added metrics.

                Raises:
                    Exception: Raises any exception thrown by the original function after recording the error.
                """
                reg = PiscesLxCoreMetricsRegistry.instance()
                labels = {"provider": provider, "model": model, "task": task}

                # Define metrics
                req_c = reg.counter("llm.stream.requests_total", help_text="Total streaming LLM requests", labels=["provider", "model", "task"])
                err_c = reg.counter("llm.stream.errors_total", help_text="Total streaming LLM errors", labels=["provider", "model", "task"])
                fail_c = reg.counter("llm.stream.failures_total", help_text="Streaming LLM failures by class", labels=["provider", "model", "task", "error_class"])
                finish_c = reg.counter("llm.stream.finish_reason_total", help_text="Streaming finish reasons", labels=["provider", "model", "task", "reason"])
                tok_c = reg.counter("llm.stream.output_tokens_total", help_text="Total streamed output tokens", labels=["provider", "model", "task"])
                ttft_h = reg.histogram("llm.stream.ttft_ms", help_text="Time to first token (ms)", labels=["provider", "model", "task"])
                e2e_h = reg.histogram("llm.stream.e2e_latency_ms", help_text="End-to-end latency for stream (ms)", labels=["provider", "model", "task"])
                tps_g = reg.gauge("llm.stream.tokens_per_second", help_text="Streaming tokens per second", labels=["provider", "model", "task"])
                active_g = reg.gauge("llm.stream.active", help_text="Streaming active (1 while streaming)", labels=["provider", "model", "task"])

                # Record the start time and increment the request counter
                start_ts = time.time()
                req_c.inc(1.0, labels=labels)
                active_g.set(1.0, labels=labels)
                first_token_emitted = False
                out_tokens = 0

                try:
                    result = func(*args, **kwargs)

                    def _iter() -> Iterator[Any]:
                        """Iterator that adds observability metrics to each chunk.

                        Yields:
                            Any: Chunks from the original function.
                        """
                        nonlocal first_token_emitted, out_tokens
                        for chunk in result:  # type: ignore
                            now = time.time()
                            if not first_token_emitted:
                                ttft_ms = (now - start_ts) * 1000.0
                                ttft_h.observe(ttft_ms, labels=labels)
                                first_token_emitted = True

                            # Count tokens for this chunk
                            ct = 0
                            try:
                                if chunk_token_fn is not None:
                                    ct = int(chunk_token_fn(chunk) or 0)
                                elif chunk_text_fn is not None:
                                    txt = chunk_text_fn(chunk)
                                    ct = int((len(txt) if isinstance(txt, str) else 0) / 4)
                                else:
                                    if isinstance(chunk, dict):
                                        if "text" in chunk and isinstance(chunk["text"], str):
                                            ct = int(len(chunk["text"]) / 4)
                                        elif "delta" in chunk and isinstance(chunk["delta"], str):
                                            ct = int(len(chunk["delta"]) / 4)
                            except Exception as log_e:
                                PiscesLxCoreLog().debug("TOKEN_COUNTING_FAILED", event="TOKEN_COUNTING_FAILED", chunk_type=type(chunk).__name__, error=str(log_e))
                                ct = 0

                            if ct > 0:
                                out_tokens += ct
                                tok_c.inc(ct, labels=labels)

                            # Update instantaneous TPS
                            try:
                                elapsed = max(1e-6, now - start_ts)
                                tps_g.set(out_tokens / elapsed, labels=labels)
                            except Exception as log_e:
                                PiscesLxCoreLog().debug("TPS_UPDATE_FAILED", event="TPS_UPDATE_FAILED", out_tokens=out_tokens, elapsed=elapsed, error=str(log_e))

                            yield chunk

                        # Record end-to-end latency
                        end_ms = (time.time() - start_ts) * 1000.0
                        e2e_h.observe(end_ms, labels=labels)

                    return _iter()
                except Exception as e:
                    # Increment error counters
                    err_c.inc(1.0, labels=labels)
                    try:
                        fail_c.inc(1.0, labels={**labels, "error_class": type(e).__name__})
                    except Exception as log_e:
                        PiscesLxCoreLog().debug("FAILURE_COUNTER_UPDATE_FAILED", event="FAILURE_COUNTER_UPDATE_FAILED", error_class=type(e).__name__, error=str(log_e))
                    raise
                finally:
                    try:
                        active_g.set(0.0, labels=labels)
                    except Exception as log_e:
                        PiscesLxCoreLog().debug("ACTIVE_GAUGE_RESET_FAILED", event="ACTIVE_GAUGE_RESET_FAILED", labels=labels, error=str(log_e))

            return _wrapped
        return _decorator

    @staticmethod
    def report_llm_kv_cache(provider: str, model: str, task: str,
                            hit_ratio: Optional[float] = None,
                            evictions: Optional[float] = None,
                            usage_percent: Optional[float] = None) -> None:
        """Report LLM KV cache related metrics as gauges or counters.

        Call this function from model serving code when KV cache signals are available.

        Args:
            provider (str): The LLM provider.
            model (str): The LLM model.
            task (str): The task type.
            hit_ratio (float, optional): KV cache hit ratio. Defaults to None.
            evictions (float, optional): Number of KV cache evictions. Defaults to None.
            usage_percent (float, optional): KV cache usage percentage. Defaults to None.
        """
        reg = PiscesLxCoreMetricsRegistry.instance()
        labels = {"provider": provider, "model": model, "task": task}
        try:
            if hit_ratio is not None:
                reg.gauge("llm.kv_cache.hit_ratio", help_text="LLM KV cache hit ratio", labels=["provider", "model", "task"]).set(float(hit_ratio), labels=labels)
            if usage_percent is not None:
                reg.gauge("llm.kv_cache.usage_percent", help_text="LLM KV cache usage percent", labels=["provider", "model", "task"]).set(float(usage_percent), labels=labels)
            if evictions is not None:
                reg.counter("llm.kv_cache.evictions_total", help_text="LLM KV cache evictions", labels=["provider", "model", "task"]).inc(float(evictions), labels=labels)
        except Exception as log_e:
            PiscesLxCoreLog().debug("KV_CACHE_REPORT_FAILED", event="KV_CACHE_REPORT_FAILED", provider=provider, model=model, task=task, error=str(log_e))

    @staticmethod
    def auto_cached_logged(namespace: str, ttl: int = 60, soft_expiry: bool = True) -> Callable:
        """Create a decorator that adds auto-caching and logging functionality to a function.
        The cache key is derived from the function's qualified name and a SHA1 hash of the arguments.

        Args:
            namespace (str): The cache namespace.
            ttl (int, optional): Time-to-live for cache entries in seconds. Defaults to 60.
            soft_expiry (bool, optional): Whether to allow stale cache entries once. Defaults to True.

        Returns:
            Callable: A decorator function.
        """
        def _decorator(func: Callable) -> Callable:
            """Decorate a function to add auto-caching and logging functionality.

            Args:
                func (Callable): The function to be decorated.

            Returns:
                Callable: A wrapped function with auto-caching and logging.
            """
            @wraps(func)
            def _wrapped(*args, **kwargs):
                """The wrapped function that adds auto-caching and logging functionality.

                Args:
                    *args: Positional arguments passed to the original function.
                    **kwargs: Keyword arguments passed to the original function.

                Returns:
                    Any: The result of the original function, possibly from cache.

                Raises:
                    Exception: Raises any exception thrown by the original function after logging the error.
                """
                # Get logger, cache, and hook bus
                log = PiscesLxCoreObservabilityService.instance().get_logger()
                cache = get_default_cache("observability")
                bus = get_global_hook_bus()

                # Build cache key
                key = PiscesLxCoreDecorators._build_key(func, args, kwargs)

                # Record the start time and log the start of the span
                t0 = time.time()
                span_fields = {"ns": namespace, "func": func.__name__}
                log.info("span.start", event="span.start", **span_fields)

                try:
                    # Get or set the cache entry
                    result = cache.get_or_set(
                        namespace,
                        key,
                        ttl,
                        producer=lambda: func(*args, **kwargs),
                        allow_stale_once=soft_expiry,
                        async_refresh=True,
                    )

                    # Calculate the duration and log the end of the span
                    duration = int((time.time() - t0) * 1000)
                    end_fields = {"namespace": namespace, "func": func.__name__, "duration_ms": duration}
                    log.info("span.end", event="span.end", **end_fields)
                    return result
                except Exception as e:
                    # Calculate the duration and log the error
                    duration = int((time.time() - t0) * 1000)
                    err_fields = {"namespace": namespace, "func": func.__name__, "duration_ms": duration, "error": str(e)}
                    log.error("span.error", event="span.error", **err_fields)
                    raise

            return _wrapped
        return _decorator

    @staticmethod
    def _build_key(func: Callable, args: tuple, kwargs: dict) -> str:
        """Build a cache key based on the function's qualified name and its arguments.

        Args:
            func (Callable): The function.
            args (tuple): Positional arguments passed to the function.
            kwargs (dict): Keyword arguments passed to the function.

        Returns:
            str: A SHA1 hash of the function's qualified name and arguments as the cache key.
        """
        try:
            payload = json.dumps({"args": args, "kwargs": kwargs}, default=str, ensure_ascii=False)
        except Exception:
            payload = str((args, kwargs))
        basis = f"{func.__qualname__}|{payload}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()

    @staticmethod
    def _make_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
        """Build a W3C traceparent string.
        The format is 00-{trace_id}-{span_id}-{flags}. The IDs are normalized to hex-lowercase
        with lengths of 16/16 nibbles for simplicity.

        Args:
            trace_id (str): The trace ID.
            span_id (str): The span ID.
            sampled (bool, optional): Whether the trace is sampled. Defaults to True.

        Returns:
            str: A W3C traceparent string.
        """
        # Normalize to 16-byte (32 hex) trace id and 8-byte (16 hex) span id if shorter
        t = (trace_id.replace("-", "")[:32]).ljust(32, '0')
        s = (span_id.replace("-", "")[:16]).ljust(16, '0')
        flags = "01" if sampled else "00"
        return f"00-{t}-{s}-{flags}"

    @staticmethod
    def _parse_traceparent(tp: str) -> tuple:
        """Parse a W3C traceparent string and return the trace ID and parent span ID.

        Args:
            tp (str): The traceparent string.

        Returns:
            tuple: A tuple containing the trace ID and parent span ID if parsing succeeds, otherwise (None, None).
        """
        try:
            parts = tp.strip().split('-')
            if len(parts) >= 4 and len(parts[1]) >= 16 and len(parts[2]) >= 16:
                return parts[1], parts[2]
        except Exception:
            pass
        return None, None

    # -------- LLM Observability Decorators --------
    @staticmethod
    def observe_llm_generation(provider: str = "unknown", model: str = "unknown", task: str = "inference",
                               count_fn: Optional[Callable[[Any], dict]] = None) -> Callable:
        """Create a decorator to observe LLM generation.
        This decorator records metrics such as request count, latency, token count, etc.

        Args:
            provider (str, optional): The LLM provider. Defaults to "unknown".
            model (str, optional): The LLM model. Defaults to "unknown".
            task (str, optional): The task type. Defaults to "inference".
            count_fn (Callable[[Any], dict], optional): Function to extract token counts and latency from the result. Defaults to None.

        Returns:
            Callable: A decorator function.
        """
        def _decorator(func: Callable) -> Callable:
            """Decorate an LLM generation function to add observability metrics.

            Args:
                func (Callable): The function to be decorated.

            Returns:
                Callable: A wrapped function with observability metrics.
            """
            @wraps(func)
            def _wrapped(*args, **kwargs):
                """The wrapped function that adds observability metrics to an LLM generation function.

                Args:
                    *args: Positional arguments passed to the original function.
                    **kwargs: Keyword arguments passed to the original function.

                Returns:
                    Any: The result of the original function.

                Raises:
                    Exception: Raises any exception thrown by the original function after recording the error.
                """
                reg = PiscesLxCoreMetricsRegistry.instance()
                labels = {"provider": provider, "model": model, "task": task}

                # Define metrics
                req_c = reg.counter("llm.requests_total", help_text="Total LLM requests", labels=["provider", "model", "task"])
                err_c = reg.counter("llm.errors_total", help_text="Total LLM errors", labels=["provider", "model", "task"])
                fail_c = reg.counter("llm.failures_total", help_text="LLM failures by class", labels=["provider", "model", "task", "error_class"])
                finish_c = reg.counter("llm.finish_reason_total", help_text="LLM finish reason", labels=["provider", "model", "task", "reason"])
                tok_c = reg.counter("llm.tokens_total", help_text="Total LLM tokens", labels=["provider", "model", "task", "type"])
                lat_h = reg.histogram("llm.latency_ms", help_text="LLM end-to-end latency (ms)", labels=["provider", "model", "task"])
                pre_h = reg.histogram("llm.prefill_latency_ms", help_text="LLM prefill latency (ms)", labels=["provider", "model", "task"])
                dec_h = reg.histogram("llm.decode_latency_ms", help_text="LLM decode latency (ms)", labels=["provider", "model", "task"])
                tps_g = reg.gauge("llm.tokens_per_second", help_text="LLM tokens per second", labels=["provider", "model", "task"])

                # Record the start time and increment the request counter
                t0 = time.time()
                req_c.inc(1.0, labels=labels)

                try:
                    # Create an optional OTel span
                    with start_span("llm.generate", kind="internal", attributes={"provider": provider, "model": model, "task": task}):
                        result = func(*args, **kwargs)

                    # Calculate the end-to-end latency and record it
                    dur_ms = (time.time() - t0) * 1000.0
                    lat_h.observe(dur_ms, labels=labels)

                    # Extract token counts and latency
                    counts = {}
                    if count_fn is not None:
                        try:
                            counts = count_fn(result) or {}
                        except Exception:
                            counts = {}

                    # Use actual tokenizer for accurate token counting
                    if not counts:
                        input_text = kwargs.get("input_text") or kwargs.get("prompt")
                        
                        # Try to get actual token counts using tokenizer if available
                        try:
                            from transformers import AutoTokenizer
                            # Use a lightweight tokenizer for token counting
                            tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
                            
                            inp_tokens = 0
                            if isinstance(input_text, str) and input_text:
                                inp_tokens = len(tokenizer.encode(input_text, add_special_tokens=True))
                            
                            out_tokens = 0
                            if isinstance(result, str) and result:
                                out_tokens = len(tokenizer.encode(result, add_special_tokens=True))
                            elif isinstance(result, dict) and "text" in result and isinstance(result["text"], str):
                                out_tokens = len(tokenizer.encode(result["text"], add_special_tokens=True))
                            
                            counts = {
                                "input_tokens": inp_tokens,
                                "output_tokens": out_tokens,
                                "total_tokens": inp_tokens + out_tokens,
                            }
                        except Exception:
                            # Fallback to character-based estimation only if tokenizer fails
                            input_text = kwargs.get("input_text") or kwargs.get("prompt")
                            try:
                                inp_len = len(input_text) if isinstance(input_text, str) else 0
                            except Exception:
                                inp_len = 0

                            out_len = 0
                            try:
                                if isinstance(result, str):
                                    out_len = len(result)
                                elif isinstance(result, dict) and "text" in result and isinstance(result["text"], str):
                                    out_len = len(result["text"])
                            except Exception:
                                out_len = 0

                            # Improved character-to-token ratio (3.5 chars per token on average)
                            counts = {
                                "input_tokens": int(inp_len / 3.5) if inp_len else 0,
                                "output_tokens": int(out_len / 3.5) if out_len else 0,
                                "total_tokens": int(inp_len / 3.5) + int(out_len / 3.5),
                            }

                    it = int(counts.get("input_tokens", 0) or 0)
                    ot = int(counts.get("output_tokens", 0) or 0)
                    tt = int(counts.get("total_tokens", it + ot) or (it + ot))
                    pre_ms = float(counts.get("prefill_ms", 0.0) or 0.0)
                    dec_ms = float(counts.get("decode_ms", 0.0) or 0.0)

                    # Record token counts
                    if it:
                        tok_c.inc(it, labels={**labels, "type": "input"})
                    if ot:
                        tok_c.inc(ot, labels={**labels, "type": "output"})
                    if tt:
                        tok_c.inc(tt, labels={**labels, "type": "total"})

                    # Record prefill and decode latency
                    if pre_ms > 0:
                        pre_h.observe(pre_ms, labels=labels)
                    if dec_ms > 0:
                        dec_h.observe(dec_ms, labels=labels)

                    # Calculate and record tokens per second
                    try:
                        if dur_ms > 0:
                            tps_g.set(ot / (dur_ms / 1000.0), labels=labels)
                    except Exception:
                        pass

                    # Classify finish reason if available
                    try:
                        reason = None
                        if isinstance(result, dict):
                            reason = result.get("finish_reason") or result.get("stop_reason")
                        if isinstance(reason, str) and reason:
                            finish_c.inc(1.0, labels={**labels, "reason": reason})
                    except Exception:
                        pass

                    return result
                except Exception as e:
                    # Increment error counters
                    err_c.inc(1.0, labels=labels)
                    try:
                        fail_c.inc(1.0, labels={**labels, "error_class": type(e).__name__})
                    except Exception:
                        pass
                    raise

            return _wrapped
        return _decorator
