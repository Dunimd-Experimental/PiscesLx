#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
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

"""
PiscesL1 Large Language Model Framework

DMSC-based infrastructure with LLM-specific extensions.

Core Components (DMSC):
    - PiscesLxLogger: Structured logging (DMSC)
    - PiscesLxMetrics: Metrics collection (DMSC)
    - PiscesLxConfiguration: Configuration management (DMSC)
    - PiscesLxFilesystem: File operations (DMSC)
    - PiscesLxCoreContext: Service context (DMSC)

LLM-Specific Components (PiscesL1):
    - PiscesLxTokenManager: Token counting
    - PiscesLxPromptTemplate: Prompt templates
    - PiscesLxLLMClient: LLM client
    - PiscesLxSemanticCache: Semantic caching
    - PiscesLxBatchProcessor: Batch processing
"""

import asyncio
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Dict, List,
    Optional, Type,
)

sys.path.append(str(Path(__file__).parent.parent))

from configs.version import VERSION
from dmsc import (
    DMSCAppBuilder,
    DMSCLogConfig,
    DMSCServiceContext,
    DMSCPythonServiceModule,
    DMSCPythonAsyncServiceModule,
    DMSCHookKind,
    DMSCDevice,
    DMSCDeviceType,
    DMSCDeviceStatus,
    DMSCDeviceCapabilities,
    DMSCDeviceHealthMetrics,
    DMSCResourcePool,
    DMSCResourcePoolStatus,
    DMSCResourcePoolStatistics,
    DMSCConnectionPoolStatistics,
    DMSCSystemMetricsCollector,
    DMSCSystemMetrics,
    DMSCCPUMetrics,
    DMSCMemoryMetrics,
    DMSCDiskMetrics,
    DMSCNetworkMetrics,
)


class _PiscesLxRuntime:
    """Internal DMSC runtime manager - not exposed externally."""
    
    _instance: Optional['_PiscesLxRuntime'] = None
    _lock: threading.RLock = threading.RLock()
    _ctx: Optional[DMSCServiceContext] = None
    
    def __new__(cls) -> '_PiscesLxRuntime':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def context(cls) -> DMSCServiceContext:
        if cls._ctx is None:
            with cls._lock:
                if cls._ctx is None:
                    cls._ctx = DMSCAppBuilder().build().get_context()
        return cls._ctx
    
    @classmethod
    def reinitialize(cls, config_path: Optional[str] = None) -> DMSCServiceContext:
        with cls._lock:
            builder = DMSCAppBuilder()
            if config_path:
                builder = builder.with_config(config_path)
            builder = builder.with_logging(DMSCLogConfig.default())
            cls._ctx = builder.build().get_context()
        return cls._ctx


class PiscesLxCoreLogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PiscesLxCoreErrorCode(Enum):
    INTERNAL_ERROR = ("P-001", "Internal server error")
    CONFIG_PARSE_ERROR = ("P-002", "Configuration parse error")
    FILESYSTEM_ERROR = ("P-003", "Filesystem operation error")
    TOKEN_LIMIT_EXCEEDED = ("P-004", "Token limit exceeded")
    LLM_REQUEST_FAILED = ("P-005", "LLM request failed")
    VALIDATION_ERROR = ("P-006", "Validation error")
    SERVICE_UNAVAILABLE = ("P-007", "Service unavailable")
    TIMEOUT_ERROR = ("P-008", "Operation timeout")


@dataclass(frozen=True)
class PiscesLxCoreErrorContext:
    component: str = "unknown"
    operation: str = "unknown"
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class PiscesLxCoreError(Exception):
    def __init__(
        self,
        code: PiscesLxCoreErrorCode,
        message: Optional[str] = None,
        context: Optional[PiscesLxCoreErrorContext] = None
    ):
        self.code = code
        self.message = message or code.value[1]
        self.context = context or PiscesLxCoreErrorContext()
        super().__init__(self.message)


class PiscesLxLogger:
    """DMSC-based structured logging."""
    
    def __init__(
        self, 
        name: str, 
        context: Optional[Dict[str, Any]] = None,
        file_path: Optional[str] = None,
        enable_file: bool = False
    ):
        self.name = name
        self._context = context or {}
        self._ctx = _PiscesLxRuntime.context()
        self._file_path = file_path
        self._enable_file = enable_file
        self._file_handler = None
        
        if enable_file and file_path:
            self._init_file_handler(file_path)
    
    def _init_file_handler(self, file_path: str) -> None:
        """Initialize file handler for logging to file."""
        import os
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._file_handler = open(file_path, 'a', encoding='utf-8')
        except Exception:
            self._file_handler = None
    
    def _write_to_file(self, level: str, msg: str) -> None:
        """Write log message to file with DMSC-compatible format."""
        if self._file_handler:
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            level_fixed = level[:5].ljust(5)
            self._file_handler.write(f"{timestamp} | {level_fixed} | {self.name} | event={msg} | {msg}\n")
            self._file_handler.flush()
    
    def debug(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        logger = self._ctx.logger
        if hasattr(logger, 'debug'):
            logger.debug(self.name, formatted)
        else:
            print(f"[DEBUG] {self.name}: {formatted}")
        self._write_to_file("DEBUG", formatted)
    
    def info(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        logger = self._ctx.logger
        if hasattr(logger, 'info'):
            logger.info(self.name, formatted)
        else:
            print(f"[INFO] {self.name}: {formatted}")
        self._write_to_file("INFO", formatted)
    
    def warning(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        logger = self._ctx.logger
        if hasattr(logger, 'warn'):
            logger.warn(self.name, formatted)
        elif hasattr(logger, 'warning'):
            logger.warning(self.name, formatted)
        else:
            print(f"[WARNING] {self.name}: {formatted}")
        self._write_to_file("WARNING", formatted)
    
    def error(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        logger = self._ctx.logger
        if hasattr(logger, 'error'):
            logger.error(self.name, formatted)
        else:
            print(f"[ERROR] {self.name}: {formatted}")
        self._write_to_file("ERROR", formatted)
    
    def critical(self, msg: str, **kwargs) -> None:
        formatted = self._format(msg, **kwargs)
        logger = self._ctx.logger
        if hasattr(logger, 'critical'):
            logger.critical(self.name, formatted)
        elif hasattr(logger, 'error'):
            logger.error(self.name, formatted)
        else:
            print(f"[CRITICAL] {self.name}: {formatted}")
        self._write_to_file("CRITICAL", formatted)
    
    def _format(self, msg: str, **kwargs) -> str:
        if self._context:
            ctx_str = " ".join(f"{k}={v}" for k, v in self._context.items())
            msg = f"[{ctx_str}] {msg}"
        if kwargs:
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            msg = f"{msg} {extra}"
        return msg
    
    def with_context(self, **context) -> 'PiscesLxLogger':
        return PiscesLxLogger(self.name, {**self._context, **context})


class PiscesLxMetrics:
    """DMSC-based metrics collection."""
    
    def __init__(self):
        self._registry = _PiscesLxRuntime.context().metrics_registry()
    
    def counter(self, name: str, value: float = 1) -> float:
        from dmsc import DMSCMetricConfig, DMSCMetric, DMSCMetricType
        config = DMSCMetricConfig(name=name, metric_type=DMSCMetricType.Counter, help=f"Counter {name}")
        metric = DMSCMetric(config)
        self._registry.register(metric)
        metric.record(value)
        return value
    
    def gauge(self, name: str, value: float) -> None:
        from dmsc import DMSCMetricConfig, DMSCMetric, DMSCMetricType
        config = DMSCMetricConfig(name=name, metric_type=DMSCMetricType.Gauge, help=f"Gauge {name}")
        metric = DMSCMetric(config)
        self._registry.register(metric)
        metric.record(value)
    
    def histogram(self, name: str, value: float) -> None:
        from dmsc import DMSCMetricConfig, DMSCMetric, DMSCMetricType
        config = DMSCMetricConfig(name=name, metric_type=DMSCMetricType.Histogram, help=f"Histogram {name}")
        metric = DMSCMetric(config)
        self._registry.register(metric)
        metric.record(value)
    
    def timer(self, name: str, duration_ms: float) -> None:
        self.histogram(f"{name}_duration_ms", duration_ms)
    
    def export_prometheus(self) -> str:
        return self._registry.export_prometheus()


class PiscesLxConfiguration:
    """DMSC-based configuration management."""
    
    def __init__(self):
        self._config = _PiscesLxRuntime.context().config()
        self._local_configs: Dict[str, Any] = {}
    
    def load_from_dict(self, config: Dict[str, Any]) -> None:
        self._local_configs.update(config)
    
    def load_from_env(self, prefix: str = "PISCES_") -> None:
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self._local_configs[config_key] = value
    
    def load_from_file(self, file_path: str) -> None:
        self._config.add_file_source(file_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._local_configs
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                result = self._config.get(key)
                return result if result is not None else default
        return value
    
    def set(self, key: str, value: Any) -> None:
        self._local_configs[key] = value


class PiscesLxFilesystem:
    """DMSC-based filesystem operations."""
    
    def __init__(self, base_path: Optional[str] = None):
        self._fs = _PiscesLxRuntime.context().fs()
        self.base_path = base_path or os.getcwd()
    
    def read(self, path: str) -> str:
        return self._fs.read_text(path)
    
    def write(self, path: str, content: str) -> None:
        self._fs.atomic_write_text(path, content)
    
    def read_bytes(self, path: str) -> bytes:
        return self._fs.read_text(path).encode('utf-8')
    
    def write_bytes(self, path: str, content: bytes) -> None:
        self._fs.atomic_write_bytes(path, content)
    
    def exists(self, path: str) -> bool:
        return self._fs.exists(path)
    
    def delete(self, path: str) -> bool:
        try:
            self._fs.remove_file(path)
            return True
        except Exception:
            return False
    
    def list_dir(self, path: str) -> List[str]:
        return os.listdir(path)
    
    def mkdir(self, path: str) -> None:
        self._fs.safe_mkdir(path)


class PiscesLxCoreContext:
    """DMSC-based service context."""
    
    def __init__(self, name: str = "piscesl1"):
        self.name = name
        self._request_id: Optional[str] = None
        self._correlation_id: Optional[str] = None
    
    @classmethod
    def create(cls, name: str = "piscesl1") -> 'PiscesLxCoreContext':
        return cls(name)
    
    def __enter__(self) -> 'PiscesLxCoreContext':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.logger.error(f"Context exited with exception: {exc_val}")
    
    async def __aenter__(self) -> 'PiscesLxCoreContext':
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.logger.error(f"Async context exited with exception: {exc_val}")
    
    @property
    def logger(self) -> PiscesLxLogger:
        return PiscesLxLogger(self.name)
    
    @property
    def metrics(self) -> PiscesLxMetrics:
        return PiscesLxMetrics()
    
    @property
    def config(self) -> PiscesLxConfiguration:
        return PiscesLxConfiguration()
    
    @property
    def filesystem(self) -> PiscesLxFilesystem:
        return PiscesLxFilesystem()
    
    def set_request_id(self, request_id: str) -> None:
        self._request_id = request_id
    
    def get_request_id(self) -> Optional[str]:
        return self._request_id
    
    def set_correlation_id(self, correlation_id: str) -> None:
        self._correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        return self._correlation_id
    
    async def log_info(self, message: str, **kwargs) -> None:
        self.logger.info(message, **kwargs)
    
    async def log_error(self, message: str, **kwargs) -> None:
        self.logger.error(message, **kwargs)
    
    async def log_debug(self, message: str, **kwargs) -> None:
        self.logger.debug(message, **kwargs)


class PiscesLxTokenManager:
    """LLM token counting and management."""
    
    DEFAULT_TOKEN_LIMITS: Dict[str, int] = {}
    
    _instance: Optional['PiscesLxTokenManager'] = None
    _lock: threading.RLock = threading.RLock()
    
    def __new__(cls) -> 'PiscesLxTokenManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._limits = cls.DEFAULT_TOKEN_LIMITS.copy()
        return cls._instance
    
    def count_tokens(self, text: str, model: str = "pisces-l1") -> int:
        return len(text) // 4
    
    def get_limit(self, model: str) -> int:
        return self._limits.get(model, -1)
    
    def set_limit(self, model: str, limit: int) -> None:
        self._limits[model] = limit
    
    def is_within_limit(self, text: str, model: str = "pisces-l1") -> bool:
        limit = self.get_limit(model)
        if limit == -1:
            return True
        return self.count_tokens(text, model) <= limit


class PiscesLxPromptTemplate:
    """Prompt template engine."""
    
    def __init__(self, template: str):
        self.template = template
        self._variables = re.findall(r'\{(\w+)\}', template)
    
    def render(self, **kwargs) -> str:
        result = self.template
        for var in self._variables:
            if var in kwargs:
                result = result.replace(f'{{{var}}}', str(kwargs[var]))
        return result
    
    def get_variables(self) -> List[str]:
        return self._variables.copy()


@dataclass
class PiscesLxLLMResponse:
    content: str
    model: str
    token_count: int
    duration_ms: float
    finish_reason: str = "complete"
    metadata: Dict[str, Any] = field(default_factory=dict)


class PiscesLxLLMClient:
    """LLM client with retry logic."""
    
    def __init__(self, ctx: PiscesLxCoreContext, model: str = "pisces-l1"):
        self.ctx = ctx
        self.model = model
        self._token_manager = PiscesLxTokenManager()
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> PiscesLxLLMResponse:
        start_time = time.time()
        token_count = self._token_manager.count_tokens(prompt, self.model)
        duration_ms = (time.time() - start_time) * 1000
        
        return PiscesLxLLMResponse(
            content="",
            model=self.model,
            token_count=token_count,
            duration_ms=duration_ms,
        )
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        yield ""


class PiscesLxSemanticCache:
    """Semantic-based response caching."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self._cache: Dict[str, Any] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._similarity_threshold = similarity_threshold
        self._lock = threading.RLock()
    
    def get(self, embedding: List[float]) -> Optional[Any]:
        with self._lock:
            for key, cached_emb in self._embeddings.items():
                if self._cosine_similarity(embedding, cached_emb) >= self._similarity_threshold:
                    return self._cache[key]
            return None
    
    def set(self, embedding: List[float], response: Any) -> None:
        with self._lock:
            key = hashlib.md5(str(embedding).encode()).hexdigest()
            self._cache[key] = response
            self._embeddings[key] = embedding
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class PiscesLxBatchProcessor:
    """Batch processing with rate limiting."""
    
    def __init__(
        self,
        max_concurrent: int = 5,
        rate_limit: Optional[int] = None,
        timeout: float = 30.0
    ):
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_all(
        self,
        items: List[Any],
        processor: Callable[[Any], Awaitable[Any]]
    ) -> List[Any]:
        async def process_item(item):
            async with self._semaphore:
                return await processor(item)
        
        tasks = [process_item(item) for item in items]
        return await asyncio.gather(*tasks)


class PiscesLxRateLimiter:
    """Rate limiting using token bucket algorithm."""
    
    def __init__(self, rate: float = 10.0, capacity: int = 100):
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()
        self._lock = threading.RLock()
    
    def acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_update = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False
    
    async def acquire_async(self, tokens: int = 1) -> bool:
        return self.acquire(tokens)


class PiscesLxTracing:
    """Distributed tracing."""
    
    def __init__(self, service_name: str = "piscesl1"):
        self.service_name = service_name
        self._spans: Dict[str, List[Dict[str, Any]]] = {}
        self._current_span: Optional[str] = None
        self._lock = threading.RLock()
    
    def start_span(self, name: str, parent_id: Optional[str] = None) -> str:
        span_id = str(uuid.uuid4())
        span = {
            "id": span_id,
            "name": name,
            "parent_id": parent_id,
            "start_time": time.time(),
            "events": [],
        }
        with self._lock:
            if span_id not in self._spans:
                self._spans[span_id] = []
            self._spans[span_id].append(span)
            self._current_span = span_id
        return span_id
    
    def end_span(self, span_id: str) -> None:
        with self._lock:
            if span_id in self._spans:
                for span in self._spans[span_id]:
                    if span["id"] == span_id:
                        span["end_time"] = time.time()
                        span["duration_ms"] = (span["end_time"] - span["start_time"]) * 1000
                        break
    
    def add_event(self, span_id: str, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if span_id in self._spans:
                for span in self._spans[span_id]:
                    if span["id"] == span_id:
                        span["events"].append({
                            "name": name,
                            "timestamp": time.time(),
                            "attributes": attributes or {},
                        })
                        break


class PiscesLxHookKind(Enum):
    """Lifecycle hook types."""
    BEFORE_INIT = "before_init"
    AFTER_INIT = "after_init"
    BEFORE_START = "before_start"
    AFTER_START = "after_start"
    BEFORE_SHUTDOWN = "before_shutdown"
    AFTER_SHUTDOWN = "after_shutdown"


class PiscesLxHooks:
    """Lifecycle hooks management."""
    
    def __init__(self):
        self._hooks: Dict[PiscesLxHookKind, List[Callable[[PiscesLxCoreContext], Awaitable[None]]]] = {
            kind: [] for kind in PiscesLxHookKind
        }
    
    def register(self, hook_kind: PiscesLxHookKind, callback: Callable[[PiscesLxCoreContext], Awaitable[None]]) -> None:
        self._hooks[hook_kind].append(callback)
    
    async def emit(self, hook_kind: PiscesLxHookKind, ctx: PiscesLxCoreContext) -> None:
        for callback in self._hooks[hook_kind]:
            await callback(ctx)
    
    def on_before_init(self, callback: Callable[[PiscesLxCoreContext], Awaitable[None]]) -> None:
        self.register(PiscesLxHookKind.BEFORE_INIT, callback)
    
    def on_after_init(self, callback: Callable[[PiscesLxCoreContext], Awaitable[None]]) -> None:
        self.register(PiscesLxHookKind.AFTER_INIT, callback)
    
    def on_before_start(self, callback: Callable[[PiscesLxCoreContext], Awaitable[None]]) -> None:
        self.register(PiscesLxHookKind.BEFORE_START, callback)
    
    def on_after_start(self, callback: Callable[[PiscesLxCoreContext], Awaitable[None]]) -> None:
        self.register(PiscesLxHookKind.AFTER_START, callback)
    
    def on_before_shutdown(self, callback: Callable[[PiscesLxCoreContext], Awaitable[None]]) -> None:
        self.register(PiscesLxHookKind.BEFORE_SHUTDOWN, callback)
    
    def on_after_shutdown(self, callback: Callable[[PiscesLxCoreContext], Awaitable[None]]) -> None:
        self.register(PiscesLxHookKind.AFTER_SHUTDOWN, callback)


class PiscesLxServiceModule:
    """Base class for sync service modules."""
    
    def __init__(self, name: str):
        self._name = name
        self._ctx: Optional[PiscesLxCoreContext] = None
        self._is_critical = True
        self._priority = 0
        self._dependencies: List[str] = []
    
    def name(self) -> str:
        return self._name
    
    def is_critical(self) -> bool:
        return self._is_critical
    
    def priority(self) -> int:
        return self._priority
    
    def dependencies(self) -> List[str]:
        return self._dependencies
    
    def init(self, ctx) -> None:
        self._ctx = PiscesLxCoreContext(self._name)
        self.on_init(self._ctx)
    
    def start(self, ctx) -> None:
        if self._ctx:
            self.on_start(self._ctx)
    
    def shutdown(self, ctx) -> None:
        if self._ctx:
            self.on_shutdown(self._ctx)
    
    def on_init(self, ctx: PiscesLxCoreContext) -> None:
        """Called during module initialization. Override to implement custom init logic."""
        ctx.logger.debug(f"Module '{self._name}' initialized")
    
    def on_start(self, ctx: PiscesLxCoreContext) -> None:
        """Called when module starts. Override to implement custom start logic."""
        ctx.logger.debug(f"Module '{self._name}' started")
    
    def on_shutdown(self, ctx: PiscesLxCoreContext) -> None:
        """Called during module shutdown. Override to implement custom cleanup logic."""
        ctx.logger.debug(f"Module '{self._name}' shutdown")


class PiscesLxAsyncServiceModule:
    """Base class for async service modules."""
    
    def __init__(self, name: str):
        self._name = name
        self._ctx: Optional[PiscesLxCoreContext] = None
        self._is_critical = True
        self._priority = 0
        self._dependencies: List[str] = []
    
    def name(self) -> str:
        return self._name
    
    def is_critical(self) -> bool:
        return self._is_critical
    
    def priority(self) -> int:
        return self._priority
    
    def dependencies(self) -> List[str]:
        return self._dependencies
    
    async def init(self, ctx) -> None:
        self._ctx = PiscesLxCoreContext(self._name)
        await self.on_init(self._ctx)
    
    async def start(self, ctx) -> None:
        if self._ctx:
            await self.on_start(self._ctx)
    
    async def shutdown(self, ctx) -> None:
        if self._ctx:
            await self.on_shutdown(self._ctx)
    
    async def on_init(self, ctx: PiscesLxCoreContext) -> None:
        """Called during async module initialization. Override to implement custom init logic."""
        ctx.logger.debug(f"Async module '{self._name}' initialized")
    
    async def on_start(self, ctx: PiscesLxCoreContext) -> None:
        """Called when async module starts. Override to implement custom start logic."""
        ctx.logger.debug(f"Async module '{self._name}' started")
    
    async def on_shutdown(self, ctx: PiscesLxCoreContext) -> None:
        """Called during async module shutdown. Override to implement custom cleanup logic."""
        ctx.logger.debug(f"Async module '{self._name}' shutdown")


class PiscesLxDeviceType(Enum):
    """Device type enumeration."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    NPU = "npu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    ACCELERATOR = "accelerator"
    CUSTOM = "custom"


class PiscesLxDeviceStatus(Enum):
    """Device status enumeration."""
    UNKNOWN = "unknown"
    AVAILABLE = "available"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"
    ALLOCATED = "allocated"


@dataclass
class PiscesLxDeviceCapabilities:
    """Device capabilities structure."""
    compute_units: int = 0
    memory_gb: float = 0.0
    storage_gb: float = 0.0
    bandwidth_gbps: float = 0.0
    custom: Dict[str, float] = field(default_factory=dict)
    
    def meets_requirements(self, required: 'PiscesLxDeviceCapabilities') -> bool:
        if self.compute_units < required.compute_units:
            return False
        if self.memory_gb < required.memory_gb:
            return False
        if self.storage_gb < required.storage_gb:
            return False
        if self.bandwidth_gbps < required.bandwidth_gbps:
            return False
        return True
    
    @classmethod
    def from_dmsc(cls, caps: DMSCDeviceCapabilities) -> 'PiscesLxDeviceCapabilities':
        return cls(
            compute_units=caps.compute_units or 0,
            memory_gb=caps.memory_gb or 0.0,
            storage_gb=caps.storage_gb or 0.0,
            bandwidth_gbps=caps.bandwidth_gbps or 0.0,
            custom=getattr(caps, 'custom', {}) or {},
        )


@dataclass
class PiscesLxDeviceHealthMetrics:
    """Device health metrics structure."""
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    temperature_celsius: float = 0.0
    error_count: int = 0
    throughput: int = 0
    network_latency_ms: float = 0.0
    disk_iops: int = 0
    battery_level_percent: float = 100.0
    response_time_ms: float = 0.0
    uptime_seconds: int = 0
    
    @property
    def health_score(self) -> float:
        score = 100.0
        if self.cpu_usage_percent > 80:
            score -= (self.cpu_usage_percent - 80) * 0.5
        if self.memory_usage_percent > 80:
            score -= (self.memory_usage_percent - 80) * 0.5
        if self.temperature_celsius > 70:
            score -= (self.temperature_celsius - 70) * 0.3
        if self.error_count > 0:
            score -= min(self.error_count * 5, 30)
        return max(0.0, min(100.0, score))
    
    @property
    def is_healthy(self) -> bool:
        return self.health_score >= 50.0
    
    @classmethod
    def from_dmsc(cls, metrics: DMSCDeviceHealthMetrics) -> 'PiscesLxDeviceHealthMetrics':
        return cls(
            cpu_usage_percent=metrics.cpu_usage_percent,
            memory_usage_percent=metrics.memory_usage_percent,
            temperature_celsius=metrics.temperature_celsius,
            error_count=metrics.error_count,
            throughput=metrics.throughput,
            network_latency_ms=metrics.network_latency_ms,
            disk_iops=metrics.disk_iops,
            battery_level_percent=metrics.battery_level_percent,
            response_time_ms=metrics.response_time_ms,
            uptime_seconds=metrics.uptime_seconds,
        )


class PiscesLxDevice:
    """Device representation with type, status, capabilities, and health metrics."""
    
    def __init__(
        self,
        device_id: str,
        device_type: PiscesLxDeviceType = PiscesLxDeviceType.CPU,
        name: Optional[str] = None,
    ):
        self._id = device_id
        self._type = device_type
        self._name = name or device_id
        self._status = PiscesLxDeviceStatus.UNKNOWN
        self._capabilities = PiscesLxDeviceCapabilities()
        self._health_metrics = PiscesLxDeviceHealthMetrics()
        self._metadata: Dict[str, Any] = {}
    
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def device_type(self) -> PiscesLxDeviceType:
        return self._type
    
    @property
    def status(self) -> PiscesLxDeviceStatus:
        return self._status
    
    def set_status(self, status: PiscesLxDeviceStatus) -> None:
        self._status = status
    
    @property
    def capabilities(self) -> PiscesLxDeviceCapabilities:
        return self._capabilities
    
    def set_capabilities(self, caps: PiscesLxDeviceCapabilities) -> None:
        self._capabilities = caps
    
    @property
    def health_metrics(self) -> PiscesLxDeviceHealthMetrics:
        return self._health_metrics
    
    def set_health_metrics(self, metrics: PiscesLxDeviceHealthMetrics) -> None:
        self._health_metrics = metrics
    
    @property
    def health_score(self) -> float:
        return self._health_metrics.health_score
    
    @property
    def is_healthy(self) -> bool:
        return self._health_metrics.is_healthy and self._status not in [
            PiscesLxDeviceStatus.ERROR,
            PiscesLxDeviceStatus.OFFLINE,
            PiscesLxDeviceStatus.MAINTENANCE,
        ]
    
    @property
    def is_available(self) -> bool:
        return self._status == PiscesLxDeviceStatus.AVAILABLE and self.is_healthy
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
    
    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value
    
    @classmethod
    def from_dmsc(cls, device: DMSCDevice) -> 'PiscesLxDevice':
        type_map = {
            DMSCDeviceType.CPU: PiscesLxDeviceType.CPU,
            DMSCDeviceType.GPU: PiscesLxDeviceType.GPU,
            DMSCDeviceType.TPU: PiscesLxDeviceType.TPU,
            DMSCDeviceType.NPU: PiscesLxDeviceType.NPU,
            DMSCDeviceType.Memory: PiscesLxDeviceType.MEMORY,
            DMSCDeviceType.Storage: PiscesLxDeviceType.STORAGE,
            DMSCDeviceType.Network: PiscesLxDeviceType.NETWORK,
            DMSCDeviceType.Accelerator: PiscesLxDeviceType.ACCELERATOR,
            DMSCDeviceType.Custom: PiscesLxDeviceType.CUSTOM,
        }
        status_map = {
            DMSCDeviceStatus.Unknown: PiscesLxDeviceStatus.UNKNOWN,
            DMSCDeviceStatus.Available: PiscesLxDeviceStatus.AVAILABLE,
            DMSCDeviceStatus.Busy: PiscesLxDeviceStatus.BUSY,
            DMSCDeviceStatus.Error: PiscesLxDeviceStatus.ERROR,
            DMSCDeviceStatus.Offline: PiscesLxDeviceStatus.OFFLINE,
            DMSCDeviceStatus.Maintenance: PiscesLxDeviceStatus.MAINTENANCE,
            DMSCDeviceStatus.Degraded: PiscesLxDeviceStatus.DEGRADED,
            DMSCDeviceStatus.Allocated: PiscesLxDeviceStatus.ALLOCATED,
        }
        
        result = cls(
            device_id=device.id,
            device_type=type_map.get(device.device_type, PiscesLxDeviceType.CUSTOM),
            name=device.name,
        )
        result._status = status_map.get(device.status, PiscesLxDeviceStatus.UNKNOWN)
        result._capabilities = PiscesLxDeviceCapabilities.from_dmsc(device.capabilities)
        result._health_metrics = PiscesLxDeviceHealthMetrics.from_dmsc(device.health_metrics)
        return result


@dataclass
class PiscesLxResourcePoolStatus:
    """Resource pool status structure."""
    total_devices: int = 0
    available_devices: int = 0
    allocated_devices: int = 0
    total_capacity: float = 0.0
    available_capacity: float = 0.0
    allocated_capacity: float = 0.0
    utilization_percent: float = 0.0
    average_health_score: float = 0.0
    status_distribution: Dict[str, int] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        return self.available_devices > 0 or self.allocated_devices > 0
    
    @classmethod
    def from_dmsc(cls, status: DMSCResourcePoolStatus) -> 'PiscesLxResourcePoolStatus':
        return cls(
            total_devices=status.total_devices,
            available_devices=status.available_devices,
            allocated_devices=status.allocated_devices,
            total_capacity=status.total_capacity,
            available_capacity=status.available_capacity,
            allocated_capacity=status.allocated_capacity,
            utilization_percent=status.utilization_percent,
            average_health_score=status.average_health_score,
            status_distribution=getattr(status, 'status_distribution', {}) or {},
        )


@dataclass
class PiscesLxResourcePoolStatistics:
    """Resource pool statistics structure."""
    total_devices: int = 0
    available_devices: int = 0
    busy_devices: int = 0
    offline_devices: int = 0
    error_devices: int = 0
    total_compute_units: int = 0
    total_memory_gb: float = 0.0
    total_storage_gb: float = 0.0
    total_bandwidth_gbps: float = 0.0
    average_health_score: float = 0.0
    status_distribution: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def from_dmsc(cls, stats: DMSCResourcePoolStatistics) -> 'PiscesLxResourcePoolStatistics':
        return cls(
            total_devices=stats.total_devices,
            available_devices=stats.available_devices,
            busy_devices=stats.busy_devices,
            offline_devices=stats.offline_devices,
            error_devices=stats.error_devices,
            total_compute_units=stats.total_compute_units,
            total_memory_gb=stats.total_memory_gb,
            total_storage_gb=stats.total_storage_gb,
            total_bandwidth_gbps=stats.total_bandwidth_gbps,
            average_health_score=stats.average_health_score,
            status_distribution=getattr(stats, 'status_distribution', {}) or {},
        )


@dataclass
class PiscesLxConnectionPoolStatistics:
    """Connection pool statistics structure."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    unhealthy_connections: int = 0
    total_successful_ops: int = 0
    total_failed_ops: int = 0
    average_response_time_ms: float = 0.0
    health_check_interval_secs: int = 30
    last_health_check_secs: int = 0
    
    @property
    def health_percentage(self) -> float:
        if self.total_connections == 0:
            return 0.0
        return (self.total_connections - self.unhealthy_connections) / self.total_connections * 100
    
    @property
    def success_rate(self) -> float:
        total = self.total_successful_ops + self.total_failed_ops
        if total == 0:
            return 100.0
        return self.total_successful_ops / total * 100
    
    @classmethod
    def from_dmsc(cls, stats: DMSCConnectionPoolStatistics) -> 'PiscesLxConnectionPoolStatistics':
        return cls(
            total_connections=stats.total_connections,
            active_connections=stats.active_connections,
            idle_connections=stats.idle_connections,
            unhealthy_connections=stats.unhealthy_connections,
            total_successful_ops=stats.total_successful_ops,
            total_failed_ops=stats.total_failed_ops,
            average_response_time_ms=stats.average_response_time_ms,
            health_check_interval_secs=stats.health_check_interval_secs,
            last_health_check_secs=stats.last_health_check_secs,
        )


class PiscesLxResourcePool:
    """Resource pool for managing device resources."""
    
    def __init__(self, name: str):
        self._name = name
        self._devices: Dict[str, PiscesLxDevice] = {}
        self._lock = threading.RLock()
    
    @property
    def name(self) -> str:
        return self._name
    
    def add_device(self, device: PiscesLxDevice) -> None:
        with self._lock:
            self._devices[device.id] = device
    
    def remove_device(self, device_id: str) -> bool:
        with self._lock:
            if device_id in self._devices:
                del self._devices[device_id]
                return True
            return False
    
    def get_device(self, device_id: str) -> Optional[PiscesLxDevice]:
        return self._devices.get(device_id)
    
    def get_all_devices(self) -> List[PiscesLxDevice]:
        return list(self._devices.values())
    
    def get_available_devices(self) -> List[PiscesLxDevice]:
        return [d for d in self._devices.values() if d.is_available]
    
    def get_devices_by_type(self, device_type: PiscesLxDeviceType) -> List[PiscesLxDevice]:
        return [d for d in self._devices.values() if d.device_type == device_type]
    
    def get_devices_by_status(self, status: PiscesLxDeviceStatus) -> List[PiscesLxDevice]:
        return [d for d in self._devices.values() if d.status == status]
    
    def get_healthy_devices(self) -> List[PiscesLxDevice]:
        return [d for d in self._devices.values() if d.is_healthy]
    
    def get_status(self) -> PiscesLxResourcePoolStatus:
        with self._lock:
            devices = list(self._devices.values())
            total = len(devices)
            available = sum(1 for d in devices if d.status == PiscesLxDeviceStatus.AVAILABLE)
            allocated = sum(1 for d in devices if d.status == PiscesLxDeviceStatus.ALLOCATED)
            
            total_cap = sum(d.capabilities.compute_units for d in devices)
            available_cap = sum(d.capabilities.compute_units for d in devices if d.is_available)
            
            status_dist: Dict[str, int] = {}
            for d in devices:
                status_name = d.status.value
                status_dist[status_name] = status_dist.get(status_name, 0) + 1
            
            avg_health = 0.0
            if devices:
                avg_health = sum(d.health_score for d in devices) / len(devices)
            
            return PiscesLxResourcePoolStatus(
                total_devices=total,
                available_devices=available,
                allocated_devices=allocated,
                total_capacity=float(total_cap),
                available_capacity=float(available_cap),
                allocated_capacity=float(total_cap - available_cap),
                utilization_percent=(total_cap - available_cap) / total_cap * 100 if total_cap > 0 else 0.0,
                average_health_score=avg_health,
                status_distribution=status_dist,
            )
    
    def get_statistics(self) -> PiscesLxResourcePoolStatistics:
        status = self.get_status()
        devices = list(self._devices.values())
        
        return PiscesLxResourcePoolStatistics(
            total_devices=status.total_devices,
            available_devices=status.available_devices,
            busy_devices=sum(1 for d in devices if d.status == PiscesLxDeviceStatus.BUSY),
            offline_devices=sum(1 for d in devices if d.status == PiscesLxDeviceStatus.OFFLINE),
            error_devices=sum(1 for d in devices if d.status == PiscesLxDeviceStatus.ERROR),
            total_compute_units=sum(d.capabilities.compute_units for d in devices),
            total_memory_gb=sum(d.capabilities.memory_gb for d in devices),
            total_storage_gb=sum(d.capabilities.storage_gb for d in devices),
            total_bandwidth_gbps=sum(d.capabilities.bandwidth_gbps for d in devices),
            average_health_score=status.average_health_score,
            status_distribution=status.status_distribution,
        )
    
    def is_healthy(self) -> bool:
        return self.get_status().is_healthy
    
    def select_best_device(
        self,
        requirements: Optional[PiscesLxDeviceCapabilities] = None,
        device_type: Optional[PiscesLxDeviceType] = None,
    ) -> Optional[PiscesLxDevice]:
        candidates = self.get_available_devices()
        
        if device_type:
            candidates = [d for d in candidates if d.device_type == device_type]
        
        if requirements:
            candidates = [d for d in candidates if d.capabilities.meets_requirements(requirements)]
        
        if not candidates:
            return None
        
        return max(candidates, key=lambda d: d.health_score)
    
    @classmethod
    def from_dmsc(cls, pool: DMSCResourcePool) -> 'PiscesLxResourcePool':
        result = cls(pool.name)
        return result


class PiscesLxSystemMetrics:
    """System metrics wrapper with convenient access methods."""
    
    def __init__(self, metrics: DMSCSystemMetrics):
        self._metrics = metrics
    
    @property
    def timestamp(self) -> int:
        return self._metrics.timestamp()
    
    @property
    def cpu(self) -> DMSCCPUMetrics:
        return self._metrics.cpu()
    
    @property
    def memory(self) -> DMSCMemoryMetrics:
        return self._metrics.memory()
    
    @property
    def disk(self) -> DMSCDiskMetrics:
        return self._metrics.disk()
    
    @property
    def network(self) -> DMSCNetworkMetrics:
        return self._metrics.network()
    
    @property
    def cpu_usage_percent(self) -> float:
        return self.cpu.total_usage_percent
    
    @property
    def memory_usage_percent(self) -> float:
        return self.memory.usage_percent
    
    @property
    def disk_usage_percent(self) -> float:
        return self.disk.usage_percent
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cpu': {
                'total_usage_percent': self.cpu.total_usage_percent,
                'per_core_usage': list(self.cpu.per_core_usage),
                'context_switches': self.cpu.context_switches,
                'interrupts': self.cpu.interrupts,
            },
            'memory': {
                'total_bytes': self.memory.total_bytes,
                'used_bytes': self.memory.used_bytes,
                'free_bytes': self.memory.free_bytes,
                'usage_percent': self.memory.usage_percent,
                'swap_total_bytes': self.memory.swap_total_bytes,
                'swap_used_bytes': self.memory.swap_used_bytes,
                'swap_free_bytes': self.memory.swap_free_bytes,
                'swap_usage_percent': self.memory.swap_usage_percent,
            },
            'disk': {
                'total_bytes': self.disk.total_bytes,
                'used_bytes': self.disk.used_bytes,
                'free_bytes': self.disk.free_bytes,
                'usage_percent': self.disk.usage_percent,
            },
            'network': {
                'total_received_bytes': self.network.total_received_bytes,
                'total_transmitted_bytes': self.network.total_transmitted_bytes,
                'received_bytes_per_sec': self.network.received_bytes_per_sec,
                'transmitted_bytes_per_sec': self.network.transmitted_bytes_per_sec,
            },
        }


class PiscesLxSystemMonitor:
    """System monitor for collecting system metrics."""
    
    _instance: Optional['PiscesLxSystemMonitor'] = None
    _lock: threading.RLock = threading.RLock()
    
    def __new__(cls) -> 'PiscesLxSystemMonitor':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._collector = DMSCSystemMetricsCollector()
        return cls._instance
    
    def collect(self) -> PiscesLxSystemMetrics:
        return PiscesLxSystemMetrics(self._collector.collect())
    
    def refresh(self) -> None:
        self._collector.refresh()
    
    def get_cpu_usage(self) -> float:
        return self.collect().cpu_usage_percent
    
    def get_memory_usage(self) -> float:
        return self.collect().memory_usage_percent
    
    def get_disk_usage(self) -> float:
        return self.collect().disk_usage_percent
    
    def get_memory_info(self) -> DMSCMemoryMetrics:
        return self.collect().memory
    
    def get_cpu_info(self) -> DMSCCPUMetrics:
        return self.collect().cpu
    
    def get_disk_info(self) -> DMSCDiskMetrics:
        return self.collect().disk
    
    def get_network_info(self) -> DMSCNetworkMetrics:
        return self.collect().network


__all__ = [
    "PiscesLxCoreLogLevel",
    "PiscesLxCoreErrorCode",
    "PiscesLxCoreErrorContext",
    "PiscesLxCoreError",
    "PiscesLxLogger",
    "PiscesLxMetrics",
    "PiscesLxConfiguration",
    "PiscesLxFilesystem",
    "PiscesLxCoreContext",
    "PiscesLxTokenManager",
    "PiscesLxPromptTemplate",
    "PiscesLxLLMResponse",
    "PiscesLxLLMClient",
    "PiscesLxSemanticCache",
    "PiscesLxBatchProcessor",
    "PiscesLxRateLimiter",
    "PiscesLxTracing",
    "PiscesLxHookKind",
    "PiscesLxHooks",
    "PiscesLxServiceModule",
    "PiscesLxAsyncServiceModule",
    "PiscesLxDeviceType",
    "PiscesLxDeviceStatus",
    "PiscesLxDeviceCapabilities",
    "PiscesLxDeviceHealthMetrics",
    "PiscesLxDevice",
    "PiscesLxResourcePoolStatus",
    "PiscesLxResourcePoolStatistics",
    "PiscesLxConnectionPoolStatistics",
    "PiscesLxResourcePool",
    "PiscesLxSystemMetrics",
    "PiscesLxSystemMonitor",
]
