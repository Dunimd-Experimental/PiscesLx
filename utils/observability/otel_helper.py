#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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
from contextlib import contextmanager
from typing import Any, Dict, Optional
import logging

# Get environment variable to determine if OpenTelemetry tracing is enabled
_OTEL_ENABLED = os.environ.get("PISCES_OTEL_ENABLE", "0").lower() in ("1", "true")

try:
    if _OTEL_ENABLED:
        from opentelemetry import trace as _otel_trace  # type: ignore
        from opentelemetry.trace import SpanKind as _SpanKind  # type: ignore
    else:
        _otel_trace = None  # type: ignore
        _SpanKind = None  # type: ignore
except ImportError:
    _OTEL_ENABLED = False
    _otel_trace = None  # type: ignore
    _SpanKind = None  # type: ignore
    logging.warning("Failed to import OpenTelemetry packages. Tracing will be disabled.")

def is_enabled() -> bool:
    """Check if OpenTelemetry tracing is enabled and the tracer is available.

    Returns:
        bool: True if tracing is enabled and the tracer is available, False otherwise.
    """
    return bool(_OTEL_ENABLED and _otel_trace is not None)

def get_tracer(instrumentation_name: str = "pisceslx.observability") -> Any:
    """Retrieve an OpenTelemetry tracer. If unavailable, return a no-op tracer.

    Args:
        instrumentation_name (str, optional): The name of the instrumentation. 
            Defaults to "pisceslx.observability".

    Returns:
        Any: An OpenTelemetry tracer if available, otherwise a no-op tracer instance.
    """
    if is_enabled():
        try:
            return _otel_trace.get_tracer(instrumentation_name)  # type: ignore
        except Exception as e:
            logging.warning(f"Failed to get OpenTelemetry tracer: {str(e)}. Using no-op tracer instead.")
    # No-op tracer class
    class _NoopTracer:
        def start_as_current_span(self, name: str, kind: Optional[Any] = None) -> Any:
            """Start a no-op span context.

            Args:
                name (str): The name of the span.
                kind (Optional[Any], optional): The kind of the span. Defaults to None.

            Returns:
                Any: A no-op context manager.
            """
            return _noop_context()
    return _NoopTracer()

@contextmanager
def _noop_context() -> Any:
    """Provide a no-op context manager.

    Yields:
        None: Always yields None.
    """
    yield None

@contextmanager
def start_span(name: str, kind: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None) -> Any:
    """Start a new tracing span. If tracing is unavailable, use a no-op span.

    Args:
        name (str): The name of the span.
        kind (Optional[str], optional): The kind of the span as a string. 
            Valid values are "internal", "server", "client", "producer", "consumer". Defaults to None.
        attributes (Optional[Dict[str, Any]], optional): The attributes to set on the span. Defaults to None.

    Yields:
        Any: An OpenTelemetry span if tracing is available, otherwise None.
    """
    if is_enabled():
        try:
            tracer = get_tracer()
            # Map kind string to OTel SpanKind if available
            _kind = None
            if kind and _SpanKind is not None:
                try:
                    kind_map = {
                        "internal": getattr(_SpanKind, "INTERNAL", None),
                        "server": getattr(_SpanKind, "SERVER", None),
                        "client": getattr(_SpanKind, "CLIENT", None),
                        "producer": getattr(_SpanKind, "PRODUCER", None),
                        "consumer": getattr(_SpanKind, "CONSUMER", None),
                    }
                    _kind = kind_map.get(kind.lower())
                except Exception as e:
                    logging.warning(f"Failed to map span kind: {str(e)}. Using None as span kind.")
                    _kind = None
            with tracer.start_as_current_span(name, kind=_kind) as span:  # type: ignore
                if attributes and span is not None:
                    try:
                        for k, v in attributes.items():
                            try:
                                span.set_attribute(k, v)  # type: ignore
                            except Exception as e:
                                logging.warning(f"Failed to set attribute {k}: {str(e)}")
                    except Exception as e:
                        logging.warning(f"Failed to set span attributes: {str(e)}")
                yield span
            return
        except Exception as e:
            logging.warning(f"Failed to start OpenTelemetry span: {str(e)}. Using no-op context instead.")
    # Fallback to no-op context
    with _noop_context():
        yield None
