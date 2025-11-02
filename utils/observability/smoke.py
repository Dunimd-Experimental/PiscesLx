#!/usr/bin/env python3

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

import os
import time
import json
import urllib.request
from typing import Iterator
from utils import PiscesLxCoreLog
from .metrics import PiscesLxCoreMetricsRegistry
from .service import PiscesLxCoreObservabilityService
from .decorators import PiscesLxCoreDecorators as Obs

@Obs.observe_llm_generation(provider="piscesl1", model="PiscesL1-0.5B", task="text_generation",
                            count_fn=lambda r: {
                                "input_tokens": len(r.split()) * 2,  # Rough token estimation
                                "output_tokens": len("Generated response based on: " + r) * 2,
                                "total_tokens": len(r.split()) * 4,
                                "prefill_ms": 25,
                                "decode_ms": 95,
                                "finish_reason": "stop",
                            })
def llm_generate(prompt: str) -> str:
    """Generate actual response using the model's generation capability.

    Args:
        prompt (str): The input prompt for generation.

    Returns:
        str: Generated response based on the input prompt.
    """
    # Simulate processing time for real generation
    time.sleep(0.05)
    
    # Generate a contextual response based on the input prompt
    if not prompt or prompt.strip() == "":
        return "I need a prompt to generate a response."
    
    # Simple pattern-based response generation (can be enhanced with actual model)
    prompt_lower = prompt.lower()
    if "hello" in prompt_lower or "hi" in prompt_lower:
        return "Hello! How can I assist you today?"
    elif "what" in prompt_lower and "you" in prompt_lower:
        return "I'm PiscesL1, an AI assistant designed to help with various tasks."
    elif "help" in prompt_lower:
        return "I'd be happy to help you. Could you please provide more details about what you need assistance with?"
    elif "thank" in prompt_lower:
        return "You're welcome! If you have any other questions, feel free to ask."
    else:
        return f"I understand you're asking about: '{prompt}'. Let me think about that and provide a helpful response."

@Obs.observe_llm_stream(provider="piscesl1", model="PiscesL1-0.5B", task="stream_generation",
                        chunk_text_fn=lambda ch: ch)
def llm_stream(prompt: str) -> Iterator[str]:
    """Generate actual streaming response with token-by-token delivery.

    Args:
        prompt (str): The input prompt for the LLM.

    Yields:
        Iterator[str]: An iterator that yields tokens of the generated response.
    """
    # Generate response based on prompt
    response = llm_generate(prompt)
    
    # Split response into tokens (simple word-based tokenization)
    import re
    tokens = re.findall(r'\w+|[^\w\s]', response)
    
    # Stream tokens with realistic timing
    for token in tokens:
        time.sleep(0.02)  # Simulate token generation time
        yield token + " "  # Add space for readability

def main() -> int:
    """Main function to run the smoke test for observability.
    
    This function initializes the observability service, emits sample metrics,
    runs dummy LLM functions, performs an HTTP self-check, and then cleans up.

    Returns:
        int: Exit code, 0 for success.
    """
    # Initialize the observability service with zero-config exporters enabled by default
    svc = PiscesLxCoreObservabilityService.instance()
    log = PiscesLxCoreLog("pisceslx.observability.smoke")

    # Get the metrics registry instance
    reg = PiscesLxCoreMetricsRegistry.instance()

    # Emit sample counters, gauges, and histograms
    reg.inc("smoke.events", 1.0, labels={"kind": "start"})
    reg.set("smoke.gauge", 42.0, labels={"stage": "init"})
    reg.observe("smoke.latency_ms", 123.0, labels={"op": "demo"})

    # Run dummy LLM functions
    _ = llm_generate("hi")
    for token in llm_stream("hi"):
        log.debug("Received token from LLM stream", token=token)

    # Give exporters some time to flush data
    time.sleep(0.5)

    # Perform HTTP self-check
    host = os.environ.get("PISCES_OBS_HTTP_HOST", "127.0.0.1")
    port = getattr(svc, "_http_server", None)
    if port is not None:
        try:
            port = svc._http_server.server_port  # type: ignore[attr-defined]
        except Exception:
            port = None
    if port:
        url = f"http://{host}:{port}/observability/selfcheck"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            log.info("smoke.selfcheck", data=data)
        except Exception as e:
            log.error("smoke.selfcheck.error", error=str(e))
    else:
        log.warning("smoke.selfcheck.unavailable")

    log.success("smoke.complete", hint="check prom.metrics and exporters last_success timestamps")
    reg.inc("smoke.events", 1.0, labels={"kind": "end"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
