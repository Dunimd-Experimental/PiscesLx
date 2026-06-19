#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of EnTA.
# The EnTA project belongs to the Dunimd Team.
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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.



"""
Local backend -- Hugging Face Transformers (CPU/GPU inference).

This backend runs models locally using the Hugging Face ``transformers``
library.  It supports any causal language model available on the Hugging
Face Hub, including Llama, Mistral, Qwen, DeepSeek, Gemma, Phi, and many
more.

Key characteristics:
- Fully offline, no API calls
- Supports CPU and GPU inference
- True token-by-token streaming via ``TextIteratorStreamer``
- Tool calling via text parsing (model-dependent)
- Context window determined by model configuration
- No built-in prompt caching or thinking support
- Requires significant local compute resources

Architecture:
- Uses ``transformers`` pipeline for model/tokenizer loading
- Streaming via HuggingFace ``TextIteratorStreamer`` (background thread)
- Tool calls parsed from generated text with balanced-brace JSON extraction
- Model and tokenizer are loaded lazily on first ``chat()`` call
- GPU memory is released on ``aclose()``

Note:
    This backend is designed for development and testing.  For production
    use, consider using :class:`OllamaBackend` or a cloud API backend for
    better performance and reliability.
"""

import json
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import Any

from enta.backends.base import BaseBackend
from enta.utils.types import (
    BackendEvent,
    create_backend_error,
    create_backend_finish,
    create_backend_text,
    create_backend_tool_call,
    create_backend_tool_call_delta,
)


class LocalBackend(BaseBackend):
    """Local backend using Hugging Face Transformers.

    Loads and runs any Hugging Face causal language model locally.  The
    model is loaded lazily on the first ``chat()`` call to avoid blocking
    initialisation.

    Streaming uses HuggingFace's ``TextIteratorStreamer`` to produce true
    token-by-token output.  A background thread runs ``model.generate()``
    while the async event loop consumes tokens from the streamer queue.

    Tool calling is detected from the tokenizer's chat template.  Models
    with function-calling templates (Llama 3.1+, Qwen 2.5, etc.) report
    ``supports_tool_calling() == True``.  Tool call JSON is extracted
    from generated text using balanced-brace parsing for reliability  # noqa: E402
    with nested arguments.

    Args:
        model_name: Hugging Face model ID (e.g., ``"meta-llama/Llama-3.2-3B"``).
            Defaults to ``"Qwen/Qwen2.5-1.5B-Instruct"``.
        device: Device to run inference on (``"cpu"``, ``"cuda"``, ``"auto"``).
            Defaults to ``"cpu"``.
        **kwargs: Additional arguments passed to ``transformers.pipeline``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        """Initialise the local backend.

        Args:
            model_name: Hugging Face model ID.  Defaults to
                ``"Qwen/Qwen2.5-1.5B-Instruct"``.
            device: Device for inference.  ``"cpu"``, ``"cuda"``, or ``"auto"``.
            **kwargs: Additional arguments for ``transformers.pipeline``.
        """
        self.model_name = model_name
        self.device = device
        self._pipeline_kwargs = kwargs
        self._model = None
        self._tokenizer = None
        self._pipe = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._tool_support: bool | None = None

    async def _ensure_model(self) -> None:
        """Lazy-load the model and tokenizer on first use.

        Uses ``transformers.pipeline`` with ``task="text-generation"`` to
        load the model.  The pipeline is configured with the specified device
        and any additional kwargs passed during initialisation.

        After loading, tool-calling support is detected from the tokenizer's
        chat template by checking for function-calling template markers.

        Raises:
            ImportError: If ``transformers`` or ``torch`` is not installed.
        """
        if self._pipe is not None:
            return
        try:
            from transformers import pipeline
            self._pipe = pipeline(
                "text-generation",
                model=self.model_name,
                device=self.device,
                **self._pipeline_kwargs,
            )
            self._model = self._pipe.model
            self._tokenizer = self._pipe.tokenizer
            self._detect_tool_support()
        except ImportError as e:
            raise ImportError(
                "transformers/torch not installed. Install with: pip install enta[local]"
            ) from e

    def _detect_tool_support(self) -> None:
        """Detect whether the loaded model supports tool calling.

        Checks the tokenizer's chat template for function-calling markers
        such as ``'tool_calls'``, ``'function'``, or ``'tools'``.
        """
        if self._tokenizer is None:
            self._tool_support = False
            return
        template = getattr(self._tokenizer, "chat_template", None)
        if template is None:
            self._tool_support = False
            return
        if isinstance(template, str):
            template_lower = template.lower()
            self._tool_support = any(
                marker in template_lower
                for marker in ("tool_calls", '"function"', '"tools"')
            )
        elif isinstance(template, list):
            template_str = json.dumps(template).lower()
            self._tool_support = any(
                marker in template_str
                for marker in ("tool_calls", '"function"', '"tools"')
            )
        else:
            self._tool_support = False

    def _parse_tool_calls_from_text(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls from generated text using balanced-brace extraction.

        Uses character-level brace counting to find complete JSON objects,
        which correctly handles nested arguments unlike simple regex.
        Attempts to parse multiple tool call objects from the text.

        Args:
            text: The generated text to parse.

        Returns:
            A list of parsed tool call dictionaries, each containing
            ``id``, ``name``, and ``arguments`` keys.
        """
        tool_calls: list[dict[str, Any]] = []
        pos = 0
        while pos < len(text):
            brace_start = text.find("{", pos)
            if brace_start == -1:
                break
            depth = 0
            in_string = False
            escape = False
            end = -1
            for i in range(brace_start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                pos = brace_start + 1
                continue
            candidate = text[brace_start:end]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_args = (
                        parsed.get("arguments")
                        or parsed.get("parameters", {})
                    )
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "name": parsed.get("name", ""),
                        "arguments": json.dumps(tool_args),
                    })
                elif isinstance(parsed, dict) and "function" in parsed:
                    func = parsed["function"]
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "name": func.get("name", ""),
                        "arguments": json.dumps(func.get("arguments", {})),
                    })
            except json.JSONDecodeError:
                pass
            pos = end
        return tool_calls

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        _tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stream: bool = True,
        _enable_caching: bool = False,
    ) -> AsyncGenerator[BackendEvent, None]:
        """Send a chat completion request and stream back events.

        Runs inference locally using the Hugging Face pipeline.  Streaming
        mode uses ``TextIteratorStreamer`` for true token-by-token output.

        Args:
            messages: Conversation history in OpenAI message format.
            tools: Optional tool definitions (used for tool call parsing).
            tool_choice: Tool selection strategy (``"auto"``, ``"any"``, ``"none"``).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stream: If True (default), uses token-by-token streaming.
            enable_caching: Not supported for local models (ignored).

        Yields:
            :class:`BackendText`, :class:`BackendToolCallDelta`,
            :class:`BackendToolCall`, :class:`BackendFinish`, or
            :class:`BackendError`.
        """
        try:
            await self._ensure_model()
        except ImportError as e:
            yield create_backend_error(str(e))
            return

        import asyncio

        prompt = self._format_messages(messages, tools)

        try:
            if stream:
                self._last_full_output = ""
                async for event in self._stream_generate(
                    prompt, max_tokens, temperature,
                ):
                    yield event
                full_output = self._last_full_output
            else:
                loop = asyncio.get_running_loop()
                outputs = await loop.run_in_executor(
                    self._executor,
                    lambda: self._pipe(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        return_full_text=False,
                    ),
                )
                text = outputs[0]["generated_text"]
                yield create_backend_text(text)
                full_output = text
                yield create_backend_finish("stop")
                return

            tool_calls = self._parse_tool_calls_from_text(full_output)
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    yield create_backend_tool_call_delta(i, "name", tc["name"])
                    yield create_backend_tool_call_delta(i, "arguments", tc["arguments"])
                    yield create_backend_tool_call(
                        id=tc["id"],
                        name=tc["name"],
                        arguments=tc["arguments"],
                    )
                yield create_backend_finish("tool_calls")

        except Exception as e:
            yield create_backend_error(str(e))

    async def _stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> AsyncGenerator[BackendEvent, None]:
        """Generate tokens one at a time using TextIteratorStreamer.

        Uses HuggingFace's TextIteratorStreamer with background threads
        for true token-by-token streaming without blocking the event loop.
        A generation thread runs ``model.generate()``, a feeder thread
        reads the streamer and pushes tokens into an ``asyncio.Queue``,
        and the async generator consumes from the queue.

        Yields BackendText events, then BackendFinish("stop").
        Stores the complete text in ``self._last_full_output``.
        """
        import asyncio

        from transformers import TextIteratorStreamer

        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = getattr(self._model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }

        gen_thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        gen_thread.start()

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _feed_queue() -> None:
            try:
                for text in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception:
                pass
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        feed_thread = Thread(target=_feed_queue)
        feed_thread.start()

        full_output = ""
        while True:
            text = await queue.get()
            if text is None:
                break
            full_output += text
            yield create_backend_text(text)

        gen_thread.join()
        feed_thread.join()
        self._last_full_output = full_output
        yield create_backend_finish("stop")

    def _format_messages(
        self,
        messages: list[dict[str, Any]],
        _tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Format messages into a prompt string for the local model.

        Uses the tokenizer's ``apply_chat_template`` method if available,
        otherwise falls back to a simple concatenation format.

        Args:
            messages: OpenAI-format message list.
            tools: Optional tool definitions (added to the system prompt).

        Returns:
            A formatted prompt string ready for model input.
        """
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                content = " ".join(text_parts)
            formatted += f"<|{role}|>\n{content}\n"
        formatted += "<|assistant|>\n"
        return formatted

    def supports_tool_calling(self) -> bool:
        """Return whether the loaded model supports tool calling.

        Tool calling support is determined by inspecting the tokenizer's
        chat template at load time.  Models with function-calling templates
        (Llama 3.1+, Qwen 2.5, etc.) return True.  If the model hasn't
        been loaded yet, returns False (conservative default).
        """
        if self._tool_support is not None:
            return self._tool_support
        return False

    def context_window_size(self) -> int:
        """Return the model's context window size from its configuration.

        Reads ``max_position_embeddings`` from the model config if available.
        Falls back to 4096 if the model is not loaded or the config lacks
        this attribute.
        """
        if self._model is not None and hasattr(self._model, "config"):
            config = self._model.config
            if hasattr(config, "max_position_embeddings"):
                return config.max_position_embeddings
        return 4096

    async def aclose(self) -> None:
        """Release model resources and GPU memory.

        Moves the model to CPU, deletes references, and clears GPU cache
        if CUDA is available.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._executor.shutdown, True)
        if self._model is not None:
            try:
                import torch
                self._model = self._model.to("cpu")
                del self._model
                del self._tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self._model = None
        self._tokenizer = None

    def supports_thinking(self) -> bool:
        """Local models do not natively support thinking tokens."""
        return False
