#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright © 2025-2026 Wenze Wei. All Rights Reserved.
#
# This file is part of Encre.
# The Encre project belongs to the Dunimd Team.
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

from __future__ import annotations
import asyncio
import json
import subprocess
from threading import Lock
from typing import Any


class EncreLSPClient:
    def __init__(self, server_name: str) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._initialized = False
        self._request_id = 0
        self._lock = Lock()
        self._server_name = server_name
        self._pending_requests: dict[int, asyncio.Future[Any]] = {}
        self._response_buffer: bytearray = bytearray()
        self._reader_task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event | None = None

    async def start(self, command: str, args: list[str], cwd: str) -> None:
        self._shutdown_event = asyncio.Event()
        self._process = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
        )
        self._reader_task = asyncio.create_task(self._read_responses())

    async def initialize(self, root_uri: str) -> dict[str, Any]:
        params = {
            "processId": None,
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {"linkSupport": True},
                    "references": {},
                    "documentSymbol": {
                        "hierarchicalDocumentSymbolSupport": True,
                    },
                },
            },
        }
        result = await self.send_request("initialize", params)
        self._initialized = True
        await self.send_notification("initialized", {})
        return result

    async def send_request(self, method: str, params: dict[str, Any]) -> Any:
        with self._lock:
            self._request_id += 1
            request_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending_requests[request_id] = future

        self._write_message(message)

        try:
            return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise

    async def send_notification(self, method: str, params: dict[str, Any]) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._write_message(message)

    async def stop(self) -> None:
        """Stop the LSP server. Deprecated: use close() instead."""
        await self.close()

    async def close(self) -> None:
        """Terminate the LSP subprocess, cancel reader task, close pipes."""
        if not self._process or self._process.stdin is None:
            return
        try:
            await self.send_request("shutdown", {})
        except Exception:
            pass
        self._write_message({"jsonrpc": "2.0", "method": "exit"})
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._shutdown_event:
            self._shutdown_event.set()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        # Close remaining pending futures
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
        # Close pipes
        if self._process.stdin:
            try:
                self._process.stdin.close()
            except Exception:
                pass
        if self._process.stdout:
            try:
                self._process.stdout.close()
            except Exception:
                pass
        if self._process.stderr:
            try:
                self._process.stderr.close()
            except Exception:
                pass

    def _write_message(self, message: dict[str, Any]) -> None:
        if not self._process or self._process.stdin is None:
            return
        content = json.dumps(message, ensure_ascii=False)
        content_bytes = content.encode("utf-8")
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n".encode("utf-8")
        with self._lock:
            self._process.stdin.write(header + content_bytes)
            self._process.stdin.flush()

    async def _read_responses(self) -> None:
        if not self._process or self._process.stdout is None:
            return

        loop = asyncio.get_running_loop()
        buffer = bytearray()
        stdout = self._process.stdout

        while True:
            if self._shutdown_event and self._shutdown_event.is_set():
                break

            try:
                chunk = await loop.run_in_executor(None, lambda: stdout.read(4096))
            except (ValueError, OSError):
                break

            if not chunk:
                if self._process.poll() is not None:
                    break
                await asyncio.sleep(0.01)
                continue

            buffer.extend(chunk)

            while True:
                header_end = buffer.find(b"\r\n\r\n")
                if header_end == -1:
                    break

                header = buffer[:header_end].decode("utf-8", errors="replace")
                content_length = 0
                for line in header.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        try:
                            content_length = int(line.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                        break

                if content_length <= 0:
                    buffer = buffer[header_end + 4:]
                    continue

                body_start = header_end + 4
                if len(buffer) < body_start + content_length:
                    break

                body_bytes = buffer[body_start : body_start + content_length]
                buffer = buffer[body_start + content_length:]

                try:
                    message = json.loads(body_bytes.decode("utf-8"))
                except json.JSONDecodeError:
                    continue

                self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and "result" in message:
            future = self._pending_requests.pop(message["id"], None)
            if future and not future.done():
                future.set_result(message["result"])
        elif "id" in message and "error" in message:
            future = self._pending_requests.pop(message["id"], None)
            if future and not future.done():
                future.set_exception(
                    RuntimeError(
                        f"LSP error {message['error'].get('code', 0)}: "
                        f"{message['error'].get('message', 'unknown')}"
                    )
                )
