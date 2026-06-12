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

"""MCP (Model Context Protocol) client implementation.

Implements the standard JSON-RPC 2.0 based MCP protocol with support for:
- Two transport modes: stdio (subprocess) and HTTP (streamable HTTP with SSE)
- Full MCP lifecycle: initialize -> initialized -> tools/list -> tools/call
- Tool schema discovery and caching
- MCP resources (resources/list, resources/read)
- MCP prompts (prompts/list, prompts/get)
- Proper JSON-RPC 2.0 error codes
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import httpx

from encre.tools.base import EncreTool

logger = logging.getLogger("encre.tools.mcp")

# ──────────────────────────────────────────────────────────────────────
# JSON-RPC 2.0 constants
# ──────────────────────────────────────────────────────────────────────

JSONRPC_VERSION = "2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"
CLIENT_NAME = "encre-mcp-client"
CLIENT_VERSION = "1.0.0"

# Standard JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# MCP-specific error codes (reserved range -32000 to -32099)
SERVER_NOT_INITIALIZED = -32002

JSONRPC_ERROR_MESSAGES: dict[int, str] = {
    PARSE_ERROR: "Parse error",
    INVALID_REQUEST: "Invalid Request",
    METHOD_NOT_FOUND: "Method not found",
    INVALID_PARAMS: "Invalid params",
    INTERNAL_ERROR: "Internal error",
    SERVER_NOT_INITIALIZED: "Server not initialized",
}


# ──────────────────────────────────────────────────────────────────────
# JSON-RPC 2.0 message helpers
# ──────────────────────────────────────────────────────────────────────

def _make_request(method: str, params: dict[str, Any] | None = None,
                  request_id: int | str = 0) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request object."""
    msg: dict[str, Any] = {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


def _make_notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 notification (no id field)."""
    msg: dict[str, Any] = {
        "jsonrpc": JSONRPC_VERSION,
        "method": method,
    }
    if params is not None:
        msg["params"] = params
    return msg


def _is_valid_message(msg: dict[str, Any]) -> bool:
    """Check if a dict is a valid JSON-RPC 2.0 message."""
    return (
        isinstance(msg, dict)
        and msg.get("jsonrpc") == JSONRPC_VERSION
        and "method" in msg
    )


def _is_error_response(msg: dict[str, Any]) -> bool:
    """Check if a JSON-RPC message is an error response."""
    return "error" in msg and "id" in msg


def _is_success_response(msg: dict[str, Any]) -> bool:
    """Check if a JSON-RPC message is a success response."""
    return "result" in msg and "id" in msg


def _extract_error(msg: dict[str, Any]) -> tuple[int, str]:
    """Extract error code and message from a JSON-RPC error response."""
    error = msg.get("error", {})
    code = error.get("code", INTERNAL_ERROR)
    message = error.get("message", JSONRPC_ERROR_MESSAGES.get(code, "Unknown error"))
    return code, message


# ──────────────────────────────────────────────────────────────────────
# Transport layer
# ──────────────────────────────────────────────────────────────────────


class MCPTransport(ABC):
    """Abstract transport for MCP client-server communication."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish the transport connection."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Tear down the transport connection."""

    @abstractmethod
    async def send_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC message and return the response.

        Args:
            message: A complete JSON-RPC 2.0 message dict.

        Returns:
            The response message dict.

        Raises:
            MCPError: On transport or protocol errors.
        """

    @abstractmethod
    async def send_notification(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (fire-and-forget).

        Args:
            message: A JSON-RPC notification dict (no id field).
        """

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the transport is currently connected."""


class MCPError(Exception):
    """Base exception for MCP protocol errors."""


class MCPTransportError(MCPError):
    """Raised when the transport layer encounters an error."""


class MCPProtocolError(MCPError):
    """Raised for JSON-RPC protocol-level errors."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.data = data


# ──────────────────────────────────────────────────────────────────────
# Stdio transport (subprocess with Content-Length framing)
# ──────────────────────────────────────────────────────────────────────


class StdioTransport(MCPTransport):
    """MCP transport over stdin/stdout of a subprocess.

    Implements the MCP stdio framing protocol:
    - Messages are framed with a Content-Length header followed by \r\n\r\n
    - The JSON body follows the header.

    Example wire format::

        Content-Length: 42\r\n\r\n{"jsonrpc":"2.0","id":1,"result":{}}
    """

    def __init__(self, command: str | list[str], env: dict[str, str] | None = None,
                 cwd: str | None = None) -> None:
        """Initialize stdio transport.

        Args:
            command: Shell command string or list of args to spawn the MCP server.
            env: Optional environment variables to pass to the subprocess.
            cwd: Optional working directory for the subprocess.
        """
        self._raw_command = command
        self._env = env
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._buffer = b""
        self._connected = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected and self._process is not None and self._process.returncode is None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Spawn the MCP server subprocess and attach to its stdio."""
        if self._connected:
            return

        if isinstance(self._raw_command, str):
            if sys.platform == "win32":
                args = self._raw_command
            else:
                args = shlex.split(self._raw_command)
        else:
            args = list(self._raw_command)

        logger.debug("Spawning MCP server: %s", args)

        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self._env,
                cwd=self._cwd,
            )
        except FileNotFoundError as exc:
            raise MCPTransportError(
                f"MCP server command not found: {args[0] if args else self._raw_command}"
            ) from exc
        except Exception as exc:
            raise MCPTransportError(
                f"Failed to spawn MCP server: {exc}"
            ) from exc

        self._connected = True
        self._request_id = 0
        logger.info("MCP stdio transport connected (pid=%s)", self._process.pid)

    async def disconnect(self) -> None:
        """Terminate the MCP server subprocess."""
        if not self._connected or self._process is None:
            return

        logger.debug("Disconnecting MCP stdio transport (pid=%s)", self._process.pid)

        try:
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("MCP server did not terminate gracefully, killing")
                    self._process.kill()
                    await self._process.wait()
        except ProcessLookupError:
            pass  # Already exited
        except Exception as exc:
            logger.warning("Error while terminating MCP server: %s", exc)

        self._connected = False
        self._process = None
        self._buffer = b""

    # ------------------------------------------------------------------
    # Message I/O
    # ------------------------------------------------------------------

    async def send_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a request and wait for the matching response."""
        if not self.is_connected or self._process is None:
            raise MCPTransportError("Stdio transport is not connected")

        # Assign request ID
        self._request_id += 1
        message["id"] = self._request_id

        async with self._lock:
            await self._write_message(message)
            return await self._read_response(self._request_id)

    async def send_notification(self, message: dict[str, Any]) -> None:
        """Send a notification (no response expected)."""
        if not self.is_connected or self._process is None:
            raise MCPTransportError("Stdio transport is not connected")

        # Notifications have no id
        message.pop("id", None)

        async with self._lock:
            await self._write_message(message)

    # ------------------------------------------------------------------
    # Internal: write
    # ------------------------------------------------------------------

    async def _write_message(self, message: dict[str, Any]) -> None:
        """Write a JSON-RPC message to the subprocess stdin with framing."""
        assert self._process is not None
        assert self._process.stdin is not None

        body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self._process.stdin.write(header + body)
        await self._process.stdin.drain()

        logger.debug("Sent MCP request: id=%s method=%s", message.get("id", "<none>"), message.get("method"))

    # ------------------------------------------------------------------
    # Internal: read
    # ------------------------------------------------------------------

    async def _read_response(self, expected_id: int) -> dict[str, Any]:
        """Read and return the response matching *expected_id*.

        Reads framed messages from stdout until the matching response
        arrives.  Server-pushed notifications (no id) are logged and
        skipped.
        """
        assert self._process is not None
        assert self._process.stdout is not None

        while True:
            msg = await self._read_one_message()
            msg_id = msg.get("id")

            if msg_id is None:
                # Notification from server — handle/log it but keep reading
                logger.debug("Received server notification: method=%s", msg.get("method"))
                self._handle_server_notification(msg)
                continue

            if msg_id == expected_id:
                return msg

            # Mismatched id — should not happen in sequential usage
            logger.warning(
                "Received response with unexpected id=%s (expected %s)", msg_id, expected_id
            )
            # Return it anyway to avoid deadlocks
            return msg

    async def _read_one_message(self) -> dict[str, Any]:
        """Read a single Content-Length-framed message from stdout."""
        assert self._process is not None
        assert self._process.stdout is not None

        content_length: int | None = None

        while True:
            # Check if the server process has died
            if self._process.returncode is not None:
                stderr_data = b""
                if self._process.stderr:
                    try:
                        stderr_data = await self._process.stderr.read()
                    except Exception:
                        pass
                stderr_text = stderr_data.decode("utf-8", errors="replace")[:500] if stderr_data else ""
                raise MCPTransportError(
                    f"MCP server exited with code {self._process.returncode}. "
                    f"Stderr: {stderr_text}"
                )

            if content_length is None:
                # Search for Content-Length header in the buffer
                header_end = self._buffer.find(b"\r\n\r\n")
                if header_end == -1:
                    # Read more data
                    chunk = await self._process.stdout.read(4096)
                    if not chunk:
                        await asyncio.sleep(0.01)
                        continue
                    self._buffer += chunk
                    continue

                # Parse the header
                header = self._buffer[:header_end].decode("utf-8", errors="replace")
                self._buffer = self._buffer[header_end + 4:]  # strip header + \r\n\r\n

                # Extract Content-Length value
                content_length = self._parse_content_length(header)
                if content_length is None:
                    raise MCPTransportError(
                        f"Invalid MCP frame header (no Content-Length): {header!r}"
                    )

            # Wait until we have enough bytes for the body
            if len(self._buffer) < content_length:
                needed = content_length - len(self._buffer)
                chunk = await self._process.stdout.read(max(needed, 4096))
                if not chunk:
                    await asyncio.sleep(0.01)
                    continue
                self._buffer += chunk
                continue

            # Extract the JSON body
            body_bytes = self._buffer[:content_length]
            self._buffer = self._buffer[content_length:]
            content_length = None  # reset for next message

            try:
                msg = json.loads(body_bytes.decode("utf-8"))
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse MCP message JSON: %s", exc)
                raise MCPProtocolError(
                    PARSE_ERROR,
                    f"Failed to parse JSON-RPC message: {exc}",
                ) from exc

            return msg

    @staticmethod
    def _parse_content_length(header: str) -> int | None:
        """Parse Content-Length from an MCP frame header."""
        for line in header.split("\r\n"):
            line = line.strip()
            if line.lower().startswith("content-length:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return None
        return None

    def _handle_server_notification(self, msg: dict[str, Any]) -> None:
        """Handle a notification pushed by the MCP server.

        The MCP spec defines server→client notifications such as:
        ``notifications/tools/list_changed``, ``notifications/resources/list_changed``,
        ``notifications/prompts/list_changed``, etc.
        """
        method = msg.get("method", "")
        logger.debug("MCP server notification: %s params=%s", method, msg.get("params"))


# ──────────────────────────────────────────────────────────────────────
# HTTP transport (Streamable HTTP with SSE)
# ──────────────────────────────────────────────────────────────────────


class HttpTransport(MCPTransport):
    """MCP transport over HTTP with optional SSE streaming.

    Implements the MCP Streamable HTTP transport:
    - Client sends POST requests with JSON-RPC 2.0 bodies to the server endpoint.
    - Server can respond with ``Content-Type: application/json`` or
      ``Content-Type: text/event-stream`` for streaming responses.
    - Server→client notifications are delivered via SSE when the client
      establishes a streaming session.
    """

    def __init__(self, server_url: str,
                 timeout: float = 60.0,
                 headers: dict[str, str] | None = None) -> None:
        """Initialize HTTP transport.

        Args:
            server_url: The MCP server endpoint URL.
            timeout: HTTP request timeout in seconds.
            headers: Optional extra HTTP headers to include in requests.
        """
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout
        self._extra_headers = headers or {}
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._connected = False
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Create the HTTP client session."""
        if self._connected:
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout, connect=10.0),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "User-Agent": f"{CLIENT_NAME}/{CLIENT_VERSION}",
                **self._extra_headers,
            },
        )
        self._connected = True
        self._request_id = 0
        logger.info("MCP HTTP transport connected to %s", self._server_url)

    async def disconnect(self) -> None:
        """Close the HTTP client session."""
        if not self._connected or self._client is None:
            return

        logger.debug("Disconnecting MCP HTTP transport from %s", self._server_url)
        await self._client.aclose()
        self._client = None
        self._connected = False
        self._session_id = None

    # ------------------------------------------------------------------
    # Message I/O
    # ------------------------------------------------------------------

    async def send_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request via HTTP POST and return the response."""
        if not self.is_connected or self._client is None:
            raise MCPTransportError("HTTP transport is not connected")

        self._request_id += 1
        message["id"] = self._request_id

        async with self._lock:
            return await self._post_message(message)

    async def send_notification(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC notification via HTTP POST (fire-and-forget)."""
        if not self.is_connected or self._client is None:
            raise MCPTransportError("HTTP transport is not connected")

        message.pop("id", None)

        async with self._lock:
            await self._post_message(message, is_notification=True)

    # ------------------------------------------------------------------
    # Internal: HTTP POST
    # ------------------------------------------------------------------

    async def _post_message(self, message: dict[str, Any],
                            is_notification: bool = False) -> dict[str, Any]:
        """Post a JSON-RPC message and parse the HTTP response."""
        assert self._client is not None

        headers: dict[str, str] = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        request_body = json.dumps(message, ensure_ascii=False).encode("utf-8")
        logger.debug("HTTP POST to %s: id=%s method=%s",
                     self._server_url, message.get("id", "<none>"), message.get("method"))

        try:
            response = await self._client.post(
                self._server_url,
                content=request_body,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            raise MCPTransportError(
                f"MCP HTTP request timed out to {self._server_url}"
            ) from exc
        except httpx.ConnectError as exc:
            raise MCPTransportError(
                f"Failed to connect to MCP server at {self._server_url}: {exc}"
            ) from exc
        except Exception as exc:
            raise MCPTransportError(
                f"MCP HTTP transport error: {exc}"
            ) from exc

        # Track session ID (set by server on initialize response)
        session_id = response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id

        content_type = response.headers.get("Content-Type", "")

        # Notifications (fire-and-forget) may return 202 Accepted with no body
        if is_notification and not response.content:
            return {"_notification": True}

        # Empty body from any request — nothing to parse
        if not response.content:
            return {}

        if "text/event-stream" in content_type:
            return await self._parse_sse_response(response)
        elif "application/json" in content_type:
            return response.json()
        else:
            # Try to parse as JSON anyway
            try:
                return response.json()
            except Exception:
                response_text = response.text[:1000]
                raise MCPTransportError(
                    f"Unexpected MCP HTTP response (status={response.status_code}, "
                    f"content-type={content_type}): {response_text}"
                ) from None

    # ------------------------------------------------------------------
    # SSE stream parsing
    # ------------------------------------------------------------------

    async def _parse_sse_response(self, response: httpx.Response) -> dict[str, Any]:
        """Parse an SSE (text/event-stream) response into a single result.

        SSE format::

            data: <json>\n
            \n

        Multiple events may be streamed.  Events with non-null ``id`` fields
        are JSON-RPC responses/errors; others are server notifications.
        Returns the final JSON-RPC response or error event.
        """
        last_response: dict[str, Any] | None = None
        data_buffer = ""

        async for raw_line in response.aiter_lines():
            line = raw_line.rstrip("\n").rstrip("\r")

            if line == "":
                # Empty line = event delimiter
                if data_buffer:
                    try:
                        event_data = json.loads(data_buffer)
                    except json.JSONDecodeError:
                        data_buffer = ""
                        continue

                    data_buffer = ""

                    event_id = event_data.get("id")
                    if event_id is not None:
                        # This is a JSON-RPC response or error
                        last_response = event_data
                    elif "method" in event_data:
                        # Server → client notification
                        logger.debug("SSE server notification: method=%s", event_data.get("method"))
                    else:
                        # Some other event — might be a partial result; capture it
                        last_response = event_data

                continue

            if line.startswith("data:"):
                # Strip "data:" prefix (and optional leading space)
                payload = line[5:]
                if payload.startswith(" "):
                    payload = payload[1:]
                data_buffer += payload
            # Ignore event:, id:, retry:, and comment lines

        # Flush any remaining buffered data
        if data_buffer:
            try:
                last_response = json.loads(data_buffer)
            except json.JSONDecodeError:
                pass

        if last_response is None:
            raise MCPTransportError("SSE stream ended without a valid JSON-RPC response")

        return last_response


# ──────────────────────────────────────────────────────────────────────
# MCP Client — protocol-level logic
# ──────────────────────────────────────────────────────────────────────


class MCPClient:
    """Low-level MCP client implementing the JSON-RPC 2.0 protocol.

    Handles:
    - Session lifecycle (initialize → initialized)
    - Tool discovery (tools/list)
    - Tool execution (tools/call)
    - Resource access (resources/list, resources/read)
    - Prompt access (prompts/list, prompts/get)
    - Error propagation with proper JSON-RPC codes

    Typical usage::

        transport = StdioTransport("npx -y @anthropic/mcp-server-filesystem /tmp")
        async with MCPClient(transport) as client:
            tools = await client.list_tools()
            result = await client.call_tool("read_file", {"path": "/tmp/foo.txt"})
    """

    def __init__(self, transport: MCPTransport) -> None:
        self._transport = transport
        self._initialized = False
        self._server_info: dict[str, Any] = {}
        self._server_capabilities: dict[str, Any] = {}
        self._tools_cache: dict[str, dict[str, Any]] | None = None
        self._tools_list: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "MCPClient":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> dict[str, Any]:
        """Perform the MCP initialize handshake.

        1. Connect the transport.
        2. Send ``initialize`` request.
        3. Send ``notifications/initialized`` notification.

        Returns:
            The ``initialize`` response result dict containing server info
            and capabilities.
        """
        if self._initialized:
            return self._server_info

        # Connect transport
        await self._transport.connect()

        # Send initialize request
        init_params = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
            },
            "clientInfo": {
                "name": CLIENT_NAME,
                "version": CLIENT_VERSION,
            },
        }

        logger.info("Sending MCP initialize request (protocol=%s)", MCP_PROTOCOL_VERSION)
        response = await self._send_request("initialize", init_params)

        result = response.get("result", {})
        self._server_info = result
        self._server_capabilities = result.get("capabilities", {})
        self._initialized = True

        logger.info(
            "MCP initialized: server=%s v%s, capabilities=%s",
            result.get("serverInfo", {}).get("name", "unknown"),
            result.get("protocolVersion", "unknown"),
            list(self._server_capabilities.keys()),
        )

        # Send initialized notification
        await self._send_notification("notifications/initialized")

        return result

    async def close(self) -> None:
        """Disconnect the transport and release resources."""
        if self._transport.is_connected:
            await self._transport.disconnect()
        self._initialized = False
        self._tools_cache = None
        self._tools_list = None

    # ------------------------------------------------------------------
    # Tool operations
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """Discover available tools from the MCP server.

        Calls ``tools/list`` and caches the result.  Subsequent calls
        return the cached value.  Use ``invalidate_cache()`` to force
        a re-fetch.

        Returns:
            A list of tool schema dicts, each with ``name``, ``description``,
            and ``inputSchema`` keys.
        """
        self._ensure_initialized()

        if self._tools_list is not None:
            return self._tools_list

        response = await self._send_request("tools/list")
        tools: list[dict[str, Any]] = response.get("result", {}).get("tools", [])

        self._tools_list = tools
        self._tools_cache = {t["name"]: t for t in tools}
        logger.info("Discovered %d tools from MCP server", len(tools))
        return tools

    async def get_tool_schema(self, name: str) -> dict[str, Any] | None:
        """Get the schema for a specific tool by name."""
        if self._tools_cache is None:
            await self.list_tools()
        return self._tools_cache.get(name) if self._tools_cache else None

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Call a tool on the MCP server.

        Args:
            name: The tool name.
            arguments: Tool arguments dict.

        Returns:
            The tool result content as a list of content item dicts.
            Each item has a ``type`` key (``"text"``, ``"image"``, ``"resource"``)
            and corresponding content fields.
        """
        self._ensure_initialized()

        params: dict[str, Any] = {"name": name}
        if arguments is not None:
            params["arguments"] = arguments

        logger.debug("Calling MCP tool: %s", name)
        response = await self._send_request("tools/call", params)

        result = response.get("result", {})
        content: list[dict[str, Any]] = result.get("content", [])

        # Check for isError flag
        if result.get("isError"):
            text_parts = [
                item.get("text", "")
                for item in content
                if item.get("type") == "text"
            ]
            error_text = "\n".join(text_parts)
            raise MCPError(f"MCP tool '{name}' returned an error: {error_text}")

        return content

    def invalidate_cache(self) -> None:
        """Clear cached tool schemas so the next ``list_tools()`` call
        re-fetches from the server."""
        self._tools_cache = None
        self._tools_list = None

    # ------------------------------------------------------------------
    # Resource operations
    # ------------------------------------------------------------------

    async def list_resources(self) -> list[dict[str, Any]]:
        """List available resources from the MCP server.

        Returns:
            A list of resource descriptors.
        """
        self._ensure_initialized()
        response = await self._send_request("resources/list")
        return response.get("result", {}).get("resources", [])

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """Read a resource from the MCP server.

        Args:
            uri: The resource URI to read.

        Returns:
            The resource content dict.
        """
        self._ensure_initialized()
        response = await self._send_request("resources/read", {"uri": uri})
        return response.get("result", {})

    # ------------------------------------------------------------------
    # Prompt operations
    # ------------------------------------------------------------------

    async def list_prompts(self) -> list[dict[str, Any]]:
        """List available prompts from the MCP server.

        Returns:
            A list of prompt descriptors.
        """
        self._ensure_initialized()
        response = await self._send_request("prompts/list")
        return response.get("result", {}).get("prompts", [])

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Retrieve a prompt from the MCP server.

        Args:
            name: The prompt name.
            arguments: Optional arguments for the prompt template.

        Returns:
            The prompt result dict containing ``messages`` and optional
            ``description``.
        """
        self._ensure_initialized()
        params: dict[str, Any] = {"name": name}
        if arguments:
            params["arguments"] = arguments
        response = await self._send_request("prompts/get", params)
        return response.get("result", {})

    # ------------------------------------------------------------------
    # Generic request helpers
    # ------------------------------------------------------------------

    async def _send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and return the parsed response.

        Raises:
            MCPProtocolError: If the server returned a JSON-RPC error.
        """
        message = _make_request(method, params)
        response = await self._transport.send_message(message)

        if _is_error_response(response):
            code, msg = _extract_error(response)
            raise MCPProtocolError(code, msg, response.get("error", {}).get("data"))

        return response

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification."""
        message = _make_notification(method, params)
        await self._transport.send_notification(message)

    def _ensure_initialized(self) -> None:
        """Raise if the client has not been initialized."""
        if not self._initialized:
            raise MCPError(
                "MCP client not initialized. Call initialize() first or use as async context manager."
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def server_info(self) -> dict[str, Any]:
        """Information about the connected MCP server."""
        return self._server_info

    @property
    def server_capabilities(self) -> dict[str, Any]:
        """Capabilities advertised by the MCP server."""
        return self._server_capabilities

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# ──────────────────────────────────────────────────────────────────────
# EncreTool adapter — bridge between MCP and encre tool registry
# ──────────────────────────────────────────────────────────────────────


class EncreMCPTool(EncreTool):
    """Encre tool that proxies calls to an MCP server using JSON-RPC 2.0.

    Supports two transport modes:

    **Stdio mode** (subprocess)::

        tool = EncreMCPTool(command="npx -y @anthropic/mcp-server-filesystem /tmp")
        await tool._connect()
        tools = await tool.discover_tools()
        result = await tool.call_tool("read_file", {"path": "/tmp/foo.txt"})

    **HTTP mode** (streamable HTTP)::

        tool = EncreMCPTool(server_url="http://localhost:8000/mcp")
        await tool._connect()
        result = await tool.execute(tool_name="some_tool", arguments={...})

    The ``execute`` method implements the standard ``EncreTool`` interface so the
    tool can be used via the encre tool registry like any other tool.
    """

    name: ClassVar[str] = "mcp"
    description: ClassVar[str] = (
        "Call a tool exposed via the Model Context Protocol (MCP). "
        "MCP servers can provide filesystem access, database queries, "
        "API integrations, and other capabilities."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "server_url": {
                "type": "string",
                "description": (
                    "The MCP server URL (for HTTP transport). "
                    "Ignored when the tool was initialized with a command."
                ),
            },
            "tool_name": {
                "type": "string",
                "description": "The name of the MCP tool to call.",
            },
            "arguments": {
                "type": "object",
                "description": "The arguments to pass to the MCP tool.",
            },
        },
        "required": ["tool_name"],
    }

    def __init__(self, command: str = "", server_url: str = "",
                 env: dict[str, str] | None = None,
                 cwd: str | None = None,
                 http_timeout: float = 60.0,
                 auto_connect: bool = True,
                 **kwargs: Any) -> None:
        """Initialize the MCP tool.

        Args:
            command: Shell command (or list of args) to spawn an MCP server
                     via stdio.  Example: ``"npx -y @modelcontextprotocol/server-filesystem /tmp"``.
            server_url: URL of an MCP HTTP server.  Example: ``"http://localhost:8000/mcp"``.
            env: Extra environment variables for the subprocess (stdio mode only).
            cwd: Working directory for the subprocess (stdio mode only).
            http_timeout: HTTP request timeout in seconds (HTTP mode only).
            auto_connect: If True (default), lazily connect on first use.
                          If False, the caller must call ``_connect()`` explicitly.
        """
        super().__init__()
        self._command = command
        self._server_url = server_url
        self._env = env
        self._cwd = cwd
        self._http_timeout = http_timeout
        self._auto_connect = auto_connect
        self._client: MCPClient | None = None
        self._connection_lock = asyncio.Lock()

        # Determine the transport mode
        if command:
            self._transport: MCPTransport = StdioTransport(command, env=env, cwd=cwd)
        elif server_url:
            self._transport = HttpTransport(server_url, timeout=http_timeout)
        else:
            # No transport configured — will be set up lazily when
            # execute() provides a server_url (backward-compat).
            self._transport = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Establish the MCP connection and perform initialize handshake.

        This is called automatically on first use if ``auto_connect=True``.
        Call it explicitly when ``auto_connect=False``.
        """
        if self._client is not None and self._client.is_initialized:
            return

        async with self._connection_lock:
            # Double-check after acquiring lock
            if self._client is not None and self._client.is_initialized:
                return

            if self._transport is None:
                raise MCPError(
                    "Cannot connect: no transport configured. "
                    "Provide a 'command' or 'server_url' to the constructor, "
                    "or pass 'server_url' to execute()."
                )

            self._client = MCPClient(self._transport)
            await self._client.initialize()
            logger.info("EncreMCPTool connected to MCP server")

    async def _disconnect(self) -> None:
        """Close the MCP connection and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("EncreMCPTool disconnected")

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    async def discover_tools(self) -> list[dict[str, Any]]:
        """Discover available tools from the MCP server.

        Calls ``tools/list`` on the server and caches the results.
        Subsequent calls return the cached schemas.

        Returns:
            A list of tool schema dicts.  Each dict has:
            - ``name`` (str): tool name
            - ``description`` (str): human-readable description
            - ``inputSchema`` (dict): JSON Schema for the tool's input
        """
        if self._client is None:
            await self._connect()
        assert self._client is not None
        return await self._client.list_tools()

    # ------------------------------------------------------------------
    # Tool execution (low-level)
    # ------------------------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Call a named tool on the MCP server.

        Args:
            name: The MCP tool name.
            arguments: Tool arguments.

        Returns:
            The tool's text output as a string.  If the tool returns
            multiple content items, they are joined with newlines.
        """
        if self._client is None:
            await self._connect()
        assert self._client is not None

        content = await self._client.call_tool(name, arguments)

        # Convert MCP content items to a single string
        text_parts: list[str] = []
        for item in content:
            item_type = item.get("type", "text")
            if item_type == "text":
                text_parts.append(item.get("text", ""))
            elif item_type == "image":
                mime = item.get("mimeType", "image/png")
                data = item.get("data", "")
                text_parts.append(f"[Image: {mime}, data={len(data)} bytes]")
            elif item_type == "resource":
                resource = item.get("resource", {})
                text_parts.append(f"[Resource: {resource.get('uri', 'unknown')}]")
            else:
                text_parts.append(json.dumps(item, ensure_ascii=False))

        return "\n".join(text_parts)

    # ------------------------------------------------------------------
    # EncreTool.execute — standard tool interface
    # ------------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> str:
        """Execute an MCP tool call (standard ``EncreTool`` interface).

        Accepts:
            tool_name (str): The MCP tool name to call. Required.
            arguments (dict): The tool arguments. Optional.
            server_url (str): MCP server URL for one-off HTTP connections.
                Ignored if the tool was initialized with ``command``.
        """
        tool_name = kwargs.get("tool_name", "")
        arguments = kwargs.get("arguments", {})
        server_url = kwargs.get("server_url", "")

        if not tool_name:
            return "Error: No MCP tool_name provided"

        # If a server_url is passed and the tool has no pre-configured
        # transport, create a temporary HTTP transport (backward compat).
        if self._transport is None and server_url:
            temp_transport = HttpTransport(server_url, timeout=self._http_timeout)
            temp_client = MCPClient(temp_transport)
            try:
                await temp_client.initialize()
                content = await temp_client.call_tool(tool_name, arguments)
            except MCPProtocolError as exc:
                return f"Error: MCP protocol error [{exc.code}]: {exc.message}"
            except MCPError as exc:
                return f"Error: MCP error: {exc}"
            except Exception as exc:
                return f"Error calling MCP tool '{tool_name}': {exc}"
            finally:
                await temp_client.close()

            return self._format_content(content)

        # Normal path: use the established (or auto-connect) client
        try:
            result = await self.call_tool(tool_name, arguments)
            return result
        except MCPProtocolError as exc:
            return f"Error: MCP protocol error [{exc.code}]: {exc.message}"
        except MCPTransportError as exc:
            return f"Error: MCP transport error: {exc}"
        except MCPError as exc:
            return f"Error: MCP error: {exc}"
        except Exception as exc:
            return f"Error calling MCP tool '{tool_name}': {exc}"

    # ------------------------------------------------------------------
    # Resource / Prompt access (convenience methods)
    # ------------------------------------------------------------------

    async def list_resources(self) -> list[dict[str, Any]]:
        """List resources available on the MCP server."""
        if self._client is None:
            await self._connect()
        assert self._client is not None
        return await self._client.list_resources()

    async def read_resource(self, uri: str) -> str:
        """Read a resource from the MCP server. Returns text content."""
        if self._client is None:
            await self._connect()
        assert self._client is not None

        result = await self._client.read_resource(uri)
        contents = result.get("contents", [])
        return self._format_content(contents)

    async def list_prompts(self) -> list[dict[str, Any]]:
        """List prompts available on the MCP server."""
        if self._client is None:
            await self._connect()
        assert self._client is not None
        return await self._client.list_prompts()

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Retrieve a prompt from the MCP server."""
        if self._client is None:
            await self._connect()
        assert self._client is not None
        return await self._client.get_prompt(name, arguments)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_content(content: list[dict[str, Any]]) -> str:
        """Format MCP content items as a human-readable string."""
        parts: list[str] = []
        for item in content:
            item_type = item.get("type", "text")
            if item_type == "text":
                parts.append(item.get("text", ""))
            elif item_type == "image":
                mime = item.get("mimeType", "image/png")
                data_len = len(item.get("data", ""))
                parts.append(f"[Image: {mime}, {data_len} bytes]")
            elif item_type == "resource":
                uri = item.get("resource", {}).get("uri", "unknown")
                parts.append(f"[Resource: {uri}]")
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)

    def is_concurrency_safe(self, input_data: dict[str, Any]) -> bool:
        """Stdio transport is not concurrency-safe; HTTP is.

        Returns True only for HTTP transport, since subprocess stdin/stdout
        message framing requires sequential access.
        """
        if isinstance(self._transport, HttpTransport):
            return True
        return False

    # ------------------------------------------------------------------
    # Convenience: register all discovered tools into a registry
    # ------------------------------------------------------------------

    async def register_with(self, registry: Any, prefix: str = "mcp__") -> list[str]:
        """Discover tools and register each as an individual tool in *registry*.

        Each discovered MCP tool is wrapped in an ``_MCPDiscoveredTool`` instance
        and registered under ``{prefix}{tool_name}``.

        Args:
            registry: A ``ToolRegistry`` instance.
            prefix: Prefix to prepend to each registered tool name.

        Returns:
            A list of the registered tool names.
        """
        tools = await self.discover_tools()
        registered: list[str] = []

        for tool_schema in tools:
            tool_name = tool_schema["name"]
            registered_name = f"{prefix}{tool_name}"

            wrapper = _MCPDiscoveredTool(
                mcp_tool=self,
                tool_name=tool_name,
                schema=tool_schema,
                registered_name=registered_name,
            )
            registry.register(wrapper)
            registered.append(registered_name)

        logger.info("Registered %d MCP tools into registry with prefix '%s'", len(registered), prefix)
        return registered


# ──────────────────────────────────────────────────────────────────────
# Internal: individual discovered tool wrapper
# ──────────────────────────────────────────────────────────────────────


class _MCPDiscoveredTool(EncreTool):
    """A single MCP-discovered tool wrapped as a EncreTool.

    This is created automatically by ``EncreMCPTool.register_with()``.
    Each instance proxies to its parent ``EncreMCPTool`` for execution.
    """

    def __init__(self, mcp_tool: EncreMCPTool, tool_name: str,
                 schema: dict[str, Any], registered_name: str) -> None:
        super().__init__()
        self._mcp_tool = mcp_tool
        self._tool_name = tool_name
        self._registered_name = registered_name

        # Set the ClassVar-like attributes (these are instance overrides
        # but EncreTool uses ClassVar… we set them as instance attrs which
        # the base class methods will read correctly via self.*)
        self.name = registered_name  # type: ignore[misc]
        self.description = schema.get("description", f"MCP tool: {tool_name}")  # type: ignore[misc]
        self.input_schema = schema.get("inputSchema", {"type": "object", "properties": {}})  # type: ignore[misc]

    async def execute(self, **kwargs: Any) -> str:
        """Proxy execution to the parent MCP tool."""
        # Filter out any kwargs not in the schema
        schema_props = self.input_schema.get("properties", {})
        filtered_args = {k: v for k, v in kwargs.items() if k in schema_props}
        return await self._mcp_tool.call_tool(self._tool_name, filtered_args)

    def is_concurrency_safe(self, input_data: dict[str, Any]) -> bool:
        return self._mcp_tool.is_concurrency_safe(input_data)
