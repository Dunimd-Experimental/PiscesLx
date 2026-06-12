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

"""MCP server lifecycle manager.

Loads server specs from a config file (or in-memory dict), spawns each
MCP server, registers their tools into a ``ToolRegistry`` with a per-server
prefix, and exposes start / stop / restart / reload primitives. Supports
the same JSON shape Claude Desktop / Claude Code uses for ``mcpServers``::

    {
        "mcpServers": {
            "fs": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                "env": {"FOO": "bar"}
            },
            "remote": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer xxx"}
            }
        }
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

from encre.tools.mcp import (
    HttpTransport,
    MCPClient,
    MCPError,
    MCPProtocolError,
    MCPTransport,
    MCPTransportError,
    StdioTransport,
    _MCPDiscoveredTool,
)
from encre.tools.registry import ToolRegistry

logger = logging.getLogger("encre.tools.mcp_manager")


# ──────────────────────────────────────────────────────────────────────
# Server spec
# ──────────────────────────────────────────────────────────────────────


@dataclass
class MCPServerSpec:
    """Declarative description of one MCP server.

    Exactly one of (command/args) or url must be set. ``enabled=False``
    skips the server during reconcile.
    """

    name: str
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 60.0
    enabled: bool = True
    prefix: str | None = None  # default: f"mcp__{name}__"

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> "MCPServerSpec":
        if not isinstance(raw, dict):
            raise ValueError(f"Server {name!r} entry must be an object")
        command = str(raw.get("command", "")).strip()
        args = list(raw.get("args", []) or [])
        url = str(raw.get("url", "")).strip()
        if not command and not url:
            raise ValueError(
                f"Server {name!r}: must specify either 'command' or 'url'"
            )
        if command and url:
            raise ValueError(
                f"Server {name!r}: cannot specify both 'command' and 'url'"
            )
        env = {str(k): str(v) for k, v in (raw.get("env") or {}).items()}
        headers = {str(k): str(v) for k, v in (raw.get("headers") or {}).items()}
        return cls(
            name=name,
            command=command,
            args=[str(a) for a in args],
            url=url,
            env=env,
            cwd=raw.get("cwd"),
            headers=headers,
            timeout=float(raw.get("timeout", 60.0)),
            enabled=bool(raw.get("enabled", True)),
            prefix=raw.get("prefix"),
        )

    def fingerprint(self) -> tuple[Any, ...]:
        """Identity tuple used to decide whether a running server's config has
        changed and it needs restarting."""
        return (
            self.command,
            tuple(self.args),
            self.url,
            tuple(sorted(self.env.items())),
            self.cwd,
            tuple(sorted(self.headers.items())),
            self.timeout,
        )

    def build_transport(self) -> MCPTransport:
        if self.command:
            cmd_line = [self.command] + list(self.args) if self.args else self.command
            return StdioTransport(cmd_line, env=self.env, cwd=self.cwd)
        return HttpTransport(self.url, timeout=self.timeout, headers=self.headers)

    @property
    def resolved_prefix(self) -> str:
        return self.prefix if self.prefix is not None else f"mcp__{self.name}__"


# ──────────────────────────────────────────────────────────────────────
# Manager
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _ServerRecord:
    spec: MCPServerSpec
    client: MCPClient
    transport: MCPTransport
    registered_names: list[str] = field(default_factory=list)
    last_error: str | None = None


class MCPManager:
    """Manages multiple MCP servers and bridges their tools into a registry.

    Thread-safety: relies on asyncio. Methods are not safe to call from
    multiple loops; one event loop per manager.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._servers: dict[str, _ServerRecord] = {}
        self._lock = asyncio.Lock()
        self._config_path: str | None = None

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def parse_config(raw: dict[str, Any]) -> list[MCPServerSpec]:
        """Parse the standard ``{"mcpServers": {...}}`` config shape."""
        servers_node = raw.get("mcpServers")
        if servers_node is None:
            # Tolerate the flat shape too
            servers_node = raw if all(isinstance(v, dict) for v in raw.values()) else {}
        out: list[MCPServerSpec] = []
        for name, entry in (servers_node or {}).items():
            out.append(MCPServerSpec.from_dict(str(name), entry))
        return out

    @staticmethod
    def load_config_file(path: str) -> list[MCPServerSpec]:
        p = pathlib.Path(path)
        if not p.is_file():
            return []
        with p.open("r", encoding="utf-8") as fh:
            try:
                raw = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"MCP config {path} is not valid JSON: {exc}") from exc
        return MCPManager.parse_config(raw)

    def bind_config_file(self, path: str) -> None:
        """Remember a config file path so ``reload()`` can re-read it."""
        self._config_path = os.fspath(path)

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start_server(self, spec: MCPServerSpec) -> _ServerRecord:
        """Start a server from a spec and register its tools."""
        if not spec.enabled:
            raise ValueError(f"Server {spec.name!r} is disabled")
        if spec.name in self._servers:
            raise ValueError(f"Server {spec.name!r} is already running")

        transport = spec.build_transport()
        client = MCPClient(transport)
        try:
            await client.initialize()
        except (MCPProtocolError, MCPTransportError, MCPError) as exc:
            try:
                await client.close()
            except Exception:
                pass
            raise RuntimeError(
                f"MCP server {spec.name!r} failed to initialize: {exc}"
            ) from exc

        rec = _ServerRecord(spec=spec, client=client, transport=transport)
        # Register discovered tools into the registry
        try:
            tools = await client.list_tools()
        except Exception as exc:
            try:
                await client.close()
            except Exception:
                pass
            raise RuntimeError(
                f"MCP server {spec.name!r} initialized but tools/list failed: {exc}"
            ) from exc

        proxy = _ManagedClientProxy(client)
        for tool_schema in tools:
            tool_name = tool_schema["name"]
            registered_name = f"{spec.resolved_prefix}{tool_name}"
            wrapper = _MCPDiscoveredTool(
                mcp_tool=proxy,  # type: ignore[arg-type]
                tool_name=tool_name,
                schema=tool_schema,
                registered_name=registered_name,
            )
            self._registry.register(wrapper)
            rec.registered_names.append(registered_name)

        self._servers[spec.name] = rec
        logger.info(
            "MCP server %r started; registered %d tools (prefix=%s)",
            spec.name, len(rec.registered_names), spec.resolved_prefix,
        )
        return rec

    async def stop_server(self, name: str) -> bool:
        rec = self._servers.pop(name, None)
        if rec is None:
            return False
        # Remove tools from the registry
        for n in rec.registered_names:
            self._registry._tools.pop(n, None)  # type: ignore[attr-defined]
        try:
            await rec.client.close()
        except Exception:
            pass
        logger.info("MCP server %r stopped; unregistered %d tools",
                    name, len(rec.registered_names))
        return True

    async def restart_server(self, name: str) -> _ServerRecord:
        rec = self._servers.get(name)
        if rec is None:
            raise ValueError(f"No such MCP server: {name}")
        spec = rec.spec
        await self.stop_server(name)
        return await self.start_server(spec)

    async def start_all(self, specs: list[MCPServerSpec]) -> dict[str, str]:
        """Start every enabled spec. Returns ``{name: 'ok' | 'error: ...'}``."""
        results: dict[str, str] = {}
        for spec in specs:
            if not spec.enabled:
                results[spec.name] = "disabled"
                continue
            try:
                await self.start_server(spec)
                results[spec.name] = "ok"
            except Exception as exc:
                results[spec.name] = f"error: {exc}"
                logger.warning("Failed to start MCP server %r: %s", spec.name, exc)
        return results

    async def stop_all(self) -> None:
        for name in list(self._servers.keys()):
            await self.stop_server(name)

    # ------------------------------------------------------------------
    # Reconcile (hot reload)
    # ------------------------------------------------------------------

    async def reconcile(self, specs: list[MCPServerSpec]) -> dict[str, str]:
        """Bring the running set in line with ``specs``.

        - New specs are started.
        - Removed specs are stopped.
        - Changed specs (different fingerprint) are restarted.
        - Unchanged specs are left alone.
        """
        async with self._lock:
            desired = {s.name: s for s in specs if s.enabled}
            current = dict(self._servers)
            actions: dict[str, str] = {}

            # Stop removed/disabled
            for name in list(current.keys()):
                if name not in desired:
                    await self.stop_server(name)
                    actions[name] = "stopped"

            # Start new or restart changed
            for name, spec in desired.items():
                if name not in self._servers:
                    try:
                        await self.start_server(spec)
                        actions[name] = "started"
                    except Exception as exc:
                        actions[name] = f"start_failed: {exc}"
                else:
                    if self._servers[name].spec.fingerprint() != spec.fingerprint():
                        try:
                            await self.stop_server(name)
                            await self.start_server(spec)
                            actions[name] = "restarted"
                        except Exception as exc:
                            actions[name] = f"restart_failed: {exc}"
                    else:
                        actions[name] = "unchanged"

            return actions

    async def reload(self) -> dict[str, str]:
        """Re-read the bound config file and reconcile."""
        if self._config_path is None:
            raise RuntimeError("No config file bound. Call bind_config_file() first.")
        specs = self.load_config_file(self._config_path)
        return await self.reconcile(specs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def status(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, rec in self._servers.items():
            out.append({
                "name": name,
                "transport": "stdio" if rec.spec.command else "http",
                "command": rec.spec.command,
                "args": rec.spec.args,
                "url": rec.spec.url,
                "prefix": rec.spec.resolved_prefix,
                "tools": list(rec.registered_names),
                "connected": rec.client.is_initialized,
                "last_error": rec.last_error,
            })
        return out

    def list_tool_names(self, server: str | None = None) -> list[str]:
        if server is not None:
            rec = self._servers.get(server)
            return list(rec.registered_names) if rec else []
        names: list[str] = []
        for rec in self._servers.values():
            names.extend(rec.registered_names)
        return names


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


class _ManagedClientProxy:
    """Looks enough like a ``EncreMCPTool`` for ``_MCPDiscoveredTool`` to call.

    The base ``_MCPDiscoveredTool`` calls ``mcp_tool.call_tool`` and asks for
    concurrency-safety. We expose only what's needed and route through a
    shared ``MCPClient`` instance.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client
        # _MCPDiscoveredTool's is_concurrency_safe() asks the proxy; SSE/HTTP
        # transports are safe for parallel calls under the client lock.
        self._transport = client._transport  # type: ignore[attr-defined]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        content = await self._client.call_tool(name, arguments)
        parts: list[str] = []
        for item in content:
            t = item.get("type", "text")
            if t == "text":
                parts.append(item.get("text", ""))
            elif t == "image":
                parts.append(
                    json.dumps({
                        "type": "image",
                        "mime": item.get("mimeType", "image/png"),
                        "base64": item.get("data", ""),
                    }, ensure_ascii=False)
                )
            elif t == "resource":
                parts.append(json.dumps(item.get("resource", item), ensure_ascii=False))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)

    def is_concurrency_safe(self, input_data: dict[str, Any]) -> bool:
        return isinstance(self._transport, HttpTransport)


# ──────────────────────────────────────────────────────────────────────
# Convenience bootstrap
# ──────────────────────────────────────────────────────────────────────


def default_mcp_config_path() -> str:
    """Resolve the default mcp.json location.

    Order:
    1. ``$ENCRE_MCP_CONFIG`` env var if set.
    2. ``<data_dir>/mcp.json`` where data_dir = ``encre.config.get_data_dir()``.
    """
    override = os.environ.get("ENCRE_MCP_CONFIG")
    if override:
        return override
    try:
        from encre.config import get_data_dir
        return str(pathlib.Path(get_data_dir()) / "mcp.json")
    except Exception:
        return str(pathlib.Path.home() / ".encre" / "mcp.json")


async def bootstrap_mcp_servers(
    registry: ToolRegistry,
    config_path: str | None = None,
    raise_on_error: bool = False,
) -> tuple[MCPManager, dict[str, str]]:
    """Start every MCP server declared in the config file and bridge their
    tools into ``registry``. Safe to call when no config file exists
    (returns an empty MCPManager).

    Returns the manager (so callers can later ``stop_all()`` or ``reload()``)
    along with a per-server status dict.
    """
    path = config_path or default_mcp_config_path()
    mgr = MCPManager(registry)
    mgr.bind_config_file(path)

    if not pathlib.Path(path).is_file():
        return mgr, {}

    try:
        specs = MCPManager.load_config_file(path)
    except Exception as exc:
        logger.warning("Skipping MCP bootstrap; config %s is invalid: %s", path, exc)
        if raise_on_error:
            raise
        return mgr, {"_config_error": str(exc)}

    results = await mgr.start_all(specs)
    return mgr, results


__all__ = [
    "MCPServerSpec",
    "MCPManager",
    "bootstrap_mcp_servers",
    "default_mcp_config_path",
]
