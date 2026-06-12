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
"""MCP server provider catalog.

Each entry is a real service provider that offers an MCP server, with its
config in the standard ``.mcp.json`` format.  The frontend uses this catalog
to let users pick a provider and auto-fill the configuration form.

To extend: add to ``MCP_PROVIDERS`` below.
"""

from typing import Any


MCP_PROVIDERS: list[dict[str, Any]] = [
    # ── GitHub ──────────────────────────────────────────────────────────
    {
        "id": "github",
        "label": "GitHub",
        "description": "GitHub API — manage repositories, issues, pull requests, code search, and Actions",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
        },
        "env_fields": {"GITHUB_TOKEN": {"label": "GitHub Personal Access Token", "secret": True}},
        "docs": "https://github.com/modelcontextprotocol/servers/tree/main/src/github",
    },
    # ── Brave Search ────────────────────────────────────────────────────
    {
        "id": "brave-search",
        "label": "Brave Search",
        "description": "Web search via Brave Search API",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic-ai/mcp-brave-search"],
        },
        "env_fields": {"BRAVE_API_KEY": {"label": "Brave Search API Key", "secret": True}},
        "docs": "https://github.com/anthropics/anthropic-quickstarts/tree/main/mcp/brave-search",
    },
    # ── Stripe ──────────────────────────────────────────────────────────
    {
        "id": "stripe",
        "label": "Stripe",
        "description": "Stripe payment platform — customers, products, invoices, payment links, balance",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@stripe/mcp", "--tools=all"],
        },
        "env_fields": {"STRIPE_SECRET_KEY": {"label": "Stripe Secret Key (sk_...)", "secret": True}},
        "docs": "https://github.com/stripe/ai",
    },
    # ── Supabase ────────────────────────────────────────────────────────
    {
        "id": "supabase",
        "label": "Supabase",
        "description": "Supabase project management — database, storage, functions, auth",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@supabase/mcp-server-supabase@latest", "--read-only"],
        },
        "env_fields": {"SUPABASE_ACCESS_TOKEN": {"label": "Supabase Personal Access Token (sbp_...)", "secret": True}},
        "docs": "https://supabase.com/docs/guides/integration/mcp",
    },
    # ── Vercel ──────────────────────────────────────────────────────────
    {
        "id": "vercel",
        "label": "Vercel",
        "description": "Vercel platform — deployments, environment variables, project management",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "--package", "@vercel/sdk", "--", "mcp", "start"],
        },
        "env_fields": {"VERCEL_TOKEN": {"label": "Vercel API Token", "secret": True}},
        "docs": "https://www.npmjs.com/package/@vercel/sdk",
    },
    # ── Cloudflare ──────────────────────────────────────────────────────
    {
        "id": "cloudflare",
        "label": "Cloudflare",
        "description": "Cloudflare — Workers, DNS, KV, analytics, WAF configuration",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["mcp-remote", "https://remote-mcp-server.your-account.workers.dev/sse"],
        },
        "env_fields": {"CLOUDFLARE_API_TOKEN": {"label": "Cloudflare API Token", "secret": True}},
        "docs": "https://github.com/cloudflare/mcp-server-cloudflare",
    },
    # ── Sentry ──────────────────────────────────────────────────────────
    {
        "id": "sentry",
        "label": "Sentry",
        "description": "Sentry error tracking — issues, events, performance monitoring",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@getsentry/sentry-mcp-stdio"],
        },
        "env_fields": {"SENTRY_AUTH_TOKEN": {"label": "Sentry Auth Token", "secret": True}},
        "docs": "https://github.com/getsentry/sentry-mcp-stdio",
    },
    # ── Prisma ──────────────────────────────────────────────────────────
    {
        "id": "prisma",
        "label": "Prisma",
        "description": "Prisma ORM — schema management, migrations, database queries, Prisma Postgres",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "prisma", "mcp"],
        },
        "env_fields": {},
        "docs": "https://www.prismagraphql.com/blog/prisma-orm-6-6-0-esm-support-d1-migrations-and-prisma-mcp-server",
    },
    # ── Notion ──────────────────────────────────────────────────────────
    {
        "id": "notion",
        "label": "Notion",
        "description": "Notion workspace — semantic search, pages, databases",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@notionhq/notion-mcp-server"],
        },
        "env_fields": {"NOTION_TOKEN": {"label": "Notion Integration Token (ntn_...)", "secret": True}},
        "docs": "https://www.npmjs.com/package/@notionhq/notion-mcp-server",
    },
    # ── Figma ───────────────────────────────────────────────────────────
    {
        "id": "figma",
        "label": "Figma",
        "description": "Figma design — read file data, extract components, styles, and assets",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "figma-developer-mcp", "--stdio"],
        },
        "env_fields": {"FIGMA_API_KEY": {"label": "Figma Personal Access Token", "secret": True}},
        "docs": "https://github.com/GLips/Figma-Context-MCP",
    },
    # ── Slack ───────────────────────────────────────────────────────────
    {
        "id": "slack",
        "label": "Slack",
        "description": "Slack workspace — read messages, channels, users, and post messages",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@anthropic-ai/mcp-slack"],
        },
        "env_fields": {
            "SLACK_BOT_TOKEN": {"label": "Slack Bot Token (xoxb-...)", "secret": True},
            "SLACK_TEAM_ID": {"label": "Slack Team ID", "secret": False},
        },
        "docs": "https://github.com/anthropics/anthropic-quickstarts/tree/main/mcp/slack",
    },
    # ── Obsidian ────────────────────────────────────────────────────────
    {
        "id": "obsidian",
        "label": "Obsidian",
        "description": "Obsidian vault access — read, write, search notes, manage tags",
        "config": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "@bitbonsai/mcpvault@latest"],
        },
        "env_fields": {},
        "docs": "https://github.com/bitbonsai/mcpvault",
    },
]


def get_mcp_provider(provider_id: str) -> dict[str, Any] | None:
    """Return the provider entry for ``provider_id`` or None if unknown."""
    for p in MCP_PROVIDERS:
        if p["id"] == provider_id:
            return p
    return None


def mcp_catalog_payload() -> dict[str, Any]:
    """Serializable snapshot used by the frontend MCP form."""
    return {"providers": MCP_PROVIDERS}


__all__ = [
    "MCP_PROVIDERS",
    "get_mcp_provider",
    "mcp_catalog_payload",
]
