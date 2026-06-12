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

import ipaddress
import socket
from urllib.parse import urlparse


class EncreSSRFGuard:
    BLOCKED_V4: list[str] = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
        "100.64.0.0/10",
        "0.0.0.0/8",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "198.18.0.0/15",
        "127.0.0.0/8",
    ]
    BLOCKED_V6: list[str] = [
        "fc00::/7",
        "fe80::/10",
        "::1",
        "::ffff:0:0/96",
        "2001:db8::/32",
        "ff00::/8",
    ]
    WHITELIST_V4: list[str] = []
    WHITELIST_V6: list[str] = []

    def __init__(self) -> None:
        self._blocked_v4: list[ipaddress.IPv4Network] = [ipaddress.IPv4Network(n) for n in self.BLOCKED_V4]
        self._blocked_v6: list[ipaddress.IPv6Network] = [ipaddress.IPv6Network(n) for n in self.BLOCKED_V6]
        self._whitelist_v4: list[ipaddress.IPv4Network] = [ipaddress.IPv4Network(n) for n in self.WHITELIST_V4]
        self._whitelist_v6: list[ipaddress.IPv6Network] = [ipaddress.IPv6Network(n) for n in self.WHITELIST_V6]
        self._dns_cache: dict[str, list[ipaddress.IPv4Address | ipaddress.IPv6Address]] = {}
        self._dns_cache_ttl: float = 300.0

    def _resolve_hostname(self, hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        import time
        now = time.time()
        if hostname in self._dns_cache:
            cached_at, addrs = self._dns_cache[hostname]
            if now - cached_at < self._dns_cache_ttl:
                return addrs
        addrs: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        try:
            addr = ipaddress.ip_address(hostname)
            addrs = [addr]
        except ValueError:
            try:
                infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
                seen: set[str] = set()
                for _, _, _, _, sockaddr in infos:
                    ip_str = sockaddr[0]
                    if ip_str not in seen:
                        seen.add(ip_str)
                        addrs.append(ipaddress.ip_address(ip_str))
            except socket.gaierror:
                pass
        self._dns_cache[hostname] = (now, addrs)
        return addrs

    def is_blocked_hostname(self, hostname: str) -> bool:
        if not hostname:
            return False
        addrs = self._resolve_hostname(hostname)
        return any(self.is_blocked_address(addr) for addr in addrs)

    def is_blocked_address(self, addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        if isinstance(addr, ipaddress.IPv6Address):
            if addr.ipv4_mapped:
                mapped = ipaddress.IPv4Address(addr.ipv4_mapped)
                return self._is_blocked_v4(mapped)
            return self._is_blocked_v6(addr)
        return self._is_blocked_v4(addr)

    def _is_blocked_v4(self, addr: ipaddress.IPv4Address) -> bool:
        for net in self._whitelist_v4:
            if addr in net:
                return False
        return any(addr in net for net in self._blocked_v4)

    def _is_blocked_v6(self, addr: ipaddress.IPv6Address) -> bool:
        for net in self._whitelist_v6:
            if addr in net:
                return False
        return any(addr in net for net in self._blocked_v6)

    def validate_url(self, url_str: str) -> bool:
        try:
            parsed = urlparse(url_str)
        except Exception:
            return False
        if not parsed.hostname:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        return not self.is_blocked_hostname(parsed.hostname)

    def extract_safe_hostname(self, url_str: str) -> str | None:
        try:
            parsed = urlparse(url_str)
        except Exception:
            return None
        if not parsed.hostname:
            return None
        if self.is_blocked_hostname(parsed.hostname):
            return None
        return parsed.hostname

    def clear_dns_cache(self) -> None:
        self._dns_cache.clear()
