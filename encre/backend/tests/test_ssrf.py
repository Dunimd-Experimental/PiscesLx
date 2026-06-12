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

"""Tests for the SSRF guard URL validation and hostname blocking."""

import pytest


class TestEncreSSRFGuardCreation:
    def test_create_guard(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard is not None
        assert guard._dns_cache == {}
        assert guard._dns_cache_ttl == 300.0

    def test_default_blocked_v4(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        # There should be blocked v4 subnets
        assert len(guard._blocked_v4) > 0
        assert len(guard._blocked_v6) > 0

    def test_default_whitelist_empty(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert len(guard._whitelist_v4) == 0
        assert len(guard._whitelist_v6) == 0


class TestValidateUrl:
    def test_public_url_is_safe(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        # Use public IPs that are not in any blocked private/reserved range
        assert guard.validate_url("https://8.8.8.8/path?q=1") is True
        assert guard.validate_url("https://1.1.1.1") is True

    def test_http_url_is_safe(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.validate_url("http://93.184.216.34") is True  # example.com IP

    def test_localhost_is_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.validate_url("http://127.0.0.1:8080/api") is False
        assert guard.validate_url("https://127.0.0.1") is False

    def test_private_ipv4_is_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        # 192.168.x.x
        assert guard.validate_url("http://192.168.1.1") is False
        # 10.x.x.x
        assert guard.validate_url("https://10.0.0.1/admin") is False
        # 172.16.x.x
        assert guard.validate_url("http://172.16.0.1") is False

    def test_ipv6_localhost_is_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.validate_url("http://[::1]:8080") is False

    def test_zero_ip_is_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.validate_url("http://0.0.0.0") is False

    def test_link_local_is_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.validate_url("http://169.254.1.1") is False

    def test_non_http_scheme_is_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.validate_url("ftp://8.8.8.8") is False
        assert guard.validate_url("file:///etc/passwd") is False
        assert guard.validate_url("ssh://8.8.8.8") is False

    def test_invalid_url_returns_false(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.validate_url("not-a-url") is False
        assert guard.validate_url("") is False

    def test_no_hostname_returns_false(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        # URL without a hostname
        assert guard.validate_url("http://") is False


class TestIsBlockedHostname:
    def test_empty_hostname_not_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.is_blocked_hostname("") is False

    def test_localhost_is_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.is_blocked_hostname("127.0.0.1") is True
        assert guard.is_blocked_hostname("localhost") is True
        assert guard.is_blocked_hostname("::1") is True

    def test_private_ips_are_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.is_blocked_hostname("192.168.1.100") is True
        assert guard.is_blocked_hostname("10.0.0.5") is True
        assert guard.is_blocked_hostname("172.16.5.5") is True

    def test_public_ip_is_not_blocked(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.is_blocked_hostname("8.8.8.8") is False
        assert guard.is_blocked_hostname("1.1.1.1") is False


class TestExtractSafeHostname:
    def test_safe_url_extracts_hostname(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        hostname = guard.extract_safe_hostname("https://8.8.8.8/path")
        assert hostname == "8.8.8.8"

    def test_blocked_url_returns_none(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        hostname = guard.extract_safe_hostname("http://127.0.0.1/secret")
        assert hostname is None

    def test_invalid_url_returns_none(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert guard.extract_safe_hostname("not_a_url") is None
        assert guard.extract_safe_hostname("") is None

    def test_https_with_port(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        hostname = guard.extract_safe_hostname("https://1.1.1.1:443/v1")
        assert hostname == "1.1.1.1"


class TestDNSResolution:
    def test_hostname_with_dns(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        # 8.8.8.8 is a public IP address, should not be blocked
        assert guard.is_blocked_hostname("8.8.8.8") is False

    def test_dns_cache_for_ip_address(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        assert len(guard._dns_cache) == 0
        guard.is_blocked_hostname("8.8.8.8")
        # IP addresses also get cached
        assert "8.8.8.8" in guard._dns_cache


class TestClearDNSCache:
    def test_clear_dns_cache_empties_cache(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        # Resolve something to populate cache
        guard.is_blocked_hostname("8.8.8.8")
        assert len(guard._dns_cache) >= 1
        guard.clear_dns_cache()
        assert len(guard._dns_cache) == 0

    def test_clear_dns_cache_is_idempotent(self):
        from encre.ssrf import EncreSSRFGuard
        guard = EncreSSRFGuard()
        guard.clear_dns_cache()
        guard.clear_dns_cache()
        assert len(guard._dns_cache) == 0
