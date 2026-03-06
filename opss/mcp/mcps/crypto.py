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
#
# DISCLAIMER: Users must comply with applicable AI regulations.
# Non-compliance may result in service termination or legal liability.

"""
Crypto Tool - Cryptocurrency price and market data
"""

from typing import Any, Dict, List, Optional

from .base import POPSSMCPToolBase, POPSSMCPToolResult


class CryptoPriceTool(POPSSMCPToolBase):
    name = "crypto_price"
    description = "Get cryptocurrency prices and market data from CoinGecko"
    category = "finance"
    tags = ["crypto", "price", "market", "bitcoin", "ethereum"]
    
    parameters = {
        "type": "object",
        "properties": {
            "coin_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of coin IDs (e.g., ['bitcoin', 'ethereum'])"
            },
            "vs_currency": {
                "type": "string",
                "description": "Target currency (e.g., 'usd', 'eur', 'cny')",
                "default": "usd"
            },
            "include_market_cap": {
                "type": "boolean",
                "description": "Include market cap data",
                "default": True
            },
            "include_24h_vol": {
                "type": "boolean",
                "description": "Include 24h volume data",
                "default": True
            },
            "include_24h_change": {
                "type": "boolean",
                "description": "Include 24h price change",
                "default": True
            }
        },
        "required": ["coin_ids"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        coin_ids = arguments.get("coin_ids", [])
        vs_currency = arguments.get("vs_currency", "usd")
        include_market_cap = arguments.get("include_market_cap", True)
        include_24h_vol = arguments.get("include_24h_vol", True)
        include_24h_change = arguments.get("include_24h_change", True)
        
        if not coin_ids:
            return self._create_error_result("coin_ids is required", "ValidationError")
        
        try:
            import requests
            
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": vs_currency,
                "include_market_cap": str(include_market_cap).lower(),
                "include_24hr_vol": str(include_24h_vol).lower(),
                "include_24hr_change": str(include_24h_change).lower(),
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for coin_id in coin_ids:
                if coin_id in data:
                    coin_data = data[coin_id]
                    result = {
                        "coin_id": coin_id,
                        "price": coin_data.get(vs_currency),
                    }
                    if include_market_cap:
                        result["market_cap"] = coin_data.get(f"{vs_currency}_market_cap")
                    if include_24h_vol:
                        result["volume_24h"] = coin_data.get(f"{vs_currency}_24h_vol")
                    if include_24h_change:
                        result["change_24h"] = coin_data.get(f"{vs_currency}_24h_change")
                    results.append(result)
            
            return self._create_success_result({
                "currency": vs_currency,
                "coins": results,
            })
            
        except ImportError:
            return self._create_error_result(
                "requests required. Install with: pip install requests",
                "DependencyError"
            )
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)


class CryptoTrendingTool(POPSSMCPToolBase):
    name = "crypto_trending"
    description = "Get trending cryptocurrencies from CoinGecko"
    category = "finance"
    tags = ["crypto", "trending", "market"]
    
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        try:
            import requests
            
            url = "https://api.coingecko.com/api/v3/search/trending"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            trending = []
            for item in data.get("coins", [])[:10]:
                coin = item.get("item", {})
                trending.append({
                    "id": coin.get("id"),
                    "name": coin.get("name"),
                    "symbol": coin.get("symbol"),
                    "market_cap_rank": coin.get("market_cap_rank"),
                    "price_btc": coin.get("price_btc"),
                })
            
            return self._create_success_result({
                "trending": trending,
            })
            
        except ImportError:
            return self._create_error_result(
                "requests required. Install with: pip install requests",
                "DependencyError"
            )
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)


class CryptoSearchTool(POPSSMCPToolBase):
    name = "crypto_search"
    description = "Search for cryptocurrencies by name or symbol"
    category = "finance"
    tags = ["crypto", "search"]
    
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (name or symbol)"
            }
        },
        "required": ["query"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        query = arguments.get("query", "")
        
        if not query:
            return self._create_error_result("Query is required", "ValidationError")
        
        try:
            import requests
            
            url = "https://api.coingecko.com/api/v3/search"
            params = {"query": query}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            coins = []
            for coin in data.get("coins", [])[:10]:
                coins.append({
                    "id": coin.get("id"),
                    "name": coin.get("name"),
                    "symbol": coin.get("symbol"),
                    "market_cap_rank": coin.get("market_cap_rank"),
                    "thumb": coin.get("thumb"),
                })
            
            return self._create_success_result({
                "query": query,
                "coins": coins,
            })
            
        except ImportError:
            return self._create_error_result(
                "requests required. Install with: pip install requests",
                "DependencyError"
            )
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)
