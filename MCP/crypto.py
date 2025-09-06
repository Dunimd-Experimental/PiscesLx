#!/usr/bin/env/python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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

import json
from MCP import mcp

@mcp.tool()
def crypto_price(symbol: str = "BTC", currency: str = "USD") -> Dict[str, Any]:
    """Get current cryptocurrency price for a given symbol."""
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": symbol.lower(),
            "vs_currencies": currency.lower(),
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_24hr_vol": "true"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if symbol.lower() in data:
            price_data = data[symbol.lower()]
            return {
                "success": True,
                "symbol": symbol.upper(),
                "currency": currency.upper(),
                "price": price_data.get(currency.lower()),
                "change_24h": price_data.get(f"{currency.lower()}_24h_change"),
                "market_cap": price_data.get(f"{currency.lower()}_market_cap"),
                "volume_24h": price_data.get(f"{currency.lower()}_24h_vol")
            }
        else:
            return {
                "success": False,
                "error": f"Symbol '{symbol}' not found"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Network error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def crypto_trending(limit: int = 10) -> Dict[str, Any]:
    """Get trending cryptocurrencies from CoinGecko."""
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        trending = []
        
        for item in data.get("coins", [])[:limit]:
            coin = item.get("item", {})
            trending.append({
                "id": coin.get("id"),
                "symbol": coin.get("symbol"),
                "name": coin.get("name"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "price_btc": coin.get("price_btc")
            })
        
        return {
            "success": True,
            "trending": trending,
            "count": len(trending)
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Network error: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }