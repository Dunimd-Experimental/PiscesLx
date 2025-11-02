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

import sys
import json
import requests
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.mcp import PiscesLxCoreMCPPlaza

mcp = PiscesLxCoreMCPPlaza()

@mcp.tool()
def crypto_price(symbol: str = "BTC", currency: str = "USD") -> Dict[str, Any]:
    """
    Fetch the current price and related data for a given cryptocurrency.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., BTC, ETH). Defaults to "BTC".
        currency (str): Target currency for price conversion (e.g., USD, EUR). Defaults to "USD".
        
    Returns:
        Dict[str, Any]: A dictionary containing price data or error information.
                        On success, includes symbol, currency, price, 24h change, market cap, and volume.
                        On failure, includes an error message.
    """
    try:
        # Prepare API request parameters
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": symbol.lower(),
            "vs_currencies": currency.lower(),
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_24hr_vol": "true"
        }
        
        # Execute HTTP GET request with timeout
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        
        # Check if requested symbol exists in response
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
        # Handle network-related errors
        return {
            "success": False,
            "error": f"Network error: {str(e)}"
        }
    except Exception as e:
        # Handle unexpected errors
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def crypto_trending(limit: int = 10) -> Dict[str, Any]:
    """
    Retrieve trending cryptocurrencies from CoinGecko API.
    
    Args:
        limit (int): Maximum number of trending coins to retrieve. Defaults to 10.
        
    Returns:
        Dict[str, Any]: A dictionary containing trending coin data or error information.
                        On success, includes a list of trending coins with their details.
                        On failure, includes an error message.
    """
    try:
        # Execute HTTP GET request to fetch trending coins
        url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        trending = []
        
        # Process each trending coin up to the specified limit
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
        # Handle network-related errors
        return {
            "success": False,
            "error": f"Network error: {str(e)}"
        }
    except Exception as e:
        # Handle unexpected errors
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
