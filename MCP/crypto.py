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

"""
Cryptocurrency data retrieval module for accessing price and trending information
from the CoinGecko API.
"""

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
    Fetch current market data for a specified cryptocurrency from CoinGecko API.
    
    Args:
        symbol (str): Cryptocurrency identifier (e.g., btc, eth).
                      Case-insensitive.
        currency (str): Target currency for price quoting (e.g., usd, eur).
                        Defaults to 'USD'.
    
    Returns:
        Dict[str, Any]: A dictionary containing cryptocurrency market data:
            - success (bool): Indicates if the request was successful.
            - symbol (str): Uppercase version of the input symbol.
            - currency (str): Uppercase version of the input currency.
            - price (float): Current price in the target currency.
            - change_24h (float): Percentage change over the last 24 hours.
            - market_cap (float): Market capitalization in the target currency.
            - volume_24h (float): Trading volume in the last 24 hours.
            - error (str): Error message if the request failed.
            - error_type (str): Type of exception encountered, if any.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": symbol.lower(),
            "vs_currencies": currency.lower(),
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        payload = response.json()
        if symbol.lower() not in payload:
            return {
                "success": False,
                "error": f"Symbol '{symbol}' not found",
                "error_type": "ValueError"
            }

        price_info = payload[symbol.lower()]
        return {
            "success": True,
            "symbol": symbol.upper(),
            "currency": currency.upper(),
            "price": price_info.get(currency.lower()),
            "change_24h": price_info.get(f"{currency.lower()}_24h_change"),
            "market_cap": price_info.get(f"{currency.lower()}_market_cap"),
            "volume_24h": price_info.get(f"{currency.lower()}_24h_vol"),
        }

    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "error": f"Network error: {exc}",
            "error_type": type(exc).__name__
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__
        }

@mcp.tool()
def crypto_trending(limit: int = 10) -> Dict[str, Any]:
    """
    Retrieve trending cryptocurrencies from CoinGecko API.
    
    Args:
        limit (int): Maximum number of trending coins to retrieve.
                    Defaults to 10.
    
    Returns:
        Dict[str, Any]: A dictionary containing:
            - success (bool): Indicates if the request was successful.
            - trending (list): List of dictionaries with coin details.
              Each dictionary includes:
                - id (str): Coin ID.
                - symbol (str): Coin symbol.
                - name (str): Full name of the coin.
                - market_cap_rank (int): Market cap rank.
                - price_btc (float): Price in Bitcoin.
            - count (int): Number of coins retrieved.
            - error (str): Error message if the request failed.
            - error_type (str): Type of exception encountered, if any.
    """
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        payload = response.json()
        coins = [
            {
                "id": item["item"].get("id"),
                "symbol": item["item"].get("symbol"),
                "name": item["item"].get("name"),
                "market_cap_rank": item["item"].get("market_cap_rank"),
                "price_btc": item["item"].get("price_btc"),
            }
            for item in payload.get("coins", [])[:limit]
        ]

        return {
            "success": True,
            "trending": coins,
            "count": len(coins)
        }

    except requests.exceptions.RequestException as exc:
        return {
            "success": False,
            "error": f"Network error: {exc}",
            "error_type": type(exc).__name__
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__
        }
