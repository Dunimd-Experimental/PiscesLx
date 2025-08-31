#!/usr/bin/env python3

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
import datetime
from typing import Dict, Any

def crypto_price(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve cryptocurrency price information.

    Args:
        arguments (Dict[str, Any]): A dictionary containing request parameters.
            - symbols (List[str], optional): List of cryptocurrency symbols. Defaults to ["BTC", "ETH"].
            - currency (str, optional): The currency in which prices are quoted. Defaults to "USD".

    Returns:
        Dict[str, Any]: A dictionary containing the request result.
            - success (bool): Indicates whether the request was successful.
            - data (Dict[str, Any]): Contains price data, currency, and timestamp.
    """
    symbols = arguments.get("symbols", ["BTC", "ETH"])
    currency = arguments.get("currency", "USD")
    
    # Mock cryptocurrency price data
    mock_prices = {
        "BTC": {
            "symbol": "BTC",
            "name": "Bitcoin",
            "price": 43567.89,
            "change_24h": 2.34,
            "change_percent_24h": 5.67,
            "market_cap": 854321000000,
            "volume_24h": 23456789000,
            "last_updated": "2024-12-19T10:30:00Z"
        },
        "ETH": {
            "symbol": "ETH",
            "name": "Ethereum",
            "price": 2234.56,
            "change_24h": -45.67,
            "change_percent_24h": -2.00,
            "market_cap": 268456000000,
            "volume_24h": 12345678000,
            "last_updated": "2024-12-19T10:30:00Z"
        },
        "SOL": {
            "symbol": "SOL",
            "name": "Solana",
            "price": 198.76,
            "change_24h": 12.34,
            "change_percent_24h": 6.58,
            "market_cap": 87654000000,
            "volume_24h": 2345678000,
            "last_updated": "2024-12-19T10:30:00Z"
        },
        "USDT": {
            "symbol": "USDT",
            "name": "Tether",
            "price": 1.00,
            "change_24h": 0.00,
            "change_percent_24h": 0.00,
            "market_cap": 91000000000,
            "volume_24h": 45678900000,
            "last_updated": "2024-12-19T10:30:00Z"
        }
    }
    
    result = {}
    for symbol in symbols:
        symbol = symbol.upper()
        if symbol in mock_prices:
            result[symbol] = mock_prices[symbol]
    
    return {
        "success": True,
        "data": {
            "prices": result,
            "currency": currency,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
    }

def crypto_trending(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve trending cryptocurrencies.

    Args:
        arguments (Dict[str, Any]): A dictionary containing request parameters.
            - limit (int, optional): The number of trending cryptocurrencies to return. Defaults to 5, maximum 10.

    Returns:
        Dict[str, Any]: A dictionary containing the request result.
            - success (bool): Indicates whether the request was successful.
            - data (Dict[str, Any]): Contains trending cryptocurrency data and timestamp.
    """
    limit = min(arguments.get("limit", 5), 10)
    
    # Mock trending data
    trending = [
        {
            "symbol": "BTC",
            "name": "Bitcoin",
            "price": 43567.89,
            "change_percent_24h": 5.67,
            "market_cap_rank": 1
        },
        {
            "symbol": "SOL",
            "name": "Solana",
            "price": 198.76,
            "change_percent_24h": 6.58,
            "market_cap_rank": 5
        },
        {
            "symbol": "DOGE",
            "name": "Dogecoin",
            "price": 0.321,
            "change_percent_24h": 12.34,
            "market_cap_rank": 8
        }
    ]
    
    return {
        "success": True,
        "data": {
            "trending": trending[:limit],
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
    }

# Integrate with Pisces L1 MCP Plaza
from . import register_custom_tool

# Register tools to MCP Plaza
register_custom_tool(
    name="crypto_price",
    description="Retrieve real-time cryptocurrency prices and market data.",
    parameters={
        "type": "object",
        "properties": {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "default": ["BTC", "ETH"],
                "description": "List of cryptocurrency symbols (BTC, ETH, SOL, USDT, etc.)"
            },
            "currency": {
                "type": "string",
                "default": "USD",
                "description": "Pricing currency (USD, CNY, EUR, etc.)"
            }
        }
    },
    function=crypto_price,
    category="Finance"
)

register_custom_tool(
    name="crypto_trending",
    description="Retrieve current trending cryptocurrency data.",
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "number",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
                "description": "Number of trending cryptocurrencies to return."
            }
        }
    },
    function=crypto_trending,
    category="Finance"
)