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

import pytz
import datetime
from MCP import mcp
from typing import Dict, Any, Optional

@mcp.tool()
def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """Get the current time in a specific timezone."""
    try:
        if timezone not in pytz.all_timezones:
            return {
                "success": False,
                "error": f"Invalid timezone: {timezone}",
                "available_timezones": pytz.all_timezones[:10]  # Return first 10 as example
            }
        
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)
        
        return {
            "success": True,
            "datetime": now.isoformat(),
            "timezone": timezone,
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "microsecond": now.microsecond,
            "weekday": now.strftime("%A"),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S %Z")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def parse_time(time_string: str, input_format: str = "%Y-%m-%d %H:%M:%S", timezone: str = "UTC") -> Dict[str, Any]:
    """Parse a time string into a structured format."""
    try:
        if timezone not in pytz.all_timezones:
            return {
                "success": False,
                "error": f"Invalid timezone: {timezone}"
            }
        
        tz = pytz.timezone(timezone)
        dt = datetime.datetime.strptime(time_string, input_format)
        dt = tz.localize(dt)
        
        return {
            "success": True,
            "datetime": dt.isoformat(),
            "timezone": timezone,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "second": dt.second,
            "weekday": dt.strftime("%A"),
            "formatted": dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid time format: {str(e)}",
            "example_format": "2024-01-15 14:30:00"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def time_difference(start_time: str, end_time: str, timezone: str = "UTC") -> Dict[str, Any]:
    """Calculate the difference between two times."""
    try:
        if timezone not in pytz.all_timezones:
            return {
                "success": False,
                "error": f"Invalid timezone: {timezone}"
            }
        
        tz = pytz.timezone(timezone)
        
        # Parse ISO format times
        start_dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = datetime.datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        # Convert to specified timezone
        if start_dt.tzinfo is None:
            start_dt = tz.localize(start_dt)
        else:
            start_dt = start_dt.astimezone(tz)
            
        if end_dt.tzinfo is None:
            end_dt = tz.localize(end_dt)
        else:
            end_dt = end_dt.astimezone(tz)
        
        delta = end_dt - start_dt
        
        return {
            "success": True,
            "difference_seconds": delta.total_seconds(),
            "difference_days": delta.days,
            "difference_hours": delta.total_seconds() / 3600,
            "difference_minutes": delta.total_seconds() / 60,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "human_readable": str(delta)
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid time format: {str(e)}",
            "expected_format": "ISO format (YYYY-MM-DDTHH:MM:SS)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def format_time(iso_time: str, output_format: str = "%Y-%m-%d %H:%M:%S", timezone: str = "UTC") -> Dict[str, Any]:
    """Format an ISO time string to a custom format."""
    try:
        if timezone not in pytz.all_timezones:
            return {
                "success": False,
                "error": f"Invalid timezone: {timezone}"
            }
        
        tz = pytz.timezone(timezone)
        dt = datetime.datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
        
        if dt.tzinfo is None:
            dt = tz.localize(dt)
        else:
            dt = dt.astimezone(tz)
        
        formatted = dt.strftime(output_format)
        
        return {
            "success": True,
            "formatted_time": formatted,
            "timezone": timezone,
            "iso_time": dt.isoformat(),
            "format_used": output_format
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid time format: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def add_duration(base_time: str, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0, timezone: str = "UTC") -> Dict[str, Any]:
    """Add a duration to a base time."""
    try:
        if timezone not in pytz.all_timezones:
            return {
                "success": False,
                "error": f"Invalid timezone: {timezone}"
            }
        
        tz = pytz.timezone(timezone)
        base_dt = datetime.datetime.fromisoformat(base_time.replace('Z', '+00:00'))
        
        if base_dt.tzinfo is None:
            base_dt = tz.localize(base_dt)
        else:
            base_dt = base_dt.astimezone(tz)
        
        delta = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        new_dt = base_dt + delta
        
        return {
            "success": True,
            "original_time": base_dt.isoformat(),
            "new_time": new_dt.isoformat(),
            "duration_added": {
                "days": days,
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds
            },
            "formatted": new_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid time format: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
def list_timezones() -> Dict[str, Any]:
    """List all available timezones."""
    try:
        timezones = pytz.all_timezones
        
        return {
            "success": True,
            "timezones": timezones,
            "count": len(timezones),
            "common_timezones": [
                "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
                "Europe/London", "Europe/Paris", "Asia/Tokyo", "Asia/Shanghai", "Australia/Sydney"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }