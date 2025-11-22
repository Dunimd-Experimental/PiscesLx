#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
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

import sys
import pytz
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.mcp import PiscesLxCoreMCPPlaza

# Create mcp instance for tool registration
mcp = PiscesLxCoreMCPPlaza()

@mcp.tool()
def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """
    Get the current time in a specific timezone.
    
    Args:
        timezone (str): The timezone to get the current time for. Defaults to "UTC".
        
    Returns:
        Dict[str, Any]: A dictionary containing the current time information.
            - success (bool): Whether the operation was successful.
            - datetime (str): ISO formatted datetime string.
            - timezone (str): The requested timezone.
            - year (int): Current year.
            - month (int): Current month.
            - day (int): Current day.
            - hour (int): Current hour.
            - minute (int): Current minute.
            - second (int): Current second.
            - microsecond (int): Current microsecond.
            - weekday (str): Full name of the weekday.
            - formatted (str): Formatted datetime string in "%Y-%m-%d %H:%M:%S %Z" format.
            
    Raises:
        Exception: If there's an error processing the timezone or getting the current time.
    """
    try:
        if timezone not in pytz.all_timezones:
            return {
                "success": False,
                "error": f"Invalid timezone: {timezone}",
                "available_timezones": pytz.all_timezones[:10]
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
    """
    Parse a time string into a structured format.
    
    Args:
        time_string (str): The time string to parse.
        input_format (str): The format of the input time string. Defaults to "%Y-%m-%d %H:%M:%S".
        timezone (str): The timezone to use for the parsed time. Defaults to "UTC".
        
    Returns:
        Dict[str, Any]: A dictionary containing the parsed time information.
            - success (bool): Whether the operation was successful.
            - datetime (str): ISO formatted datetime string.
            - timezone (str): The requested timezone.
            - year (int): Year from the parsed time.
            - month (int): Month from the parsed time.
            - day (int): Day from the parsed time.
            - hour (int): Hour from the parsed time.
            - minute (int): Minute from the parsed time.
            - second (int): Second from the parsed time.
            - weekday (str): Full name of the weekday.
            - formatted (str): Formatted datetime string in "%Y-%m-%d %H:%M:%S %Z" format.
            
    Raises:
        ValueError: If the time string doesn't match the input format.
        Exception: If there's an error processing the timezone or parsing the time.
    """
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
    """
    Calculate the difference between two times.
    
    Args:
        start_time (str): The start time in ISO format.
        end_time (str): The end time in ISO format.
        timezone (str): The timezone to use for the calculation. Defaults to "UTC".
        
    Returns:
        Dict[str, Any]: A dictionary containing the time difference information.
            - success (bool): Whether the operation was successful.
            - difference_seconds (float): Time difference in seconds.
            - difference_days (int): Time difference in days.
            - difference_hours (float): Time difference in hours.
            - difference_minutes (float): Time difference in minutes.
            - start_time (str): ISO formatted start time.
            - end_time (str): ISO formatted end time.
            - human_readable (str): Human-readable representation of the time difference.
            
    Raises:
        ValueError: If the time strings don't match the expected ISO format.
        Exception: If there's an error processing the timezone or calculating the difference.
    """
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
    """
    Format an ISO time string to a custom format.
    
    Args:
        iso_time (str): The ISO formatted time string to format.
        output_format (str): The desired output format. Defaults to "%Y-%m-%d %H:%M:%S".
        timezone (str): The timezone to use for formatting. Defaults to "UTC".
        
    Returns:
        Dict[str, Any]: A dictionary containing the formatted time information.
            - success (bool): Whether the operation was successful.
            - formatted_time (str): The formatted time string.
            - timezone (str): The requested timezone.
            - iso_time (str): ISO formatted datetime string.
            - format_used (str): The format string used for formatting.
            
    Raises:
        ValueError: If the ISO time string is invalid.
        Exception: If there's an error processing the timezone or formatting the time.
    """
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
    """
    Add a duration to a base time.
    
    Args:
        base_time (str): The base time in ISO format.
        days (int): Number of days to add. Defaults to 0.
        hours (int): Number of hours to add. Defaults to 0.
        minutes (int): Number of minutes to add. Defaults to 0.
        seconds (int): Number of seconds to add. Defaults to 0.
        timezone (str): The timezone to use for the calculation. Defaults to "UTC".
        
    Returns:
        Dict[str, Any]: A dictionary containing the new time information.
            - success (bool): Whether the operation was successful.
            - original_time (str): ISO formatted original time.
            - new_time (str): ISO formatted new time after adding the duration.
            - duration_added (dict): Dictionary containing the duration components.
                - days (int): Days added.
                - hours (int): Hours added.
                - minutes (int): Minutes added.
                - seconds (int): Seconds added.
            - formatted (str): Formatted datetime string in "%Y-%m-%d %H:%M:%S %Z" format.
            
    Raises:
        ValueError: If the base time string is invalid.
        Exception: If there's an error processing the timezone or adding the duration.
    """
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
    """
    List all available timezones.
    
    Returns:
        Dict[str, Any]: A dictionary containing the timezone information.
            - success (bool): Whether the operation was successful.
            - timezones (list): List of all available timezones.
            - count (int): Total number of available timezones.
            - common_timezones (list): List of commonly used timezones.
            
    Raises:
        Exception: If there's an error retrieving the timezone list.
    """
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
