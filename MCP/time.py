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
import logging
from typing import Dict, Any
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

try:
    # Try to import get_localzone_name from tzlocal
    from tzlocal import get_localzone_name
    def get_local_timezone():
        """
        Get the local timezone name.

        Attempts to get the local timezone name using tzlocal.
        If an error occurs, falls back to "UTC".

        Returns:
            str: The local timezone name or "UTC" if an error occurs.
        """
        try:
            return get_localzone_name()
        except:
            return "UTC"
except ImportError:
    def get_local_timezone():
        """
        Fallback function to get the timezone name.

        Returns "UTC" when tzlocal module is not available.

        Returns:
            str: "UTC" as the default timezone.
        """
        return "UTC"

# Import register_tool from simple_mcp module
from .simple_mcp import register_tool

# Initialize logger for the current module
logger = logging.getLogger(__name__)

class TimeTool:
    """
    A tool for time and timezone conversion operations.

    This class provides functionality to get the current time, convert time between timezones,
    and list common timezones.
    """
    
    def __init__(self):
        """
        Initialize the TimeTool instance.

        Sets the tool name, description, and determines the local timezone.
        """
        self.name = "time"
        self.description = "Time and timezone conversion functionality"
        self.local_timezone = get_local_timezone()
        
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the schema for the time tool operations.

        Defines the structure and requirements for the input parameters of the time operations.

        Returns:
            Dict[str, Any]: A dictionary representing the JSON schema for the tool.
        """
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Time operation to perform",
                    "enum": ["get_current_time", "convert_time", "list_timezones"]
                },
                "timezone": {
                    "type": "string",
                    "description": f"IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Use '{self.local_timezone}' as local timezone."
                },
                "source_timezone": {
                    "type": "string",
                    "description": "Source IANA timezone name for conversion"
                },
                "target_timezone": {
                    "type": "string",
                    "description": "Target IANA timezone name for conversion"
                },
                "time": {
                    "type": "string",
                    "description": "Time to convert in 24-hour format (HH:MM)"
                }
            },
            "required": ["operation"]
        }
    
    def _get_zoneinfo(self, timezone_name: str) -> ZoneInfo:
        """
        Get the ZoneInfo object for a given timezone name.

        Args:
            timezone_name (str): The IANA timezone name.

        Returns:
            ZoneInfo: The ZoneInfo object corresponding to the given timezone name.

        Raises:
            ValueError: If the provided timezone name is invalid.
        """
        try:
            return ZoneInfo(timezone_name)
        except Exception as e:
            raise ValueError(f"Invalid timezone '{timezone_name}': {str(e)}")
    
    def _get_current_time(self, timezone_name: str) -> Dict[str, Any]:
        """
        Get the current time in the specified timezone.

        Args:
            timezone_name (str): The IANA timezone name.

        Returns:
            Dict[str, Any]: A dictionary containing the current time information or an error message.
        """
        try:
            timezone = self._get_zoneinfo(timezone_name)
            current_time = datetime.now(timezone)
            
            return {
                "success": True,
                "data": {
                    "timezone": timezone_name,
                    "datetime": current_time.isoformat(timespec="seconds"),
                    "time": current_time.strftime("%H:%M:%S"),
                    "date": current_time.strftime("%Y-%m-%d"),
                    "day_of_week": current_time.strftime("%A"),
                    "is_dst": bool(current_time.dst()),
                    "utc_offset": str(current_time.utcoffset() or timedelta()),
                    "epoch": int(current_time.timestamp())
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _convert_time(self, source_tz: str, time_str: str, target_tz: str) -> Dict[str, Any]:
        """
        Convert a given time from a source timezone to a target timezone.

        Args:
            source_tz (str): The source IANA timezone name.
            time_str (str): The time to convert in 24-hour format (HH:MM).
            target_tz (str): The target IANA timezone name.

        Returns:
            Dict[str, Any]: A dictionary containing the conversion result or an error message.
        """
        try:
            source_timezone = self._get_zoneinfo(source_tz)
            target_timezone = self._get_zoneinfo(target_tz)
            
            # Parse the input time string
            try:
                parsed_time = datetime.strptime(time_str, "%H:%M").time()
            except ValueError:
                return {
                    "success": False,
                    "error": "Invalid time format. Expected HH:MM [24-hour format]"
                }
            
            # Create a datetime object with today's date in the source timezone
            now = datetime.now(source_timezone)
            source_time = datetime(
                now.year, now.month, now.day,
                parsed_time.hour, parsed_time.minute,
                tzinfo=source_timezone
            )
            
            # Convert the source time to the target timezone
            target_time = source_time.astimezone(target_timezone)
            
            # Calculate the difference in hours between the two timezones
            source_offset = source_time.utcoffset() or timedelta()
            target_offset = target_time.utcoffset() or timedelta()
            hours_difference = (target_offset - source_offset).total_seconds() / 3600
            
            if hours_difference.is_integer():
                time_diff_str = f"{hours_difference:+.1f}h"
            else:
                time_diff_str = f"{hours_difference:+.2f}h"
            
            return {
                "success": True,
                "data": {
                    "source": {
                        "timezone": source_tz,
                        "datetime": source_time.isoformat(timespec="seconds"),
                        "time": source_time.strftime("%H:%M"),
                        "is_dst": bool(source_time.dst())
                    },
                    "target": {
                        "timezone": target_tz,
                        "datetime": target_time.isoformat(timespec="seconds"),
                        "time": target_time.strftime("%H:%M"),
                        "is_dst": bool(target_time.dst())
                    },
                    "time_difference": time_diff_str,
                    "hours_difference": hours_difference
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _list_timezones(self) -> Dict[str, Any]:
        """
        List common IANA timezone names.

        Returns:
            Dict[str, Any]: A dictionary containing the local timezone, common timezones, and their count.
        """
        common_timezones = [
            "UTC",
            "America/New_York",
            "America/Chicago", 
            "America/Denver",
            "America/Los_Angeles",
            "America/Sao_Paulo",
            "Europe/London",
            "Europe/Paris",
            "Europe/Berlin",
            "Europe/Moscow",
            "Asia/Tokyo",
            "Asia/Shanghai",
            "Asia/Kolkata",
            "Asia/Dubai",
            "Australia/Sydney",
            "Australia/Melbourne",
            "Pacific/Auckland",
            "Africa/Cairo",
            "Africa/Johannesburg",
            "America/Mexico_City"
        ]
        
        return {
            "success": True,
            "data": {
                "local_timezone": self.local_timezone,
                "common_timezones": common_timezones,
                "count": len(common_timezones)
            }
        }
    
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a time operation based on the provided operation name and parameters.

        Args:
            operation (str): The name of the operation to perform.
            **kwargs: Additional parameters required by the operation.

        Returns:
            Dict[str, Any]: The result of the operation or an error message.
        """
        if operation == "get_current_time":
            timezone = kwargs.get("timezone", self.local_timezone)
            return self._get_current_time(timezone)
            
        elif operation == "convert_time":
            source_tz = kwargs.get("source_timezone", self.local_timezone)
            time_str = kwargs.get("time")
            target_tz = kwargs.get("target_timezone")
            
            if not time_str:
                return {
                    "success": False,
                    "error": "Missing required parameter: time"
                }
            
            if not target_tz:
                return {
                    "success": False,
                    "error": "Missing required parameter: target_timezone"
                }
            
            return self._convert_time(source_tz, time_str, target_tz)
            
        elif operation == "list_timezones":
            return self._list_timezones()
            
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}"
            }

# Register the TimeTool instance as a tool
time_tool = TimeTool()
register_tool(
    time_tool.name,
    time_tool.description,
    time_tool.get_schema(),
    time_tool.execute
)