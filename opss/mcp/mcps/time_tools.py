#!/usr/bin/env/python3
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

"""
Time Tools - Time, date, and timezone utilities
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import POPSSMCPToolBase, POPSSMCPToolResult


class CurrentTimeTool(POPSSMCPToolBase):
    name = "current_time"
    description = "Get current time in specified timezone"
    category = "utility"
    tags = ["time", "date", "timezone"]
    
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone name (e.g., 'UTC', 'Asia/Shanghai', 'America/New_York')",
                "default": "UTC"
            },
            "format": {
                "type": "string",
                "description": "Output format: 'iso', 'unix', 'custom'",
                "default": "iso"
            },
            "custom_format": {
                "type": "string",
                "description": "Custom strftime format (used when format='custom')"
            }
        },
        "required": []
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        tz_name = arguments.get("timezone", "UTC")
        fmt = arguments.get("format", "iso")
        custom_format = arguments.get("custom_format")
        
        try:
            import pytz
            
            tz = pytz.timezone(tz_name)
            now = datetime.now(tz)
            
            if fmt == "iso":
                result = now.isoformat()
            elif fmt == "unix":
                result = int(now.timestamp())
            elif fmt == "custom" and custom_format:
                result = now.strftime(custom_format)
            else:
                result = now.isoformat()
            
            return self._create_success_result({
                "timezone": tz_name,
                "datetime": result,
                "unix_timestamp": int(now.timestamp()),
                "day_of_week": now.strftime("%A"),
                "is_dst": bool(now.dst()),
            })
            
        except ImportError:
            now = datetime.now(timezone.utc)
            return self._create_success_result({
                "timezone": "UTC",
                "datetime": now.isoformat(),
                "unix_timestamp": int(now.timestamp()),
                "day_of_week": now.strftime("%A"),
                "note": "pytz not installed, using UTC only"
            })
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)


class TimeConvertTool(POPSSMCPToolBase):
    name = "time_convert"
    description = "Convert time between timezones"
    category = "utility"
    tags = ["time", "convert", "timezone"]
    
    parameters = {
        "type": "object",
        "properties": {
            "time": {
                "type": "string",
                "description": "Time to convert (ISO format or unix timestamp)"
            },
            "from_timezone": {
                "type": "string",
                "description": "Source timezone",
                "default": "UTC"
            },
            "to_timezone": {
                "type": "string",
                "description": "Target timezone"
            }
        },
        "required": ["time", "to_timezone"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        time_str = arguments.get("time", "")
        from_tz = arguments.get("from_timezone", "UTC")
        to_tz = arguments.get("to_timezone", "UTC")
        
        if not time_str or not to_tz:
            return self._create_error_result("time and to_timezone are required", "ValidationError")
        
        try:
            import pytz
            
            if time_str.isdigit():
                dt = datetime.fromtimestamp(int(time_str), tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            
            if dt.tzinfo is None:
                source_tz = pytz.timezone(from_tz)
                dt = source_tz.localize(dt)
            
            target_tz = pytz.timezone(to_tz)
            converted = dt.astimezone(target_tz)
            
            return self._create_success_result({
                "original_time": dt.isoformat(),
                "original_timezone": from_tz,
                "converted_time": converted.isoformat(),
                "converted_timezone": to_tz,
                "unix_timestamp": int(converted.timestamp()),
            })
            
        except ImportError:
            return self._create_error_result(
                "pytz required. Install with: pip install pytz",
                "DependencyError"
            )
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)


class TimezoneListTool(POPSSMCPToolBase):
    name = "timezone_list"
    description = "List available timezones"
    category = "utility"
    tags = ["time", "timezone", "list"]
    
    parameters = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "string",
                "description": "Filter timezones by region (e.g., 'Asia', 'America', 'Europe')"
            }
        },
        "required": []
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        filter_region = arguments.get("filter", "")
        
        try:
            import pytz
            
            all_timezones = pytz.all_timezones
            
            if filter_region:
                filtered = [tz for tz in all_timezones if tz.startswith(filter_region)]
            else:
                filtered = all_timezones[:50]
            
            return self._create_success_result({
                "count": len(filtered),
                "total": len(all_timezones),
                "timezones": filtered,
            })
            
        except ImportError:
            return self._create_success_result({
                "count": 1,
                "total": 1,
                "timezones": ["UTC"],
                "note": "pytz not installed, only UTC available"
            })
        except Exception as e:
            return self._create_error_result(str(e), type(e).__name__)


class StopwatchTool(POPSSMCPToolBase):
    name = "stopwatch"
    description = "Start, stop, or check stopwatch timer"
    category = "utility"
    tags = ["time", "timer", "stopwatch"]
    
    _start_times: Dict[str, float] = {}
    
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: 'start', 'stop', 'check'",
                "enum": ["start", "stop", "check"]
            },
            "timer_id": {
                "type": "string",
                "description": "Timer identifier",
                "default": "default"
            }
        },
        "required": ["action"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> POPSSMCPToolResult:
        import time
        
        action = arguments.get("action", "check")
        timer_id = arguments.get("timer_id", "default")
        
        if action == "start":
            self._start_times[timer_id] = time.time()
            return self._create_success_result({
                "action": "started",
                "timer_id": timer_id,
                "start_time": datetime.now().isoformat(),
            })
        
        elif action == "stop":
            if timer_id not in self._start_times:
                return self._create_error_result(f"Timer '{timer_id}' not started", "TimerError")
            
            elapsed = time.time() - self._start_times[timer_id]
            del self._start_times[timer_id]
            
            return self._create_success_result({
                "action": "stopped",
                "timer_id": timer_id,
                "elapsed_seconds": round(elapsed, 3),
                "elapsed_human": self._format_duration(elapsed),
            })
        
        elif action == "check":
            if timer_id not in self._start_times:
                return self._create_error_result(f"Timer '{timer_id}' not started", "TimerError")
            
            elapsed = time.time() - self._start_times[timer_id]
            
            return self._create_success_result({
                "action": "checked",
                "timer_id": timer_id,
                "elapsed_seconds": round(elapsed, 3),
                "elapsed_human": self._format_duration(elapsed),
            })
        
        else:
            return self._create_error_result(f"Invalid action: {action}", "ValidationError")
    
    def _format_duration(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs:.1f}s"
        elif minutes > 0:
            return f"{minutes}m {secs:.1f}s"
        else:
            return f"{secs:.3f}s"
