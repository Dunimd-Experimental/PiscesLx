#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of PiscesL1.
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
import time
import logging
from typing import Any, Dict

class PiscesLxCoreLogBaseFormatter:
    """Base formatter that provides common emoji mapping and message parsing logic."""
    
    _EMOJI = {
        "INFO": "🟢 INFO",
        "DEBUG": "🔵 DEBUG", 
        "WARNING": "🟡 WARN",
        "ERROR": "🔴 ERROR",
        "CRITICAL": "🔴 CRITICAL",
        "SUCCESS": "✅ SUCCESS"
    }
    
    def _parse_message(self, record: logging.LogRecord) -> tuple[str, str]:
        """
        Parse the log record message and extract the tag and message content.

        Args:
            record (logging.LogRecord): The log record to be parsed.

        Returns:
            tuple[str, str]: A tuple containing the tag and message.
        """
        tag = "PiscesL1"
        message = ""
        
        try:
            if isinstance(record.msg, dict):
                # Convert the log message to a dictionary
                data = dict(record.msg)
                # Get the tag from the 'event' key, default to 'PiscesLx Core'
                tag = data.get("event", "PiscesLx Core")
                # Get the message from the 'message' key, default to an empty string
                message = data.get("message", "")
                # Collect additional fields and append them to the message
                extra_fields = []
                for key, value in data.items():
                    if key not in ["event", "message"]:
                        extra_fields.append(f"{key}={value}")
                if extra_fields:
                    message = f"{message} {' '.join(extra_fields)}" if message else " ".join(extra_fields)
            else:
                # Convert the log message to a string
                message = str(record.msg) if record.msg else ""
        except Exception:
            # Fallback to string representation of the message if an error occurs
            message = str(record.msg) if record.msg else ""
        
        return tag, message
    
    def _format_timestamp(self, record: logging.LogRecord) -> str:
        """
        Format the timestamp of the log record into a simplified format.

        Args:
            record (logging.LogRecord): The log record containing the timestamp to be formatted.

        Returns:
            str: A string representing the formatted timestamp in 'MMDD HH:mm:ss' format.
        """
        return time.strftime("%m%d %H:%M:%S", time.localtime(record.created))

    def _format_iso_utc(self, record: logging.LogRecord) -> str:
        """Format created time as ISO8601 in UTC with microseconds and 'Z'."""
        try:
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
            # Ensure microseconds are present, and append 'Z' suffix
            return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
        except Exception:
            # Fallback using time.gmtime
            return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime(record.created))

class PiscesLxCoreLogJsonFormatter(PiscesLxCoreLogBaseFormatter):
    """
    A formatter that uses colorful emojis to format log messages into a structured string.

    The output format is: `MMDD HH:mm:ss | EMOJI LEVEL | [TAG] | message`.
    Supports both dictionary and string log messages. For dictionary messages, it merges additional fields;
    for string messages, it uses the string directly as the message.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record into a string with emoji, timestamp, tag, and message.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: A formatted log string in the format `MMDD HH:mm:ss | ISO_TIMESTAMP | EMOJI | [TAG] | message`.
        """
        local_ts = self._format_timestamp(record)
        iso_ts = self._format_iso_utc(record)
        # Extract only the emoji without the level text
        level_emoji = self._EMOJI.get(record.levelname, "ℹ️ INFO")
        # Remove the level text, keeping only the emoji
        emoji_only = level_emoji.split()[0] if ' ' in level_emoji else level_emoji
        tag, message = self._parse_message(record)
        
        import re
        message = re.sub(r'\s*timestamp=[^\s]+\s*level=\w+\s*$', '', message)
        
        return f"{local_ts} | {iso_ts} | {emoji_only} | [{tag}] | {message}"

class PiscesLxCoreLogConsoleFormatter(PiscesLxCoreLogBaseFormatter):
    """
    A console formatter that uses colorful emojis and includes a timestamp.

    Formats log messages into the pattern: `MMDD HH:mm:ss | EMOJI | [TAG] | message`.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record into a string with emoji, timestamp, tag, and message for console output.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: A formatted log string in the format `MMDD HH:mm:ss | ISO_TIMESTAMP | EMOJI | [TAG] | message`.
        """
        timestamp = self._format_timestamp(record)
        iso_ts = self._format_iso_utc(record)
        level_emoji = self._EMOJI.get(record.levelname, "ℹ️ INFO")
        # Extract only the emoji without the level text
        emoji_only = level_emoji.split()[0] if ' ' in level_emoji else level_emoji
        tag, message = self._parse_message(record)
        
        # 移除消息末尾的timestamp和level字段
        import re
        message = re.sub(r'\s*timestamp=[^\s]+\s*level=\w+\s*$', '', message)
        
        return f"{timestamp} | {iso_ts} | {emoji_only} | [{tag}] | {message}"
    
    def debug_format(self, record: logging.LogRecord) -> str:
        """
        Generate a debug string to inspect the content of a log record.

        Args:
            record (logging.LogRecord): The log record to debug.

        Returns:
            str: A debug string containing the message, its type, and arguments of the log record.
        """
        return f"DEBUG: record.msg={repr(record.msg)}, type={type(record.msg)}, args={record.args}"

class PiscesLxCoreRealJsonFormatter(logging.Formatter):
    """
    A strict JSON formatter designed for machine consumption.

    Includes standard fields such as 'ts', 'level', 'tag', 'message', 'logger', and merges any extra fields
    provided in record.msg if it's a dictionary. Also incorporates additional fields from record.extra if present.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record into a JSON string.

        Args:
            record (logging.LogRecord): The log record to be formatted.

        Returns:
            str: A JSON-formatted string containing log information.
        """
        payload = {}
        # Add base fields to the payload using a simplified timestamp format
        payload["ts"] = time.strftime("%m%d %H:%M:%S", time.localtime(record.created))
        payload["level"] = record.levelname
        payload["logger"] = record.name
        
        # Initialize tag and message
        tag = "PiscesL1"
        message = ""
        
        try:
            if isinstance(record.msg, dict):
                # Convert the log message to a dictionary
                data = dict(record.msg)
                # Get the tag from the 'event' key, default to 'PiscesLx Core'
                tag = data.get("event", "PiscesLx Core")
                # Get the message from the 'message' key
                message = data.get("message", message)
                # Update payload with additional fields
                payload.update({k: v for k, v in data.items() if k not in ("event", "message")})
            else:
                # Convert the log message to a string
                message = str(record.msg) if record.msg else ""
        except Exception:
            # Fallback to string representation of the message if an error occurs
            message = str(record.msg) if record.msg else ""
        
        payload["tag"] = tag
        payload["message"] = message
        
        # Include extra fields from the record if available
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        
        return json.dumps(payload, ensure_ascii=False)