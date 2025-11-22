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

import os
import sys
import time
import logging
import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from .context import PiscesLxCoreLogContext
from .handlers import build_rotating_file_handler
from typing import Optional, Dict, Any, Callable, List
from .formatters import PiscesLxCoreLogJsonFormatter,PiscesLxCoreLogConsoleFormatter,PiscesLxCoreRealJsonFormatter

SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

_CONSOLE_HANDLER_NAME = "PiscesLxConsole"
_FILE_HANDLER_NAME = "PiscesLxFile"

@dataclass
class LogPattern:
    """Represents a log pattern with its occurrence statistics.

    Attributes:
        pattern (str): The log pattern string.
        count (int): The number of times the pattern has occurred. Defaults to 0.
        first_occurrence (datetime): The first occurrence time of the pattern. 
                                   Defaults to the current UTC time.
        last_occurrence (datetime): The last occurrence time of the pattern. 
                                  Defaults to the current UTC time.
        severity (str): The severity level of the log pattern. Defaults to "info".
    """
    pattern: str
    count: int = 0
    first_occurrence: datetime = field(default_factory=datetime.utcnow)
    last_occurrence: datetime = field(default_factory=datetime.utcnow)
    severity: str = "info"

@dataclass
class LogAnalysisResult:
    """Represents the result of log analysis.

    Attributes:
        total_logs (int): The total number of logs analyzed. Defaults to 0.
        error_count (int): The number of error logs. Defaults to 0.
        warning_count (int): The number of warning logs. Defaults to 0.
        info_count (int): The number of info logs. Defaults to 0.
        debug_count (int): The number of debug logs. Defaults to 0.
        patterns (Dict[str, LogPattern]): A dictionary of log patterns. 
                                       Defaults to an empty dictionary.
        top_errors (List[str]): A list of the most common error messages. 
                              Defaults to an empty list.
        timestamp (datetime): The time when the analysis was performed. 
                            Defaults to the current UTC time.
    """
    total_logs: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    debug_count: int = 0
    patterns: Dict[str, LogPattern] = field(default_factory=dict)
    top_errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

class LogAnalyzer:
    """A class for analyzing log records and detecting patterns."""
    
    def __init__(self, window_size: int = 10000):
        """Initialize the LogAnalyzer instance.

        Args:
            window_size (int, optional): The maximum number of logs to keep in the buffer. 
                                       Defaults to 10000.
        """
        self.window_size = window_size
        self.log_buffer = deque(maxlen=window_size)
        self.patterns: Dict[str, LogPattern] = {}
        self.lock = threading.RLock()
        self.analysis_callbacks: List[Callable[[LogAnalysisResult], None]] = []
    
    def add_log(self, log_record: Dict[str, Any]):
        """Add a log record for analysis.

        Args:
            log_record (Dict[str, Any]): The log record to be added.
        """
        with self.lock:
            self.log_buffer.append(log_record)
            
            # Update pattern statistics
            if "message" in log_record:
                message = log_record["message"]
                # Simple pattern extraction (may need more complex pattern recognition in real applications)
                pattern_key = message[:50]  # Use the first 50 characters of the message as the pattern key
                if pattern_key not in self.patterns:
                    self.patterns[pattern_key] = LogPattern(
                        pattern=message,
                        first_occurrence=datetime.utcnow(),
                        severity=log_record.get("level", "info")
                    )
                self.patterns[pattern_key].count += 1
                self.patterns[pattern_key].last_occurrence = datetime.utcnow()
    
    def add_analysis_callback(self, callback: Callable[[LogAnalysisResult], None]):
        """Add a callback function to be called after analysis.

        Args:
            callback (Callable[[LogAnalysisResult], None]): The callback function.
        """
        self.analysis_callbacks.append(callback)
    
    def analyze(self) -> LogAnalysisResult:
        """Analyze the logs in the buffer and return the analysis result.

        Returns:
            LogAnalysisResult: The result of the log analysis.
        """
        with self.lock:
            result = LogAnalysisResult()
            result.total_logs = len(self.log_buffer)
            
            # Count logs by level
            for log in self.log_buffer:
                level = log.get("level", "info").lower()
                if level == "error":
                    result.error_count += 1
                elif level == "warning":
                    result.warning_count += 1
                elif level == "info":
                    result.info_count += 1
                elif level == "debug":
                    result.debug_count += 1
            
            # Get the most common log patterns
            sorted_patterns = sorted(
                self.patterns.items(), 
                key=lambda x: x[1].count, 
                reverse=True
            )
            result.patterns = dict(sorted_patterns[:20])  # Keep the top 20 patterns
            
            # Get the most common error messages
            error_patterns = [
                pattern for pattern in self.patterns.values() 
                if pattern.severity == "error"
            ]
            error_patterns.sort(key=lambda x: x.count, reverse=True)
            result.top_errors = [p.pattern for p in error_patterns[:10]]
            
            # Call callback functions
            for callback in self.analysis_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    # Keep the main process running and log a low-noise debug message for troubleshooting
                    try:
                        from utils.log.core import PiscesLxCoreLog as _L
                        _L("pisceslx.log").debug(
                            "log analysis callback error",
                            event="log.analysis.callback.error",
                            callback=getattr(callback, "__name__", str(callback)),
                            error=str(e),
                            error_class=type(e).__name__,
                        )
                    except Exception:
                        pass
            
            return result
    
    def get_patterns_in_time_window(self, minutes: int = 60) -> Dict[str, LogPattern]:
        """Get log patterns that occurred within the specified time window.

        Args:
            minutes (int, optional): The time window in minutes. Defaults to 60.

        Returns:
            Dict[str, LogPattern]: A dictionary of log patterns that occurred within the time window.
        """
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            return {
                k: v for k, v in self.patterns.items() 
                if v.last_occurrence >= cutoff_time
            }

class SecurityAuditor:
    """A class for auditing log records for sensitive information."""
    
    def __init__(self):
        """Initialize the SecurityAuditor instance."""
        self.sensitive_patterns = [
            r"password\s*[:=]\s*['\"][^'\"]+['\"]",
            r"secret\s*[:=]\s*['\"][^'\"]+['\"]",
            r"key\s*[:=]\s*['\"][^'\"]+['\"]",
            r"token\s*[:=]\s*['\"][^'\"]+['\"]",
        ]
        self.audit_log = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def audit_log_record(self, log_record: Dict[str, Any]) -> bool:
        """Audit a log record for sensitive information.

        Args:
            log_record (Dict[str, Any]): The log record to be audited.

        Returns:
            bool: True if sensitive information is found, False otherwise.
        """
        import re
        
        message = str(log_record.get("message", ""))
        # Check for sensitive information
        for pattern in self.sensitive_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                with self.lock:
                    self.audit_log.append({
                        "timestamp": datetime.utcnow(),
                        "pattern": pattern,
                        "message_preview": message[:100],
                        "log_level": log_record.get("level", "info")
                    })
                return True
        return False
    
    def get_audit_report(self) -> List[Dict[str, Any]]:
        """Get the audit report containing records with sensitive information.

        Returns:
            List[Dict[str, Any]]: A list of audit records.
        """
        with self.lock:
            return list(self.audit_log)

class RealTimeLogMonitor:
    """A class for monitoring log files in real-time."""
    
    def __init__(self, log_file_path: str):
        """Initialize the RealTimeLogMonitor instance.

        Args:
            log_file_path (str): The path to the log file to monitor.
        """
        self.log_file_path = log_file_path
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.monitor_thread = None
        self.stop_monitor = threading.Event()
        self.last_position = 0
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback function to be called when a new log is detected.

        Args:
            callback (Callable[[Dict[str, Any]], None]): The callback function.
        """
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """Start monitoring the log file."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.stop_monitor.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring the log file."""
        self.stop_monitor.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """The main monitoring loop that checks for new logs in the file."""
        # Get the initial position of the file
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                f.seek(0, 2)  # Move to the end of the file
                self.last_position = f.tell()
        except Exception:
            pass
        
        while not self.stop_monitor.wait(1.0):  # Check every second
            try:
                with open(self.log_file_path, 'r', encoding='utf-8') as f:
                    f.seek(self.last_position)
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        
                        # Parse the log line
                        log_record = self._parse_log_line(line)
                        if log_record:
                            for callback in self.callbacks:
                                try:
                                    callback(log_record)
                                except Exception:
                                    pass
                    
                    self.last_position = f.tell()
            except Exception:
                pass
    
    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a log line into a dictionary.

        Args:
            line (str): The log line to parse.

        Returns:
            Optional[Dict[str, Any]]: The parsed log record, or a basic record if parsing fails.
                                     Returns None if an unexpected error occurs.
        """
        try:
            import json
            return json.loads(line)
        except Exception:
            # If not in JSON format, create a basic record
            return {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": "info",
                "message": line.strip()
            }

class PiscesLxCoreLogManager:
    """A singleton manager class for advanced logging functionality."""
    
    _instances: Dict[str, 'PiscesLxCoreLogManager'] = {}
    _lock = threading.Lock()
    
    def __init__(self, name: str = "pisces"):
        """Initialize the PiscesLxCoreLogManager instance.

        Args:
            name (str, optional): The name of the logger manager. Defaults to "pisces".
        """
        self.name = name
        self._sampling_rates: Dict[str, float] = {}
        self._anomaly_detectors: Dict[str, Callable] = {}
        self._trace_contexts: Dict[str, str] = {}
        self._log_buffer: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self.log_analyzer = LogAnalyzer()
        self.security_auditor = SecurityAuditor()
        self.real_time_monitor: Optional[RealTimeLogMonitor] = None
        self.log_file_path: Optional[str] = None
        
    @classmethod
    def get_instance(cls, name: str = "pisces") -> 'PiscesLxCoreLogManager':
        """Get a singleton instance of the PiscesLxCoreLogManager.

        Args:
            name (str, optional): The name of the logger manager. Defaults to "pisces".

        Returns:
            PiscesLxCoreLogManager: The singleton instance of the manager.
        """
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name)
            return cls._instances[name]
    
    def set_sampling_rate(self, event: str, rate: float) -> None:
        """Set the sampling rate for a specific event.

        Args:
            event (str): The name of the event.
            rate (float): The sampling rate, should be between 0.0 and 1.0.
        """
        self._sampling_rates[event] = max(0.0, min(1.0, rate))
    
    def get_sampling_rate(self, event: str) -> float:
        """Get the sampling rate for a specific event.

        Args:
            event (str): The name of the event.

        Returns:
            float: The sampling rate, defaults to 1.0 if not set.
        """
        return self._sampling_rates.get(event, 1.0)
    
    def add_anomaly_detector(self, name: str, detector: Callable[[Dict[str, Any]], bool]) -> None:
        """Add an anomaly detector function.

        Args:
            name (str): The name of the anomaly detector.
            detector (Callable[[Dict[str, Any]], bool]): The detector function.
        """
        self._anomaly_detectors[name] = detector
    
    def generate_trace_id(self) -> str:
        """Generate a distributed tracing ID.

        Returns:
            str: The generated tracing ID.
        """
        return f"trace_{int(time.time() * 1000000)}_{threading.get_ident()}"
    
    def set_trace_context(self, trace_id: str) -> None:
        """Set the tracing context for the current thread.

        Args:
            trace_id (str): The tracing ID to set.
        """
        self._trace_contexts[threading.get_ident()] = trace_id
    
    def get_trace_context(self) -> Optional[str]:
        """Get the tracing context for the current thread.

        Returns:
            Optional[str]: The tracing ID if set, None otherwise.
        """
        return self._trace_contexts.get(threading.get_ident())
    
    def should_log(self, event: str) -> bool:
        """Determine whether a log should be recorded based on the sampling rate.

        Args:
            event (str): The name of the event.

        Returns:
            bool: True if the log should be recorded, False otherwise.
        """
        rate = self.get_sampling_rate(event)
        if rate >= 1.0:
            return True
        import random
        return random.random() < rate
    
    def detect_anomalies(self, record: Dict[str, Any]) -> List[str]:
        """Detect anomalies in a log record using registered anomaly detectors.

        Args:
            record (Dict[str, Any]): The log record to check.

        Returns:
            List[str]: A list of names of the anomaly detectors that triggered.
        """
        anomalies = []
        for name, detector in self._anomaly_detectors.items():
            try:
                if detector(record):
                    anomalies.append(name)
            except Exception:
                pass
        return anomalies
    
    def add_log_for_analysis(self, log_record: Dict[str, Any]):
        """Add a log record for analysis and security auditing.

        Args:
            log_record (Dict[str, Any]): The log record to add.
        """
        self.log_analyzer.add_log(log_record)
        
        # Perform security audit
        self.security_auditor.audit_log_record(log_record)
    
    def get_log_analysis(self) -> LogAnalysisResult:
        """Get the result of log analysis.

        Returns:
            LogAnalysisResult: The result of the log analysis.
        """
        return self.log_analyzer.analyze()
    
    def add_analysis_callback(self, callback: Callable[[LogAnalysisResult], None]):
        """Add a callback function to be called after log analysis.

        Args:
            callback (Callable[[LogAnalysisResult], None]): The callback function.
        """
        self.log_analyzer.add_analysis_callback(callback)
    
    def get_security_audit_report(self) -> List[Dict[str, Any]]:
        """Get the security audit report.

        Returns:
            List[Dict[str, Any]]: A list of audit records.
        """
        return self.security_auditor.get_audit_report()
    
    def start_real_time_monitoring(self, log_file_path: str):
        """Start real-time monitoring of a log file.

        Args:
            log_file_path (str): The path to the log file to monitor.
        """
        self.log_file_path = log_file_path
        self.real_time_monitor = RealTimeLogMonitor(log_file_path)
        
        # Add a callback to send logs to the analyzer
        def log_callback(log_record: Dict[str, Any]):
            self.add_log_for_analysis(log_record)
        
        self.real_time_monitor.add_callback(log_callback)
        self.real_time_monitor.start_monitoring()
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring of the log file."""
        if self.real_time_monitor:
            self.real_time_monitor.stop_monitoring()
            self.real_time_monitor = None

class PiscesLxCoreLog:
    """A flagship structured logger with advanced features.

    Features:
    - Multiple handlers: console + rotating file (by size or time)
    - JSON structured logs (merging user dictionary and context)
    - Dynamic level changes
    - Intelligent sampling based on log importance and system load
    - Anomaly detection in log patterns
    - Distributed tracing support
    - Asynchronous logging capabilities
    - Log buffering for high-throughput scenarios
    - No external dependencies by default
    - Intelligent log analysis and security auditing
    """

    def __init__(
        self,
        name: str = "pisces",
        level: Optional[str] = None,
        console: bool = True,
        file_path: Optional[str] = None,
        rotate_when: str = "time",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 7,
        json_format_console: bool = False,
        enable_file: bool = True,
        file_format: str = "text",  # "text" | "json"
    ) -> None:
        """Initialize the PiscesLxCoreLog instance.

        Args:
            name (str, optional): The name of the logger. Defaults to "pisces".
            level (Optional[str], optional): The logging level. If None, it will be determined automatically. Defaults to None.
            console (bool, optional): Whether to enable the console handler. Defaults to True.
            file_path (Optional[str], optional): The path to the log file. If None, a default path will be generated. Defaults to None.
            rotate_when (str, optional): The rotation condition for the file handler. Defaults to "time".
            max_bytes (int, optional): The maximum size of the log file before rotation. Defaults to 10 * 1024 * 1024.
            backup_count (int, optional): The number of backup log files to keep. Defaults to 7.
            json_format_console (bool, optional): Whether to use JSON format for the console handler. Defaults to False.
            enable_file (bool, optional): Whether to enable the file handler. Defaults to True.
            file_format (str, optional): The format of the file log, either "text" or "json". Defaults to "text".
        """
        self._logger = logging.getLogger(name)
        self._logger.propagate = False
        self._manager = PiscesLxCoreLogManager.get_instance(name)

        # Automatically set the logging level: DEBUG if in an interactive TTY, otherwise INFO
        auto_level = "DEBUG" if getattr(sys.stdout, "isatty", lambda: False)() else "INFO"
        self.set_level(level or auto_level)

        if console:
            # Remove existing console handlers with our name to avoid duplicates
            for h in list(self._logger.handlers):
                if getattr(h, 'name', '') == _CONSOLE_HANDLER_NAME:
                    self._logger.removeHandler(h)
            ch = logging.StreamHandler()
            ch.set_name(_CONSOLE_HANDLER_NAME)
            ch.setFormatter(
                PiscesLxCoreLogJsonFormatter() if json_format_console else PiscesLxCoreLogConsoleFormatter()
            )
            self._logger.addHandler(ch)

        if file_path is None and enable_file:
            try:
                from utils.config.loader import PiscesLxCoreConfigLoader
                loader = PiscesLxCoreConfigLoader()
                file_path = str(loader._project_root / ".pisceslx" / "logs" / "app.log")
            except Exception:
                # If project root resolution fails, fall back to the home directory
                home = os.path.expanduser("~")
                file_path = os.path.join(home, ".pisceslx", "logs", "app.log")

        if enable_file and file_path:
            try:
                # Normalize any provided path to .pisceslx/logs/<basename>
                try:
                    from utils.fs.core import PiscesLxCoreFS as _FS
                    _fs = _FS()
                    file_path = str(_fs.normalize_under_category('logs', file_path))
                except Exception:
                    pass
                # Remove existing file handlers with our name to avoid duplicates
                for h in list(self._logger.handlers):
                    if getattr(h, 'name', '') == _FILE_HANDLER_NAME:
                        self._logger.removeHandler(h)
                fh = build_rotating_file_handler(file_path, rotate_when, max_bytes, backup_count)
                fh.set_name(_FILE_HANDLER_NAME)
                if (file_format or "text").lower() == "json":
                    fh.setFormatter(PiscesLxCoreRealJsonFormatter())
                else:
                    fh.setFormatter(PiscesLxCoreLogJsonFormatter())
                self._logger.addHandler(fh)
                
                # Start real-time monitoring
                self._manager.start_real_time_monitoring(file_path)
            except Exception as e:
                # Gracefully degrade to console-only logging if file handler fails
                self._logger.warning({
                    "event": "LOG",
                    "message": "File handler disabled",
                    "path": file_path,
                    "error": str(e),
                })

    def set_level(self, level: str) -> None:
        """Set the logging level of the logger.

        Args:
            level (str): The logging level to set.
        """
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        self._logger.setLevel(level_map.get(level.upper(), logging.INFO))

    def debug(self, msg: Any, **kwargs: Any) -> None:
        """Log a debug message.

        Args:
            msg (Any): The message to log.
            **kwargs: Additional context information to include in the log.
        """
        self._log("DEBUG", msg, **kwargs)

    def info(self, msg: Any, **kwargs: Any) -> None:
        """Log an info message.

        Args:
            msg (Any): The message to log.
            **kwargs: Additional context information to include in the log.
        """
        self._log("INFO", msg, **kwargs)

    def warning(self, msg: Any, **kwargs: Any) -> None:
        """Log a warning message.

        Args:
            msg (Any): The message to log.
            **kwargs: Additional context information to include in the log.
        """
        self._log("WARNING", msg, **kwargs)

    def error(self, msg: Any, **kwargs: Any) -> None:
        """Log an error message.

        Args:
            msg (Any): The message to log.
            **kwargs: Additional context information to include in the log.
        """
        self._log("ERROR", msg, **kwargs)

    def critical(self, msg: Any, **kwargs: Any) -> None:
        """Log a critical message.

        Args:
            msg (Any): The message to log.
            **kwargs: Additional context information to include in the log.
        """
        self._log("CRITICAL", msg, **kwargs)

    def success(self, msg: Any, **kwargs: Any) -> None:
        """Log a success message.

        Args:
            msg (Any): The message to log.
            **kwargs: Additional context information to include in the log.
        """
        self._log("SUCCESS", msg, **kwargs)

    def _log(self, level: str, msg: Any, **kwargs: Any) -> None:
        """Internal method to log messages.

        Args:
            level (str): The logging level.
            msg (Any): The message to log.
            **kwargs: Additional context information to include in the log.
        """
        # Check if the log should be recorded based on the sampling rate
        event_key = kwargs.get("event", "default")
        if not self._manager.should_log(event_key):
            return

        # Prepare the log record
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "message": str(msg),
        }

        # Add additional context information
        if kwargs:
            record.update(kwargs)

        # Add tracing context
        trace_context = self._manager.get_trace_context()
        if trace_context:
            record["trace_id"] = trace_context

        # Detect anomalies
        anomalies = self._manager.detect_anomalies(record)
        if anomalies:
            record["anomalies"] = anomalies

        # Send to the analyzer
        self._manager.add_log_for_analysis(record)

        # Record the log
        if level == "SUCCESS":
            self._logger.log(SUCCESS_LEVEL, record)
        else:
            level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            self._logger.log(level_map.get(level, logging.INFO), record)

    def get_analysis(self) -> LogAnalysisResult:
        """Get the result of log analysis.

        Returns:
            LogAnalysisResult: The result of the log analysis.
        """
        return self._manager.get_log_analysis()

    def add_analysis_callback(self, callback: Callable[[LogAnalysisResult], None]):
        """Add a callback function to be called after log analysis.

        Args:
            callback (Callable[[LogAnalysisResult], None]): The callback function.
        """
        self._manager.add_analysis_callback(callback)

    def get_security_report(self) -> List[Dict[str, Any]]:
        """Get the security audit report.

        Returns:
            List[Dict[str, Any]]: A list of audit records.
        """
        return self._manager.get_security_audit_report()

    def alert(self, msg: Any, **kwargs: Any) -> None:
        """Log an alert message.

        Args:
            msg (Any): The alert message to log.
            **kwargs: Additional context information to include in the log.
        """
        kwargs["level"] = "alert"
        self._log("ERROR", f"ALERT: {msg}", **kwargs)

    def close(self):
        """Close the logger and stop real-time monitoring."""
        self._manager.stop_real_time_monitoring()

# --- helper: control external library loggers via our utils API ---
def set_external_logger_level(logger_name: str, level: str = "ERROR") -> None:
    """Set level for an external logger (wrapper to avoid importing logging in callers).

    Args:
        logger_name: name of the external logger, e.g. "modelscope".
        level: one of DEBUG/INFO/WARNING/ERROR/CRITICAL.
    """
    try:
        import logging as _logging
        level_map = {
            "DEBUG": _logging.DEBUG,
            "INFO": _logging.INFO,
            "WARNING": _logging.WARNING,
            "ERROR": _logging.ERROR,
            "CRITICAL": _logging.CRITICAL,
        }
        _logging.getLogger(logger_name).setLevel(level_map.get(level.upper(), _logging.ERROR))
    except Exception:
        pass