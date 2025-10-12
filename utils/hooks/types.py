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

import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from utils.log.core import PiscesLxCoreLog
logger = PiscesLxCoreLog("PiscesLx.Utils.Hooks.Types")
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable

@dataclass
class PiscesLxCoreEventMetrics:
    """
    Data class for storing event metrics.
    
    Attributes:
        event_type (str): The type of the event.
        count (int): The number of event executions, default is 0.
        total_time (float): The total execution time of the event, default is 0.0.
        errors (int): The number of errors during event execution, default is 0.
        last_executed (Optional[float]): The timestamp of the last execution, default is None.
    """
    event_type: str
    count: int = 0
    total_time: float = 0.0
    errors: int = 0
    last_executed: Optional[float] = None
    
    @property
    def average_time(self) -> float:
        """
        Calculate the average execution time of the event.
        
        Returns:
            float: The average execution time, ensuring the denominator is at least 1 to avoid division by zero.
        """
        return self.total_time / max(self.count, 1)
    
    @property
    def error_rate(self) -> float:
        """
        Calculate the error rate of the event.
        
        Returns:
            float: The error rate, ensuring the denominator is at least 1 to avoid division by zero.
        """
        return self.errors / max(self.count, 1)

@dataclass
class PiscesLxCoreExecutionResult:
    """
    Data class for storing the execution result of an event.
    
    Attributes:
        event_type (str): The type of the event.
        execution_time (float): The execution time of the event.
        executed (int): The number of successful executions.
        errors (int): The number of errors during execution.
        result (Any): The result of the execution, default is None.
        exception (Optional[str]): The exception information if an error occurs, default is None.
    """
    event_type: str
    execution_time: float
    executed: int
    errors: int
    result: Any = None
    exception: Optional[str] = None

class PiscesLxCoreAlgorithmicListener(ABC):
    """
    Abstract base class for algorithm listeners.
    """
    
    def __init__(
        self,
        priority: int = 0,
        once: bool = False,
        max_executions: Optional[int] = None,
        event_types: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the algorithm listener.
        
        Args:
            priority (int): The priority of the listener, default is 0.
            once (bool): Whether the listener should execute only once, default is False.
            max_executions (Optional[int]): The maximum number of executions, default is None.
            event_types (Optional[List[str]]): The list of event types to listen for, default is None.
        """
        self.priority = priority
        self.once = once
        self.max_executions = max_executions
        self.event_types = event_types or []
        self._executions = 0
        self._created_at = time.time()
        self._last_executed: Optional[float] = None
    
    @property
    @abstractmethod
    def callback(self) -> Union[Callable[..., Any], Callable[..., Awaitable[Any]]]:
        """
        Get the callback function of the listener.
        
        Returns:
            Union[Callable[..., Any], Callable[..., Awaitable[Any]]]: The callback function.
        """
        pass
    
    def should_execute(self) -> bool:
        """
        Determine whether the listener should be executed.
        
        Returns:
            bool: True if the listener should be executed, False otherwise.
        """
        if self.once and self._executions > 0:
            return False
        
        if self.max_executions is not None and self._executions >= self.max_executions:
            return False
        
        return True
    
    def executed(self) -> None:
        """
        Mark the listener as executed, updating the execution count and last execution timestamp.
        """
        self._executions += 1
        self._last_executed = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get the statistical information of the listener.
        
        Returns:
            Dict[str, Any]: A dictionary containing statistical information.
        """
        return {
            "executions": self._executions,
            "created_at": self._created_at,
            "last_executed": self._last_executed,
            "should_execute": self.should_execute(),
            "priority": self.priority,
            "once": self.once,
            "max_executions": self.max_executions,
            "event_types": self.event_types
        }

class PiscesLxCoreFunctionListener(PiscesLxCoreAlgorithmicListener):
    """
    Listener class for function events.
    """
    
    def __init__(
        self,
        callback: Union[Callable[..., Any], Callable[..., Awaitable[Any]]],
        priority: int = 0,
        once: bool = False,
        max_executions: Optional[int] = None,
        event_types: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the function listener.
        
        Args:
            callback (Union[Callable[..., Any], Callable[..., Awaitable[Any]]]): The callback function.
            priority (int): The priority of the listener, default is 0.
            once (bool): Whether the listener should execute only once, default is False.
            max_executions (Optional[int]): The maximum number of executions, default is None.
            event_types (Optional[List[str]]): The list of event types to listen for, default is None.
        """
        super().__init__(priority, once, max_executions, event_types)
        self._callback = callback
    
    @property
    def callback(self) -> Union[Callable[..., Any], Callable[..., Awaitable[Any]]]:
        """
        Get the callback function of the listener.
        
        Returns:
            Union[Callable[..., Any], Callable[..., Awaitable[Any]]]: The callback function.
        """
        return self._callback
