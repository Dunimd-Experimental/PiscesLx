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

import logging
from typing import Any, Dict
from datetime import datetime

# Set up logging for the module
logger = logging.getLogger(__name__)

def build_device_report_payload(service: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a device report payload by augmenting the provided data with service-related information.
    
    Args:
        service (Any): Service object containing device health, metrics, and state information.
        data (Dict[str, Any]): Initial data dictionary to be augmented.
    
    Returns:
        Dict[str, Any]: Augmented payload dictionary with additional device-related information.
        If an error occurs during the process, the original data dictionary is returned.
    """
    try:
        # Create a copy of the input data to prevent modifying the original
        payload = dict(data)
        
        # Retrieve the current health snapshot of the service
        health = {}
        try:
            if hasattr(service, 'get_health_snapshot'):
                health = service.get_health_snapshot()
        except Exception as e:
            logger.warning(f"Failed to get health snapshot: {e}")
        
        # Retrieve anomaly insights
        anomalies = []
        try:
            if hasattr(service, 'metrics_collector') and hasattr(service.metrics_collector, 'get_anomaly_insights'):
                anomaly_insights = service.metrics_collector.get_anomaly_insights()
                anomalies = [vars(a) if hasattr(a, '__dict__') else str(a) for a in anomaly_insights]
        except Exception as e:
            logger.warning(f"Failed to get anomaly insights: {e}")
        
        # Retrieve the service state history
        history = []
        try:
            history = list(getattr(service, "state_history", []))
        except Exception as e:
            logger.warning(f"Failed to get state history: {e}")
        
        # Retrieve environment information
        environment = {}
        try:
            environment = getattr(service, "env_info", {})
        except Exception as e:
            logger.warning(f"Failed to get environment info: {e}")

        # Update the payload with the collected information
        payload.update({
            "health_snapshot": health,
            "anomaly_detection": anomalies,
            "service_state_history": history,
            "environment": environment,
        })
        
        return payload
    except Exception as e:
        logger.error(f"Failed to build device report payload: {e}")
        return data

def build_session_report_payload(service: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a session report payload by augmenting the provided data with session-related information.
    
    Args:
        service (Any): Service object containing session start time, metrics, and state information.
        data (Dict[str, Any]): Initial data dictionary to be augmented.
    
    Returns:
        Dict[str, Any]: Augmented payload dictionary with additional session-related information.
        If an error occurs during the process, the original data dictionary is returned.
    """
    try:
        # Create a copy of the input data to prevent modifying the original
        payload = dict(data)
        
        # Calculate the session duration
        duration = 0.0
        try:
            started_at = getattr(service, "started_at", None)
            if started_at:
                # Handle different datetime formats
                try:
                    start_dt = datetime.fromisoformat(started_at.replace("Z", ""))
                    duration = (datetime.utcnow() - start_dt).total_seconds()
                except ValueError:
                    logger.warning(f"Invalid started_at format: {started_at}")
                    duration = 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate session duration: {e}")
            duration = 0.0
        
        # Get the count of detected anomalies
        anomalies_cnt = 0
        try:
            if hasattr(service, 'metrics_collector') and hasattr(service.metrics_collector, 'get_anomaly_insights'):
                anomaly_insights = service.metrics_collector.get_anomaly_insights()
                anomalies_cnt = len(anomaly_insights) if anomaly_insights else 0
        except Exception as e:
            logger.warning(f"Failed to get anomaly count: {e}")
        
        # Get the number of state transitions
        state_transitions = 0
        try:
            state_history = getattr(service, "state_history", [])
            state_transitions = len(state_history) if state_history else 0
        except Exception as e:
            logger.warning(f"Failed to get state history: {e}")
        
        # Calculate the resource efficiency
        resource_efficiency = 0.0
        try:
            if hasattr(service, '_calculate_resource_efficiency'):
                resource_efficiency = service._calculate_resource_efficiency()
            else:
                logger.warning("Service does not have _calculate_resource_efficiency method")
        except Exception as e:
            logger.warning(f"Failed to calculate resource efficiency: {e}")
        
        # Get the final health snapshot
        final_health = {}
        try:
            if hasattr(service, 'get_health_snapshot'):
                final_health = service.get_health_snapshot()
        except Exception as e:
            logger.warning(f"Failed to get final health snapshot: {e}")

        # Update the payload with the session summary and final health snapshot
        payload.update({
            "session_summary": {
                "duration_seconds": duration,
                "service_state_transitions": state_transitions,
                "anomalies_detected": anomalies_cnt,
                "resource_efficiency": resource_efficiency,
            },
            "final_health_snapshot": final_health,
        })
        
        return payload
    except Exception as e:
        logger.error(f"Failed to build session report payload: {e}")
        return data
