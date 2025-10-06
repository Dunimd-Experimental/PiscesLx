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

import os
import json
from typing import Dict, Any

# Grafana dashboard configuration
GRAFANA_DASHBOARD_CONFIG = {
    "llm_observability": "grafana_llm_observability.json",
    "system_monitoring": "grafana_system_monitoring.json",
    "performance_analysis": "grafana_performance_analysis.json"
}

def load_dashboard_config(dashboard_name: str) -> Dict[str, Any]:
    """
    Load the dashboard configuration file.

    Args:
        dashboard_name (str): The name of the dashboard.

    Returns:
        Dict[str, Any]: The dashboard configuration dictionary.

    Raises:
        ValueError: If the dashboard name is unknown.
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the configuration file format is invalid.
    """
    if dashboard_name not in GRAFANA_DASHBOARD_CONFIG:
        raise ValueError(f"Unknown dashboard name: {dashboard_name}")
    
    config_file = GRAFANA_DASHBOARD_CONFIG[dashboard_name]
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    
    from utils.config.loader import load_config_from_file
    return load_config_from_file(config_path)

def get_available_dashboards() -> Dict[str, str]:
    """
    Get a list of available dashboards.

    Returns:
        Dict[str, str]: A mapping from dashboard names to file names.
    """
    return GRAFANA_DASHBOARD_CONFIG.copy()

def validate_dashboard_config(config: Dict[str, Any]) -> bool:
    """
    Validate the effectiveness of the dashboard configuration.

    Args:
        config (Dict[str, Any]): The dashboard configuration dictionary.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    required_fields = ["title", "panels"]
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            return False
    
    # Check panel configuration
    if not isinstance(config["panels"], list):
        return False
    
    for panel in config["panels"]:
        if not isinstance(panel, dict):
            return False
        if "type" not in panel or "title" not in panel:
            return False
    
    return True

# Intentionally do not re-export anything at subpackage level to enforce single entry via utils.__init__.