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
import json
import yaml
import uuid
import hashlib
from utils.log.core import PiscesLxCoreLog
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from utils import PiscesLxCoreConfigManagerFacade
from utils import PiscesLxCoreDeviceFacade
from utils import PiscesLxCoreObservabilityFacade
from utils import PiscesLxCoreMetricsRegistry
from utils import PiscesLxCoreCacheManagerFacade

class FunctionTemplateManager:
    """Manager for function templates with comprehensive utils integration."""
    
    def __init__(self, base_dir: str = None, config_manager=None, device_manager=None, 
                 observability=None, metrics_registry=None):
        """Initialize the template manager with comprehensive utils support.

        Args:
            base_dir (str): Base directory for templates. If None, uses cache manager for flexible storage.
            config_manager: Optional config manager instance.
            device_manager: Optional device manager instance.
            observability: Optional observability facade.
            metrics_registry: Optional metrics registry.
        """
        # Initialize utils components
        self.config_manager = config_manager or PiscesLxCoreConfigManagerFacade()
        # Use cache manager to get a flexible directory for template storage
        cache_manager = PiscesLxCoreCacheManagerFacade.get_instance()
        
        if base_dir is None:
            self.base_dir = Path(cache_manager.get_cache_dir("func_templates"))
        else:
            self.base_dir = Path(base_dir)
        
        # Directory to store current templates
        self.templates_dir = self.base_dir / "templates"
        # Directory to store template versions
        self.versions_dir = self.base_dir / "versions"
        
        # Create necessary directories if they don't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        self.device_manager = device_manager
        self.observability = observability
        self.metrics_registry = metrics_registry
        self.templates = {}
        self.logger = PiscesLxCoreLog(__name__)
        
        # Collect initialization metrics
        if self.observability:
            init_metrics = {
                "cache_dir_size": self._get_cache_size(),
                "available_templates": len(self.list_templates()),
                "device_info": self.device_manager.get_device_info() if self.device_manager else {}
            }
            self.observability.record_event("func_templates.init", init_metrics)
        
        self.logger.info(f"FunctionTemplateManager initialized with comprehensive utils support")
    
    def _get_cache_size(self) -> int:
        """Get cache directory size in bytes."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.base_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size
        except OSError:
            return 0
    
    def _validate_template_integrity(self, template: dict) -> bool:
        """Validate template integrity and compliance."""
        required_fields = ["name", "version", "parameters", "return_type"]
        return all(field in template for field in required_fields)
    
    def _optimize_template_storage(self, template: dict) -> dict:
        """Optimize template for storage based on device capabilities."""
        if not self.device_manager:
            return template
        
        device_info = self.device_manager.get_device_info()
        
        # Optimize based on available memory
        if device_info.get('memory_available_mb', 4096) < 2048:  # Less than 2GB
            # Remove large metadata for memory-constrained devices
            optimized = template.copy()
            optimized.pop('detailed_description', None)
            optimized.pop('examples', None)
            return optimized
        
        return template
    
    def save_template(self, name: str, function_code: str, category: str = "default", 
                     description: str = "", tags: List[str] = None) -> str:
        """Save a new function template with comprehensive metadata and utils integration.

        Args:
            name (str): Name of the function template.
            function_code (str): Source code of the function.
            category (str, optional): Category of the template. Defaults to "default".
            description (str, optional): Description of the template. Defaults to "".
            tags (List[str], optional): Tags associated with the template. Defaults to None.

        Returns:
            str: ID of the newly saved template.
        """
        import time
        start_time = time.time()
        
        if tags is None:
            tags = []
            
        template_id = str(uuid.uuid4())
        template_data = {
            "id": template_id,
            "name": name,
            "function_code": function_code,
            "category": category,
            "description": description,
            "tags": tags,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": 1,
            "size": len(function_code),
            "hash": __import__('hashlib').sha256(function_code.encode()).hexdigest()[:16],
            "parameters": [],  # Will be extracted from code
            "return_type": "unknown",  # Will be extracted from code
            "device_optimized": False
        }
        
        # Validate template before saving
        if not self._validate_template_integrity(template_data):
            self.logger.error(f"Template {name} failed integrity validation, not saving")
            return None
        
        # Optimize for storage
        optimized_template = self._optimize_template_storage(template_data)
        
        template_file = self.templates_dir / f"{template_id}.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_template, f, ensure_ascii=False, indent=2)
        
        # Create the initial version of the template
        self._save_version(template_id, optimized_template)
        
        save_time = time.time() - start_time
        
        # Record metrics
        if self.metrics_registry:
            self.metrics_registry.record_metric("func_templates.saved", 1, {"template": name})
            self.metrics_registry.record_metric("func_templates.save_time", save_time, {"template": name})
        
        # Record observability event
        if self.observability:
            self.observability.record_event("func_templates.saved", {
                "template_name": name,
                "save_time": save_time,
                "optimized": optimized_template != template_data,
                "cache_size": self._get_cache_size()
            })
        
        self.logger.info(f"Saved template: {name} (optimized: {optimized_template != template_data})")
        return template_id
    
    def update_template(self, template_id: str, name: str = None, function_code: str = None,
                     category: str = None, description: str = None, tags: List[str] = None) -> bool:
        """Update an existing template and create a new version.

        Args:
            template_id (str): ID of the template to update.
            name (str, optional): New name of the template. Defaults to None.
            function_code (str, optional): New function code. Defaults to None.
            category (str, optional): New category. Defaults to None.
            description (str, optional): New description. Defaults to None.
            tags (List[str], optional): New tags. Defaults to None.

        Returns:
            bool: True if the update is successful, False otherwise.
        """
        template_file = self.templates_dir / f"{template_id}.json"
        if not template_file.exists():
            return False
        
        with open(template_file, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        
        # Update template fields if new values are provided
        if name is not None:
            template_data["name"] = name
        if function_code is not None:
            template_data["function_code"] = function_code
        if category is not None:
            template_data["category"] = category
        if description is not None:
            template_data["description"] = description
        if tags is not None:
            template_data["tags"] = tags
        
        template_data["updated_at"] = datetime.now().isoformat()
        template_data["version"] += 1
        
        # Save the updated template data
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)
        
        # Save the updated template as a new version
        self._save_version(template_id, template_data)
        
        return True
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a template by ID with utils integration.

        Args:
            template_id (str): ID of the template to retrieve.

        Returns:
            Optional[Dict[str, Any]]: Template data if found, None otherwise.
        """
        import time
        start_time = time.time()
        
        try:
            template_file = self.templates_dir / f"{template_id}.json"
            if not template_file.exists():
                if self.metrics_registry:
                    self.metrics_registry.record_metric("func_templates.not_found", 1, {"template": template_id})
                return None
            
            with open(template_file, 'r', encoding='utf-8') as f:
                template = json.load(f)
            
            load_time = time.time() - start_time
            
            # Record metrics
            if self.metrics_registry:
                self.metrics_registry.record_metric("func_templates.loaded", 1, {"template": template_id})
                self.metrics_registry.record_metric("func_templates.load_time", load_time, {"template": template_id})
            
            # Record observability event
            if self.observability:
                self.observability.record_event("func_templates.loaded", {
                    "template_id": template_id,
                    "load_time": load_time,
                    "template_size": len(json.dumps(template))
                })
            
            return template
            
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load template {template_id}: {e}")
            if self.metrics_registry:
                self.metrics_registry.record_metric("func_templates.load_error", 1, {"template": template_id, "error_type": type(e).__name__})
            return None
    
    def list_templates(self, category: str = None) -> List[Dict[str, Any]]:
        """List all templates with utils integration, optionally filtered by category.

        Args:
            category (str, optional): Category to filter templates. Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of template data.
        """
        import time
        start_time = time.time()
        templates = []
        
        try:
            for template_file in self.templates_dir.glob("*.json"):
                try:
                    with open(template_file, 'r', encoding='utf-8') as f:
                        template = json.load(f)
                        if category is None or template.get("category") == category:
                            templates.append(template)
                except (json.JSONDecodeError, IOError) as e:
                    self.logger.error(f"Failed to load template file {template_file}: {e}")
            
            list_time = time.time() - start_time
            
            # Record metrics
            if self.metrics_registry:
                self.metrics_registry.record_metric("func_templates.listed", 1, {"category": category or "all"})
                self.metrics_registry.record_metric("func_templates.list_time", list_time, {"category": category or "all"})
                self.metrics_registry.record_metric("func_templates.total_count", len(templates))
            
            # Record observability event
            if self.observability:
                self.observability.record_event("func_templates.listed", {
                    "template_count": len(templates),
                    "list_time": list_time,
                    "category": category or "all",
                    "cache_size": self._get_cache_size()
                })
            
            # Sort templates by the updated time in descending order
            templates.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return templates
            
        except OSError as e:
            self.logger.error(f"Failed to list templates: {e}")
            if self.metrics_registry:
                self.metrics_registry.record_metric("func_templates.list_error", 1, {"category": category or "all", "error_type": type(e).__name__})
            return []
    
    def delete_template(self, template_id: str) -> bool:
        """Delete a template and all its versions.

        Args:
            template_id (str): ID of the template to delete.

        Returns:
            bool: True if the deletion is successful.
        """
        template_file = self.templates_dir / f"{template_id}.json"
        if template_file.exists():
            template_file.unlink()
        
        # Delete all version files of the template
        version_dir = self.versions_dir / template_id
        if version_dir.exists():
            for version_file in version_dir.glob("*.json"):
                version_file.unlink()
            version_dir.rmdir()
        
        return True
    
    def get_categories(self) -> List[str]:
        """Get all unique categories.

        Returns:
            List[str]: List of unique categories.
        """
        categories = set()
        for template in self.list_templates():
            categories.add(template.get("category", "default"))
        return sorted(list(categories))
    
    def _save_version(self, template_id: str, template_data: Dict[str, Any]):
        """Save a version of the template.

        Args:
            template_id (str): ID of the template.
            template_data (Dict[str, Any]): Template data to save.
        """
        version_dir = self.versions_dir / template_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        version_file = version_dir / f"v{template_data['version']}.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)
    
    def list_versions(self, template_id: str) -> List[Dict[str, Any]]:
        """List all versions of a template.

        Args:
            template_id (str): ID of the template.

        Returns:
            List[Dict[str, Any]]: List of version data.
        """
        version_dir = self.versions_dir / template_id
        if not version_dir.exists():
            return []
        
        versions = []
        for version_file in version_dir.glob("*.json"):
            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    versions.append(json.load(f))
            except (json.JSONDecodeError, IOError):
                continue
        
        # Sort versions by version number in descending order
        versions.sort(key=lambda x: x.get("version", 0), reverse=True)
        return versions
    
    def rollback_to_version(self, template_id: str, version: int) -> bool:
        """Rollback template to a specific version.

        Args:
            template_id (str): ID of the template.
            version (int): Version number to rollback to.

        Returns:
            bool: True if the rollback is successful, False otherwise.
        """
        version_file = self.versions_dir / template_id / f"v{version}.json"
        if not version_file.exists():
            return False
        
        with open(version_file, 'r', encoding='utf-8') as f:
            version_data = json.load(f)
        
        # Update the current template with the specified version data
        template_file = self.templates_dir / f"{template_id}.json"
        version_data["updated_at"] = datetime.now().isoformat()
        version_data["version"] += 1  # Increment version number after rollback
        
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(version_data, f, ensure_ascii=False, indent=2)
        
        # Save the rolled-back template as a new version
        self._save_version(template_id, version_data)
        
        return True
    
    def export_template(self, template_id: str) -> Optional[str]:
        """Export template as JSON string.

        Args:
            template_id (str): ID of the template to export.

        Returns:
            Optional[str]: JSON string of the template data without internal fields, None if template not found.
        """
        template = self.get_template(template_id)
        if not template:
            return None
        
        # Remove internal fields that should not be exported
        export_data = template.copy()
        export_data.pop("id", None)
        export_data.pop("created_at", None)
        export_data.pop("updated_at", None)
        export_data.pop("version", None)
        
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def import_template(self, json_str: str, new_name: str = None) -> Optional[str]:
        """Import template from JSON string.

        Args:
            json_str (str): JSON string containing template data.
            new_name (str, optional): New name for the imported template. Defaults to None.

        Returns:
            Optional[str]: ID of the imported template, None if import fails.
        """
        try:
            template_data = json.loads(json_str)
            
            # Use the provided name or the original name in the template data
            name = new_name or template_data.get("name", "Imported Template")
            
            return self.save_template(
                name=name,
                function_code=template_data.get("function_code", ""),
                category=template_data.get("category", "imported"),
                description=template_data.get("description", ""),
                tags=template_data.get("tags", [])
            )
        except (json.JSONDecodeError, KeyError):
            return None
    
    def export_all_templates(self) -> str:
        """Export all templates as JSON.

        Returns:
            str: JSON string containing all templates and export time.
        """
        templates = self.list_templates()
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "templates": templates
        }
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def import_multiple_templates(self, json_str: str) -> List[str]:
        """Import multiple templates from JSON.

        Args:
            json_str (str): JSON string containing multiple templates.

        Returns:
            List[str]: List of IDs of the imported templates.
        """
        try:
            data = json.loads(json_str)
            templates = data.get("templates", [])
            
            imported_ids = []
            for template_data in templates:
                name = template_data.get("name", "Imported Template")
                template_id = self.save_template(
                    name=name,
                    function_code=template_data.get("function_code", ""),
                    category=template_data.get("category", "imported"),
                    description=template_data.get("description", ""),
                    tags=template_data.get("tags", [])
                )
                imported_ids.append(template_id)
            
            return imported_ids
        except (json.JSONDecodeError, KeyError):
            return []