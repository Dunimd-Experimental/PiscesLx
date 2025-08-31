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
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from .simple_mcp import register_tool

logger = logging.getLogger(__name__)

class MemoryTool:
    """Knowledge graph memory management tool
    
    This class provides functionality for managing a knowledge graph memory, 
    including operations such as creating entities and relations, adding observations,
    searching, reading, and deleting data.
    """
    
    def __init__(self):
        """Initialize the MemoryTool instance.
        
        Set up the tool's name, description, and memory file path.
        Ensure the memory file exists and initialize it if necessary.
        """
        self.name = "memory"
        self.description = "Knowledge graph memory for storing and retrieving information"
        self.memory_file = Path(os.getenv('MCP_MEMORY_FILE', 'memory.json')).expanduser()
        
        # Create the memory file if it doesn't exist
        if not self.memory_file.exists():
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_memory({"entities": [], "relations": []})
    
    def _load_memory(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load memory data from the memory file.
        
        Returns:
            A dictionary containing two lists: 'entities' and 'relations'.
            If the file is not found or corrupted, return an empty memory structure.
        """
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"entities": [], "relations": []}
    
    def _save_memory(self, memory: Dict[str, List[Dict[str, Any]]]):
        """Save memory data to the memory file.
        
        Args:
            memory: A dictionary containing the memory data to be saved.
        """
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema for memory operations.
        
        Returns:
            A dictionary defining the schema for memory operations, including 
            available operations and their parameters.
        """
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Memory operation to perform",
                    "enum": [
                        "create_entity", "create_relation", "add_observation",
                        "search", "read", "delete_entity", "delete_relation",
                        "delete_observation"
                    ]
                },
                "entity_name": {
                    "type": "string",
                    "description": "Name of the entity"
                },
                "entity_type": {
                    "type": "string",
                    "description": "Type/category of the entity"
                },
                "from_entity": {
                    "type": "string",
                    "description": "Source entity name (for relations)"
                },
                "to_entity": {
                    "type": "string", 
                    "description": "Target entity name (for relations)"
                },
                "relation_type": {
                    "type": "string",
                    "description": "Type of relationship"
                },
                "observation": {
                    "type": "string",
                    "description": "Observation to add to entity"
                },
                "observations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of observations to add/delete"
                },
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "entity_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entity names"
                }
            },
            "required": ["operation"]
        }
    
    def _create_entity(self, entity_name: str, entity_type: str) -> Dict[str, Any]:
        """Create a new entity in the memory.
        
        Args:
            entity_name: The name of the entity to create.
            entity_type: The type of the entity.
            
        Returns:
            A dictionary indicating the success status and relevant data or error message.
        """
        memory = self._load_memory()
        
        # Check if the entity already exists
        if any(e["name"] == entity_name for e in memory["entities"]):
            return {
                "success": False,
                "error": f"Entity '{entity_name}' already exists"
            }
        
        entity = {
            "name": entity_name,
            "entityType": entity_type,
            "observations": [],
            "created": datetime.now().isoformat()
        }
        
        memory["entities"].append(entity)
        self._save_memory(memory)
        
        return {
            "success": True,
            "data": {"entity": entity, "created": True}
        }
    
    def _create_relation(self, from_entity: str, to_entity: str, relation_type: str) -> Dict[str, Any]:
        """Create a new relationship between two entities in the memory.
        
        Args:
            from_entity: The name of the source entity.
            to_entity: The name of the target entity.
            relation_type: The type of the relationship.
            
        Returns:
            A dictionary indicating the success status and relevant data or error message.
        """
        memory = self._load_memory()
        
        # Verify that both entities exist
        entities = {e["name"] for e in memory["entities"]}
        if from_entity not in entities or to_entity not in entities:
            return {
                "success": False,
                "error": "One or both entities do not exist"
            }
        
        relation = {
            "from": from_entity,
            "to": to_entity,
            "relationType": relation_type,
            "created": datetime.now().isoformat()
        }
        
        memory["relations"].append(relation)
        self._save_memory(memory)
        
        return {
            "success": True,
            "data": {"relation": relation, "created": True}
        }
    
    def _add_observation(self, entity_name: str, observation: str) -> Dict[str, Any]:
        """Add an observation to an existing entity in the memory.
        
        Args:
            entity_name: The name of the entity to add the observation to.
            observation: The observation to add.
            
        Returns:
            A dictionary indicating the success status and relevant data or error message.
        """
        memory = self._load_memory()
        
        entity = next((e for e in memory["entities"] if e["name"] == entity_name), None)
        if not entity:
            return {
                "success": False,
                "error": f"Entity '{entity_name}' not found"
            }
        
        if observation not in entity["observations"]:
            entity["observations"].append(observation)
            self._save_memory(memory)
        
        return {
            "success": True,
            "data": {"entity": entity, "observation_added": True}
        }
    
    def _search_memory(self, query: str) -> Dict[str, Any]:
        """Search the memory for entities and relations matching the query.
        
        Args:
            query: The search query string.
            
        Returns:
            A dictionary indicating the success status and the search results.
        """
        memory = self._load_memory()
        query_lower = query.lower()
        
        # Find entities that match the query
        matching_entities = []
        for entity in memory["entities"]:
            if (query_lower in entity["name"].lower() or
                query_lower in entity["entityType"].lower() or
                any(query_lower in obs.lower() for obs in entity["observations"])):
                matching_entities.append(entity)
        
        # Filter relations to those involving matching entities
        entity_names = {e["name"] for e in matching_entities}
        matching_relations = [
            r for r in memory["relations"]
            if r["from"] in entity_names and r["to"] in entity_names
        ]
        
        return {
            "success": True,
            "data": {
                "entities": matching_entities,
                "relations": matching_relations,
                "query": query,
                "total_entities": len(matching_entities),
                "total_relations": len(matching_relations)
            }
        }
    
    def _read_memory(self, entity_names: List[str] = None) -> Dict[str, Any]:
        """Read specific entities and their related relations or the entire memory.
        
        Args:
            entity_names: A list of entity names to read. If None, read the entire memory.
            
        Returns:
            A dictionary indicating the success status and the read data.
        """
        memory = self._load_memory()
        
        if entity_names:
            entities = [e for e in memory["entities"] if e["name"] in entity_names]
            entity_names_set = {e["name"] for e in entities}
            relations = [
                r for r in memory["relations"]
                if r["from"] in entity_names_set and r["to"] in entity_names_set
            ]
        else:
            entities = memory["entities"]
            relations = memory["relations"]
        
        return {
            "success": True,
            "data": {
                "entities": entities,
                "relations": relations,
                "total_entities": len(entities),
                "total_relations": len(relations)
            }
        }
    
    def _delete_entity(self, entity_name: str) -> Dict[str, Any]:
        """Delete an entity and all its related relations from the memory.
        
        Args:
            entity_name: The name of the entity to delete.
            
        Returns:
            A dictionary indicating the success status and deletion results.
        """
        memory = self._load_memory()
        
        # Remove the entity
        original_entity_count = len(memory["entities"])
        memory["entities"] = [e for e in memory["entities"] if e["name"] != entity_name]
        
        # Remove all relations involving the deleted entity
        original_relation_count = len(memory["relations"])
        memory["relations"] = [
            r for r in memory["relations"]
            if r["from"] != entity_name and r["to"] != entity_name
        ]
        
        self._save_memory(memory)
        
        return {
            "success": True,
            "data": {
                "entity_deleted": original_entity_count > len(memory["entities"]),
                "relations_deleted": original_relation_count > len(memory["relations"]),
                "entity_name": entity_name
            }
        }
    
    def _delete_observation(self, entity_name: str, observation: str) -> Dict[str, Any]:
        """Delete an observation from an existing entity in the memory.
        
        Args:
            entity_name: The name of the entity to delete the observation from.
            observation: The observation to delete.
            
        Returns:
            A dictionary indicating the success status and relevant data or error message.
        """
        memory = self._load_memory()
        
        entity = next((e for e in memory["entities"] if e["name"] == entity_name), None)
        if not entity:
            return {
                "success": False,
                "error": f"Entity '{entity_name}' not found"
            }
        
        original_obs_count = len(entity["observations"])
        entity["observations"] = [obs for obs in entity["observations"] if obs != observation]
        
        if original_obs_count > len(entity["observations"]):
            self._save_memory(memory)
            return {
                "success": True,
                "data": {"observation_deleted": True, "entity": entity}
            }
        else:
            return {
                "success": False,
                "error": f"Observation not found in entity '{entity_name}'"
            }
    
    def execute(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute a memory operation based on the provided operation type and parameters.
        
        Args:
            operation: The type of memory operation to perform.
            **kwargs: Additional parameters required by the operation.
            
        Returns:
            A dictionary indicating the success status and operation results or error message.
        """
        if operation == "create_entity":
            entity_name = kwargs.get('entity_name')
            entity_type = kwargs.get('entity_type', 'general')
            if not entity_name:
                return {"success": False, "error": "entity_name is required"}
            return self._create_entity(entity_name, entity_type)
            
        elif operation == "create_relation":
            from_entity = kwargs.get('from_entity')
            to_entity = kwargs.get('to_entity')
            relation_type = kwargs.get('relation_type')
            if not all([from_entity, to_entity, relation_type]):
                return {"success": False, "error": "from_entity, to_entity, and relation_type are required"}
            return self._create_relation(from_entity, to_entity, relation_type)
            
        elif operation == "add_observation":
            entity_name = kwargs.get('entity_name')
            observation = kwargs.get('observation')
            if not all([entity_name, observation]):
                return {"success": False, "error": "entity_name and observation are required"}
            return self._add_observation(entity_name, observation)
            
        elif operation == "search":
            query = kwargs.get('query', '')
            if not query:
                return {"success": False, "error": "query is required for search"}
            return self._search_memory(query)
            
        elif operation == "read":
            entity_names = kwargs.get('entity_names')
            return self._read_memory(entity_names)
            
        elif operation == "delete_entity":
            entity_name = kwargs.get('entity_name')
            if not entity_name:
                return {"success": False, "error": "entity_name is required"}
            return self._delete_entity(entity_name)
            
        elif operation == "delete_observation":
            entity_name = kwargs.get('entity_name')
            observation = kwargs.get('observation')
            if not all([entity_name, observation]):
                return {"success": False, "error": "entity_name and observation are required"}
            return self._delete_observation(entity_name, observation)
            
        else:
            return {"success": False, "error": f"Unsupported operation: {operation}"}

# Integrate with Pisces L1 MCP Square
from . import register_custom_tool

# Register the memory tool to the MCP Square
register_custom_tool(
    name="memory",
    description="Knowledge graph memory management for storing and retrieving information",
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "description": "Type of memory operation",
                "enum": [
                    "create_entity", "create_relation", "add_observation",
                    "search", "read", "delete_entity", "delete_relation",
                    "delete_observation"
                ]
            },
            "entity_name": {
                "type": "string",
                "description": "Name of the entity"
            },
            "entity_type": {
                "type": "string",
                "description": "Type of the entity"
            },
            "from_entity": {
                "type": "string",
                "description": "Source entity of the relation"
            },
            "to_entity": {
                "type": "string",
                "description": "Target entity of the relation"
            },
            "relation_type": {
                "type": "string",
                "description": "Type of the relation"
            },
            "observation": {
                "type": "string",
                "description": "Observation record"
            },
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "entity_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of entity names"
            }
        },
        "required": ["operation"]
    },
    function=MemoryTool().execute,
    category="Memory"
)