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
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import PiscesLxCoreCacheManagerFacade

class JSONLCreator:
    """
    A specialized creator for JSONL files designed for managing the PiscesL1 dataset.
    Supports the creation of new .json files in JSONL format (newline-delimited JSON).
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the JSONLCreator instance.

        Args:
            base_path (str, optional): The base path for storing JSONL files. 
                                     If None, use the data cache directory from the cache manager.
        """
        if base_path is None:
            cache_manager = PiscesLxCoreCacheManagerFacade.get_instance()
            base_path = cache_manager.get_cache_dir("data_cache")
        self.base_path = Path(base_path)
        # Create the base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_jsonl_file(self, filename: str, data: List[Dict[str, Any]], 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new JSONL format file.
        
        Args:
            filename (str): Name of the file (should end with .json).
            data (List[Dict[str, Any]]): List of JSON-serializable dictionaries.
            metadata (Optional[Dict[str, Any]]): Optional metadata to include as first line comment.
            
        Returns:
            str: Full path of the created file.
        """
        if not filename.endswith('.json'):
            filename += '.json'
            
        file_path = self.base_path / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write metadata as a comment at the beginning of the file if provided
                if metadata:
                    meta_line = f"# {json.dumps(metadata, ensure_ascii=False)}\n"
                    f.write(meta_line)
                
                # Write each data record as a separate line in JSON format
                for record in data:
                    json_line = json.dumps(record, ensure_ascii=False)
                    f.write(json_line + '\n')
                    
            return str(file_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to create JSONL file {filename}: {str(e)}")
    
    def create_from_template(self, template_name: str, count: int = 100, 
                           template_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a JSONL file from predefined templates.
        
        Args:
            template_name (str): Name of the template ('multimodal', 'text', 'conversation').
            count (int, optional): Number of records to generate. Defaults to 100.
            template_data (Optional[Dict[str, Any]]): Template-specific parameters.
            
        Returns:
            str: Full path of the created file.
        """
        templates = {
            'multimodal': self._generate_multimodal_template,
            'text': self._generate_text_template,
            'conversation': self._generate_conversation_template
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}")
            
        data = templates[template_name](count, template_data or {})
        filename = f"{template_name}_template_{count}.json"
        
        metadata = {
            "template": template_name,
            "count": count,
            "generator": "PiscesL1 JSONLCreator",
            "type": "template"
        }
        
        return self.create_jsonl_file(filename, data, metadata)
    
    def create_from_data(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Create a JSONL file from existing data.
        
        Args:
            data (List[Dict[str, Any]]): List of dictionaries containing the data.
            output_path (str): Path where the JSONL file will be created.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in data:
                    json.dump(record, f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            raise RuntimeError(f"Failed to create JSONL from data: {str(e)}")
            
    def append_to_jsonl(self, filename: str, new_data: List[Dict[str, Any]]) -> int:
        """
        Append new data to an existing JSONL file.
        
        Args:
            filename (str): Name of the existing JSONL file.
            new_data (List[Dict[str, Any]]): List of new records to append.
            
        Returns:
            int: Total number of lines in the file after appending.
        """
        if not filename.endswith('.json'):
            filename += '.json'
            
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {filename}")
        
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                lines_added = 0
                for record in new_data:
                    json_line = json.dumps(record, ensure_ascii=False)
                    f.write(json_line + '\n')
                    lines_added += 1
                    
            return self._count_lines(file_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to append to JSONL file {filename}: {str(e)}")
    
    def _generate_multimodal_template(self, count: int, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a template for multimodal training data.

        Args:
            count (int): Number of records to generate.
            params (Dict[str, Any]): Parameters for customizing the template.

        Returns:
            List[Dict[str, Any]]: A list of multimodal training data records.
        """
        data = []
        for i in range(count):
            record = {
                "id": f"mm_{i:06d}",
                "text": params.get("text", f"Sample text content {i}"),
                "image": params.get("image_path", f"images/sample_{i}.jpg"),
                "audio": params.get("audio_path", f"audio/sample_{i}.wav"),
                "metadata": {
                    "created": "auto-generated",
                    "modality": "multimodal",
                    "index": i
                }
            }
            data.append(record)
        return data
    
    def _generate_text_template(self, count: int, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a template for text-only training data.

        Args:
            count (int): Number of records to generate.
            params (Dict[str, Any]): Parameters for customizing the template.

        Returns:
            List[Dict[str, Any]]: A list of text-only training data records.
        """
        data = []
        for i in range(count):
            record = {
                "id": f"text_{i:06d}",
                "prompt": params.get("prompt", f"Question about topic {i}?"),
                "response": params.get("response", f"Answer to question {i}"),
                "metadata": {
                    "type": "text_qa",
                    "index": i
                }
            }
            data.append(record)
        return data
    
    def _generate_conversation_template(self, count: int, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate a template for conversation training data.

        Args:
            count (int): Number of records to generate.
            params (Dict[str, Any]): Parameters for customizing the template.

        Returns:
            List[Dict[str, Any]]: A list of conversation training data records.
        """
        data = []
        for i in range(count):
            record = {
                "id": f"conv_{i:06d}",
                "conversation": [
                    {"role": "user", "content": f"User message {i}"},
                    {"role": "assistant", "content": f"Assistant response {i}"}
                ],
                "metadata": {
                    "type": "conversation",
                    "turns": 2,
                    "index": i
                }
            }
            data.append(record)
        return data
    
    def _count_lines(self, file_path: Path) -> int:
        """
        Count the number of lines in a file efficiently.

        Args:
            file_path (Path): Path to the file.

        Returns:
            int: Number of lines in the file. Returns 0 if an error occurs.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def get_file_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about a JSONL file.
        
        Args:
            filename (str): Name of the JSONL file.
            
        Returns:
            Dict[str, Any]: File information including existence status, path, line count, size, and metadata status.
        """
        if not filename.endswith('.json'):
            filename += '.json'
            
        file_path = self.base_path / filename
        
        if not file_path.exists():
            return {"exists": False, "error": "File not found"}
        
        try:
            line_count = self._count_lines(file_path)
            file_size = file_path.stat().st_size
            
            # Read the first line to check if the file has metadata
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                has_metadata = first_line.startswith('#')
            
            return {
                "exists": True,
                "path": str(file_path),
                "line_count": line_count,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "has_metadata": has_metadata
            }
            
        except Exception as e:
            return {"exists": True, "error": str(e)}

# Global instance for easy access (lazy initialization)
_jsonl_creator: Optional[JSONLCreator] = None

def _get_jsonl_creator() -> JSONLCreator:
    """
    Get the lazy singleton instance of JSONLCreator.

    Returns:
        JSONLCreator: The singleton instance of JSONLCreator.
    """
    global _jsonl_creator
    if _jsonl_creator is None:
        _jsonl_creator = JSONLCreator()
    return _jsonl_creator

def create_jsonl_dataset(filename: str, data: List[Dict[str, Any]], 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    A convenience function to create a JSONL dataset.
    
    Args:
        filename (str): Name of the JSONL file.
        data (List[Dict[str, Any]]): List of JSON records.
        metadata (Optional[Dict[str, Any]]): Optional metadata.
        
    Returns:
        str: Path of the created file.
    """
    return _get_jsonl_creator().create_jsonl_file(filename, data, metadata)

def create_template_dataset(template_type: str, record_count: int = 10, base_path: str = None) -> str:
    """
    A unified function for creating template datasets, designed for direct external module calls.
    
    Args:
        template_type (str): Template type ('text', 'conversation', 'multimodal').
        record_count (int, optional): Number of records. Defaults to 10.
        base_path (str, optional): Base path. Defaults to the cache manager's data directory.
        
    Returns:
        str: Full path of the created file.
    """
    if base_path is None:
        from utils import PiscesLxCoreCacheManagerFacade
        cache_manager = PiscesLxCoreCacheManagerFacade.get_instance()
        base_path = cache_manager.get_or_create_cache_dir("data_cache")
    creator = JSONLCreator(base_path)
    return creator.create_from_template(template_type, record_count)
