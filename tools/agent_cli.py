#!/usr/bin/env python3

# Copyright © 2025 Wenze Wei. All Rights Reserved.
#
# This file is part of Pisces L1.
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
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Any
from utils.log import RIGHT, DEBUG, ERROR

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import PiscesConfig, PiscesModel
from model.multimodal import PiscesAgent
from model.tokenizer import BPETokenizer

def load_agent(config_path: str, model_path: str = None):
    """Load PiscesModel with integrated agent functionality"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = PiscesConfig(**config_dict)
    
    # Initialize tokenizer
    tokenizer = BPETokenizer(vocab_path=os.path.join(tokenizer_path, 'vocab.json'), merges_path=os.path.join(tokenizer_path, 'merges.txt'))
    
    # Initialize integrated model with agent
    from model import PiscesModel
    model = PiscesModel(config)
    
    # Load model weights if provided
    if model_path and Path(model_path).exists():
        print(f"Loading model weights from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    return model, tokenizer


def register_example_tools(agent: PiscesAgent):
    """Register some example tools for the agent"""
    
    def calculator(expression: str) -> Dict[str, Any]:
        """Basic calculator tool"""
        try:
            result = eval(expression)
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e)}
    
    def file_reader(filepath: str) -> Dict[str, Any]:
        """Read file contents"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            return {"content": content, "filepath": filepath}
        except Exception as e:
            return {"error": str(e)}
    
    def web_search(query: str) -> Dict[str, Any]:
        """Simulated web search tool"""
        return {
            "query": query,
            "results": [
                f"Result 1 for {query}",
                f"Result 2 for {query}",
                f"Result 3 for {query}"
            ]
        }
    
    # Register tools as capabilities
    agent.register_capability("calculator", "Basic calculator tool", {"expression": "str"}, calculator)
    agent.register_capability("file_reader", "Read file contents", {"filepath": "str"}, file_reader)
    agent.register_capability("web_search", "Simulated web search tool", {"query": "str"}, web_search)


def run_interactive_agent(model, tokenizer):
    """Run agent in interactive mode (Integrated)"""
    RIGHT("Pisces L1 Agent - Interactive Mode (Integrated)")
    DEBUG("Type 'exit' to quit, 'help' for commands")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['exit', 'quit']:
                RIGHT("Goodbye!")
                break
            if user_input.lower() == 'help':
                DEBUG("Commands: exit, help")
                continue
            if not user_input:
                continue
            
            # Tokenize input
            input_ids = tokenizer.encode(user_input)
            input_ids = torch.tensor([input_ids])
            
            # Run agent mode
            result = model.forward(
                input_ids=input_ids,
                agent_mode=True,
                task=user_input
            )
            RIGHT(f"Result: {result}")
            
        except KeyboardInterrupt:
            RIGHT("\nExiting...")
            break
        except Exception as e:
            ERROR(f"Error: {e}")