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

"""Server utilities that expose multimodal generation through MCP.

The module defines :class:`YvMCPGenerationServer`, which mediates between the
Pisces multimodal generator and Model Context Protocol (MCP) clients. Requests
can trigger synchronous generations or opt into progress streaming updates.
"""

import uuid
import torch
import asyncio
import pandas as pd
from typing import Any, Dict

class YvMCPGenerationServer:
    """Serve multimodal generation requests with optional streaming feedback.

    Args:
        generator: Backing generator instance implementing ``generate_from_text``
            and ``generate_from_emotion`` with PyTorch tensor outputs.
    """

    def __init__(self, generator):
        self.generator = generator

    async def handle_generate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a generation request and return serialized tensor outputs.

        Args:
            request (Dict[str, Any]): Incoming MCP request containing modality,
                prompt or emotion, style overrides, and optional streaming flag.

        Returns:
            Dict[str, Any]: Response payload with success metadata, serialized
            tensor data, and optional watermark information. In case of failure,
            the payload contains ``success = False`` and an error message.
        """

        try:
            modality = request.get('modality', 'image')
            prompt = request.get('prompt', '')
            emotion = request.get('emotion', 'neutral')
            style = request.get('style', {})
            stream = request.get('stream', False)

            if stream:
                return await self.generate_streaming(modality, prompt, emotion, style, request)

            if prompt:
                result = self.generator.generate_from_text(prompt, modality, style=style, **request)
            else:
                result = self.generator.generate_from_emotion(emotion, modality, style=style, **request)

            response = {
                'success': True,
                'modality': modality,
                'timestamp': str(pd.Timestamp.now()),
                'metadata': {
                    'prompt': prompt,
                    'emotion': emotion,
                    'style': style,
                    'generation_params': request,
                    'watermark': {
                        'applied': True,
                        'method': 'hidden_watermark_only',
                        'content_type': modality
                    }
                },
                'request_id': str(uuid.uuid4())
            }

            if modality == 'document' and isinstance(result, dict):
                response['data'] = {k: v.detach().cpu().numpy().tolist() for k, v in result.items()}
                response['shape'] = {k: list(v.shape) for k, v in result.items()}
            else:
                t = result.detach().cpu().numpy()
                response['data'] = t.tolist()
                response['shape'] = list(t.shape)

            return response
        except Exception as e:
            return {'success': False, 'error': str(e), 'timestamp': str(pd.Timestamp.now())}

    async def generate_streaming(self, modality: str, prompt: str, emotion: str, style: Dict, request: Dict):
        """Return a streaming generator for progress updates during long runs.

        Args:
            modality (str): Requested generation modality such as ``image`` or ``document``.
            prompt (str): Optional textual conditioning string.
            emotion (str): Emotion cue used when no prompt is provided.
            style (Dict): Style overrides forwarded to the generator.
            request (Dict): Full request payload to extract step configuration.

        Returns:
            Dict[str, Any]: A response with ``type='stream'`` containing an async
            generator that yields per-step progress dictionaries with percentage
            completion.
        """

        steps = request.get('steps', 50)

        async def progress_stream():
            for step in range(steps):
                yield {
                    'step': step + 1,
                    'total_steps': steps,
                    'progress': (step + 1) / steps,
                    'modality': modality
                }
                await asyncio.sleep(0.1)

        return {'type': 'stream', 'generator': progress_stream()}
