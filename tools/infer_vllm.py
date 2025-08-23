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

from vllm import LLM, SamplingParams

# A class for interacting with the VLLM model for text generation.
class VLLMEngine:
    def __init__(self, model_path, dtype="auto", gpu_memory_utilization=0.9, tensor_parallel_size=1):
        """
        Initialize the VLLMEngine instance.

        Args:
            model_path (str): Path to the model to be loaded.
            dtype (str, optional): Data type for the model. Defaults to "auto".
            gpu_memory_utilization (float, optional): Proportion of GPU memory to use. Defaults to 0.9.
            tensor_parallel_size (int, optional): Number of GPUs to use for tensor parallelism. Defaults to 1.
        """
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size
        )

    def infer(self, prompt, temperature=0.7, max_tokens=512, top_p=0.95, stop=None):
        """
        Generate text based on the given prompt.

        Args:
            prompt (str): The input prompt for text generation.
            temperature (float, optional): Controls randomness in sampling. Defaults to 0.7.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 512.
            top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
            stop (list, optional): List of stop sequences. Defaults to None.

        Returns:
            str: The generated text.
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
