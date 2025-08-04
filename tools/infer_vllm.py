from vllm import LLM, SamplingParams

class VLLMEngine:
    def __init__(self, model_path, dtype="auto", gpu_memory_utilization=0.9, tensor_parallel_size=1):
        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size
        )

    def infer(self, prompt, temperature=0.7, max_tokens=512, top_p=0.95, stop=None):
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
