import torch

from evaluation.target_models.causal_lm import CausalLM


class LLaMA13B(CausalLM):

    def __init__(self):
        super().__init__(name="/path/to/Llama-2-13b-chat-hf")
