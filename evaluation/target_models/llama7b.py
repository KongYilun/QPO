import torch

from evaluation.target_models.causal_lm import CausalLM


class LLaMA7B(CausalLM):

    def __init__(self):
        super().__init__(name="/home/kyl/code/QPO/llama2-7b-chat")
