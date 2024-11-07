import torch

from evaluation.target_models.causal_lm import CausalLM


class Vicuna13B(CausalLM):

    def __init__(self):
        super().__init__(name="/path/to/vicuna-13b")
