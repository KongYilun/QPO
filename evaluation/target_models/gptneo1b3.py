import torch

from evaluation.target_models.causal_lm import CausalLM


class GPTNeo1B3(CausalLM):

    def __init__(self):
        super().__init__("/path/to/gpt-neo-1.3B")
