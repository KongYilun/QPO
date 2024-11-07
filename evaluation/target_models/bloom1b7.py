import torch

from evaluation.target_models.causal_lm import CausalLM


class Bloom1B7(CausalLM):

    def __init__(self):
        super().__init__("bigscience/bloom-1b7")
