import torch

from evaluation.target_models.causal_lm import CausalLM


class Bloom3B(CausalLM):

    def __init__(self):
        super().__init__("bigscience/bloom-3b")
