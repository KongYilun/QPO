import torch

from evaluation.target_models.causal_lm import CausalLM


class Bloom1B1(CausalLM):

    def __init__(self):
        super().__init__("models/bigscience/bloom-1b1")
