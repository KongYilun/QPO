import torch

from evaluation.target_models.causal_lm import CausalLM


class OPT1B3(CausalLM):

    def __init__(self):
        super().__init__("evaluation/target_models/facebook/opt-1.3b")
