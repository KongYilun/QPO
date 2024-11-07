import torch

from evaluation.target_models.causal_lm import CausalLM


class OPT13B(CausalLM):

    def __init__(self):
        super().__init__("facebook/opt-13b")
