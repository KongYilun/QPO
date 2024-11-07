from typing import List, Dict, Any

from evaluation.target_models.causal_lm import CausalLM
from evaluation.templates.query_based_few_shot_template import QueryBasedFewShotTemplate


class QueryBasedDecoder:

    def __init__(self, template: QueryBasedFewShotTemplate):
        self.template = template

    def decode(
        self,
        model: CausalLM,
        demonstrations: List[Dict[str, Any]],
        test_examples: List[Dict[str, Any]],
    ) -> List[dict]:
        raise NotImplementedError
