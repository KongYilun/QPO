from typing import Any, Dict

from evaluation.nlp_data.dataset import Dataset
from evaluation.decoders.decoder import Decoder
from evaluation.metrics.utils import quasi_exact_match
from evaluation.target_models.base import BaseModel
from evaluation.templates.few_shot_template import FewShotTemplate


class Metric:

    def __init__(
        self,
        model: BaseModel,
        dataset: Dataset,
        template: FewShotTemplate,
        decoder: Decoder,
    ):

        """
            Parent class for all metrics.

            model: model to evaluate.
            dataset: dataset to evaluate on.
            template: template to use for generating prompts.
            decoder: decoder to use for decoding.
        """

        self.model = model
        self.dataset = dataset
        self.template = template
        self.decoder = decoder
        self.eq_metric = quasi_exact_match

    def create_inputs(self) -> Any:
        # has to be implemented by child classes
        raise NotImplementedError

    def evaluate(self, inputs: Any) -> Dict:
        # has to be implemented by child classes
        raise NotImplementedError
