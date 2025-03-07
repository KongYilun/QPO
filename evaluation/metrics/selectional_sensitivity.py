import copy
import statistics
from typing import Any, List, Tuple, Dict

import datasets
from tqdm import tqdm

from evaluation.nlp_data.dataset import Dataset
from evaluation.decoders.decoder import Decoder
from evaluation.metrics.metric import Metric
from evaluation.target_models.base import BaseModel
from evaluation.templates.few_shot_template import FewShotTemplate


class SelectionalSensitivityMetric(Metric):
    def __init__(
        self,
        model: BaseModel,
        dataset: Dataset,
        template: FewShotTemplate,
        decoder: Decoder,
        num_demonstrations: int,
        num_combinations: int,
        num_test_instances: int,
    ):

        """
            Metric for evaluating selectional sensitivity.

            model: model to evaluate.
            dataset: dataset to evaluate on.
            template: template to use for generating prompts.
            decoder: decoder to use for decoding.
            num_demonstrations: K for K-shot learning.
            num_combinations: number of combinations of K-shot learning to try.
            num_test_instances: number of test instances to evaluate on.
        """

        super().__init__(model, dataset, template, decoder)
        self.num_demonstrations = num_demonstrations
        self.num_combinations = num_combinations
        self.num_test_instances = num_test_instances

    def create_inputs(self) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        # create inputs for calculating selectional sensitivity
        
        demonstrations_list = []
        for seed in range(self.num_combinations):
            demonstration_instances = self.dataset.sample_instances("train", self.num_demonstrations, seed=seed)
            demonstrations_list.append(demonstration_instances)
        
        test_instances = self.dataset.sample_instances("test", self.num_test_instances)
        return (demonstrations_list, test_instances)

    def evaluate(
        self, 
        inputs: Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]]]
    ) -> Dict[str, Any]:

        # unpack inputs
        demonstrations_list, test_instances = inputs

        # remove labels from test instances
        test_instances_no_label = datasets.Dataset.from_list(test_instances).remove_columns([self.dataset.label_key])
        test_instance_labels = [test_instance[self.dataset.label_key] for test_instance in test_instances]

        # evaluate on each combination of demonstrations
        accuracies = []
        for demonstrations in tqdm(demonstrations_list):
            predicted_outputs = [
                output["prediction"]
                for output in self.decoder.decode(
                    self.model,
                    demonstrations,
                    test_instances_no_label,
                )
            ]

            # This metric uses exact match for correctness
            correctness_indicators = [
                self.eq_metric(predicted_output, gt_output)
                for gt_output, predicted_output in zip(
                    test_instance_labels, predicted_outputs
                )
            ]

            # compute accuracy
            accuracies.append(statistics.mean(correctness_indicators))

        # return mean accuracy, standard deviation to selection, and list of accuracies
        return {
            "selectional_sensitivity": statistics.stdev(accuracies),
            "all_selectional_accuracies": accuracies,
        }
