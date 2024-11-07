import yaml
import unicodedata
import re


from evaluation.nlp_data import *
from evaluation.target_models import *
from evaluation.templates import *
from evaluation.metrics import *
from evaluation.decoders import *

def get_decoder(decoder_name: str, template: FewShotTemplate, dataset: Dataset) -> Decoder:
    decoder_name = slugify(decoder_name)
    if decoder_name == "query_based_constrained_label_generation":
        return QueryBasedConstrainedLabelGeneration(template)
    elif decoder_name == "nucleus_generation":
        return NucleusGeneration(template)
    elif decoder_name == "greedy_generation":
        return GreedyGeneration(template)
    elif decoder_name == "query_based_constrained_per_example_label_generation":
        return QueryBasedConstrainedPerExampleLabelGeneration(template, dataset)
    else:
        raise KeyError(f"Unrecognized decoder {decoder_name}")
    
    
def get_metric(
    metric_name: str,
    model: BaseModel,
    dataset: Dataset,
    template: FewShotTemplate,
    decoder: Decoder,
    metric_config: dict
) -> Metric:
    metric_name = slugify(metric_name)
    metric_to_class_map = {
        "zero_shot_accuracy": ZeroShotAccuracyMetric,
        "few_shot_accuracy": FewShotAccuracyMetric,
        "perturbational_accuracy": PerturbationalAccuracyMetric,
        "selectional_sensitivity": SelectionalSensitivityMetric,
        "permutational_sensitivity": PermutationalSensitivityMetric,
        "query_based_zero_shot_accuracy": QueryBasedZeroShotAccuracyMetric,
        "query_based_few_shot_accuracy": QueryBasedFewShotAccuracyMetric,
    }
    if metric_name not in metric_to_class_map:
        raise KeyError(f"Unrecognized metric {metric_name}")

    metric_class = metric_to_class_map[metric_name]
    return metric_class(
        model=model,
        dataset=dataset,
        template=template,
        decoder=decoder,
        **metric_config[metric_name],
    )


def slugify(value, allow_unicode=False) -> str:
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (unicodedata.normalize("NFKD",
                                       value).encode("ascii",
                                                     "ignore").decode("ascii"))
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")

def get_model(model_name):
    if model_name!='gpt3.5':
        model_name = slugify(model_name)
    model_to_class_map = {
        "gptneo1b3": GPTNeo1B3,
        "gptneo2b7": GPTNeo2B7,
        "gptneox20b": GPTNeoX20B,
        "bloom1b1": Bloom1B1,
        "bloom1b7": Bloom1B7,
        "bloom3b": Bloom3B,
        "bloom7b1": Bloom7B1,
        "vicuna13b": Vicuna13B,
        "llama7b": LLaMA7B,
        "llama13b": LLaMA13B,
        "opt1b3": OPT1B3,
        "opt2b7": OPT2B7,
        "opt6b7": OPT6B7,
        "opt13b": OPT13B,
        "stablelmbase3b": StableLMBase3B,
        "stablelmbase7b": StableLMBase7B,
        "stablelmtuned3b": StableLMTuned3B,
        "stablelmtuned7b": StableLMTuned7B,
        "gpt3.5": None
    }
    if model_name not in model_to_class_map:
        raise KeyError(f"Unrecognized model {model_name}")
    if model_name=='gpt3.5':
        return model_to_class_map[model_name]
    else:
        return model_to_class_map[model_name]()


def default_decoder_name(task_type: str) -> str:
    if task_type == "CLS":
        return "constrained_label_generation"
    elif task_type == "MCQ":
        return "constrained_per_example_label_generation"
    elif task_type == "GQA":
        return "greedy_generation"
    else:
        raise KeyError(f"Unrecognized task type {task_type}")
    
def query_based_decoder_name(task_type: str) -> str:
    if task_type == "CLS":
        return "query_based_constrained_label_generation"
    elif task_type == "MCQ":
        return "query_based_constrained_per_example_label_generation"
    elif task_type == "GQA":
        return "greedy_generation"
    else:
        raise KeyError(f"Unrecognized task type {task_type}")
    
def get_metric_name_config(metric_config):
    with open(metric_config, "r") as f:
        metric_config = yaml.safe_load(f)
        metric_name = list(metric_config.keys())[0]
    return metric_name, metric_config