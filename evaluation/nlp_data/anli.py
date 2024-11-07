import datasets

from evaluation.nlp_data.dataset import Dataset


class ANLI(Dataset):

    def __init__(self):
        dataset = datasets.load_from_disk("evaluation/nlp_dataset/anli")
        super().__init__("anli",
                         "CLS",
                         dataset["train_r1"],
                         dataset["dev_r1"],
                         dataset["test_r1"],
                         text_keys=["premise", "hypothesis"],
                         label_key="label")
