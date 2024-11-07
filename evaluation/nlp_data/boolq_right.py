import datasets

from evaluation.nlp_data.dataset import Dataset


class BoolQ(Dataset):

    def __init__(self):
        dataset = datasets.load_from_disk("evaluation/nlp_dataset/boolq")
        sub_dataset=dataset['train'].train_test_split(test_size=0.1, seed=1)
        super().__init__("boolq",
                         "CLS",
                         sub_dataset["train"],
                         sub_dataset["test"],
                         dataset["validation"],
                         text_keys=["question", "passage"],
                         label_key="answer")
