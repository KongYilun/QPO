import datasets

from evaluation.nlp_data.dataset import Dataset


class BoolQ(Dataset):

    def __init__(self):
        dataset = datasets.load_from_disk("evaluation/nlp_dataset/boolq")
        test_dataset=dataset['validation'].train_test_split(test_size=0.1, seed=1)
        sub_dataset=dataset['train'].train_test_split(test_size=0.1, seed=1)
        train_dev_set=sub_dataset['train'].train_test_split(test_size=0.1, seed=1)
        super().__init__("boolq",
                         "CLS",
                         train_dev_set["train"],
                         sub_dataset["test"],
                         train_dev_set["test"],
                         test_dataset["train"],
                         text_keys=["question", "passage"],
                         label_key="answer")
