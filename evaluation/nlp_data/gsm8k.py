import datasets

from evaluation.nlp_data.dataset import Dataset


class GSM8K(Dataset):

    def __init__(self):
        dataset = datasets.load_from_disk("evaluation/nlp_dataset/gsm8k")
        sub_dataset=dataset['train'].train_test_split(test_size=0.5, seed=1)
        train_dev_set=sub_dataset['train'].train_test_split(test_size=0.1, seed=1)
        #train,collect,dev,test
        super().__init__("gsm8k", "MATH", train_dev_set["train"], sub_dataset["test"],train_dev_set["test"],  dataset['test'])
