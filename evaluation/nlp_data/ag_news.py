import datasets

from evaluation.nlp_data.dataset import Dataset


class AGNews(Dataset):

    def __init__(self):
        dataset = datasets.load_from_disk("evaluation/nlp_dataset/ag_news")
        sub_dataset=dataset['test'].train_test_split(test_size=0.1, seed=1)
        train_dev_set=dataset['train'].train_test_split(test_size=0.1, seed=1)
        super().__init__("ag_news", "CLS", train_dev_set["train"], sub_dataset["test"],train_dev_set["test"],  sub_dataset['train'])
