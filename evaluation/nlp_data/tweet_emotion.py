import datasets

from evaluation.nlp_data.dataset import Dataset


class TweetEmotion(Dataset):

    def __init__(self):
        dataset = datasets.load_from_disk("evaluation/nlp_dataset/tweet_emotion")
        train_dev_set=dataset['train'].train_test_split(test_size=0.2, seed=1)
        super().__init__("tweet_emotion",
                         "CLS",
                         train_dev_set["train"],
                         dataset["validation"],
                         train_dev_set["test"],
                         dataset["test"],
                         text_keys=["text"],
                         label_key="label")
