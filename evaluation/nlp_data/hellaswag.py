import datasets

from evaluation.nlp_data.dataset import Dataset


class HellaSwag(Dataset):

    def __init__(self):
        dataset = datasets.load_from_disk("evaluation/nlp_dataset/hellaswag")
        sub_dataset=dataset['train'].train_test_split(test_size=0.1, seed=1)
        train_dev_set=sub_dataset['train'].train_test_split(test_size=0.1, seed=1)
        super().__init__("hellaswag",
                         "MCQ",
                         train_dev_set["train"],
                         sub_dataset["test"],
                         train_dev_set["test"],
                         dataset["validation"],
                         text_keys=["ctx", "activity_label"],
                         label_key="label")

    def get_choices_per_instance(self, instance):
            return instance['endings']