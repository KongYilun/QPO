import datasets

from evaluation.nlp_data.dataset import Dataset


class CosmosQA(Dataset):

    def __init__(self):
        dataset = datasets.load_from_disk("evaluation/nlp_dataset/cosmos_qa")
        sub_dataset=dataset['train'].train_test_split(test_size=0.1, seed=1)
        train_dev_set=sub_dataset['train'].train_test_split(test_size=0.1, seed=1)
        super().__init__("cosmos_qa",
                         "MCQ",
                         sub_dataset["train"],
                         sub_dataset["test"],
                         train_dev_set["test"],
                         dataset["validation"],
                         text_keys=["context", "question"],
                         label_key="label")#train_dev_set["train"],

    def get_choices_per_instance(self, instance):
        return [
            instance['answer0'], 
            instance['answer1'], 
            instance['answer2'],
            instance['answer3']
        ]
