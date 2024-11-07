from evaluation.nlp_data.ag_news import AGNews
from evaluation.nlp_data.anli import ANLI
from evaluation.nlp_data.boolq import BoolQ
from evaluation.nlp_data.cosmos_qa import CosmosQA
from evaluation.nlp_data.dataset import Dataset
from evaluation.nlp_data.hellaswag import HellaSwag
from evaluation.nlp_data.nq_open import NQOpen
from evaluation.nlp_data.imdb import IMDB
from evaluation.nlp_data.trivia_qa import TriviaQA
from evaluation.nlp_data.tweet_emotion import TweetEmotion
from evaluation.nlp_data.gsm8k import GSM8K
from evaluation.nlp_data.svamp import SVAMP


def get_dataset(name: str) -> Dataset:
    name2dataset = {
        "ag_news": AGNews,
        "imdb": IMDB,
        "anli": ANLI,
        "boolq": BoolQ,
        "tweet_emotion": TweetEmotion,
        "hellaswag": HellaSwag,
        "cosmos_qa": CosmosQA,
        "nq_open": NQOpen,
        "trivia_qa": TriviaQA,
        "gsm8k": GSM8K,
        "svamp": SVAMP
    }
    if not name in name2dataset:
        raise KeyError(f"Unrecognized dataset {name}")
    return name2dataset[name]()
