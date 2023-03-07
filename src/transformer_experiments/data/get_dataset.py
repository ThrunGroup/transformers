from get_billsum_dataset import get_billsum_dataset
from utils.constants import BILLSUM


def get_dataset(dataset_name: str, tokenizer, model):
    """
    Tokenize a dataset and get additional data collator and metrics necessary for training the models

    :param dataset_name: Name of the dataset to build
    :param tokenizer: Tokenizer to tokenize the inputs
    :param model: Model to finetune / evaluate
    :return: tokenized dataset, data collator, and compute_metrics function
    """
    if dataset_name == BILLSUM:
        return get_billsum_dataset(tokenizer, model)

    assert False, f"No dataset named {dataset_name} available."
