from .get_billsum_dataset import Billsum
from .get_squad_dataset import SQuAD
from .get_openwebtext_dataset import OpenWebText
from utils.constants import BILLSUM, SQUAD, OPENWEBTEXT


def get_dataset(dataset_name: str, tokenizer, model):
    """
    Tokenize a dataset and get additional data collator and metrics necessary for training the models

    :param dataset_name: Name of the dataset to build
    :param tokenizer: Tokenizer to tokenize the inputs
    :param model: Model to finetune / evaluate
    :return: tokenized dataset, data collator, and compute_metrics function
    """
    if dataset_name == BILLSUM:
        dataset = Billsum(tokenizer, model)
    elif dataset_name == SQUAD:
        dataset = SQuAD(tokenizer, model)
    elif dataset_name == OPENWEBTEXT:
        dataset = OpenWebText(tokenizer, model)
    else:
        assert False, f"No dataset named {dataset_name} available."

    return dataset
