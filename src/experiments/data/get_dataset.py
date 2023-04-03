from .get_billsum_dataset import get_billsum_dataset
from utils.constants import BILLSUM


def get_dataset(
    dataset_name: str, tokenizer, model, train_sample_ratio: int, test_sample_ratio: int, seed: int = 0,
):
    """
    Tokenize a dataset and get additional data collator and metrics necessary for training the models

    :param dataset_name: Name of the dataset to build
    :param tokenizer: Tokenizer to tokenize the inputs
    :param model: Model to finetune / evaluate
    :param train_sample_ratio: Sample size (proportion) of train data.
    :param test_sample_ratio: Sample size (proportion) of test data.
    :param seed: random seed for subsampling data
    :return: tokenized dataset, data collator, and compute_metrics function
    """
    if dataset_name == BILLSUM:
        return get_billsum_dataset(
            tokenizer,
            model,
            train_sample_ratio=train_sample_ratio,
            test_sample_ratio=test_sample_ratio,
            subsample_seed=seed,
        )

    assert False, f"No dataset named {dataset_name} available."
