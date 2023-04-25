from transformers import Trainer, TrainingArguments
from typing import List
import pandas as pd
import time
import os

from utils.constants import GPT2, BILLSUM
from accelerators.apply_accelerator import apply_accelerator
from data.get_dataset import get_dataset
from load_models import get_naive_model_and_tokenizer, load_model, list_checkpoint_models
from utils.parse_string import get_model_type


def evaluate_model(model_name: str,
                   dataset_name: str,
                   log_dir_name: str = "exp_logs",
                   model=None,
                   tokenizer=None,
                   test_dataset=None,
                   trainer=None):
    """
    Evaluate a model on a dataset

    :param model_name: Name of the model to finetune
    :param dataset_name: Name of the dataset to train on
    :param log_dir_name: Directory to save the logs
    :param model: The PyTorch Model to evaluate
    :param tokenizer: Tokenizer to tokenize the dataset
    :param test_dataset: Dataset to evaluate the model on
    :param trainer: HuggingFace Trainer to evaluate the model
    :returns: Dictionary of evaluation metrics and inference time
    """
    if not (model and tokenizer and test_dataset and trainer):
        # Load the model, tokenizer, dataset, and trainer if not provided
        # This block is executed usually when running this function as its own outside `create_model`

        # Load a pretrained original model from HuggingFace
        model_type = get_model_type(model_name)
        model, tokenizer = get_naive_model_and_tokenizer(model_type)
        model = load_model(model_name)

        # Get dataset
        dataset = get_dataset(dataset_name, tokenizer, model)
        _, test_dataset = dataset.get_tokenized_dataset()
        data_collator = dataset.get_data_collator()
        compute_metrics = dataset.get_compute_metrics()

        # Set up the evaluation pipeline
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    # Evaluate
    start_time = time.time()
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    end_time = time.time()
    inference_time = end_time - start_time

    # Save the experiment logs
    log_dict = {
        **metrics,
        "inference_time": inference_time,
    }

    print("Metrics: ", metrics)
    print("Inference time: ", inference_time)

    log_df = pd.DataFrame([log_dict], index=[0])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, log_dir_name)
    filename = os.path.join(
        log_dir,
        model_name
        + ".csv",
    )
    os.makedirs(log_dir, exist_ok=True)
    log_df.to_csv(filename, index=False)
    print("Saved the file to ", filename)

    return log_dict


if __name__ == "__main__":
    # Get one checkpoint
    print(list_checkpoint_models())
    checkpoint_models = [
        # "gpt2_None__accelerated_None_froze_0-10_dataset_billsum",
        "gpt2_SVD_k:64_accelerated_11_froze_0-10_dataset_billsum_trained_accelerated_layers",
    ]

    for model in checkpoint_models:
        # model_name = checkpoint_models[-1]
        print("Evaluating ", model)
        evaluate_model(model, BILLSUM)
