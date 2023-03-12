from transformers import Trainer, TrainingArguments
from typing import List
import pandas as pd
import time
import os

from utils.constants import TRANSFORMER_XL, SVD, PCA, GPT2, GPT2_LARGE
from accelerators.apply_accelerator import apply_accelerator
from data.get_dataset import get_dataset
from load_models import get_naive_model_and_tokenizer, load_model
from utils.parse_string import get_model_type


def evaluate_model(model_name: str,
                   dataset_name: str,
                   log_dir_name: str = "./exp_logs",
                   model=None,
                   tokenizer=None,
                   dataset=None,
                   trainer=None):
    """
    Evaluate a model on a dataset

    :param model_name: Name of the model to finetune
    :param dataset_name: Name of the dataset to train on
    :param log_dir_name: Directory to save the logs
    :param model: The PyTorch Model to evaluate
    :param tokenizer: Tokenizer to tokenize the dataset
    :param dataset: Dataset to evaluate the model on
    :param trainer: HuggingFace Trainer to evaluate the model
    """
    if not (model and tokenizer and dataset and trainer):
        # Load the model, tokenizer, dataset, and trainer if not provided
        # This block is executed usually when running this function as its own outside `create_model`

        # Load a pretrained original model from HuggingFace
        model_type = get_model_type(model_name)
        model, tokenizer = get_naive_model_and_tokenizer(model_type)
        model = load_model(model_name)

        # Get dataset
        dataset, data_collator, compute_metrics = get_dataset(dataset_name, tokenizer, model)

        # Set up the evaluation pipeline
        training_args = TrainingArguments(
            output_dir=model_name,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

    # Evaluate
    start_time = time.time()
    metrics = trainer.evaluate(eval_dataset=dataset["test"])
    end_time = time.time()
    inference_time = end_time - start_time

    # Save the experiment logs
    log_dict = {
        **metrics,
        "inference_time": inference_time,
    }

    log_df = pd.DataFrame(log_dict)

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = (os.path.join(parent_dir, log_dir_name))
    filename = os.path.join(
        log_dir,
        model_name
        + ".csv",
    )
    os.makedirs(log_dir, exist_ok=True)
    log_df.to_csv(filename, index=False)


def inference_pipeline(
        model_names: List[str] = [GPT2], accelerators: List[str] = [None, "SVD"],
):
    # Set up the tokenizer and models
    models = {}
    tokenizers = {}
    for accelerator in accelerators:
        for model_name in model_names:
            model, tokenizer = get_naive_model_and_tokenizer(model_name)
            apply_accelerator(model_name, model, accelerator, k=200)
            new_model_name = model_name + f"+ {accelerator}"
            models[new_model_name] = model
            tokenizers[new_model_name] = tokenizer

    # Benchmark the inference time for each model
    # Encode input sequence
    for model_name, model in models.items():
        print(f"Model: ", model_name)
        tokenizer = tokenizers[model_name]
        input_ids = tokenizer.encode(
            "Once upon a week, ",
            return_tensors="pt",
        )  # Encode input sequence
        start_time = time.time()
        output_ids = model.generate(input_ids, max_length=50)  # Generate text
        end_time = time.time()
        inference_time = end_time - start_time

        # Decode output sequence
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\tModel output: ", output_text)
        print(f"\tInference time: {inference_time:.4f} seconds")


if __name__ == "__main__":
    inference_pipeline(model_names=[GPT2])
