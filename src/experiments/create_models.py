import os
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from typing import List
import torch.nn as nn

from accelerators.apply_accelerator import apply_accelerator
from data.get_dataset import get_dataset
from load_models import get_naive_model_and_tokenizer
from evaluate_models import evaluate_model
from utils.parse_string import parse_string, dict_to_string
from utils.constants import (
    # Accelerators
    SVD,

    # Models
    GPT2,
    GPT2_MEDIUM,

    # Parameters
    NUM_BLOCKS_GPT2,
    NUM_BLOCKS_GPT2_MEDIUM,

    # Datasets
    BILLSUM,
    OPENWEBTEXT,
    SQUAD,
)


def freeze_layers(model,
                  layers_to_freeze: List[str] = None,
                  train_accelerated_layers: bool = False,
                  accelerated_layers: List[str] = None):
    """
        Freeze specific layers of a PyTorch model during finetuning.

    :param model: The PyTorch model to finetune.
    :param layers_to_freeze: A list of layer names to freeze. If not provided, all layers will be frozen.
    :param train_accelerated_layers: Whether to train the accelerated layers. If true, these layers will not be frozen.
    :param accelerated_layers: A list of names of the accelerated layers.
                               If not provided, every layer will be considered as accelerated.
    """
    param_names = list(model.named_parameters())

    layers_to_freeze = [f"transformer.h.{i}." for i in layers_to_freeze]
    param_names_to_freeze = [param_name for param_name, _ in param_names if
                             any(layer_to_freeze in param_name for layer_to_freeze in layers_to_freeze)]

    # Freeze the specified layers
    for name, param in model.named_parameters():
        if name in param_names_to_freeze:
            param.requires_grad = False

        if train_accelerated_layers and name in accelerated_layers:
            param.requires_grad = True
            print("This layer is not frozen: ", name)


def finetune(model_name: str, model, tokenizer, dataset_name: str, num_epochs: int = 10,
             resume_from_checkpoint: bool = False,
             do_evaluation: bool = True):
    """
    Finetune the model on a dataset

    :param model_name: Name of the model to finetune
    :param model: PyTorch model to finetune
    :param tokenizer: Tokenizer to tokenize the dataset
    :param dataset_name: Name of the dataset to train on
    :param num_epochs: Number of epochs to train the model
    :param resume_from_checkpoint: Whether to resume the training from the checkpoint
    :param do_evaluation: Whether to evaluate the model after training and save the metrics
    """
    dataset = get_dataset(dataset_name, tokenizer, model)
    train_dataset, test_dataset = dataset.get_tokenized_dataset()
    data_collator = dataset.get_data_collator()
    compute_metrics = dataset.get_compute_metrics()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(current_dir, "checkpoints", model_name)

    training_args = TrainingArguments(
        output_dir=checkpoint_path,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        num_train_epochs=num_epochs,
        logging_dir="exp_logs",
        save_total_limit=1,
        load_best_model_at_end=True,  # load the best model when finished training
        metric_for_best_model="rouge1",  # use rouge1 as the metric to determine best model
        greater_is_better=True,  # the higher the rouge score, the better the model
        gradient_accumulation_steps=4,
        fp16=True,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=8, early_stopping_threshold=0.0001)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    print("\n*********************")
    print("Training ", model_name)
    print("*********************\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if do_evaluation:
        evaluate_model(model_name, dataset_name,
                       model=model,
                       tokenizer=tokenizer,
                       test_dataset=test_dataset,
                       trainer=trainer)


def create_model(model_type: str,
                 dataset_name: str,
                 num_epochs: int = 5,
                 layers_to_freeze: str = None,
                 layers_to_accelerate: str = None,
                 train_accelerated_layers: bool = False,
                 do_evaluation: bool = True,
                 accelerator_type: str = None,
                 **accelerator_args):
    """
    Load a naive pretrained model from HuggingFace, apply the accelerator to FC layers,
    and finetune it on the specified dataset.

    :param model_type: Type of the model (TXL, GPT, etc.)
    :param dataset_name: Name of the dataset to train on
    :param num_epochs: Number of epochs to finetune the model
    :param layers_to_freeze: Layers to freeze (e.g. "1-3,7,9" freezes the 1,2,3,7,9th layer)
    :param layers_to_accelerate: Layers to accelerate (e.g. "1-3,7,9" accelerates the 1,2,3,7,9th layer)
    :param train_accelerated_layers: Whether to train the accelerated layers. If true, these layers will not be frozen.
    :param do_evaluation: Whether to evaluate the model after training and save the metrics
    :param accelerator_type: Type of the accelerator to apply (None, SVD, etc.)
    :param accelerator_args: Optional arguments for the accelerator
    """
    # Load a pretrained original model from HuggingFace
    model, tokenizer = get_naive_model_and_tokenizer(model_type)

    # Make a name for this model so that it's easy to load it later
    accelerator_args_string = dict_to_string(accelerator_args)
    model_name = f"{model_type}" \
                 f"_{accelerator_type}" \
                 f"_{accelerator_args_string}" \
                 f"_accelerated_{layers_to_accelerate}" \
                 f"_froze_{layers_to_freeze}" \
                 f"_epochs_{num_epochs}" \
                 f"_dataset_{dataset_name}"

    if train_accelerated_layers:
        model_name += "_trained_accelerated_layers"

    print("Creating ", model_name)

    # Get which layers to freeze and accelerate
    num_total_blocks = len(model.transformer.h)
    if layers_to_freeze := parse_string(layers_to_freeze):
        assert sorted(layers_to_freeze)[-1] <= num_total_blocks, \
            f"There are only {num_total_blocks} blocks in this model"
    if layers_to_accelerate := parse_string(layers_to_accelerate):
        assert sorted(layers_to_accelerate)[-1] <= num_total_blocks, \
            f"There are only {num_total_blocks} blocks in this model"

    # Accelerate & Finetune the model
    accelerated_layers = apply_accelerator(model_type, model, layers_to_accelerate, accelerator_type,
                                           **accelerator_args)
    # model.resize_token_embeddings(len(tokenizer))
    # model.transformer.output = nn.Sequential(
    #     model.transformer.output,
    #     nn.Linear(model.config.n_embd, 128)
    # )
    freeze_layers(model, layers_to_freeze, train_accelerated_layers, accelerated_layers)
    finetune(model_name, model, tokenizer, dataset_name, num_epochs, do_evaluation=do_evaluation)


if __name__ == "__main__":
    # Train a model without any accelerator
    # create_model(model_type=GPT2,
    #              dataset_name=BILLSUM,
    #              num_epochs=30,
    #              layers_to_freeze=f"0-{NUM_BLOCKS_GPT2 - 2}")
    #
    # # # Only train the accelerated FC layer and the very final linear layer
    # create_model(model_type=GPT2,
    #              dataset_name=BILLSUM,
    #              num_epochs=30,
    #              layers_to_freeze=f"0-{NUM_BLOCKS_GPT2 - 3}",
    #              layers_to_accelerate=f"{NUM_BLOCKS_GPT2 - 3}-{NUM_BLOCKS_GPT2 - 1}",
    #              train_accelerated_layers=True,
    #              accelerator_type=SVD,
    #              k=64)

    create_model(model_type=GPT2_MEDIUM,
                 dataset_name=BILLSUM,
                 num_epochs=30,
                 layers_to_freeze=f"0-{NUM_BLOCKS_GPT2_MEDIUM - 4}")

    create_model(model_type=GPT2_MEDIUM,
                 dataset_name=BILLSUM,
                 num_epochs=30,
                 layers_to_freeze=f"0-{NUM_BLOCKS_GPT2_MEDIUM - 4}",  # GPT2_MEDIUM
                 layers_to_accelerate=f"{NUM_BLOCKS_GPT2_MEDIUM - 3}-{NUM_BLOCKS_GPT2_MEDIUM - 1}",  # GPT2_MEDIUM
                 train_accelerated_layers=True,
                 accelerator_type=SVD,
                 k=64)