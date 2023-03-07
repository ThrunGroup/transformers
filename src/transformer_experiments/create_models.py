from transformers import AutoTokenizer, TransfoXLLMHeadModel, GPT2LMHeadModel, GPT2Tokenizer, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

from utils.constants import TRANSFORMER_XL, SVD, PCA, GPT2, GPT2_LARGE
from accelerators.apply_accelerator import apply_accelerator
from data.get_dataset import get_dataset
from load_models import get_naive_model_and_tokenizer
from utils.constants import BILLSUM


def freeze_layers(model, num_layers_to_freeze: int = -1):
    """
    Freeze layers of the model

    :param model: PyTorch model to finetune
    :param num_layers_to_freeze: Number of layers to freeze
    """
    param_names = list(model.named_parameters())
    num_total_layers = len(model.transformer.h)
    if num_layers_to_freeze == -1:
        num_layers_to_freeze = num_total_layers

    layers_to_freeze = [f"transformer.h.{i}." for i in range(num_layers_to_freeze)]
    param_names_to_freeze = [param_name for param_name, _ in param_names if
                             any(layer_to_freeze in param_name for layer_to_freeze in layers_to_freeze)]

    # Freeze the specified layers
    for name, param in model.named_parameters():
        if name in param_names_to_freeze:
            param.requires_grad = False


def finetune(model_name: str, model, tokenizer, dataset_name: str):
    """
    Finetune the model on a dataset

    :param model_name: Name of the model to finetune
    :param model: PyTorch model to finetune
    :param tokenizer: Tokenizer to tokenize the dataset
    :param dataset: Name of the dataset to train on
    :return:
    """
    dataset, data_collator, compute_metrics = get_dataset(dataset_name, tokenizer, model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def create_model(model_name: str,
                 model_type: str,
                 dataset_name: str,
                 num_layers_to_freeze: int = -1,
                 accelerator_type: str = None,
                 **accelerator_args):
    """
    Load a naive pretrained model from HuggingFace, apply the accelerator to FC layers,
    and finetune it on the specified dataset.

    :param model_name: Name of the new model to save
    :param model_type: Type of the model (TXL, GPT, etc.)
    :param dataset_name: Name of the dataset to train on
    :param num_layers_to_freeze: Number of layers to freeze during finetuning
    :param accelerator_type: Type of the accelerator to apply (None, SVD, etc.)
    :param accelerator_args: Optional arguments for the accelerator
    """
    model, tokenizer = get_naive_model_and_tokenizer(model_type)
    apply_accelerator(model_type, model, accelerator_type, **accelerator_args)
    freeze_layers(model, num_layers_to_freeze)
    finetune(model_name, model, tokenizer, dataset_name)


if __name__ == "__main__":
    create_model(model_name="gpt2_svd_billsum",
                 model_type=GPT2,
                 dataset_name=BILLSUM,
                 num_layers_to_freeze=-1,
                 accelerator_type=SVD,
                 k=10)
