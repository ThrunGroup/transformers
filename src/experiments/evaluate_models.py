from transformers import Trainer, TrainingArguments
from typing import List
from tqdm import tqdm
import pandas as pd
import time
import os
import torch
from torch.ao.quantization import default_dynamic_qconfig, QConfigMapping, get_default_qconfig

# Note that this is temporary, we'll expose these functions to torch.ao.quantization after official releasee
from torch.quantization.quantize_fx import prepare_fx, convert_fx


from utils.constants import (
    GPT2,
    BILLSUM,
    WIKITEXT2,
    QUANTIZATION,
    SVD,
    OPT_13B,
    OPT_30B,
    OPT_1_3B,
    OPT_2_7B,
    OPT_350M,
    OPT_125M,
    OPT,
    BLOOM,
    BLOOM_560M,
    BLOOM_1b,
    BLOOM_3b,
    BLOOM_7b1,
    EXAMPLE_INPUT,
    StaticQ,
    DynamicQ,
    QUANTIZATION_GPU,
)
from utils.utils import print_size_of_model
from utils.parse_string import get_model_type, get_subsampling_ratio, get_seed
from accelerators.quantization import quantization
from accelerators.apply_accelerator import apply_accelerator
from data.get_dataset import get_dataset
from load_models import get_naive_model_and_tokenizer, load_model, list_checkpoint_models
from datasets import load_dataset


def evaluate_model(
    model_name: str,
    dataset_name: str,
    log_dir_name: str = "exp_logs",
    model=None,
    tokenizer=None,
    dataset=None,
    trainer=None,
):
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
        subsampling_ratio = get_subsampling_ratio(model_name)
        seed = get_seed(model_name)

        # Get dataset
        dataset, data_collator, compute_metrics = get_dataset(
            dataset_name,
            tokenizer,
            model,
            train_sample_ratio=subsampling_ratio,
            test_sample_ratio=subsampling_ratio,
            seed=seed,
        )

        # Set up the evaluation pipeline
        trainer = Trainer(
            model=model, tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics,
        )

    # Evaluate
    metrics = trainer.evaluate(eval_dataset=dataset["test"])

    # Save the experiment logs
    len_data = len(dataset["test"])
    inference_time_per_sample = metrics["eval_runtime"] / len_data
    log_dict = {
        **metrics,
        "inference_time": inference_time_per_sample,
    }

    print("Metrics: ", metrics)
    print("Inference time: ", inference_time_per_sample)

    log_df = pd.DataFrame([log_dict], index=[0])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, log_dir_name)
    filename = os.path.join(log_dir, model_name + ".csv",)
    os.makedirs(log_dir, exist_ok=True)
    log_df.to_csv(filename, index=False)


def inference_pipeline(
    model_names: List[str] = [GPT2], accelerators: List[str] = [None, "SVD"],
):
    # Set up the tokenizer and models
    models = {}
    tokenizers = {}
    torch.no_grad()
    for model_name in model_names:
        for accelerator in accelerators:
            model, tokenizer = get_naive_model_and_tokenizer(model_name)
            if accelerator == QUANTIZATION:
                model = quantization(
                    model,
                    quantization_type=DynamicQ,
                    example_input=tokenizer.encode(EXAMPLE_INPUT, return_tensors="pt",),
                )
            else:
                apply_accelerator(model_name, model, accelerator_type=accelerator, k=10)
            print(f"{model_name} size:")
            # print_size_of_model(model)
            new_model_name = model_name + f" + {accelerator}"
            models[new_model_name] = model
            tokenizers[new_model_name] = tokenizer

    # Benchmark the inference time for each model
    # Encode input sequence
    for model_name, model in models.items():
        print(f"Model: ", model_name)
        tokenizer = tokenizers[model_name]
        input_ids = tokenizer.encode("Once upon a week, ", return_tensors="pt",)  # Encode input sequence
        start_time = time.time()
        output_ids = model.generate(input_ids, max_length=50)  # Generate text
        end_time = time.time()
        inference_time = end_time - start_time

        # Decode output sequence
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\tModel output: ", output_text)
        print(f"\tInference time: {inference_time:.4f} seconds")


def inference_perplexity(
    model_names: List[str] = [GPT2],
    accelerators: List[str] = [None],
    dataset: str = WIKITEXT2,
    num_sentences: int = 100,
    stride: int = 1024,
    is_log: bool = True,
    is_plot: bool = True,
):
    # Set up the tokenizer and models
    models = {}
    tokenizers = {}
    for model_name in model_names:
        for accelerator in accelerators:
            if accelerator == QUANTIZATION_GPU:
                model, tokenizer = get_naive_model_and_tokenizer(model_name, load_in_8bit=True)
            else:
                model, tokenizer = get_naive_model_and_tokenizer(model_name, load_in_8bit=False)
                if accelerator == QUANTIZATION:
                    model = quantization(model, DynamicQ)
                else:
                    apply_accelerator(model_name, model, accelerator_type=accelerator, k=10)
            # print(f"{model_name} + {accelerator} size:")
            # print_size_of_model(model)
            new_model_name = model_name + f"+{accelerator}"
            models[new_model_name] = model
            tokenizers[new_model_name] = tokenizer

    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        if dataset is WIKITEXT2:
            test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        else:
            raise NotImplementedError(f"{dataset} is not implemented")

        encodings = tokenizer("\n\n".join(test["text"][:num_sentences]), return_tensors="pt")
        if GPT2 in model_name:
            max_length = model.config.n_positions
        elif OPT in model_name:
            max_length = model.config.max_position_embeddings
        else:
            max_length = stride
        stride = stride
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        inference_time = 0
        num_samples = 0
        for begin_loc in tqdm(range(0, seq_len, stride)):
            num_samples += 1
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                start_time = time.time()
                if accelerator == QUANTIZATION:
                    outputs = model(input_ids, labels=target_ids)
                else:
                    outputs = model(input_ids, labels=target_ids)
                inference_time += time.time() - start_time

                # loss is calculated using CrossEntropyLoss which averages over input tokens.
                # Multiply it with trg_len to get the summation instead of average.
                # We will take average over all the tokens to get the true average
                # in the last step of this example.
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        added_array = torch.stack(nlls).sum() / end_loc
        added_array = added_array.cuda()
        ppl = torch.exp(added_array)
        inference_time /= num_samples

        if is_log:
            curr_dir = os.path.dirname(__file__)
            log_dir = os.path.join(curr_dir, "evaluation_logs", f"{dataset}-n_{num_sentences}-s_{stride}")
            os.makedirs(log_dir, exist_ok=True)
            print(log_dir)

            log_dict = {"inference_time": [inference_time], "perplexity": [float(ppl)]}
            log_df = pd.DataFrame(log_dict)
            log_df.to_csv(os.path.join(log_dir, f"{model_name}.csv"), index=False)

        print(f"Model:{model_name}")
        print(f"Perplexity: {ppl}\n")
        print(f"Inference time: {inference_time}\n")


if __name__ == "__main__":
    inference_perplexity([OPT_125M], [QUANTIZATION_GPU, None])
    # inference_perplexity([OPT_350M], [None, QUANTIZATION])
    # # Get one checkpoint
    # checkpoint_models = [list_checkpoint_models()[1]]
    # for model in checkpoint_models:
    #     # model_name = checkpoint_models[-1]
    #     print("Evaluating ", model)
    #     evaluate_model(model, BILLSUM)
