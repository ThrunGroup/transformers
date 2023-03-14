import os.path
import pandas as pd

from transformers import AutoTokenizer, TransfoXLLMHeadModel, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

import time
import torch
import matplotlib.pyplot as plt
from typing import List
from tqdm import tqdm

from utils.constants import TRANSFORMER_XL, SVD, PCA, GPT2, GPT2_MEDIUM, GPT2_LARGE, WIKITEXT2
from accelerators.apply_accelerator import apply_accelerator
from load_models import get_naive_model_and_tokenizer


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
    num_sentences: int = 200,
    stride: int = 1024,
    is_log: bool = True,
    is_plot: bool = True,
):
    # Set up the tokenizer and models
    models = {}
    tokenizers = {}
    for accelerator in accelerators:
        for model_name in model_names:
            model, tokenizer = get_naive_model_and_tokenizer(model_name)
            apply_accelerator(
                model_name=model_name, model=model, layers_to_accelerate=None, accelerator_type=accelerator, k=10
            )
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
        max_length = model.config.n_positions
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

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
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



if __name__ == "__main__":
    # inference_pipeline(model_names=[GPT2])
    inference_perplexity([GPT2, GPT2_MEDIUM], [None, SVD])
